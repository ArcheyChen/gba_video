#!/usr/bin/env python3
"""
gba_encode.py  v5  ——  把视频/图片序列转成 GBA Mode3 YUV9 数据（支持条带帧间差分 + 向量量化）
输出 video_data.c / video_data.h
默认 5 s @ 30 fps，可用 --duration / --fps 修改
支持条带处理，每个条带独立进行I/P帧编码 + 码表压缩
"""

import argparse, cv2, numpy as np, pathlib, textwrap
import struct
import concurrent.futures
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist

WIDTH, HEIGHT = 240, 160
DEFAULT_STRIP_COUNT = 4
CODEBOOK_SIZE = 256  # 码表大小

Y_COEFF  = np.array([0.28571429,  0.57142857,  0.14285714])
CB_COEFF = np.array([-0.14285714, -0.28571429,  0.42857143])
CR_COEFF = np.array([ 0.35714286, -0.28571429, -0.07142857])
BLOCK_W, BLOCK_H = 2, 2
BYTES_PER_BLOCK  = 7  # 4Y + d_r + d_g + d_b

# 帧类型标识
FRAME_TYPE_I = 0x00  # I帧（关键帧）
FRAME_TYPE_P = 0x01  # P帧（差分帧）

def calculate_strip_heights(height: int, strip_count: int) -> list:
    """计算每个条带的高度，确保每个条带高度都是4的倍数"""
    # 确保总高度能被4整除
    if height % 4 != 0:
        raise ValueError(f"视频高度 {height} 必须是4的倍数")
    
    # 计算每个条带的基础高度（必须是4的倍数）
    base_height = (height // strip_count // 4) * 4
    
    # 计算剩余的高度
    remaining_height = height - (base_height * strip_count)
    
    strip_heights = []
    for i in range(strip_count):
        current_height = base_height
        # 将剩余高度以4的倍数分配给前面的条带
        if remaining_height >= 4:
            current_height += 4
            remaining_height -= 4
        strip_heights.append(current_height)
    
    # 验证总高度
    if sum(strip_heights) != height:
        raise ValueError(f"条带高度分配错误: {strip_heights} 总和 {sum(strip_heights)} != {height}")
    
    # 验证每个条带高度都是4的倍数
    for i, h in enumerate(strip_heights):
        if h % 4 != 0:
            raise ValueError(f"条带 {i} 高度 {h} 不是4的倍数")
    
    return strip_heights

def pack_yuv420_strip(frame_bgr: np.ndarray, strip_y: int, strip_height: int) -> np.ndarray:
    """
    向量化实现，把指定条带的 240×strip_height×3 BGR → YUV420：每 2×2 像素 7 Byte
    布局按行优先：(Y>>1) (Y>>1) (Y>>1) (Y>>1) d_r d_g d_b
    返回形状为 (strip_blocks_h, blocks_w, 7) 的数组，每个元素是一个2x2块
    """
    strip_bgr = frame_bgr[strip_y:strip_y + strip_height, :, :]
    B = strip_bgr[:, :, 0].astype(np.float32)
    G = strip_bgr[:, :, 1].astype(np.float32)
    R = strip_bgr[:, :, 2].astype(np.float32)

    Y  = (R*Y_COEFF[0]  + G*Y_COEFF[1]  + B*Y_COEFF[2]).round()
    Cb = (R*CB_COEFF[0] + G*CB_COEFF[1] + B*CB_COEFF[2]).round()
    Cr = (R*CR_COEFF[0] + G*CR_COEFF[1] + B*CR_COEFF[2]).round()

    Y  = np.clip(Y,  0, 255).astype(np.uint8)
    Cb = np.clip(Cb, -128, 127).astype(np.int16)
    Cr = np.clip(Cr, -128, 127).astype(np.int16)

    h, w = strip_bgr.shape[:2]
    blocks_h = h // BLOCK_H
    blocks_w = w // BLOCK_W

    # reshape为块结构: (blocks_h, 2, blocks_w, 2)
    Y_blocks  = Y.reshape(blocks_h, BLOCK_H, blocks_w, BLOCK_W)
    Cb_blocks = Cb.reshape(blocks_h, BLOCK_H, blocks_w, BLOCK_W)
    Cr_blocks = Cr.reshape(blocks_h, BLOCK_H, blocks_w, BLOCK_W)

    # 预处理Y值：右移1位
    y_flat = (Y_blocks.transpose(0,2,1,3).reshape(blocks_h, blocks_w, 4) >> 1).astype(np.uint8)
    
    # Cb/Cr平均
    cb_mean = np.clip(Cb_blocks.mean(axis=(1,3)).round(), -128, 127).astype(np.int16)
    cr_mean = np.clip(Cr_blocks.mean(axis=(1,3)).round(), -128, 127).astype(np.int16)
    
    # 预计算差值并右移1位: d_r = Cr>>1, d_g = (-(Cb>>1)-Cr)>>1, d_b = Cb>>1
    d_r = np.clip(cr_mean, -128, 127).astype(np.int8)  # Cr>>1
    d_g = np.clip((-(cb_mean >> 1) - cr_mean) >> 1, -128, 127).astype(np.int8)  # (-(Cb>>1)-Cr)>>1
    d_b = np.clip(cb_mean, -128, 127).astype(np.int8)  # Cb>>1

    # 合并：4个Y值(>>1) + d_r + d_g + d_b
    block_array = np.zeros((blocks_h, blocks_w, BYTES_PER_BLOCK), dtype=np.uint8)
    block_array[..., 0:4] = y_flat
    block_array[..., 4] = d_r.view(np.uint8)  # d_r
    block_array[..., 5] = d_g.view(np.uint8)  # d_g
    block_array[..., 6] = d_b.view(np.uint8)  # d_b
    
    return block_array

def calculate_block_diff(block1: np.ndarray, block2: np.ndarray) -> float:
    """计算两个块的差异度（使用Y通道的平均绝对差值）"""
    # 只比较Y通道（前4个字节）
    y_diff = np.abs(block1[:4].astype(np.int16) - block2[:4].astype(np.int16))
    return y_diff.mean()  # 使用平均差值，更敏感

def encode_strip_differential(current_blocks: np.ndarray, prev_blocks: np.ndarray, 
                            diff_threshold: float, force_i_threshold: float = 0.7) -> tuple:
    """
    差分编码当前条带（存储需要更新的完整块数据）
    返回: (编码数据, 是否为I帧)
    """
    if prev_blocks is None or current_blocks.shape != prev_blocks.shape:
        return encode_strip_i_frame(current_blocks), True
    
    blocks_h, blocks_w = current_blocks.shape[:2]
    total_blocks = blocks_h * blocks_w
    
    if total_blocks == 0:
        return b'', True
    
    # 计算每个块的差异
    block_diffs = np.zeros((blocks_h, blocks_w))
    for by in range(blocks_h):
        for bx in range(blocks_w):
            block_diffs[by, bx] = calculate_block_diff(
                current_blocks[by, bx], prev_blocks[by, bx]
            )
    
    # 统计需要更新的块数
    blocks_to_update = (block_diffs > diff_threshold).sum()
    update_ratio = blocks_to_update / total_blocks
    
    # 如果需要更新的块太多，则使用I帧
    if update_ratio > force_i_threshold:
        return encode_strip_i_frame(current_blocks), True
    
    # 否则编码为P帧
    data = bytearray()
    data.append(FRAME_TYPE_P)
    
    # 存储需要更新的块数（2字节）
    data.extend(struct.pack('<H', blocks_to_update))
    
    # 存储每个需要更新的块的索引和数据
    block_idx = 0
    for by in range(blocks_h):
        for bx in range(blocks_w):
            if block_diffs[by, bx] > diff_threshold:
                # 存储块索引（2字节）
                data.extend(struct.pack('<H', block_idx))
                # 存储完整的块数据
                data.extend(current_blocks[by, bx].tobytes())
            block_idx += 1
    
    return bytes(data), False

def encode_strip_i_frame(blocks: np.ndarray) -> bytes:
    """编码条带I帧（完整条带）"""
    data = bytearray()
    data.append(FRAME_TYPE_I)
    if blocks.size > 0:
        data.extend(blocks.flatten().tobytes())
    return bytes(data)

def generate_codebook(blocks_data: np.ndarray, codebook_size: int = CODEBOOK_SIZE, max_iter: int = 100) -> tuple:
    """
    使用K-Means聚类生成码表
    blocks_data: shape (N, 6) 的块数据数组
    返回: (codebook, effective_size) - 码表和有效码字数量
    """
    if len(blocks_data) == 0:
        return np.zeros((codebook_size, BYTES_PER_BLOCK), dtype=np.uint8), 0
    
    # 确保blocks_data是2D数组 (N, 6)
    if blocks_data.ndim > 2:
        blocks_data = blocks_data.reshape(-1, BYTES_PER_BLOCK)
    
    # 去重，统计实际不同的块数量
    # 将每个6字节块转换为一个字符串来进行去重
    blocks_as_tuples = [tuple(block) for block in blocks_data]
    unique_tuples = list(set(blocks_as_tuples))
    unique_blocks = np.array(unique_tuples, dtype=np.uint8)
    
    effective_size = min(len(unique_blocks), codebook_size)
    
    print(f"    原始块数: {len(blocks_data)}, 唯一块数: {len(unique_blocks)}, 有效码字数: {effective_size}")
    
    # 如果唯一块数小于等于码表大小，直接使用
    if len(unique_blocks) <= codebook_size:
        codebook = np.zeros((codebook_size, BYTES_PER_BLOCK), dtype=np.uint8)
        codebook[:len(unique_blocks)] = unique_blocks
        # 用最后一个块填充剩余位置，避免未初始化数据
        if len(unique_blocks) > 0:
            for i in range(len(unique_blocks), codebook_size):
                codebook[i] = unique_blocks[-1]
        return codebook, effective_size
    
    # 使用MiniBatchKMeans进行聚类
    kmeans = MiniBatchKMeans(
        n_clusters=codebook_size, 
        random_state=42, 
        batch_size=min(1000, len(blocks_data)),
        max_iter=max_iter,
        n_init=3
    )
    kmeans.fit(blocks_data.astype(np.float32))
    
    # 将聚类中心转换回uint8，确保在有效范围内
    codebook = np.clip(kmeans.cluster_centers_.round(), 0, 255).astype(np.uint8)
    
    # 验证Cb/Cr在有效范围内 (-128到127，以uint8存储)
    for i in range(codebook_size):
        # Cb (索引4) 和 Cr (索引5) 需要特殊处理
        cb_val = codebook[i, 4].view(np.int8)
        cr_val = codebook[i, 5].view(np.int8)
        if cb_val < -128 or cb_val > 127 or cr_val < -128 or cr_val > 127:
            print(f"    警告: 码字{i} Cb/Cr值超出范围: Cb={cb_val}, Cr={cr_val}")
            # 裁剪到有效范围
            codebook[i, 4] = np.clip(cb_val, -128, 127).astype(np.int8).view(np.uint8)
            codebook[i, 5] = np.clip(cr_val, -128, 127).astype(np.int8).view(np.uint8)
    
    return codebook, codebook_size

def quantize_blocks(blocks_data: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """
    使用码表对块进行量化
    返回每个块对应的码表索引
    """
    if len(blocks_data) == 0:
        return np.array([], dtype=np.uint8)
    
    # 计算每个块与码表中所有条目的距离
    distances = cdist(blocks_data.astype(np.float32), codebook.astype(np.float32), metric='euclidean')
    
    # 找到最近的码表索引
    indices = np.argmin(distances, axis=1).astype(np.uint8)
    
    # 验证索引范围
    max_idx = indices.max() if len(indices) > 0 else 0
    if max_idx >= CODEBOOK_SIZE:
        print(f"警告: 量化索引超出范围: 最大索引={max_idx}, 码表大小={CODEBOOK_SIZE}")
        indices = np.clip(indices, 0, CODEBOOK_SIZE - 1)
    
    return indices

def encode_strip_i_frame_vq(blocks: np.ndarray, codebook: np.ndarray) -> bytes:
    """编码条带I帧（带向量量化）"""
    data = bytearray()
    data.append(FRAME_TYPE_I)
    
    if blocks.size > 0:
        # 展平块数据并量化
        blocks_flat = blocks.reshape(-1, BYTES_PER_BLOCK)
        indices = quantize_blocks(blocks_flat, codebook)
        
        # 存储码表大小和码表数据
        data.extend(struct.pack('<H', CODEBOOK_SIZE))
        data.extend(codebook.flatten().tobytes())
        
        # 存储量化索引
        data.extend(indices.tobytes())
    
    return bytes(data)

def encode_strip_differential_vq(current_blocks: np.ndarray, prev_blocks: np.ndarray, 
                                codebook: np.ndarray, diff_threshold: float, 
                                force_i_threshold: float = 0.7) -> tuple:
    """
    差分编码当前条带（使用向量量化）
    返回: (编码数据, 是否为I帧)
    """
    if prev_blocks is None or current_blocks.shape != prev_blocks.shape:
        return encode_strip_i_frame_vq(current_blocks, codebook), True
    
    blocks_h, blocks_w = current_blocks.shape[:2]
    total_blocks = blocks_h * blocks_w
    
    if total_blocks == 0:
        return b'', True
    
    # 计算每个块的差异
    block_diffs = np.zeros((blocks_h, blocks_w))
    for by in range(blocks_h):
        for bx in range(blocks_w):
            block_diffs[by, bx] = calculate_block_diff(
                current_blocks[by, bx], prev_blocks[by, bx]
            )
    
    # 统计需要更新的块数
    blocks_to_update = (block_diffs > diff_threshold).sum()
    update_ratio = blocks_to_update / total_blocks
    
    # 如果需要更新的块太多，则使用I帧
    if update_ratio > force_i_threshold:
        return encode_strip_i_frame_vq(current_blocks, codebook), True
    
    # 否则编码为P帧
    data = bytearray()
    data.append(FRAME_TYPE_P)
    
    # 存储需要更新的块数（2字节）
    data.extend(struct.pack('<H', blocks_to_update))
    
    # 收集需要更新的块数据并量化
    if blocks_to_update > 0:
        updated_blocks = []
        updated_indices = []
        
        block_idx = 0
        for by in range(blocks_h):
            for bx in range(blocks_w):
                if block_diffs[by, bx] > diff_threshold:
                    updated_blocks.append(current_blocks[by, bx])
                    updated_indices.append(block_idx)
                block_idx += 1
        
        # 量化更新的块
        updated_blocks = np.array(updated_blocks)
        quantized_indices = quantize_blocks(updated_blocks, codebook)
        
        # 存储块索引和量化索引
        for i, (block_idx, quant_idx) in enumerate(zip(updated_indices, quantized_indices)):
            data.extend(struct.pack('<H', block_idx))  # 块索引
            data.append(quant_idx)  # 量化索引
    
    return bytes(data), False

def process_strip_parallel(args):
    """并行处理单个条带的编码任务"""
    (strip_blocks, prev_strip_blocks, strip_codebook, frame_idx, 
     i_frame_interval, diff_threshold, force_i_threshold, is_first_frame) = args
    
    # 决定是否强制I帧
    force_i_frame = (frame_idx % i_frame_interval == 0) or is_first_frame
    
    if force_i_frame or prev_strip_blocks is None:
        # 编码为I帧
        strip_data = encode_strip_i_frame_vq(strip_blocks, strip_codebook)
        is_i_frame = True
    else:
        # 尝试差分编码
        strip_data, is_i_frame = encode_strip_differential_vq(
            strip_blocks, prev_strip_blocks, strip_codebook,
            diff_threshold, force_i_threshold
        )
    
    return strip_data, is_i_frame

def generate_gop_codebooks(frames: list, strip_count: int, i_frame_interval: int, 
                          kmeans_max_iter: int = 100) -> dict:
    """为每个GOP（Group of Pictures）的每个条带生成码表"""
    print("正在为每个GOP生成条带码表...")
    
    gop_codebooks = {}  # {gop_start_frame: [strip_codebooks]}
    
    # 确定所有I帧的位置
    i_frame_positions = []
    for frame_idx in range(len(frames)):
        if frame_idx % i_frame_interval == 0:
            i_frame_positions.append(frame_idx)
    
    # 为每个GOP生成码表
    for gop_idx, gop_start in enumerate(i_frame_positions):
        # 确定GOP的结束位置
        if gop_idx + 1 < len(i_frame_positions):
            gop_end = i_frame_positions[gop_idx + 1]
        else:
            gop_end = len(frames)
        
        print(f"  处理GOP {gop_idx}: 帧 {gop_start} 到 {gop_end-1}")
        
        gop_codebooks[gop_start] = []
        
        # 为GOP中的每个条带生成码表
        for strip_idx in range(strip_count):
            print(f"    生成条带 {strip_idx} 的码表...")
            
            # 收集该GOP中该条带的所有块数据
            strip_blocks_samples = []
            
            for frame_idx in range(gop_start, gop_end):
                strip_blocks = frames[frame_idx][strip_idx]
                if strip_blocks.size > 0:
                    blocks_flat = strip_blocks.reshape(-1, BYTES_PER_BLOCK)
                    strip_blocks_samples.append(blocks_flat)
            
            # 合并所有样本
            if strip_blocks_samples:
                all_blocks = np.vstack(strip_blocks_samples)
                codebook, effective_size = generate_codebook(all_blocks, CODEBOOK_SIZE, kmeans_max_iter)
                total_samples = len(all_blocks)
            else:
                codebook = np.zeros((CODEBOOK_SIZE, BYTES_PER_BLOCK), dtype=np.uint8)
                effective_size = 0
                total_samples = 0
            
            gop_codebooks[gop_start].append({
                'codebook': codebook,
                'total_samples': total_samples,
                'effective_size': effective_size,
                'utilization': effective_size / CODEBOOK_SIZE if CODEBOOK_SIZE > 0 else 0
            })
            
            print(f"      GOP{gop_idx} 条带{strip_idx}: 样本数{total_samples}, 有效码字{effective_size}, 利用率{effective_size/CODEBOOK_SIZE*100:.1f}%")
    
    return gop_codebooks

def get_current_codebooks(frame_idx: int, gop_codebooks: dict, i_frame_interval: int) -> list:
    """获取当前帧应该使用的码表"""
    # 找到当前帧所属的GOP起始位置
    gop_start = (frame_idx // i_frame_interval) * i_frame_interval
    
    if gop_start in gop_codebooks:
        return [strip_data['codebook'] for strip_data in gop_codebooks[gop_start]]
    else:
        # 如果找不到，使用第一个GOP的码表
        first_gop = min(gop_codebooks.keys())
        return [strip_data['codebook'] for strip_data in gop_codebooks[first_gop]]

def process_strip_parallel_with_gop(args):
    """并行处理单个条带的编码任务（使用GOP码表）"""
    (strip_blocks, prev_strip_blocks, strip_codebook, frame_idx, 
     i_frame_interval, diff_threshold, force_i_threshold, is_first_frame) = args
    
    # 决定是否强制I帧
    force_i_frame = (frame_idx % i_frame_interval == 0) or is_first_frame
    
    if force_i_frame or prev_strip_blocks is None:
        # 编码为I帧
        strip_data = encode_strip_i_frame_vq(strip_blocks, strip_codebook)
        is_i_frame = True
    else:
        # 尝试差分编码
        strip_data, is_i_frame = encode_strip_differential_vq(
            strip_blocks, prev_strip_blocks, strip_codebook,
            diff_threshold, force_i_threshold
        )
    
    return strip_data, is_i_frame

def write_header(path_h: pathlib.Path, frame_cnt: int, total_bytes: int, strip_count: int, strip_heights: list):
    guard = "VIDEO_DATA_H"
    strip_heights_str = ', '.join(map(str, strip_heights))
    
    with path_h.open("w", encoding="utf-8") as f:
        f.write(textwrap.dedent(f"""\
            #ifndef {guard}
            #define {guard}

            #define VIDEO_FRAME_COUNT   {frame_cnt}
            #define VIDEO_WIDTH         {WIDTH}
            #define VIDEO_HEIGHT        {HEIGHT}
            #define VIDEO_TOTAL_BYTES   {total_bytes}
            #define VIDEO_STRIP_COUNT   {strip_count}
            #define CODEBOOK_SIZE       {CODEBOOK_SIZE}
            
            // 帧类型定义
            #define FRAME_TYPE_I        0x00
            #define FRAME_TYPE_P        0x01
            
            // 块参数
            #define BLOCK_WIDTH         2
            #define BLOCK_HEIGHT        2
            #define BYTES_PER_BLOCK     7

            // 条带高度数组
            extern const unsigned char strip_heights[VIDEO_STRIP_COUNT];
            
            extern const unsigned char video_data[VIDEO_TOTAL_BYTES];
            extern const unsigned int frame_offsets[VIDEO_FRAME_COUNT];

            #endif // {guard}
            """))

def write_source(path_c: pathlib.Path, data: bytes, frame_offsets: list, strip_heights: list):
    with path_c.open("w", encoding="utf-8") as f:
        f.write('#include "video_data.h"\n\n')
        
        # 写入条带高度数组
        f.write("const unsigned char strip_heights[] = {\n")
        f.write("    " + ', '.join(map(str, strip_heights)) + "\n")
        f.write("};\n\n")
        
        # 写入帧偏移表
        f.write("const unsigned int frame_offsets[] = {\n")
        for i in range(0, len(frame_offsets), 8):
            chunk = ', '.join(f"{offset}" for offset in frame_offsets[i:i+8])
            f.write("    " + chunk + ",\n")
        f.write("};\n\n")
        
        # 写入视频数据
        f.write("const unsigned char video_data[] = {\n")
        per_line = 16
        for i in range(0, len(data), per_line):
            chunk = ', '.join(f"0x{v:02X}" for v in data[i:i+per_line])
            f.write("    " + chunk + ",\n")
        f.write("};\n")

def main():
    pa = argparse.ArgumentParser(description="Encode to GBA YUV9 with strip-based inter-frame compression and vector quantization")
    pa.add_argument("input")
    pa.add_argument("--duration", type=float, default=5.0)
    pa.add_argument("--fps", type=int, default=30)
    pa.add_argument("--out", default="video_data")
    pa.add_argument("--strip-count", type=int, default=DEFAULT_STRIP_COUNT,
                   help=f"条带数量（默认{DEFAULT_STRIP_COUNT}）")
    pa.add_argument("--i-frame-interval", type=int, default=30, 
                   help="间隔多少帧插入一个I帧（默认30）")
    pa.add_argument("--diff-threshold", type=float, default=2.0,
                   help="差异阈值，超过此值的块将被更新（默认2.0，Y通道平均差值）")
    pa.add_argument("--force-i-threshold", type=float, default=0.7,
                   help="当需要更新的块比例超过此值时，强制生成I帧（默认0.7）")
    pa.add_argument("--kmeans-max-iter", type=int, default=200,
                   help="K-Means聚类最大迭代次数（默认200）")
    pa.add_argument("--threads", type=int, default=None,
                   help="并行处理线程数（默认为CPU核心数）")
    args = pa.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit("❌ 打不开输入文件")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    every = int(round(src_fps / args.fps))
    grab_max = int(args.duration * src_fps)

    # 计算条带高度
    strip_heights = calculate_strip_heights(HEIGHT, args.strip_count)
    print(f"条带配置: {args.strip_count} 个条带，高度分别为: {strip_heights}")

    frames = []
    idx = 0
    print("正在提取帧...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        while idx < grab_max:
            ret, frm = cap.read()
            if not ret:
                break
            if idx % every == 0:
                frm = cv2.resize(frm, (WIDTH, HEIGHT), cv2.INTER_AREA)
                # 多线程处理每个条带
                strip_y_list = []
                y = 0
                for strip_height in strip_heights:
                    strip_y_list.append((frm, y, strip_height))
                    y += strip_height
                # 提交任务
                future_to_idx = {
                    executor.submit(pack_yuv420_strip, *args): i
                    for i, args in enumerate(strip_y_list)
                }
                frame_strips = [None] * len(strip_y_list)
                for future in concurrent.futures.as_completed(future_to_idx):
                    i = future_to_idx[future]
                    frame_strips[i] = future.result()
                frames.append(frame_strips)
                
                if len(frames) % 30 == 0:
                    print(f"  已提取 {len(frames)} 帧")
            idx += 1
    cap.release()

    if not frames:
        raise SystemExit("❌ 没有任何帧被采样")

    print(f"总共提取了 {len(frames)} 帧")

    # 为每个GOP生成码表
    gop_codebooks = generate_gop_codebooks(frames, args.strip_count, args.i_frame_interval, args.kmeans_max_iter)

    # 编码所有帧
    print("正在编码帧...")
    encoded_frames = []
    frame_offsets = []
    current_offset = 0
    prev_strips = [None] * args.strip_count
    i_frame_count = [0] * args.strip_count
    p_frame_count = [0] * args.strip_count
    
    # 计算总的码表统计
    all_codebook_stats = []
    for gop_start, gop_data in gop_codebooks.items():
        all_codebook_stats.extend(gop_data)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        for frame_idx, current_strips in enumerate(frames):
            frame_offsets.append(current_offset)
            
            # 获取当前帧应该使用的码表
            current_codebooks = get_current_codebooks(frame_idx, gop_codebooks, args.i_frame_interval)
            
            # 准备并行编码任务
            tasks = []
            for strip_idx, current_strip in enumerate(current_strips):
                task_args = (
                    current_strip, prev_strips[strip_idx], current_codebooks[strip_idx],
                    frame_idx, args.i_frame_interval, args.diff_threshold, 
                    args.force_i_threshold, frame_idx == 0
                )
                tasks.append(task_args)
            
            # 并行编码所有条带
            future_to_strip = {
                executor.submit(process_strip_parallel_with_gop, task): strip_idx
                for strip_idx, task in enumerate(tasks)
            }
            
            frame_data = bytearray()
            strip_results = [None] * args.strip_count
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_strip):
                strip_idx = future_to_strip[future]
                strip_data, is_i_frame = future.result()
                strip_results[strip_idx] = (strip_data, is_i_frame)
                
                if is_i_frame:
                    i_frame_count[strip_idx] += 1
                else:
                    p_frame_count[strip_idx] += 1
            
            # 按顺序组装帧数据
            for strip_idx, (strip_data, _) in enumerate(strip_results):
                # 存储条带长度（2字节）+ 条带数据
                frame_data.extend(struct.pack('<H', len(strip_data)))
                frame_data.extend(strip_data)
                
                # 更新参考条带
                prev_strips[strip_idx] = current_strips[strip_idx].copy() if current_strips[strip_idx].size > 0 else None
            
            encoded_frames.append(bytes(frame_data))
            current_offset += len(frame_data)
            
            if frame_idx % 30 == 0 or frame_idx == len(frames) - 1:
                print(f"  已编码 {frame_idx + 1}/{len(frames)} 帧")
    
    # 合并所有数据
    all_data = b''.join(encoded_frames)
    
    # 写入文件
    write_header(pathlib.Path(args.out).with_suffix(".h"), len(frames), len(all_data), 
                args.strip_count, strip_heights)
    write_source(pathlib.Path(args.out).with_suffix(".c"), all_data, frame_offsets, strip_heights)
    
    # 统计信息
    total_original_size = 0
    for strip_height in strip_heights:
        blocks_in_strip = (strip_height // BLOCK_H) * (WIDTH // BLOCK_W)
        total_original_size += blocks_in_strip * BYTES_PER_BLOCK
    
    original_size = len(frames) * total_original_size
    compressed_size = len(all_data)
    compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
    
    # 计算平均码表利用率
    avg_utilization = np.mean([stat['utilization'] for stat in all_codebook_stats]) * 100
    avg_effective_size = np.mean([stat['effective_size'] for stat in all_codebook_stats])
    
    print(f"\n✅ 编码完成：")
    print(f"   总帧数: {len(frames)}")
    print(f"   条带数: {args.strip_count}")
    print(f"   条带高度: {strip_heights}")
    print(f"   码表大小: {CODEBOOK_SIZE}")
    print(f"   GOP数量: {len(gop_codebooks)}")
    print(f"   平均有效码字数: {avg_effective_size:.1f}")
    print(f"   平均码表利用率: {avg_utilization:.1f}%")
    print(f"   K-Means迭代次数: {args.kmeans_max_iter}")
    
    for strip_idx in range(args.strip_count):
        print(f"   条带{strip_idx}: I帧{i_frame_count[strip_idx]}, P帧{p_frame_count[strip_idx]}")
    
    print(f"   原始大小: {original_size:,} bytes")
    print(f"   压缩后大小: {compressed_size:,} bytes")
    print(f"   压缩比: {compression_ratio:.2f}x")
    print(f"   I帧间隔: {args.i_frame_interval}")
    print(f"   差异阈值: {args.diff_threshold}")

if __name__ == "__main__":
    main()