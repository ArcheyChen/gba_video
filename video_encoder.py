#!/usr/bin/env python3
"""
gba_encode.py  v4  ——  把视频/图片序列转成 GBA Mode3 YUV9 数据（支持条带帧间差分）
输出 video_data.c / video_data.h
默认 5 s @ 30 fps，可用 --duration / --fps 修改
支持条带处理，每个条带独立进行I/P帧编码
"""

import argparse, cv2, numpy as np, pathlib, textwrap
import struct

WIDTH, HEIGHT = 240, 160
DEFAULT_STRIP_COUNT = 4

Y_COEFF  = np.array([0.28571429,  0.57142857,  0.14285714])
CB_COEFF = np.array([-0.14285714, -0.28571429,  0.42857143])
CR_COEFF = np.array([ 0.35714286, -0.28571429, -0.07142857])
BLOCK_W, BLOCK_H = 4, 4
BYTES_PER_BLOCK  = 18                                   # 16Y + Cb + Cr

# 帧类型标识
FRAME_TYPE_I = 0x00  # I帧（关键帧）
FRAME_TYPE_P = 0x01  # P帧（差分帧）

def calculate_strip_heights(height: int, strip_count: int) -> list:
    """计算每个条带的高度，处理不能整除的情况"""
    base_height = height // strip_count
    remainder = height % strip_count
    
    strip_heights = []
    for i in range(strip_count):
        # 余数分配给前面的条带
        current_height = base_height + (1 if i < remainder else 0)
        strip_heights.append(current_height)
    
    return strip_heights

def pack_yuv9_strip(frame_bgr: np.ndarray, strip_y: int, strip_height: int) -> np.ndarray:
    """
    把指定条带的 240×strip_height×3 BGR → YUV9：每 4×4 像素 18 Byte
    布局按行优先：Y00..Y03 Y10..Y13 Y20..Y23 Y30..Y33 Cb Cr
    返回形状为 (strip_blocks_h, blocks_w, 18) 的数组，每个元素是一个4x4块
    """
    # 提取当前条带
    strip_bgr = frame_bgr[strip_y:strip_y + strip_height, :, :]
    
    B = strip_bgr[:, :, 0].astype(np.float32)
    G = strip_bgr[:, :, 1].astype(np.float32)
    R = strip_bgr[:, :, 2].astype(np.float32)

    Y  = (R*Y_COEFF[0]  + G*Y_COEFF[1]  + B*Y_COEFF[2]).round()
    Cb = (R*CB_COEFF[0] + G*CB_COEFF[1] + B*CB_COEFF[2]).round()
    Cr = (R*CR_COEFF[0] + G*CR_COEFF[1] + B*CR_COEFF[2]).round()

    Y  = np.clip(Y,  0, 255).astype(np.uint8)
    
    # 重组为块数组
    blocks_h = strip_height // BLOCK_H
    blocks_w = WIDTH // BLOCK_W
    block_array = np.zeros((blocks_h, blocks_w, BYTES_PER_BLOCK), dtype=np.uint8)
    
    for by in range(blocks_h):
        for bx in range(blocks_w):
            y = by * BLOCK_H
            x = bx * BLOCK_W
            # 确保不超出条带边界
            if y + BLOCK_H <= strip_height:
                # 16 Y值
                block_array[by, bx, :16] = Y[y:y+4, x:x+4].flatten()
                # Cb和Cr的平均值
                block_array[by, bx, 16] = Cb[y:y+4, x:x+4].mean().round().astype(np.uint8)
                block_array[by, bx, 17] = Cr[y:y+4, x:x+4].mean().round().astype(np.uint8)
    
    return block_array

def calculate_block_diff(block1: np.ndarray, block2: np.ndarray) -> float:
    """计算两个块的差异度（使用Y通道的平均绝对差值）"""
    # 只比较Y通道（前16个字节）
    y_diff = np.abs(block1[:16].astype(np.int16) - block2[:16].astype(np.int16))
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
            
            // 帧类型定义
            #define FRAME_TYPE_I        0x00
            #define FRAME_TYPE_P        0x01
            
            // 块参数
            #define BLOCK_WIDTH         4
            #define BLOCK_HEIGHT        4
            #define BYTES_PER_BLOCK     18

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
    pa = argparse.ArgumentParser(description="Encode to GBA YUV9 with strip-based inter-frame compression")
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
    while idx < grab_max:
        ret, frm = cap.read()
        if not ret:
            break
        if idx % every == 0:
            frm = cv2.resize(frm, (WIDTH, HEIGHT), cv2.INTER_AREA)
            
            # 将帧分割成条带
            frame_strips = []
            strip_y = 0
            for strip_height in strip_heights:
                strip_blocks = pack_yuv9_strip(frm, strip_y, strip_height)
                frame_strips.append(strip_blocks)
                strip_y += strip_height
            
            frames.append(frame_strips)
        idx += 1
    cap.release()

    if not frames:
        raise SystemExit("❌ 没有任何帧被采样")

    # 编码所有帧
    encoded_frames = []
    frame_offsets = []
    current_offset = 0
    prev_strips = [None] * args.strip_count
    i_frame_count = [0] * args.strip_count
    p_frame_count = [0] * args.strip_count
    
    for frame_idx, current_strips in enumerate(frames):
        frame_offsets.append(current_offset)
        
        # 决定是否强制I帧
        force_i_frame = (frame_idx % args.i_frame_interval == 0) or (frame_idx == 0)
        
        frame_data = bytearray()
        
        # 编码每个条带
        for strip_idx, current_strip in enumerate(current_strips):
            if force_i_frame or prev_strips[strip_idx] is None:
                # 编码为I帧
                strip_data = encode_strip_i_frame(current_strip)
                is_i_frame = True
                i_frame_count[strip_idx] += 1
            else:
                # 尝试差分编码
                strip_data, is_i_frame = encode_strip_differential(
                    current_strip, prev_strips[strip_idx], 
                    args.diff_threshold, args.force_i_threshold
                )
                if is_i_frame:
                    i_frame_count[strip_idx] += 1
                else:
                    p_frame_count[strip_idx] += 1
            
            # 存储条带长度（2字节）+ 条带数据
            frame_data.extend(struct.pack('<H', len(strip_data)))
            frame_data.extend(strip_data)
            
            # 更新参考条带
            prev_strips[strip_idx] = current_strip.copy() if current_strip.size > 0 else None
        
        encoded_frames.append(bytes(frame_data))
        current_offset += len(frame_data)
        
        # 打印编码信息
        strip_types = []
        for strip_idx in range(args.strip_count):
            if frame_idx == 0 or force_i_frame:
                strip_types.append("I")
            else:
                # 这里需要根据实际编码结果来判断，简化处理
                strip_types.append("?")
        
        print(f"帧 {frame_idx:4d}: 条带[{'/'.join(strip_types)}], {len(frame_data):6d} bytes")
    
    # 合并所有数据
    all_data = b''.join(encoded_frames)
    
    # 验证帧偏移
    print(f"\n帧偏移验证:")
    print(f"   前5帧偏移: {frame_offsets[:5]}")
    print(f"   最后一帧偏移: {frame_offsets[-1]}")
    print(f"   总数据大小: {len(all_data)}")
    
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
    
    print(f"\n✅ 编码完成：")
    print(f"   总帧数: {len(frames)}")
    print(f"   条带数: {args.strip_count}")
    print(f"   条带高度: {strip_heights}")
    
    for strip_idx in range(args.strip_count):
        print(f"   条带{strip_idx}: I帧{i_frame_count[strip_idx]}, P帧{p_frame_count[strip_idx]}")
    
    print(f"   原始大小: {original_size:,} bytes")
    print(f"   压缩后大小: {compressed_size:,} bytes")
    print(f"   压缩比: {compression_ratio:.2f}x")
    print(f"   I帧间隔: {args.i_frame_interval}")
    print(f"   差异阈值: {args.diff_threshold}")

if __name__ == "__main__":
    main()