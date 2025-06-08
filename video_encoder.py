#!/usr/bin/env python3
"""
gba_encode.py  v6  ——  把视频/图片序列转成 GBA Mode3 YUV9 数据（支持条带帧间差分 + 双码本向量量化）
输出 video_data.c / video_data.h
默认 5 s @ 30 fps，可用 --duration / --fps 修改，或使用 --full-duration 编码整个视频
支持条带处理，每个条带独立进行I/P帧编码 + 双码本压缩（细节码本255项，色块码本256项）
"""

import argparse, cv2, numpy as np, pathlib, textwrap
import struct
import concurrent.futures
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist

WIDTH, HEIGHT = 240, 160
DEFAULT_STRIP_COUNT = 4
DETAIL_CODEBOOK_SIZE = 255  # 细节码本大小（0xFF保留）
DEFAULT_COLOR_CODEBOOK_SIZE = 256   # 默认色块码本大小

Y_COEFF  = np.array([0.28571429,  0.57142857,  0.14285714])
CB_COEFF = np.array([-0.14285714, -0.28571429,  0.42857143])
CR_COEFF = np.array([ 0.35714286, -0.28571429, -0.07142857])
BLOCK_W, BLOCK_H = 2, 2
BYTES_PER_BLOCK  = 7  # 4Y + d_r + d_g + d_b

# 新增常量
ZONE_HEIGHT_PIXELS = 16  # 每个区域的像素高度
ZONE_HEIGHT_BIG_BLOCKS = ZONE_HEIGHT_PIXELS // (BLOCK_H * 2)  # 每个区域的4x4大块行数 (16像素 = 4行4x4大块)

# 帧类型标识
FRAME_TYPE_I = 0x00  # I帧（关键帧）
FRAME_TYPE_P = 0x01  # P帧（差分帧）

def calculate_strip_heights(height: int, strip_count: int) -> list:
    """计算每个条带的高度，确保每个条带高度都是4的倍数"""
    if height % 4 != 0:
        raise ValueError(f"视频高度 {height} 必须是4的倍数")
    
    base_height = (height // strip_count // 4) * 4
    remaining_height = height - (base_height * strip_count)
    
    strip_heights = []
    for i in range(strip_count):
        current_height = base_height
        if remaining_height >= 4:
            current_height += 4
            remaining_height -= 4
        strip_heights.append(current_height)
    
    if sum(strip_heights) != height:
        raise ValueError(f"条带高度分配错误: {strip_heights} 总和 {sum(strip_heights)} != {height}")
    
    for i, h in enumerate(strip_heights):
        if h % 4 != 0:
            raise ValueError(f"条带 {i} 高度 {h} 不是4的倍数")
    
    return strip_heights

def pack_yuv420_strip(frame_bgr: np.ndarray, strip_y: int, strip_height: int) -> np.ndarray:
    """向量化实现，把指定条带的 240×strip_height×3 BGR → YUV420"""
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

    Y_blocks  = Y.reshape(blocks_h, BLOCK_H, blocks_w, BLOCK_W)
    Cb_blocks = Cb.reshape(blocks_h, BLOCK_H, blocks_w, BLOCK_W)
    Cr_blocks = Cr.reshape(blocks_h, BLOCK_H, blocks_w, BLOCK_W)

    y_flat = (Y_blocks.transpose(0,2,1,3).reshape(blocks_h, blocks_w, 4) >> 1).astype(np.uint8)
    cb_mean = np.clip(Cb_blocks.mean(axis=(1,3)).round(), -128, 127).astype(np.int16)
    cr_mean = np.clip(Cr_blocks.mean(axis=(1,3)).round(), -128, 127).astype(np.int16)
    
    d_r = np.clip(cr_mean, -128, 127).astype(np.int8)
    d_g = np.clip((-(cb_mean >> 1) - cr_mean) >> 1, -128, 127).astype(np.int8)
    d_b = np.clip(cb_mean, -128, 127).astype(np.int8)

    block_array = np.zeros((blocks_h, blocks_w, BYTES_PER_BLOCK), dtype=np.uint8)
    block_array[..., 0:4] = y_flat
    block_array[..., 4] = d_r.view(np.uint8)
    block_array[..., 5] = d_g.view(np.uint8)
    block_array[..., 6] = d_b.view(np.uint8)
    
    return block_array

def calculate_block_variance(blocks_4x4: list) -> float:
    """计算4x4块的方差，用于判断是否为纯色块"""
    # 将4个2x2块合并为一个4x4的Y值数组
    y_values = []
    for block in blocks_4x4:
        y_values.extend(block[:4])  # 只取Y值
    
    y_array = np.array(y_values)
    return np.var(y_array)

def classify_4x4_blocks(blocks: np.ndarray, variance_threshold: float = 5.0) -> tuple:
    """将4x4块分类为纯色块和纹理块"""
    blocks_h, blocks_w = blocks.shape[:2]
    big_blocks_h = blocks_h // 2
    big_blocks_w = blocks_w // 2
    
    color_blocks = []  # 纯色块
    detail_blocks = []  # 纹理块
    block_types = {}   # 记录每个4x4块的类型 {(big_by, big_bx): 'color' or 'detail'}
    
    for big_by in range(big_blocks_h):
        for big_bx in range(big_blocks_w):
            # 收集4x4大块内的4个2x2小块
            blocks_4x4 = []
            for sub_by in range(2):
                for sub_bx in range(2):
                    by = big_by * 2 + sub_by
                    bx = big_bx * 2 + sub_bx
                    if by < blocks_h and bx < blocks_w:
                        blocks_4x4.append(blocks[by, bx])
                    else:
                        blocks_4x4.append(np.zeros(BYTES_PER_BLOCK, dtype=np.uint8))
            
            # 计算方差判断是否为纯色块
            variance = calculate_block_variance(blocks_4x4)
            
            if variance < variance_threshold:
                # 纯色块：计算平均值作为代表
                avg_block = np.mean(blocks_4x4, axis=0).round().astype(np.uint8)
                # 对于d_r, d_g, d_b需要特殊处理
                for i in range(4, 7):
                    avg_val = np.mean([b[i].view(np.int8) for b in blocks_4x4])
                    avg_block[i] = np.clip(avg_val, -128, 127).astype(np.int8).view(np.uint8)
                
                color_blocks.append(avg_block)
                block_types[(big_by, big_bx)] = 'color'
            else:
                # 纹理块：保留所有4个2x2块
                detail_blocks.extend(blocks_4x4)
                block_types[(big_by, big_bx)] = 'detail'
    
    return color_blocks, detail_blocks, block_types

def generate_dual_codebooks(all_color_blocks: list, all_detail_blocks: list, 
                          color_codebook_size: int = DEFAULT_COLOR_CODEBOOK_SIZE,
                          kmeans_max_iter: int = 100) -> tuple:
    """生成双码本：色块码本和细节码本"""
    # 生成色块码本
    if all_color_blocks:
        color_blocks_array = np.array(all_color_blocks)
        color_codebook, _ = generate_codebook(color_blocks_array, color_codebook_size, kmeans_max_iter)
    else:
        color_codebook = np.zeros((color_codebook_size, BYTES_PER_BLOCK), dtype=np.uint8)
    
    # 生成细节码本（只用255项，0xFF保留）
    if all_detail_blocks:
        detail_blocks_array = np.array(all_detail_blocks)
        detail_codebook, _ = generate_codebook(detail_blocks_array, DETAIL_CODEBOOK_SIZE, kmeans_max_iter)
        # 添加一个占位块使总大小为256
        detail_codebook_full = np.zeros((256, BYTES_PER_BLOCK), dtype=np.uint8)
        detail_codebook_full[:DETAIL_CODEBOOK_SIZE] = detail_codebook[:DETAIL_CODEBOOK_SIZE]
        detail_codebook_full[255] = detail_codebook_full[254]  # 复制最后一个有效项作为占位
    else:
        detail_codebook_full = np.zeros((256, BYTES_PER_BLOCK), dtype=np.uint8)
    
    return color_codebook, detail_codebook_full

def encode_strip_i_frame_dual_vq(blocks: np.ndarray, color_codebook: np.ndarray, 
                                detail_codebook: np.ndarray, block_types: dict) -> bytes:
    """编码条带I帧（双码本）"""
    data = bytearray()
    data.append(FRAME_TYPE_I)
    
    if blocks.size > 0:
        blocks_h, blocks_w = blocks.shape[:2]
        big_blocks_h = blocks_h // 2
        big_blocks_w = blocks_w // 2
        
        # 存储两个码本
        # 先存储细节码本（256项，包括占位）
        data.extend(detail_codebook.flatten().tobytes())
        # 再存储色块码本（可配置项数）
        data.extend(color_codebook.flatten().tobytes())
        
        # 按4x4大块的顺序编码
        for big_by in range(big_blocks_h):
            for big_bx in range(big_blocks_w):
                if (big_by, big_bx) in block_types and block_types[(big_by, big_bx)] == 'color':
                    # 色块：标记0xFF + 1个色块码本索引
                    data.append(0xFF)
                    
                    # 收集4x4块用于计算平均值
                    blocks_4x4 = []
                    for sub_by in range(2):
                        for sub_bx in range(2):
                            by = big_by * 2 + sub_by
                            bx = big_bx * 2 + sub_bx
                            if by < blocks_h and bx < blocks_w:
                                blocks_4x4.append(blocks[by, bx])
                    
                    # 计算平均块并量化
                    avg_block = np.mean(blocks_4x4, axis=0).round().astype(np.uint8)
                    for i in range(4, 7):
                        avg_val = np.mean([b[i].view(np.int8) for b in blocks_4x4])
                        avg_block[i] = np.clip(avg_val, -128, 127).astype(np.int8).view(np.uint8)
                    
                    color_idx = quantize_blocks(avg_block.reshape(1, -1), color_codebook)[0]
                    data.append(color_idx)
                else:
                    # 纹理块：4个细节码本索引
                    for sub_by in range(2):
                        for sub_bx in range(2):
                            by = big_by * 2 + sub_by
                            bx = big_bx * 2 + sub_bx
                            if by < blocks_h and bx < blocks_w:
                                block = blocks[by, bx]
                                detail_idx = quantize_blocks(block.reshape(1, -1), detail_codebook[:DETAIL_CODEBOOK_SIZE])[0]
                                data.append(detail_idx)
                            else:
                                data.append(0)
    
    return bytes(data)

def encode_strip_differential_dual_vq(current_blocks: np.ndarray, prev_blocks: np.ndarray,
                                     color_codebook: np.ndarray, detail_codebook: np.ndarray,
                                     block_types: dict, diff_threshold: float,
                                     force_i_threshold: float = 0.7) -> tuple:
    """差分编码当前条带（双码本）- 使用区域优化"""
    if prev_blocks is None or current_blocks.shape != prev_blocks.shape:
        return encode_strip_i_frame_dual_vq(current_blocks, color_codebook, detail_codebook, block_types), True
    
    blocks_h, blocks_w = current_blocks.shape[:2]
    total_blocks = blocks_h * blocks_w
    
    if total_blocks == 0:
        return b'', True
    
    # 计算块差异
    current_flat = current_blocks.reshape(-1, BYTES_PER_BLOCK)
    prev_flat = prev_blocks.reshape(-1, BYTES_PER_BLOCK)
    
    y_current = current_flat[:, :4].astype(np.int16)
    y_prev = prev_flat[:, :4].astype(np.int16)
    y_diff = np.abs(y_current - y_prev)
    block_diffs_flat = y_diff.mean(axis=1)
    block_diffs = block_diffs_flat.reshape(blocks_h, blocks_w)
    
    big_blocks_h = blocks_h // 2
    big_blocks_w = blocks_w // 2
    
    # 计算区域数量
    zones_count = (big_blocks_h + ZONE_HEIGHT_BIG_BLOCKS - 1) // ZONE_HEIGHT_BIG_BLOCKS  # 向上取整
    if zones_count > 8:
        zones_count = 8  # 限制最多8个区域（u8 bitmap）
    
    # 按区域组织更新
    zone_detail_updates = [[] for _ in range(zones_count)]  # 每个区域的纹理块更新
    zone_color_updates = [[] for _ in range(zones_count)]   # 每个区域的色块更新
    total_updated_blocks = 0
    
    for big_by in range(big_blocks_h):
        for big_bx in range(big_blocks_w):
            needs_update = False
            positions = [
                (big_by * 2, big_bx * 2),
                (big_by * 2, big_bx * 2 + 1),
                (big_by * 2 + 1, big_bx * 2),
                (big_by * 2 + 1, big_bx * 2 + 1)
            ]
            
            for by, bx in positions:
                if by < blocks_h and bx < blocks_w:
                    if block_diffs[by, bx] > diff_threshold:
                        needs_update = True
                        break
            
            if needs_update:
                # 计算属于哪个区域
                zone_idx = min(big_by // ZONE_HEIGHT_BIG_BLOCKS, zones_count - 1)
                # 计算在区域内的相对坐标
                zone_relative_by = big_by % ZONE_HEIGHT_BIG_BLOCKS
                zone_relative_idx = zone_relative_by * big_blocks_w + big_bx
                
                total_updated_blocks += 4
                
                if (big_by, big_bx) in block_types and block_types[(big_by, big_bx)] == 'color':
                    # 色块更新
                    blocks_4x4 = []
                    for by, bx in positions:
                        if by < blocks_h and bx < blocks_w:
                            blocks_4x4.append(current_blocks[by, bx])
                    
                    avg_block = np.mean(blocks_4x4, axis=0).round().astype(np.uint8)
                    for i in range(4, 7):
                        avg_val = np.mean([b[i].view(np.int8) for b in blocks_4x4])
                        avg_block[i] = np.clip(avg_val, -128, 127).astype(np.int8).view(np.uint8)
                    
                    color_idx = quantize_blocks(avg_block.reshape(1, -1), color_codebook)[0]
                    zone_color_updates[zone_idx].append((zone_relative_idx, color_idx))
                else:
                    # 纹理块更新
                    indices = []
                    for by, bx in positions:
                        if by < blocks_h and bx < blocks_w:
                            block = current_blocks[by, bx]
                            detail_idx = quantize_blocks(block.reshape(1, -1), detail_codebook[:DETAIL_CODEBOOK_SIZE])[0]
                            indices.append(detail_idx)
                        else:
                            indices.append(0)
                    zone_detail_updates[zone_idx].append((zone_relative_idx, indices))
    
    # 判断是否需要I帧
    update_ratio = total_updated_blocks / total_blocks
    if update_ratio > force_i_threshold:
        return encode_strip_i_frame_dual_vq(current_blocks, color_codebook, detail_codebook, block_types), True
    
    # 编码P帧
    data = bytearray()
    data.append(FRAME_TYPE_P)
    
    # 生成区域bitmap
    zone_bitmap = 0
    for zone_idx in range(zones_count):
        if zone_detail_updates[zone_idx] or zone_color_updates[zone_idx]:
            zone_bitmap |= (1 << zone_idx)
    
    data.append(zone_bitmap)
    
    # 按区域编码更新
    for zone_idx in range(zones_count):
        if zone_bitmap & (1 << zone_idx):
            # 编码该区域的更新
            detail_updates = zone_detail_updates[zone_idx]
            color_updates = zone_color_updates[zone_idx]
            
            # 存储纹理块更新数量和色块更新数量
            data.extend(struct.pack('<H', len(detail_updates)))
            data.extend(struct.pack('<H', len(color_updates)))
            
            # 存储纹理块更新
            for relative_idx, indices in detail_updates:
                data.append(relative_idx)  # 使用u8而不是u16
                for idx in indices:
                    data.append(idx)
            
            # 存储色块更新
            for relative_idx, color_idx in color_updates:
                data.append(relative_idx)  # 使用u8而不是u16
                data.append(color_idx)
    
    return bytes(data), False

def generate_codebook(blocks_data: np.ndarray, codebook_size: int, max_iter: int = 100) -> tuple:
    """使用K-Means聚类生成码表"""
    if len(blocks_data) == 0:
        return np.zeros((codebook_size, BYTES_PER_BLOCK), dtype=np.uint8), 0
    
    if blocks_data.ndim > 2:
        blocks_data = blocks_data.reshape(-1, BYTES_PER_BLOCK)
    
    blocks_as_tuples = [tuple(block) for block in blocks_data]
    unique_tuples = list(set(blocks_as_tuples))
    unique_blocks = np.array(unique_tuples, dtype=np.uint8)
    
    effective_size = min(len(unique_blocks), codebook_size)
    
    if len(unique_blocks) <= codebook_size:
        codebook = np.zeros((codebook_size, BYTES_PER_BLOCK), dtype=np.uint8)
        codebook[:len(unique_blocks)] = unique_blocks
        if len(unique_blocks) > 0:
            for i in range(len(unique_blocks), codebook_size):
                codebook[i] = unique_blocks[-1]
        return codebook, effective_size
    
    kmeans = MiniBatchKMeans(
        n_clusters=codebook_size,
        random_state=42,
        batch_size=min(1000, len(blocks_data)),
        max_iter=max_iter,
        n_init=3
    )
    blocks_for_clustering = convert_blocks_for_clustering(blocks_data)
    kmeans.fit(blocks_for_clustering)
    codebook = convert_codebook_from_clustering(kmeans.cluster_centers_)
    
    return codebook, codebook_size

def quantize_blocks(blocks_data: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """使用码表对块进行量化"""
    if len(blocks_data) == 0:
        return np.array([], dtype=np.uint8)
    
    blocks_for_clustering = convert_blocks_for_clustering(blocks_data)
    codebook_for_clustering = convert_blocks_for_clustering(codebook)
    
    blocks_expanded = blocks_for_clustering[:, np.newaxis, :]
    codebook_expanded = codebook_for_clustering[np.newaxis, :, :]
    
    diff = blocks_expanded - codebook_expanded
    squared_distances = np.sum(diff * diff, axis=2)
    indices = np.argmin(squared_distances, axis=1).astype(np.uint8)
    
    return indices

def convert_blocks_for_clustering(blocks_data: np.ndarray) -> np.ndarray:
    """将块数据转换为正确的聚类格式"""
    if len(blocks_data) == 0:
        return blocks_data.astype(np.float32)
    
    if blocks_data.ndim > 2:
        blocks_data = blocks_data.reshape(-1, BYTES_PER_BLOCK)
    
    blocks_float = blocks_data.astype(np.float32)
    
    for i in range(4, BYTES_PER_BLOCK):
        blocks_float[:, i] = blocks_data[:, i].view(np.int8).astype(np.float32)
    
    return blocks_float

def convert_codebook_from_clustering(codebook_float: np.ndarray) -> np.ndarray:
    """将聚类结果转换回正确的块格式"""
    codebook = np.zeros_like(codebook_float, dtype=np.uint8)
    
    codebook[:, 0:4] = np.clip(codebook_float[:, 0:4].round(), 0, 255).astype(np.uint8)
    
    for i in range(4, BYTES_PER_BLOCK):
        clipped_values = np.clip(codebook_float[:, i].round(), -128, 127).astype(np.int8)
        codebook[:, i] = clipped_values.view(np.uint8)
    
    return codebook

def generate_gop_dual_codebooks(frames: list, strip_count: int, i_frame_interval: int,
                              variance_threshold: float, color_codebook_size: int = DEFAULT_COLOR_CODEBOOK_SIZE,
                              kmeans_max_iter: int = 100) -> dict:
    """为每个GOP生成双码本"""
    print("正在为每个GOP生成双码本...")
    
    gop_codebooks = {}
    
    i_frame_positions = []
    for frame_idx in range(len(frames)):
        if frame_idx % i_frame_interval == 0:
            i_frame_positions.append(frame_idx)
    
    for gop_idx, gop_start in enumerate(i_frame_positions):
        if gop_idx + 1 < len(i_frame_positions):
            gop_end = i_frame_positions[gop_idx + 1]
        else:
            gop_end = len(frames)
        
        print(f"  处理GOP {gop_idx}: 帧 {gop_start} 到 {gop_end-1}")
        
        gop_codebooks[gop_start] = []
        
        for strip_idx in range(strip_count):
            all_color_blocks = []
            all_detail_blocks = []
            block_types_list = []
            
            for frame_idx in range(gop_start, gop_end):
                strip_blocks = frames[frame_idx][strip_idx]
                if strip_blocks.size > 0:
                    color_blocks, detail_blocks, block_types = classify_4x4_blocks(strip_blocks, variance_threshold)
                    all_color_blocks.extend(color_blocks)
                    all_detail_blocks.extend(detail_blocks)
                    block_types_list.append((frame_idx, block_types))
            
            # 生成双码本
            color_codebook, detail_codebook = generate_dual_codebooks(
                all_color_blocks, all_detail_blocks, color_codebook_size, kmeans_max_iter
            )
            
            gop_codebooks[gop_start].append({
                'color_codebook': color_codebook,
                'detail_codebook': detail_codebook,
                'block_types_list': block_types_list,
                'color_blocks_count': len(all_color_blocks),
                'detail_blocks_count': len(all_detail_blocks) // 4  # 4个2x2块组成一个4x4块
            })
            
            print(f"    条带{strip_idx}: 色块{len(all_color_blocks)}, 纹理块{len(all_detail_blocks)//4}")
    
    return gop_codebooks

def write_header(path_h: pathlib.Path, frame_cnt: int, total_bytes: int, strip_count: int, 
                strip_heights: list, color_codebook_size: int):
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
            #define DETAIL_CODEBOOK_SIZE {DETAIL_CODEBOOK_SIZE}
            #define COLOR_CODEBOOK_SIZE  {color_codebook_size}
            
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
        
        f.write("const unsigned char strip_heights[] = {\n")
        f.write("    " + ', '.join(map(str, strip_heights)) + "\n")
        f.write("};\n\n")
        
        f.write("const unsigned int frame_offsets[] = {\n")
        for i in range(0, len(frame_offsets), 8):
            chunk = ', '.join(f"{offset}" for offset in frame_offsets[i:i+8])
            f.write("    " + chunk + ",\n")
        f.write("};\n\n")
        
        f.write("const unsigned char video_data[] = {\n")
        per_line = 16
        for i in range(0, len(data), per_line):
            chunk = ', '.join(f"0x{v:02X}" for v in data[i:i+per_line])
            f.write("    " + chunk + ",\n")
        f.write("};\n")

def main():
    pa = argparse.ArgumentParser(description="Encode to GBA YUV9 with dual codebook")
    pa.add_argument("input")
    pa.add_argument("--duration", type=float, default=5.0)
    pa.add_argument("--full-duration", action="store_true")
    pa.add_argument("--fps", type=int, default=30)
    pa.add_argument("--out", default="video_data")
    pa.add_argument("--strip-count", type=int, default=DEFAULT_STRIP_COUNT)
    pa.add_argument("--i-frame-interval", type=int, default=60)
    pa.add_argument("--diff-threshold", type=float, default=2.0)
    pa.add_argument("--force-i-threshold", type=float, default=0.7)
    pa.add_argument("--variance-threshold", type=float, default=5.0,
                   help="方差阈值，用于区分纯色块和纹理块（默认5.0）")
    pa.add_argument("--color-codebook-size", type=int, default=DEFAULT_COLOR_CODEBOOK_SIZE,
                   help=f"色块码本大小（默认{DEFAULT_COLOR_CODEBOOK_SIZE}）")
    pa.add_argument("--kmeans-max-iter", type=int, default=200)
    pa.add_argument("--threads", type=int, default=None)
    args = pa.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit("❌ 打不开输入文件")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    every = int(round(src_fps / args.fps))
    
    if args.full_duration:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        grab_max = total_frames
        actual_duration = total_frames / src_fps
        print(f"编码整个视频: {total_frames} 帧，时长 {actual_duration:.2f} 秒")
    else:
        grab_max = int(args.duration * src_fps)
        print(f"编码时长: {args.duration} 秒 ({grab_max} 帧)")

    strip_heights = calculate_strip_heights(HEIGHT, args.strip_count)
    print(f"条带配置: {args.strip_count} 个条带，高度分别为: {strip_heights}")
    print(f"码本配置: 细节码本{DETAIL_CODEBOOK_SIZE}项，色块码本{args.color_codebook_size}项")

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
                strip_y_list = []
                y = 0
                for strip_height in strip_heights:
                    strip_y_list.append((frm, y, strip_height))
                    y += strip_height
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

    # 生成双码本
    gop_codebooks = generate_gop_dual_codebooks(
        frames, args.strip_count, args.i_frame_interval, 
        args.variance_threshold, args.color_codebook_size, args.kmeans_max_iter
    )

    # 编码所有帧
    print("正在编码帧...")
    encoded_frames = []
    frame_offsets = []
    current_offset = 0
    prev_strips = [None] * args.strip_count
    
    for frame_idx, current_strips in enumerate(frames):
        frame_offsets.append(current_offset)
        
        # 找到当前GOP
        gop_start = (frame_idx // args.i_frame_interval) * args.i_frame_interval
        gop_data = gop_codebooks[gop_start]
        
        frame_data = bytearray()
        
        for strip_idx, current_strip in enumerate(current_strips):
            strip_gop_data = gop_data[strip_idx]
            color_codebook = strip_gop_data['color_codebook']
            detail_codebook = strip_gop_data['detail_codebook']
            
            # 找到当前帧的block_types
            block_types = None
            for fid, bt in strip_gop_data['block_types_list']:
                if fid == frame_idx:
                    block_types = bt
                    break
            
            force_i_frame = (frame_idx % args.i_frame_interval == 0) or frame_idx == 0
            
            if force_i_frame or prev_strips[strip_idx] is None:
                strip_data = encode_strip_i_frame_dual_vq(
                    current_strip, color_codebook, detail_codebook, block_types
                )
                is_i_frame = True
            else:
                strip_data, is_i_frame = encode_strip_differential_dual_vq(
                    current_strip, prev_strips[strip_idx],
                    color_codebook, detail_codebook, block_types,
                    args.diff_threshold, args.force_i_threshold
                )
            
            frame_data.extend(struct.pack('<H', len(strip_data)))
            frame_data.extend(strip_data)
            
            prev_strips[strip_idx] = current_strip.copy() if current_strip.size > 0 else None
        
        encoded_frames.append(bytes(frame_data))
        current_offset += len(frame_data)
        
        if frame_idx % 30 == 0 or frame_idx == len(frames) - 1:
            print(f"  已编码 {frame_idx + 1}/{len(frames)} 帧")
    
    all_data = b''.join(encoded_frames)
    
    write_header(pathlib.Path(args.out).with_suffix(".h"), len(frames), len(all_data), 
                args.strip_count, strip_heights, args.color_codebook_size)
    write_source(pathlib.Path(args.out).with_suffix(".c"), all_data, frame_offsets, strip_heights)
    
    # 统计信息
    total_color_blocks = sum(sum(d['color_blocks_count'] for d in gop_data) for gop_data in gop_codebooks.values())
    total_detail_blocks = sum(sum(d['detail_blocks_count'] for d in gop_data) for gop_data in gop_codebooks.values())
    
    print(f"\n✅ 编码完成：")
    print(f"   总帧数: {len(frames)}")
    print(f"   条带数: {args.strip_count}")
    print(f"   GOP数量: {len(gop_codebooks)}")
    print(f"   总色块数: {total_color_blocks}")
    print(f"   总纹理块数: {total_detail_blocks}")
    print(f"   压缩后大小: {len(all_data):,} bytes")

if __name__ == "__main__":
    main()