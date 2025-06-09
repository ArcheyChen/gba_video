#!/usr/bin/env python3
"""
gba_encode.py  v8  ——  把视频/图片序列转成 GBA Mode3 YUV9 数据（支持条带帧间差分 + 统一码本向量量化）
输出 video_data.c / video_data.h
默认 5 s @ 30 fps，可用 --duration / --fps 修改，或使用 --full-duration 编码整个视频
支持条带处理，每个条带独立进行I/P帧编码 + 统一码本压缩（有效255项，0xFF保留作为色块标记）
现在使用4x4码表和8x8大块
"""

import argparse, cv2, numpy as np, pathlib, textwrap
import struct
import concurrent.futures
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from collections import defaultdict
import statistics

WIDTH, HEIGHT = 240, 160
DEFAULT_STRIP_COUNT = 4
DEFAULT_UNIFIED_CODEBOOK_SIZE = 256   # 统一码本大小
EFFECTIVE_UNIFIED_CODEBOOK_SIZE = 255  # 有效码本大小（0xFF保留）

Y_COEFF  = np.array([0.28571429,  0.57142857,  0.14285714])
CB_COEFF = np.array([-0.14285714, -0.28571429,  0.42857143])
CR_COEFF = np.array([ 0.35714286, -0.28571429, -0.07142857])
BLOCK_W, BLOCK_H = 2, 2
BYTES_PER_BLOCK  = 19  # 16Y + d_r + d_g + d_b (4x4块)

# 新增常量 - 8x8大块
BIG_BLOCK_W = 8  # 8x8大块宽度
BIG_BLOCK_H = 8  # 8x8大块高度
ZONE_HEIGHT_PIXELS = HEIGHT  # 每个区域覆盖整个条带高度
ZONE_WIDTH_PIXELS = WIDTH    # 每个区域覆盖整个条带宽度

# 帧类型标识
FRAME_TYPE_I = 0x00  # I帧（关键帧）
FRAME_TYPE_P = 0x01  # P帧（差分帧）

def calculate_strip_heights(height: int, strip_count: int) -> list:
    """计算每个条带的高度，确保每个条带高度都是8的倍数（适配8x8大块）"""
    if height % 8 != 0:
        raise ValueError(f"视频高度 {height} 必须是8的倍数")
    
    base_height = (height // strip_count // 8) * 8
    remaining_height = height - (base_height * strip_count)
    
    strip_heights = []
    for i in range(strip_count):
        current_height = base_height
        if remaining_height >= 8:
            current_height += 8
            remaining_height -= 8
        strip_heights.append(current_height)
    
    if sum(strip_heights) != height:
        raise ValueError(f"条带高度分配错误: {strip_heights} 总和 {sum(strip_heights)} != {height}")
    
    for i, h in enumerate(strip_heights):
        if h % 8 != 0:
            raise ValueError(f"条带 {i} 高度 {h} 不是8的倍数")
    
    return strip_heights

def pack_yuv420_strip(frame_bgr: np.ndarray, strip_y: int, strip_height: int) -> np.ndarray:
    """向量化实现，把指定条带的 240×strip_height×3 BGR → YUV420，现在输出4x4块"""
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
    # 改为4x4块处理
    blocks_h = h // 4
    blocks_w = w // 4

    Y_blocks  = Y.reshape(blocks_h, 4, blocks_w, 4)
    Cb_blocks = Cb.reshape(blocks_h, 4, blocks_w, 4)
    Cr_blocks = Cr.reshape(blocks_h, 4, blocks_w, 4)

    # 4x4块的Y值直接量化
    y_flat = (Y_blocks.transpose(0,2,1,3).reshape(blocks_h, blocks_w, 16) >> 1).astype(np.uint8)
    cb_mean = np.clip(Cb_blocks.mean(axis=(1,3)).round(), -128, 127).astype(np.int16)
    cr_mean = np.clip(Cr_blocks.mean(axis=(1,3)).round(), -128, 127).astype(np.int16)
    
    d_r = np.clip(cr_mean, -128, 127).astype(np.int8)
    d_g = np.clip((-(cb_mean >> 1) - cr_mean) >> 1, -128, 127).astype(np.int8)
    d_b = np.clip(cb_mean, -128, 127).astype(np.int8)

    block_array = np.zeros((blocks_h, blocks_w, BYTES_PER_BLOCK), dtype=np.uint8)
    block_array[..., 0:16] = y_flat  # 16个Y值
    block_array[..., 16] = d_r.view(np.uint8)
    block_array[..., 17] = d_g.view(np.uint8)
    block_array[..., 18] = d_b.view(np.uint8)
    
    return block_array

def calculate_block_variance(blocks_8x8: list) -> float:
    """计算8x8块的方差，用于判断是否为纯色块"""
    # 将4个4x4块合并为一个8x8的Y值数组
    y_values = []
    for block in blocks_8x8:
        y_values.extend(block[:16])  # 只取Y值
    
    y_array = np.array(y_values)
    return np.var(y_array)

def classify_8x8_blocks(blocks: np.ndarray, variance_threshold: float = 5.0) -> tuple:
    """将8x8块分类为纯色块和纹理块"""
    blocks_h, blocks_w = blocks.shape[:2]
    big_blocks_h = blocks_h // 2  # 每2行4x4块组成1行8x8块
    big_blocks_w = blocks_w // 2  # 每2列4x4块组成1列8x8块
    
    color_blocks = []  # 纯色块
    detail_blocks = []  # 纹理块
    block_types = {}   # 记录每个8x8块的类型 {(big_by, big_bx): 'color' or 'detail'}
    
    for big_by in range(big_blocks_h):
        for big_bx in range(big_blocks_w):
            # 收集8x8大块内的4个4x4小块
            blocks_8x8 = []
            for sub_by in range(2):
                for sub_bx in range(2):
                    by = big_by * 2 + sub_by
                    bx = big_bx * 2 + sub_bx
                    if by < blocks_h and bx < blocks_w:
                        blocks_8x8.append(blocks[by, bx])
                    else:
                        blocks_8x8.append(np.zeros(BYTES_PER_BLOCK, dtype=np.uint8))
            
            # 计算方差判断是否为纯色块
            variance = calculate_block_variance(blocks_8x8)
            
            if variance < variance_threshold:
                # 纯色块：计算平均值作为代表
                avg_block = np.mean(blocks_8x8, axis=0).round().astype(np.uint8)
                # 对于d_r, d_g, d_b需要特殊处理
                for i in range(16, 19):
                    avg_val = np.mean([b[i].view(np.int8) for b in blocks_8x8])
                    avg_block[i] = np.clip(avg_val, -128, 127).astype(np.int8).view(np.uint8)
                
                color_blocks.append(avg_block)
                block_types[(big_by, big_bx)] = 'color'
            else:
                # 纹理块：保留所有4个4x4块
                detail_blocks.extend(blocks_8x8)
                block_types[(big_by, big_bx)] = 'detail'
    
    return color_blocks, detail_blocks, block_types

def classify_8x8_blocks_unified(blocks: np.ndarray, variance_threshold: float = 5.0) -> tuple:
    """将8x8块分类为纯色块和纹理块，用于统一码本"""
    blocks_h, blocks_w = blocks.shape[:2]
    big_blocks_h = blocks_h // 2
    big_blocks_w = blocks_w // 2
    
    all_blocks = []  # 所有4x4块
    block_types = {}   # 记录每个8x8块的类型和对应的4x4块索引
    
    for big_by in range(big_blocks_h):
        for big_bx in range(big_blocks_w):
            # 收集8x8大块内的4个4x4小块
            blocks_8x8 = []
            for sub_by in range(2):
                for sub_bx in range(2):
                    by = big_by * 2 + sub_by
                    bx = big_bx * 2 + sub_bx
                    if by < blocks_h and bx < blocks_w:
                        blocks_8x8.append(blocks[by, bx])
                    else:
                        blocks_8x8.append(np.zeros(BYTES_PER_BLOCK, dtype=np.uint8))
            
            # 计算方差判断是否为纯色块
            variance = calculate_block_variance(blocks_8x8)
            
            if variance < variance_threshold:
                # 纯色块：计算平均值作为一个4x4块
                avg_block = np.mean(blocks_8x8, axis=0).round().astype(np.uint8)
                for i in range(16, 19):
                    avg_val = np.mean([b[i].view(np.int8) for b in blocks_8x8])
                    avg_block[i] = np.clip(avg_val, -128, 127).astype(np.int8).view(np.uint8)
                
                block_idx = len(all_blocks)
                all_blocks.append(avg_block)
                block_types[(big_by, big_bx)] = ('color', [block_idx])
            else:
                # 纹理块：保留所有4个4x4块
                block_indices = []
                for block in blocks_8x8:
                    block_idx = len(all_blocks)
                    all_blocks.append(block)
                    block_indices.append(block_idx)
                block_types[(big_by, big_bx)] = ('detail', block_indices)
    
    return all_blocks, block_types

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

def generate_unified_codebook(all_blocks: list, codebook_size: int = DEFAULT_UNIFIED_CODEBOOK_SIZE,
                             kmeans_max_iter: int = 100) -> np.ndarray:
    """生成统一码本（保留0xFF作为特殊标记）"""
    if all_blocks:
        blocks_array = np.array(all_blocks)
        # 只使用255项有效码本，保留0xFF
        effective_size = min(codebook_size - 1, EFFECTIVE_UNIFIED_CODEBOOK_SIZE)
        codebook, _ = generate_codebook(blocks_array, effective_size, kmeans_max_iter)
        
        # 创建完整的256项码本
        full_codebook = np.zeros((codebook_size, BYTES_PER_BLOCK), dtype=np.uint8)
        full_codebook[:effective_size] = codebook[:effective_size]
        # 第255项（索引255/0xFF）复制最后一个有效项作为占位
        if effective_size > 0:
            full_codebook[255] = full_codebook[effective_size - 1]
    else:
        full_codebook = np.zeros((codebook_size, BYTES_PER_BLOCK), dtype=np.uint8)
    
    return full_codebook

def quantize_blocks_unified(blocks_data: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """使用统一码表对块进行量化（避免产生0xFF）"""
    if len(blocks_data) == 0:
        return np.array([], dtype=np.uint8)
    
    # 只使用前255项进行量化
    effective_codebook = codebook[:EFFECTIVE_UNIFIED_CODEBOOK_SIZE]
    
    blocks_for_clustering = convert_blocks_for_clustering(blocks_data)
    codebook_for_clustering = convert_blocks_for_clustering(effective_codebook)
    
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
    
    for i in range(16, BYTES_PER_BLOCK):  # 从第16个字节开始是颜色差分
        blocks_float[:, i] = blocks_data[:, i].view(np.int8).astype(np.float32)
    
    return blocks_float

def convert_codebook_from_clustering(codebook_float: np.ndarray) -> np.ndarray:
    """将聚类结果转换回正确的块格式"""
    codebook = np.zeros_like(codebook_float, dtype=np.uint8)
    
    codebook[:, 0:16] = np.clip(codebook_float[:, 0:16].round(), 0, 255).astype(np.uint8)
    
    for i in range(16, BYTES_PER_BLOCK):
        clipped_values = np.clip(codebook_float[:, i].round(), -128, 127).astype(np.int8)
        codebook[:, i] = clipped_values.view(np.uint8)
    
    return codebook

def encode_strip_i_frame_unified(blocks: np.ndarray, unified_codebook: np.ndarray, 
                                block_types: dict) -> bytes:
    """编码条带I帧（统一码本）"""
    data = bytearray()
    data.append(FRAME_TYPE_I)
    
    if blocks.size > 0:
        blocks_h, blocks_w = blocks.shape[:2]
        big_blocks_h = blocks_h // 2
        big_blocks_w = blocks_w // 2
        
        # 存储统一码本
        data.extend(unified_codebook.flatten().tobytes())
        
        # 按8x8大块的顺序编码
        for big_by in range(big_blocks_h):
            for big_bx in range(big_blocks_w):
                if (big_by, big_bx) in block_types:
                    block_type, block_indices = block_types[(big_by, big_bx)]
                    
                    if block_type == 'color':
                        # 色块：标记0xFF + 1个码本索引
                        data.append(0xFF)
                        
                        # 从原始blocks重建平均块
                        blocks_8x8 = []
                        for sub_by in range(2):
                            for sub_bx in range(2):
                                by = big_by * 2 + sub_by
                                bx = big_bx * 2 + sub_bx
                                if by < blocks_h and bx < blocks_w:
                                    blocks_8x8.append(blocks[by, bx])
                        
                        avg_block = np.mean(blocks_8x8, axis=0).round().astype(np.uint8)
                        for i in range(16, 19):
                            avg_val = np.mean([b[i].view(np.int8) for b in blocks_8x8])
                            avg_block[i] = np.clip(avg_val, -128, 127).astype(np.int8).view(np.uint8)
                        
                        unified_idx = quantize_blocks_unified(avg_block.reshape(1, -1), unified_codebook)[0]
                        data.append(unified_idx)
                    else:
                        # 纹理块：4个码本索引
                        for sub_by in range(2):
                            for sub_bx in range(2):
                                by = big_by * 2 + sub_by
                                bx = big_bx * 2 + sub_bx
                                if by < blocks_h and bx < blocks_w:
                                    block = blocks[by, bx]
                                    unified_idx = quantize_blocks_unified(block.reshape(1, -1), unified_codebook)[0]
                                    data.append(unified_idx)
                                else:
                                    data.append(0)
    
    return bytes(data)

def encode_strip_differential_unified(current_blocks: np.ndarray, prev_blocks: np.ndarray,
                                     unified_codebook: np.ndarray, block_types: dict, 
                                     diff_threshold: float, force_i_threshold: float = 0.7) -> tuple:
    """差分编码当前条带（统一码本）"""
    if prev_blocks is None or current_blocks.shape != prev_blocks.shape:
        i_frame_data = encode_strip_i_frame_unified(current_blocks, unified_codebook, block_types)
        return i_frame_data, True, 0, 0, 0
    
    blocks_h, blocks_w = current_blocks.shape[:2]
    total_blocks = blocks_h * blocks_w
    
    if total_blocks == 0:
        return b'', True, 0, 0, 0
    
    # 计算块差异
    current_flat = current_blocks.reshape(-1, BYTES_PER_BLOCK)
    prev_flat = prev_blocks.reshape(-1, BYTES_PER_BLOCK)
    
    y_current = current_flat[:, :16].astype(np.int16)  # 16个Y值
    y_prev = prev_flat[:, :16].astype(np.int16)
    y_diff = np.abs(y_current - y_prev)
    block_diffs_flat = y_diff.mean(axis=1)
    block_diffs = block_diffs_flat.reshape(blocks_h, blocks_w)
    
    big_blocks_h = blocks_h // 2
    big_blocks_w = blocks_w // 2
    
    # 检查8x8大块总数是否超出u8范围
    max_big_blocks = big_blocks_h * big_blocks_w
    if max_big_blocks > 255:
        print(f"警告: 8x8大块数量 {max_big_blocks} 超出u8范围，强制使用I帧")
        i_frame_data = encode_strip_i_frame_unified(current_blocks, unified_codebook, block_types)
        return i_frame_data, True, 0, 0, 0
    
    # 现在每个条带就是一个zone
    zone_detail_updates = []
    zone_color_updates = []
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
                # 计算在条带内的相对坐标
                big_block_idx = big_by * big_blocks_w + big_bx
                
                # 确保索引在u8范围内
                if big_block_idx > 255:
                    print(f"警告: big_block_idx {big_block_idx} 超出u8范围，跳过此块")
                    continue
                
                total_updated_blocks += 4
                
                if (big_by, big_bx) in block_types and block_types[(big_by, big_bx)][0] == 'color':
                    # 色块更新
                    blocks_8x8 = []
                    for by, bx in positions:
                        if by < blocks_h and bx < blocks_w:
                            blocks_8x8.append(current_blocks[by, bx])
                    
                    avg_block = np.mean(blocks_8x8, axis=0).round().astype(np.uint8)
                    for i in range(16, 19):
                        avg_val = np.mean([b[i].view(np.int8) for b in blocks_8x8])
                        avg_block[i] = np.clip(avg_val, -128, 127).astype(np.int8).view(np.uint8)
                    
                    color_idx = quantize_blocks_unified(avg_block.reshape(1, -1), unified_codebook)[0]
                    zone_color_updates.append((big_block_idx, color_idx))
                else:
                    # 纹理块更新
                    indices = []
                    for by, bx in positions:
                        if by < blocks_h and bx < blocks_w:
                            block = current_blocks[by, bx]
                            unified_idx = quantize_blocks_unified(block.reshape(1, -1), unified_codebook)[0]
                            indices.append(unified_idx)
                        else:
                            indices.append(0)
                    zone_detail_updates.append((big_block_idx, indices))
    
    # 检查更新数量是否超出u8范围
    if len(zone_detail_updates) > 255 or len(zone_color_updates) > 255:
        print(f"警告: 更新数量超出u8范围 (纹理:{len(zone_detail_updates)}, 色块:{len(zone_color_updates)})，强制使用I帧")
        i_frame_data = encode_strip_i_frame_unified(current_blocks, unified_codebook, block_types)
        return i_frame_data, True, 0, 0, 0
    
    # 判断是否需要I帧
    update_ratio = total_updated_blocks / total_blocks
    if update_ratio > force_i_threshold:
        i_frame_data = encode_strip_i_frame_unified(current_blocks, unified_codebook, block_types)
        return i_frame_data, True, 0, 0, 0
    
    # 编码P帧
    data = bytearray()
    data.append(FRAME_TYPE_P)
    
    # 编码更新
    data.append(len(zone_detail_updates))
    data.append(len(zone_color_updates))
    
    # 存储纹理块更新
    for big_block_idx, indices in zone_detail_updates:
        data.append(big_block_idx)
        for idx in indices:
            data.append(idx)
    
    # 存储色块更新
    for big_block_idx, unified_idx in zone_color_updates:
        data.append(big_block_idx)
        data.append(unified_idx)
    
    return bytes(data), False, 1, len(zone_color_updates), len(zone_detail_updates)

def generate_gop_unified_codebooks(frames: list, strip_count: int, i_frame_interval: int,
                                  variance_threshold: float, codebook_size: int = DEFAULT_UNIFIED_CODEBOOK_SIZE,
                                  kmeans_max_iter: int = 100) -> dict:
    """为每个GOP生成统一码本"""
    print("正在为每个GOP生成统一码本...")
    
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
            all_blocks = []
            block_types_list = []
            
            for frame_idx in range(gop_start, gop_end):
                strip_blocks = frames[frame_idx][strip_idx]
                if strip_blocks.size > 0:
                    frame_blocks, block_types = classify_8x8_blocks_unified(strip_blocks, variance_threshold)
                    all_blocks.extend(frame_blocks)
                    block_types_list.append((frame_idx, block_types))
            
            # 生成统一码本
            unified_codebook = generate_unified_codebook(all_blocks, codebook_size, kmeans_max_iter)
            
            gop_codebooks[gop_start].append({
                'unified_codebook': unified_codebook,
                'block_types_list': block_types_list,
                'total_blocks_count': len(all_blocks)
            })
            
            # 统计色块和纹理块数量
            color_count = 0
            detail_count = 0
            for _, block_types in block_types_list:
                for (big_by, big_bx), (block_type, _) in block_types.items():
                    if block_type == 'color':
                        color_count += 1
                    else:
                        detail_count += 1
            
            print(f"    条带{strip_idx}: 总块数{len(all_blocks)}, 色块{color_count}, 纹理块{detail_count}")
    
    return gop_codebooks

class EncodingStats:
    """编码统计类"""
    def __init__(self):
        # 帧统计
        self.total_frames_processed = 0  # 实际处理的帧数（条带级别）
        self.total_i_frames = 0
        self.forced_i_frames = 0  # 强制I帧（GOP开始）
        self.threshold_i_frames = 0  # 超阈值I帧
        self.total_p_frames = 0
        
        # 大小统计
        self.total_i_frame_bytes = 0
        self.total_p_frame_bytes = 0
        self.total_codebook_bytes = 0  # 只计算I帧中的码本数据
        self.total_index_bytes = 0     # 只计算I帧中的索引数据
        self.total_p_overhead_bytes = 0  # P帧的开销数据（bitmap等）
        
        # P帧块更新统计
        self.p_frame_updates = []  # 每个P帧的更新块数
        self.zone_usage = defaultdict(int)  # 区域使用次数
        
        # 细节统计
        self.color_block_bytes = 0
        self.detail_block_bytes = 0
        self.color_update_count = 0
        self.detail_update_count = 0
        
        # 条带统计
        self.strip_stats = defaultdict(lambda: {
            'i_frames': 0, 'p_frames': 0, 
            'i_bytes': 0, 'p_bytes': 0
        })
    
    def add_i_frame(self, strip_idx, size_bytes, is_forced=True, codebook_size=0, index_size=0):
        self.total_frames_processed += 1
        self.total_i_frames += 1
        if is_forced:
            self.forced_i_frames += 1
        else:
            self.threshold_i_frames += 1
        
        self.total_i_frame_bytes += size_bytes
        self.total_codebook_bytes += codebook_size
        self.total_index_bytes += index_size
        
        self.strip_stats[strip_idx]['i_frames'] += 1
        self.strip_stats[strip_idx]['i_bytes'] += size_bytes
    
    def add_p_frame(self, strip_idx, size_bytes, updates_count, zone_count, 
                   color_updates=0, detail_updates=0):
        self.total_frames_processed += 1
        self.total_p_frames += 1
        self.total_p_frame_bytes += size_bytes
        self.p_frame_updates.append(updates_count)
        self.zone_usage[zone_count] += 1
        
        # P帧开销：帧类型(1) + bitmap(1) + 每个区域的计数(2*zones)
        overhead = 2 + zone_count * 2  # 大致估算
        self.total_p_overhead_bytes += overhead
        
        self.color_update_count += color_updates
        self.detail_update_count += detail_updates
        
        self.strip_stats[strip_idx]['p_frames'] += 1
        self.strip_stats[strip_idx]['p_bytes'] += size_bytes
    
    def add_block_stats(self, color_bytes, detail_bytes):
        self.color_block_bytes += color_bytes
        self.detail_block_bytes += detail_bytes
    
    def print_summary(self, total_frames, total_bytes):
        print(f"\n📊 编码统计报告")
        print(f"=" * 60)
        
        # 计算条带级别的统计
        strip_count = len(self.strip_stats) if self.strip_stats else 1
        
        # 基本统计
        print(f"🎬 帧统计:")
        print(f"   视频帧数: {total_frames}")
        print(f"   条带总数: {strip_count}")
        print(f"   处理的条带帧: {self.total_frames_processed}")
        print(f"   I帧条带: {self.total_i_frames} ({self.total_i_frames/self.total_frames_processed*100:.1f}%)")
        print(f"     - 强制I帧: {self.forced_i_frames}")
        print(f"     - 超阈值I帧: {self.threshold_i_frames}")
        print(f"   P帧条带: {self.total_p_frames} ({self.total_p_frames/self.total_frames_processed*100:.1f}%)")
        
        # 大小统计
        print(f"\n💾 空间占用:")
        print(f"   总大小: {total_bytes:,} bytes ({total_bytes/1024:.1f} KB)")
        print(f"   I帧数据: {self.total_i_frame_bytes:,} bytes ({self.total_i_frame_bytes/total_bytes*100:.1f}%)")
        print(f"   P帧数据: {self.total_p_frame_bytes:,} bytes ({self.total_p_frame_bytes/total_bytes*100:.1f}%)")
        
        if self.total_i_frames > 0:
            print(f"   平均I帧大小: {self.total_i_frame_bytes/self.total_i_frames:.1f} bytes")
        if self.total_p_frames > 0:
            print(f"   平均P帧大小: {self.total_p_frame_bytes/self.total_p_frames:.1f} bytes")
        
        # 数据构成统计（修正）
        print(f"\n🎨 数据构成:")
        print(f"   码本数据: {self.total_codebook_bytes:,} bytes ({self.total_codebook_bytes/total_bytes*100:.1f}%)")
        print(f"   I帧索引: {self.total_index_bytes:,} bytes ({self.total_index_bytes/total_bytes*100:.1f}%)")
        
        # P帧数据构成
        p_frame_data_bytes = self.total_p_frame_bytes - self.total_p_overhead_bytes
        print(f"   P帧更新数据: {p_frame_data_bytes:,} bytes ({p_frame_data_bytes/total_bytes*100:.1f}%)")
        print(f"   P帧开销: {self.total_p_overhead_bytes:,} bytes ({self.total_p_overhead_bytes/total_bytes*100:.1f}%)")
        
        # 其他数据
        other_bytes = total_bytes - (self.total_codebook_bytes + self.total_index_bytes + self.total_p_frame_bytes)
        if other_bytes > 0:
            print(f"   其他数据: {other_bytes:,} bytes ({other_bytes/total_bytes*100:.1f}%)")
        
        # 块类型统计
        print(f"\n🧩 块类型分布:")
        if self.color_block_bytes > 0 or self.detail_block_bytes > 0:
            total_block_data = self.color_block_bytes + self.detail_block_bytes
            print(f"   色块索引: {self.color_block_bytes} 个 ({self.color_block_bytes/total_block_data*100:.1f}%)")
            print(f"   纹理块索引: {self.detail_block_bytes} 个 ({self.detail_block_bytes/total_block_data*100:.1f}%)")
        
        # P帧更新统计
        if self.p_frame_updates:
            avg_updates = statistics.mean(self.p_frame_updates)
            median_updates = statistics.median(self.p_frame_updates)
            max_updates = max(self.p_frame_updates)
            min_updates = min(self.p_frame_updates)
            
            print(f"\n⚡ P帧更新分析:")
            print(f"   平均更新块数: {avg_updates:.1f}")
            print(f"   中位数更新块数: {median_updates}")
            print(f"   最大更新块数: {max_updates}")
            print(f"   最小更新块数: {min_updates}")
            print(f"   色块更新总数: {self.color_update_count:,}")
            print(f"   纹理块更新总数: {self.detail_update_count:,}")
        
        # 区域使用统计
        if self.zone_usage:
            print(f"\n🗺️  区域使用分布:")
            for zone_count in sorted(self.zone_usage.keys()):
                frames_count = self.zone_usage[zone_count]
                if self.total_p_frames > 0:
                    print(f"   {zone_count}个区域: {frames_count}次 ({frames_count/self.total_p_frames*100:.1f}%)")
        
        # 条带统计
        print(f"\n📏 条带统计:")
        for strip_idx in sorted(self.strip_stats.keys()):
            stats = self.strip_stats[strip_idx]
            total_strip_frames = stats['i_frames'] + stats['p_frames']
            total_strip_bytes = stats['i_bytes'] + stats['p_bytes']
            if total_strip_frames > 0:
                print(f"   条带{strip_idx}: {total_strip_frames}帧, {total_strip_bytes:,}bytes, "
                      f"平均{total_strip_bytes/total_strip_frames:.1f}bytes/帧")
        
        # 压缩效率
        raw_size = total_frames * WIDTH * HEIGHT * 2  # 假设16位像素
        compression_ratio = raw_size / total_bytes if total_bytes > 0 else 0
        print(f"\n📈 压缩效率:")
        print(f"   原始大小估算: {raw_size:,} bytes ({raw_size/1024/1024:.1f} MB)")
        print(f"   压缩比: {compression_ratio:.1f}:1")
        print(f"   压缩率: {(1-total_bytes/raw_size)*100:.1f}%")

# 全局统计对象
encoding_stats = EncodingStats()

def main():
    pa = argparse.ArgumentParser(description="Encode to GBA YUV9 with unified codebook")
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
    pa.add_argument("--codebook-size", type=int, default=DEFAULT_UNIFIED_CODEBOOK_SIZE,
                   help=f"统一码本大小（默认{DEFAULT_UNIFIED_CODEBOOK_SIZE}）")
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
    print(f"码本配置: 统一码本{args.codebook_size}项")
    
    # 添加调试信息：计算每个条带的8x8大块数量
    for i, strip_height in enumerate(strip_heights):
        big_blocks_h = (strip_height // 4) // 2  # 4x4块行数除以2
        big_blocks_w = (WIDTH // 4) // 2          # 4x4块列数除以2
        total_big_blocks = big_blocks_h * big_blocks_w
        print(f"  条带{i}: 高度{strip_height}, 8x8大块数量: {big_blocks_h}×{big_blocks_w}={total_big_blocks}")
        if total_big_blocks > 255:
            print(f"  ⚠️  警告: 条带{i}的8x8大块数量{total_big_blocks}超出u8范围(255)!")

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

    # 生成统一码本
    gop_codebooks = generate_gop_unified_codebooks(
        frames, args.strip_count, args.i_frame_interval, 
        args.variance_threshold, args.codebook_size, args.kmeans_max_iter
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
            unified_codebook = strip_gop_data['unified_codebook']
            
            # 找到当前帧的block_types
            block_types = None
            for fid, bt in strip_gop_data['block_types_list']:
                if fid == frame_idx:
                    block_types = bt
                    break
            
            force_i_frame = (frame_idx % args.i_frame_interval == 0) or frame_idx == 0
            
            if force_i_frame or prev_strips[strip_idx] is None:
                strip_data = encode_strip_i_frame_unified(
                    current_strip, unified_codebook, block_types
                )
                is_i_frame = True
                
                # 计算码本和索引大小
                codebook_size = args.codebook_size * BYTES_PER_BLOCK
                index_size = len(strip_data) - 1 - codebook_size
                
                encoding_stats.add_i_frame(
                    strip_idx, len(strip_data), 
                    is_forced=force_i_frame,
                    codebook_size=codebook_size,
                    index_size=max(0, index_size)
                )
            else:
                strip_data, is_i_frame, used_zones, color_updates, detail_updates = encode_strip_differential_unified(
                    current_strip, prev_strips[strip_idx],
                    unified_codebook, block_types,
                    args.diff_threshold, args.force_i_threshold
                )
                
                if is_i_frame:
                    codebook_size = args.codebook_size * BYTES_PER_BLOCK
                    index_size = len(strip_data) - 1 - codebook_size
                    
                    encoding_stats.add_i_frame(
                        strip_idx, len(strip_data), 
                        is_forced=False,
                        codebook_size=codebook_size,
                        index_size=max(0, index_size)
                    )
                else:
                    total_updates = color_updates + detail_updates
                    
                    encoding_stats.add_p_frame(
                        strip_idx, len(strip_data), total_updates, used_zones,
                        color_updates, detail_updates
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
                args.strip_count, strip_heights, args.codebook_size)
    write_source(pathlib.Path(args.out).with_suffix(".c"), all_data, frame_offsets, strip_heights)
    
    # 打印详细统计
    encoding_stats.print_summary(len(frames), len(all_data))

def write_header(path_h: pathlib.Path, frame_cnt: int, total_bytes: int, strip_count: int, 
                strip_heights: list, codebook_size: int):
    guard = "VIDEO_DATA_H"
    
    with path_h.open("w", encoding="utf-8") as f:
        f.write(textwrap.dedent(f"""\
            #ifndef {guard}
            #define {guard}

            #define VIDEO_FRAME_COUNT   {frame_cnt}
            #define VIDEO_WIDTH         {WIDTH}
            #define VIDEO_HEIGHT        {HEIGHT}
            #define VIDEO_TOTAL_BYTES   {total_bytes}
            #define VIDEO_STRIP_COUNT   {strip_count}
            #define UNIFIED_CODEBOOK_SIZE {codebook_size}
            #define EFFECTIVE_UNIFIED_CODEBOOK_SIZE {EFFECTIVE_UNIFIED_CODEBOOK_SIZE}
            
            // 帧类型定义
            #define FRAME_TYPE_I        0x00
            #define FRAME_TYPE_P        0x01
            
            // 特殊标记
            #define COLOR_BLOCK_MARKER  0xFF
            
            // 块参数 - 4x4块和8x8大块
            #define BLOCK_WIDTH         4
            #define BLOCK_HEIGHT        4
            #define BIG_BLOCK_WIDTH     8
            #define BIG_BLOCK_HEIGHT    8
            #define BYTES_PER_BLOCK     {BYTES_PER_BLOCK}

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

if __name__ == "__main__":
    main()