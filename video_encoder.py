#!/usr/bin/env python3

import argparse, cv2, numpy as np, pathlib, textwrap
import struct
import concurrent.futures
import multiprocessing as mp
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from collections import defaultdict
import statistics
from numba import jit, njit, types
from numba.typed import List

from dither_opt import apply_dither_optimized

WIDTH, HEIGHT = 240, 160
DEFAULT_UNIFIED_CODEBOOK_SIZE = 256   # 统一码本大小
EFFECTIVE_UNIFIED_CODEBOOK_SIZE = 254  # 有效码本大小（0xFF保留）

# 标记常量
COLOR_BLOCK_MARKER = 0xFF

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

@njit
def clip_value(value, min_val, max_val):
    """Numba兼容的clip函数"""
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    else:
        return value

@njit
def pack_yuv420_frame_numba(bgr_frame):
    """Numba加速的整帧YUV420转换"""
    blocks_h = HEIGHT // BLOCK_H
    blocks_w = WIDTH // BLOCK_W
    
    block_array = np.zeros((blocks_h, blocks_w, BYTES_PER_BLOCK), dtype=np.uint8)
    
    for by in range(blocks_h):
        for bx in range(blocks_w):
            # 提取2x2块
            y_start = by * BLOCK_H
            x_start = bx * BLOCK_W
            
            # BGR to YUV conversion for 2x2 block
            cb_sum = 0.0
            cr_sum = 0.0
            y_values = np.zeros(4, dtype=np.uint8)
            
            idx = 0
            for dy in range(BLOCK_H):
                for dx in range(BLOCK_W):
                    if y_start + dy < HEIGHT and x_start + dx < WIDTH:
                        b = float(bgr_frame[y_start + dy, x_start + dx, 0])
                        g = float(bgr_frame[y_start + dy, x_start + dx, 1])  
                        r = float(bgr_frame[y_start + dy, x_start + dx, 2])
                        
                        y = r * 0.28571429 + g * 0.57142857 + b * 0.14285714
                        cb = r * (-0.14285714) + g * (-0.28571429) + b * 0.42857143
                        cr = r * 0.35714286 + g * (-0.28571429) + b * (-0.07142857)
                        
                        y_values[idx] = np.uint8(clip_value(y / 2.0, 0.0, 255.0))
                        cb_sum += cb
                        cr_sum += cr
                        idx += 1
            
            # Store Y values
            block_array[by, bx, 0:4] = y_values
            
            # Compute and store chroma
            cb_mean = cb_sum / 4.0
            cr_mean = cr_sum / 4.0
            
            d_r = clip_value(cr_mean, -128.0, 127.0)
            d_g = clip_value((-(cb_mean/2.0) - cr_mean) / 2.0, -128.0, 127.0)  
            d_b = clip_value(cb_mean, -128.0, 127.0)
            
            # 将有符号值转换为无符号字节存储
            block_array[by, bx, 4] = np.uint8(np.int8(d_r).view(np.uint8))
            block_array[by, bx, 5] = np.uint8(np.int8(d_g).view(np.uint8))
            block_array[by, bx, 6] = np.uint8(np.int8(d_b).view(np.uint8))
    
    return block_array

def pack_yuv420_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """使用Numba加速的整帧YUV转换包装函数"""
    return pack_yuv420_frame_numba(frame_bgr)

def calculate_block_variance(blocks_4x4: list) -> float:
    """计算4x4块的方差，用于判断是否为纯色块"""
    # 将4个2x2块合并为一个4x4的Y值数组
    y_values = []
    for block in blocks_4x4:
        y_values.extend(block[:4])  # 只取Y值
    
    y_array = np.array(y_values)
    return np.var(y_array)

@njit
def calculate_block_variance_numba(y_values):
    """Numba加速的方差计算"""
    mean_val = np.mean(y_values)
    variance = 0.0
    for val in y_values:
        diff = val - mean_val
        variance += diff * diff
    return variance / len(y_values)

def calculate_2x2_block_variance(block: np.ndarray) -> float:
    """计算单个2x2块的方差，用于判断是否为纯色"""
    y_values = block[:4].astype(np.float64)
    return calculate_block_variance_numba(y_values)

def classify_4x4_blocks(blocks: np.ndarray, variance_threshold: float = 5.0) -> tuple:
    """将4x4块分类为大色块和纹理块"""
    blocks_h, blocks_w = blocks.shape[:2]
    big_blocks_h = blocks_h // 2
    big_blocks_w = blocks_w // 2
    
    color_blocks = []  # 大色块（用下采样的2x2块表示）
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
            
            # 检查每个2x2子块是否内部一致（而非整个4x4块）
            all_2x2_blocks_are_uniform = True
            for block in blocks_4x4:
                if calculate_2x2_block_variance(block) > variance_threshold:
                    all_2x2_blocks_are_uniform = False
                    break
            
            if all_2x2_blocks_are_uniform:
                # 大色块：每个2x2块内部都一致，可以下采样为一个2x2块
                # 每个2x2块取其内部的平均值，然后用这4个平均值组成一个下采样的2x2块
                downsampled_block = np.zeros(BYTES_PER_BLOCK, dtype=np.uint8)
                
                # 提取每个2x2块的内部平均值
                y_values = []
                d_r_values = []
                d_g_values = []
                d_b_values = []
                
                for block in blocks_4x4:
                    # 每个2x2块内部一致，取其4个Y值的平均
                    avg_y = np.mean(block[:4])
                    y_values.append(int(avg_y))
                    
                    # 色度分量直接使用（已经是该2x2块的平均色度）
                    d_r_values.append(block[4].view(np.int8))
                    d_g_values.append(block[5].view(np.int8))
                    d_b_values.append(block[6].view(np.int8))
                
                # 构建下采样的2x2块：用4个2x2子块的平均值构成新的2x2块
                downsampled_block[:4] = np.array(y_values, dtype=np.uint8)
                downsampled_block[4] = np.clip(np.mean(d_r_values), -128, 127).astype(np.int8).view(np.uint8)
                downsampled_block[5] = np.clip(np.mean(d_g_values), -128, 127).astype(np.int8).view(np.uint8)
                downsampled_block[6] = np.clip(np.mean(d_b_values), -128, 127).astype(np.int8).view(np.uint8)
                
                color_blocks.append(downsampled_block)
                block_types[(big_by, big_bx)] = 'color'
            else:
                # 纹理块：至少有一个2x2子块内部不一致，保留所有4个2x2块
                detail_blocks.extend(blocks_4x4)
                block_types[(big_by, big_bx)] = 'detail'
    
    return color_blocks, detail_blocks, block_types

def classify_4x4_blocks_unified(blocks: np.ndarray, variance_threshold: float = 5.0) -> tuple:
    """将4x4块分类为大色块和纹理块，用于统一码本"""
    blocks_h, blocks_w = blocks.shape[:2]
    big_blocks_h = blocks_h // 2
    big_blocks_w = blocks_w // 2
    
    all_blocks = []  # 所有2x2块
    block_types = {}   # 记录每个4x4块的类型和对应的2x2块索引
    
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
            
            # 检查每个2x2子块是否内部一致（而非整个4x4块）
            all_2x2_blocks_are_uniform = True
            for block in blocks_4x4:
                if calculate_2x2_block_variance(block) > variance_threshold:
                    all_2x2_blocks_are_uniform = False
                    break
            
            if all_2x2_blocks_are_uniform:
                # 大色块：每个2x2子块内部都一致，用下采样的2x2块表示
                downsampled_block = np.zeros(BYTES_PER_BLOCK, dtype=np.uint8)
                
                y_values = []
                d_r_values = []
                d_g_values = []
                d_b_values = []
                
                for block in blocks_4x4:
                    # 每个2x2块内部一致，取其内部平均值
                    avg_y = np.mean(block[:4])
                    y_values.append(int(avg_y))
                    d_r_values.append(block[4].view(np.int8))
                    d_g_values.append(block[5].view(np.int8))
                    d_b_values.append(block[6].view(np.int8))
                
                # 用4个2x2子块的平均值构成下采样块
                downsampled_block[:4] = np.array(y_values, dtype=np.uint8)
                downsampled_block[4] = np.clip(np.mean(d_r_values), -128, 127).astype(np.int8).view(np.uint8)
                downsampled_block[5] = np.clip(np.mean(d_g_values), -128, 127).astype(np.int8).view(np.uint8)
                downsampled_block[6] = np.clip(np.mean(d_b_values), -128, 127).astype(np.int8).view(np.uint8)
                
                block_idx = len(all_blocks)
                all_blocks.append(downsampled_block)
                block_types[(big_by, big_bx)] = ('color', [block_idx])
            else:
                # 纹理块：至少有一个2x2子块内部不一致，保留所有4个2x2块
                block_indices = []
                for block in blocks_4x4:
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
    
    # 移除去重操作，直接使用原始数据进行聚类
    # 这样K-Means可以基于数据的真实分布（包括频次）进行更好的聚类
    effective_size = min(len(blocks_data), codebook_size)
    
    if len(blocks_data) <= codebook_size:
        # 如果数据量小于码本大小，需要去重避免重复
        blocks_as_tuples = [tuple(block) for block in blocks_data]
        unique_tuples = list(set(blocks_as_tuples))
        unique_blocks = np.array(unique_tuples, dtype=np.uint8)
        
        codebook = np.zeros((codebook_size, BYTES_PER_BLOCK), dtype=np.uint8)
        codebook[:len(unique_blocks)] = unique_blocks
        if len(unique_blocks) > 0:
            for i in range(len(unique_blocks), codebook_size):
                codebook[i] = unique_blocks[-1]
        return codebook, len(unique_blocks)
    
    # 对于大数据集，直接进行K-Means聚类
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

@njit
def quantize_blocks_distance_numba(blocks_for_clustering, codebook_for_clustering):
    """Numba加速的块量化距离计算"""
    n_blocks = blocks_for_clustering.shape[0]
    n_codebook = codebook_for_clustering.shape[0]
    indices = np.zeros(n_blocks, dtype=np.uint8)
    
    for i in range(n_blocks):
        min_dist = np.inf
        best_idx = 0
        
        for j in range(n_codebook):
            dist = 0.0
            for k in range(BYTES_PER_BLOCK):
                diff = blocks_for_clustering[i, k] - codebook_for_clustering[j, k]
                dist += diff * diff
            
            if dist < min_dist:
                min_dist = dist
                best_idx = j
        
        indices[i] = best_idx
    
    return indices

def quantize_blocks_unified(blocks_data: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """使用统一码表对块进行量化（避免产生0xFF）"""
    if len(blocks_data) == 0:
        return np.array([], dtype=np.uint8)
    
    # 只使用前255项进行量化
    effective_codebook = codebook[:EFFECTIVE_UNIFIED_CODEBOOK_SIZE]
    
    blocks_for_clustering = convert_blocks_for_clustering(blocks_data)
    codebook_for_clustering = convert_blocks_for_clustering(effective_codebook)
    
    # 使用Numba加速的距离计算
    indices = quantize_blocks_distance_numba(blocks_for_clustering, codebook_for_clustering)
    
    return indices

@njit
def compute_2x2_block_differences_numba(current_flat, prev_flat, blocks_h, blocks_w):
    """Numba加速的2x2块差异计算"""
    block_diffs = np.zeros((blocks_h, blocks_w), dtype=np.float64)
    
    for i in range(blocks_h * blocks_w):
        y_diff_sum = 0.0
        for j in range(4):  # 只计算Y分量差异
            current_val = int(current_flat[i, j])
            prev_val = int(prev_flat[i, j])
            if current_val >= prev_val:
                diff = current_val - prev_val
            else:
                diff = prev_val - current_val
            y_diff_sum += diff
        block_diffs[i // blocks_w, i % blocks_w] = y_diff_sum / 4.0
    
    return block_diffs

def encode_i_frame_unified(blocks: np.ndarray, unified_codebook: np.ndarray, 
                          block_types: dict) -> bytes:
    """编码I帧（统一码本）"""
    data = bytearray()
    data.append(FRAME_TYPE_I)
    
    if blocks.size > 0:
        blocks_h, blocks_w = blocks.shape[:2]
        big_blocks_h = blocks_h // 2
        big_blocks_w = blocks_w // 2
        
        # 存储统一码本
        data.extend(unified_codebook.flatten().tobytes())
        
        # 按4x4大块的顺序编码
        for big_by in range(big_blocks_h):
            for big_bx in range(big_blocks_w):
                # 处理block_types为None的情况
                if block_types is None or (big_by, big_bx) not in block_types:
                    # 默认为纹理块处理
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
                else:
                    block_type, block_indices = block_types[(big_by, big_bx)]
                    
                    if block_type == 'color':
                        # 色块：标记0xFF + 1个码本索引
                        data.append(COLOR_BLOCK_MARKER)
                        
                        # 从原始blocks重建平均块
                        blocks_4x4 = []
                        for sub_by in range(2):
                            for sub_bx in range(2):
                                by = big_by * 2 + sub_by
                                bx = big_bx * 2 + sub_bx
                                if by < blocks_h and bx < blocks_w:
                                    blocks_4x4.append(blocks[by, bx])
                        
                        avg_block = np.mean(blocks_4x4, axis=0).round().astype(np.uint8)
                        for i in range(4, 7):
                            avg_val = np.mean([b[i].view(np.int8) for b in blocks_4x4])
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

def encode_p_frame_unified(current_blocks: np.ndarray, prev_blocks: np.ndarray,
                          unified_codebook: np.ndarray, block_types: dict,
                          diff_threshold: float, force_i_threshold: float = 0.7) -> tuple:
    """差分编码P帧（统一码本）"""
    if prev_blocks is None or current_blocks.shape != prev_blocks.shape:
        i_frame_data = encode_i_frame_unified(current_blocks, unified_codebook, block_types)
        return i_frame_data, True, 0, 0, 0
    
    blocks_h, blocks_w = current_blocks.shape[:2]
    total_blocks = blocks_h * blocks_w
    
    if total_blocks == 0:
        return b'', True, 0, 0, 0
    
    # 使用Numba加速的2x2块差异计算
    current_flat = current_blocks.reshape(-1, BYTES_PER_BLOCK)
    prev_flat = prev_blocks.reshape(-1, BYTES_PER_BLOCK)
    block_diffs = compute_2x2_block_differences_numba(current_flat, prev_flat, blocks_h, blocks_w)
    
    big_blocks_h = blocks_h // 2
    big_blocks_w = blocks_w // 2
    
    # 计算区域数量
    zones_count = (big_blocks_h + ZONE_HEIGHT_BIG_BLOCKS - 1) // ZONE_HEIGHT_BIG_BLOCKS
    
    # 按区域组织更新
    zone_detail_updates = [[] for _ in range(zones_count)]
    zone_color_updates = [[] for _ in range(zones_count)]
    total_updated_blocks = 0
    
    for big_by in range(big_blocks_h):
        for big_bx in range(big_blocks_w):
            # 检查4x4大块内每个2x2子块是否需要更新
            positions = [
                (big_by * 2, big_bx * 2),
                (big_by * 2, big_bx * 2 + 1),
                (big_by * 2 + 1, big_bx * 2),
                (big_by * 2 + 1, big_bx * 2 + 1)
            ]
            
            # 检查每个2x2子块是否需要更新
            subblock_needs_update = []
            any_subblock_needs_update = False
            
            for by, bx in positions:
                if by < blocks_h and bx < blocks_w:
                    needs_update = block_diffs[by, bx] > diff_threshold
                    subblock_needs_update.append(needs_update)
                    if needs_update:
                        any_subblock_needs_update = True
                else:
                    subblock_needs_update.append(False)
            
            if any_subblock_needs_update:
                # 计算属于哪个区域
                zone_idx = min(big_by // ZONE_HEIGHT_BIG_BLOCKS, zones_count - 1)
                # 计算在区域内的相对坐标
                zone_relative_by = big_by % ZONE_HEIGHT_BIG_BLOCKS
                zone_relative_idx = zone_relative_by * big_blocks_w + big_bx
                
                # 统计实际更新的子块数
                actual_updated_subblocks = sum(subblock_needs_update)
                total_updated_blocks += actual_updated_subblocks
                
                # 检查block_types是否为None或不包含当前块
                is_color_block = (block_types is not None and 
                                (big_by, big_bx) in block_types and 
                                block_types[(big_by, big_bx)][0] == 'color')
                
                if is_color_block:
                    # 色块更新：如果任何子块需要更新，就更新整个色块
                    blocks_4x4 = []
                    for by, bx in positions:
                        if by < blocks_h and bx < blocks_w:
                            blocks_4x4.append(current_blocks[by, bx])
                    
                    avg_block = np.mean(blocks_4x4, axis=0).round().astype(np.uint8)
                    for i in range(4, 7):
                        avg_val = np.mean([b[i].view(np.int8) for b in blocks_4x4])
                        avg_block[i] = np.clip(avg_val, -128, 127).astype(np.int8).view(np.uint8)
                    
                    color_idx = quantize_blocks_unified(avg_block.reshape(1, -1), unified_codebook)[0]
                    zone_color_updates[zone_idx].append((zone_relative_idx, color_idx))
                else:
                    # 纹理块更新：为每个2x2子块生成索引，不需要更新的用0xFF标记
                    indices = []
                    for i, (by, bx) in enumerate(positions):
                        if by < blocks_h and bx < blocks_w and subblock_needs_update[i]:
                            # 需要更新的子块：量化并获取索引
                            block = current_blocks[by, bx]
                            unified_idx = quantize_blocks_unified(block.reshape(1, -1), unified_codebook)[0]
                            indices.append(unified_idx)
                        else:
                            # 不需要更新的子块：使用跳过标记
                            indices.append(COLOR_BLOCK_MARKER)  # 0xFF
                    
                    # 检查是否所有子块都不需要更新（防止发送全FF的块）
                    if any(idx != COLOR_BLOCK_MARKER for idx in indices):
                        zone_detail_updates[zone_idx].append((zone_relative_idx, indices))
    
    # 判断是否需要I帧
    update_ratio = total_updated_blocks / total_blocks
    if update_ratio > force_i_threshold:
        i_frame_data = encode_i_frame_unified(current_blocks, unified_codebook, block_types)
        return i_frame_data, True, 0, 0, 0
    
    # 编码P帧
    data = bytearray()
    data.append(FRAME_TYPE_P)
    
    # 统计使用的区域数量
    used_zones = 0
    total_color_updates = 0
    total_detail_updates = 0
    
    # 生成两个区域bitmap
    detail_zone_bitmap = 0
    color_zone_bitmap = 0
    
    for zone_idx in range(zones_count):
        if zone_detail_updates[zone_idx]:
            detail_zone_bitmap |= (1 << zone_idx)
            total_detail_updates += len(zone_detail_updates[zone_idx])
        if zone_color_updates[zone_idx]:
            color_zone_bitmap |= (1 << zone_idx)
            total_color_updates += len(zone_color_updates[zone_idx])
    
    # 计算实际使用的区域数（两个bitmap的并集）
    combined_bitmap = detail_zone_bitmap | color_zone_bitmap
    used_zones = bin(combined_bitmap).count('1')
    
    # 写入两个u16 bitmap
    data.extend(struct.pack('<H', detail_zone_bitmap))
    data.extend(struct.pack('<H', color_zone_bitmap))
    
    # 按区域编码纹理块更新
    for zone_idx in range(zones_count):
        if detail_zone_bitmap & (1 << zone_idx):
            detail_updates = zone_detail_updates[zone_idx]
            data.append(len(detail_updates))
            
            # 存储纹理块更新
            for relative_idx, indices in detail_updates:
                data.append(relative_idx)
                for idx in indices:
                    data.append(idx)
    
    # 按区域编码色块更新
    for zone_idx in range(zones_count):
        if color_zone_bitmap & (1 << zone_idx):
            color_updates = zone_color_updates[zone_idx]
            data.append(len(color_updates))
            
            # 存储色块更新
            for relative_idx, unified_idx in color_updates:
                data.append(relative_idx)
                data.append(unified_idx)
    
    return bytes(data), False, used_zones, total_color_updates, total_detail_updates

@njit
def identify_updated_blocks_numba(block_diffs, diff_threshold, blocks_h, blocks_w):
    """Numba加速的更新块识别"""
    big_blocks_h = blocks_h // 2
    big_blocks_w = blocks_w // 2
    updated_positions = []
    
    for big_by in range(big_blocks_h):
        for big_bx in range(big_blocks_w):
            needs_update = False
            
            # 检查4个2x2子块的位置
            for sub_by in range(2):
                for sub_bx in range(2):
                    by = big_by * 2 + sub_by
                    bx = big_bx * 2 + sub_bx
                    
                    if by < blocks_h and bx < blocks_w:
                        if block_diffs[by, bx] > diff_threshold:
                            needs_update = True
                            break
                if needs_update:
                    break
            
            if needs_update:
                updated_positions.append((big_by, big_bx))
    
    return updated_positions

def identify_updated_big_blocks(current_blocks: np.ndarray, prev_blocks: np.ndarray,
                               diff_threshold: float) -> set:
    """识别需要更新的4x4大块位置 - Numba加速版本"""
    if prev_blocks is None or current_blocks.shape != prev_blocks.shape:
        # 如果没有前一帧，所有大块都需要更新
        blocks_h, blocks_w = current_blocks.shape[:2]
        big_blocks_h = blocks_h // 2
        big_blocks_w = blocks_w // 2
        return {(big_by, big_bx) for big_by in range(big_blocks_h) for big_bx in range(big_blocks_w)}
    
    blocks_h, blocks_w = current_blocks.shape[:2]
    
    # 使用Numba加速的块差异计算
    current_flat = current_blocks.reshape(-1, BYTES_PER_BLOCK)
    prev_flat = prev_blocks.reshape(-1, BYTES_PER_BLOCK)
    block_diffs = compute_2x2_block_differences_numba(current_flat, prev_flat, blocks_h, blocks_w)
    
    # 使用Numba加速的更新块识别
    updated_list = identify_updated_blocks_numba(block_diffs, diff_threshold, blocks_h, blocks_w)
    
    return set(updated_list)

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

def extract_effective_blocks_from_big_blocks(blocks: np.ndarray, big_block_positions: set,
                                           variance_threshold: float = 5.0) -> list:
    """从指定的4x4大块位置提取有效的2x2块"""
    blocks_h, blocks_w = blocks.shape[:2]
    effective_blocks = []
    
    for big_by, big_bx in big_block_positions:
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
        
        # 检查是否为色块（每个2x2子块内部一致）
        all_2x2_blocks_are_uniform = True
        for block in blocks_4x4:
            if calculate_2x2_block_variance(block) > variance_threshold:
                all_2x2_blocks_are_uniform = False
                break
        
        if all_2x2_blocks_are_uniform:
            # 色块：生成下采样的2x2块
            downsampled_block = np.zeros(BYTES_PER_BLOCK, dtype=np.uint8)
            
            y_values = []
            d_r_values = []
            d_g_values = []
            d_b_values = []
            
            for block in blocks_4x4:
                # 每个2x2块内部一致，取其内部平均值
                avg_y = np.mean(block[:4])
                y_values.append(int(avg_y))
                d_r_values.append(block[4].view(np.int8))
                d_g_values.append(block[5].view(np.int8))
                d_b_values.append(block[6].view(np.int8))
            
            # 用4个2x2子块的平均值构成下采样块
            downsampled_block[:4] = np.array(y_values, dtype=np.uint8)
            downsampled_block[4] = np.clip(np.mean(d_r_values), -128, 127).astype(np.int8).view(np.uint8)
            downsampled_block[5] = np.clip(np.mean(d_g_values), -128, 127).astype(np.int8).view(np.uint8)
            downsampled_block[6] = np.clip(np.mean(d_b_values), -128, 127).astype(np.int8).view(np.uint8)
            
            effective_blocks.append(downsampled_block)
        else:
            # 纹理块：添加所有4个2x2块
            effective_blocks.extend(blocks_4x4)
    
    return effective_blocks

def process_single_gop_frame(args_tuple):
    """处理单个GOP的单帧 - 用于多进程"""
    (gop_start, gop_end, frame_data_list, variance_threshold, 
     diff_threshold, codebook_size, kmeans_max_iter, i_frame_weight) = args_tuple
    
    try:
        effective_blocks = []
        block_types_list = []
        
        # 处理GOP中的每一帧
        prev_blocks = None
        
        for frame_idx in range(gop_start, gop_end):
            relative_frame_idx = frame_idx - gop_start
            if relative_frame_idx >= len(frame_data_list):
                break
                
            frame_blocks = frame_data_list[relative_frame_idx]
            if frame_blocks.size == 0:
                continue
            
            # 确定帧类型和需要更新的大块
            is_i_frame = (frame_idx == gop_start)  # GOP第一帧是I帧
            
            if is_i_frame:
                # I帧：所有大块都有效
                blocks_h, blocks_w = frame_blocks.shape[:2]
                big_blocks_h = blocks_h // 2
                big_blocks_w = blocks_w // 2
                updated_big_blocks = {(big_by, big_bx) for big_by in range(big_blocks_h) for big_bx in range(big_blocks_w)}
            else:
                # P帧：只有更新的大块有效
                updated_big_blocks = identify_updated_big_blocks(frame_blocks, prev_blocks, diff_threshold)
            
            # 从有效大块中提取2x2块
            frame_effective_blocks = extract_effective_blocks_from_big_blocks(
                frame_blocks, updated_big_blocks, variance_threshold)
            
            # I帧块加权：复制多次以增加在聚类中的影响力
            if is_i_frame:
                weighted_blocks = frame_effective_blocks * i_frame_weight  # 复制i_frame_weight次
                effective_blocks.extend(weighted_blocks)
            else:
                effective_blocks.extend(frame_effective_blocks)
            
            # 生成完整的block_types用于编码（所有大块，不只是有效的）
            frame_blocks_list, block_types = classify_4x4_blocks_unified(frame_blocks, variance_threshold)
            block_types_list.append((frame_idx, block_types))
            
            prev_blocks = frame_blocks.copy()
        
        # 使用有效块生成统一码本
        unified_codebook = generate_unified_codebook(effective_blocks, codebook_size, kmeans_max_iter)
        
        return {
            'gop_start': gop_start,
            'unified_codebook': unified_codebook,
            'block_types_list': block_types_list,
            'total_blocks_count': len(effective_blocks),
            'success': True
        }
        
    except Exception as e:
        return {
            'gop_start': gop_start,
            'error': str(e),
            'success': False
        }

def generate_gop_unified_codebooks(frames: list, i_frame_interval: int,
                                  variance_threshold: float, diff_threshold: float,
                                  codebook_size: int = DEFAULT_UNIFIED_CODEBOOK_SIZE,
                                  kmeans_max_iter: int = 100, i_frame_weight: int = 3,
                                  max_workers: int = None) -> dict:
    """为每个GOP生成统一码本（多进程版本）"""
    print("正在为每个GOP生成统一码本（多进程，基于有效块，I帧加权）...")
    
    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 1)  # 留一个核心给系统
    
    print(f"使用 {max_workers} 个进程并行处理")
    
    gop_codebooks = {}
    
    # 确定I帧位置
    i_frame_positions = []
    for frame_idx in range(len(frames)):
        if frame_idx % i_frame_interval == 0:
            i_frame_positions.append(frame_idx)
    
    # 准备任务参数
    tasks = []
    for gop_idx, gop_start in enumerate(i_frame_positions):
        if gop_idx + 1 < len(i_frame_positions):
            gop_end = i_frame_positions[gop_idx + 1]
        else:
            gop_end = len(frames)
        
        # 提取该GOP的所有帧数据
        frame_data_list = []
        for frame_idx in range(gop_start, gop_end):
            if frame_idx < len(frames):
                frame_data_list.append(frames[frame_idx])
            else:
                break
        
        if frame_data_list:  # 只有当有数据时才添加任务
            task_args = (
                gop_start, gop_end, frame_data_list,
                variance_threshold, diff_threshold, codebook_size,
                kmeans_max_iter, i_frame_weight
            )
            tasks.append(task_args)
    
    total_tasks = len(tasks)
    print(f"总共 {len(i_frame_positions)} 个GOP，{total_tasks} 个处理任务")
    
    # 使用多进程处理
    completed_tasks = 0
    
    # 设置进程启动方法（避免某些平台的问题）
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 如果已经设置过就跳过
    
    with mp.Pool(processes=max_workers) as pool:
        # 提交所有任务
        results = []
        for task_args in tasks:
            result = pool.apply_async(process_single_gop_frame, (task_args,))
            results.append(result)
        
        # 收集结果并显示进度
        processed_results = []
        for i, result in enumerate(results):
            try:
                task_result = result.get(timeout=300)  # 5分钟超时
                processed_results.append(task_result)
                completed_tasks += 1
                
                if completed_tasks % max(1, total_tasks // 20) == 0 or completed_tasks == total_tasks:
                    progress = completed_tasks / total_tasks * 100
                    print(f"  进度: {completed_tasks}/{total_tasks} ({progress:.1f}%)")
                    
            except Exception as e:
                print(f"  ⚠️ 任务 {i} 处理失败: {e}")
                # 创建一个失败的结果
                task_args = tasks[i]
                processed_results.append({
                    'gop_start': task_args[0],
                    'success': False,
                    'error': str(e)
                })
    
    # 组织结果
    failed_count = 0
    for result in processed_results:
        if not result['success']:
            print(f"  ❌ GOP {result['gop_start']} 处理失败: {result.get('error', '未知错误')}")
            failed_count += 1
            continue
        
        gop_start = result['gop_start']
        gop_codebooks[gop_start] = {
            'unified_codebook': result['unified_codebook'],
            'block_types_list': result['block_types_list'],
            'total_blocks_count': result['total_blocks_count']
        }
    
    if failed_count > 0:
        print(f"  ⚠️ 共有 {failed_count} 个任务处理失败")
    else:
        print(f"  ✅ 所有 {total_tasks} 个任务处理完成")
    
    # 验证结果完整性
    for gop_start in i_frame_positions:
        if gop_start not in gop_codebooks:
            print(f"  ⚠️ GOP {gop_start} 缺少数据，使用默认码本")
            # 创建默认码本
            default_codebook = np.zeros((codebook_size, BYTES_PER_BLOCK), dtype=np.uint8)
            gop_codebooks[gop_start] = {
                'unified_codebook': default_codebook,
                'block_types_list': [],
                'total_blocks_count': 0
            }
    
    return gop_codebooks

class EncodingStats:
    """编码统计类"""
    def __init__(self):
        # 帧统计
        self.total_frames_processed = 0
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
    
    def add_i_frame(self, size_bytes, is_forced=True, codebook_size=0, index_size=0):
        self.total_frames_processed += 1
        self.total_i_frames += 1
        if is_forced:
            self.forced_i_frames += 1
        else:
            self.threshold_i_frames += 1
        
        self.total_i_frame_bytes += size_bytes
        self.total_codebook_bytes += codebook_size
        self.total_index_bytes += index_size
    
    def add_p_frame(self, size_bytes, updates_count, zone_count, 
                   color_updates=0, detail_updates=0):
        self.total_frames_processed += 1
        self.total_p_frames += 1
        self.total_p_frame_bytes += size_bytes
        self.p_frame_updates.append(updates_count)
        self.zone_usage[zone_count] += 1
        
        # P帧开销：帧类型(1) + bitmap(2) + 每个区域的计数(2*zones)
        overhead = 3 + zone_count * 2
        self.total_p_overhead_bytes += overhead
        
        self.color_update_count += color_updates
        self.detail_update_count += detail_updates
    
    def print_summary(self, total_frames, total_bytes):
        print(f"\n📊 编码统计报告")
        print(f"=" * 60)
        
        # 基本统计
        print(f"🎬 帧统计:")
        print(f"   视频帧数: {total_frames}")
        print(f"   I帧: {self.total_i_frames} ({self.total_i_frames/total_frames*100:.1f}%)")
        print(f"     - 强制I帧: {self.forced_i_frames}")
        print(f"     - 超阈值I帧: {self.threshold_i_frames}")
        print(f"   P帧: {self.total_p_frames} ({self.total_p_frames/total_frames*100:.1f}%)")
        
        # 大小统计
        print(f"\n💾 空间占用:")
        print(f"   总大小: {total_bytes:,} bytes ({total_bytes/1024:.1f} KB)")
        print(f"   I帧数据: {self.total_i_frame_bytes:,} bytes ({self.total_i_frame_bytes/total_bytes*100:.1f}%)")
        print(f"   P帧数据: {self.total_p_frame_bytes:,} bytes ({self.total_p_frame_bytes/total_bytes*100:.1f}%)")
        
        if self.total_i_frames > 0:
            print(f"   平均I帧大小: {self.total_i_frame_bytes/self.total_i_frames:.1f} bytes")
        if self.total_p_frames > 0:
            print(f"   平均P帧大小: {self.total_p_frame_bytes/self.total_p_frames:.1f} bytes")
        
        # 数据构成统计
        print(f"\n🎨 数据构成:")
        print(f"   码本数据: {self.total_codebook_bytes:,} bytes ({self.total_codebook_bytes/total_bytes*100:.1f}%)")
        print(f"   I帧索引: {self.total_index_bytes:,} bytes ({self.total_index_bytes/total_bytes*100:.1f}%)")
        
        # P帧数据构成
        p_frame_data_bytes = self.total_p_frame_bytes - self.total_p_overhead_bytes
        print(f"   P帧更新数据: {p_frame_data_bytes:,} bytes ({p_frame_data_bytes/total_bytes*100:.1f}%)")
        print(f"   P帧开销: {self.total_p_overhead_bytes:,} bytes ({self.total_p_overhead_bytes/total_bytes*100:.1f}%)")
        
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
    pa.add_argument("--i-frame-interval", type=int, default=60)
    pa.add_argument("--diff-threshold", type=float, default=2.0)
    pa.add_argument("--force-i-threshold", type=float, default=0.7)
    pa.add_argument("--variance-threshold", type=float, default=5.0,
                   help="方差阈值，用于区分纯色块和纹理块（默认5.0）")
    pa.add_argument("--codebook-size", type=int, default=DEFAULT_UNIFIED_CODEBOOK_SIZE,
                   help=f"统一码本大小（默认{DEFAULT_UNIFIED_CODEBOOK_SIZE}）")
    pa.add_argument("--kmeans-max-iter", type=int, default=200)
    pa.add_argument("--threads", type=int, default=None)
    pa.add_argument("--i-frame-weight", type=int, default=3,
                   help="I帧块在聚类中的权重倍数（默认3）")
    pa.add_argument("--max-workers", type=int, default=None,
                   help="GOP处理的最大进程数（默认为CPU核心数-1）")
    pa.add_argument("--dither", action="store_true",
                   help="启用Floyd-Steinberg抖动算法提升画质")
    args = pa.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit("❌ 打不开输入文件")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    # 计算实际输出FPS：如果目标FPS高于源FPS，使用源FPS
    actual_output_fps = min(args.fps, src_fps)
    every = int(round(src_fps / actual_output_fps))
    
    if args.full_duration:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        grab_max = total_frames
        actual_duration = total_frames / src_fps
        print(f"编码整个视频: {total_frames} 帧，时长 {actual_duration:.2f} 秒")
    else:
        grab_max = int(args.duration * src_fps)
        print(f"编码时长: {args.duration} 秒 ({grab_max} 帧)")

    print(f"源视频FPS: {src_fps:.2f}, 目标FPS: {args.fps}, 实际输出FPS: {actual_output_fps:.2f}")
    print(f"码本配置: 统一码本{args.codebook_size}项")
    if args.dither:
        print(f"🎨 已启用抖动算法（蛇形扫描）")
    
    frames = []
    idx = 0
    print("正在提取帧...")
    
    while idx < grab_max:
        ret, frm = cap.read()
        if not ret:
            break
        if idx % every == 0:
            frm = cv2.resize(frm, (WIDTH, HEIGHT), cv2.INTER_AREA)
            frm = cv2.GaussianBlur(frm, (3, 3), 0.41)
            if args.dither:
                frm = apply_dither_optimized(frm)
            
            frame_blocks = pack_yuv420_frame(frm)
            frames.append(frame_blocks)
            
            if len(frames) % 30 == 0:
                print(f"  已提取 {len(frames)} 帧")
        idx += 1
    cap.release()

    if not frames:
        raise SystemExit("❌ 没有任何帧被采样")

    print(f"总共提取了 {len(frames)} 帧")

    # 生成统一码本（传入max_workers参数）
    gop_codebooks = generate_gop_unified_codebooks(
        frames, args.i_frame_interval, 
        args.variance_threshold, args.diff_threshold, args.codebook_size, 
        args.kmeans_max_iter, args.i_frame_weight, args.max_workers
    )

    # 基于 GOP 内 P 帧纹理块使用频次，对每个码本项降序重排
    import numpy as _np
    for gop_start, gop_data in gop_codebooks.items():
        codebook = gop_data['unified_codebook']
        counts = _np.zeros(len(codebook), dtype=int)
        # GOP 范围：起始帧下一个到下一个 I 帧
        gop_end = min(gop_start + args.i_frame_interval, len(frames))
        for fid in range(gop_start + 1, gop_end):
            cur = frames[fid]
            prev = frames[fid - 1]
            # 识别更新的大块
            updated = identify_updated_big_blocks(cur, prev, args.diff_threshold)
            # 取出该帧的 block_types
            bt_map = None
            for fno, bt in gop_data['block_types_list']:
                if fno == fid:
                    bt_map = bt; break
            # 累加每个纹理子块的索引使用次数
            for by, bx in updated:
                is_color = bt_map and bt_map.get((by, bx), ('detail',))[0] == 'color'
                if not is_color:
                    for sy in (0,1):
                        for sx in (0,1):
                            y, x = by*2+sy, bx*2+sx
                            if y < cur.shape[0] and x < cur.shape[1]:
                                b = cur[y, x]
                                idx = quantize_blocks_unified(b.reshape(1, -1), codebook)[0]
                                counts[idx] += 1
        # 根据 counts 降序排序，stable 保持相同频次项原序
        order = _np.argsort(-counts, kind='stable')
        gop_data['unified_codebook'] = codebook[order]
    
    # 编码所有帧
    print("正在编码帧...")
    encoded_frames = []
    frame_offsets = []
    current_offset = 0
    prev_frame = None
    
    for frame_idx, current_frame in enumerate(frames):
        frame_offsets.append(current_offset)
        
        # 找到当前GOP
        gop_start = (frame_idx // args.i_frame_interval) * args.i_frame_interval
        gop_data = gop_codebooks[gop_start]
        
        unified_codebook = gop_data['unified_codebook']
        
        # 找到当前帧的block_types，处理缺失的情况
        block_types = None
        for fid, bt in gop_data['block_types_list']:
            if fid == frame_idx:
                block_types = bt
                break
        
        # 如果block_types仍然为None，生成默认的block_types
        if block_types is None:
            print(f"  ⚠️ 帧 {frame_idx} 缺少block_types，使用默认分类")
            # 临时生成block_types
            _, block_types = classify_4x4_blocks_unified(current_frame, args.variance_threshold)
        
        force_i_frame = (frame_idx % args.i_frame_interval == 0) or frame_idx == 0
        
        if force_i_frame or prev_frame is None:
            frame_data = encode_i_frame_unified(
                current_frame, unified_codebook, block_types
            )
            is_i_frame = True
            
            # 计算码本和索引大小
            codebook_size = args.codebook_size * BYTES_PER_BLOCK
            index_size = len(frame_data) - 1 - codebook_size
            
            encoding_stats.add_i_frame(
                len(frame_data), 
                is_forced=force_i_frame,
                codebook_size=codebook_size,
                index_size=max(0, index_size)
            )
        else:
            frame_data, is_i_frame, used_zones, color_updates, detail_updates = encode_p_frame_unified(
                current_frame, prev_frame,
                unified_codebook, block_types,
                args.diff_threshold, args.force_i_threshold
            )
            
            if is_i_frame:
                codebook_size = args.codebook_size * BYTES_PER_BLOCK
                index_size = len(frame_data) - 1 - codebook_size
                
                encoding_stats.add_i_frame(
                    len(frame_data), 
                    is_forced=False,
                    codebook_size=codebook_size,
                    index_size=max(0, index_size)
                )
            else:
                total_updates = color_updates + detail_updates
                
                encoding_stats.add_p_frame(
                    len(frame_data), total_updates, used_zones,
                    color_updates, detail_updates
                )
        
        encoded_frames.append(frame_data)
        current_offset += len(frame_data)
        
        prev_frame = current_frame.copy() if current_frame.size > 0 else None
        
        if frame_idx % 30 == 0 or frame_idx == len(frames) - 1:
            print(f"  已编码 {frame_idx + 1}/{len(frames)} 帧")
    
    all_data = b''.join(encoded_frames)
    
    write_header(pathlib.Path(args.out).with_suffix(".h"), len(frames), len(all_data), 
                args.codebook_size, actual_output_fps)
    write_source(pathlib.Path(args.out).with_suffix(".c"), all_data, frame_offsets)
    
    # 打印详细统计
    encoding_stats.print_summary(len(frames), len(all_data))

def write_header(path_h: pathlib.Path, frame_cnt: int, total_bytes: int, codebook_size: int, output_fps: float):
    guard = "VIDEO_DATA_H"
    
    with path_h.open("w", encoding="utf-8") as f:
        f.write(textwrap.dedent(f"""\
            #ifndef {guard}
            #define {guard}

            #define VIDEO_FRAME_COUNT   {frame_cnt}
            #define VIDEO_WIDTH         {WIDTH}
            #define VIDEO_HEIGHT        {HEIGHT}
            #define VIDEO_TOTAL_BYTES   {total_bytes}
            #define VIDEO_FPS           {int(round(output_fps*10000))}
            #define UNIFIED_CODEBOOK_SIZE {codebook_size}
            #define EFFECTIVE_UNIFIED_CODEBOOK_SIZE {EFFECTIVE_UNIFIED_CODEBOOK_SIZE}
            
            // 帧类型定义
            #define FRAME_TYPE_I        0x00
            #define FRAME_TYPE_P        0x01
            
            // 特殊标记
            #define COLOR_BLOCK_MARKER  0xFF
            
            // 块参数
            #define BLOCK_WIDTH         2
            #define BLOCK_HEIGHT        2
            #define BYTES_PER_BLOCK     7
            
            extern const unsigned char video_data[VIDEO_TOTAL_BYTES];
            extern const unsigned int frame_offsets[VIDEO_FRAME_COUNT];

            #endif // {guard}
            """))

def write_source(path_c: pathlib.Path, data: bytes, frame_offsets: list):
    with path_c.open("w", encoding="utf-8") as f:
        f.write('#include "video_data.h"\n\n')
        
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