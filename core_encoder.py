#!/usr/bin/env python3

import numpy as np
import struct
from sklearn.cluster import MiniBatchKMeans
from numba import jit, njit, types
from numba.typed import List
from collections import defaultdict

from dither_opt import apply_dither_optimized

# 常量定义
WIDTH, HEIGHT = 240, 160
DEFAULT_UNIFIED_CODEBOOK_SIZE = 256   # 统一码本大小
EFFECTIVE_UNIFIED_CODEBOOK_SIZE = 254  # 有效码本大小（0xFF保留）

# 标记常量
COLOR_BLOCK_MARKER = 0xFF

Y_COEFF  = np.array([0.28571429,  0.57142857,  0.14285714])
CB_COEFF = np.array([-0.14285714, -0.28571429,  0.42857143])
CR_COEFF = np.array([ 0.35714286, -0.28571429, -0.07142857])
BLOCK_W, BLOCK_H = 2, 2
BYTES_PER_BLOCK  = 6  # 4Y + Cb + Cr

# 新增常量
ZONE_HEIGHT_PIXELS = 16  # 每个区域的像素高度
ZONE_HEIGHT_BIG_BLOCKS = ZONE_HEIGHT_PIXELS // (BLOCK_H * 2)  # 每个区域的4x4大块行数 (16像素 = 4行4x4大块)

MINI_CODEBOOK_SIZE = 16  # 每个分段码本的大小
MEDIUM_CODEBOOK_SIZE = 64  # 每个中等码本的大小
DEFAULT_ENABLED_SEGMENTS_BITMAP = 0xFFFF  # 默认启用前16段的bitmap (1111111111111111)
DEFAULT_ENABLED_MEDIUM_SEGMENTS_BITMAP = 0x0F  # 默认启用前4个中码表段的bitmap (00001111)
SKIP_MARKER_4BIT = 0xF   # 4bit跳过标记（已废弃）
SKIP_MARKER_6BIT = 0x3F  # 6bit跳过标记（已废弃）

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
                        
                        # 不再预处理Y值，直接存储完整Y
                        y_values[idx] = np.uint8(clip_value(y, 0.0, 255.0))
                        cb_sum += cb
                        cr_sum += cr
                        idx += 1
            
            # Store Y values
            block_array[by, bx, 0:4] = y_values
            
            # Compute and store chroma - 直接存储Cb和Cr，不再预处理
            cb_mean = cb_sum / 4.0
            cr_mean = cr_sum / 4.0
            
            # 直接存储Cb和Cr（有符号值转换为无符号字节存储）
            block_array[by, bx, 4] = np.uint8(np.int8(clip_value(cb_mean, -128.0, 127.0)))
            block_array[by, bx, 5] = np.uint8(np.int8(clip_value(cr_mean, -128.0, 127.0)))
    
    return block_array

def pack_yuv420_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """使用Numba加速的整帧YUV转换包装函数"""
    return pack_yuv420_frame_numba(frame_bgr)

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

def calculate_color_block_distance(color_block: np.ndarray, codebook_entry: np.ndarray) -> float:
    """计算色块与码本项的距离"""
    # 将色度分量转换为有符号值进行比较
    color_block_float = color_block.astype(np.float32)
    codebook_float = codebook_entry.astype(np.float32)
    
    # 色度分量需要转换为有符号值
    for i in range(4, BYTES_PER_BLOCK):
        color_block_float[i] = color_block[i].view(np.int8).astype(np.float32)
        codebook_float[i] = codebook_entry[i].view(np.int8).astype(np.float32)
    
    # 计算欧几里得距离
    diff = color_block_float - codebook_float
    distance = np.sqrt(np.sum(diff * diff))
    
    return distance

@njit(cache=True, fastmath=True)
def classify_4x4_blocks_unified_numba(blocks, variance_threshold=5.0):
    blocks_h, blocks_w = blocks.shape[:2]
    big_blocks_h = blocks_h // 2
    big_blocks_w = blocks_w // 2
    all_blocks = []
    block_types_list = []  # list of (big_by, big_bx, type, indices)
    for big_by in range(big_blocks_h):
        for big_bx in range(big_blocks_w):
            blocks_4x4 = []
            for sub_by in range(2):
                for sub_bx in range(2):
                    by = big_by * 2 + sub_by
                    bx = big_bx * 2 + sub_bx
                    if by < blocks_h and bx < blocks_w:
                        blocks_4x4.append(blocks[by, bx])
                    else:
                        blocks_4x4.append(np.zeros(BYTES_PER_BLOCK, dtype=np.uint8))
            all_2x2_blocks_are_uniform = True
            for block in blocks_4x4:
                y = block[:4].astype(np.float32)
                v = ((y[0] - y[1]) ** 2 + (y[0] - y[2]) ** 2 + (y[0] - y[3]) ** 2 + (y[1] - y[2]) ** 2 + (y[1] - y[3]) ** 2 + (y[2] - y[3]) ** 2) / 6.0
                if v > variance_threshold:
                    all_2x2_blocks_are_uniform = False
                    break
            if all_2x2_blocks_are_uniform:
                downsampled_block = np.zeros(BYTES_PER_BLOCK, dtype=np.uint8)
                y_values = np.zeros(4, dtype=np.uint8)
                cb_values = np.zeros(4, dtype=np.int8)
                cr_values = np.zeros(4, dtype=np.int8)
                for i in range(4):
                    block = blocks_4x4[i]
                    y_values[i] = int(np.mean(block[:4]))
                    cb_values[i] = np.int8(block[4])
                    cr_values[i] = np.int8(block[5])
                downsampled_block[:4] = y_values
                val_cb = np.mean(cb_values)
                val_cb = min(max(val_cb, -128), 127)
                downsampled_block[4] = np.int8(val_cb)
                val_cr = np.mean(cr_values)
                val_cr = min(max(val_cr, -128), 127)
                downsampled_block[5] = np.int8(val_cr)
                block_idx = len(all_blocks)
                all_blocks.append(downsampled_block)
                block_types_list.append((big_by, big_bx, 0, [block_idx]))  # 0=color
            else:
                block_indices = []
                for block in blocks_4x4:
                    block_idx = len(all_blocks)
                    all_blocks.append(block)
                    block_indices.append(block_idx)
                block_types_list.append((big_by, big_bx, 1, block_indices))  # 1=detail
    return all_blocks, block_types_list

def classify_4x4_blocks_unified(blocks: np.ndarray, variance_threshold: float = 5.0) -> tuple:
    """4x4块分类为大色块和纹理块，用于统一码本，外部接口不变"""
    all_blocks, block_types_list = classify_4x4_blocks_unified_numba(blocks, variance_threshold)
    block_types = {}
    for big_by, big_bx, typ, indices in block_types_list:
        if typ == 0:
            block_types[(big_by, big_bx)] = ('color', indices)
        else:
            block_types[(big_by, big_bx)] = ('detail', indices)
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

def quantize_blocks_unified_segmented(blocks_data: np.ndarray, codebook: np.ndarray) -> tuple:
    """使用分段码本对块进行量化，返回段索引和段内索引"""
    if len(blocks_data) == 0:
        return np.array([], dtype=np.uint8), np.array([], dtype=np.uint8)
    
    # 只使用前255项进行量化
    effective_codebook = codebook[:EFFECTIVE_UNIFIED_CODEBOOK_SIZE]
    
    blocks_for_clustering = convert_blocks_for_clustering(blocks_data)
    codebook_for_clustering = convert_blocks_for_clustering(effective_codebook)
    
    # 使用Numba加速的距离计算
    indices = quantize_blocks_distance_numba(blocks_for_clustering, codebook_for_clustering)
    
    # 将索引转换为段索引和段内索引
    segment_indices = indices // MINI_CODEBOOK_SIZE
    within_segment_indices = indices % MINI_CODEBOOK_SIZE
    
    return segment_indices, within_segment_indices

def quantize_blocks_unified_medium(blocks_data: np.ndarray, codebook: np.ndarray) -> tuple:
    """使用中等码本对块进行量化，返回段索引和段内索引"""
    if len(blocks_data) == 0:
        return np.array([], dtype=np.uint8), np.array([], dtype=np.uint8)
    
    # 只使用前255项进行量化（4个中码表 × 64项，最后一段可能不满）
    max_medium_items = min(4 * MEDIUM_CODEBOOK_SIZE, EFFECTIVE_UNIFIED_CODEBOOK_SIZE)
    effective_codebook = codebook[:max_medium_items]
    
    blocks_for_clustering = convert_blocks_for_clustering(blocks_data)
    codebook_for_clustering = convert_blocks_for_clustering(effective_codebook)
    
    # 使用Numba加速的距离计算
    indices = quantize_blocks_distance_numba(blocks_for_clustering, codebook_for_clustering)
    
    # 将索引转换为段索引和段内索引
    segment_indices = indices // MEDIUM_CODEBOOK_SIZE
    within_segment_indices = indices % MEDIUM_CODEBOOK_SIZE
    
    # 修正最后一段的越界（最后一段实际项数可能小于64）
    last_segment = (max_medium_items - 1) // MEDIUM_CODEBOOK_SIZE
    last_segment_size = max_medium_items - last_segment * MEDIUM_CODEBOOK_SIZE
    mask = (segment_indices == last_segment) & (within_segment_indices >= last_segment_size)
    within_segment_indices[mask] = last_segment_size - 1  # 越界的都指向最后一个有效项
    
    return segment_indices, within_segment_indices

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

def encode_i_frame_unified(blocks: np.ndarray, unified_codebook: np.ndarray, 
                          block_types: dict, color_fallback_threshold: float = 50.0) -> bytes:
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
                        # 色块：先尝试色块编码，如果距离过大则回退到纹理块
                        # 从原始blocks重建平均块
                        blocks_4x4 = []
                        for sub_by in range(2):
                            for sub_bx in range(2):
                                by = big_by * 2 + sub_by
                                bx = big_bx * 2 + sub_bx
                                if by < blocks_h and bx < blocks_w:
                                    blocks_4x4.append(blocks[by, bx])
                        
                        avg_block = np.mean(blocks_4x4, axis=0).round().astype(np.uint8)
                        for i in range(4, 6):  # 现在只有6个元素：4Y + Cb + Cr
                            avg_val = np.mean([b[i].view(np.int8) for b in blocks_4x4])
                            avg_block[i] = np.clip(avg_val, -128, 127).astype(np.int8).view(np.uint8)
                        
                        # 计算色块与最佳码本项的距离
                        unified_idx = quantize_blocks_unified(avg_block.reshape(1, -1), unified_codebook)[0]
                        best_codebook_entry = unified_codebook[unified_idx]
                        color_distance = calculate_color_block_distance(avg_block, best_codebook_entry)
                        
                        # 如果距离过大（阈值设为50），回退到纹理块模式
                        if color_distance > color_fallback_threshold:
                            # 回退到纹理块：4个码本索引
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
                            # 色块编码：标记0xFF + 1个码本索引
                            data.append(COLOR_BLOCK_MARKER)
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
                          diff_threshold: float, force_i_threshold: float = 0.7,
                          enabled_segments_bitmap: int = DEFAULT_ENABLED_SEGMENTS_BITMAP,
                          enabled_medium_segments_bitmap: int = DEFAULT_ENABLED_MEDIUM_SEGMENTS_BITMAP,
                          color_fallback_threshold: float = 50.0) -> tuple:
    """差分编码P帧（统一码本，三级分段编码）- 新的bitmap+bitstream格式"""
    # 将bitmap参数转换为np.uint16类型
    enabled_segments_bitmap = np.uint16(enabled_segments_bitmap)
    enabled_medium_segments_bitmap = np.uint8(enabled_medium_segments_bitmap)
    
    if prev_blocks is None or current_blocks.shape != prev_blocks.shape:
        i_frame_data = encode_i_frame_unified(current_blocks, unified_codebook, block_types)
        return i_frame_data, True, 0, 0, 0, 0, 0, 0, 0, 0, 0, set(), set(), [], [], []
    
    blocks_h, blocks_w = current_blocks.shape[:2]
    total_blocks = blocks_h * blocks_w
    
    if total_blocks == 0:
        return b'', True, 0, 0, 0, 0, 0, 0, 0, 0, 0, set(), set(), [], [], []
    
    # 使用Numba加速的2x2块差异计算
    current_flat = current_blocks.reshape(-1, BYTES_PER_BLOCK)
    prev_flat = prev_blocks.reshape(-1, BYTES_PER_BLOCK)
    block_diffs = compute_2x2_block_differences_numba(current_flat, prev_flat, blocks_h, blocks_w)
    
    big_blocks_h = blocks_h // 2
    big_blocks_w = blocks_w // 2
    
    # 计算区域数量
    zones_count = (big_blocks_h + ZONE_HEIGHT_BIG_BLOCKS - 1) // ZONE_HEIGHT_BIG_BLOCKS
    
    # 获取启用的段索引列表
    enabled_segments = []
    for seg_idx in range(16):  # 最多16段
        # 使用np.uint16确保位运算安全
        if enabled_segments_bitmap & (np.uint16(1) << np.uint16(seg_idx)):
            enabled_segments.append(seg_idx)
    
    # 计算最大启用段索引
    max_enabled_segment = max(enabled_segments) if enabled_segments else -1
    
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
                    for i in range(4, 6):  # 现在只有6个元素：4Y + Cb + Cr
                        avg_val = np.mean([b[i].view(np.int8) for b in blocks_4x4])
                        avg_block[i] = np.clip(avg_val, -128, 127).astype(np.int8).view(np.uint8)
                    
                    # 计算色块与最佳码本项的距离
                    color_idx = quantize_blocks_unified(avg_block.reshape(1, -1), unified_codebook)[0]
                    best_codebook_entry = unified_codebook[color_idx]
                    color_distance = calculate_color_block_distance(avg_block, best_codebook_entry)
                    
                    # 如果距离过大（阈值设为50），回退到纹理块模式
                    if color_distance > color_fallback_threshold:
                        # 回退到纹理块：收集更新信息
                        update_info = {
                            'zone_relative_idx': zone_relative_idx,
                            'subblock_needs_update': subblock_needs_update,
                            'positions': positions
                        }
                        
                        # 计算所有子块的编码信息
                        small_segment_indices = []
                        small_within_indices = []
                        medium_segment_indices = []
                        medium_within_indices = []
                        full_indices = []
                        
                        for i, (by, bx) in enumerate(positions):
                            if by < blocks_h and bx < blocks_w:
                                block = current_blocks[by, bx]
                                # 小码表编码
                                small_seg_idx, small_within_idx = quantize_blocks_unified_segmented(block.reshape(1, -1), unified_codebook)
                                small_segment_indices.append(small_seg_idx[0])
                                small_within_indices.append(small_within_idx[0])
                                # 中码表编码
                                medium_seg_idx, medium_within_idx = quantize_blocks_unified_medium(block.reshape(1, -1), unified_codebook)
                                medium_segment_indices.append(medium_seg_idx[0])
                                medium_within_indices.append(medium_within_idx[0])
                                # 完整索引
                                full_idx = quantize_blocks_unified(block.reshape(1, -1), unified_codebook)[0]
                                full_indices.append(full_idx)
                            else:
                                # 无效位置
                                small_segment_indices.append(0)
                                small_within_indices.append(0)
                                medium_segment_indices.append(0)
                                medium_within_indices.append(0)
                                full_indices.append(0)
                        
                        update_info.update({
                            'small_segment_indices': small_segment_indices,
                            'small_within_indices': small_within_indices,
                            'medium_segment_indices': medium_segment_indices,
                            'medium_within_indices': medium_within_indices,
                            'full_indices': full_indices
                        })
                        
                        zone_detail_updates[zone_idx].append(update_info)
                    else:
                        # 色块编码：添加到色块更新列表
                        zone_color_updates[zone_idx].append((zone_relative_idx, color_idx))
                else:
                    # 纹理块更新：收集更新信息
                    update_info = {
                        'zone_relative_idx': zone_relative_idx,
                        'subblock_needs_update': subblock_needs_update,
                        'positions': positions
                    }
                    
                    # 计算所有子块的编码信息
                    small_segment_indices = []
                    small_within_indices = []
                    medium_segment_indices = []
                    medium_within_indices = []
                    full_indices = []
                    
                    for i, (by, bx) in enumerate(positions):
                        if by < blocks_h and bx < blocks_w:
                            block = current_blocks[by, bx]
                            # 小码表编码
                            small_seg_idx, small_within_idx = quantize_blocks_unified_segmented(block.reshape(1, -1), unified_codebook)
                            small_segment_indices.append(small_seg_idx[0])
                            small_within_indices.append(small_within_idx[0])
                            # 中码表编码
                            medium_seg_idx, medium_within_idx = quantize_blocks_unified_medium(block.reshape(1, -1), unified_codebook)
                            medium_segment_indices.append(medium_seg_idx[0])
                            medium_within_indices.append(medium_within_idx[0])
                            # 完整索引
                            full_idx = quantize_blocks_unified(block.reshape(1, -1), unified_codebook)[0]
                            full_indices.append(full_idx)
                        else:
                            # 无效位置
                            small_segment_indices.append(0)
                            small_within_indices.append(0)
                            medium_segment_indices.append(0)
                            medium_within_indices.append(0)
                            full_indices.append(0)
                    
                    update_info.update({
                        'small_segment_indices': small_segment_indices,
                        'small_within_indices': small_within_indices,
                        'medium_segment_indices': medium_segment_indices,
                        'medium_within_indices': medium_within_indices,
                        'full_indices': full_indices
                    })
                    
                    zone_detail_updates[zone_idx].append(update_info)
    
    # 判断是否需要I帧
    update_ratio = total_updated_blocks / total_blocks
    if update_ratio > force_i_threshold:
        i_frame_data = encode_i_frame_unified(current_blocks, unified_codebook, block_types)
        return i_frame_data, True, 0, 0, 0, 0, 0, 0, 0, 0, 0, set(), set(), [], [], []
    
    # 编码P帧
    data = bytearray()
    data.append(FRAME_TYPE_P)
    
    # 统计使用的区域数量
    used_zones = 0
    total_color_updates = 0
    total_detail_updates = 0
    
    # 码表使用统计
    small_updates = 0
    medium_updates = 0
    full_updates = 0
    small_bytes = 0
    medium_bytes = 0
    full_bytes = 0
    small_segments = defaultdict(int)
    medium_segments = defaultdict(int)
    small_blocks_per_update = []
    medium_blocks_per_update = []
    full_blocks_per_update = []
    
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
    
    # 按区域编码纹理块更新（新的bitmap+bitstream格式）
    for zone_idx in range(zones_count):
        if detail_zone_bitmap & (1 << zone_idx):
            detail_updates = zone_detail_updates[zone_idx]
            
            # 分离三种编码模式并使用新格式
            def encode_zone_with_new_format(updates_list, get_segment_info_func, get_indices_func, bits_per_index, mode_name):
                nonlocal small_updates, medium_updates, full_updates
                nonlocal small_bytes, medium_bytes, full_bytes
                nonlocal small_segments, medium_segments
                nonlocal small_blocks_per_update, medium_blocks_per_update, full_blocks_per_update
                
                # 按段分组
                segments_data = defaultdict(list)
                for update_info in updates_list:
                    segment_info = get_segment_info_func(update_info)
                    if segment_info:
                        seg_idx, used_segments = segment_info
                        for seg in used_segments:
                            segments_data[seg].append(update_info)
                
                # 为每个有效段编码
                for seg_idx in sorted(segments_data.keys()):
                    seg_updates = segments_data[seg_idx]
                    
                    # 统计更新次数
                    if mode_name == 'small':
                        small_updates += len(seg_updates)
                        for _ in range(len(seg_updates)):
                            small_segments[seg_idx] += 1
                    elif mode_name == 'medium':
                        medium_updates += len(seg_updates)
                        for _ in range(len(seg_updates)):
                            medium_segments[seg_idx] += 1
                    elif mode_name == 'full':
                        full_updates += len(seg_updates)
                    
                    # 新格式编码
                    num_blocks = len(seg_updates)
                    data.append(num_blocks)
                    
                    # 计算bitmap和位置数据的大小
                    bitmap_indices_size = (num_blocks >> 1) * 3  # 每2个块用3字节：1个bitmap + 2个位置
                    if num_blocks & 1:
                        bitmap_indices_size += 2  # 最后一个奇数块用2字节：1个bitmap + 1个位置
                    
                    # 收集bitmap和位置信息，以及所有有效子块的索引
                    bitmap_and_position_data = bytearray()
                    all_valid_indices = []
                    
                    # 处理每2个块为一组
                    for i in range(0, num_blocks, 2):
                        if i + 1 < num_blocks:
                            # 处理一对块
                            update1 = seg_updates[i]
                            update2 = seg_updates[i + 1]
                            
                            # 生成8位bitmap（前4位是第一个块，后4位是第二个块）
                            bitmap = 0
                            indices1 = get_indices_func(update1, seg_idx)
                            indices2 = get_indices_func(update2, seg_idx)
                            
                            # 按bitmap位顺序收集索引
                            for j in range(4):
                                if update1['subblock_needs_update'][j]:
                                    bitmap |= (1 << j)
                                    all_valid_indices.append(indices1[j])
                            for j in range(4):
                                if update2['subblock_needs_update'][j]:
                                    bitmap |= (1 << (j + 4))
                                    all_valid_indices.append(indices2[j])
                            
                            bitmap_and_position_data.append(bitmap)
                            bitmap_and_position_data.append(update1['zone_relative_idx'])
                            bitmap_and_position_data.append(update2['zone_relative_idx'])
                        else:
                            # 处理最后一个奇数块
                            update1 = seg_updates[i]
                            
                            # 生成4位bitmap（只用低4位，高4位填0）
                            bitmap = 0
                            indices1 = get_indices_func(update1, seg_idx)
                            
                            for j in range(4):
                                if update1['subblock_needs_update'][j]:
                                    bitmap |= (1 << j)
                                    all_valid_indices.append(indices1[j])
                            
                            bitmap_and_position_data.append(bitmap)
                            bitmap_and_position_data.append(update1['zone_relative_idx'])
                    
                    # 写入bitmap和位置信息
                    data.extend(bitmap_and_position_data)
                    
                    # 写入bitstream
                    if all_valid_indices:
                        # 计算需要的位数
                        total_bits = len(all_valid_indices) * bits_per_index
                        total_bytes = (total_bits + 7) // 8  # 向上取整到字节
                        
                        # 打包索引到bitstream
                        bitstream = bytearray(total_bytes)
                        bit_pos = 0
                        
                        for idx in all_valid_indices:
                            # 将索引写入bitstream
                            for bit in range(bits_per_index):
                                if idx & (1 << bit):
                                    byte_pos = bit_pos // 8
                                    bit_in_byte = bit_pos % 8
                                    bitstream[byte_pos] |= (1 << bit_in_byte)
                                bit_pos += 1
                        
                        data.extend(bitstream)
                        
                        # 统计字节数
                        total_segment_bytes = 1 + len(bitmap_and_position_data) + len(bitstream)  # num_blocks + bitmap_indices + bitstream
                        if mode_name == 'small':
                            small_bytes += total_segment_bytes
                            small_blocks_per_update.extend([sum(update['subblock_needs_update']) for update in seg_updates])
                        elif mode_name == 'medium':
                            medium_bytes += total_segment_bytes
                            medium_blocks_per_update.extend([sum(update['subblock_needs_update']) for update in seg_updates])
                        elif mode_name == 'full':
                            full_bytes += total_segment_bytes
                            full_blocks_per_update.extend([sum(update['subblock_needs_update']) for update in seg_updates])
            
            # 小码表编码
            def get_small_segment_info(update_info):
                small_segment_indices = update_info['small_segment_indices']
                small_within_indices = update_info['small_within_indices']
                subblock_needs_update = update_info['subblock_needs_update']
                
                used_segments = set()
                for i, (seg_idx, within_idx) in enumerate(zip(small_segment_indices, small_within_indices)):
                    if subblock_needs_update[i] and seg_idx < 16 and (enabled_segments_bitmap & (np.uint16(1) << np.uint16(seg_idx))):
                        used_segments.add(seg_idx)
                
                if len(used_segments) == 1:
                    return list(used_segments)[0], used_segments
                return None
            
            def get_small_indices(update_info, seg_idx):
                indices = []
                for i in range(4):
                    if update_info['small_segment_indices'][i] == seg_idx:
                        indices.append(update_info['small_within_indices'][i])
                    else:
                        indices.append(0)  # 占位，实际不会用到
                return indices
            
            # 中码表编码
            def get_medium_segment_info(update_info):
                medium_segment_indices = update_info['medium_segment_indices']
                medium_within_indices = update_info['medium_within_indices']
                subblock_needs_update = update_info['subblock_needs_update']
                
                used_segments = set()
                for i, (seg_idx, within_idx) in enumerate(zip(medium_segment_indices, medium_within_indices)):
                    if subblock_needs_update[i] and seg_idx < 4 and (enabled_medium_segments_bitmap & (np.uint8(1) << np.uint8(seg_idx))):
                        used_segments.add(seg_idx)
                
                if len(used_segments) == 1:
                    return list(used_segments)[0], used_segments
                return None
            
            def get_medium_indices(update_info, seg_idx):
                indices = []
                for i in range(4):
                    if update_info['medium_segment_indices'][i] == seg_idx:
                        indices.append(update_info['medium_within_indices'][i])
                    else:
                        indices.append(0)  # 占位，实际不会用到
                return indices
            
            # 完整索引编码
            def get_full_segment_info(update_info):
                # 完整索引不需要段信息检查，直接返回段0
                return 0, {0}
            
            def get_full_indices(update_info, seg_idx):
                return update_info['full_indices']
            
            # 分类更新并使用新格式编码
            small_updates_list = []
            medium_updates_list = []
            full_updates_list = []
            
            for update_info in detail_updates:
                # 检查小码表可行性
                small_info = get_small_segment_info(update_info)
                if small_info:
                    small_updates_list.append(update_info)
                else:
                    # 检查中码表可行性
                    medium_info = get_medium_segment_info(update_info)
                    if medium_info:
                        medium_updates_list.append(update_info)
                    else:
                        # 使用完整索引
                        full_updates_list.append(update_info)
            
            # 按段分组编码
            # 小码表段编码
            small_segments_grouped = defaultdict(list)
            for update_info in small_updates_list:
                info = get_small_segment_info(update_info)
                if info:
                    seg_idx, _ = info
                    small_segments_grouped[seg_idx].append(update_info)
            
            # 写入小码表启用bitmap
            actual_enabled_segments_bitmap = np.uint16(0)
            for seg_idx in small_segments_grouped.keys():
                if seg_idx < 16:
                    actual_enabled_segments_bitmap |= (np.uint16(1) << np.uint16(seg_idx))
            
            data.extend(struct.pack('<H', actual_enabled_segments_bitmap))
            
            # 编码每个小码表段
            for seg_idx in range(16):
                if actual_enabled_segments_bitmap & (np.uint16(1) << np.uint16(seg_idx)):
                    seg_updates = small_segments_grouped[seg_idx]
                    # 修复：直接传递seg_updates列表，而不是嵌套在列表中
                    encode_zone_with_new_format(seg_updates, lambda x: (seg_idx, {seg_idx}), 
                                              lambda update_info, s_idx: get_small_indices(update_info, s_idx), 4, 'small')
            
            # 中码表段编码
            medium_segments_grouped = defaultdict(list)
            for update_info in medium_updates_list:
                info = get_medium_segment_info(update_info)
                if info:
                    seg_idx, _ = info
                    medium_segments_grouped[seg_idx].append(update_info)
            
            # 写入中码表启用bitmap
            actual_enabled_medium_segments_bitmap = np.uint8(0)
            for seg_idx in medium_segments_grouped.keys():
                if seg_idx < 4:
                    actual_enabled_medium_segments_bitmap |= (np.uint8(1) << np.uint8(seg_idx))
            
            data.append(actual_enabled_medium_segments_bitmap)
            
            # 编码每个中码表段
            for seg_idx in range(4):
                if actual_enabled_medium_segments_bitmap & (np.uint8(1) << np.uint8(seg_idx)):
                    seg_updates = medium_segments_grouped[seg_idx]
                    # 计算本段实际项数
                    if seg_idx == 3:
                        # 最后一段
                        max_medium_items = min(4 * MEDIUM_CODEBOOK_SIZE, EFFECTIVE_UNIFIED_CODEBOOK_SIZE)
                        last_segment_size = max_medium_items - 3 * MEDIUM_CODEBOOK_SIZE
                    else:
                        last_segment_size = MEDIUM_CODEBOOK_SIZE
                    # 传递本段实际项数给encode_zone_with_new_format（如需用到）
                    encode_zone_with_new_format(seg_updates, lambda x: (seg_idx, {seg_idx}), 
                                              lambda update_info, s_idx: get_medium_indices(update_info, s_idx), 6, 'medium')
            
            # 完整索引编码
            if full_updates_list:
                # 修复：直接传递full_updates_list，而不是嵌套在列表中
                encode_zone_with_new_format(full_updates_list, get_full_segment_info, get_full_indices, 8, 'full')
            else:
                data.append(0)  # 没有完整索引更新
    
    # 按区域编码色块更新（逻辑不变）
    for zone_idx in range(zones_count):
        if color_zone_bitmap & (1 << zone_idx):
            color_updates = zone_color_updates[zone_idx]
            data.append(len(color_updates))
            
            # 存储色块更新
            for relative_idx, unified_idx in color_updates:
                data.append(relative_idx)
                data.append(unified_idx)
    
    return bytes(data), False, used_zones, total_color_updates, total_detail_updates, small_updates, medium_updates, full_updates, small_bytes, medium_bytes, full_bytes, small_segments, medium_segments, small_blocks_per_update, medium_blocks_per_update, full_blocks_per_update

@njit(cache=True, fastmath=True)
def accumulate_unchanged_blocks_numba(current_blocks_flat: np.ndarray, prev_blocks_flat: np.ndarray, 
                                    block_diffs: np.ndarray, diff_threshold: float) -> None:
    """将未更新的块从前一帧累积到当前帧 - Numba加速版本
    
    Args:
        current_blocks_flat: 当前帧的扁平化块数据 (会被原地修改)
        prev_blocks_flat: 前一帧的扁平化块数据
        block_diffs: 块差异数组
        diff_threshold: 差异阈值
    """
    total_blocks = block_diffs.size
    
    for i in range(total_blocks):
        if block_diffs.flat[i] <= diff_threshold:
            # 如果差异小于阈值，将前一帧的块复制到当前帧
            for j in range(BYTES_PER_BLOCK):
                current_blocks_flat[i, j] = prev_blocks_flat[i, j]

def accumulate_unchanged_blocks(current_blocks: np.ndarray, prev_blocks: np.ndarray, 
                              diff_threshold: float) -> np.ndarray:
    """将未更新的块从前一帧累积到当前帧，以避免渐变残影
    
    Args:
        current_blocks: 当前帧的块数据
        prev_blocks: 前一帧的块数据
        diff_threshold: 差异阈值
    
    Returns:
        修改后的当前帧块数据（累积了未更新的块）
    """
    if prev_blocks is None or current_blocks.shape != prev_blocks.shape:
        return current_blocks.copy()
    
    blocks_h, blocks_w = current_blocks.shape[:2]
    
    # 创建当前帧的副本以避免修改原始数据
    accumulated_blocks = current_blocks.copy()
    
    # 计算块差异
    current_flat = current_blocks.reshape(-1, BYTES_PER_BLOCK)
    prev_flat = prev_blocks.reshape(-1, BYTES_PER_BLOCK)
    accumulated_flat = accumulated_blocks.reshape(-1, BYTES_PER_BLOCK)
    
    block_diffs = compute_2x2_block_differences_numba(current_flat, prev_flat, blocks_h, blocks_w)
    
    # 使用Numba加速的累积更新
    accumulate_unchanged_blocks_numba(accumulated_flat, prev_flat, block_diffs, diff_threshold)
    
    return accumulated_blocks
