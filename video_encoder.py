#!/usr/bin/env python3

import argparse, cv2, numpy as np, pathlib, textwrap
import struct
import concurrent.futures
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from collections import defaultdict
import statistics
from numba import jit, njit, types
from numba.typed import List

from dither_opt import apply_dither_optimized

WIDTH, HEIGHT = 240, 160
DEFAULT_STRIP_COUNT = 4
DEFAULT_UNIFIED_CODEBOOK_SIZE = 256   # 统一码本大小
EFFECTIVE_UNIFIED_CODEBOOK_SIZE = 254  # 有效码本大小（0xFF保留）
DEFAULT_BIG_BLOCK_CODEBOOK_SIZE = 256  # 4x4大块码表大小

# 标记常量
# COLOR_BLOCK_MARKER = 0xFF
BIG_BLOCK_MARKER = 0xFE

Y_COEFF  = np.array([0.28571429,  0.57142857,  0.14285714])
CB_COEFF = np.array([-0.14285714, -0.28571429,  0.42857143])
CR_COEFF = np.array([ 0.35714286, -0.28571429, -0.07142857])
BLOCK_W, BLOCK_H = 2, 2
BYTES_PER_BLOCK  = 7  # 4Y + d_r + d_g + d_b
BYTES_PER_BIG_BLOCK = 28  # 16Y + 4*(d_r + d_g + d_b)

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
def pack_yuv420_strip_numba(bgr_strip, strip_height, width):
    """Numba加速的YUV420转换"""
    blocks_h = strip_height // BLOCK_H
    blocks_w = width // BLOCK_W
    
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
                    if y_start + dy < strip_height and x_start + dx < width:
                        b = float(bgr_strip[y_start + dy, x_start + dx, 0])
                        g = float(bgr_strip[y_start + dy, x_start + dx, 1])  
                        r = float(bgr_strip[y_start + dy, x_start + dx, 2])
                        
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

def pack_yuv420_strip(frame_bgr: np.ndarray, strip_y: int, strip_height: int) -> np.ndarray:
    """使用Numba加速的YUV转换包装函数"""
    strip_bgr = frame_bgr[strip_y:strip_y + strip_height, :, :]
    return pack_yuv420_strip_numba(strip_bgr, strip_height, WIDTH)

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

def generate_big_block_codebook(big_blocks: list, codebook_size: int = DEFAULT_BIG_BLOCK_CODEBOOK_SIZE, 
                               max_iter: int = 100) -> np.ndarray:
    """生成4x4大块码表"""
    if len(big_blocks) == 0:
        return np.zeros((codebook_size, BYTES_PER_BIG_BLOCK), dtype=np.uint8)
    
    big_blocks_array = np.array(big_blocks)
    if len(big_blocks_array) <= codebook_size:
        # 数据量小于码表大小
        codebook = np.zeros((codebook_size, BYTES_PER_BIG_BLOCK), dtype=np.uint8)
        codebook[:len(big_blocks_array)] = big_blocks_array
        if len(big_blocks_array) > 0:
            for i in range(len(big_blocks_array), codebook_size):
                codebook[i] = big_blocks_array[-1]
        return codebook
    
    # 使用K-Means聚类
    big_blocks_for_clustering = convert_big_blocks_for_clustering(big_blocks_array)
    kmeans = MiniBatchKMeans(
        n_clusters=codebook_size,
        random_state=42,
        batch_size=min(1000, len(big_blocks_array)),
        max_iter=max_iter,
        n_init=3
    )
    kmeans.fit(big_blocks_for_clustering)
    codebook = convert_big_block_codebook_from_clustering(kmeans.cluster_centers_)
    
    return codebook

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

def convert_big_blocks_for_clustering(big_blocks: np.ndarray) -> np.ndarray:
    """将4x4大块转换为聚类格式"""
    if len(big_blocks) == 0:
        return big_blocks.astype(np.float32)
    
    if big_blocks.ndim > 2:
        big_blocks = big_blocks.reshape(-1, BYTES_PER_BIG_BLOCK)
    
    big_blocks_float = big_blocks.astype(np.float32)
    
    # 色度分量需要转换为有符号数
    for i in range(16, BYTES_PER_BIG_BLOCK):
        big_blocks_float[:, i] = big_blocks[:, i].view(np.int8).astype(np.float32)
    
    return big_blocks_float

def convert_big_block_codebook_from_clustering(codebook_float: np.ndarray) -> np.ndarray:
    """将聚类结果转换回4x4大块格式"""
    codebook = np.zeros_like(codebook_float, dtype=np.uint8)
    
    # Y分量
    codebook[:, 0:16] = np.clip(codebook_float[:, 0:16].round(), 0, 255).astype(np.uint8)
    
    # 色度分量
    for i in range(16, BYTES_PER_BIG_BLOCK):
        clipped_values = np.clip(codebook_float[:, i].round(), -128, 127).astype(np.int8)
        codebook[:, i] = clipped_values.view(np.uint8)
    
    return codebook

def quantize_big_blocks(big_blocks: list, big_block_codebook: np.ndarray) -> tuple:
    """量化4x4大块，返回索引和重建的块"""
    if len(big_blocks) == 0:
        return np.array([], dtype=np.uint8), []
    
    big_blocks_array = np.array(big_blocks)
    big_blocks_for_clustering = convert_big_blocks_for_clustering(big_blocks_array)
    codebook_for_clustering = convert_big_blocks_for_clustering(big_block_codebook)
    
    # 计算距离和找到最近的码字
    distances = cdist(big_blocks_for_clustering, codebook_for_clustering, metric='euclidean')
    indices = np.argmin(distances, axis=1).astype(np.uint8)
    
    # 重建块
    reconstructed_big_blocks = [big_block_codebook[idx] for idx in indices]
    
    return indices, reconstructed_big_blocks

def calculate_distortion_sad(original_blocks: list, reconstructed_blocks: list) -> float:
    """计算失真度量 - SAD (Sum of Absolute Differences)"""
    if len(original_blocks) != len(reconstructed_blocks):
        return float('inf')
    
    total_sad = 0.0
    for orig, recon in zip(original_blocks, reconstructed_blocks):
        # 只计算Y分量的SAD
        y_orig = orig[:4].astype(np.float32)
        y_recon = recon[:4].astype(np.float32)
        total_sad += np.sum(np.abs(y_orig - y_recon))
    
    return total_sad / len(original_blocks)  # 平均SAD

def calculate_distortion_mse(original_blocks: list, reconstructed_blocks: list) -> float:
    """计算失真度量 - MSE (Mean Squared Error)"""
    if len(original_blocks) != len(reconstructed_blocks):
        return float('inf')
    
    total_mse = 0.0
    for orig, recon in zip(original_blocks, reconstructed_blocks):
        # 只计算Y分量的MSE
        y_orig = orig[:4].astype(np.float32)
        y_recon = recon[:4].astype(np.float32)
        total_mse += np.sum((y_orig - y_recon) ** 2)
    
    return total_mse / (len(original_blocks) * 4)  # 平均MSE

# 默认使用SAD
calculate_distortion = calculate_distortion_sad

def classify_4x4_blocks_with_big_codebook(blocks: np.ndarray, big_block_codebook: np.ndarray,
                                        variance_threshold: float = 5.0, 
                                        distortion_threshold: float = 10.0) -> tuple:
    """使用4x4大块码表对4x4块进行分类"""
    blocks_h, blocks_w = blocks.shape[:2]
    big_blocks_h = blocks_h // 2
    big_blocks_w = blocks_w // 2
    
    big_block_indices = {}  # 使用4x4大块码表的块
    small_blocks = []       # 需要用2x2小块码表的块
    block_types = {}        # 记录每个4x4块的类型
    
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
            
            # 尝试用4x4大块码表
            big_block = pack_big_block_from_2x2_blocks(blocks_4x4)
            indices, reconstructed = quantize_big_blocks([big_block], big_block_codebook)
            
            if len(reconstructed) > 0:
                # 计算失真
                reconstructed_2x2_blocks = unpack_big_block_to_2x2_blocks(reconstructed[0])
                distortion = calculate_distortion(blocks_4x4, reconstructed_2x2_blocks)
                
                if distortion <= distortion_threshold:
                    # 失真可接受，使用4x4大块码表
                    big_block_indices[(big_by, big_bx)] = indices[0]
                    block_types[(big_by, big_bx)] = 'big_block'
                else:
                    # 失真太大，使用2x2小块码表
                    small_blocks.extend(blocks_4x4)
                    block_types[(big_by, big_bx)] = 'small_blocks'
            else:
                # 量化失败，使用2x2小块码表
                small_blocks.extend(blocks_4x4)
                block_types[(big_by, big_bx)] = 'small_blocks'
    
    return big_block_indices, small_blocks, block_types

def encode_strip_i_frame_with_big_blocks(blocks: np.ndarray, big_block_codebook: np.ndarray,
                                       small_block_codebook: np.ndarray, block_types: dict,
                                       big_block_indices: dict) -> bytes:
    """编码I帧条带（删除色块支持）"""
    data = bytearray()
    data.append(FRAME_TYPE_I)
    
    if blocks.size > 0:
        blocks_h, blocks_w = blocks.shape[:2]
        big_blocks_h = blocks_h // 2
        big_blocks_w = blocks_w // 2
        
        # 存储4x4大块码表
        data.extend(big_block_codebook.flatten().tobytes())
        
        # 存储2x2小块码表
        data.extend(small_block_codebook.flatten().tobytes())
        
        # 按4x4大块的顺序编码
        for big_by in range(big_blocks_h):
            for big_bx in range(big_blocks_w):
                if (big_by, big_bx) in block_types:
                    block_type = block_types[(big_by, big_bx)]
                    
                    if block_type == 'big_block':
                        # 4x4大块：0xFE + 1个大块码表索引
                        data.append(BIG_BLOCK_MARKER)
                        big_idx = big_block_indices[(big_by, big_bx)]
                        data.append(big_idx)
                        
                    else:  # small_blocks
                        # 纹理块：4个小块码表索引
                        for sub_by in range(2):
                            for sub_bx in range(2):
                                by = big_by * 2 + sub_by
                                bx = big_bx * 2 + sub_bx
                                if by < blocks_h and bx < blocks_w:
                                    block = blocks[by, bx]
                                    small_idx = quantize_blocks_unified(block.reshape(1, -1), small_block_codebook)[0]
                                    data.append(small_idx)
                                else:
                                    data.append(0)
    
    return bytes(data)

def generate_gop_codebooks_with_big_blocks(frames: list, strip_count: int, i_frame_interval: int,
                                         variance_threshold: float, diff_threshold: float,
                                         distortion_threshold: float = 10.0,
                                         big_block_codebook_size: int = DEFAULT_BIG_BLOCK_CODEBOOK_SIZE,
                                         small_block_codebook_size: int = EFFECTIVE_UNIFIED_CODEBOOK_SIZE,
                                         kmeans_max_iter: int = 100, i_frame_weight: int = 3) -> dict:
    """为每个GOP生成4x4大块码表和2x2小块码表（删除色块支持）"""
    print("正在为每个GOP生成大块码表和小块码表...")
    
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
            all_big_blocks = []
            all_small_blocks = []
            block_types_list = []
            
            # 处理GOP中的每一帧
            prev_strip_blocks = None
            
            for frame_idx in range(gop_start, gop_end):
                strip_blocks = frames[frame_idx][strip_idx]
                if strip_blocks.size == 0:
                    continue
                
                # 确定需要处理的大块
                is_i_frame = (frame_idx == gop_start)
                
                if is_i_frame:
                    blocks_h, blocks_w = strip_blocks.shape[:2]
                    big_blocks_h = blocks_h // 2
                    big_blocks_w = blocks_w // 2
                    updated_big_blocks = {(big_by, big_bx) for big_by in range(big_blocks_h) for big_bx in range(big_blocks_w)}
                else:
                    updated_big_blocks = identify_updated_big_blocks(strip_blocks, prev_strip_blocks, diff_threshold)
                
                # 从有效大块中提取数据用于训练码表
                for big_by, big_bx in updated_big_blocks:
                    blocks_4x4 = []
                    for sub_by in range(2):
                        for sub_bx in range(2):
                            by = big_by * 2 + sub_by
                            bx = big_bx * 2 + sub_bx
                            if by < strip_blocks.shape[0] and bx < strip_blocks.shape[1]:
                                blocks_4x4.append(strip_blocks[by, bx])
                            else:
                                blocks_4x4.append(np.zeros(BYTES_PER_BLOCK, dtype=np.uint8))
                    
                    # 添加到训练数据
                    big_block = pack_big_block_from_2x2_blocks(blocks_4x4)
                    if is_i_frame:
                        all_big_blocks.extend([big_block] * i_frame_weight)
                        all_small_blocks.extend(blocks_4x4 * i_frame_weight)
                    else:
                        all_big_blocks.append(big_block)
                        all_small_blocks.extend(blocks_4x4)
                
                prev_strip_blocks = strip_blocks.copy()
            
            # 生成码表
            big_block_codebook = generate_big_block_codebook(all_big_blocks, big_block_codebook_size, kmeans_max_iter)
            small_block_codebook = generate_unified_codebook_simplified(
                all_small_blocks, small_block_codebook_size, kmeans_max_iter)
            
            # 为每一帧生成分类信息
            for frame_idx in range(gop_start, gop_end):
                strip_blocks = frames[frame_idx][strip_idx]
                if strip_blocks.size == 0:
                    continue
                
                big_block_indices, _, block_types = classify_4x4_blocks_with_big_codebook(
                    strip_blocks, big_block_codebook, variance_threshold, distortion_threshold)
                block_types_list.append((frame_idx, block_types, big_block_indices))
            
            gop_codebooks[gop_start].append({
                'big_block_codebook': big_block_codebook,
                'small_block_codebook': small_block_codebook,
                'block_types_list': block_types_list,
                'distortion_threshold': distortion_threshold
            })
            
            print(f"    条带{strip_idx}: 大块{len(all_big_blocks)}个, 小块{len(all_small_blocks)}个")
    
    return gop_codebooks

def generate_unified_codebook_simplified(small_blocks: list, 
                                       codebook_size: int = EFFECTIVE_UNIFIED_CODEBOOK_SIZE,
                                       kmeans_max_iter: int = 100) -> np.ndarray:
    """生成2x2小块的统一码表（254项，避免0xFE）"""
    if small_blocks:
        blocks_array = np.array(small_blocks)
        codebook, _ = generate_codebook(blocks_array, codebook_size, kmeans_max_iter)
        
        # 创建254项码表
        full_codebook = np.zeros((codebook_size, BYTES_PER_BLOCK), dtype=np.uint8)
        actual_size = min(len(codebook), codebook_size)
        full_codebook[:actual_size] = codebook[:actual_size]
        
        # 填充剩余项
        if actual_size > 0:
            for i in range(actual_size, codebook_size):
                full_codebook[i] = full_codebook[actual_size - 1]
    else:
        full_codebook = np.zeros((codebook_size, BYTES_PER_BLOCK), dtype=np.uint8)
    
    return full_codebook

def encode_strip_p_frame_with_big_blocks(current_blocks: np.ndarray, prev_blocks: np.ndarray,
                                       big_block_codebook: np.ndarray, small_block_codebook: np.ndarray,
                                       block_types: dict, big_block_indices: dict,
                                       diff_threshold: float, force_i_threshold: float = 0.7,
                                       variance_threshold: float = 5.0, distortion_threshold: float = 10.0) -> tuple:
    """编码P帧条带（删除色块支持）"""
    if prev_blocks is None or current_blocks.shape != prev_blocks.shape:
        i_frame_data = encode_strip_i_frame_with_big_blocks(
            current_blocks, big_block_codebook, small_block_codebook, block_types, big_block_indices)
        return i_frame_data, True, 0, 0, 0
    
    blocks_h, blocks_w = current_blocks.shape[:2]
    total_blocks = blocks_h * blocks_w
    
    if total_blocks == 0:
        return b'', True, 0, 0, 0
    
    # 识别需要更新的大块
    updated_big_blocks = identify_updated_big_blocks(current_blocks, prev_blocks, diff_threshold)
    
    big_blocks_h = blocks_h // 2
    big_blocks_w = blocks_w // 2
    total_big_blocks = big_blocks_h * big_blocks_w
    
    # 判断是否需要I帧
    update_ratio = len(updated_big_blocks) / total_big_blocks if total_big_blocks > 0 else 0
    if update_ratio > force_i_threshold:
        i_frame_data = encode_strip_i_frame_with_big_blocks(
            current_blocks, big_block_codebook, small_block_codebook, block_types, big_block_indices)
        return i_frame_data, True, 0, 0, 0
    
    # 计算区域数量
    zones_count = (big_blocks_h + ZONE_HEIGHT_BIG_BLOCKS - 1) // ZONE_HEIGHT_BIG_BLOCKS
    if zones_count > 8:
        zones_count = 8
    
    # 按区域组织更新
    zone_detail_updates = [[] for _ in range(zones_count)]
    zone_big_block_updates = [[] for _ in range(zones_count)]
    
    for big_by, big_bx in updated_big_blocks:
        # 计算属于哪个区域
        zone_idx = min(big_by // ZONE_HEIGHT_BIG_BLOCKS, zones_count - 1)
        zone_relative_by = big_by % ZONE_HEIGHT_BIG_BLOCKS
        zone_relative_idx = zone_relative_by * big_blocks_w + big_bx
        
        if (big_by, big_bx) in block_types:
            block_type = block_types[(big_by, big_bx)]
            
            if block_type == 'big_block':
                # 4x4大块更新
                big_idx = big_block_indices[(big_by, big_bx)]
                zone_big_block_updates[zone_idx].append((zone_relative_idx, big_idx))
                
            else:  # small_blocks
                # 纹理块更新
                indices = []
                for sub_by in range(2):
                    for sub_bx in range(2):
                        by = big_by * 2 + sub_by
                        bx = big_bx * 2 + sub_bx
                        if by < blocks_h and bx < blocks_w:
                            block = current_blocks[by, bx]
                            small_idx = quantize_blocks_unified(block.reshape(1, -1), small_block_codebook)[0]
                            indices.append(small_idx)
                        else:
                            indices.append(0)
                zone_detail_updates[zone_idx].append((zone_relative_idx, indices))
    
    # 编码P帧
    data = bytearray()
    data.append(FRAME_TYPE_P)
    
    # 统计使用的区域数量
    used_zones = 0
    total_detail_updates = 0
    total_big_block_updates = 0
    
    # 生成区域bitmap
    zone_bitmap = 0
    for zone_idx in range(zones_count):
        if zone_detail_updates[zone_idx] or zone_big_block_updates[zone_idx]:
            zone_bitmap |= (1 << zone_idx)
            used_zones += 1
            total_detail_updates += len(zone_detail_updates[zone_idx])
            total_big_block_updates += len(zone_big_block_updates[zone_idx])
    
    data.append(zone_bitmap)
    
    # 按区域编码更新（现在只有2种类型）
    for zone_idx in range(zones_count):
        if zone_bitmap & (1 << zone_idx):
            detail_updates = zone_detail_updates[zone_idx]
            big_block_updates = zone_big_block_updates[zone_idx]
            
            data.append(len(detail_updates))
            data.append(len(big_block_updates))
            
            # 存储纹理块更新
            for relative_idx, indices in detail_updates:
                data.append(relative_idx)
                for idx in indices:
                    data.append(idx)
            
            # 存储4x4大块更新
            for relative_idx, big_idx in big_block_updates:
                data.append(relative_idx)
                data.append(big_idx)
    
    total_updates = total_detail_updates + total_big_block_updates
    return bytes(data), False, used_zones, 0, total_detail_updates

def pack_big_block_from_2x2_blocks(blocks_2x2: list) -> np.ndarray:
    """将4个2x2块组合成一个4x4大块"""
    big_block = np.zeros(BYTES_PER_BIG_BLOCK, dtype=np.uint8)
    
    # 存储16个Y值和4组色度信息
    y_offset = 0
    chroma_offset = 16
    
    for i, block in enumerate(blocks_2x2):
        # 复制4个Y值
        big_block[y_offset:y_offset+4] = block[:4]
        y_offset += 4
        
        # 复制色度信息
        big_block[chroma_offset:chroma_offset+3] = block[4:7]
        chroma_offset += 3
    
    return big_block

def unpack_big_block_to_2x2_blocks(big_block: np.ndarray) -> list:
    """将4x4大块拆分成4个2x2块"""
    blocks_2x2 = []
    
    for i in range(4):
        block = np.zeros(BYTES_PER_BLOCK, dtype=np.uint8)
        # 复制Y值
        y_start = i * 4
        block[:4] = big_block[y_start:y_start+4]
        # 复制色度信息
        chroma_start = 16 + i * 3
        block[4:7] = big_block[chroma_start:chroma_start+3]
        blocks_2x2.append(block)
    
    return blocks_2x2

def quantize_blocks_unified(blocks_data: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """使用统一码表对块进行量化（避免产生0xFE）"""
    if len(blocks_data) == 0:
        return np.array([], dtype=np.uint8)
    
    # 只使用前254项进行量化
    effective_codebook = codebook[:EFFECTIVE_UNIFIED_CODEBOOK_SIZE]
    
    blocks_for_clustering = convert_blocks_for_clustering(blocks_data)
    codebook_for_clustering = convert_blocks_for_clustering(effective_codebook)
    
    # 使用Numba加速的距离计算
    indices = quantize_blocks_distance_numba(blocks_for_clustering, codebook_for_clustering)
    
    return indices

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
    block_diffs = compute_block_differences_numba(current_flat, prev_flat, blocks_h, blocks_w)
    
    # 使用Numba加速的更新块识别
    updated_list = identify_updated_blocks_numba(block_diffs, diff_threshold, blocks_h, blocks_w)
    
    return set(updated_list)

@njit
def compute_block_differences_numba(current_flat, prev_flat, blocks_h, blocks_w):
    """Numba加速的块差异计算"""
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

class EncodingStats:
    """编码统计类 - 修复统计问题"""
    def __init__(self):
        # 帧统计
        self.total_frames_processed = 0
        self.total_i_frames = 0
        self.forced_i_frames = 0
        self.threshold_i_frames = 0
        self.total_p_frames = 0
        
        # 大小统计
        self.total_i_frame_bytes = 0
        self.total_p_frame_bytes = 0
        self.total_big_block_codebook_bytes = 0
        self.total_small_block_codebook_bytes = 0
        self.total_index_bytes = 0
        self.total_p_overhead_bytes = 0
        
        # 块类型统计 - 修复
        self.big_block_count = 0
        self.small_block_count = 0
        
        # P帧块更新统计 - 新增详细统计
        self.p_frame_updates = []
        self.zone_usage = defaultdict(int)
        self.detail_update_count = 0
        self.big_block_update_count = 0
        self.detail_update_bytes = 0  # 纹理块更新字节数
        self.big_block_update_bytes = 0  # 大块更新字节数
        
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
        
        # 修复码本统计 - 分别计算大块和小块码本
        big_codebook_bytes = DEFAULT_BIG_BLOCK_CODEBOOK_SIZE * BYTES_PER_BIG_BLOCK
        small_codebook_bytes = EFFECTIVE_UNIFIED_CODEBOOK_SIZE * BYTES_PER_BLOCK
        self.total_big_block_codebook_bytes += big_codebook_bytes
        self.total_small_block_codebook_bytes += small_codebook_bytes
        
        # 索引大小 = 总大小 - 帧类型标记 - 两个码本大小
        actual_index_size = size_bytes - 1 - big_codebook_bytes - small_codebook_bytes
        self.total_index_bytes += max(0, actual_index_size)
        
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
        overhead = 2 + zone_count * 2  # 现在只有2种块类型
        self.total_p_overhead_bytes += overhead
        
        # 详细更新统计
        self.detail_update_count += detail_updates
        big_block_updates = updates_count - detail_updates
        self.big_block_update_count += big_block_updates
        
        # 计算更新数据字节数
        detail_bytes = detail_updates * 5  # 1字节位置 + 4字节索引
        big_block_bytes = big_block_updates * 2  # 1字节位置 + 1字节索引
        self.detail_update_bytes += detail_bytes
        self.big_block_update_bytes += big_block_bytes
        
        self.strip_stats[strip_idx]['p_frames'] += 1
        self.strip_stats[strip_idx]['p_bytes'] += size_bytes
    
    def add_block_type_stats(self, big_blocks, small_blocks):
        self.big_block_count += big_blocks
        self.small_block_count += small_blocks
    
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
        
        # 数据构成统计
        print(f"\n🎨 数据构成:")
        print(f"   大块码本数据: {self.total_big_block_codebook_bytes:,} bytes ({self.total_big_block_codebook_bytes/total_bytes*100:.1f}%)")
        print(f"   小块码本数据: {self.total_small_block_codebook_bytes:,} bytes ({self.total_small_block_codebook_bytes/total_bytes*100:.1f}%)")
        print(f"   I帧索引: {self.total_index_bytes:,} bytes ({self.total_index_bytes/total_bytes*100:.1f}%)")
        
        # P帧数据构成
        p_frame_data_bytes = self.total_p_frame_bytes - self.total_p_overhead_bytes
        print(f"   P帧更新数据: {p_frame_data_bytes:,} bytes ({p_frame_data_bytes/total_bytes*100:.1f}%)")
        print(f"     - 纹理块更新: {self.detail_update_bytes:,} bytes ({self.detail_update_bytes/total_bytes*100:.1f}%)")
        print(f"     - 大块更新: {self.big_block_update_bytes:,} bytes ({self.big_block_update_bytes/total_bytes*100:.1f}%)")
        print(f"   P帧开销: {self.total_p_overhead_bytes:,} bytes ({self.total_p_overhead_bytes/total_bytes*100:.1f}%)")
        
        # 块类型统计
        print(f"\n🧩 块类型分布:")
        total_block_types = self.big_block_count + self.small_block_count
        if total_block_types > 0:
            print(f"   4x4大块: {self.big_block_count} 个 ({self.big_block_count/total_block_types*100:.1f}%)")
            print(f"   2x2纹理块: {self.small_block_count} 个 ({self.small_block_count/total_block_types*100:.1f}%)")
        
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
            print(f"   纹理块更新总数: {self.detail_update_count:,}")
            print(f"   大块更新总数: {self.big_block_update_count:,}")
        
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
        raw_size = total_frames * WIDTH * HEIGHT * 2
        compression_ratio = raw_size / total_bytes if total_bytes > 0 else 0
        print(f"\n📈 压缩效率:")
        print(f"   原始大小估算: {raw_size:,} bytes ({raw_size/1024/1024:.1f} MB)")
        print(f"   压缩比: {compression_ratio:.1f}:1")
        print(f"   压缩率: {(1-total_bytes/raw_size)*100:.1f}%")

# 全局统计对象
encoding_stats = EncodingStats()

def main():
    pa = argparse.ArgumentParser(description="Encode to GBA YUV9 with big block codebook")
    pa.add_argument("input")
    pa.add_argument("--duration", type=float, default=5.0)
    pa.add_argument("--full-duration", action="store_true")
    pa.add_argument("--fps", type=int, default=30)
    pa.add_argument("--out", default="video_data")
    pa.add_argument("--strip-count", type=int, default=DEFAULT_STRIP_COUNT)
    pa.add_argument("--i-frame-interval", type=int, default=60)
    pa.add_argument("--diff-threshold", type=float, default=2.0)
    pa.add_argument("--force-i-threshold", type=float, default=0.7)
    pa.add_argument("--variance-threshold", type=float, default=5.0)
    pa.add_argument("--distortion-threshold", type=float, default=10.0,
                   help="失真阈值，用于决定是否使用4x4大块码表（默认10.0）")
    pa.add_argument("--big-block-codebook-size", type=int, default=DEFAULT_BIG_BLOCK_CODEBOOK_SIZE)
    pa.add_argument("--small-block-codebook-size", type=int, default=EFFECTIVE_UNIFIED_CODEBOOK_SIZE)
    pa.add_argument("--kmeans-max-iter", type=int, default=200)
    pa.add_argument("--threads", type=int, default=None)
    pa.add_argument("--i-frame-weight", type=int, default=3)
    pa.add_argument("--dither", action="store_true")

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
    print(f"码本配置: 大块码表{args.big_block_codebook_size}项, 小块码表{args.small_block_codebook_size}项")
    if args.dither:
        print(f"🎨 已启用抖动算法（蛇形扫描）")
    
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
                # if args.dither:
                #     frm = apply_dither_optimized(frm)
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

    # 生成码表
    gop_codebooks = generate_gop_codebooks_with_big_blocks(
        frames, args.strip_count, args.i_frame_interval, 
        args.variance_threshold, args.diff_threshold, args.distortion_threshold,
        args.big_block_codebook_size, args.small_block_codebook_size,
        args.kmeans_max_iter, args.i_frame_weight
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
            big_block_codebook = strip_gop_data['big_block_codebook']
            small_block_codebook = strip_gop_data['small_block_codebook']
            
            # 找到当前帧的分类信息
            block_types = None
            big_block_indices = None
            for fid, bt, bbi in strip_gop_data['block_types_list']:
                if fid == frame_idx:
                    block_types = bt
                    big_block_indices = bbi
                    break
            
            force_i_frame = (frame_idx % args.i_frame_interval == 0) or frame_idx == 0
            
            if force_i_frame or prev_strips[strip_idx] is None:
                strip_data = encode_strip_i_frame_with_big_blocks(
                    current_strip, big_block_codebook, small_block_codebook, 
                    block_types, big_block_indices
                )
                is_i_frame = True
                
                # 计算码本和索引大小
                big_codebook_size = args.big_block_codebook_size * BYTES_PER_BIG_BLOCK
                small_codebook_size = args.small_block_codebook_size * BYTES_PER_BLOCK
                index_size = len(strip_data) - 1 - big_codebook_size - small_codebook_size
                
                encoding_stats.add_i_frame(
                    strip_idx, len(strip_data), 
                    is_forced=force_i_frame,
                    codebook_size=big_codebook_size + small_codebook_size,
                    index_size=max(0, index_size)
                )
            else:
                strip_data, is_i_frame, used_zones, color_updates, detail_updates = encode_strip_p_frame_with_big_blocks(
                    current_strip, prev_strips[strip_idx],
                    big_block_codebook, small_block_codebook, block_types, big_block_indices,
                    args.diff_threshold, args.force_i_threshold, args.variance_threshold, args.distortion_threshold
                )
                
                if is_i_frame:
                    big_codebook_size = args.big_block_codebook_size * BYTES_PER_BIG_BLOCK
                    small_codebook_size = args.small_block_codebook_size * BYTES_PER_BLOCK
                    index_size = len(strip_data) - 1 - big_codebook_size - small_codebook_size
                    
                    encoding_stats.add_i_frame(
                        strip_idx, len(strip_data), 
                        is_forced=False,
                        codebook_size=big_codebook_size + small_codebook_size,
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
                args.strip_count, strip_heights, args.big_block_codebook_size, args.small_block_codebook_size)
    write_source(pathlib.Path(args.out).with_suffix(".c"), all_data, frame_offsets, strip_heights)
    
    # 打印详细统计
    encoding_stats.print_summary(len(frames), len(all_data))

def write_header(path_h: pathlib.Path, frame_cnt: int, total_bytes: int, strip_count: int, 
                strip_heights: list, big_block_codebook_size: int, small_block_codebook_size: int):
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
            #define BIG_BLOCK_CODEBOOK_SIZE {big_block_codebook_size}
            #define SMALL_BLOCK_CODEBOOK_SIZE {small_block_codebook_size}
            #define EFFECTIVE_UNIFIED_CODEBOOK_SIZE {EFFECTIVE_UNIFIED_CODEBOOK_SIZE}
            
            // 帧类型定义
            #define FRAME_TYPE_I        0x00
            #define FRAME_TYPE_P        0x01
            
            // 特殊标记（删除色块标记）
            #define BIG_BLOCK_MARKER    0xFE
            
            // 块参数
            #define BLOCK_WIDTH         2
            #define BLOCK_HEIGHT        2
            #define BYTES_PER_BLOCK     7
            #define BYTES_PER_BIG_BLOCK 28

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