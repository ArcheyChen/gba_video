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
EFFECTIVE_UNIFIED_CODEBOOK_SIZE = 255  # 有效码本大小（0xFF保留）
DEFAULT_4X4_CODEBOOK_SIZE = 128  # 4x4块码表大小

# 标记常量
BLOCK_4X4_MARKER = 0xFF

Y_COEFF  = np.array([0.28571429,  0.57142857,  0.14285714])
CB_COEFF = np.array([-0.14285714, -0.28571429,  0.42857143])
CR_COEFF = np.array([ 0.35714286, -0.28571429, -0.07142857])
BLOCK_W, BLOCK_H = 2, 2
BYTES_PER_2X2_BLOCK  = 7  # 4Y + d_r + d_g + d_b
BYTES_PER_4X4_BLOCK = 28  # 16Y + 4*(d_r + d_g + d_b)

# 新增常量 - 改为8x8单位
SUPER_BLOCK_SIZE = 8  # 8x8超级块
ZONE_HEIGHT_PIXELS = 16  # 每个区域的像素高度
ZONE_HEIGHT_SUPER_BLOCKS = ZONE_HEIGHT_PIXELS // SUPER_BLOCK_SIZE  # 每个区域的8x8超级块行数 (16像素 = 2行8x8超级块)

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
    
    block_array = np.zeros((blocks_h, blocks_w, BYTES_PER_2X2_BLOCK), dtype=np.uint8)
    
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

def generate_4x4_codebook(blocks_4x4: list, codebook_size: int = DEFAULT_4X4_CODEBOOK_SIZE, 
                         max_iter: int = 100) -> np.ndarray:
    """生成4x4块码表"""
    if len(blocks_4x4) == 0:
        return np.zeros((codebook_size, BYTES_PER_4X4_BLOCK), dtype=np.uint8)
    
    blocks_4x4_array = np.array(blocks_4x4)
    if len(blocks_4x4_array) <= codebook_size:
        # 数据量小于码表大小
        codebook = np.zeros((codebook_size, BYTES_PER_4X4_BLOCK), dtype=np.uint8)
        codebook[:len(blocks_4x4_array)] = blocks_4x4_array
        if len(blocks_4x4_array) > 0:
            for i in range(len(blocks_4x4_array), codebook_size):
                codebook[i] = blocks_4x4_array[-1]
        return codebook
    
    # 使用K-Means聚类
    blocks_4x4_for_clustering = convert_4x4_blocks_for_clustering(blocks_4x4_array)
    kmeans = MiniBatchKMeans(
        n_clusters=codebook_size,
        random_state=42,
        batch_size=min(1000, len(blocks_4x4_array)),
        max_iter=max_iter,
        n_init=3
    )
    kmeans.fit(blocks_4x4_for_clustering)
    codebook = convert_4x4_codebook_from_clustering(kmeans.cluster_centers_)
    
    return codebook

def convert_4x4_blocks_for_clustering(blocks_4x4: np.ndarray) -> np.ndarray:
    """将4x4块转换为聚类格式"""
    if len(blocks_4x4) == 0:
        return blocks_4x4.astype(np.float32)
    
    if blocks_4x4.ndim > 2:
        blocks_4x4 = blocks_4x4.reshape(-1, BYTES_PER_4X4_BLOCK)
    
    blocks_4x4_float = blocks_4x4.astype(np.float32)
    
    # 色度分量需要转换为有符号数
    for i in range(16, BYTES_PER_4X4_BLOCK):
        blocks_4x4_float[:, i] = blocks_4x4[:, i].view(np.int8).astype(np.float32)
    
    return blocks_4x4_float

def convert_4x4_codebook_from_clustering(codebook_float: np.ndarray) -> np.ndarray:
    """将聚类结果转换回4x4块格式"""
    codebook = np.zeros_like(codebook_float, dtype=np.uint8)
    
    # Y分量
    codebook[:, 0:16] = np.clip(codebook_float[:, 0:16].round(), 0, 255).astype(np.uint8)
    
    # 色度分量
    for i in range(16, BYTES_PER_4X4_BLOCK):
        clipped_values = np.clip(codebook_float[:, i].round(), -128, 127).astype(np.int8)
        codebook[:, i] = clipped_values.view(np.uint8)
    
    return codebook

def quantize_4x4_blocks(blocks_4x4: list, codebook_4x4: np.ndarray) -> tuple:
    """量化4x4块，返回索引和重建的块"""
    if len(blocks_4x4) == 0:
        return np.array([], dtype=np.uint8), []
    
    blocks_4x4_array = np.array(blocks_4x4)
    blocks_4x4_for_clustering = convert_4x4_blocks_for_clustering(blocks_4x4_array)
    codebook_4x4_for_clustering = convert_4x4_blocks_for_clustering(codebook_4x4)
    
    # 计算距离和找到最近的码字
    distances = cdist(blocks_4x4_for_clustering, codebook_4x4_for_clustering, metric='euclidean')
    indices = np.argmin(distances, axis=1).astype(np.uint8)
    
    # 重建块
    reconstructed_4x4_blocks = [codebook_4x4[idx] for idx in indices]
    
    return indices, reconstructed_4x4_blocks

def classify_8x8_super_blocks_with_4x4_codebook(blocks: np.ndarray, codebook_4x4: np.ndarray,
                                              variance_threshold: float = 5.0, 
                                              distortion_threshold: float = 10.0) -> tuple:
    """使用4x4块码表对8x8超级块进行分类"""
    blocks_h, blocks_w = blocks.shape[:2]
    super_blocks_h = blocks_h // 4  # 8x8超级块 = 4个2x2块的行数
    super_blocks_w = blocks_w // 4  # 8x8超级块 = 4个2x2块的列数
    
    block_4x4_indices = {}  # 使用4x4块码表的超级块
    blocks_2x2 = []         # 需要用2x2块码表的块
    block_types = {}        # 记录每个8x8超级块的类型
    
    for super_by in range(super_blocks_h):
        for super_bx in range(super_blocks_w):
            # 收集8x8超级块内的16个2x2块 - 按行优先顺序
            blocks_8x8 = []
            for sub_by in range(4):  # 4行2x2块
                for sub_bx in range(4):  # 4列2x2块
                    by = super_by * 4 + sub_by
                    bx = super_bx * 4 + sub_bx
                    if by < blocks_h and bx < blocks_w:
                        blocks_8x8.append(blocks[by, bx])
                    else:
                        blocks_8x8.append(np.zeros(BYTES_PER_2X2_BLOCK, dtype=np.uint8))
            
            # 将16个2x2块重组为4个4x4块
            blocks_4x4_in_super = []
            for quad_idx in range(4):  # 4个4x4块
                quad_by = quad_idx // 2
                quad_bx = quad_idx % 2
                blocks_2x2_in_4x4 = []
                for sub_by in range(2):
                    for sub_bx in range(2):
                        block_idx = (quad_by * 2 + sub_by) * 4 + (quad_bx * 2 + sub_bx)
                        blocks_2x2_in_4x4.append(blocks_8x8[block_idx])
                block_4x4 = pack_4x4_block_from_2x2_blocks(blocks_2x2_in_4x4)
                blocks_4x4_in_super.append(block_4x4)
            
            # 尝试用4x4块码表
            indices, reconstructed = quantize_4x4_blocks(blocks_4x4_in_super, codebook_4x4)
            
            if len(reconstructed) > 0:
                # 计算失真
                reconstructed_2x2_blocks = []
                for block_4x4 in reconstructed:
                    reconstructed_2x2_blocks.extend(unpack_4x4_block_to_2x2_blocks(block_4x4))
                distortion = calculate_distortion(blocks_8x8, reconstructed_2x2_blocks)
                
                if distortion <= distortion_threshold:
                    # 失真可接受，使用4x4块码表
                    block_4x4_indices[(super_by, super_bx)] = indices
                    block_types[(super_by, super_bx)] = '4x4_blocks'
                else:
                    # 失真太大，使用2x2块码表
                    blocks_2x2.extend(blocks_8x8)
                    block_types[(super_by, super_bx)] = '2x2_blocks'
            else:
                # 量化失败，使用2x2块码表
                blocks_2x2.extend(blocks_8x8)
                block_types[(super_by, super_bx)] = '2x2_blocks'
    
    return block_4x4_indices, blocks_2x2, block_types

def encode_strip_i_frame_with_4x4_blocks(blocks: np.ndarray, codebook_4x4: np.ndarray,
                                        codebook_2x2: np.ndarray, block_types: dict,
                                        block_4x4_indices: dict) -> bytes:
    """编码I帧条带"""
    data = bytearray()
    data.append(FRAME_TYPE_I)
    
    if blocks.size > 0:
        blocks_h, blocks_w = blocks.shape[:2]
        super_blocks_h = blocks_h // 4
        super_blocks_w = blocks_w // 4
        
        # 存储4x4块码表
        data.extend(codebook_4x4.flatten().tobytes())
        
        # 存储2x2块码表
        data.extend(codebook_2x2.flatten().tobytes())
        
        # 按8x8超级块的顺序编码
        for super_by in range(super_blocks_h):
            for super_bx in range(super_blocks_w):
                if (super_by, super_bx) in block_types:
                    block_type = block_types[(super_by, super_bx)]
                    
                    if block_type == '4x4_blocks':
                        # 4x4块：0xFF + 4个4x4块码表索引
                        data.append(BLOCK_4X4_MARKER)
                        indices_4x4 = block_4x4_indices[(super_by, super_bx)]
                        for idx in indices_4x4:
                            data.append(idx)
                        
                    else:  # 2x2_blocks
                        # 纹理块：16个2x2块码表索引，按行优先顺序
                        for sub_by in range(4):
                            for sub_bx in range(4):
                                by = super_by * 4 + sub_by
                                bx = super_bx * 4 + sub_bx
                                if by < blocks_h and bx < blocks_w:
                                    block = blocks[by, bx]
                                    idx_2x2 = quantize_blocks_unified(block.reshape(1, -1), codebook_2x2)[0]
                                    data.append(idx_2x2)
                                else:
                                    data.append(0)
    
    return bytes(data)

def generate_gop_codebooks_with_4x4_blocks(frames: list, strip_count: int, i_frame_interval: int,
                                         variance_threshold: float, diff_threshold: float,
                                         distortion_threshold: float = 10.0,
                                         codebook_4x4_size: int = DEFAULT_4X4_CODEBOOK_SIZE,
                                         codebook_2x2_size: int = EFFECTIVE_UNIFIED_CODEBOOK_SIZE,
                                         kmeans_max_iter: int = 100, i_frame_weight: int = 3) -> dict:
    """为每个GOP生成4x4块码表和2x2块码表"""
    print("正在为每个GOP生成4x4块码表和2x2块码表...")
    
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
            all_4x4_blocks = []
            all_2x2_blocks = []
            block_types_list = []
            
            # 处理GOP中的每一帧
            prev_strip_blocks = None
            
            for frame_idx in range(gop_start, gop_end):
                strip_blocks = frames[frame_idx][strip_idx]
                if strip_blocks.size == 0:
                    continue
                
                # 确定需要处理的8x8超级块
                is_i_frame = (frame_idx == gop_start)
                
                if is_i_frame:
                    blocks_h, blocks_w = strip_blocks.shape[:2]
                    super_blocks_h = blocks_h // 4
                    super_blocks_w = blocks_w // 4
                    updated_super_blocks = {(super_by, super_bx) for super_by in range(super_blocks_h) for super_bx in range(super_blocks_w)}
                else:
                    updated_super_blocks = identify_updated_8x8_super_blocks(strip_blocks, prev_strip_blocks, diff_threshold)
                
                # 从有效8x8超级块中提取数据用于训练码表
                for super_by, super_bx in updated_super_blocks:
                    blocks_8x8 = []
                    for sub_by in range(4):
                        for sub_bx in range(4):
                            by = super_by * 4 + sub_by
                            bx = super_bx * 4 + sub_bx
                            if by < strip_blocks.shape[0] and bx < strip_blocks.shape[1]:
                                blocks_8x8.append(strip_blocks[by, bx])
                            else:
                                blocks_8x8.append(np.zeros(BYTES_PER_2X2_BLOCK, dtype=np.uint8))
                    
                    # 将16个2x2块重组为4个4x4块
                    blocks_4x4_in_super = []
                    for quad_idx in range(4):
                        quad_by = quad_idx // 2
                        quad_bx = quad_idx % 2
                        blocks_2x2_in_4x4 = []
                        for sub_by in range(2):
                            for sub_bx in range(2):
                                block_idx = (quad_by * 2 + sub_by) * 4 + (quad_bx * 2 + sub_bx)
                                blocks_2x2_in_4x4.append(blocks_8x8[block_idx])
                        block_4x4 = pack_4x4_block_from_2x2_blocks(blocks_2x2_in_4x4)
                        blocks_4x4_in_super.append(block_4x4)
                    
                    # 添加到训练数据
                    if is_i_frame:
                        all_4x4_blocks.extend(blocks_4x4_in_super * i_frame_weight)
                        all_2x2_blocks.extend(blocks_8x8 * i_frame_weight)
                    else:
                        all_4x4_blocks.extend(blocks_4x4_in_super)
                        all_2x2_blocks.extend(blocks_8x8)
                
                prev_strip_blocks = strip_blocks.copy()
            
            # 生成码表
            codebook_4x4 = generate_4x4_codebook(all_4x4_blocks, codebook_4x4_size, kmeans_max_iter)
            codebook_2x2 = generate_unified_codebook_simplified(
                all_2x2_blocks, codebook_2x2_size, kmeans_max_iter)
            
            # 为每一帧生成分类信息
            for frame_idx in range(gop_start, gop_end):
                strip_blocks = frames[frame_idx][strip_idx]
                if strip_blocks.size == 0:
                    continue
                
                block_4x4_indices, _, block_types = classify_8x8_super_blocks_with_4x4_codebook(
                    strip_blocks, codebook_4x4, variance_threshold, distortion_threshold)
                block_types_list.append((frame_idx, block_types, block_4x4_indices))
            
            gop_codebooks[gop_start].append({
                'codebook_4x4': codebook_4x4,
                'codebook_2x2': codebook_2x2,
                'block_types_list': block_types_list,
                'distortion_threshold': distortion_threshold
            })
            
            print(f"    条带{strip_idx}: 4x4块{len(all_4x4_blocks)}个, 2x2块{len(all_2x2_blocks)}个")
    
    return gop_codebooks

def pack_4x4_block_from_2x2_blocks(blocks_2x2: list) -> np.ndarray:
    """将4个2x2块组合成一个4x4块"""
    block_4x4 = np.zeros(BYTES_PER_4X4_BLOCK, dtype=np.uint8)
    
    # 直接按行优先顺序存储4个YUV_Struct
    # blocks_2x2的顺序应该是：[左上, 右上, 左下, 右下]
    for i, block in enumerate(blocks_2x2):
        if len(block) >= BYTES_PER_2X2_BLOCK:
            start_offset = i * BYTES_PER_2X2_BLOCK
            block_4x4[start_offset:start_offset + BYTES_PER_2X2_BLOCK] = block[:BYTES_PER_2X2_BLOCK]
    
    return block_4x4

def unpack_4x4_block_to_2x2_blocks(block_4x4: np.ndarray) -> list:
    """将4x4块拆分成4个2x2块"""
    blocks_2x2 = []
    
    for i in range(4):
        start_offset = i * BYTES_PER_2X2_BLOCK
        block = block_4x4[start_offset:start_offset + BYTES_PER_2X2_BLOCK].copy()
        blocks_2x2.append(block)
    
    return blocks_2x2

def identify_updated_8x8_super_blocks(current_blocks: np.ndarray, prev_blocks: np.ndarray,
                                    diff_threshold: float) -> set:
    """识别需要更新的8x8超级块位置"""
    if prev_blocks is None or current_blocks.shape != prev_blocks.shape:
        # 如果没有前一帧，所有超级块都需要更新
        blocks_h, blocks_w = current_blocks.shape[:2]
        super_blocks_h = blocks_h // 4
        super_blocks_w = blocks_w // 4
        return {(super_by, super_bx) for super_by in range(super_blocks_h) for super_bx in range(super_blocks_w)}
    
    blocks_h, blocks_w = current_blocks.shape[:2]
    
    # 使用Numba加速的块差异计算
    current_flat = current_blocks.reshape(-1, BYTES_PER_2X2_BLOCK)
    prev_flat = prev_blocks.reshape(-1, BYTES_PER_2X2_BLOCK)
    block_diffs = compute_block_differences_numba(current_flat, prev_flat, blocks_h, blocks_w)
    
    # 使用Numba加速的更新块识别
    updated_list = identify_updated_8x8_super_blocks_numba(block_diffs, diff_threshold, blocks_h, blocks_w)
    
    return set(updated_list)

@njit
def identify_updated_8x8_super_blocks_numba(block_diffs, diff_threshold, blocks_h, blocks_w):
    """Numba加速的8x8超级块更新识别"""
    super_blocks_h = blocks_h // 4
    super_blocks_w = blocks_w // 4
    updated_positions = []
    
    for super_by in range(super_blocks_h):
        for super_bx in range(super_blocks_w):
            needs_update = False
            
            # 检查16个2x2子块的位置
            for sub_by in range(4):
                for sub_bx in range(4):
                    by = super_by * 4 + sub_by
                    bx = super_bx * 4 + sub_bx
                    
                    if by < blocks_h and bx < blocks_w:
                        if block_diffs[by, bx] > diff_threshold:
                            needs_update = True
                            break
                if needs_update:
                    break
            
            if needs_update:
                updated_positions.append((super_by, super_bx))
    
    return updated_positions

def encode_strip_p_frame_with_4x4_blocks(current_blocks: np.ndarray, prev_blocks: np.ndarray,
                                        codebook_4x4: np.ndarray, codebook_2x2: np.ndarray,
                                        block_types: dict, block_4x4_indices: dict,
                                        diff_threshold: float, force_i_threshold: float = 0.7,
                                        variance_threshold: float = 5.0, distortion_threshold: float = 10.0) -> tuple:
    """编码P帧条带"""
    if prev_blocks is None or current_blocks.shape != prev_blocks.shape:
        i_frame_data = encode_strip_i_frame_with_4x4_blocks(
            current_blocks, codebook_4x4, codebook_2x2, block_types, block_4x4_indices)
        return i_frame_data, True, 0, 0, 0
    
    blocks_h, blocks_w = current_blocks.shape[:2]
    total_blocks = blocks_h * blocks_w
    
    if total_blocks == 0:
        return b'', True, 0, 0, 0
    
    # 识别需要更新的8x8超级块
    updated_super_blocks = identify_updated_8x8_super_blocks(current_blocks, prev_blocks, diff_threshold)
    
    super_blocks_h = blocks_h // 4
    super_blocks_w = blocks_w // 4
    total_super_blocks = super_blocks_h * super_blocks_w
    
    # 判断是否需要I帧
    update_ratio = len(updated_super_blocks) / total_super_blocks if total_super_blocks > 0 else 0
    if update_ratio > force_i_threshold:
        i_frame_data = encode_strip_i_frame_with_4x4_blocks(
            current_blocks, codebook_4x4, codebook_2x2, block_types, block_4x4_indices)
        return i_frame_data, True, 0, 0, 0
    
    # 计算区域数量 - 基于8x8超级块
    zones_count = (super_blocks_h + ZONE_HEIGHT_SUPER_BLOCKS - 1) // ZONE_HEIGHT_SUPER_BLOCKS
    
    # 按区域组织更新
    zone_4x4_updates = [[] for _ in range(zones_count)]
    zone_2x2_updates = [[] for _ in range(zones_count)]
    
    for super_by, super_bx in updated_super_blocks:
        # 计算属于哪个区域
        zone_idx = min(super_by // ZONE_HEIGHT_SUPER_BLOCKS, zones_count - 1)
        zone_relative_by = super_by % ZONE_HEIGHT_SUPER_BLOCKS
        zone_relative_idx = zone_relative_by * super_blocks_w + super_bx
        
        if (super_by, super_bx) in block_types:
            block_type = block_types[(super_by, super_bx)]
            
            if block_type == '4x4_blocks':
                # 4x4块更新
                indices_4x4 = block_4x4_indices[(super_by, super_bx)]
                zone_4x4_updates[zone_idx].append((zone_relative_idx, indices_4x4))
                
            else:  # 2x2_blocks
                # 2x2块更新
                indices = []
                for sub_by in range(4):
                    for sub_bx in range(4):
                        by = super_by * 4 + sub_by
                        bx = super_bx * 4 + sub_bx
                        if by < blocks_h and bx < blocks_w:
                            block = current_blocks[by, bx]
                            idx_2x2 = quantize_blocks_unified(block.reshape(1, -1), codebook_2x2)[0]
                            indices.append(idx_2x2)
                        else:
                            indices.append(0)
                zone_2x2_updates[zone_idx].append((zone_relative_idx, indices))
    
    # 编码P帧
    data = bytearray()
    data.append(FRAME_TYPE_P)
    
    # 统计使用的区域数量
    used_zones = 0
    total_4x4_updates = 0
    total_2x2_updates = 0
    
    # 生成区域bitmap
    zone_bitmap = 0
    for zone_idx in range(zones_count):
        if zone_4x4_updates[zone_idx] or zone_2x2_updates[zone_idx]:
            zone_bitmap |= (1 << zone_idx)
            used_zones += 1
            total_4x4_updates += len(zone_4x4_updates[zone_idx])
            total_2x2_updates += len(zone_2x2_updates[zone_idx])
    
    data.extend(struct.pack('<H', zone_bitmap))
    
    # 按区域编码更新（现在只有2种类型）
    for zone_idx in range(zones_count):
        if zone_bitmap & (1 << zone_idx):
            updates_2x2 = zone_2x2_updates[zone_idx]
            updates_4x4 = zone_4x4_updates[zone_idx]
            
            data.append(len(updates_2x2))
            data.append(len(updates_4x4))
            
            # 存储纹理块更新
            for relative_idx, indices in updates_2x2:
                data.append(relative_idx)
                for idx in indices:
                    data.append(idx)
            
            # 存储4x4块更新
            for relative_idx, indices_4x4 in updates_4x4:
                data.append(relative_idx)
                for idx in indices_4x4:
                    data.append(idx)
    
    total_updates = total_4x4_updates + total_2x2_updates
    return bytes(data), False, used_zones, total_4x4_updates, total_2x2_updates

def quantize_blocks_unified(blocks_data: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """使用统一码表对块进行量化（避免产生0xFE）"""
    if len(blocks_data) == 0:
        return np.array([], dtype=np.uint8)
    
    # 只使用前若干项进行量化，因为最后几项用于特殊标记
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
            # Y分量（前4个字节）使用2倍权重，计算SAD
            for k in range(4):
                diff = blocks_for_clustering[i, k] - codebook_for_clustering[j, k]
                dist += 2.0 * abs(diff)
            
            # 色度分量（后3个字节）使用1倍权重，计算SAD
            # 注意：这里的数据已经在convert_blocks_for_clustering中转换为有符号数
            for k in range(4, BYTES_PER_2X2_BLOCK):
                diff = blocks_for_clustering[i, k] - codebook_for_clustering[j, k]
                dist += abs(diff)
            
            if dist < min_dist:
                min_dist = dist
                best_idx = j
        
        indices[i] = best_idx
    
    return indices

def main():
    pa = argparse.ArgumentParser(description="Encode to GBA YUV9 with 4x4 block codebook")
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
                   help="失真阈值，用于决定是否使用4x4块码表（默认10.0）")
    pa.add_argument("--codebook-4x4-size", type=int, default=DEFAULT_4X4_CODEBOOK_SIZE)
    pa.add_argument("--codebook-2x2-size", type=int, default=EFFECTIVE_UNIFIED_CODEBOOK_SIZE)
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
    print(f"码本配置: 4x4块码表{args.codebook_4x4_size}项, 2x2块码表{args.codebook_2x2_size}项")
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
    gop_codebooks = generate_gop_codebooks_with_4x4_blocks(
        frames, args.strip_count, args.i_frame_interval, 
        args.variance_threshold, args.diff_threshold, args.distortion_threshold,
        args.codebook_4x4_size, args.codebook_2x2_size,
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
            codebook_4x4 = strip_gop_data['codebook_4x4']
            codebook_2x2 = strip_gop_data['codebook_2x2']
            
            # 找到当前帧的分类信息
            block_types = None
            block_4x4_indices = None
            for fid, bt, bbi in strip_gop_data['block_types_list']:
                if fid == frame_idx:
                    block_types = bt
                    block_4x4_indices = bbi
                    break
            
            force_i_frame = (frame_idx % args.i_frame_interval == 0) or frame_idx == 0
            
            if force_i_frame or prev_strips[strip_idx] is None:
                strip_data = encode_strip_i_frame_with_4x4_blocks(
                    current_strip, codebook_4x4, codebook_2x2, 
                    block_types, block_4x4_indices
                )
                is_i_frame = True
                
                # 计算码本和索引大小
                codebook_4x4_size = args.codebook_4x4_size * BYTES_PER_4X4_BLOCK
                codebook_2x2_size = args.codebook_2x2_size * BYTES_PER_2X2_BLOCK
                index_size = len(strip_data) - 1 - codebook_4x4_size - codebook_2x2_size
                
                encoding_stats.add_i_frame(
                    strip_idx, len(strip_data), 
                    is_forced=force_i_frame,
                    codebook_size=codebook_4x4_size + codebook_2x2_size,
                    index_size=max(0, index_size)
                )
            else:
                strip_data, is_i_frame, used_zones, updates_4x4, updates_2x2 = encode_strip_p_frame_with_4x4_blocks(
                    current_strip, prev_strips[strip_idx],
                    codebook_4x4, codebook_2x2, block_types, block_4x4_indices,
                    args.diff_threshold, args.force_i_threshold, args.variance_threshold, args.distortion_threshold
                )
                
                if is_i_frame:
                    codebook_4x4_size = args.codebook_4x4_size * BYTES_PER_4X4_BLOCK
                    codebook_2x2_size = args.codebook_2x2_size * BYTES_PER_2X2_BLOCK
                    index_size = len(strip_data) - 1 - codebook_4x4_size - codebook_2x2_size
                    
                    encoding_stats.add_i_frame(
                        strip_idx, len(strip_data), 
                        is_forced=False,
                        codebook_size=codebook_4x4_size + codebook_2x2_size,
                        index_size=max(0, index_size)
                    )
                else:
                    total_updates = updates_4x4 + updates_2x2
                    
                    encoding_stats.add_p_frame(
                        strip_idx, len(strip_data), total_updates, used_zones,
                        updates_4x4, updates_2x2
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
                args.strip_count, strip_heights, args.codebook_4x4_size, args.codebook_2x2_size)
    write_source(pathlib.Path(args.out).with_suffix(".c"), all_data, frame_offsets, strip_heights)
    
    # 打印详细统计
    encoding_stats.print_summary(len(frames), len(all_data))

def write_header(path_h: pathlib.Path, frame_cnt: int, total_bytes: int, strip_count: int, 
                strip_heights: list, codebook_4x4_size: int, codebook_2x2_size: int):
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
            #define CODEBOOK_4X4_SIZE {codebook_4x4_size}
            #define CODEBOOK_2X2_SIZE {codebook_2x2_size}
            #define EFFECTIVE_UNIFIED_CODEBOOK_SIZE {EFFECTIVE_UNIFIED_CODEBOOK_SIZE}

            #define BLOCK_4X4_MARKER {BLOCK_4X4_MARKER}
            
            // 帧类型定义
            #define FRAME_TYPE_I        0x00
            #define FRAME_TYPE_P        0x01
            
            // 块参数
            #define BLOCK_WIDTH         2
            #define BLOCK_HEIGHT        2
            #define BYTES_PER_2X2_BLOCK     7
            #define BYTES_PER_4X4_BLOCK 28
            #define SUPER_BLOCK_SIZE    8

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

def generate_codebook(blocks_data: np.ndarray, codebook_size: int, max_iter: int = 100) -> tuple:
    """使用K-Means聚类生成码表"""
    if len(blocks_data) == 0:
        return np.zeros((codebook_size, BYTES_PER_2X2_BLOCK), dtype=np.uint8), 0
    
    if blocks_data.ndim > 2:
        blocks_data = blocks_data.reshape(-1, BYTES_PER_2X2_BLOCK)
    
    # 移除去重操作，直接使用原始数据进行聚类
    # 这样K-Means可以基于数据的真实分布（包括频次）进行更好的聚类
    effective_size = min(len(blocks_data), codebook_size)
    
    if len(blocks_data) <= codebook_size:
        # 如果数据量小于码本大小，需要去重避免重复
        blocks_as_tuples = [tuple(block) for block in blocks_data]
        unique_tuples = list(set(blocks_as_tuples))
        unique_blocks = np.array(unique_tuples, dtype=np.uint8)
        
        codebook = np.zeros((codebook_size, BYTES_PER_2X2_BLOCK), dtype=np.uint8)
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

def convert_blocks_for_clustering(blocks_data: np.ndarray) -> np.ndarray:
    """将块数据转换为正确的聚类格式"""
    if len(blocks_data) == 0:
        return blocks_data.astype(np.float32)
    
    if blocks_data.ndim > 2:
        blocks_data = blocks_data.reshape(-1, BYTES_PER_2X2_BLOCK)
    
    blocks_float = blocks_data.astype(np.float32)
    
    for i in range(4, BYTES_PER_2X2_BLOCK):
        blocks_float[:, i] = blocks_data[:, i].view(np.int8).astype(np.float32)
    
    return blocks_float

def convert_codebook_from_clustering(codebook_float: np.ndarray) -> np.ndarray:
    """将聚类结果转换回正确的块格式"""
    codebook = np.zeros_like(codebook_float, dtype=np.uint8)
    
    codebook[:, 0:4] = np.clip(codebook_float[:, 0:4].round(), 0, 255).astype(np.uint8)
    
    for i in range(4, BYTES_PER_2X2_BLOCK):
        clipped_values = np.clip(codebook_float[:, i].round(), -128, 127).astype(np.int8)
        codebook[:, i] = clipped_values.view(np.uint8)
    
    return codebook

def calculate_distortion_sad(original_blocks: list, reconstructed_blocks: list) -> float:
    """计算失真度量 - SAD (Sum of Absolute Differences)"""
    if len(original_blocks) != len(reconstructed_blocks):
        return float('inf')
    
    total_sad = 0.0
    for orig, recon in zip(original_blocks, reconstructed_blocks):
        # Y分量的SAD（需要乘2还原）
        y_orig = orig[:4].astype(np.float32) * 2.0  # 还原Y分量
        y_recon = recon[:4].astype(np.float32) * 2.0  # 还原Y分量
        y_sad = np.sum(np.abs(y_orig - y_recon))
        
        # CrCb分量的SAD（有符号数转换）
        chroma_orig = orig[4:7].view(np.int8).astype(np.float32)  # d_r, d_g, d_b
        chroma_recon = recon[4:7].view(np.int8).astype(np.float32)
        chroma_sad = np.sum(np.abs(chroma_orig - chroma_recon))
        
        # 可以调整权重，这里Y和色度等权重
        total_sad += y_sad + chroma_sad
    
    return total_sad / len(original_blocks)  # 平均SAD

# 默认使用SAD
calculate_distortion = calculate_distortion_sad

def generate_unified_codebook_simplified(small_blocks: list, 
                                       codebook_size: int = EFFECTIVE_UNIFIED_CODEBOOK_SIZE,
                                       kmeans_max_iter: int = 100) -> np.ndarray:
    """生成2x2小块的统一码表（254项，避免0xFE）"""
    if small_blocks:
        blocks_array = np.array(small_blocks)
        codebook, _ = generate_codebook(blocks_array, codebook_size, kmeans_max_iter)
        
        # 创建254项码表
        full_codebook = np.zeros((codebook_size, BYTES_PER_2X2_BLOCK), dtype=np.uint8)
        actual_size = min(len(codebook), codebook_size)
        full_codebook[:actual_size] = codebook[:actual_size]
        
        # 填充剩余项
        if actual_size > 0:
            for i in range(actual_size, codebook_size):
                full_codebook[i] = full_codebook[actual_size - 1]
    else:
        full_codebook = np.zeros((codebook_size, BYTES_PER_2X2_BLOCK), dtype=np.uint8)
    
    return full_codebook

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
        self.total_4x4_codebook_bytes = 0
        self.total_2x2_codebook_bytes = 0
        self.total_index_bytes = 0
        self.total_p_overhead_bytes = 0
        
        # 块类型统计 - 修复
        self.block_4x4_count = 0
        self.block_2x2_count = 0
        
        # P帧块更新统计 - 新增详细统计
        self.p_frame_updates = []
        self.zone_usage = defaultdict(int)
        self.detail_update_count = 0
        self.block_4x4_update_count = 0
        self.block_2x2_update_count = 0
        self.detail_update_bytes = 0  # 纹理块更新字节数
        self.block_4x4_update_bytes = 0  # 大块更新字节数
        
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
        
        # 修复码本统计 - 分别计算4x4和2x2码本
        codebook_4x4_bytes = DEFAULT_4X4_CODEBOOK_SIZE * BYTES_PER_4X4_BLOCK
        codebook_2x2_bytes = EFFECTIVE_UNIFIED_CODEBOOK_SIZE * BYTES_PER_2X2_BLOCK
        self.total_4x4_codebook_bytes += codebook_4x4_bytes
        self.total_2x2_codebook_bytes += codebook_2x2_bytes
        
        # 索引大小 = 总大小 - 帧类型标记 - 两个码本大小
        actual_index_size = size_bytes - 1 - codebook_4x4_bytes - codebook_2x2_bytes
        self.total_index_bytes += max(0, actual_index_size)
        
        self.strip_stats[strip_idx]['i_frames'] += 1
        self.strip_stats[strip_idx]['i_bytes'] += size_bytes
    
    def add_p_frame(self, strip_idx, size_bytes, updates_count, zone_count, 
               updates_4x4=0, updates_2x2=0):  # 修改参数名和顺序
        self.total_frames_processed += 1
        self.total_p_frames += 1
        self.total_p_frame_bytes += size_bytes
        self.p_frame_updates.append(updates_count)
        self.zone_usage[zone_count] += 1
        
        # P帧开销：帧类型(1) + bitmap(2) + 每个区域的计数(2*zones)
        overhead = 3 + zone_count * 2  # 现在只有2种块类型
        self.total_p_overhead_bytes += overhead
        
        # 详细更新统计
        self.detail_update_count += updates_2x2
        self.block_4x4_update_count += updates_4x4  # 直接使用传入的值
        
        # 计算更新数据字节数
        detail_bytes = updates_2x2 * 17  # 1字节位置 + 16字节索引
        block_4x4_bytes = updates_4x4 * 5  # 1字节位置 + 4字节索引
        self.detail_update_bytes += detail_bytes
        self.block_4x4_update_bytes += block_4x4_bytes
        
        self.strip_stats[strip_idx]['p_frames'] += 1
        self.strip_stats[strip_idx]['p_bytes'] += size_bytes
    
    def add_block_type_stats(self, block_4x4s, block_2x2s):
        self.block_4x4_count += block_4x4s
        self.block_2x2_count += block_2x2s
    
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
        print(f"   4x4块码本数据: {self.total_4x4_codebook_bytes:,} bytes ({self.total_4x4_codebook_bytes/total_bytes*100:.1f}%)")
        print(f"   2x2块码本数据: {self.total_2x2_codebook_bytes:,} bytes ({self.total_2x2_codebook_bytes/total_bytes*100:.1f}%)")
        print(f"   I帧索引: {self.total_index_bytes:,} bytes ({self.total_index_bytes/total_bytes*100:.1f}%)")
        
        # P帧数据构成
        p_frame_data_bytes = self.total_p_frame_bytes - self.total_p_overhead_bytes
        print(f"   P帧更新数据: {p_frame_data_bytes:,} bytes ({p_frame_data_bytes/total_bytes*100:.1f}%)")
        print(f"     - 2x2块更新: {self.detail_update_bytes:,} bytes ({self.detail_update_bytes/total_bytes*100:.1f}%)")
        print(f"     - 4x4块更新: {self.block_4x4_update_bytes:,} bytes ({self.block_4x4_update_bytes/total_bytes*100:.1f}%)")
        print(f"   P帧开销: {self.total_p_overhead_bytes:,} bytes ({self.total_p_overhead_bytes/total_bytes*100:.1f}%)")
        
        # 块类型统计
        print(f"\n🧩 块类型分布:")
        total_block_types = self.block_4x4_count + self.block_2x2_count
        if total_block_types > 0:
            print(f"   4x4块: {self.block_4x4_count} 个 ({self.block_4x4_count/total_block_types*100:.1f}%)")
            print(f"   2x2块: {self.block_2x2_count} 个 ({self.block_2x2_count/total_block_types*100:.1f}%)")
        
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
            print(f"   2x2块更新总数: {self.detail_update_count:,}")
            print(f"   4x4块更新总数: {self.block_4x4_update_count:,}")
        
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

if __name__ == "__main__":
    main()