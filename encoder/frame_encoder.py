#!/usr/bin/env python3
"""
frame_encoder.py - I帧和P帧编码
"""

import struct
import numpy as np
from config import (FRAME_TYPE_I, FRAME_TYPE_P, CODEBOOK_SIZE, BYTES_PER_BLOCK)
from utils import calculate_block_diff
from codebook import quantize_blocks


def encode_strip_i_frame(blocks: np.ndarray) -> bytes:
    """编码条带I帧（完整条带）"""
    data = bytearray()
    data.append(FRAME_TYPE_I)
    if blocks.size > 0:
        data.extend(blocks.flatten().tobytes())
    return bytes(data)


def encode_strip_i_frame_vq(blocks: np.ndarray, codebook: np.ndarray) -> bytes:
    """编码条带I帧（带向量量化，按4x4大块组织，批量优化）"""
    data = bytearray()
    data.append(FRAME_TYPE_I)
    
    if blocks.size > 0:
        blocks_h, blocks_w = blocks.shape[:2]
        big_blocks_h = blocks_h // 2  # 4x4大块的行数
        big_blocks_w = blocks_w // 2  # 4x4大块的列数
        
        # 存储码表大小和码表数据
        data.extend(struct.pack('<H', CODEBOOK_SIZE))
        data.extend(codebook.flatten().tobytes())
        
        # 批量量化：收集所有需要量化的块
        all_blocks_to_quantize = []
        block_positions = []
        
        # 按4x4大块的顺序组织量化索引：12/34排布
        for big_by in range(big_blocks_h):
            for big_bx in range(big_blocks_w):
                # 收集4x4大块内的4个2x2小块
                for sub_by in range(2):
                    for sub_bx in range(2):
                        by = big_by * 2 + sub_by
                        bx = big_bx * 2 + sub_bx
                        
                        if by < blocks_h and bx < blocks_w:
                            all_blocks_to_quantize.append(blocks[by, bx])
                            block_positions.append((big_by, big_bx, sub_by, sub_bx))
                        else:
                            # 如果超出边界，使用零块
                            all_blocks_to_quantize.append(np.zeros(BYTES_PER_BLOCK, dtype=np.uint8))
                            block_positions.append((big_by, big_bx, sub_by, sub_bx))
        
        # 批量量化所有块
        if all_blocks_to_quantize:
            all_blocks_array = np.array(all_blocks_to_quantize)
            all_quantized_indices = quantize_blocks(all_blocks_array, codebook)
            
            # 重新组织为4x4大块格式
            idx = 0
            for big_by in range(big_blocks_h):
                for big_bx in range(big_blocks_w):
                    # 按12/34顺序存储4个量化索引
                    for _ in range(4):
                        data.append(all_quantized_indices[idx])
                        idx += 1
    
    return bytes(data)


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


def encode_strip_differential_vq(current_blocks: np.ndarray, prev_blocks: np.ndarray, 
                                codebook: np.ndarray, diff_threshold: float, 
                                force_i_threshold: float = 0.7) -> tuple:
    """
    差分编码当前条带（使用向量量化，按4x4大块组织，批量优化）
    返回: (编码数据, 是否为I帧)
    """
    if prev_blocks is None or current_blocks.shape != prev_blocks.shape:
        return encode_strip_i_frame_vq(current_blocks, codebook), True
    
    blocks_h, blocks_w = current_blocks.shape[:2]
    total_blocks = blocks_h * blocks_w
    
    if total_blocks == 0:
        return b'', True
    
    # 批量计算所有块的差异
    current_flat = current_blocks.reshape(-1, BYTES_PER_BLOCK)
    prev_flat = prev_blocks.reshape(-1, BYTES_PER_BLOCK)
    
    # 只比较Y通道（前4个字节）
    y_current = current_flat[:, :4].astype(np.int16)
    y_prev = prev_flat[:, :4].astype(np.int16)
    y_diff = np.abs(y_current - y_prev)
    block_diffs_flat = y_diff.mean(axis=1)  # 每个块的平均Y差值
    
    # 重塑回原来的形状
    block_diffs = block_diffs_flat.reshape(blocks_h, blocks_w)
    
    # 按4x4大块组织：每个大块包含4个2x2小块
    big_blocks_h = blocks_h // 2
    big_blocks_w = blocks_w // 2
    
    # 收集需要更新的4x4大块
    updated_big_blocks = []
    blocks_to_quantize = []
    quantize_positions = []
    
    for big_by in range(big_blocks_h):
        for big_bx in range(big_blocks_w):
            # 检查当前4x4大块中的4个2x2小块是否有任何一个需要更新
            needs_update = False
            
            # 4x4大块中的4个2x2小块位置
            positions = [
                (big_by * 2, big_bx * 2),      # 左上
                (big_by * 2, big_bx * 2 + 1),  # 右上
                (big_by * 2 + 1, big_bx * 2),  # 左下
                (big_by * 2 + 1, big_bx * 2 + 1)  # 右下
            ]
            
            for by, bx in positions:
                if by < blocks_h and bx < blocks_w:
                    if block_diffs[by, bx] > diff_threshold:
                        needs_update = True
                        break
            
            # 如果4x4大块中有任何小块需要更新，则记录整个大块
            if needs_update:
                big_block_idx = big_by * big_blocks_w + big_bx
                
                # 收集要量化的块
                current_big_block_data = []
                for by, bx in positions:
                    if by < blocks_h and bx < blocks_w:
                        current_big_block_data.append(current_blocks[by, bx])
                        blocks_to_quantize.append(current_blocks[by, bx])
                        quantize_positions.append((big_block_idx, len(current_big_block_data) - 1))
                    else:
                        # 如果超出边界，使用零块
                        zero_block = np.zeros(BYTES_PER_BLOCK, dtype=np.uint8)
                        current_big_block_data.append(zero_block)
                        blocks_to_quantize.append(zero_block)
                        quantize_positions.append((big_block_idx, len(current_big_block_data) - 1))
                
                updated_big_blocks.append((big_block_idx, current_big_block_data))
    
    # 计算更新比例（基于2x2小块数量）
    total_updated_small_blocks = len(updated_big_blocks) * 4  # 每个大块包含4个小块
    update_ratio = total_updated_small_blocks / total_blocks
    
    # 如果需要更新的块太多，则使用I帧
    if update_ratio > force_i_threshold:
        return encode_strip_i_frame_vq(current_blocks, codebook), True
    
    # 批量量化所有需要更新的块
    quantized_indices_flat = []
    if blocks_to_quantize:
        blocks_array = np.array(blocks_to_quantize)
        quantized_indices_flat = quantize_blocks(blocks_array, codebook)
    
    # 重新组织量化结果
    big_block_quantized = {}
    idx = 0
    for big_block_idx, big_block_data in updated_big_blocks:
        quantized_indices = []
        for _ in range(4):  # 每个大块4个小块
            quantized_indices.append(quantized_indices_flat[idx])
            idx += 1
        big_block_quantized[big_block_idx] = quantized_indices
    
    # 否则编码为P帧
    data = bytearray()
    data.append(FRAME_TYPE_P)
    
    # 存储需要更新的4x4大块数（2字节）
    data.extend(struct.pack('<H', len(updated_big_blocks)))
    
    # 存储每个4x4大块的数据
    for big_block_idx, _ in updated_big_blocks:
        data.extend(struct.pack('<H', big_block_idx))  # 4x4大块索引
        for quant_idx in big_block_quantized[big_block_idx]:  # 4个2x2小块的量化索引
            data.append(quant_idx)
    
    return bytes(data), False