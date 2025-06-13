import numpy as np
from numba import njit
from const_def import *
from block_quantizer import quantize_4x4_blocks,quantize_blocks_unified
from codebook_generator import convert_4x4_blocks_for_clustering

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

def unpack_4x4_block_to_2x2_blocks(block_4x4: np.ndarray) -> list:
    """将4x4块拆分成4个2x2块"""
    blocks_2x2 = []
    
    for i in range(4):
        start_offset = i * BYTES_PER_2X2_BLOCK
        block = block_4x4[start_offset:start_offset + BYTES_PER_2X2_BLOCK].copy()
        blocks_2x2.append(block)
    
    return blocks_2x2

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


def classify_4x4_blocks_in_8x8_super_block(blocks_8x8: list, codebook_4x4: np.ndarray,
                                          codebook_2x2: np.ndarray, distortion_threshold: float = 10.0) -> tuple:
    """对8x8超级块内的4个4x4子块进行分类"""
    block_4x4_usage = {}  # 记录哪些4x4子块使用4x4码表
    
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
        blocks_4x4_in_super.append((quad_idx, block_4x4, blocks_2x2_in_4x4))
    
    # 对每个4x4块独立决定使用哪种码表
    for quad_idx, block_4x4, blocks_2x2_in_4x4 in blocks_4x4_in_super:
        # 尝试4x4码表
        indices_4x4, reconstructed_4x4 = quantize_4x4_blocks([block_4x4], codebook_4x4)
        if len(reconstructed_4x4) > 0:
            reconstructed_2x2_from_4x4 = unpack_4x4_block_to_2x2_blocks(reconstructed_4x4[0])
            distortion_4x4 = calculate_distortion(blocks_2x2_in_4x4, reconstructed_2x2_from_4x4)
            
            # 尝试2x2码表
            indices_2x2 = []
            reconstructed_2x2_from_2x2 = []
            for block_2x2 in blocks_2x2_in_4x4:
                idx = quantize_blocks_unified(block_2x2.reshape(1, -1), codebook_2x2)[0]
                indices_2x2.append(idx)
                reconstructed_2x2_from_2x2.append(codebook_2x2[idx])
            distortion_2x2 = calculate_distortion(blocks_2x2_in_4x4, reconstructed_2x2_from_2x2)
            
            # 选择失真更小的方案
            if distortion_4x4 <= distortion_2x2 and distortion_4x4 <= distortion_threshold:
                block_4x4_usage[quad_idx] = ('4x4', indices_4x4[0])
            else:
                block_4x4_usage[quad_idx] = ('2x2', indices_2x2)
        else:
            # 4x4量化失败，使用2x2
            indices_2x2 = []
            for block_2x2 in blocks_2x2_in_4x4:
                idx = quantize_blocks_unified(block_2x2.reshape(1, -1), codebook_2x2)[0]
                indices_2x2.append(idx)
            block_4x4_usage[quad_idx] = ('2x2', indices_2x2)
    
    return block_4x4_usage