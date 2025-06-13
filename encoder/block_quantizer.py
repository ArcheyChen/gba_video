import numpy as np
from codebook_generator import convert_4x4_blocks_for_clustering

from scipy.spatial.distance import cdist
from numba import njit
from const_def import *

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


def quantize_blocks_unified(blocks_data: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """使用统一码表对块进行量化（避免产生0xFE和0xFF）"""
    if len(blocks_data) == 0:
        return np.array([], dtype=np.uint8)
    
    # 只使用前253项进行量化，因为0xFE和0xFF用于特殊标记
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