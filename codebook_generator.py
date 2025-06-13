from const_def import *
import numpy as np
from sklearn.cluster import MiniBatchKMeans

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

    