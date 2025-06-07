#!/usr/bin/env python3
"""
codebook.py - 向量量化码表生成和管理
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from config import CODEBOOK_SIZE, BYTES_PER_BLOCK


def generate_codebook(blocks_data: np.ndarray, codebook_size: int = CODEBOOK_SIZE, max_iter: int = 100) -> tuple:
    """
    使用K-Means聚类生成码表
    blocks_data: shape (N, 7) 的块数据数组
    返回: (codebook, effective_size) - 码表和有效码字数量
    """
    if len(blocks_data) == 0:
        return np.zeros((codebook_size, BYTES_PER_BLOCK), dtype=np.uint8), 0
    
    # 确保blocks_data是2D数组 (N, 7)
    if blocks_data.ndim > 2:
        blocks_data = blocks_data.reshape(-1, BYTES_PER_BLOCK)
    
    # 去重，统计实际不同的块数量
    # 将每个7字节块转换为一个字符串来进行去重
    blocks_as_tuples = [tuple(block) for block in blocks_data]
    unique_tuples = list(set(blocks_as_tuples))
    unique_blocks = np.array(unique_tuples, dtype=np.uint8)
    
    effective_size = min(len(unique_blocks), codebook_size)
    
    print(f"    原始块数: {len(blocks_data)}, 唯一块数: {len(unique_blocks)}, 有效码字数: {effective_size}")
    
    # 如果唯一块数小于等于码表大小，直接使用
    if len(unique_blocks) <= codebook_size:
        codebook = np.zeros((codebook_size, BYTES_PER_BLOCK), dtype=np.uint8)
        codebook[:len(unique_blocks)] = unique_blocks
        # 用最后一个块填充剩余位置，避免未初始化数据
        if len(unique_blocks) > 0:
            for i in range(len(unique_blocks), codebook_size):
                codebook[i] = unique_blocks[-1]
        return codebook, effective_size
    
    # 使用MiniBatchKMeans进行聚类，正确处理有符号数
    kmeans = MiniBatchKMeans(
        n_clusters=codebook_size, 
        random_state=42, 
        batch_size=min(1000, len(blocks_data)),
        max_iter=max_iter,
        n_init=3
    )
    # 将块数据正确转换为聚类格式（处理有符号数）
    blocks_for_clustering = convert_blocks_for_clustering(blocks_data)
    kmeans.fit(blocks_for_clustering)
    
    # 将聚类中心转换回正确的块格式
    codebook = convert_codebook_from_clustering(kmeans.cluster_centers_)
    
    return codebook, codebook_size


def quantize_blocks(blocks_data: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """
    使用码表对块进行量化（优化版本）
    返回每个块对应的码表索引
    """
    if len(blocks_data) == 0:
        return np.array([], dtype=np.uint8)
    
    # 优化：使用向量化计算替代cdist，正确处理有符号数
    # 将块数据和码表都转换为正确的聚类格式
    blocks_for_clustering = convert_blocks_for_clustering(blocks_data)
    codebook_for_clustering = convert_blocks_for_clustering(codebook)
    
    # 展开为 (N, 1, D) - (1, M, D) = (N, M, D)
    blocks_expanded = blocks_for_clustering[:, np.newaxis, :]  # (N, 1, D)
    codebook_expanded = codebook_for_clustering[np.newaxis, :, :]  # (1, M, D)
    
    # 计算平方差
    diff = blocks_expanded - codebook_expanded  # (N, M, D)
    squared_distances = np.sum(diff * diff, axis=2)  # (N, M)
    
    # 找到最近的码表索引
    indices = np.argmin(squared_distances, axis=1).astype(np.uint8)
    
    # 验证索引范围
    max_idx = indices.max() if len(indices) > 0 else 0
    if max_idx >= CODEBOOK_SIZE:
        print(f"警告: 量化索引超出范围: 最大索引={max_idx}, 码表大小={CODEBOOK_SIZE}")
        indices = np.clip(indices, 0, CODEBOOK_SIZE - 1)
    
    return indices


def convert_blocks_for_clustering(blocks_data: np.ndarray) -> np.ndarray:
    """
    将块数据转换为正确的聚类格式，处理有符号数问题
    索引0-3是Y值(uint8)，索引4-6是d_r, d_g, d_b(int8)
    """
    if len(blocks_data) == 0:
        return blocks_data.astype(np.float32)
    
    # 确保是2D数组
    if blocks_data.ndim > 2:
        blocks_data = blocks_data.reshape(-1, BYTES_PER_BLOCK)
    
    # 创建float32副本
    blocks_float = blocks_data.astype(np.float32)
    
    # 将索引4-6的值从uint8转换为int8（有符号），再转为float32
    # 这样-1会变成-1.0而不是255.0
    for i in range(4, BYTES_PER_BLOCK):
        blocks_float[:, i] = blocks_data[:, i].view(np.int8).astype(np.float32)
    
    return blocks_float


def convert_codebook_from_clustering(codebook_float: np.ndarray) -> np.ndarray:
    """
    将聚类结果转换回正确的块格式
    索引0-3是Y值(uint8)，索引4-6是d_r, d_g, d_b(int8)
    """
    codebook = np.zeros_like(codebook_float, dtype=np.uint8)
    
    # Y值（索引0-3）直接裁剪为uint8
    codebook[:, 0:4] = np.clip(codebook_float[:, 0:4].round(), 0, 255).astype(np.uint8)
    
    # d_r, d_g, d_b（索引4-6）需要裁剪为int8范围，然后转为uint8存储
    for i in range(4, BYTES_PER_BLOCK):
        # 裁剪到int8范围[-128, 127]，然后用view转为uint8存储
        clipped_values = np.clip(codebook_float[:, i].round(), -128, 127).astype(np.int8)
        codebook[:, i] = clipped_values.view(np.uint8)
    
    return codebook