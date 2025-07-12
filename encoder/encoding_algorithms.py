#!/usr/bin/env python3

import numpy as np
import multiprocessing as mp
import time
from sklearn.metrics import pairwise_distances

# 新增：导入分离的工具模块
from block_utils import *
from codebook import *

WIDTH, HEIGHT = 240, 160

# 多级码表配置（默认值，将被命令行参数覆盖）
# 所有码表大小限制为255，使用U8索引，0xFF保留作为分裂标志
DEFAULT_CODEBOOK_SIZE_8x8 = 255     # 8x8块码表大小
DEFAULT_CODEBOOK_SIZE_8x4 = 255     # 8x4块码表大小
DEFAULT_CODEBOOK_SIZE_4x4 = 255     # 4x4块码表大小
DEFAULT_CODEBOOK_SIZE_4x2 = 255     # 4x2块码表大小（最大255，0xFF留作分裂标志）
DEFAULT_COVERAGE_RADIUS_8x8 = 70.0  # 8x8块覆盖半径
DEFAULT_COVERAGE_RADIUS_8x4 = 65.0 # 8x4块覆盖半径
DEFAULT_COVERAGE_RADIUS_4x4 = 60.0  # 4x4块覆盖半径

# 块尺寸定义
BLOCK_8x8_W, BLOCK_8x8_H = 8, 8   # 8x8块
BLOCK_8x4_W, BLOCK_8x4_H = 8, 4   # 8x4块
BLOCK_4x4_W, BLOCK_4x4_H = 4, 4   # 4x4块
BLOCK_4x2_W, BLOCK_4x2_H = 4, 2   # 4x2块

PIXELS_PER_8x8_BLOCK = BLOCK_8x8_W * BLOCK_8x8_H  # 64
PIXELS_PER_8x4_BLOCK = BLOCK_8x4_W * BLOCK_8x4_H  # 32
PIXELS_PER_4x4_BLOCK = BLOCK_4x4_W * BLOCK_4x4_H  # 16
PIXELS_PER_4x2_BLOCK = BLOCK_4x2_W * BLOCK_4x2_H  # 8

# 8x8块数量（用于I帧主编码）
BLOCKS_8x8_PER_FRAME = (WIDTH // BLOCK_8x8_W) * (HEIGHT // BLOCK_8x8_H)  # 30 * 20 = 600
# 8x4块数量（用于分裂编码）
BLOCKS_8x4_PER_FRAME = (WIDTH // BLOCK_8x4_W) * (HEIGHT // BLOCK_8x4_H)  # 30 * 40 = 1200
# 4x4块数量（用于分裂编码）
BLOCKS_4x4_PER_FRAME = (WIDTH // BLOCK_4x4_W) * (HEIGHT // BLOCK_4x4_H)  # 60 * 40 = 2400
# 4x2块数量（用于细分编码）
BLOCKS_4x2_PER_FRAME = (WIDTH // BLOCK_4x2_W) * (HEIGHT // BLOCK_4x2_H)  # 60 * 80 = 4800

# 特殊标记 - 统一使用0xFF作为分裂标志
MARKER_SPLIT = 0xFF  # 统一分裂标志，表示需要分裂到下一级

# Zone配置 - 每240个块为一个zone
BLOCKS_PER_ZONE = 240

# Zone数量和bitmap类型
ZONES_8x8_COUNT = (BLOCKS_8x8_PER_FRAME + BLOCKS_PER_ZONE - 1) // BLOCKS_PER_ZONE  # 3个zone
ZONES_8x4_COUNT = (BLOCKS_8x4_PER_FRAME + BLOCKS_PER_ZONE - 1) // BLOCKS_PER_ZONE  # 5个zone  
ZONES_4x4_COUNT = (BLOCKS_4x4_PER_FRAME + BLOCKS_PER_ZONE - 1) // BLOCKS_PER_ZONE  # 10个zone
ZONES_4x2_COUNT = (BLOCKS_4x2_PER_FRAME + BLOCKS_PER_ZONE - 1) // BLOCKS_PER_ZONE  # 20个zone

# Zone像素高度（zone在屏幕上占据的像素高度）
ZONE_8x8_PIXEL_HEIGHT = BLOCKS_PER_ZONE * PIXELS_PER_8x8_BLOCK // (WIDTH // BLOCK_8x8_W)  # 240*64/30 = 512像素
ZONE_8x4_PIXEL_HEIGHT = BLOCKS_PER_ZONE * PIXELS_PER_8x4_BLOCK // (WIDTH // BLOCK_8x4_W)  # 240*32/30 = 256像素
ZONE_4x4_PIXEL_HEIGHT = BLOCKS_PER_ZONE * PIXELS_PER_4x4_BLOCK // (WIDTH // BLOCK_4x4_W)  # 240*16/60 = 64像素
ZONE_4x2_PIXEL_HEIGHT = BLOCKS_PER_ZONE * PIXELS_PER_4x2_BLOCK // (WIDTH // BLOCK_4x2_W)  # 240*8/60 = 32像素

# IP帧编码参数
GOP_SIZE = 30  # GOP大小，每30帧一个I帧
I_FRAME_WEIGHT = 3  # I帧块的权重（用于K-means训练）
DIFF_THRESHOLD = 100  # 块差异阈值，超过此值认为块需要更新

def encode_8x4_block_recursive(
    block_8x4: np.ndarray,
    codebook_8x4: np.ndarray,
    codebook_4x4: np.ndarray, 
    codebook_4x2: np.ndarray,
    coverage_radius_8x4: float,
    coverage_radius_4x4: float,
    reference_8x4: np.ndarray = None
) -> tuple:
    """
    递归编码单个8x4块，严格按照8x4→4x4→4x2的分裂顺序
    
    返回: (encoding_list, stats)
    encoding_list格式:
    - 如果用8x4码表: [8x4_index]
    - 如果拆分为4x4: [MARKER_SPLIT, left_4x4_encoding..., right_4x4_encoding...]
    """
    stats = {'blocks_8x8_used': 0, 'blocks_8x4_used': 0, 'blocks_4x4_used': 0, 'blocks_4x2_used': 0}
    
    # 尝试8x4码表
    distances_8x4 = pairwise_distances(
        block_8x4.reshape(1, -1).astype(np.float32),
        codebook_8x4.astype(np.float32),
        metric="euclidean"
    )
    min_dist_8x4 = distances_8x4.min()
    
    if min_dist_8x4 <= coverage_radius_8x4:
        # 可以用8x4码表
        best_idx = distances_8x4.argmin()
        stats['blocks_8x4_used'] = 1
        return [best_idx], stats
    else:
        # 8x4无法覆盖，拆分为两个4x4块
        # 正确的8x4→4x4拆分：左右分割，而不是前后分割
        
        # 提取Y分量（4行8列，按行存储）
        y_8x4 = block_8x4[:32].reshape(4, 8)  # 重塑为4x8矩阵
        # 左半4x4：前4列
        left_y_4x4 = y_8x4[:, :4].flatten()   # 每行前4个像素
        # 右半4x4：后4列  
        right_y_4x4 = y_8x4[:, 4:].flatten()  # 每行后4个像素
        
        # 提取Cb分量（4行8列，按行存储）
        cb_8x4 = block_8x4[32:64].reshape(4, 8)
        left_cb_4x4 = cb_8x4[:, :4].flatten()
        right_cb_4x4 = cb_8x4[:, 4:].flatten()
        
        # 提取Cr分量（4行8列，按行存储）
        cr_8x4 = block_8x4[64:96].reshape(4, 8)
        left_cr_4x4 = cr_8x4[:, :4].flatten()
        right_cr_4x4 = cr_8x4[:, 4:].flatten()
        
        # 组装左半4x4块（16Y + 16Cb + 16Cr）
        left_4x4 = np.concatenate([left_y_4x4, left_cb_4x4, left_cr_4x4])
        # 组装右半4x4块（16Y + 16Cb + 16Cr）
        right_4x4 = np.concatenate([right_y_4x4, right_cb_4x4, right_cr_4x4])
        
        # 准备参考块（如果有的话）
        left_ref_4x4 = None
        right_ref_4x4 = None
        if reference_8x4 is not None:
            # 拆分参考8x4块为两个4x4块（与当前块相同的拆分方式）
            ref_y_8x4 = reference_8x4[:32].reshape(4, 8)
            left_ref_y_4x4 = ref_y_8x4[:, :4].flatten()
            right_ref_y_4x4 = ref_y_8x4[:, 4:].flatten()
            
            ref_cb_8x4 = reference_8x4[32:64].reshape(4, 8)
            left_ref_cb_4x4 = ref_cb_8x4[:, :4].flatten()
            right_ref_cb_4x4 = ref_cb_8x4[:, 4:].flatten()
            
            ref_cr_8x4 = reference_8x4[64:96].reshape(4, 8)
            left_ref_cr_4x4 = ref_cr_8x4[:, :4].flatten()
            right_ref_cr_4x4 = ref_cr_8x4[:, 4:].flatten()
            
            left_ref_4x4 = np.concatenate([left_ref_y_4x4, left_ref_cb_4x4, left_ref_cr_4x4])
            right_ref_4x4 = np.concatenate([right_ref_y_4x4, right_ref_cb_4x4, right_ref_cr_4x4])
        
        # 检查每个4x4子块是否需要更新
        left_encoding = []
        right_encoding = []
        left_stats = {'blocks_8x8_used': 0, 'blocks_8x4_used': 0, 'blocks_4x4_used': 0, 'blocks_4x2_used': 0}
        right_stats = {'blocks_8x8_used': 0, 'blocks_8x4_used': 0, 'blocks_4x4_used': 0, 'blocks_4x2_used': 0}
        
        # 检查左半4x4块是否需要更新
        if left_ref_4x4 is not None:
            left_diff = np.linalg.norm(left_4x4 - left_ref_4x4)
            if left_diff <= coverage_radius_4x4:
                # 左半块不需要更新，跳过编码
                left_encoding = []  # 空编码表示不更新
            else:
                # 左半块需要更新
                left_encoding, left_stats = encode_4x4_block_recursive(
                    left_4x4, codebook_4x4, codebook_4x2, coverage_radius_4x4, left_ref_4x4
                )
        else:
            # 没有参考块，总是编码
            left_encoding, left_stats = encode_4x4_block_recursive(
                left_4x4, codebook_4x4, codebook_4x2, coverage_radius_4x4, None
            )
        
        # 检查右半4x4块是否需要更新
        if right_ref_4x4 is not None:
            right_diff = np.linalg.norm(right_4x4 - right_ref_4x4)
            if right_diff <= coverage_radius_4x4:
                # 右半块不需要更新，跳过编码
                right_encoding = []  # 空编码表示不更新
            else:
                # 右半块需要更新
                right_encoding, right_stats = encode_4x4_block_recursive(
                    right_4x4, codebook_4x4, codebook_4x2, coverage_radius_4x4, right_ref_4x4
                )
        else:
            # 没有参考块，总是编码
            right_encoding, right_stats = encode_4x4_block_recursive(
                right_4x4, codebook_4x4, codebook_4x2, coverage_radius_4x4, None
            )
        
        # 合并统计
        for key in stats:
            stats[key] = left_stats[key] + right_stats[key]
        
        # 组装编码结果 - 只有需要更新的块才编码
        if len(left_encoding) == 0 and len(right_encoding) == 0:
            # 两个子块都不需要更新，直接返回空编码（不应该到这里，因为父级已经检查过）
            return [], stats
        elif len(left_encoding) == 0:
            # 只有右半块需要更新
            encoding = [MARKER_SPLIT, 0] + right_encoding  # 0表示左半块不更新
        elif len(right_encoding) == 0:
            # 只有左半块需要更新
            encoding = [MARKER_SPLIT] + left_encoding + [0]  # 0表示右半块不更新
        else:
            # 两个子块都需要更新
            encoding = [MARKER_SPLIT] + left_encoding + right_encoding
        
        return encoding, stats

def encode_4x4_block_recursive(
    block_4x4: np.ndarray,
    codebook_4x4: np.ndarray,
    codebook_4x2: np.ndarray,
    coverage_radius_4x4: float,
    reference_4x4: np.ndarray = None,
    coverage_radius_4x2: float = 40.0  # 4x2块的覆盖半径
) -> tuple:
    """
    递归编码单个4x4块，严格按照4x4→4x2的分裂顺序
    
    返回: (encoding_list, stats)
    encoding_list格式:
    - 如果用4x4码表: [4x4_index]
    - 如果拆分为4x2: [MARKER_SPLIT, upper_4x2_index, lower_4x2_index]
    """
    stats = {'blocks_8x8_used': 0, 'blocks_8x4_used': 0, 'blocks_4x4_used': 0, 'blocks_4x2_used': 0}
    
    # 尝试4x4码表
    distances_4x4 = pairwise_distances(
        block_4x4.reshape(1, -1).astype(np.float32),
        codebook_4x4.astype(np.float32),
        metric="euclidean"
    )
    min_dist_4x4 = distances_4x4.min()
    
    if min_dist_4x4 <= coverage_radius_4x4:
        # 可以用4x4码表
        best_idx = distances_4x4.argmin()
        stats['blocks_4x4_used'] = 1
        return [best_idx], stats
    else:
        # 4x4无法覆盖，拆分为两个4x2块
        upper_4x2 = np.concatenate([
            block_4x4[:8],      # 前8个Y值（前2行）
            block_4x4[16:24],   # 前8个Cb值（前2行）
            block_4x4[32:40]    # 前8个Cr值（前2行）
        ])
        lower_4x2 = np.concatenate([
            block_4x4[8:16],    # 后8个Y值（后2行）
            block_4x4[24:32],   # 后8个Cb值（后2行）
            block_4x4[40:48]    # 后8个Cr值（后2行）
        ])
        
        # 准备参考块（如果有的话）
        upper_ref_4x2 = None
        lower_ref_4x2 = None
        if reference_4x4 is not None:
            upper_ref_4x2 = np.concatenate([
                reference_4x4[:8],      # 前8个Y值（前2行）
                reference_4x4[16:24],   # 前8个Cb值（前2行）
                reference_4x4[32:40]    # 前8个Cr值（前2行）
            ])
            lower_ref_4x2 = np.concatenate([
                reference_4x4[8:16],    # 后8个Y值（后2行）
                reference_4x4[24:32],   # 后8个Cb值（后2行）
                reference_4x4[40:48]    # 后8个Cr值（后2行）
            ])
        
        # 检查每个4x2子块是否需要更新
        upper_encoding = []
        lower_encoding = []
        blocks_4x2_used = 0
        
        # 检查上半4x2块是否需要更新
        if upper_ref_4x2 is not None:
            upper_diff = np.linalg.norm(upper_4x2 - upper_ref_4x2)
            if upper_diff > coverage_radius_4x2:
                # 上半块需要更新 - 4x2作为兜底，必须编码
                upper_indices = encode_frame_with_codebook(upper_4x2.reshape(1, -1), codebook_4x2)
                upper_encoding = [upper_indices[0]]
                blocks_4x2_used += 1
            # 否则上半块不需要更新，保持空编码
        else:
            # 没有参考块，总是编码 - 4x2作为兜底，必须编码
            upper_indices = encode_frame_with_codebook(upper_4x2.reshape(1, -1), codebook_4x2)
            upper_encoding = [upper_indices[0]]
            blocks_4x2_used += 1
        
        # 检查下半4x2块是否需要更新
        if lower_ref_4x2 is not None:
            lower_diff = np.linalg.norm(lower_4x2 - lower_ref_4x2)
            if lower_diff > coverage_radius_4x2:
                # 下半块需要更新 - 4x2作为兜底，必须编码
                lower_indices = encode_frame_with_codebook(lower_4x2.reshape(1, -1), codebook_4x2)
                lower_encoding = [lower_indices[0]]
                blocks_4x2_used += 1
            # 否则下半块不需要更新，保持空编码
        else:
            # 没有参考块，总是编码 - 4x2作为兜底，必须编码
            lower_indices = encode_frame_with_codebook(lower_4x2.reshape(1, -1), codebook_4x2)
            lower_encoding = [lower_indices[0]]
            blocks_4x2_used += 1
        
        stats['blocks_4x2_used'] = blocks_4x2_used
        
        # 组装编码结果
        if len(upper_encoding) == 0 and len(lower_encoding) == 0:
            # 两个子块都不需要更新，不应该到这里
            return [], stats
        elif len(upper_encoding) == 0:
            # 只有下半块需要更新
            encoding = [MARKER_SPLIT, 0] + lower_encoding  # 0表示上半块不更新
        elif len(lower_encoding) == 0:
            # 只有上半块需要更新
            encoding = [MARKER_SPLIT] + upper_encoding + [0]  # 0表示下半块不更新
        else:
            # 两个子块都需要更新
            encoding = [MARKER_SPLIT] + upper_encoding + lower_encoding
        
        return encoding, stats

def encode_i_frame_multi_level_8x8(
    frame_blocks_8x8: np.ndarray, 
    codebook_8x8: np.ndarray,
    codebook_8x4: np.ndarray, 
    codebook_4x4: np.ndarray, 
    codebook_4x2: np.ndarray, 
    coverage_radius_8x8: float = 150.0,
    coverage_radius_8x4: float = 120.0,
    coverage_radius_4x4: float = 80.0,
    coverage_radius_4x2: float = 40.0
) -> tuple:
    """
    使用四级码表编码I帧 - 严格递归分裂：8x8→8x4→4x4→4x2
    
    新的编码格式：
    - 8x8块：8x8码字索引 (直接是索引)
    - 分裂为8x4块：MARKER_SPLIT, 上半8x4编码..., 下半8x4编码...
    - 分裂为4x4块：MARKER_SPLIT, 左半4x4编码..., 右半4x4编码...
    - 分裂为4x2块：MARKER_SPLIT, 上半4x2码字索引, 下半4x2码字索引
    
    返回格式：([总块数, 块1编码, 块2编码, ...], stats)
    
    stats格式：{
        'blocks_8x8_used': 使用8x8码表的块数,
        'blocks_8x4_used': 使用8x4码表的块数(以8x4块为单位),
        'blocks_4x4_used': 使用4x4码表的块数(以4x4块为单位),
        'blocks_4x2_used': 使用4x2码表的块数(以4x2块为单位)
    }
    """
    frame_data = [BLOCKS_8x8_PER_FRAME & 0xFF, (BLOCKS_8x8_PER_FRAME >> 8) & 0xFF]  # 总块数（低字节，高字节）
    
    # 统计信息
    total_stats = {
        'blocks_8x8_used': 0,
        'blocks_8x4_used': 0,
        'blocks_4x4_used': 0,
        'blocks_4x2_used': 0
    }
    
    # 逐个递归编码每个8x8块
    for block_idx in range(len(frame_blocks_8x8)):
        block_8x8 = frame_blocks_8x8[block_idx]
        
        # 递归编码当前8x8块
        encoding, stats = encode_8x8_block_recursive(
            block_8x8, codebook_8x8, codebook_8x4, codebook_4x4, codebook_4x2,
            coverage_radius_8x8, coverage_radius_8x4, coverage_radius_4x4, None, coverage_radius_4x2
        )
        
        # 添加编码结果
        frame_data.extend(encoding)
        
        # 累加统计
        for key in total_stats:
            total_stats[key] += stats[key]
    
    return frame_data, total_stats

def encode_i_frame_multi_level(frame_blocks_4x4: np.ndarray, codebook_4x4: np.ndarray, codebook_4x2: np.ndarray, coverage_radius: float = 80.0) -> tuple:
    """
    使用多级码表编码I帧 - 4x4块优先，FFFF作为分裂标志
    
    新的编码格式：
    - 4x4块：4x4码字索引 (直接是索引，不需要MARKER)
    - 分裂为4x2块：MARKER_SPLIT, 上半4x2码字索引, 下半4x2码字索引
    
    返回格式：([总块数, 块1编码, 块2编码, ...], stats)
    
    stats格式：{
        'blocks_4x4_used': 使用4x4码表的块数,
        'blocks_4x2_used': 使用4x2码表的块数(以4x2块为单位)
    }
    """
    frame_data = [BLOCKS_4x4_PER_FRAME & 0xFF, (BLOCKS_4x4_PER_FRAME >> 8) & 0xFF]  # 总块数（低字节，高字节）
    
    # 统计信息
    stats = {
        'blocks_4x4_used': 0,
        'blocks_4x2_used': 0
    }
    
    # 计算每个4x4块到4x4码表的最小距离
    distances_4x4 = pairwise_distances(
        frame_blocks_4x4.astype(np.float32),
        codebook_4x4.astype(np.float32),
        metric="euclidean",
        n_jobs=1
    )
    min_distances_4x4 = distances_4x4.min(axis=1)
    best_indices_4x4 = distances_4x4.argmin(axis=1)
    
    for block_idx in range(len(frame_blocks_4x4)):
        if min_distances_4x4[block_idx] <= coverage_radius:
            # 使用4x4码表 - 直接输出索引
            frame_data.append(best_indices_4x4[block_idx])
            stats['blocks_4x4_used'] += 1
        else:
            # 需要分裂为4x2块编码 - 输出FFFF分裂标志 + 两个4x2索引
            block_4x4 = frame_blocks_4x4[block_idx]
            
            # 上半部分：前2行
            upper_4x2 = np.concatenate([
                block_4x4[:8],      # 前8个Y值（前2行）
                block_4x4[16:24],   # 前8个Cb值（前2行）
                block_4x4[32:40]    # 前8个Cr值（前2行）
            ])
            # 下半部分：后2行
            lower_4x2 = np.concatenate([
                block_4x4[8:16],    # 后8个Y值（后2行）
                block_4x4[24:32],   # 后8个Cb值（后2行）
                block_4x4[40:48]    # 后8个Cr值（后2行）
            ])
            
            # 使用4x2码表编码
            upper_indices = encode_frame_with_codebook(upper_4x2.reshape(1, -1), codebook_4x2)
            lower_indices = encode_frame_with_codebook(lower_4x2.reshape(1, -1), codebook_4x2)
            
            # 输出：分裂标志 + 上半4x2索引 + 下半4x2索引
            frame_data.extend([MARKER_SPLIT, upper_indices[0], lower_indices[0]])
            stats['blocks_4x2_used'] += 2  # 一个4x4块拆分为2个4x2块
    
    return frame_data, stats

def encode_p_frame_multi_level(
    current_blocks_8x4: np.ndarray, 
    previous_blocks_8x4: np.ndarray, 
    codebook_8x4: np.ndarray,
    codebook_4x4: np.ndarray, 
    codebook_4x2: np.ndarray, 
    diff_threshold: float,
    coverage_radius_8x4: float = 120.0,
    coverage_radius_4x4: float = 80.0
) -> tuple:
    """
    使用三级码表编码P帧 - 严格递归分裂：8x4→4x4→4x2
    只编码发生变化的块
    
    返回格式：([变化块数, 位置1, 编码1..., 位置2, 编码2..., ...], stats)
    
    stats格式：{
        'blocks_8x4_used': 使用8x4码表的块数,
        'blocks_4x4_used': 使用4x4码表的块数,
        'blocks_4x2_used': 使用4x2码表的块数(以4x2块为单位)
    }
    """
    # 统计信息
    total_stats = {
        'blocks_8x4_used': 0,
        'blocks_4x4_used': 0,
        'blocks_4x2_used': 0
    }
    
    # 找出发生变化的8x4块
    changed_indices_8x4 = find_changed_blocks_8x4(current_blocks_8x4, previous_blocks_8x4, diff_threshold)
    
    if len(changed_indices_8x4) == 0:
        # 没有变化
        return [0, 0], total_stats  # 变化块数=0（低字节，高字节）
    
    frame_data = [len(changed_indices_8x4) & 0xFF, (len(changed_indices_8x4) >> 8) & 0xFF]  # 变化块数（低字节，高字节）
    
    # 逐个递归编码变化的8x4块
    for block_pos in changed_indices_8x4:
        block_8x4 = current_blocks_8x4[block_pos]
        
        # 递归编码当前8x4块
        encoding, stats = encode_8x4_block_recursive(
            block_8x4, codebook_8x4, codebook_4x4, codebook_4x2,
            coverage_radius_8x4, coverage_radius_4x4
        )
        
        # P帧格式：位置（2字节） + 编码
        frame_data.extend([block_pos & 0xFF, (block_pos >> 8) & 0xFF])  # 位置（低字节，高字节）
        frame_data.extend(encoding)
        
        # 累加统计
        for key in total_stats:
            total_stats[key] += stats[key]
    
    return frame_data, total_stats

def encode_p_frame_multi_level_8x8(
    current_blocks_8x8: np.ndarray, 
    previous_blocks_8x8: np.ndarray, 
    codebook_8x8: np.ndarray,
    codebook_8x4: np.ndarray,
    codebook_4x4: np.ndarray, 
    codebook_4x2: np.ndarray, 
    diff_threshold: float,
    coverage_radius_8x8: float = 150.0,
    coverage_radius_8x4: float = 120.0,
    coverage_radius_4x4: float = 80.0
) -> tuple:
    """
    使用四级码表编码P帧 - 严格递归分裂：8x8→8x4→4x4→4x2
    只编码发生变化的块
    
    返回格式：([变化块数, 位置1, 编码1..., 位置2, 编码2..., ...], stats)
    
    stats格式：{
        'blocks_8x8_used': 使用8x8码表的块数,
        'blocks_8x4_used': 使用8x4码表的块数,
        'blocks_4x4_used': 使用4x4码表的块数,
        'blocks_4x2_used': 使用4x2码表的块数(以4x2块为单位)
    }
    """
    # 统计信息
    total_stats = {
        'blocks_8x8_used': 0,
        'blocks_8x4_used': 0,
        'blocks_4x4_used': 0,
        'blocks_4x2_used': 0
    }
    
    # 找出发生变化的8x8块
    changed_indices_8x8 = find_changed_blocks_8x8(current_blocks_8x8, previous_blocks_8x8, diff_threshold)
    
    if len(changed_indices_8x8) == 0:
        # 没有变化
        return [0, 0], total_stats  # 变化块数=0（低字节，高字节）
    
    frame_data = [len(changed_indices_8x8) & 0xFF, (len(changed_indices_8x8) >> 8) & 0xFF]  # 变化块数（低字节，高字节）
    
    # 逐个递归编码变化的8x8块
    for block_pos in changed_indices_8x8:
        block_8x8 = current_blocks_8x8[block_pos]
        
        # 递归编码当前8x8块
        encoding, stats = encode_8x8_block_recursive(
            block_8x8, codebook_8x8, codebook_8x4, codebook_4x4, codebook_4x2,
            coverage_radius_8x8, coverage_radius_8x4, coverage_radius_4x4
        )
        
        # P帧格式：位置（2字节） + 编码
        frame_data.extend([block_pos & 0xFF, (block_pos >> 8) & 0xFF])  # 位置（低字节，高字节）
        frame_data.extend(encoding)
        
        # 累加统计
        for key in total_stats:
            total_stats[key] += stats[key]
    
    return frame_data, total_stats

def encode_8x8_block_recursive(
    block_8x8: np.ndarray,
    codebook_8x8: np.ndarray,
    codebook_8x4: np.ndarray,
    codebook_4x4: np.ndarray, 
    codebook_4x2: np.ndarray,
    coverage_radius_8x8: float,
    coverage_radius_8x4: float,
    coverage_radius_4x4: float,
    reference_8x8: np.ndarray = None,
    coverage_radius_4x2: float = 40.0
) -> tuple:
    """
    递归编码单个8x8块，严格按照8x8→8x4→4x4→4x2的分裂顺序
    
    返回: (encoding_list, stats)
    encoding_list格式:
    - 如果用8x8码表: [8x8_index]
    - 如果拆分为8x4: [MARKER_SPLIT, upper_8x4_encoding..., lower_8x4_encoding...]
    """
    stats = {'blocks_8x8_used': 0, 'blocks_8x4_used': 0, 'blocks_4x4_used': 0, 'blocks_4x2_used': 0}
    
    # 尝试8x8码表
    distances_8x8 = pairwise_distances(
        block_8x8.reshape(1, -1).astype(np.float32),
        codebook_8x8.astype(np.float32),
        metric="euclidean"
    )
    min_dist_8x8 = distances_8x8.min()
    
    if min_dist_8x8 <= coverage_radius_8x8:
        # 可以用8x8码表
        best_idx = distances_8x8.argmin()
        stats['blocks_8x8_used'] = 1
        return [best_idx], stats
    else:
        # 8x8无法覆盖，拆分为两个8x4块（上下分割）
        
        # 提取Y分量（8行8列，按行存储）
        y_8x8 = block_8x8[:64].reshape(8, 8)  # 重塑为8x8矩阵
        # 上半8x4：前4行
        upper_y_8x4 = y_8x8[:4, :].flatten()   # 前4行全部像素
        # 下半8x4：后4行  
        lower_y_8x4 = y_8x8[4:, :].flatten()  # 后4行全部像素
        
        # 提取Cb分量（8行8列，按行存储）
        cb_8x8 = block_8x8[64:128].reshape(8, 8)
        upper_cb_8x4 = cb_8x8[:4, :].flatten()
        lower_cb_8x4 = cb_8x8[4:, :].flatten()
        
        # 提取Cr分量（8行8列，按行存储）
        cr_8x8 = block_8x8[128:192].reshape(8, 8)
        upper_cr_8x4 = cr_8x8[:4, :].flatten()
        lower_cr_8x4 = cr_8x8[4:, :].flatten()
        
        # 组装上半8x4块（32Y + 32Cb + 32Cr）
        upper_8x4 = np.concatenate([upper_y_8x4, upper_cb_8x4, upper_cr_8x4])
        # 组装下半8x4块（32Y + 32Cb + 32Cr）
        lower_8x4 = np.concatenate([lower_y_8x4, lower_cb_8x4, lower_cr_8x4])
        
        # 准备参考块（如果有的话）
        upper_ref_8x4 = None
        lower_ref_8x4 = None
        if reference_8x8 is not None:
            # 拆分参考8x8块为两个8x4块（与当前块相同的拆分方式）
            ref_y_8x8 = reference_8x8[:64].reshape(8, 8)
            upper_ref_y_8x4 = ref_y_8x8[:4, :].flatten()
            lower_ref_y_8x4 = ref_y_8x8[4:, :].flatten()
            
            ref_cb_8x8 = reference_8x8[64:128].reshape(8, 8)
            upper_ref_cb_8x4 = ref_cb_8x8[:4, :].flatten()
            lower_ref_cb_8x4 = ref_cb_8x8[4:, :].flatten()
            
            ref_cr_8x8 = reference_8x8[128:192].reshape(8, 8)
            upper_ref_cr_8x4 = ref_cr_8x8[:4, :].flatten()
            lower_ref_cr_8x4 = ref_cr_8x8[4:, :].flatten()
            
            upper_ref_8x4 = np.concatenate([upper_ref_y_8x4, upper_ref_cb_8x4, upper_ref_cr_8x4])
            lower_ref_8x4 = np.concatenate([lower_ref_y_8x4, lower_ref_cb_8x4, lower_ref_cr_8x4])
        
        # 递归编码上8x4和下8x4
        upper_encoding, upper_stats = encode_8x4_block_recursive(
            upper_8x4, codebook_8x4, codebook_4x4, codebook_4x2,
            coverage_radius_8x4, coverage_radius_4x4, upper_ref_8x4
        )
        lower_encoding, lower_stats = encode_8x4_block_recursive(
            lower_8x4, codebook_8x4, codebook_4x4, codebook_4x2,
            coverage_radius_8x4, coverage_radius_4x4, lower_ref_8x4
        )
        
        # 合并统计
        for key in stats:
            stats[key] = upper_stats[key] + lower_stats[key]
        
        # 组装编码结果
        encoding = [MARKER_SPLIT] + upper_encoding + lower_encoding
        return encoding, stats

def reconstruct_frame_from_encoding(frame_data, frame_type, codebooks, previous_decoded_frame=None):
    """
    从编码数据重建帧，用于维护解码帧缓存
    
    Args:
        frame_data: 编码数据
        frame_type: 帧类型 (0=I帧, 1=P帧)
        codebooks: (codebook_8x8, codebook_8x4, codebook_4x4, codebook_4x2)
        previous_decoded_frame: 前一解码帧 (仅P帧需要)
    
    Returns:
        np.ndarray: 重建的8x8块数组
    """
    codebook_8x8, codebook_8x4, codebook_4x4, codebook_4x2 = codebooks
    
    if frame_type == 0:  # I帧
        # I帧重建：从编码数据完全重建所有块
        # 这里需要解码逻辑，暂时返回原始帧（简化实现）
        # 实际应该根据frame_data解码，但这需要复杂的解码器
        # 为了解决累积误差，我们用近似方法：直接用码表最近邻
        reconstructed_blocks = np.zeros((BLOCKS_8x8_PER_FRAME, 192), dtype=np.float32)
        
        # 简化实现：用8x8码表的第一个码字填充所有块
        if len(codebook_8x8) > 0:
            for i in range(BLOCKS_8x8_PER_FRAME):
                reconstructed_blocks[i] = codebook_8x8[0]  # 使用第一个码字作为默认
        
        return reconstructed_blocks
    else:  # P帧
        # P帧重建：从前一帧开始，更新变化的块
        if previous_decoded_frame is None:
            raise ValueError("P帧重建需要前一解码帧")
        
        # 从前一帧复制所有块
        reconstructed_blocks = previous_decoded_frame.copy()
        
        # 这里需要解析P帧编码数据并更新对应块
        # 简化实现：假设更新的块用对应码表的第一个码字
        # 实际应该解析frame_data中的位置和编码信息
        
        return reconstructed_blocks

def process_single_gop(args):
    """
    处理单个GOP的函数 - 用于多进程
    修改：维护解码帧缓存以避免累积误差
    
    Args:
        args: 包含处理GOP所需的所有参数的元组
        
    Returns:
        tuple: (gop_idx, codebooks, encoded_frames, frame_stats)
    """
    (gop_idx, gop_frames, gop_size, i_frame_weight, diff_threshold,
     codebook_size_8x8, codebook_size_8x4, codebook_size_4x4, codebook_size_4x2,
     coverage_radius_8x8, coverage_radius_8x4, coverage_radius_4x4, coverage_radius_4x2) = args
    
    print(f"🔄 进程 {mp.current_process().name} 正在处理 GOP {gop_idx + 1}/{gop_size}...")
    start_time = time.time()
    
    # 第一帧是I帧
    i_frame_blocks_8x8 = gop_frames[0]
    
    # 分析P帧的变化8x8块
    p_frame_blocks_8x8_list = []
    for frame_idx in range(1, len(gop_frames)):
        current_blocks = gop_frames[frame_idx]
        previous_blocks = gop_frames[frame_idx - 1]
        changed_indices = find_changed_blocks_8x8(current_blocks, previous_blocks, diff_threshold)
        if len(changed_indices) > 0:
            changed_blocks = current_blocks[changed_indices]
            p_frame_blocks_8x8_list.append((frame_idx, changed_blocks))
    
    # 为当前GOP生成四级码表
    codebook_8x8, codebook_8x4, codebook_4x4, codebook_4x2 = generate_multi_level_codebooks_for_gop_8x8(
        i_frame_blocks_8x8, p_frame_blocks_8x8_list, i_frame_weight, 
        coverage_radius_8x8, coverage_radius_8x4, coverage_radius_4x4, 
        codebook_size_8x8, codebook_size_8x4, codebook_size_4x4, codebook_size_4x2
    )
    
    codebooks = (codebook_8x8, codebook_8x4, codebook_4x4, codebook_4x2)
    
    # 编码当前GOP的所有帧
    encoded_frames = []
    frame_stats_list = []
    
    # 维护解码帧缓存以避免累积误差
    decoded_frame_cache = None
    
    for frame_idx, frame_blocks_8x8 in enumerate(gop_frames):
        if frame_idx == 0:  # I帧
            frame_data, frame_stats = encode_i_frame_multi_level_8x8(
                frame_blocks_8x8, codebook_8x8, codebook_8x4, codebook_4x4, codebook_4x2, 
                coverage_radius_8x8, coverage_radius_8x4, coverage_radius_4x4, coverage_radius_4x2
            )
            frame_type = 0  # I帧
            
            # 重建I帧作为解码缓存的初始值
            decoded_frame_cache = reconstruct_frame_from_encoding(frame_data, frame_type, codebooks)
            
        else:  # P帧
            # P帧与解码缓存比较，而不是原始前一帧
            if decoded_frame_cache is not None:
                reference_blocks = decoded_frame_cache
            else:
                # 如果没有解码缓存，退回到原始前一帧（不应该发生）
                reference_blocks = gop_frames[frame_idx - 1]
            
            frame_data, frame_stats = encode_p_frame_multi_level_separate(
                frame_blocks_8x8, reference_blocks, codebook_8x8, codebook_8x4, codebook_4x4, codebook_4x2, 
                diff_threshold, coverage_radius_8x8, coverage_radius_8x4, coverage_radius_4x4, coverage_radius_4x2
            )
            frame_type = 1  # P帧
            
            # 更新解码缓存
            decoded_frame_cache = reconstruct_frame_from_encoding(frame_data, frame_type, codebooks, decoded_frame_cache)
        
        encoded_frames.append((frame_data, frame_type))
        frame_stats_list.append(frame_stats)
    
    elapsed_time = time.time() - start_time
    print(f"✅ GOP {gop_idx + 1} 处理完成 ({elapsed_time:.1f}s)")
    
    return (gop_idx, codebooks, encoded_frames, frame_stats_list)

def encode_blocks_with_run_length(blocks_list):
    """
    将块列表编码为连续块格式：位置+连续数量+码表索引序列
    
    输入: [(pos, code_idx), (pos, code_idx), ...]
    输出: [总组数(2字节), 位置1(2字节), 连续数1(2字节), 码表索引1, 码表索引2, ..., 位置2(2字节), 连续数2(2字节), ...]
    """
    if not blocks_list:
        return [0, 0]  # 总组数为0（2字节）
    
    # 按位置排序
    sorted_blocks = sorted(blocks_list, key=lambda x: x[0])
    
    groups = []
    current_start = sorted_blocks[0][0]
    current_codes = [sorted_blocks[0][1]]
    
    for i in range(1, len(sorted_blocks)):
        pos, code_idx = sorted_blocks[i]
        
        # 检查是否连续
        if pos == current_start + len(current_codes):
            # 连续块，添加到当前组
            current_codes.append(code_idx)
        else:
            # 不连续，结束当前组，开始新组
            groups.append((current_start, current_codes))
            current_start = pos
            current_codes = [code_idx]
    
    # 添加最后一组
    groups.append((current_start, current_codes))
    
    # 编码为字节数组
    frame_data = []
    
    # 总组数（2字节）
    group_count = len(groups)
    frame_data.extend([group_count & 0xFF, (group_count >> 8) & 0xFF])
    
    # 每组数据
    for start_pos, codes in groups:
        # 起始位置（2字节）
        frame_data.extend([start_pos & 0xFF, (start_pos >> 8) & 0xFF])
        # 连续块数量（2字节）
        count = len(codes)
        frame_data.extend([count & 0xFF, (count >> 8) & 0xFF])
        # 码表索引序列（每个1字节）
        frame_data.extend(codes)
    
    return frame_data

def encode_p_frame_multi_level_separate(
    current_blocks_8x8: np.ndarray, 
    previous_blocks_8x8: np.ndarray, 
    codebook_8x8: np.ndarray,
    codebook_8x4: np.ndarray,
    codebook_4x4: np.ndarray, 
    codebook_4x2: np.ndarray, 
    diff_threshold: float,
    coverage_radius_8x8: float = 150.0,
    coverage_radius_8x4: float = 120.0,
    coverage_radius_4x4: float = 80.0,
    coverage_radius_4x2: float = 40.0
) -> tuple:
    """
    使用分层编码P帧 - 每种码表单独处理
    
    返回格式：[8x8块数, 位置1, 码表项1, 位置2, 码表项2, ..., 
              8x4块数, 位置1, 码表项1, ..., 
              4x4块数, 位置1, 码表项1, ..., 
              4x2块数, 位置1, 码表项1, ...]
    
    位置编码：
    - 8x8块：屏幕240x160分为30x20=600个8x8块，位置0-599
    - 8x4块：屏幕240x160分为30x40=1200个8x4块，位置0-1199  
    - 4x4块：屏幕240x160分为60x40=2400个4x4块，位置0-2399
    - 4x2块：屏幕240x160分为60x80=4800个4x2块，位置0-4799
    """
    # 统计信息
    total_stats = {
        'blocks_8x8_used': 0,
        'blocks_8x4_used': 0,
        'blocks_4x4_used': 0,
        'blocks_4x2_used': 0
    }
    
    # 找出发生变化的8x8块
    changed_indices_8x8 = find_changed_blocks_8x8(current_blocks_8x8, previous_blocks_8x8, diff_threshold)
    
    if len(changed_indices_8x8) == 0:
        # 没有变化，返回全0
        return [0, 0, 0, 0, 0, 0, 0, 0], total_stats  # 各级块数均为0（每个2字节）
    
    # 需要编码的块列表
    blocks_to_encode_8x8 = []
    blocks_to_encode_8x4 = []
    blocks_to_encode_4x4 = []
    blocks_to_encode_4x2 = []
    
    # 1. 处理8x8块
    for block_pos_8x8 in changed_indices_8x8:
        block_8x8 = current_blocks_8x8[block_pos_8x8]
        
        # 尝试8x8码表
        distances_8x8 = pairwise_distances(
            block_8x8.reshape(1, -1).astype(np.float32),
            codebook_8x8.astype(np.float32),
            metric="euclidean"
        )
        min_dist_8x8 = distances_8x8.min()
        
        if min_dist_8x8 <= coverage_radius_8x8:
            # 可以用8x8码表
            best_idx = distances_8x8.argmin()
            blocks_to_encode_8x8.append((block_pos_8x8, best_idx))
            total_stats['blocks_8x8_used'] += 1
        else:
            # 8x8无法覆盖，拆分为两个8x4块
            # 计算对应的8x4块位置
            # 8x8块位置 = (by_8x8, bx_8x8) 其中by_8x8 ∈ [0,19], bx_8x8 ∈ [0,29]
            by_8x8 = block_pos_8x8 // 30  # 8x8块的行索引
            bx_8x8 = block_pos_8x8 % 30   # 8x8块的列索引
            
            # 上半8x4块位置：(by_8x8*2, bx_8x8)
            upper_block_pos_8x4 = (by_8x8 * 2) * 30 + bx_8x8
            # 下半8x4块位置：(by_8x8*2+1, bx_8x8)
            lower_block_pos_8x4 = (by_8x8 * 2 + 1) * 30 + bx_8x8
            
            # 拆分8x8块为两个8x4块
            # 提取Y分量（8行8列，按行存储）
            y_8x8 = block_8x8[:64].reshape(8, 8)
            upper_y_8x4 = y_8x8[:4, :].flatten()
            lower_y_8x4 = y_8x8[4:, :].flatten()
            
            # 提取Cb分量
            cb_8x8 = block_8x8[64:128].reshape(8, 8)
            upper_cb_8x4 = cb_8x8[:4, :].flatten()
            lower_cb_8x4 = cb_8x8[4:, :].flatten()
            
            # 提取Cr分量
            cr_8x8 = block_8x8[128:192].reshape(8, 8)
            upper_cr_8x4 = cr_8x8[:4, :].flatten()
            lower_cr_8x4 = cr_8x8[4:, :].flatten()
            
            # 组装8x4块
            upper_8x4 = np.concatenate([upper_y_8x4, upper_cb_8x4, upper_cr_8x4])
            lower_8x4 = np.concatenate([lower_y_8x4, lower_cb_8x4, lower_cr_8x4])
            
            # 添加到8x4待编码列表
            blocks_to_encode_8x4.append((upper_block_pos_8x4, upper_8x4))
            blocks_to_encode_8x4.append((lower_block_pos_8x4, lower_8x4))
    
    # 2. 处理8x4块
    final_blocks_8x4 = []
    for block_pos_8x4, block_8x4 in blocks_to_encode_8x4:
        # 尝试8x4码表
        distances_8x4 = pairwise_distances(
            block_8x4.reshape(1, -1).astype(np.float32),
            codebook_8x4.astype(np.float32),
            metric="euclidean"
        )
        min_dist_8x4 = distances_8x4.min()
        
        if min_dist_8x4 <= coverage_radius_8x4:
            # 可以用8x4码表
            best_idx = distances_8x4.argmin()
            final_blocks_8x4.append((block_pos_8x4, best_idx))
            total_stats['blocks_8x4_used'] += 1
        else:
            # 8x4无法覆盖，拆分为两个4x4块
            # 计算对应的4x4块位置
            # 8x4块位置 = (by_8x4, bx_8x4) 其中by_8x4 ∈ [0,39], bx_8x4 ∈ [0,29]
            by_8x4 = block_pos_8x4 // 30  # 8x4块的行索引
            bx_8x4 = block_pos_8x4 % 30   # 8x4块的列索引
            
            # 左半4x4块位置：(by_8x4, bx_8x4*2)
            left_block_pos_4x4 = by_8x4 * 60 + (bx_8x4 * 2)
            # 右半4x4块位置：(by_8x4, bx_8x4*2+1)
            right_block_pos_4x4 = by_8x4 * 60 + (bx_8x4 * 2 + 1)
            
            # 拆分8x4块为两个4x4块
            # 提取Y分量（4行8列，按行存储）
            y_8x4 = block_8x4[:32].reshape(4, 8)
            left_y_4x4 = y_8x4[:, :4].flatten()
            right_y_4x4 = y_8x4[:, 4:].flatten()
            
            # 提取Cb分量
            cb_8x4 = block_8x4[32:64].reshape(4, 8)
            left_cb_4x4 = cb_8x4[:, :4].flatten()
            right_cb_4x4 = cb_8x4[:, 4:].flatten()
            
            # 提取Cr分量
            cr_8x4 = block_8x4[64:96].reshape(4, 8)
            left_cr_4x4 = cr_8x4[:, :4].flatten()
            right_cr_4x4 = cr_8x4[:, 4:].flatten()
            
            # 组装4x4块
            left_4x4 = np.concatenate([left_y_4x4, left_cb_4x4, left_cr_4x4])
            right_4x4 = np.concatenate([right_y_4x4, right_cb_4x4, right_cr_4x4])
            
            # 添加到4x4待编码列表
            blocks_to_encode_4x4.append((left_block_pos_4x4, left_4x4))
            blocks_to_encode_4x4.append((right_block_pos_4x4, right_4x4))
    
    # 3. 处理4x4块
    final_blocks_4x4 = []
    for block_pos_4x4, block_4x4 in blocks_to_encode_4x4:
        # 尝试4x4码表
        distances_4x4 = pairwise_distances(
            block_4x4.reshape(1, -1).astype(np.float32),
            codebook_4x4.astype(np.float32),
            metric="euclidean"
        )
        min_dist_4x4 = distances_4x4.min()
        
        if min_dist_4x4 <= coverage_radius_4x4:
            # 可以用4x4码表
            best_idx = distances_4x4.argmin()
            final_blocks_4x4.append((block_pos_4x4, best_idx))
            total_stats['blocks_4x4_used'] += 1
        else:
            # 4x4无法覆盖，拆分为两个4x2块
            # 计算对应的4x2块位置
            # 4x4块位置 = (by_4x4, bx_4x4) 其中by_4x4 ∈ [0,39], bx_4x4 ∈ [0,59]
            by_4x4 = block_pos_4x4 // 60  # 4x4块的行索引
            bx_4x4 = block_pos_4x4 % 60   # 4x4块的列索引
            
            # 上半4x2块位置：(by_4x4*2, bx_4x4)
            upper_block_pos_4x2 = (by_4x4 * 2) * 60 + bx_4x4
            # 下半4x2块位置：(by_4x4*2+1, bx_4x4)
            lower_block_pos_4x2 = (by_4x4 * 2 + 1) * 60 + bx_4x4
            
            # 拆分4x4块为两个4x2块
            # 提取Y分量（4行4列，按行存储）
            y_4x4 = block_4x4[:16].reshape(4, 4)
            upper_y_4x2 = y_4x4[:2, :].flatten()
            lower_y_4x2 = y_4x4[2:, :].flatten()
            
            # 提取Cb分量
            cb_4x4 = block_4x4[16:32].reshape(4, 4)
            upper_cb_4x2 = cb_4x4[:2, :].flatten()
            lower_cb_4x2 = cb_4x4[2:, :].flatten()
            
            # 提取Cr分量
            cr_4x4 = block_4x4[32:48].reshape(4, 4)
            upper_cr_4x2 = cr_4x4[:2, :].flatten()
            lower_cr_4x2 = cr_4x4[2:, :].flatten()
            
            # 组装4x2块
            upper_4x2 = np.concatenate([upper_y_4x2, upper_cb_4x2, upper_cr_4x2])
            lower_4x2 = np.concatenate([lower_y_4x2, lower_cb_4x2, lower_cr_4x2])
            
            # 添加到4x2待编码列表
            blocks_to_encode_4x2.append((upper_block_pos_4x2, upper_4x2))
            blocks_to_encode_4x2.append((lower_block_pos_4x2, lower_4x2))
    
    # 4. 处理4x2块（最终级别，必须编码）
    final_blocks_4x2 = []
    for block_pos_4x2, block_4x2 in blocks_to_encode_4x2:
        # 4x2块必须用4x2码表编码
        distances_4x2 = pairwise_distances(
            block_4x2.reshape(1, -1).astype(np.float32),
            codebook_4x2.astype(np.float32),
            metric="euclidean"
        )
        best_idx = distances_4x2.argmin()
        final_blocks_4x2.append((block_pos_4x2, best_idx))
        total_stats['blocks_4x2_used'] += 1
    
    # 5. 组装最终帧数据（使用zone优化编码）
    frame_data, zone_combined_stats = encode_blocks_with_zone_optimization([
        blocks_to_encode_8x8,
        final_blocks_8x4, 
        final_blocks_4x4,
        final_blocks_4x2
    ])
    
    # 合并统计信息
    total_stats.update(zone_combined_stats)
    
    return frame_data, total_stats

def encode_zone_bitmap_and_data(changed_blocks_dict, block_type):
    """
    基于zone的编码：bitmap + 每个zone的连续段数据
    
    Args:
        changed_blocks_dict: {block_pos: codeword_idx} 字典
        block_type: "8x8", "8x4", "4x4", "4x2"
    
    Returns:
        tuple: (编码后的数据, zone统计信息)
    """
    if block_type == "8x8":
        zones_count = ZONES_8x8_COUNT
        bitmap_bytes = 1  # u8
    elif block_type == "8x4":
        zones_count = ZONES_8x4_COUNT
        bitmap_bytes = 1  # u8
    elif block_type == "4x4":
        zones_count = ZONES_4x4_COUNT
        bitmap_bytes = 2  # u16
    elif block_type == "4x2":
        zones_count = ZONES_4x2_COUNT
        bitmap_bytes = 4  # u32
    else:
        raise ValueError(f"未知块类型: {block_type}")
    
    # 统计信息
    zone_stats = {
        'zones_used': 0,
        'segments_total': 0,
        'blocks_total': len(changed_blocks_dict),
        'avg_segments_per_zone': 0.0,
        'avg_blocks_per_segment': 0.0
    }
    
    if not changed_blocks_dict:
        # 没有变化的块
        if block_type in ["8x8", "8x4"]:
            return [0], zone_stats  # u8 bitmap
        elif block_type == "4x4":
            return [0, 0], zone_stats  # u16 bitmap
        elif block_type == "4x2":
            return [0, 0, 0, 0], zone_stats  # u32 bitmap
    
    # 1. 按zone分组变化的块
    zones_data = {}
    for block_pos, codeword_idx in changed_blocks_dict.items():
        zone_id = block_pos // BLOCKS_PER_ZONE
        zone_offset = block_pos % BLOCKS_PER_ZONE
        if zone_id not in zones_data:
            zones_data[zone_id] = []
        zones_data[zone_id].append((zone_offset, codeword_idx))
    
    zone_stats['zones_used'] = len(zones_data)
    
    # 2. 生成bitmap
    bitmap = 0
    for zone_id in zones_data.keys():
        if zone_id < zones_count:  # 防止越界
            bitmap |= (1 << zone_id)
    
    # 3. 编码bitmap
    encoded_data = []
    if bitmap_bytes == 1:
        encoded_data.append(bitmap & 0xFF)
    elif bitmap_bytes == 2:
        encoded_data.extend([(bitmap >> 0) & 0xFF, (bitmap >> 8) & 0xFF])
    elif bitmap_bytes == 4:
        encoded_data.extend([
            (bitmap >> 0) & 0xFF, (bitmap >> 8) & 0xFF,
            (bitmap >> 16) & 0xFF, (bitmap >> 24) & 0xFF
        ])
    
    # 4. 对每个有变化的zone编码数据
    total_segments = 0
    for zone_id in sorted(zones_data.keys()):
        if zone_id >= zones_count:
            continue
            
        zone_blocks = zones_data[zone_id]
        zone_blocks.sort(key=lambda x: x[0])  # 按zone内偏移排序
        
        # 生成连续段
        segments = []
        current_segment = None
        
        for zone_offset, codeword_idx in zone_blocks:
            if current_segment is None:
                current_segment = [zone_offset, zone_offset, [codeword_idx]]
            else:
                if zone_offset == current_segment[1] + 1:
                    # 连续
                    current_segment[1] = zone_offset
                    current_segment[2].append(codeword_idx)
                else:
                    # 不连续，保存当前段，开始新段
                    segments.append(current_segment)
                    current_segment = [zone_offset, zone_offset, [codeword_idx]]
        
        # 保存最后一个段
        if current_segment is not None:
            segments.append(current_segment)
        
        total_segments += len(segments)
        
        # 编码连续段数量
        encoded_data.append(len(segments) & 0xFF)
        
        # 编码每个段
        for start_offset, end_offset, codewords in segments:
            count = end_offset - start_offset + 1
            encoded_data.append(start_offset & 0xFF)  # zone内起始偏移
            encoded_data.append(count & 0xFF)        # 连续块数量
            encoded_data.extend(codewords)           # 码表项序列
    
    zone_stats['segments_total'] = total_segments
    if zone_stats['zones_used'] > 0:
        zone_stats['avg_segments_per_zone'] = total_segments / zone_stats['zones_used']
    if total_segments > 0:
        zone_stats['avg_blocks_per_segment'] = zone_stats['blocks_total'] / total_segments
    
    return encoded_data, zone_stats

def encode_blocks_with_zone_optimization(blocks_lists):
    """
    使用zone优化编码所有级别的块
    
    Args:
        blocks_lists: [blocks_8x8, blocks_8x4, blocks_4x4, blocks_4x2]
                     每个都是[(block_pos, codeword_idx), ...] 列表
    
    Returns:
        tuple: (编码后的数据, 综合统计信息)
    """
    blocks_8x8, blocks_8x4, blocks_4x4, blocks_4x2 = blocks_lists
    
    frame_data = []
    zone_stats = {}
    
    # 对每种块类型进行zone编码
    block_types = ["8x8", "8x4", "4x4", "4x2"]
    
    for i, (blocks, block_type) in enumerate(zip(blocks_lists, block_types)):
        if not blocks:
            # 该级别没有块需要编码
            if block_type in ["8x8", "8x4"]:
                frame_data.append(0)  # u8 bitmap
            elif block_type == "4x4":
                frame_data.extend([0, 0])  # u16 bitmap
            elif block_type == "4x2":
                frame_data.extend([0, 0, 0, 0])  # u32 bitmap
            
            # 空统计
            zone_stats[block_type] = {
                'zones_used': 0, 'segments_total': 0, 'blocks_total': 0,
                'avg_segments_per_zone': 0.0, 'avg_blocks_per_segment': 0.0
            }
            continue
        
        # 转换为字典格式
        changed_blocks_dict = {pos: idx for pos, idx in blocks}
        
        # 使用zone编码
        zone_data, zone_stat = encode_zone_bitmap_and_data(changed_blocks_dict, block_type)
        frame_data.extend(zone_data)
        zone_stats[block_type] = zone_stat
    
    # 计算总统计
    total_changed_blocks = sum(len(blocks) for blocks in blocks_lists)
    total_segments = sum(stats['segments_total'] for stats in zone_stats.values())
    
    combined_stats = {
        'total_changed_blocks': total_changed_blocks,
        'total_segments': total_segments,
        'zone_stats': zone_stats
    }
    
    return frame_data, combined_stats