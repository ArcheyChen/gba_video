#!/usr/bin/env python3

import argparse, cv2, numpy as np, pathlib, textwrap
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from apricot import FacilityLocationSelection
import warnings
# 禁用sklearn相关的警告
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearnex")
from sklearnex import patch_sklearn
patch_sklearn()         # 只有这一句是新的
from numba import jit, prange

WIDTH, HEIGHT = 240, 160

# 多级码表配置（默认值，将被命令行参数覆盖）
DEFAULT_CODEBOOK_SIZE_8x4 = 64     # 8x4块码表大小
DEFAULT_CODEBOOK_SIZE_4x4 = 128     # 4x4块码表大小
DEFAULT_CODEBOOK_SIZE_4x2 = 512     # 4x2块码表大小
DEFAULT_COVERAGE_RADIUS_8x4 = 120.0 # 8x4块覆盖半径
DEFAULT_COVERAGE_RADIUS_4x4 = 80.0  # 4x4块覆盖半径

# 块尺寸定义
BLOCK_8x4_W, BLOCK_8x4_H = 8, 4   # 8x4块
BLOCK_4x4_W, BLOCK_4x4_H = 4, 4   # 4x4块
BLOCK_4x2_W, BLOCK_4x2_H = 4, 2   # 4x2块

PIXELS_PER_8x4_BLOCK = BLOCK_8x4_W * BLOCK_8x4_H  # 32
PIXELS_PER_4x4_BLOCK = BLOCK_4x4_W * BLOCK_4x4_H  # 16
PIXELS_PER_4x2_BLOCK = BLOCK_4x2_W * BLOCK_4x2_H  # 8

# 8x4块数量（用于I帧主编码）
BLOCKS_8x4_PER_FRAME = (WIDTH // BLOCK_8x4_W) * (HEIGHT // BLOCK_8x4_H)  # 30 * 40 = 1200
# 4x4块数量（用于I帧主编码）
BLOCKS_4x4_PER_FRAME = (WIDTH // BLOCK_4x4_W) * (HEIGHT // BLOCK_4x4_H)  # 60 * 40 = 2400
# 4x2块数量（用于细分编码）
BLOCKS_4x2_PER_FRAME = (WIDTH // BLOCK_4x2_W) * (HEIGHT // BLOCK_4x2_H)  # 60 * 80 = 4800

# 特殊标记
MARKER_8x4_BLOCK = 0xFFFE  # 标记这是8x4块的索引
MARKER_4x4_BLOCK = 0xFFFF  # 标记这是4x4块的索引

# IP帧编码参数
GOP_SIZE = 30  # GOP大小，每30帧一个I帧
I_FRAME_WEIGHT = 3  # I帧块的权重（用于K-means训练）
DIFF_THRESHOLD = 100  # 块差异阈值，超过此值认为块需要更新

# YUV转换系数（用于内部聚类）
Y_COEFF  = np.array([0.28571429,  0.57142857,  0.14285714])
CB_COEFF = np.array([-0.14285714, -0.28571429,  0.42857143])
CR_COEFF = np.array([ 0.35714286, -0.28571429, -0.07142857])

@jit(nopython=True, cache=True)
def convert_bgr_to_yuv(B, G, R):
    """
    使用JIT加速的BGR到YUV转换
    """
    Y  = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B
    return Y, Cb, Cr

@jit(nopython=True, cache=True)
def extract_blocks_from_yuv(Y, Cb, Cr, height, width, block_h, block_w):
    """
    使用JIT加速的块提取函数
    注意：这里Cb/Cr已经是uint8格式(0-255)，包含了128偏移
    """
    num_blocks_y = height // block_h
    num_blocks_x = width // block_w
    total_blocks = num_blocks_y * num_blocks_x
    
    # 24 = 8Y + 8Cb + 8Cr，全部使用uint8
    blocks = np.zeros((total_blocks, 24), dtype=np.uint8)
    
    block_idx = 0
    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            y_start = by * block_h
            x_start = bx * block_w
            
            # 提取8个Y值
            for py in range(block_h):
                for px in range(block_w):
                    blocks[block_idx, py * block_w + px] = Y[y_start + py, x_start + px]
            
            # 提取8个Cb值 (已加128偏移，范围0-255)
            for py in range(block_h):
                for px in range(block_w):
                    blocks[block_idx, 8 + py * block_w + px] = Cb[y_start + py, x_start + px]
            
            # 提取8个Cr值 (已加128偏移，范围0-255)
            for py in range(block_h):
                for px in range(block_w):
                    blocks[block_idx, 16 + py * block_w + px] = Cr[y_start + py, x_start + px]
            
            block_idx += 1
    
    return blocks

def extract_yuv444_blocks_4x2(frame_bgr: np.ndarray) -> np.ndarray:
    """
    把 240×160×3 BGR 转换为 YUV444 4×2 块
    返回 (num_blocks, 24) 的数组，每行是一个块的数据：8Y + 8Cb + 8Cr
    内部统一使用uint8格式：Y: 0-255, Cb/Cr: 0-255 (已加128偏移)
    """
    B = frame_bgr[:, :, 0].astype(np.float32)
    G = frame_bgr[:, :, 1].astype(np.float32)
    R = frame_bgr[:, :, 2].astype(np.float32)

    # 使用JIT加速的转换函数
    Y, Cb, Cr = convert_bgr_to_yuv(B, G, R)
    
    # 量化和裁剪，注意Cb/Cr加128偏移变为uint8
    Y  = np.clip(np.round(Y), 0, 255).astype(np.uint8)
    Cb = np.clip(np.round(Cb + 128), 0, 255).astype(np.uint8)  # 加128偏移：-128~127 -> 0~255
    Cr = np.clip(np.round(Cr + 128), 0, 255).astype(np.uint8)  # 加128偏移：-128~127 -> 0~255

    # 使用JIT加速的块提取
    blocks = extract_blocks_from_yuv(Y, Cb, Cr, HEIGHT, WIDTH, BLOCK_4x2_H, BLOCK_4x2_W)
    
    return blocks

def extract_yuv444_blocks_8x4(frame_bgr: np.ndarray) -> np.ndarray:
    """
    把 240×160×3 BGR 转换为 YUV444 8×4 块
    返回 (num_blocks, 96) 的数组，每行是一个块的数据：32Y + 32Cb + 32Cr
    内部统一使用uint8格式：Y: 0-255, Cb/Cr: 0-255 (已加128偏移)
    """
    B = frame_bgr[:, :, 0].astype(np.float32)
    G = frame_bgr[:, :, 1].astype(np.float32)
    R = frame_bgr[:, :, 2].astype(np.float32)

    # 使用JIT加速的转换函数
    Y, Cb, Cr = convert_bgr_to_yuv(B, G, R)
    
    # 量化和裁剪，注意Cb/Cr加128偏移变为uint8
    Y  = np.clip(np.round(Y), 0, 255).astype(np.uint8)
    Cb = np.clip(np.round(Cb + 128), 0, 255).astype(np.uint8)  # 加128偏移：-128~127 -> 0~255
    Cr = np.clip(np.round(Cr + 128), 0, 255).astype(np.uint8)  # 加128偏移：-128~127 -> 0~255

    # 使用JIT加速的块提取，8x4块需要96维数据
    blocks = extract_blocks_from_yuv_8x4(Y, Cb, Cr, HEIGHT, WIDTH, BLOCK_8x4_H, BLOCK_8x4_W)
    
    return blocks
    """
    把 240×160×3 BGR 转换为 YUV444 4×4 块
    返回 (num_blocks, 48) 的数组，每行是一个块的数据：16Y + 16Cb + 16Cr
    内部统一使用uint8格式：Y: 0-255, Cb/Cr: 0-255 (已加128偏移)
    """
    B = frame_bgr[:, :, 0].astype(np.float32)
    G = frame_bgr[:, :, 1].astype(np.float32)
    R = frame_bgr[:, :, 2].astype(np.float32)

    # 使用JIT加速的转换函数
    Y, Cb, Cr = convert_bgr_to_yuv(B, G, R)
    
    # 量化和裁剪，注意Cb/Cr加128偏移变为uint8
    Y  = np.clip(np.round(Y), 0, 255).astype(np.uint8)
    Cb = np.clip(np.round(Cb + 128), 0, 255).astype(np.uint8)  # 加128偏移：-128~127 -> 0~255
    Cr = np.clip(np.round(Cr + 128), 0, 255).astype(np.uint8)  # 加128偏移：-128~127 -> 0~255

    # 使用JIT加速的块提取，4x4块需要48维数据
    blocks = extract_blocks_from_yuv_4x4(Y, Cb, Cr, HEIGHT, WIDTH, BLOCK_4x4_H, BLOCK_4x4_W)
    
    return blocks

@jit(nopython=True, cache=True)
def yuv444_to_bgr555_jit(yuv444_block):
    """
    将YUV444块直接转换为BGR555格式
    输入：YUV444块，Y: 0-255, Cb/Cr: 0-255 (含128偏移)
    输出：8个BGR555值，每个用uint16表示
    """
    # 提取YUV444数据
    y_values = yuv444_block[:8].astype(np.float32)
    cb_values = yuv444_block[8:16].astype(np.float32) - 128  # 减去偏移，范围-128~127
    cr_values = yuv444_block[16:24].astype(np.float32) - 128  # 减去偏移，范围-128~127
    
    # BGR555结果
    bgr555_values = np.zeros(8, dtype=np.uint16)
    
    for i in range(8):
        Y = y_values[i]
        Cb = cb_values[i]
        Cr = cr_values[i]
        
        # YUV到RGB转换
        R = Y + 1.402 * Cr
        G = Y - 0.344136 * Cb - 0.714136 * Cr
        B = Y + 1.772 * Cb
        
        # 裁剪到0-255范围
        R = max(0.0, min(255.0, R))
        G = max(0.0, min(255.0, G))
        B = max(0.0, min(255.0, B))
        
        # 转换到5位精度 (0-31)
        R5 = int(R * 31 / 255)
        G5 = int(G * 31 / 255)
        B5 = int(B * 31 / 255)
        
        # 打包为BGR555格式: BBBBBGGGGGRRRRR (15位)
        bgr555_values[i] = (B5 << 10) | (G5 << 5) | R5
    
    return bgr555_values

def yuv444_to_bgr555(yuv444_block: np.ndarray) -> np.ndarray:
    """
    将YUV444块转换为BGR555格式
    输入：YUV444块 (24字节)
    输出：BGR555格式 (8个uint16值)
    """
    return yuv444_to_bgr555_jit(yuv444_block)

@jit(nopython=True, cache=True)
def yuv444_to_bgr555_4x4_jit(yuv444_block):
    """
    将4x4 YUV444块直接转换为BGR555格式
    输入：YUV444块，Y: 0-255, Cb/Cr: 0-255 (含128偏移)，48字节
    输出：16个BGR555值，每个用uint16表示
    """
    # 提取YUV444数据
    y_values = yuv444_block[:16].astype(np.float32)
    cb_values = yuv444_block[16:32].astype(np.float32) - 128  # 减去偏移，范围-128~127
    cr_values = yuv444_block[32:48].astype(np.float32) - 128  # 减去偏移，范围-128~127
    
    # BGR555结果
    bgr555_values = np.zeros(16, dtype=np.uint16)
    
    for i in range(16):
        y = y_values[i]
        cb = cb_values[i]
        cr = cr_values[i]
        
        # YUV到RGB转换
        R = y + 1.402 * cr
        G = y - 0.344136 * cb - 0.714136 * cr
        B = y + 1.772 * cb
        
        # 限制在[0, 255]范围内
        R = max(0, min(255, R))
        G = max(0, min(255, G))
        B = max(0, min(255, B))
        
        # 转换为5位精度
        R5 = int(R * 31 / 255)
        G5 = int(G * 31 / 255)
        B5 = int(B * 31 / 255)
        
        # 打包为BGR555格式: BBBBBGGGGGRRRRR (15位)
        bgr555_values[i] = (B5 << 10) | (G5 << 5) | R5
    
    return bgr555_values

def yuv444_to_bgr555_4x4(yuv444_block: np.ndarray) -> np.ndarray:
    """
    将4x4 YUV444块转换为BGR555格式
    输入：YUV444块 (48字节)
    输出：BGR555格式 (16个uint16值)
    """
    return yuv444_to_bgr555_4x4_jit(yuv444_block)

@jit(nopython=True, cache=True)
def yuv444_to_bgr555_8x4_jit(yuv444_block):
    """
    将8x4 YUV444块直接转换为BGR555格式
    输入：YUV444块，Y: 0-255, Cb/Cr: 0-255 (含128偏移)，96字节
    输出：32个BGR555值，每个用uint16表示
    """
    # 提取YUV444数据
    y_values = yuv444_block[:32].astype(np.float32)
    cb_values = yuv444_block[32:64].astype(np.float32) - 128  # 减去偏移，范围-128~127
    cr_values = yuv444_block[64:96].astype(np.float32) - 128  # 减去偏移，范围-128~127
    
    # BGR555结果
    bgr555_values = np.zeros(32, dtype=np.uint16)
    
    for i in range(32):
        y = y_values[i]
        cb = cb_values[i]
        cr = cr_values[i]
        
        # YUV到RGB转换
        R = y + 1.402 * cr
        G = y - 0.344136 * cb - 0.714136 * cr
        B = y + 1.772 * cb
        
        # 限制在[0, 255]范围内
        R = max(0, min(255, R))
        G = max(0, min(255, G))
        B = max(0, min(255, B))
        
        # 转换为5位精度
        R5 = int(R * 31 / 255)
        G5 = int(G * 31 / 255)
        B5 = int(B * 31 / 255)
        
        # 打包为BGR555格式: BBBBBGGGGGRRRRR (15位)
        bgr555_values[i] = (B5 << 10) | (G5 << 5) | R5
    
    return bgr555_values

def yuv444_to_bgr555_8x4(yuv444_block: np.ndarray) -> np.ndarray:
    """
    将8x4 YUV444块转换为BGR555格式
    输入：YUV444块 (96字节)
    输出：BGR555格式 (32个uint16值)
    """
    return yuv444_to_bgr555_8x4_jit(yuv444_block)

def generate_multi_level_codebooks_for_gop(
    i_frame_blocks_8x4: np.ndarray, 
    p_frame_blocks_8x4_list: list,
    i_frame_weight: int = I_FRAME_WEIGHT,
    coverage_radius_8x4: float = DEFAULT_COVERAGE_RADIUS_8x4,
    coverage_radius_4x4: float = DEFAULT_COVERAGE_RADIUS_4x4,
    codebook_size_8x4: int = DEFAULT_CODEBOOK_SIZE_8x4,
    codebook_size_4x4: int = DEFAULT_CODEBOOK_SIZE_4x4,
    codebook_size_4x2: int = DEFAULT_CODEBOOK_SIZE_4x2
) -> tuple:
    """
    为一个GOP生成多级码表（8x4 + 4x4 + 4x2）
    
    输入：
    - i_frame_blocks_8x4: I帧的所有8x4块 (BLOCKS_8x4_PER_FRAME, 96)
    - p_frame_blocks_8x4_list: P帧的变化8x4块列表，每个元素是 (frame_idx, changed_blocks_8x4)
    - i_frame_weight: I帧块的权重
    - coverage_radius_8x4: 8x4码表的覆盖半径
    - coverage_radius_4x4: 4x4码表的覆盖半径
    - codebook_size_8x4: 8x4码表大小
    - codebook_size_4x4: 4x4码表大小
    - codebook_size_4x2: 4x2码表大小
    
    返回：(codebook_8x4, codebook_4x4, codebook_4x2)
    """
    print(f"为GOP生成三级码表...")
    print(f"I帧8x4块数: {len(i_frame_blocks_8x4)}")
    print(f"P帧变化8x4块总数: {sum(len(blocks) for _, blocks in p_frame_blocks_8x4_list)}")
    
    # 第一步：收集所有8x4块用于训练
    training_blocks_8x4 = []
    
    # 添加I帧8x4块（带权重）
    for _ in range(i_frame_weight):
        training_blocks_8x4.append(i_frame_blocks_8x4)
    
    # 添加P帧的变化8x4块
    for frame_idx, changed_blocks_8x4 in p_frame_blocks_8x4_list:
        if len(changed_blocks_8x4) > 0:
            training_blocks_8x4.append(changed_blocks_8x4)
    
    if not training_blocks_8x4:
        raise ValueError("没有足够的8x4块用于生成码表")
    
    all_training_blocks_8x4 = np.vstack(training_blocks_8x4)
    print(f"总8x4训练块数: {len(all_training_blocks_8x4)} (I帧权重x{i_frame_weight})")
    
    # 第二步：使用最大覆盖方法生成8x4码表
    print("生成8x4码表（最大覆盖方法）...")
    codebook_8x4 = generate_codebook_8x4_max_coverage(
        all_training_blocks_8x4, 
        radius=coverage_radius_8x4, 
        n_neighbors=codebook_size_8x4
    )
    
    # 第三步：找出8x4码表无法很好覆盖的块，拆分为4x4块
    print("寻找8x4码表无法覆盖的块...")
    distances_8x4 = pairwise_distances(
        all_training_blocks_8x4.astype(np.float32), 
        codebook_8x4.astype(np.float32), 
        metric="euclidean", 
        n_jobs=1
    )
    min_distances_8x4 = distances_8x4.min(axis=1)
    uncovered_8x4_mask = min_distances_8x4 > coverage_radius_8x4
    uncovered_blocks_8x4 = all_training_blocks_8x4[uncovered_8x4_mask]
    
    print(f"8x4无法覆盖的块数: {len(uncovered_blocks_8x4)} / {len(all_training_blocks_8x4)}")
    
    # 第四步：将无法覆盖的8x4块拆分为4x4块
    uncovered_blocks_4x4 = []
    for block_8x4 in uncovered_blocks_8x4:
        # 将96维的8x4块拆分为两个48维的4x4块
        # 8x4块的数据布局：32Y + 32Cb + 32Cr，每个分量都是4行8列的flatten结果
        
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
        
        uncovered_blocks_4x4.extend([left_4x4, right_4x4])
    
    uncovered_blocks_4x4 = np.array(uncovered_blocks_4x4) if uncovered_blocks_4x4 else np.zeros((0, 48), dtype=np.uint8)
    print(f"拆分得到的4x4块数: {len(uncovered_blocks_4x4)}")
    
    # 第五步：使用最大覆盖方法为4x4块生成码表
    if len(uncovered_blocks_4x4) > 0:
        print("生成4x4码表（最大覆盖方法）...")
        codebook_4x4 = generate_codebook_4x4_max_coverage(
            uncovered_blocks_4x4, 
            radius=coverage_radius_4x4, 
            n_neighbors=codebook_size_4x4
        )
        
        # 第六步：找出4x4码表无法很好覆盖的块，拆分为4x2块
        print("寻找4x4码表无法覆盖的块...")
        distances_4x4 = pairwise_distances(
            uncovered_blocks_4x4.astype(np.float32), 
            codebook_4x4.astype(np.float32), 
            metric="euclidean", 
            n_jobs=1
        )
        min_distances_4x4 = distances_4x4.min(axis=1)
        uncovered_4x4_mask = min_distances_4x4 > coverage_radius_4x4
        uncovered_blocks_4x4_for_4x2 = uncovered_blocks_4x4[uncovered_4x4_mask]
        
        print(f"4x4无法覆盖的块数: {len(uncovered_blocks_4x4_for_4x2)} / {len(uncovered_blocks_4x4)}")
        
        # 第七步：将无法覆盖的4x4块拆分为4x2块
        uncovered_blocks_4x2 = []
        for block_4x4 in uncovered_blocks_4x4_for_4x2:
            # 将48维的4x4块拆分为两个24维的4x2块
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
            uncovered_blocks_4x2.extend([upper_4x2, lower_4x2])
        
        uncovered_blocks_4x2 = np.array(uncovered_blocks_4x2) if uncovered_blocks_4x2 else np.zeros((0, 24), dtype=np.uint8)
        print(f"拆分得到的4x2块数: {len(uncovered_blocks_4x2)}")
    else:
        # 没有需要4x4码表的块，创建空码表
        print("没有需要4x4编码的块，创建空码表")
        codebook_4x4 = np.zeros((codebook_size_4x4, 48), dtype=np.uint8)
        uncovered_blocks_4x2 = np.zeros((0, 24), dtype=np.uint8)
    
    # 第八步：使用K-means为4x2块生成码表
    if len(uncovered_blocks_4x2) > 0:
        print("生成4x2码表（K-means方法）...")
        train_data_4x2 = uncovered_blocks_4x2.astype(np.float32)
        
        # 如果数据量足够，使用完整的K-means
        if len(train_data_4x2) >= codebook_size_4x2:
            warm = MiniBatchKMeans(
                n_clusters=codebook_size_4x2, 
                random_state=42, 
                n_init=20, 
                max_iter=300, 
                verbose=0
            ).fit(train_data_4x2)
            print("MiniBatchKMeans预热完成")
            
            kmeans = KMeans(
                n_clusters=codebook_size_4x2, 
                init=warm.cluster_centers_, 
                random_state=42, 
                n_init=1
            )
            kmeans.fit(train_data_4x2)
            codebook_4x2 = kmeans.cluster_centers_
        else:
            # 数据量不足时，直接用现有数据填充，剩余用零填充
            codebook_4x2 = np.zeros((codebook_size_4x2, 24), dtype=np.float32)
            codebook_4x2[:len(train_data_4x2)] = train_data_4x2
        
        codebook_4x2 = np.clip(codebook_4x2, 0, 255).round().astype(np.uint8)
    else:
        # 没有需要4x2码表的块，创建空码表
        print("没有需要4x2编码的块，创建空码表")
        codebook_4x2 = np.zeros((codebook_size_4x2, 24), dtype=np.uint8)
    
    print(f"三级码表生成完成: 8x4({len(codebook_8x4)}), 4x4({len(codebook_4x4)}), 4x2({len(codebook_4x2)})")
    return codebook_8x4, codebook_4x4, codebook_4x2
    """
    为一个GOP生成多级码表（4x4 + 4x2）
    
    输入：
    - i_frame_blocks_4x4: I帧的所有4x4块 (BLOCKS_4x4_PER_FRAME, 48)
    - p_frame_blocks_4x4_list: P帧的变化4x4块列表，每个元素是 (frame_idx, changed_blocks_4x4)
    - i_frame_weight: I帧块的权重
    - coverage_radius: 4x4码表的覆盖半径
    - codebook_size_4x4: 4x4码表大小
    - codebook_size_4x2: 4x2码表大小
    
    返回：(codebook_4x4, codebook_4x2)
    """
    print(f"为GOP生成多级码表...")
    print(f"I帧4x4块数: {len(i_frame_blocks_4x4)}")
    print(f"P帧变化4x4块总数: {sum(len(blocks) for _, blocks in p_frame_blocks_4x4_list)}")
    
    # 第一步：收集所有4x4块用于训练
    training_blocks_4x4 = []
    
    # 添加I帧4x4块（带权重）
    for _ in range(i_frame_weight):
        training_blocks_4x4.append(i_frame_blocks_4x4)
    
    # 添加P帧的变化4x4块
    for frame_idx, changed_blocks_4x4 in p_frame_blocks_4x4_list:
        if len(changed_blocks_4x4) > 0:
            training_blocks_4x4.append(changed_blocks_4x4)
    
    if not training_blocks_4x4:
        raise ValueError("没有足够的4x4块用于生成码表")
    
    all_training_blocks_4x4 = np.vstack(training_blocks_4x4)
    print(f"总4x4训练块数: {len(all_training_blocks_4x4)} (I帧权重x{i_frame_weight})")
    
    # 第二步：使用最大覆盖方法生成4x4码表
    print("生成4x4码表（最大覆盖方法）...")
    codebook_4x4 = generate_codebook_4x4_max_coverage(
        all_training_blocks_4x4, 
        radius=coverage_radius, 
        n_neighbors=codebook_size_4x4
    )
    
    # 第三步：找出4x4码表无法很好覆盖的块，拆分为4x2块
    print("寻找4x4码表无法覆盖的块...")
    distances_4x4 = pairwise_distances(
        all_training_blocks_4x4.astype(np.float32), 
        codebook_4x4.astype(np.float32), 
        metric="euclidean", 
        n_jobs=1
    )
    min_distances_4x4 = distances_4x4.min(axis=1)
    uncovered_mask = min_distances_4x4 > coverage_radius
    uncovered_blocks_4x4 = all_training_blocks_4x4[uncovered_mask]
    
    print(f"4x4无法覆盖的块数: {len(uncovered_blocks_4x4)} / {len(all_training_blocks_4x4)}")
    
    # 第四步：将无法覆盖的4x4块拆分为4x2块
    uncovered_blocks_4x2 = []
    for block_4x4 in uncovered_blocks_4x4:
        # 将48维的4x4块拆分为两个24维的4x2块
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
        uncovered_blocks_4x2.extend([upper_4x2, lower_4x2])
    
    uncovered_blocks_4x2 = np.array(uncovered_blocks_4x2) if uncovered_blocks_4x2 else np.zeros((0, 24), dtype=np.uint8)
    print(f"拆分得到的4x2块数: {len(uncovered_blocks_4x2)}")
    
    # 第五步：使用K-means为4x2块生成码表
    if len(uncovered_blocks_4x2) > 0:
        print("生成4x2码表（K-means方法）...")
        train_data_4x2 = uncovered_blocks_4x2.astype(np.float32)
        
        # 如果数据量足够，使用完整的K-means
        if len(train_data_4x2) >= codebook_size_4x2:
            warm = MiniBatchKMeans(
                n_clusters=codebook_size_4x2, 
                random_state=42, 
                n_init=20, 
                max_iter=300, 
                verbose=0
            ).fit(train_data_4x2)
            print("MiniBatchKMeans预热完成")
            kmeans = KMeans(
                n_clusters=codebook_size_4x2,
                init=warm.cluster_centers_,
                n_init=1,
                max_iter=100
            ).fit(train_data_4x2)
            codebook_4x2 = kmeans.cluster_centers_
        else:
            # 数据量不够，直接使用现有数据作为码表
            print(f"数据量不足，直接使用{len(train_data_4x2)}个块作为码表")
            if len(train_data_4x2) < codebook_size_4x2:
                # 用重复数据填充码表
                repeats = codebook_size_4x2 // len(train_data_4x2) + 1
                extended_data = np.tile(train_data_4x2, (repeats, 1))[:codebook_size_4x2]
                codebook_4x2 = extended_data
            else:
                codebook_4x2 = train_data_4x2[:codebook_size_4x2]
        
        codebook_4x2 = np.clip(codebook_4x2, 0, 255).round().astype(np.uint8)
    else:
        # 没有需要4x2码表的块，创建空码表
        print("没有需要4x2编码的块，创建空码表")
        codebook_4x2 = np.zeros((codebook_size_4x2, 24), dtype=np.uint8)
    
    print(f"多级码表生成完成: 4x4({len(codebook_4x4)}), 4x2({len(codebook_4x2)})")
    return codebook_4x4, codebook_4x2

def generate_codebook_8x4_max_coverage(blocks_8x4: np.ndarray, radius: float = 120.0, n_neighbors: int = 512) -> np.ndarray:
    """
    使用最大覆盖方法为8x4块生成码表
    """
    print(f"为8x4块生成最大覆盖码表...块数: {len(blocks_8x4)}")
    
    if len(blocks_8x4) == 0:
        return np.zeros((n_neighbors, 96), dtype=np.uint8)
    
    # 转换为float32用于距离计算
    X = blocks_8x4.astype(np.float32)
    
    # 构建稀疏相似度矩阵
    print("构建稀疏相似度矩阵...")
    S = build_sparse_similarity(X, radius=radius, n_neighbors=n_neighbors)
    density = 100 * S.nnz / (len(X) ** 2)
    print(f"稀疏矩阵密度: {density:.4f}% (nnz={S.nnz:,})")
    
    # 使用FacilityLocationSelection进行最大覆盖选择
    print("执行最大覆盖选择...")
    selector = FacilityLocationSelection(
        n_samples=n_neighbors,
        metric="precomputed",
        optimizer="lazy",
        verbose=False,
    )
    selector.fit(S)
    
    # 获取选中的码字索引
    centres_idx = selector.ranking
    codebook_8x4 = X[centres_idx]
    
    # 评估覆盖率
    dists = pairwise_distances(X, codebook_8x4, metric="euclidean", n_jobs=1)
    covered = (dists.min(axis=1) <= radius)
    covered_ratio = covered.mean()
    print(f"8x4码表覆盖率: {covered.sum():,} / {len(X):,} ({covered_ratio*100:.2f}%)")
    
    return np.clip(codebook_8x4, 0, 255).round().astype(np.uint8)

def build_sparse_similarity(X: np.ndarray, radius: float, n_neighbors: int = 128) -> csr_matrix:
    """
    构建稀疏相似度矩阵，用于最大覆盖选择
    基于欧几里得距离和硬半径
    """
    nn = NearestNeighbors(
        n_neighbors=min(n_neighbors + 1, len(X)),
        metric="euclidean",
        algorithm="auto",
        n_jobs=1,
    ).fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)

    rows, cols, data = [], [], []
    for i, (d_row, idx_row) in enumerate(zip(distances, indices)):
        for d, j in zip(d_row[1:], idx_row[1:]):  # 跳过自身
            if d <= radius:
                sim = radius - d
                rows.append(i)
                cols.append(j)
                data.append(sim)
                # 确保对称性
                rows.append(j)
                cols.append(i)
                data.append(sim)

    n = len(X)
    S = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    return S

def generate_codebook_4x4_max_coverage(blocks_4x4: np.ndarray, radius: float = 80.0, n_neighbors: int = 256) -> np.ndarray:
    """
    使用最大覆盖方法为4x4块生成码表
    """
    print(f"为4x4块生成最大覆盖码表...块数: {len(blocks_4x4)}")
    
    if len(blocks_4x4) == 0:
        return np.zeros((n_neighbors, 48), dtype=np.uint8)
    
    # 转换为float32用于距离计算
    X = blocks_4x4.astype(np.float32)
    
    # 构建稀疏相似度矩阵
    print("构建稀疏相似度矩阵...")
    S = build_sparse_similarity(X, radius=radius, n_neighbors=n_neighbors)
    density = 100 * S.nnz / (len(X) ** 2)
    print(f"稀疏矩阵密度: {density:.4f}% (nnz={S.nnz:,})")
    
    # 使用FacilityLocationSelection进行最大覆盖选择
    print("执行最大覆盖选择...")
    selector = FacilityLocationSelection(
        n_samples=n_neighbors,
        metric="precomputed",
        optimizer="lazy",
        verbose=False,
    )
    selector.fit(S)
    
    # 获取选中的码字索引
    centres_idx = selector.ranking
    codebook_4x4 = X[centres_idx]
    
    # 评估覆盖率
    dists = pairwise_distances(X, codebook_4x4, metric="euclidean", n_jobs=1)
    covered = (dists.min(axis=1) <= radius)
    covered_ratio = covered.mean()
    print(f"4x4码表覆盖率: {covered.sum():,} / {len(X):,} ({covered_ratio*100:.2f}%)")
    
    return np.clip(codebook_4x4, 0, 255).round().astype(np.uint8)

@jit(nopython=True, cache=True, parallel=True)
def compute_distances_jit(blocks, codebook):
    """
    使用JIT加速计算块到码表的最小距离索引
    """
    n_blocks = blocks.shape[0]
    n_codewords = codebook.shape[0]
    indices = np.zeros(n_blocks, dtype=np.uint16)
    
    for i in prange(n_blocks):
        min_dist = np.inf
        min_idx = 0
        for j in range(n_codewords):
            dist = 0.0
            for k in range(blocks.shape[1]):
                diff = float(blocks[i, k]) - codebook[j, k]
                dist += diff * diff
            if dist < min_dist:
                min_dist = dist
                min_idx = j
        indices[i] = min_idx
    
    return indices

def encode_frame_with_codebook(blocks: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """
    使用码表对帧进行编码，返回每个块的码字索引
    """
    # 使用JIT加速的距离计算
    indices = compute_distances_jit(blocks, codebook.astype(np.float32))
    return indices

def write_header(path_h: pathlib.Path, total_frames: int, gop_count: int, gop_size: int, codebook_size_8x4: int, codebook_size_4x4: int, codebook_size_4x2: int):
    guard = "VIDEO_DATA_H"
    with path_h.open("w", encoding="utf-8") as f:
        f.write(textwrap.dedent(f"""\
            #ifndef {guard}
            #define {guard}

            #define VIDEO_FRAME_COUNT     {total_frames}
            #define VIDEO_WIDTH           {WIDTH}
            #define VIDEO_HEIGHT          {HEIGHT}
            #define VIDEO_CODEBOOK_SIZE_8x4   {codebook_size_8x4}
            #define VIDEO_CODEBOOK_SIZE_4x4   {codebook_size_4x4}
            #define VIDEO_CODEBOOK_SIZE_4x2   {codebook_size_4x2}
            #define VIDEO_BLOCKS_8x4_PER_FRAME {BLOCKS_8x4_PER_FRAME}
            #define VIDEO_BLOCKS_4x4_PER_FRAME {BLOCKS_4x4_PER_FRAME}
            #define VIDEO_BLOCKS_4x2_PER_FRAME {BLOCKS_4x2_PER_FRAME}
            #define VIDEO_BLOCK_SIZE_8x4  32
            #define VIDEO_BLOCK_SIZE_4x4  16
            #define VIDEO_BLOCK_SIZE_4x2  8
            #define VIDEO_GOP_SIZE        {gop_size}
            #define VIDEO_GOP_COUNT       {gop_count}
            #define VIDEO_MARKER_8x4      0xFFFE
            #define VIDEO_MARKER_4x4      0xFFFF

            /* 每个GOP的8x4码表：GOP_COUNT * CODEBOOK_SIZE_8x4 * BLOCK_SIZE_8x4 个uint16 */
            extern const unsigned short video_codebooks_8x4[VIDEO_GOP_COUNT][VIDEO_CODEBOOK_SIZE_8x4][VIDEO_BLOCK_SIZE_8x4];

            /* 每个GOP的4x4码表：GOP_COUNT * CODEBOOK_SIZE_4x4 * BLOCK_SIZE_4x4 个uint16 */
            extern const unsigned short video_codebooks_4x4[VIDEO_GOP_COUNT][VIDEO_CODEBOOK_SIZE_4x4][VIDEO_BLOCK_SIZE_4x4];

            /* 每个GOP的4x2码表：GOP_COUNT * CODEBOOK_SIZE_4x2 * BLOCK_SIZE_4x2 个uint16 */
            extern const unsigned short video_codebooks_4x2[VIDEO_GOP_COUNT][VIDEO_CODEBOOK_SIZE_4x2][VIDEO_BLOCK_SIZE_4x2];

            /* 帧数据：变长编码的块索引 */
            extern const unsigned short video_frame_data[];

            /* 帧起始位置：每帧在frame_data中的起始偏移 */
            extern const unsigned int video_frame_offsets[VIDEO_FRAME_COUNT + 1];

            /* 帧类型：0=I帧，1=P帧 */
            extern const unsigned char video_frame_types[VIDEO_FRAME_COUNT];

            #endif /* {guard} */
            """))

def write_source(path_c: pathlib.Path, gop_codebooks: list, encoded_frames: list, frame_offsets: list, frame_types: list, codebook_size_8x4: int, codebook_size_4x4: int, codebook_size_4x2: int):
    with path_c.open("w", encoding="utf-8") as f:
        f.write('#include "video_data.h"\n\n')
        
        # 写入所有GOP的8x4码表（BGR555格式）
        f.write("const unsigned short video_codebooks_8x4[][VIDEO_CODEBOOK_SIZE_8x4][VIDEO_BLOCK_SIZE_8x4] = {\n")
        for gop_idx, (codebook_8x4, codebook_4x4, codebook_4x2) in enumerate(gop_codebooks):
            f.write(f"    {{ // GOP {gop_idx} - 8x4码表\n")
            for i, codeword_yuv444 in enumerate(codebook_8x4):
                # 将YUV444码字转换为BGR555格式
                codeword_bgr555 = yuv444_to_bgr555_8x4(codeword_yuv444)
                
                line = "        {"
                for j, val in enumerate(codeword_bgr555):
                    line += f"0x{val:04X}"
                    if j < len(codeword_bgr555) - 1:
                        line += ","
                line += "}"
                if i < len(codebook_8x4) - 1:
                    line += ","
                f.write(line + f"  /* 8x4码字 {i} */\n")
            f.write("    }")
            if gop_idx < len(gop_codebooks) - 1:
                f.write(",")
            f.write(f"  // GOP {gop_idx}\n")
        f.write("};\n\n")
        
        # 写入所有GOP的4x4码表（BGR555格式）
        f.write("const unsigned short video_codebooks_4x4[][VIDEO_CODEBOOK_SIZE_4x4][VIDEO_BLOCK_SIZE_4x4] = {\n")
        for gop_idx, (codebook_8x4, codebook_4x4, codebook_4x2) in enumerate(gop_codebooks):
            f.write(f"    {{ // GOP {gop_idx} - 4x4码表\n")
            for i, codeword_yuv444 in enumerate(codebook_4x4):
                # 将YUV444码字转换为BGR555格式
                codeword_bgr555 = yuv444_to_bgr555_4x4(codeword_yuv444)
                
                line = "        {"
                for j, val in enumerate(codeword_bgr555):
                    line += f"0x{val:04X}"
                    if j < len(codeword_bgr555) - 1:
                        line += ","
                line += "}"
                if i < len(codebook_4x4) - 1:
                    line += ","
                f.write(line + f"  /* 4x4码字 {i} */\n")
            f.write("    }")
            if gop_idx < len(gop_codebooks) - 1:
                f.write(",")
            f.write("\n")
        f.write("};\n\n")
        
        # 写入所有GOP的4x2码表（BGR555格式）
        f.write("const unsigned short video_codebooks_4x2[][VIDEO_CODEBOOK_SIZE_4x2][VIDEO_BLOCK_SIZE_4x2] = {\n")
        for gop_idx, (codebook_8x4, codebook_4x4, codebook_4x2) in enumerate(gop_codebooks):
            f.write(f"    {{ // GOP {gop_idx} - 4x2码表\n")
            for i, codeword_yuv444 in enumerate(codebook_4x2):
                # 将YUV444码字转换为BGR555格式
                codeword_bgr555 = yuv444_to_bgr555(codeword_yuv444)
                
                line = "        {"
                for j, val in enumerate(codeword_bgr555):
                    line += f"0x{val:04X}"
                    if j < len(codeword_bgr555) - 1:
                        line += ","
                line += "}"
                if i < len(codebook_4x2) - 1:
                    line += ","
                f.write(line + f"  /* 4x2码字 {i} */\n")
            f.write("    }")
            if gop_idx < len(gop_codebooks) - 1:
                f.write(",")
            f.write("\n")
        f.write("};\n\n")
        
        # 写入帧数据（变长编码）
        f.write("const unsigned short video_frame_data[] = {\n")
        all_data = []
        for frame_data in encoded_frames:
            all_data.extend(frame_data)
        
        per_line = 16
        for i in range(0, len(all_data), per_line):
            chunk = ', '.join(f"{val:5d}" for val in all_data[i:i+per_line])
            f.write("    " + chunk + ",\n")
        f.write("};\n\n")
        
        # 写入帧偏移表
        f.write("const unsigned int video_frame_offsets[] = {\n")
        per_line = 8
        for i in range(0, len(frame_offsets), per_line):
            chunk = ', '.join(f"{offset:8d}" for offset in frame_offsets[i:i+per_line])
            f.write("    " + chunk + ",\n")
        f.write("};\n\n")
        
        # 写入帧类型表
        f.write("const unsigned char video_frame_types[] = {\n")
        per_line = 32
        for i in range(0, len(frame_types), per_line):
            chunk = ', '.join(f"{ftype}" for ftype in frame_types[i:i+per_line])
            f.write("    " + chunk + ",\n")
        f.write("};\n")

@jit(nopython=True, cache=True)
def calculate_block_difference(block1, block2):
    """
    计算两个YUV444块之间的差异
    使用平方差之和作为差异度量
    """
    diff = 0.0
    for i in range(24):  # YUV444块有24个元素
        d = float(block1[i]) - float(block2[i])
        diff += d * d
    return diff

@jit(nopython=True, cache=True)
def calculate_block_difference_8x4(block1, block2):
    """
    计算两个8x4 YUV444块之间的差异
    使用平方差之和作为差异度量
    """
    diff = 0.0
    for i in range(96):        
        diff += (float(block1[i]) - float(block2[i])) ** 2
    return diff

@jit(nopython=True, cache=True)
def calculate_block_difference_4x4(block1, block2):
    """
    计算两个4x4 YUV444块之间的差异
    使用平方差之和作为差异度量
    """
    diff = 0.0
    for i in range(48):  # 4x4 YUV444块有48个元素
        d = float(block1[i]) - float(block2[i])
        diff += d * d
    return diff

@jit(nopython=True, cache=True)
def find_changed_blocks(current_blocks, previous_blocks, threshold):
    """
    找出相对于前一帧发生变化的块
    返回变化块的索引数组
    """
    num_blocks = current_blocks.shape[0]
    # 预分配最大可能大小的数组
    temp_indices = np.zeros(num_blocks, dtype=np.int32)
    count = 0
    
    for i in range(num_blocks):
        diff = calculate_block_difference(current_blocks[i], previous_blocks[i])
        if diff > threshold:
            temp_indices[count] = i
            count += 1
    
    # 返回实际大小的数组
    if count > 0:
        return temp_indices[:count].copy()
    else:
        return np.zeros(0, dtype=np.int32)

@jit(nopython=True, cache=True)
def find_changed_blocks_8x4(current_blocks, previous_blocks, threshold):
    """
    找出相对于前一帧发生变化的8x4块
    返回变化块的索引数组
    """
    num_blocks = current_blocks.shape[0]
    # 预分配最大可能大小的数组
    temp_indices = np.zeros(num_blocks, dtype=np.int32)
    count = 0
    
    for i in range(num_blocks):
        diff = calculate_block_difference_8x4(current_blocks[i], previous_blocks[i])
        if diff > threshold:
            temp_indices[count] = i
            count += 1
    
    # 返回实际大小的数组
    if count > 0:
        return temp_indices[:count]
    else:
        return np.zeros(0, dtype=np.int32)

@jit(nopython=True, cache=True)
def find_changed_blocks_4x4(current_blocks, previous_blocks, threshold):
    """
    找出相对于前一帧发生变化的4x4块
    返回变化块的索引数组
    """
    num_blocks = current_blocks.shape[0]
    # 预分配最大可能大小的数组
    temp_indices = np.zeros(num_blocks, dtype=np.int32)
    count = 0
    
    for i in range(num_blocks):
        diff = calculate_block_difference_4x4(current_blocks[i], previous_blocks[i])
        if diff > threshold:
            temp_indices[count] = i
            count += 1
    
    # 返回实际大小的数组
    if count > 0:
        return temp_indices[:count].copy()
    else:
        return np.zeros(0, dtype=np.int32)
@jit(nopython=True, cache=True)
def extract_blocks_from_yuv_8x4(Y, Cb, Cr, height, width, block_h, block_w):
    """
    使用JIT加速的8x4块提取函数
    注意：这里Cb/Cr已经是uint8格式(0-255)，包含了128偏移
    返回 (num_blocks, 96) 数组：32Y + 32Cb + 32Cr
    """
    num_blocks_y = height // block_h
    num_blocks_x = width // block_w
    total_blocks = num_blocks_y * num_blocks_x
    
    # 96 = 32Y + 32Cb + 32Cr，全部使用uint8
    blocks = np.zeros((total_blocks, 96), dtype=np.uint8)
    
    block_idx = 0
    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            y_start, y_end = by * block_h, (by + 1) * block_h
            x_start, x_end = bx * block_w, (bx + 1) * block_w
            
            # 提取Y分量（32个值）
            y_block = Y[y_start:y_end, x_start:x_end].flatten()
            blocks[block_idx, :32] = y_block
            
            # 提取Cb分量（32个值）
            cb_block = Cb[y_start:y_end, x_start:x_end].flatten()
            blocks[block_idx, 32:64] = cb_block
            
            # 提取Cr分量（32个值）
            cr_block = Cr[y_start:y_end, x_start:x_end].flatten()
            blocks[block_idx, 64:96] = cr_block
            
            block_idx += 1
    
    return blocks

@jit(nopython=True, cache=True)
def extract_blocks_from_yuv_4x4(Y, Cb, Cr, height, width, block_h, block_w):
    """
    使用JIT加速的4x4块提取函数
    注意：这里Cb/Cr已经是uint8格式(0-255)，包含了128偏移
    返回 (num_blocks, 48) 数组：16Y + 16Cb + 16Cr
    """
    num_blocks_y = height // block_h
    num_blocks_x = width // block_w
    total_blocks = num_blocks_y * num_blocks_x
    
    # 48 = 16Y + 16Cb + 16Cr，全部使用uint8
    blocks = np.zeros((total_blocks, 48), dtype=np.uint8)
    
    block_idx = 0
    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            y_start = by * block_h
            x_start = bx * block_w
            
            # 提取16个Y值
            for py in range(block_h):
                for px in range(block_w):
                    blocks[block_idx, py * block_w + px] = Y[y_start + py, x_start + px]
            
            # 提取16个Cb值 (已加128偏移，范围0-255)
            for py in range(block_h):
                for px in range(block_w):
                    blocks[block_idx, 16 + py * block_w + px] = Cb[y_start + py, x_start + px]
            
            # 提取16个Cr值 (已加128偏移，范围0-255)
            for py in range(block_h):
                for px in range(block_w):
                    blocks[block_idx, 32 + py * block_w + px] = Cr[y_start + py, x_start + px]
            
            block_idx += 1
    
    return blocks

def main():
    pa = argparse.ArgumentParser(description="Encode to GBA IP-Frame with Multi-Level Codebook")
    pa.add_argument("input")
    pa.add_argument("--duration", type=float, default=5.0)
    pa.add_argument("--fps",      type=int,   default=30)
    pa.add_argument("--gop-size", type=int,   default=60, help="GOP大小")
    pa.add_argument("--i-weight", type=int,   default=3, help="I帧权重")
    pa.add_argument("--diff-threshold", type=float, default=100, help="P帧块差异阈值")
    pa.add_argument("--codebook-8x4", type=int, default=DEFAULT_CODEBOOK_SIZE_8x4, help="8x4码表大小")
    pa.add_argument("--codebook-4x4", type=int, default=DEFAULT_CODEBOOK_SIZE_4x4, help="4x4码表大小")
    pa.add_argument("--codebook-4x2", type=int, default=DEFAULT_CODEBOOK_SIZE_4x2, help="4x2码表大小")
    pa.add_argument("--coverage-radius-8x4", type=float, default=DEFAULT_COVERAGE_RADIUS_8x4, help="8x4块覆盖半径")
    pa.add_argument("--coverage-radius-4x4", type=float, default=DEFAULT_COVERAGE_RADIUS_4x4, help="4x4块覆盖半径")
    pa.add_argument("--out", default="video_data")
    args = pa.parse_args()

    # 使用局部变量而不是修改全局变量
    gop_size = args.gop_size
    i_frame_weight = args.i_weight
    diff_threshold = args.diff_threshold
    codebook_size_8x4 = args.codebook_8x4
    codebook_size_4x4 = args.codebook_4x4
    codebook_size_4x2 = args.codebook_4x2
    coverage_radius_8x4 = args.coverage_radius_8x4
    coverage_radius_4x4 = args.coverage_radius_4x4

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit("❌ 打不开输入文件")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    every = int(round(src_fps / args.fps))
    grab_max = int(args.duration * src_fps)

    # 读取所有帧
    print("读取视频帧...")
    all_frame_blocks = []
    idx = 0
    while idx < grab_max:
        ret, frm = cap.read()
        if not ret:
            break
        if idx % every == 0:
            frm = cv2.resize(frm, (WIDTH, HEIGHT), cv2.INTER_AREA)
            # 提取8x4块用于主要编码
            blocks_8x4 = extract_yuv444_blocks_8x4(frm)
            all_frame_blocks.append(blocks_8x4)
        idx += 1
    cap.release()

    if not all_frame_blocks:
        raise SystemExit("❌ 没有任何帧被采样")

    total_frames = len(all_frame_blocks)
    gop_count = (total_frames + gop_size - 1) // gop_size
    print(f"总帧数: {total_frames}, GOP数量: {gop_count}, GOP大小: {gop_size}")

    # 为每个GOP生成编码数据
    gop_codebooks = []
    encoded_frames = []
    frame_offsets = [0]  # 第一帧从0开始
    frame_types = []
    current_offset = 0
    
    # 统计信息
    total_stats = {
        'blocks_8x4_used': 0,
        'blocks_4x4_used': 0,
        'blocks_4x2_used': 0,
        'i_frame_stats': {'blocks_8x4_used': 0, 'blocks_4x4_used': 0, 'blocks_4x2_used': 0},
        'p_frame_stats': {'blocks_8x4_used': 0, 'blocks_4x4_used': 0, 'blocks_4x2_used': 0}
    }

    for gop_idx in range(gop_count):
        print(f"\n处理GOP {gop_idx + 1}/{gop_count}")
        
        # 确定当前GOP的帧范围
        start_frame = gop_idx * gop_size
        end_frame = min((gop_idx + 1) * gop_size, total_frames)
        gop_frames = all_frame_blocks[start_frame:end_frame]
        
        # 第一帧是I帧
        i_frame_blocks_8x4 = gop_frames[0]
        
        # 分析P帧的变化8x4块
        p_frame_blocks_8x4_list = []
        for frame_idx in range(1, len(gop_frames)):
            current_blocks = gop_frames[frame_idx]
            previous_blocks = gop_frames[frame_idx - 1]
            changed_indices = find_changed_blocks_8x4(current_blocks, previous_blocks, diff_threshold)
            if len(changed_indices) > 0:
                changed_blocks = current_blocks[changed_indices]
                p_frame_blocks_8x4_list.append((frame_idx, changed_blocks))
        
        # 为当前GOP生成三级码表
        codebook_8x4, codebook_4x4, codebook_4x2 = generate_multi_level_codebooks_for_gop(
            i_frame_blocks_8x4, p_frame_blocks_8x4_list, i_frame_weight, 
            coverage_radius_8x4, coverage_radius_4x4, codebook_size_8x4, codebook_size_4x4, codebook_size_4x2
        )

        gop_codebooks.append((codebook_8x4, codebook_4x4, codebook_4x2))
        
        # 编码当前GOP的所有帧
        for frame_idx, frame_blocks_8x4 in enumerate(gop_frames):
            global_frame_idx = start_frame + frame_idx
            
            if frame_idx == 0:  # I帧
                frame_data, frame_stats = encode_i_frame_multi_level_8x4(frame_blocks_8x4, codebook_8x4, codebook_4x4, codebook_4x2, coverage_radius_8x4, coverage_radius_4x4)
                frame_types.append(0)  # I帧
                print(f"  I帧 {global_frame_idx}: {BLOCKS_8x4_PER_FRAME} 个8x4块 (8x4码表: {frame_stats['blocks_8x4_used']}, 4x4码表: {frame_stats['blocks_4x4_used']}, 4x2码表: {frame_stats['blocks_4x2_used']})")
                
                # 更新统计
                total_stats['blocks_8x4_used'] += frame_stats['blocks_8x4_used']
                total_stats['blocks_4x4_used'] += frame_stats['blocks_4x4_used']
                total_stats['blocks_4x2_used'] += frame_stats['blocks_4x2_used']
                total_stats['i_frame_stats']['blocks_8x4_used'] += frame_stats['blocks_8x4_used']
                total_stats['i_frame_stats']['blocks_4x4_used'] += frame_stats['blocks_4x4_used']
                total_stats['i_frame_stats']['blocks_4x2_used'] += frame_stats['blocks_4x2_used']
            else:  # P帧
                # P帧只编码变化的块
                previous_blocks = gop_frames[frame_idx - 1]
                frame_data, frame_stats = encode_p_frame_multi_level(
                    frame_blocks_8x4, previous_blocks, codebook_8x4, codebook_4x4, codebook_4x2, 
                    diff_threshold, coverage_radius_8x4, coverage_radius_4x4
                )
                frame_types.append(1)  # P帧
                # print(f"  P帧 {global_frame_idx}: 变化块 (8x4码表: {frame_stats['blocks_8x4_used']}, 4x4码表: {frame_stats['blocks_4x4_used']}, 4x2码表: {frame_stats['blocks_4x2_used']})")
                
                # 更新统计
                total_stats['blocks_8x4_used'] += frame_stats['blocks_8x4_used']
                total_stats['blocks_4x4_used'] += frame_stats['blocks_4x4_used']
                total_stats['blocks_4x2_used'] += frame_stats['blocks_4x2_used']
                total_stats['p_frame_stats']['blocks_8x4_used'] += frame_stats['blocks_8x4_used']
                total_stats['p_frame_stats']['blocks_4x4_used'] += frame_stats['blocks_4x4_used']
                total_stats['p_frame_stats']['blocks_4x2_used'] += frame_stats['blocks_4x2_used']
            
            encoded_frames.append(frame_data)
            current_offset += len(frame_data)
            frame_offsets.append(current_offset)

    # 移除最后一个多余的偏移
    frame_offsets = frame_offsets[:-1]

    # 写入文件
    write_header(pathlib.Path(args.out).with_suffix(".h"), total_frames, gop_count, gop_size, codebook_size_8x4, codebook_size_4x4, codebook_size_4x2)
    write_source(pathlib.Path(args.out).with_suffix(".c"), gop_codebooks, encoded_frames, frame_offsets, frame_types, codebook_size_8x4, codebook_size_4x4, codebook_size_4x2)

    # 详细统计信息
    print("\n" + "="*60)
    print("📊 编码统计信息")
    print("="*60)
    
    # 基本信息
    total_data_size = sum(len(frame_data) for frame_data in encoded_frames)
    i_frame_count = sum(1 for ft in frame_types if ft == 0)
    p_frame_count = sum(1 for ft in frame_types if ft == 1)
    
    print(f"总帧数: {total_frames}")
    print(f"  - I帧: {i_frame_count} 帧")
    print(f"  - P帧: {p_frame_count} 帧")
    print(f"GOP数量: {gop_count}, GOP大小: {gop_size}")
    print(f"块尺寸: 8x4({BLOCKS_8x4_PER_FRAME}), 4x4({BLOCKS_4x4_PER_FRAME}), 4x2({BLOCKS_4x2_PER_FRAME})")
    print(f"码表大小: 8x4({codebook_size_8x4}), 4x4({codebook_size_4x4}), 4x2({codebook_size_4x2})")
    print(f"覆盖半径: 8x4({coverage_radius_8x4}), 4x4({coverage_radius_4x4})")
    
    # 码表使用统计
    print(f"\n📋 码表使用统计:")
    print(f"总计:")
    print(f"  - 8x4码表使用: {total_stats['blocks_8x4_used']:,} 个8x4块")
    print(f"  - 4x4码表使用: {total_stats['blocks_4x4_used']:,} 个4x4块")
    print(f"  - 4x2码表使用: {total_stats['blocks_4x2_used']:,} 个4x2块")
    
    total_8x4_blocks = i_frame_count * BLOCKS_8x4_PER_FRAME  # I帧中所有8x4块都需要编码
    total_possible_4x4_blocks = total_8x4_blocks * 2  # 每个8x4块最多拆分为2个4x4块
    total_possible_4x2_blocks = total_possible_4x4_blocks * 2  # 每个4x4块最多拆分为2个4x2块
    
    print(f"I帧统计:")
    print(f"  - 8x4码表使用: {total_stats['i_frame_stats']['blocks_8x4_used']:,} 个8x4块")
    print(f"  - 4x4码表使用: {total_stats['i_frame_stats']['blocks_4x4_used']:,} 个4x4块")
    print(f"  - 4x2码表使用: {total_stats['i_frame_stats']['blocks_4x2_used']:,} 个4x2块")
    if total_8x4_blocks > 0:
        i_8x4_ratio = total_stats['i_frame_stats']['blocks_8x4_used'] / total_8x4_blocks * 100
        print(f"  - I帧中8x4码表覆盖率: {i_8x4_ratio:.1f}%")
    
    print(f"P帧统计:")
    print(f"  - 8x4码表使用: {total_stats['p_frame_stats']['blocks_8x4_used']:,} 个8x4块")
    print(f"  - 4x4码表使用: {total_stats['p_frame_stats']['blocks_4x4_used']:,} 个4x4块")
    print(f"  - 4x2码表使用: {total_stats['p_frame_stats']['blocks_4x2_used']:,} 个4x2块")
    
    # 计算各部分大小
    # 1. 码表大小
    codebook_8x4_size_bytes = gop_count * codebook_size_8x4 * 32 * 2  # 每个8x4码字32个uint16
    codebook_4x4_size_bytes = gop_count * codebook_size_4x4 * 16 * 2  # 每个4x4码字16个uint16
    codebook_4x2_size_bytes = gop_count * codebook_size_4x2 * 8 * 2   # 每个4x2码字8个uint16
    codebook_size_bytes = codebook_8x4_size_bytes + codebook_4x4_size_bytes + codebook_4x2_size_bytes
    
    # 2. 帧数据大小
    frame_data_size_bytes = total_data_size * 2  # 每个u16是2字节
    
    # 3. 偏移表大小
    offsets_size_bytes = len(frame_offsets) * 4  # 每个u32是4字节
    
    # 4. 帧类型表大小
    frame_types_size_bytes = len(frame_types) * 1  # 每个u8是1字节
    
    # 5. I帧和P帧数据分析
    i_frame_data_size = 0
    p_frame_data_size = 0
    
    for i, (frame_data, frame_type) in enumerate(zip(encoded_frames, frame_types)):
        if frame_type == 0:  # I帧
            i_frame_data_size += len(frame_data)
        else:  # P帧
            p_frame_data_size += len(frame_data)
    
    i_frame_data_bytes = i_frame_data_size * 2
    p_frame_data_bytes = p_frame_data_size * 2
    
    # 总文件大小
    total_file_size = codebook_size_bytes + frame_data_size_bytes + offsets_size_bytes + frame_types_size_bytes
    
    print("\n💾 内存使用分析:")
    print(f"8x4码表数据: {codebook_8x4_size_bytes:,} 字节 ({codebook_8x4_size_bytes/1024:.1f} KB)")
    print(f"4x4码表数据: {codebook_4x4_size_bytes:,} 字节 ({codebook_4x4_size_bytes/1024:.1f} KB)")
    print(f"4x2码表数据: {codebook_4x2_size_bytes:,} 字节 ({codebook_4x2_size_bytes/1024:.1f} KB)")
    print(f"总码表数据: {codebook_size_bytes:,} 字节 ({codebook_size_bytes/1024:.1f} KB)")
    print(f"I帧数据: {i_frame_data_bytes:,} 字节 ({i_frame_data_bytes/1024:.1f} KB)")
    print(f"P帧数据: {p_frame_data_bytes:,} 字节 ({p_frame_data_bytes/1024:.1f} KB)")
    print(f"偏移表: {offsets_size_bytes:,} 字节 ({offsets_size_bytes/1024:.1f} KB)")
    print(f"帧类型表: {frame_types_size_bytes:,} 字节")
    print(f"总大小: {total_file_size:,} 字节 ({total_file_size/1024:.1f} KB)")
    
    print(f"\n📈 压缩效率:")
    original_size = total_frames * WIDTH * HEIGHT * 2  # 原始BGR555大小
    compression_ratio = original_size / total_file_size
    print(f"原始大小: {original_size:,} 字节 ({original_size/1024/1024:.1f} MB)")
    print(f"压缩后大小: {total_file_size:,} 字节 ({total_file_size/1024:.1f} KB)")
    print(f"压缩比: {compression_ratio:.1f}:1 ({100/compression_ratio:.1f}%)")
    
    print(f"✅ 编码完成！输出文件: {args.out}.h, {args.out}.c")

def encode_8x4_block_recursive(
    block_8x4: np.ndarray,
    codebook_8x4: np.ndarray,
    codebook_4x4: np.ndarray, 
    codebook_4x2: np.ndarray,
    coverage_radius_8x4: float,
    coverage_radius_4x4: float
) -> tuple:
    """
    递归编码单个8x4块，严格按照8x4→4x4→4x2的分裂顺序
    
    返回: (encoding_list, stats)
    encoding_list格式:
    - 如果用8x4码表: [8x4_index]
    - 如果拆分为4x4: [MARKER_8x4_BLOCK, left_4x4_encoding..., right_4x4_encoding...]
    """
    stats = {'blocks_8x4_used': 0, 'blocks_4x4_used': 0, 'blocks_4x2_used': 0}
    
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
        
        # 递归编码左4x4和右4x4
        left_encoding, left_stats = encode_4x4_block_recursive(
            left_4x4, codebook_4x4, codebook_4x2, coverage_radius_4x4
        )
        right_encoding, right_stats = encode_4x4_block_recursive(
            right_4x4, codebook_4x4, codebook_4x2, coverage_radius_4x4
        )
        
        # 合并统计
        for key in stats:
            stats[key] = left_stats[key] + right_stats[key]
        
        # 组装编码结果
        encoding = [MARKER_8x4_BLOCK] + left_encoding + right_encoding
        return encoding, stats

def encode_4x4_block_recursive(
    block_4x4: np.ndarray,
    codebook_4x4: np.ndarray,
    codebook_4x2: np.ndarray,
    coverage_radius_4x4: float
) -> tuple:
    """
    递归编码单个4x4块，严格按照4x4→4x2的分裂顺序
    
    返回: (encoding_list, stats)
    encoding_list格式:
    - 如果用4x4码表: [4x4_index]
    - 如果拆分为4x2: [MARKER_4x4_BLOCK, upper_4x2_index, lower_4x2_index]
    """
    stats = {'blocks_8x4_used': 0, 'blocks_4x4_used': 0, 'blocks_4x2_used': 0}
    
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
        
        # 使用4x2码表编码
        upper_indices = encode_frame_with_codebook(upper_4x2.reshape(1, -1), codebook_4x2)
        lower_indices = encode_frame_with_codebook(lower_4x2.reshape(1, -1), codebook_4x2)
        
        stats['blocks_4x2_used'] = 2
        encoding = [MARKER_4x4_BLOCK, upper_indices[0], lower_indices[0]]
        return encoding, stats

def encode_i_frame_multi_level_8x4(
    frame_blocks_8x4: np.ndarray, 
    codebook_8x4: np.ndarray, 
    codebook_4x4: np.ndarray, 
    codebook_4x2: np.ndarray, 
    coverage_radius_8x4: float = 120.0,
    coverage_radius_4x4: float = 80.0
) -> tuple:
    """
    使用三级码表编码I帧 - 严格递归分裂：8x4→4x4→4x2
    
    新的编码格式：
    - 8x4块：8x4码字索引 (直接是索引)
    - 分裂为4x4块：MARKER_8x4_BLOCK, 左半4x4编码..., 右半4x4编码...
    - 分裂为4x2块：MARKER_4x4_BLOCK, 上半4x2码字索引, 下半4x2码字索引
    
    返回格式：([总块数, 块1编码, 块2编码, ...], stats)
    
    stats格式：{
        'blocks_8x4_used': 使用8x4码表的块数,
        'blocks_4x4_used': 使用4x4码表的块数(以4x4块为单位),
        'blocks_4x2_used': 使用4x2码表的块数(以4x2块为单位)
    }
    """
    frame_data = [BLOCKS_8x4_PER_FRAME]  # 总块数
    
    # 统计信息
    total_stats = {
        'blocks_8x4_used': 0,
        'blocks_4x4_used': 0,
        'blocks_4x2_used': 0
    }
    
    # 逐个递归编码每个8x4块
    for block_idx in range(len(frame_blocks_8x4)):
        block_8x4 = frame_blocks_8x4[block_idx]
        
        # 递归编码当前8x4块
        encoding, stats = encode_8x4_block_recursive(
            block_8x4, codebook_8x4, codebook_4x4, codebook_4x2,
            coverage_radius_8x4, coverage_radius_4x4
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
    - 分裂为4x2块：MARKER_4x4_BLOCK, 上半4x2码字索引, 下半4x2码字索引
    
    返回格式：([总块数, 块1编码, 块2编码, ...], stats)
    
    stats格式：{
        'blocks_4x4_used': 使用4x4码表的块数,
        'blocks_4x2_used': 使用4x2码表的块数(以4x2块为单位)
    }
    """
    frame_data = [BLOCKS_4x4_PER_FRAME]  # 总块数
    
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
            frame_data.extend([MARKER_4x4_BLOCK, upper_indices[0], lower_indices[0]])
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
        return [0], total_stats  # 变化块数=0
    
    frame_data = [len(changed_indices_8x4)]  # 变化块数
    
    # 逐个递归编码变化的8x4块
    for block_pos in changed_indices_8x4:
        block_8x4 = current_blocks_8x4[block_pos]
        
        # 递归编码当前8x4块
        encoding, stats = encode_8x4_block_recursive(
            block_8x4, codebook_8x4, codebook_4x4, codebook_4x2,
            coverage_radius_8x4, coverage_radius_4x4
        )
        
        # P帧格式：位置 + 编码
        frame_data.append(block_pos)
        frame_data.extend(encoding)
        
        # 累加统计
        for key in total_stats:
            total_stats[key] += stats[key]
    
    return frame_data, total_stats

if __name__ == "__main__":
    main()