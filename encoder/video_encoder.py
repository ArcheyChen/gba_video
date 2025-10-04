#!/usr/bin/env python3

import argparse, cv2, numpy as np, pathlib, textwrap
import multiprocessing as mp
from functools import partial
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearnex import patch_sklearn
patch_sklearn()         # 只有这一句是新的
from numba import jit, prange

WIDTH, HEIGHT = 240, 160
CODEBOOK_SIZE = 2048
BLOCK_W, BLOCK_H = 4, 4
PIXELS_PER_BLOCK = BLOCK_W * BLOCK_H  # 16
BLOCKS_PER_FRAME = (WIDTH // BLOCK_W) * (HEIGHT // BLOCK_H)  # 60 * 40 = 2400

# IP帧编码参数
GOP_SIZE = 30  # GOP大小，每30帧一个I帧
I_FRAME_WEIGHT = 3  # I帧块的权重（用于K-means训练）
DIFF_THRESHOLD = 100  # 块差异阈值，超过此值认为块需要更新

# YUV转换系数（保持原来的）
Y_COEFF  = np.array([0.28571429,  0.57142857,  0.14285714])
CB_COEFF = np.array([-0.14285714, -0.28571429,  0.42857143])
CR_COEFF = np.array([ 0.35714286, -0.28571429, -0.07142857])

@jit(nopython=True, cache=True)
def convert_bgr_to_yuv(B, G, R):
    """
    使用JIT加速的BGR到YUV转换
    """
    Y  = (R*0.28571429  + G*0.57142857  + B*0.14285714)
    Cb = (R*(-0.14285714) + G*(-0.28571429) + B*0.42857143)
    Cr = (R*0.35714286 + G*(-0.28571429) + B*(-0.07142857))
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

def extract_yuv444_blocks(frame_bgr: np.ndarray) -> np.ndarray:
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

    # 使用JIT加速的块提取
    blocks = extract_blocks_from_yuv(Y, Cb, Cr, HEIGHT, WIDTH, BLOCK_H, BLOCK_W)
    
    return blocks

@jit(nopython=True, cache=True)
def yuv444_to_yuv9_jit(yuv444_block):
    """
    使用JIT加速的YUV444到YUV9转换
    输入：YUV444块，Y: 0-255, Cb/Cr: 0-255 (含128偏移)
    输出：YUV9格式，Y: 0-255, Cb/Cr: -128~127 (已减去128偏移)
    """
    # 提取YUV444数据
    y_values = yuv444_block[:16]  # 16个Y值保持不变
    cb_values = yuv444_block[16:32].astype(np.float32)  # 16个Cb值 (0-255)
    cr_values = yuv444_block[32:48].astype(np.float32)  # 16个Cr值 (0-255)
    
    # 计算Cb和Cr的平均值，然后减去128偏移
    cb_avg = np.round(np.mean(cb_values)) - 128  # 转回 -128~127 范围
    cr_avg = np.round(np.mean(cr_values)) - 128  # 转回 -128~127 范围
    
    # 手动实现clip功能，确保范围正确
    if cb_avg < -128:
        cb_avg = -128
    elif cb_avg > 127:
        cb_avg = 127
    
    if cr_avg < -128:
        cr_avg = -128
    elif cr_avg > 127:
        cr_avg = 127
    
    # 返回YUV9格式：16Y + 1Cb + 1Cr
    result = np.zeros(18, dtype=np.int16)
    result[0] = np.int16(cb_avg)            # Cb已减去128偏移
    result[1] = np.int16(cr_avg)            # Cr已减去128偏移
    result[2:18] = y_values.astype(np.int16)  # Y值直接复制
    
    return result

def yuv444_to_yuv9(yuv444_block: np.ndarray) -> np.ndarray:
    """
    将YUV444块(16Y + 16Cb + 16Cr = 48字节)转换为YUV9格式(16Y + 1Cb + 1Cr = 18字节)
    输入：YUV444块，所有分量都是uint8 (Cb/Cr含128偏移: 0-255)
    输出：YUV9格式，Y: 0-255, Cb/Cr: -128~127 (已减去128偏移)
    """
    return yuv444_to_yuv9_jit(yuv444_block)

@jit(nopython=True, cache=True)
def calculate_block_mean_yuv9(yuv444_block):
    """
    计算YUV444块的YUV9均值
    输入：YUV444块 (48字节: 16Y + 16Cb + 16Cr)，Cb/Cr含128偏移
    输出：YUV9均值 (3字节: Y_mean, Cb_mean, Cr_mean)，Cb/Cr已减去128偏移
    """
    y_values = yuv444_block[:16].astype(np.float32)
    cb_values = yuv444_block[16:32].astype(np.float32)
    cr_values = yuv444_block[32:48].astype(np.float32)

    # 计算均值
    y_mean = np.round(np.mean(y_values))
    cb_mean = np.round(np.mean(cb_values)) - 128  # 减去128偏移
    cr_mean = np.round(np.mean(cr_values)) - 128

    # 裁剪到有效范围
    y_mean = max(0, min(255, y_mean))
    cb_mean = max(-128, min(127, cb_mean))
    cr_mean = max(-128, min(127, cr_mean))

    return np.array([y_mean, cb_mean, cr_mean], dtype=np.int16)

@jit(nopython=True, cache=True)
def calculate_residual_yuv9(yuv444_block, y_mean, cb_mean, cr_mean):
    """
    计算YUV444块相对于均值的残差，转换为YUV9格式
    输入：
    - yuv444_block: YUV444块 (48字节)，Cb/Cr含128偏移
    - y_mean: Y均值 (0-255)
    - cb_mean: Cb均值 (-128~127，已减去128偏移)
    - cr_mean: Cr均值 (-128~127，已减去128偏移)

    输出：YUV9残差 (18字节: Cb_residual, Cr_residual, 16×Y_residual)
    """
    y_values = yuv444_block[:16].astype(np.float32)
    cb_values = yuv444_block[16:32].astype(np.float32)
    cr_values = yuv444_block[32:48].astype(np.float32)

    # 计算Cb/Cr平均残差（YUV9格式：4x4块共享一个Cb/Cr）
    cb_block_mean = np.mean(cb_values) - 128  # 转回-128~127范围
    cr_block_mean = np.mean(cr_values) - 128

    cb_residual = cb_block_mean - cb_mean
    cr_residual = cr_block_mean - cr_mean

    # 裁剪残差到int8范围
    cb_residual = max(-128, min(127, np.round(cb_residual)))
    cr_residual = max(-128, min(127, np.round(cr_residual)))

    # 计算Y残差（16个像素）
    y_residuals = np.zeros(16, dtype=np.int16)
    for i in range(16):
        residual = y_values[i] - y_mean
        # 裁剪到int8范围
        y_residuals[i] = max(-128, min(127, np.round(residual)))

    # 组装YUV9残差：Cb, Cr, 16×Y
    result = np.zeros(18, dtype=np.int16)
    result[0] = np.int16(cb_residual)
    result[1] = np.int16(cr_residual)
    result[2:18] = y_residuals

    return result

@jit(nopython=True, cache=True)
def calculate_residual_from_yuv_buffer(yuv444_block, yuv_buffer_block):
    """
    计算YUV444块相对于YUV缓冲区块的残差
    输入：
    - yuv444_block: 当前YUV444块 (48字节)，Cb/Cr含128偏移
    - yuv_buffer_block: 参考YUV缓冲区块 (48字节)，Cb/Cr含128偏移

    输出：YUV9残差 (18字节: Cb_residual, Cr_residual, 16×Y_residual)
    """
    y_values_curr = yuv444_block[:16].astype(np.float32)
    cb_values_curr = yuv444_block[16:32].astype(np.float32)
    cr_values_curr = yuv444_block[32:48].astype(np.float32)

    y_values_ref = yuv_buffer_block[:16].astype(np.float32)
    cb_values_ref = yuv_buffer_block[16:32].astype(np.float32)
    cr_values_ref = yuv_buffer_block[32:48].astype(np.float32)

    # 计算YUV9格式的Cb/Cr残差（4x4块的平均值之差）
    cb_curr_mean = np.mean(cb_values_curr) - 128  # 转回-128~127范围
    cr_curr_mean = np.mean(cr_values_curr) - 128
    cb_ref_mean = np.mean(cb_values_ref) - 128
    cr_ref_mean = np.mean(cr_values_ref) - 128

    cb_residual = cb_curr_mean - cb_ref_mean
    cr_residual = cr_curr_mean - cr_ref_mean

    # 裁剪残差到int8范围
    cb_residual = max(-128, min(127, np.round(cb_residual)))
    cr_residual = max(-128, min(127, np.round(cr_residual)))

    # 计算Y残差（16个像素，逐像素计算）
    y_residuals = np.zeros(16, dtype=np.int16)
    for i in range(16):
        residual = y_values_curr[i] - y_values_ref[i]
        # 裁剪到int8范围
        y_residuals[i] = max(-128, min(127, np.round(residual)))

    # 组装YUV9残差：Cb, Cr, 16×Y
    result = np.zeros(18, dtype=np.int16)
    result[0] = np.int16(cb_residual)
    result[1] = np.int16(cr_residual)
    result[2:18] = y_residuals

    return result

def generate_codebook_for_gop(i_frame_residuals: np.ndarray, p_frame_residuals_list: list, i_frame_weight: int = I_FRAME_WEIGHT) -> tuple:
    """
    为一个GOP生成残差码表，I帧块有额外权重，并按使用频率排序
    输入：
    - i_frame_residuals: I帧的所有残差块 (BLOCKS_PER_FRAME, 18)，int16格式
    - p_frame_residuals_list: P帧的变化残差块列表，每个元素是 (frame_idx, changed_residuals)
    - i_frame_weight: I帧块的权重

    返回：
    - sorted_codebook: 按使用频率降序排列的残差码表 (int8格式)
    - frequency_mapping: 原始码表索引到排序后索引的映射
    """
    print(f"为GOP生成残差码表...I帧块数: {len(i_frame_residuals)}, P帧变化块总数: {sum(len(residuals) for _, residuals in p_frame_residuals_list)}")

    # 收集所有用于训练的残差块
    training_residuals = []

    # 添加I帧残差块（带权重）
    for _ in range(i_frame_weight):
        training_residuals.append(i_frame_residuals)

    # 添加P帧的变化残差块
    for frame_idx, changed_residuals in p_frame_residuals_list:
        if len(changed_residuals) > 0:
            training_residuals.append(changed_residuals)

    if not training_residuals:
        raise ValueError("没有足够的残差块用于生成码表")

    all_training_residuals = np.vstack(training_residuals)
    print(f"总训练残差块数: {len(all_training_residuals)} (I帧权重x{i_frame_weight})")


    # 使用K-means聚类（残差范围-128~127）
    print("开始K-means聚类...")
    train_data = all_training_residuals.astype(np.float32)
    warm = MiniBatchKMeans(n_clusters=CODEBOOK_SIZE, random_state=42, n_init=20, max_iter=300, verbose=0).fit(train_data)
    print("MinibatchKMeans预热完成")
    kmeans = KMeans(
        n_clusters=CODEBOOK_SIZE,
        init=warm.cluster_centers_,
        n_init=1,
        max_iter=100
    ).fit(train_data)

    # 码表就是聚类中心（残差）
    original_codebook = kmeans.cluster_centers_

    # 确保所有值在int8范围内（-128~127）
    original_codebook = np.clip(original_codebook, -128, 127).round().astype(np.int8)

    # 计算每个码字的使用频率
    print("计算码字使用频率...")

    # 统计所有训练块对应的码字使用次数
    frequencies = np.zeros(CODEBOOK_SIZE, dtype=np.int32)

    # 对I帧残差块进行编码统计 (考虑权重)
    i_indices = compute_residual_distances_jit(i_frame_residuals, original_codebook.astype(np.float32))
    for idx in i_indices:
        frequencies[idx] += i_frame_weight

    # 对P帧变化残差块进行编码统计
    for frame_idx, changed_residuals in p_frame_residuals_list:
        if len(changed_residuals) > 0:
            p_indices = compute_residual_distances_jit(changed_residuals, original_codebook.astype(np.float32))
            for idx in p_indices:
                frequencies[idx] += 1

    # 按使用频率降序排序
    sorted_indices = np.argsort(frequencies)[::-1]  # 降序
    sorted_codebook = original_codebook[sorted_indices]

    # 创建原始索引到排序后索引的映射
    frequency_mapping = np.zeros(CODEBOOK_SIZE, dtype=np.uint16)
    for new_idx, old_idx in enumerate(sorted_indices):
        frequency_mapping[old_idx] = new_idx

    print(f"码表频率排序完成，最高频率: {frequencies[sorted_indices[0]]}, 最低频率: {frequencies[sorted_indices[-1]]}")

    return sorted_codebook, frequency_mapping

@jit(nopython=True, cache=True, parallel=True)
def compute_residual_distances_jit(residuals, codebook):
    """
    使用JIT加速的残差距离计算函数，支持并行计算
    输入：residuals和codebook都是int8/int16格式（残差范围-128~127）
    """
    num_blocks = residuals.shape[0]
    num_codewords = codebook.shape[0]
    indices = np.zeros(num_blocks, dtype=np.uint16)

    for i in prange(num_blocks):
        min_dist = np.inf
        best_idx = 0

        for j in range(num_codewords):
            dist = 0.0
            for k in range(18):  # YUV9残差有18个元素
                diff = float(residuals[i, k]) - float(codebook[j, k])
                dist += diff * diff

            if dist < min_dist:
                min_dist = dist
                best_idx = j

        indices[i] = best_idx

    return indices

@jit(nopython=True, cache=True, parallel=True)
def compute_distances_jit(blocks, codebook):
    """
    使用JIT加速的距离计算函数，支持并行计算
    输入：blocks和codebook都是uint8格式 (Cb/Cr含128偏移)
    """
    num_blocks = blocks.shape[0]
    num_codewords = codebook.shape[0]
    indices = np.zeros(num_blocks, dtype=np.uint16)

    for i in prange(num_blocks):
        min_dist = np.inf
        best_idx = 0

        for j in range(num_codewords):
            dist = 0.0
            for k in range(48):  # YUV444块有48个元素
                diff = float(blocks[i, k]) - float(codebook[j, k])
                dist += diff * diff

            if dist < min_dist:
                min_dist = dist
                best_idx = j

        indices[i] = best_idx

    return indices

def encode_frame_with_codebook(blocks: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """
    使用码表对帧进行编码，返回每个块的码字索引
    """
    # 使用JIT加速的距离计算
    indices = compute_distances_jit(blocks, codebook.astype(np.float32))
    return indices

def encode_residuals_with_codebook(residuals: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """
    使用残差码表对残差进行编码，返回每个残差块的码字索引
    """
    # 使用JIT加速的残差距离计算
    indices = compute_residual_distances_jit(residuals, codebook.astype(np.float32))
    return indices

def encode_frame_with_frequency_mapping(blocks: np.ndarray, original_codebook: np.ndarray, frequency_mapping: np.ndarray) -> np.ndarray:
    """
    使用原始码表编码，然后映射到频率排序后的索引
    """
    # 先用原始码表编码
    original_indices = compute_distances_jit(blocks, original_codebook.astype(np.float32))
    # 映射到频率排序后的索引
    mapped_indices = frequency_mapping[original_indices]
    return mapped_indices

def encode_residuals_with_frequency_mapping(residuals: np.ndarray, original_codebook: np.ndarray, frequency_mapping: np.ndarray) -> np.ndarray:
    """
    使用原始残差码表编码，然后映射到频率排序后的索引
    """
    # 先用原始残差码表编码
    original_indices = compute_residual_distances_jit(residuals, original_codebook.astype(np.float32))
    # 映射到频率排序后的索引
    mapped_indices = frequency_mapping[original_indices]
    return mapped_indices

def write_header(path_h: pathlib.Path, total_frames: int, gop_count: int, gop_size: int):
    guard = "VIDEO_DATA_H"
    with path_h.open("w", encoding="utf-8") as f:
        f.write(textwrap.dedent(f"""\
            #ifndef {guard}
            #define {guard}

            #define VIDEO_FRAME_COUNT     {total_frames}
            #define VIDEO_WIDTH           {WIDTH}
            #define VIDEO_HEIGHT          {HEIGHT}
            #define VIDEO_CODEBOOK_SIZE   {CODEBOOK_SIZE}
            #define VIDEO_BLOCKS_PER_FRAME {BLOCKS_PER_FRAME}
            #define VIDEO_BLOCK_SIZE      18
            #define VIDEO_GOP_SIZE        {gop_size}
            #define VIDEO_GOP_COUNT       {gop_count}

            /* 每个GOP的码表：GOP_COUNT * CODEBOOK_SIZE * BLOCK_SIZE 字节 */
            extern const signed char video_codebooks[VIDEO_GOP_COUNT][VIDEO_CODEBOOK_SIZE][VIDEO_BLOCK_SIZE];

            /* 帧数据：变长编码的块索引 */
            extern const unsigned short video_frame_data[];

            /* 帧起始位置：每帧在frame_data中的起始偏移 */
            extern const unsigned int video_frame_offsets[VIDEO_FRAME_COUNT + 1];

            /* 帧类型：0=I帧，1=P帧 */
            extern const unsigned char video_frame_types[VIDEO_FRAME_COUNT];

            #endif /* {guard} */
            """))

def write_source(path_c: pathlib.Path, gop_codebooks: list, encoded_frames: list, frame_offsets: list, frame_types: list):
    with path_c.open("w", encoding="utf-8") as f:
        f.write('#include "video_data.h"\n\n')

        # 写入所有GOP的残差码表
        f.write("/* 残差码表 (Residual Codebook) */\n")
        f.write("const signed char video_codebooks[][VIDEO_CODEBOOK_SIZE][VIDEO_BLOCK_SIZE] = {\n")
        for gop_idx, codebook_residuals in enumerate(gop_codebooks):
            f.write(f"    {{ // GOP {gop_idx}\n")
            for i, codeword_residuals in enumerate(codebook_residuals):
                # 残差码表已经是YUV9格式（18字节）
                line = "        {"
                for j, val in enumerate(codeword_residuals):
                    # 确保在int8范围内（-128~127）
                    val = max(-128, min(127, int(val)))
                    line += f"{val:4d}"
                    if j < len(codeword_residuals) - 1:
                        line += ","
                line += "}"
                if i < len(codebook_residuals) - 1:
                    line += ","
                f.write(line + f"  /* 残差码字 {i} */\n")
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
    for i in range(48):  # YUV444块有48个元素
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
def find_consecutive_segments(indices):
    """
    将块索引列表组织为连续段
    输入：排序后的块索引数组
    输出：段列表，每个段是 (起始位置, 块数量) 的元组
    """
    if len(indices) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    
    # 预分配最大可能的段数组
    max_segments = len(indices)
    segments = np.zeros((max_segments, 2), dtype=np.int32)
    segment_count = 0
    
    current_start = indices[0]
    current_count = 1
    
    for i in range(1, len(indices)):
        if indices[i] == indices[i-1] + 1:  # 连续
            current_count += 1
        else:  # 不连续，结束当前段
            segments[segment_count, 0] = current_start
            segments[segment_count, 1] = current_count
            segment_count += 1
            
            current_start = indices[i]
            current_count = 1
    
    # 添加最后一个段
    segments[segment_count, 0] = current_start
    segments[segment_count, 1] = current_count
    segment_count += 1
    
    # 返回实际大小的数组
    return segments[:segment_count].copy()

def main():
    pa = argparse.ArgumentParser(description="Encode to GBA IP-Frame with Codebook")
    pa.add_argument("input")
    pa.add_argument("--duration", type=float, default=5.0)
    pa.add_argument("--fps",      type=int,   default=30)
    pa.add_argument("--gop-size", type=int,   default=120, help="GOP大小")
    pa.add_argument("--i-weight", type=int,   default=3, help="I帧权重")
    pa.add_argument("--diff-threshold", type=float, default=100, help="P帧块差异阈值")
    pa.add_argument("--processes", type=int, default=mp.cpu_count(), help="并行进程数量")
    pa.add_argument("--out", default="video_data")
    args = pa.parse_args()

    # 使用局部变量而不是修改全局变量
    gop_size = args.gop_size
    i_frame_weight = args.i_weight
    diff_threshold = args.diff_threshold

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
            blocks = extract_yuv444_blocks(frm)
            all_frame_blocks.append(blocks)
        idx += 1
    cap.release()

    if not all_frame_blocks:
        raise SystemExit("❌ 没有任何帧被采样")

    total_frames = len(all_frame_blocks)
    gop_count = (total_frames + gop_size - 1) // gop_size
    print(f"总帧数: {total_frames}, GOP数量: {gop_count}, GOP大小: {gop_size}")

    # 准备GOP数据用于并行处理
    gop_data_list = []
    for gop_idx in range(gop_count):
        start_frame = gop_idx * gop_size
        end_frame = min((gop_idx + 1) * gop_size, total_frames)
        gop_frames = all_frame_blocks[start_frame:end_frame]
        gop_data_list.append((gop_idx, gop_frames, gop_size, i_frame_weight, diff_threshold))
    
    # 使用多进程并行处理GOP
    processes = min(args.processes, gop_count)  # 进程数不能超过GOP数量
    print(f"\n开始并行处理 {gop_count} 个GOP，使用 {processes} 个进程")
    
    # 创建进程池
    with mp.Pool(processes=processes) as pool:
        # 并行处理所有GOP
        results = pool.map(process_single_gop, gop_data_list)
    
    # 按GOP索引排序结果
    results.sort(key=lambda x: x[0])
    
    # 收集结果
    gop_codebooks = []
    encoded_frames = []
    frame_types = []
    
    for gop_idx, gop_codebook, gop_encoded_frames, gop_frame_types in results:
        gop_codebooks.append(gop_codebook)
        encoded_frames.extend(gop_encoded_frames)
        frame_types.extend(gop_frame_types)
    
    # 重新计算帧偏移
    frame_offsets = [0]  # 第一帧从0开始
    current_offset = 0
    for frame_data in encoded_frames:
        current_offset += len(frame_data)
        frame_offsets.append(current_offset)
    
    # 移除最后一个多余的偏移
    frame_offsets = frame_offsets[:-1]

    # 写入文件
    write_header(pathlib.Path(args.out).with_suffix(".h"), total_frames, gop_count, gop_size)
    write_source(pathlib.Path(args.out).with_suffix(".c"), gop_codebooks, encoded_frames, frame_offsets, frame_types)

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
    print(f"块尺寸: {BLOCK_W}×{BLOCK_H}, 每帧块数: {BLOCKS_PER_FRAME}")
    print(f"码表大小: {CODEBOOK_SIZE}")
    
    # 计算各部分大小
    # 1. 码表大小
    codebook_size_bytes = gop_count * CODEBOOK_SIZE * 18  # 每个码字18字节
    
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
    print(f"码表数据: {codebook_size_bytes:,} 字节 ({codebook_size_bytes/1024:.1f} KB)")
    print(f"  - {gop_count} 个GOP × {CODEBOOK_SIZE} 码字 × 18 字节")
    print(f"帧数据: {frame_data_size_bytes:,} 字节 ({frame_data_size_bytes/1024:.1f} KB)")
    print(f"  - I帧数据: {i_frame_data_bytes:,} 字节 ({i_frame_data_bytes/1024:.1f} KB)")
    print(f"  - P帧数据: {p_frame_data_bytes:,} 字节 ({p_frame_data_bytes/1024:.1f} KB)")
    print(f"偏移表: {offsets_size_bytes:,} 字节 ({offsets_size_bytes/1024:.1f} KB)")
    print(f"帧类型表: {frame_types_size_bytes:,} 字节 ({frame_types_size_bytes/1024:.1f} KB)")
    print(f"总计: {total_file_size:,} 字节 ({total_file_size/1024:.1f} KB)")
    
    # 百分比分析
    print(f"\n📈 占比分析:")
    print(f"码表占比: {codebook_size_bytes/total_file_size*100:.1f}%")
    print(f"I帧占比: {i_frame_data_bytes/total_file_size*100:.1f}%")
    print(f"P帧占比: {p_frame_data_bytes/total_file_size*100:.1f}%")
    print(f"元数据占比: {(offsets_size_bytes+frame_types_size_bytes)/total_file_size*100:.1f}%")
    
    # 压缩效率分析
    raw_frame_size = WIDTH * HEIGHT * 2  # RGB555每像素2字节
    raw_video_size = raw_frame_size * total_frames
    compression_ratio = raw_video_size / total_file_size
    
    print(f"\n🗜️ 压缩效率:")
    print(f"原始视频大小: {raw_video_size:,} 字节 ({raw_video_size/1024/1024:.1f} MB)")
    print(f"压缩后大小: {total_file_size:,} 字节 ({total_file_size/1024:.1f} KB)")
    print(f"压缩比: {compression_ratio:.1f}:1")
    print(f"压缩率: {(1-total_file_size/raw_video_size)*100:.1f}%")
    
    # 平均帧大小分析
    avg_i_frame_size = i_frame_data_bytes / i_frame_count if i_frame_count > 0 else 0
    avg_p_frame_size = p_frame_data_bytes / p_frame_count if p_frame_count > 0 else 0
    
    print(f"\n📏 帧大小分析:")
    print(f"平均I帧大小: {avg_i_frame_size:.0f} 字节")
    print(f"平均P帧大小: {avg_p_frame_size:.0f} 字节")
    if avg_i_frame_size > 0 and avg_p_frame_size > 0:
        print(f"P帧相对I帧大小: {avg_p_frame_size/avg_i_frame_size*100:.1f}%")
    
    # P帧变化统计
    total_changed_blocks = 0
    max_changed_blocks = 0
    min_changed_blocks = float('inf')
    p_frames_with_changes = 0
    total_segments = 0
    max_segments = 0
    
    # 码表段使用统计 (每256个码字为一段)
    codebook_segment_usage = np.zeros(CODEBOOK_SIZE // 256, dtype=np.int64)
    
    # P帧数据结构占用统计
    p_frame_segment_count_bytes = 0     # 段数量字段占用
    p_frame_single_count_bytes = 0      # 单块数量字段占用  
    p_frame_segment_headers_bytes = 0   # 段头(偏移+长度)占用
    p_frame_segment_indices_bytes = 0   # 段中码表索引占用
    p_frame_single_blocks_bytes = 0     # 单块数据(位置+索引)占用
    
    total_multi_segments = 0  # 连续段总数(长度>=2)
    total_single_blocks = 0   # 单块总数(长度=1)
    max_multi_segments = 0
    max_single_blocks = 0
    
    for i, (frame_data, frame_type) in enumerate(zip(encoded_frames, frame_types)):
        if frame_type == 0:  # I帧
            # I帧新格式: [块数量, Y均值×2400, Cb均值×2400, Cr均值×2400, 残差索引×2400]
            if len(frame_data) > 1:
                block_count = frame_data[0]
                # 残差索引从 1 + 2400*3 开始
                residual_start_idx = 1 + block_count * 3
                for idx_offset in range(residual_start_idx, len(frame_data)):
                    codeword_idx = frame_data[idx_offset]
                    if codeword_idx < CODEBOOK_SIZE:  # 防止越界
                        segment_idx = codeword_idx // 256
                        if segment_idx < len(codebook_segment_usage):
                            codebook_segment_usage[segment_idx] += 1
        elif frame_type == 1:  # P帧
            # P帧新格式: [段数量, 段数据..., 单块数量, 单块数据...]
            # 段数据: [起始, 长度, Y均值×N, Cb均值×N, Cr均值×N, 索引×N]
            # 单块数据: [位置, Y均值, Cb均值, Cr均值, 索引]
            if len(frame_data) >= 2:
                segment_count = frame_data[0]

                p_frame_segment_count_bytes += 2  # u16段数量
                p_frame_single_count_bytes += 2   # u16单块数量

                data_offset = 1
                segment_changed_count = 0

                # 解析连续段
                for seg in range(segment_count):
                    if data_offset + 1 >= len(frame_data):
                        break
                    seg_start = frame_data[data_offset]
                    seg_count = frame_data[data_offset + 1]
                    segment_changed_count += seg_count

                    p_frame_segment_headers_bytes += 4  # 起始+长度

                    # 跳过均值数据（Y, Cb, Cr各seg_count个）
                    data_offset += 2 + seg_count * 3

                    # 统计码表索引
                    for i in range(seg_count):
                        if data_offset + i < len(frame_data):
                            codeword_idx = frame_data[data_offset + i]
                            if codeword_idx < CODEBOOK_SIZE:
                                segment_idx = codeword_idx // 256
                                if segment_idx < len(codebook_segment_usage):
                                    codebook_segment_usage[segment_idx] += 1
                            p_frame_segment_indices_bytes += 2

                    data_offset += seg_count

                total_multi_segments += segment_count
                max_multi_segments = max(max_multi_segments, segment_count)

                # 解析单块数据
                if data_offset < len(frame_data):
                    single_count = frame_data[data_offset]
                    data_offset += 1

                    # 单块格式：[位置, Y均值, Cb均值, Cr均值, 索引] = 5个u16
                    for i in range(single_count):
                        if data_offset + 4 < len(frame_data):
                            codeword_idx = frame_data[data_offset + 4]  # 第5个是索引
                            if codeword_idx < CODEBOOK_SIZE:
                                segment_idx = codeword_idx // 256
                                if segment_idx < len(codebook_segment_usage):
                                    codebook_segment_usage[segment_idx] += 1
                            p_frame_single_blocks_bytes += 10  # 5个u16 = 10字节
                            data_offset += 5

                    total_single_blocks += single_count
                    max_single_blocks = max(max_single_blocks, single_count)
                    total_changed_blocks += segment_changed_count + single_count
                else:
                    total_changed_blocks += segment_changed_count

                if segment_changed_count + single_count > 0:
                    max_changed_blocks = max(max_changed_blocks, segment_changed_count + single_count)
                    min_changed_blocks = min(min_changed_blocks, segment_changed_count + single_count)
                    p_frames_with_changes += 1
    
    if p_frames_with_changes > 0:
        avg_changed_blocks = total_changed_blocks / p_frames_with_changes
        avg_multi_segments = total_multi_segments / p_frames_with_changes if p_frames_with_changes > 0 else 0
        avg_single_blocks = total_single_blocks / p_frames_with_changes if p_frames_with_changes > 0 else 0
        
        print(f"\n🔄 P帧变化分析:")
        print(f"有变化的P帧: {p_frames_with_changes}/{p_frame_count} ({p_frames_with_changes/p_frame_count*100:.1f}%)")
        print(f"平均变化块数: {avg_changed_blocks:.1f}")
        print(f"最大变化块数: {max_changed_blocks}")
        print(f"最小变化块数: {min_changed_blocks if min_changed_blocks != float('inf') else 0}")
        print(f"平均变化率: {avg_changed_blocks/BLOCKS_PER_FRAME*100:.1f}%")
        
        print(f"\n📏 段与单块分析:")
        print(f"总连续段数: {total_multi_segments} (长度≥2)")
        print(f"总单块数: {total_single_blocks} (长度=1)")
        print(f"平均连续段数: {avg_multi_segments:.1f}")
        print(f"平均单块数: {avg_single_blocks:.1f}")
        print(f"最大连续段数: {max_multi_segments}")
        print(f"最大单块数: {max_single_blocks}")
        if avg_multi_segments > 0:
            print(f"连续段效率: 平均每段 {(total_changed_blocks - total_single_blocks)/total_multi_segments:.1f} 块")
        
        # P帧数据结构占用详细分析
        print(f"\n📊 P帧数据结构占用分析:")
        print(f"段数量字段: {p_frame_segment_count_bytes:,} 字节 ({p_frame_segment_count_bytes/1024:.1f} KB)")
        print(f"  - {p_frame_count} 个P帧 × 2 字节/帧")
        print(f"单块数量字段: {p_frame_single_count_bytes:,} 字节 ({p_frame_single_count_bytes/1024:.1f} KB)")
        print(f"  - {p_frame_count} 个P帧 × 2 字节/帧")
        print(f"段头部数据: {p_frame_segment_headers_bytes:,} 字节 ({p_frame_segment_headers_bytes/1024:.1f} KB)")
        print(f"  - {total_multi_segments} 个段 × 4 字节/段 (偏移2字节 + 长度2字节)")
        print(f"段码表索引: {p_frame_segment_indices_bytes:,} 字节 ({p_frame_segment_indices_bytes/1024:.1f} KB)")
        print(f"  - {total_changed_blocks - total_single_blocks} 个段中块 × 2 字节/索引")
        print(f"单块数据: {p_frame_single_blocks_bytes:,} 字节 ({p_frame_single_blocks_bytes/1024:.1f} KB)")
        print(f"  - {total_single_blocks} 个单块 × 4 字节/块 (位置2字节 + 索引2字节)")
        
        # P帧内部数据占比
        total_p_frame_structure_bytes = (p_frame_segment_count_bytes + p_frame_single_count_bytes + 
                                       p_frame_segment_headers_bytes + p_frame_segment_indices_bytes + 
                                       p_frame_single_blocks_bytes)
        print(f"P帧总结构数据: {total_p_frame_structure_bytes:,} 字节 ({total_p_frame_structure_bytes/1024:.1f} KB)")
        print(f"  - 段数量字段占比: {p_frame_segment_count_bytes/total_p_frame_structure_bytes*100:.1f}%")
        print(f"  - 单块数量字段占比: {p_frame_single_count_bytes/total_p_frame_structure_bytes*100:.1f}%")
        print(f"  - 段头部占比: {p_frame_segment_headers_bytes/total_p_frame_structure_bytes*100:.1f}%")
        print(f"  - 段码表索引占比: {p_frame_segment_indices_bytes/total_p_frame_structure_bytes*100:.1f}%")
        print(f"  - 单块数据占比: {p_frame_single_blocks_bytes/total_p_frame_structure_bytes*100:.1f}%")
        
        # 优化效果分析
        old_format_bytes = (p_frame_segment_count_bytes + 
                           (total_multi_segments + total_single_blocks) * 4 +  # 如果所有块都用段格式的头部开销
                           total_changed_blocks * 2)  # 码表索引
        new_format_bytes = total_p_frame_structure_bytes
        saved_bytes = old_format_bytes - new_format_bytes
        print(f"\n💡 编码优化效果:")
        print(f"旧格式(全段)估算: {old_format_bytes:,} 字节")
        print(f"新格式(段+单块): {new_format_bytes:,} 字节")
        print(f"节省空间: {saved_bytes:,} 字节 ({saved_bytes/old_format_bytes*100:.1f}%)")
        if total_single_blocks > 0:
            print(f"单块优化效果: {total_single_blocks} 个单块节省 {total_single_blocks * 2:,} 字节头部开销")
    
    # 码表段使用统计
    print(f"\n📊 码表段使用统计 (按256码字分段):")
    total_codeword_usage = np.sum(codebook_segment_usage)
    if total_codeword_usage > 0:
        print(f"总码字使用次数: {total_codeword_usage:,}")
        for seg_idx in range(len(codebook_segment_usage)):
            usage_count = codebook_segment_usage[seg_idx]
            usage_percent = (usage_count / total_codeword_usage) * 100
            start_idx = seg_idx * 256
            end_idx = min((seg_idx + 1) * 256 - 1, CODEBOOK_SIZE - 1)
            print(f"  段{seg_idx} (码字{start_idx:4d}-{end_idx:4d}): {usage_count:8,} 次 ({usage_percent:5.1f}%)")
        
        # 计算前几个段的累积占比
        cumulative = 0
        for seg_idx in range(min(4, len(codebook_segment_usage))):
            cumulative += codebook_segment_usage[seg_idx]
            cumulative_percent = (cumulative / total_codeword_usage) * 100
            print(f"  前{seg_idx+1}段累积占比: {cumulative_percent:.1f}%")
    else:
        print("  未发现码字使用数据")
    
    print(f"\n参数设置: I帧权重={i_frame_weight}, 差异阈值={diff_threshold}, 并行进程数={processes}")
    print("="*60)

def process_single_gop(gop_data):
    """
    处理单个GOP的函数，用于多进程并行
    使用残差编码
    """
    gop_idx, gop_frames, gop_size, i_frame_weight, diff_threshold = gop_data

    print(f"\n处理GOP {gop_idx + 1} (进程 {mp.current_process().pid})")

    # 创建GOP帧的副本，用于累积误差修正
    corrected_frames = [frame.copy() for frame in gop_frames]

    # 第一帧是I帧
    i_frame_blocks = corrected_frames[0]

    # 计算I帧每个块的均值
    print(f"  计算I帧块均值...")
    i_frame_means = np.array([calculate_block_mean_yuv9(block) for block in i_frame_blocks])

    # 计算I帧每个块的残差
    print(f"  计算I帧残差...")
    i_frame_residuals = []
    for i, block in enumerate(i_frame_blocks):
        y_mean, cb_mean, cr_mean = i_frame_means[i]
        residual = calculate_residual_yuv9(block, y_mean, cb_mean, cr_mean)
        i_frame_residuals.append(residual)
    i_frame_residuals = np.array(i_frame_residuals)

    # 分析P帧的变化块，使用修正后的帧进行比较
    print(f"  分析P帧变化块...")
    p_frame_residuals_list = []
    for frame_idx in range(1, len(corrected_frames)):
        current_blocks = corrected_frames[frame_idx]
        reference_blocks = corrected_frames[frame_idx - 1]  # 使用修正后的前一帧作为参考

        # 使用numba函数找出变化的块
        changed_indices = find_changed_blocks(current_blocks, reference_blocks, diff_threshold)
        if len(changed_indices) > 0:
            changed_blocks = current_blocks[changed_indices]

            # 计算每个变化块自身的均值和残差（与I帧一致的处理方式）
            changed_residuals = []
            for curr_block in changed_blocks:
                # 计算当前块的均值
                curr_y_mean, curr_cb_mean, curr_cr_mean = calculate_block_mean_yuv9(curr_block)
                # 计算当前块相对于自身均值的残差
                residual = calculate_residual_yuv9(curr_block, curr_y_mean, curr_cb_mean, curr_cr_mean)
                changed_residuals.append(residual)

            p_frame_residuals_list.append((frame_idx, np.array(changed_residuals)))

            # 关键：对于没有更新的块，用前一帧的对应块替换当前帧的块
            # 这样可以避免累积误差
            unchanged_mask = np.ones(len(current_blocks), dtype=bool)
            unchanged_mask[changed_indices] = False
            unchanged_indices = np.where(unchanged_mask)[0]

            if len(unchanged_indices) > 0:
                # 将未更新的块替换为参考帧的对应块
                corrected_frames[frame_idx][unchanged_indices] = reference_blocks[unchanged_indices]
        else:
            p_frame_residuals_list.append((frame_idx, np.array([], dtype=np.int16).reshape(0, 18)))
            # 如果没有任何块发生变化，整帧都用前一帧替换
            corrected_frames[frame_idx] = reference_blocks.copy()

    # 为当前GOP生成残差码表
    print(f"  生成残差码表...")
    sorted_codebook, frequency_mapping = generate_codebook_for_gop(i_frame_residuals, p_frame_residuals_list, i_frame_weight)

    # 重新构建原始码表用于编码（因为我们需要用原始码表进行距离计算）
    original_codebook = np.zeros_like(sorted_codebook)
    for old_idx in range(CODEBOOK_SIZE):
        new_idx = frequency_mapping[old_idx]
        original_codebook[old_idx] = sorted_codebook[new_idx]

    # 编码当前GOP的所有帧，使用修正后的帧
    print(f"  编码GOP帧...")
    gop_encoded_frames = []
    gop_frame_types = []

    for frame_idx, frame_blocks in enumerate(corrected_frames):
        if frame_idx == 0:  # I帧
            # I帧格式：[块数量, Y均值×2400, Cb均值×2400, Cr均值×2400, 残差索引×2400]
            # 编码残差索引
            residual_indices = encode_residuals_with_frequency_mapping(i_frame_residuals, original_codebook, frequency_mapping)

            # 构建I帧数据
            frame_data = [BLOCKS_PER_FRAME]

            # 添加所有块的均值（分开存储Y, Cb, Cr）
            for mean in i_frame_means:
                y_mean, cb_mean, cr_mean = mean
                frame_data.append(int(y_mean) & 0xFFFF)  # Y: 0-255

            for mean in i_frame_means:
                y_mean, cb_mean, cr_mean = mean
                # Cb: -128~127，加128转换为0~255存储（GBA侧直接使用u8）
                frame_data.append((int(cb_mean) + 128) & 0xFFFF)

            for mean in i_frame_means:
                y_mean, cb_mean, cr_mean = mean
                # Cr: -128~127，加128转换为0~255存储（GBA侧直接使用u8）
                frame_data.append((int(cr_mean) + 128) & 0xFFFF)

            # 添加残差索引
            frame_data.extend(residual_indices.tolist())

            gop_frame_types.append(0)  # I帧
            print(f"  I帧 {gop_idx * gop_size + frame_idx}: {BLOCKS_PER_FRAME} 个块, {len(frame_data)} u16")
        else:  # P帧
            # P帧只编码变化的块（基于修正后的帧比较）
            reference_blocks = corrected_frames[frame_idx - 1]
            changed_indices = find_changed_blocks(frame_blocks, reference_blocks, diff_threshold)

            if len(changed_indices) > 0:
                # 对变化的块索引进行排序（连续段算法需要排序）
                sorted_indices = np.sort(changed_indices)
                changed_blocks = frame_blocks[sorted_indices]

                # 计算每个变化块自身的均值和残差（与I帧一致）
                changed_residuals = []
                changed_means = []  # 存储当前块的均值
                for curr_block in changed_blocks:
                    # 使用当前块的均值
                    curr_y_mean, curr_cb_mean, curr_cr_mean = calculate_block_mean_yuv9(curr_block)
                    residual = calculate_residual_yuv9(curr_block, curr_y_mean, curr_cb_mean, curr_cr_mean)
                    changed_residuals.append(residual)
                    changed_means.append((curr_y_mean, curr_cb_mean, curr_cr_mean))  # 保存当前块均值

                changed_residuals = np.array(changed_residuals)
                block_codes = encode_residuals_with_frequency_mapping(changed_residuals, original_codebook, frequency_mapping)

                # 找到连续段
                all_segments = find_consecutive_segments(sorted_indices)

                # 分离连续段(长度>=2)和单块(长度=1)
                multi_segments = []  # (seg_start, seg_count, idx0) - 记录在changed_means中的起始索引
                single_blocks = []   # 长度=1的块

                code_idx = 0
                for seg_start, seg_count in all_segments:
                    if seg_count >= 2:
                        # 连续段：保存段信息和在changed_means中的起始索引
                        multi_segments.append((seg_start, seg_count, code_idx))
                    else:
                        # 单块：保存块位置、均值和码字索引
                        single_blocks.append((seg_start, changed_means[code_idx], block_codes[code_idx]))
                    code_idx += seg_count

                # P帧新格式: [段数量, 段数据..., 单块数量, 单块数据...]
                frame_data = [len(multi_segments)]

                # 添加连续段数据（分通道成片写：Y×N, Cb×N, Cr×N, 索引×N）
                for seg_start, seg_count, idx0 in multi_segments:
                    frame_data.append(seg_start)  # 段起始位置
                    frame_data.append(seg_count)  # 段块数量

                    # 先写所有Y均值
                    for i in range(seg_count):
                        y_mean, cb_mean, cr_mean = changed_means[idx0 + i]
                        frame_data.append(int(y_mean) & 0xFFFF)

                    # 再写所有Cb均值（加128转换为0~255，GBA侧直接使用u8）
                    for i in range(seg_count):
                        y_mean, cb_mean, cr_mean = changed_means[idx0 + i]
                        frame_data.append((int(cb_mean) + 128) & 0xFFFF)

                    # 再写所有Cr均值（加128转换为0~255，GBA侧直接使用u8）
                    for i in range(seg_count):
                        y_mean, cb_mean, cr_mean = changed_means[idx0 + i]
                        frame_data.append((int(cr_mean) + 128) & 0xFFFF)

                    # 最后写所有码字索引
                    for i in range(seg_count):
                        frame_data.append(int(block_codes[idx0 + i]) & 0xFFFF)

                # 添加单块数据
                frame_data.append(len(single_blocks))
                for block_pos, (y_mean, cb_mean, cr_mean), codeword_idx in single_blocks:
                    frame_data.append(block_pos)  # 块位置
                    frame_data.append(int(y_mean) & 0xFFFF)  # Y均值
                    frame_data.append((int(cb_mean) + 128) & 0xFFFF)  # Cb均值（加128转换为0~255）
                    frame_data.append((int(cr_mean) + 128) & 0xFFFF)  # Cr均值（加128转换为0~255）
                    frame_data.append(codeword_idx)  # 码字索引

                # print(f"  P帧 {gop_idx * gop_size + frame_idx}: {len(changed_indices)} 个块变化，{len(multi_segments)} 个连续段，{len(single_blocks)} 个单块")
            else:
                # 无变化的P帧: [0段, 0单块]
                frame_data = [0, 0]
                # print(f"  P帧 {gop_idx * gop_size + frame_idx}: 无变化")

            gop_frame_types.append(1)  # P帧

        gop_encoded_frames.append(frame_data)

    return gop_idx, sorted_codebook, gop_encoded_frames, gop_frame_types

if __name__ == "__main__":
    # 设置多进程启动方法（对于跨平台兼容性）
    mp.set_start_method('spawn', force=True)
    main()
