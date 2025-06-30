#!/usr/bin/env python3

import argparse, cv2, numpy as np, pathlib, textwrap
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearnex import patch_sklearn
patch_sklearn()         # 只有这一句是新的
from numba import jit, prange

WIDTH, HEIGHT = 240, 160
CODEBOOK_SIZE = 512
BLOCK_W, BLOCK_H = 4, 2
PIXELS_PER_BLOCK = BLOCK_W * BLOCK_H  # 8
BLOCKS_PER_FRAME = (WIDTH // BLOCK_W) * (HEIGHT // BLOCK_H)  # 60 * 80 = 4800

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

def extract_yuv444_blocks(frame_bgr: np.ndarray) -> np.ndarray:
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
    y_values = yuv444_block[:8]  # 8个Y值保持不变
    cb_values = yuv444_block[8:16].astype(np.float32)  # 8个Cb值 (0-255)
    cr_values = yuv444_block[16:24].astype(np.float32)  # 8个Cr值 (0-255)
    
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
    
    # 返回YUV9格式：8Y + 1Cb + 1Cr
    result = np.zeros(10, dtype=np.int16)
    result[0] = np.int16(cb_avg)            # Cb已减去128偏移
    result[1] = np.int16(cr_avg)            # Cr已减去128偏移
    result[2:10] = y_values.astype(np.int16)  # Y值直接复制
    
    return result

def yuv444_to_yuv9(yuv444_block: np.ndarray) -> np.ndarray:
    """
    将YUV444块(8Y + 8Cb + 8Cr = 24字节)转换为YUV9格式(8Y + 1Cb + 1Cr = 10字节)
    输入：YUV444块，所有分量都是uint8 (Cb/Cr含128偏移: 0-255)
    输出：YUV9格式，Y: 0-255, Cb/Cr: -128~127 (已减去128偏移)
    """
    return yuv444_to_yuv9_jit(yuv444_block)

def generate_codebook_for_gop(i_frame_blocks: np.ndarray, p_frame_blocks_list: list, i_frame_weight: int = I_FRAME_WEIGHT) -> np.ndarray:
    """
    为一个GOP生成码表，I帧块有额外权重
    输入：
    - i_frame_blocks: I帧的所有块 (BLOCKS_PER_FRAME, 24)
    - p_frame_blocks_list: P帧的变化块列表，每个元素是 (frame_idx, changed_blocks)
    - i_frame_weight: I帧块的权重
    """
    print(f"为GOP生成码表...I帧块数: {len(i_frame_blocks)}, P帧变化块总数: {sum(len(blocks) for _, blocks in p_frame_blocks_list)}")
    
    # 收集所有用于训练的块
    training_blocks = []
    
    # 添加I帧块（带权重）
    for _ in range(i_frame_weight):
        training_blocks.append(i_frame_blocks)
    
    # 添加P帧的变化块
    for frame_idx, changed_blocks in p_frame_blocks_list:
        if len(changed_blocks) > 0:
            training_blocks.append(changed_blocks)
    
    if not training_blocks:
        raise ValueError("没有足够的块用于生成码表")
    
    all_training_blocks = np.vstack(training_blocks)
    print(f"总训练块数: {len(all_training_blocks)} (I帧权重x{i_frame_weight})")
    
    
    # 使用K-means聚类
    print("开始K-means聚类...")
    train_data = all_training_blocks.astype(np.float32)
    warm = MiniBatchKMeans(n_clusters=CODEBOOK_SIZE, random_state=42, n_init=20, max_iter=300, verbose=0).fit(train_data)
    print("MinibatchKMeans预热完成")
    kmeans = KMeans(
        n_clusters=CODEBOOK_SIZE,
        init=warm.cluster_centers_,
        n_init=1,
        max_iter=100
    ).fit(train_data)
    
    # 码表就是聚类中心
    codebook = kmeans.cluster_centers_
    
    # 确保所有值在uint8范围内
    codebook = np.clip(codebook, 0, 255)
    
    return codebook.round().astype(np.uint8)

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
            for k in range(24):  # YUV444块有24个元素
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
            #define VIDEO_BLOCK_SIZE      10
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
        
        # 写入所有GOP的码表
        f.write("const signed char video_codebooks[][VIDEO_CODEBOOK_SIZE][VIDEO_BLOCK_SIZE] = {\n")
        for gop_idx, codebook_yuv444 in enumerate(gop_codebooks):
            f.write(f"    {{ // GOP {gop_idx}\n")
            for i, codeword_yuv444 in enumerate(codebook_yuv444):
                # 将YUV444码字转换为YUV9格式
                codeword_yuv9 = yuv444_to_yuv9(codeword_yuv444)
                
                line = "        {"
                for j, val in enumerate(codeword_yuv9):
                    # 确保Cb/Cr在int8范围内，Y在uint8范围内
                    if j < 2:  # Cb, Cr
                        val = max(-128, min(127, int(val)))
                    else:  # Y values
                        val = max(0, min(255, int(val)))
                    line += f"{val:4d}"
                    if j < len(codeword_yuv9) - 1:
                        line += ","
                line += "}"
                if i < len(codebook_yuv444) - 1:
                    line += ","
                f.write(line + f"  /* 码字 {i} */\n")
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

def main():
    pa = argparse.ArgumentParser(description="Encode to GBA IP-Frame with Codebook")
    pa.add_argument("input")
    pa.add_argument("--duration", type=float, default=5.0)
    pa.add_argument("--fps",      type=int,   default=30)
    pa.add_argument("--gop-size", type=int,   default=30, help="GOP大小")
    pa.add_argument("--i-weight", type=int,   default=3, help="I帧权重")
    pa.add_argument("--diff-threshold", type=float, default=100, help="P帧块差异阈值")
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

    # 为每个GOP生成编码数据
    gop_codebooks = []
    encoded_frames = []
    frame_offsets = [0]  # 第一帧从0开始
    frame_types = []
    current_offset = 0

    for gop_idx in range(gop_count):
        print(f"\n处理GOP {gop_idx + 1}/{gop_count}")
        
        # 确定当前GOP的帧范围
        start_frame = gop_idx * gop_size
        end_frame = min((gop_idx + 1) * gop_size, total_frames)
        gop_frames = all_frame_blocks[start_frame:end_frame]
        
        # 第一帧是I帧
        i_frame_blocks = gop_frames[0]
        
        # 分析P帧的变化块
        p_frame_blocks_list = []
        for frame_idx in range(1, len(gop_frames)):
            current_blocks = gop_frames[frame_idx]
            previous_blocks = gop_frames[frame_idx - 1]
            
            # 使用numba函数找出变化的块
            changed_indices = find_changed_blocks(current_blocks, previous_blocks, diff_threshold)
            if len(changed_indices) > 0:
                changed_blocks = current_blocks[changed_indices]
                p_frame_blocks_list.append((frame_idx, changed_blocks))
                # print(f"  P帧 {frame_idx}: {len(changed_indices)} 个块发生变化")
            else:
                p_frame_blocks_list.append((frame_idx, np.array([], dtype=np.uint8).reshape(0, 24)))
                # print(f"  P帧 {frame_idx}: 无变化")
        
        # 为当前GOP生成码表
        gop_codebook = generate_codebook_for_gop(i_frame_blocks, p_frame_blocks_list, i_frame_weight)
        gop_codebooks.append(gop_codebook)
        
        # 编码当前GOP的所有帧
        for frame_idx, frame_blocks in enumerate(gop_frames):
            global_frame_idx = start_frame + frame_idx
            
            if frame_idx == 0:  # I帧
                # I帧编码所有块
                indices = encode_frame_with_codebook(frame_blocks, gop_codebook)
                frame_data = [BLOCKS_PER_FRAME] + indices.tolist()  # 前缀块数量
                frame_types.append(0)  # I帧
                print(f"  I帧 {global_frame_idx}: {BLOCKS_PER_FRAME} 个块")
            else:  # P帧
                # P帧只编码变化的块
                previous_blocks = gop_frames[frame_idx - 1]
                changed_indices = find_changed_blocks(frame_blocks, previous_blocks, diff_threshold)
                
                if len(changed_indices) > 0:
                    changed_blocks = frame_blocks[changed_indices]
                    block_indices = encode_frame_with_codebook(changed_blocks, gop_codebook)
                    
                    # P帧格式: [块数量, 位置1, 码字1, 位置2, 码字2, ...]
                    frame_data = [len(changed_indices)]
                    for pos, code in zip(changed_indices, block_indices):
                        frame_data.extend([pos, code])
                else:
                    # 无变化的P帧
                    frame_data = [0]
                
                frame_types.append(1)  # P帧
                # print(f"  P帧 {global_frame_idx}: {len(changed_indices) if len(changed_indices) > 0 else 0} 个块变化")
            
            encoded_frames.append(frame_data)
            current_offset += len(frame_data)
            frame_offsets.append(current_offset)

    # 移除最后一个多余的偏移
    frame_offsets = frame_offsets[:-1]

    # 写入文件
    write_header(pathlib.Path(args.out).with_suffix(".h"), total_frames, gop_count, gop_size)
    write_source(pathlib.Path(args.out).with_suffix(".c"), gop_codebooks, encoded_frames, frame_offsets, frame_types)

    total_data_size = sum(len(frame_data) for frame_data in encoded_frames)
    print(f"\n✅ 完成：{total_frames} 帧, {gop_count} 个GOP, 总数据大小: {total_data_size} 个u16")
    print(f"I帧权重: {i_frame_weight}, 差异阈值: {diff_threshold}")

if __name__ == "__main__":
    main()
