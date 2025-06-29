#!/usr/bin/env python3

import argparse, cv2, numpy as np, pathlib, textwrap
from sklearn.cluster import KMeans
from numba import jit, prange

WIDTH, HEIGHT = 240, 160
CODEBOOK_SIZE = 1024
BLOCK_W, BLOCK_H = 4, 4
PIXELS_PER_BLOCK = BLOCK_W * BLOCK_H  # 16
BLOCKS_PER_FRAME = (WIDTH // BLOCK_W) * (HEIGHT // BLOCK_H)  # 60 * 40 = 2400

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
    result[:16] = y_values.astype(np.int16)  # Y值直接复制
    result[16] = np.int16(cb_avg)            # Cb已减去128偏移
    result[17] = np.int16(cr_avg)            # Cr已减去128偏移
    
    return result

def yuv444_to_yuv9(yuv444_block: np.ndarray) -> np.ndarray:
    """
    将YUV444块(16Y + 16Cb + 16Cr = 48字节)转换为YUV9格式(16Y + 1Cb + 1Cr = 18字节)
    输入：YUV444块，所有分量都是uint8 (Cb/Cr含128偏移: 0-255)
    输出：YUV9格式，Y: 0-255, Cb/Cr: -128~127 (已减去128偏移)
    """
    return yuv444_to_yuv9_jit(yuv444_block)

def generate_codebook(all_blocks: np.ndarray) -> np.ndarray:
    """
    使用K-means对所有YUV444块进行聚类，生成码表
    输入：YUV444块，所有分量都是uint8 (Cb/Cr含128偏移: 0-255)
    """
    print(f"生成码表中...从 {len(all_blocks)} 个块中聚类出 {CODEBOOK_SIZE} 个码字")
    
    # 预热JIT函数
    print("预热JIT编译器...")
    dummy_blocks = np.random.randint(0, 255, (100, 48), dtype=np.uint8)
    dummy_codebook = np.random.randint(0, 255, (10, 48), dtype=np.uint8).astype(np.float32)
    _ = compute_distances_jit(dummy_blocks, dummy_codebook)
    
    # 使用K-means聚类
    print("开始K-means聚类...")
    kmeans = KMeans(n_clusters=CODEBOOK_SIZE, random_state=42, n_init=10, max_iter=300, verbose=0)
    kmeans.fit(all_blocks.astype(np.float32))
    
    # 码表就是聚类中心，转换回原来的数据类型
    codebook = kmeans.cluster_centers_
    
    # 确保所有值在uint8范围内 (0-255)
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

def write_header(path_h: pathlib.Path, frame_cnt: int):
    guard = "VIDEO_DATA_H"
    with path_h.open("w", encoding="utf-8") as f:
        f.write(textwrap.dedent(f"""\
            #ifndef {guard}
            #define {guard}

            #define VIDEO_FRAME_COUNT     {frame_cnt}
            #define VIDEO_WIDTH           {WIDTH}
            #define VIDEO_HEIGHT          {HEIGHT}
            #define VIDEO_CODEBOOK_SIZE   {CODEBOOK_SIZE}
            #define VIDEO_BLOCKS_PER_FRAME {BLOCKS_PER_FRAME}
            #define VIDEO_BLOCK_SIZE      18

            /* 码表：512 * 18 字节 (16Y + Cb + Cr) */
            extern const signed short video_codebook[VIDEO_CODEBOOK_SIZE * VIDEO_BLOCK_SIZE];

            /* I帧数据：每帧 2400 个 u16 索引 */
            extern const unsigned short video_frame_indices[VIDEO_FRAME_COUNT * VIDEO_BLOCKS_PER_FRAME];

            #endif /* {guard} */
            """))

def write_source(path_c: pathlib.Path, codebook_yuv444: np.ndarray, all_frame_indices: np.ndarray):
    with path_c.open("w", encoding="utf-8") as f:
        f.write('#include "video_data.h"\n\n')
        
        # 将YUV444码表转换为YUV9格式后写入
        f.write("const signed short video_codebook[] = {\n")
        for i, codeword_yuv444 in enumerate(codebook_yuv444):
            # 将YUV444码字转换为YUV9格式
            # 注意：这里会将Cb/Cr从uint8(0-255)转换为int8(-128~127)
            codeword_yuv9 = yuv444_to_yuv9(codeword_yuv444)
            
            line = "    "
            for j, val in enumerate(codeword_yuv9):
                line += f"{val:4d}"
                if j < len(codeword_yuv9) - 1:
                    line += ","
            line += ","
            f.write(line + f"  /* 码字 {i}: Y[0-15]={codeword_yuv9[0]}-{codeword_yuv9[15]}, Cb={codeword_yuv9[16]}, Cr={codeword_yuv9[17]} */\n")
        f.write("};\n\n")
        
        # 写帧索引数据
        f.write("const unsigned short video_frame_indices[] = {\n")
        per_line = 16
        for i in range(0, len(all_frame_indices), per_line):
            chunk = ', '.join(f"{idx:3d}" for idx in all_frame_indices[i:i+per_line])
            f.write("    " + chunk + ",\n")
        f.write("};\n")

def main():
    pa = argparse.ArgumentParser(description="Encode to GBA I-Frame with Codebook")
    pa.add_argument("input")
    pa.add_argument("--duration", type=float, default=5.0)
    pa.add_argument("--fps",      type=int,   default=30)
    pa.add_argument("--out", default="video_data")
    args = pa.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit("❌ 打不开输入文件")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    every = int(round(src_fps / args.fps))
    grab_max = int(args.duration * src_fps)

    all_blocks = []  # 收集所有帧的所有块用于聚类
    frame_blocks_list = []  # 保存每帧的块数据
    
    print("读取视频帧...")
    idx = 0
    while idx < grab_max:
        ret, frm = cap.read()
        if not ret:
            break
        if idx % every == 0:
            frm = cv2.resize(frm, (WIDTH, HEIGHT), cv2.INTER_AREA)
            blocks = extract_yuv444_blocks(frm)
            frame_blocks_list.append(blocks)
            all_blocks.append(blocks)
        idx += 1
    cap.release()

    if not frame_blocks_list:
        raise SystemExit("❌ 没有任何帧被采样")

    # 合并所有块用于K-means聚类
    all_blocks = np.vstack(all_blocks)
    print(f"总共收集了 {len(all_blocks)} 个块")
    
    # 生成码表
    codebook = generate_codebook(all_blocks)
    
    # 对每帧进行编码
    print("编码帧...")
    all_frame_indices = []
    for i, frame_blocks in enumerate(frame_blocks_list):
        indices = encode_frame_with_codebook(frame_blocks, codebook)
        all_frame_indices.extend(indices)
        if (i + 1) % 10 == 0:
            print(f"已编码 {i + 1}/{len(frame_blocks_list)} 帧")
    
    all_frame_indices = np.array(all_frame_indices, dtype=np.uint16)

    write_header(pathlib.Path(args.out).with_suffix(".h"), len(frame_blocks_list))
    write_source(pathlib.Path(args.out).with_suffix(".c"), codebook, all_frame_indices)

    print(f"✅ 完成：{len(frame_blocks_list)} 帧, 码表大小 {CODEBOOK_SIZE}, 每帧 {BLOCKS_PER_FRAME} 个块")

if __name__ == "__main__":
    main()
