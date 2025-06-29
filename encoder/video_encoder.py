#!/usr/bin/env python3
"""
gba_encode.py  v3  ——  把视频/图片序列转成 GBA Mode3 I帧+码表数据
输出 video_data.h 和 video_data.c
使用K-means生成512项码表，每帧都是I帧
默认 5 s @ 30 fps，可用 --duration / --fps 修改
"""

import argparse, cv2, numpy as np, pathlib, textwrap
from sklearn.cluster import KMeans

WIDTH, HEIGHT = 240, 160
CODEBOOK_SIZE = 512
BLOCK_W, BLOCK_H = 4, 4
PIXELS_PER_BLOCK = BLOCK_W * BLOCK_H  # 16
BLOCKS_PER_FRAME = (WIDTH // BLOCK_W) * (HEIGHT // BLOCK_H)  # 60 * 40 = 2400

# YUV转换系数（保持原来的）
Y_COEFF  = np.array([0.28571429,  0.57142857,  0.14285714])
CB_COEFF = np.array([-0.14285714, -0.28571429,  0.42857143])
CR_COEFF = np.array([ 0.35714286, -0.28571429, -0.07142857])

def extract_yuv_blocks(frame_bgr: np.ndarray) -> np.ndarray:
    """
    把 240×160×3 BGR 转换为 YUV 4×4 块
    返回 (num_blocks, 18) 的数组，每行是一个块的数据：16Y + Cb + Cr
    """
    B = frame_bgr[:, :, 0].astype(np.float32)
    G = frame_bgr[:, :, 1].astype(np.float32)
    R = frame_bgr[:, :, 2].astype(np.float32)

    Y  = (R*Y_COEFF[0]  + G*Y_COEFF[1]  + B*Y_COEFF[2]).round()
    Cb = (R*CB_COEFF[0] + G*CB_COEFF[1] + B*CB_COEFF[2]).round()
    Cr = (R*CR_COEFF[0] + G*CR_COEFF[1] + B*CR_COEFF[2]).round()

    Y  = np.clip(Y,  0, 255).astype(np.uint8)
    Cb = np.clip(Cb, -128, 127).astype(np.int8)
    Cr = np.clip(Cr, -128, 127).astype(np.int8)

    blocks = []
    for y in range(0, HEIGHT, BLOCK_H):
        for x in range(0, WIDTH, BLOCK_W):
            # 16 Y 值
            y_block = Y[y:y+4, x:x+4].flatten()
            # 1 Cb / 1 Cr：取块的平均值
            cb_avg = Cb[y:y+4, x:x+4].mean().round().astype(np.int8)
            cr_avg = Cr[y:y+4, x:x+4].mean().round().astype(np.int8)
            
            # 组合成18字节的块
            block_data = np.concatenate([y_block, [cb_avg], [cr_avg]])
            blocks.append(block_data)
    
    return np.array(blocks)

def generate_codebook(all_blocks: np.ndarray) -> np.ndarray:
    """
    使用K-means对所有块进行聚类，生成码表
    """
    print(f"生成码表中...从 {len(all_blocks)} 个块中聚类出 {CODEBOOK_SIZE} 个码字")
    
    # 使用K-means聚类
    kmeans = KMeans(n_clusters=CODEBOOK_SIZE, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(all_blocks.astype(np.float32))
    
    # 码表就是聚类中心，转换回原来的数据类型
    codebook = kmeans.cluster_centers_
    
    # 确保Y值在0-255范围内，Cb/Cr在-128到127范围内
    codebook[:, :16] = np.clip(codebook[:, :16], 0, 255)  # Y值
    codebook[:, 16:] = np.clip(codebook[:, 16:], -128, 127)  # Cb, Cr值
    
    return codebook.round().astype(np.int16)

def encode_frame_with_codebook(blocks: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """
    使用码表对帧进行编码，返回每个块的码字索引
    """
    indices = []
    
    for block in blocks:
        # 计算与所有码字的距离
        distances = np.sum((codebook.astype(np.float32) - block.astype(np.float32)) ** 2, axis=1)
        # 找到最近的码字索引
        best_idx = np.argmin(distances)
        indices.append(best_idx)
    
    return np.array(indices, dtype=np.uint16)

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

def write_source(path_c: pathlib.Path, codebook: np.ndarray, all_frame_indices: np.ndarray):
    with path_c.open("w", encoding="utf-8") as f:
        f.write('#include "video_data.h"\n\n')
        
        # 写码表
        f.write("const signed short video_codebook[] = {\n")
        for i, codeword in enumerate(codebook):
            line = "    "
            for j, val in enumerate(codeword):
                line += f"{val:4d}"
                if j < len(codeword) - 1:
                    line += ","
            line += ","
            f.write(line + f"  // 码字 {i}\n")
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
            blocks = extract_yuv_blocks(frm)
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
