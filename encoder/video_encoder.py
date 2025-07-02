#!/usr/bin/env python3

import argparse, cv2, numpy as np, pathlib, textwrap
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from apricot import FacilityLocationSelection
import warnings
from sklearnex import patch_sklearn
patch_sklearn()
from numba import jit, prange

# 新增：导入分离的工具模块
from block_utils import *
from codebook import *

WIDTH, HEIGHT = 240, 160

# 多级码表配置（默认值，将被命令行参数覆盖）
DEFAULT_CODEBOOK_SIZE_8x8 = 32     # 8x8块码表大小
DEFAULT_CODEBOOK_SIZE_8x4 = 64     # 8x4块码表大小
DEFAULT_CODEBOOK_SIZE_4x4 = 128     # 4x4块码表大小
DEFAULT_CODEBOOK_SIZE_4x2 = 256     # 4x2块码表大小
DEFAULT_COVERAGE_RADIUS_8x8 = 120.0  # 8x8块覆盖半径
DEFAULT_COVERAGE_RADIUS_8x4 = 80.0 # 8x4块覆盖半径
DEFAULT_COVERAGE_RADIUS_4x4 = 50.0  # 4x4块覆盖半径

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

# 特殊标记
MARKER_8x8_BLOCK = 0xFFFD  # 标记这是8x8块的分裂
MARKER_8x4_BLOCK = 0xFFFE  # 标记这是8x4块的分裂
MARKER_4x4_BLOCK = 0xFFFF  # 标记这是4x4块的分裂

# IP帧编码参数
GOP_SIZE = 30  # GOP大小，每30帧一个I帧
I_FRAME_WEIGHT = 3  # I帧块的权重（用于K-means训练）
DIFF_THRESHOLD = 100  # 块差异阈值，超过此值认为块需要更新

# YUV转换系数（用于内部聚类）
Y_COEFF  = np.array([0.28571429,  0.57142857,  0.14285714])
CB_COEFF = np.array([-0.14285714, -0.28571429,  0.42857143])
CR_COEFF = np.array([ 0.35714286, -0.28571429, -0.07142857])

def write_header(path_h: pathlib.Path, total_frames: int, gop_count: int, gop_size: int, codebook_size_8x8: int, codebook_size_8x4: int, codebook_size_4x4: int, codebook_size_4x2: int):
    guard = "VIDEO_DATA_H"
    with path_h.open("w", encoding="utf-8") as f:
        f.write(textwrap.dedent(f"""\
            #ifndef {guard}
            #define {guard}

            #define VIDEO_FRAME_COUNT     {total_frames}
            #define VIDEO_WIDTH           {WIDTH}
            #define VIDEO_HEIGHT          {HEIGHT}
            #define VIDEO_CODEBOOK_SIZE_8x8   {codebook_size_8x8}
            #define VIDEO_CODEBOOK_SIZE_8x4   {codebook_size_8x4}
            #define VIDEO_CODEBOOK_SIZE_4x4   {codebook_size_4x4}
            #define VIDEO_CODEBOOK_SIZE_4x2   {codebook_size_4x2}
            #define VIDEO_BLOCKS_8x8_PER_FRAME {BLOCKS_8x8_PER_FRAME}
            #define VIDEO_BLOCKS_8x4_PER_FRAME {BLOCKS_8x4_PER_FRAME}
            #define VIDEO_BLOCKS_4x4_PER_FRAME {BLOCKS_4x4_PER_FRAME}
            #define VIDEO_BLOCKS_4x2_PER_FRAME {BLOCKS_4x2_PER_FRAME}
            #define VIDEO_BLOCK_SIZE_8x8  64
            #define VIDEO_BLOCK_SIZE_8x4  32
            #define VIDEO_BLOCK_SIZE_4x4  16
            #define VIDEO_BLOCK_SIZE_4x2  8
            #define VIDEO_GOP_SIZE        {gop_size}
            #define VIDEO_GOP_COUNT       {gop_count}
            #define VIDEO_MARKER_8x8      0xFFFD
            #define VIDEO_MARKER_8x4      0xFFFE
            #define VIDEO_MARKER_4x4      0xFFFF

            /* 每个GOP的8x8码表：GOP_COUNT * CODEBOOK_SIZE_8x8 * BLOCK_SIZE_8x8 个uint16 */
            extern const unsigned short video_codebooks_8x8[VIDEO_GOP_COUNT][VIDEO_CODEBOOK_SIZE_8x8][VIDEO_BLOCK_SIZE_8x8];

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

def write_source(path_c: pathlib.Path, gop_codebooks: list, encoded_frames: list, frame_offsets: list, frame_types: list, codebook_size_8x8: int, codebook_size_8x4: int, codebook_size_4x4: int, codebook_size_4x2: int):
    with path_c.open("w", encoding="utf-8") as f:
        f.write('#include "video_data.h"\n\n')
        
        # 写入所有GOP的8x8码表（BGR555格式）
        f.write("const unsigned short video_codebooks_8x8[][VIDEO_CODEBOOK_SIZE_8x8][VIDEO_BLOCK_SIZE_8x8] = {\n")
        for gop_idx, (codebook_8x8, codebook_8x4, codebook_4x4, codebook_4x2) in enumerate(gop_codebooks):
            f.write(f"    {{ // GOP {gop_idx} - 8x8码表\n")
            for i, codeword_yuv444 in enumerate(codebook_8x8):
                # 将YUV444码字转换为BGR555格式
                codeword_bgr555 = yuv444_to_bgr555_8x8(codeword_yuv444)
                
                line = "        {"
                for j, val in enumerate(codeword_bgr555):
                    line += f"0x{val:04X}"
                    if j < len(codeword_bgr555) - 1:
                        line += ","
                line += "}"
                if i < len(codebook_8x8) - 1:
                    line += ","
                f.write(line + f"  /* 8x8码字 {i} */\n")
            f.write("    }")
            if gop_idx < len(gop_codebooks) - 1:
                f.write(",")
            f.write(f"  // GOP {gop_idx}\n")
        f.write("};\n\n")
        
        # 写入所有GOP的8x4码表（BGR555格式）
        f.write("const unsigned short video_codebooks_8x4[][VIDEO_CODEBOOK_SIZE_8x4][VIDEO_BLOCK_SIZE_8x4] = {\n")
        for gop_idx, (codebook_8x8, codebook_8x4, codebook_4x4, codebook_4x2) in enumerate(gop_codebooks):
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
        for gop_idx, (codebook_8x8, codebook_8x4, codebook_4x4, codebook_4x2) in enumerate(gop_codebooks):
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
        for gop_idx, (codebook_8x8, codebook_8x4, codebook_4x4, codebook_4x2) in enumerate(gop_codebooks):
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

def main():
    pa = argparse.ArgumentParser(description="Encode to GBA IP-Frame with Multi-Level Codebook")
    pa.add_argument("input")
    pa.add_argument("--duration", type=float, default=5.0)
    pa.add_argument("--fps",      type=int,   default=30)
    pa.add_argument("--gop-size", type=int,   default=60, help="GOP大小")
    pa.add_argument("--i-weight", type=int,   default=3, help="I帧权重")
    pa.add_argument("--diff-threshold", type=float, default=100, help="P帧块差异阈值")
    pa.add_argument("--codebook-8x8", type=int, default=DEFAULT_CODEBOOK_SIZE_8x8, help="8x8码表大小")
    pa.add_argument("--codebook-8x4", type=int, default=DEFAULT_CODEBOOK_SIZE_8x4, help="8x4码表大小")
    pa.add_argument("--codebook-4x4", type=int, default=DEFAULT_CODEBOOK_SIZE_4x4, help="4x4码表大小")
    pa.add_argument("--codebook-4x2", type=int, default=DEFAULT_CODEBOOK_SIZE_4x2, help="4x2码表大小")
    pa.add_argument("--coverage-radius-8x8", type=float, default=DEFAULT_COVERAGE_RADIUS_8x8, help="8x8块覆盖半径")
    pa.add_argument("--coverage-radius-8x4", type=float, default=DEFAULT_COVERAGE_RADIUS_8x4, help="8x4块覆盖半径")
    pa.add_argument("--coverage-radius-4x4", type=float, default=DEFAULT_COVERAGE_RADIUS_4x4, help="4x4块覆盖半径")
    pa.add_argument("--out", default="video_data")
    args = pa.parse_args()

    # 使用局部变量而不是修改全局变量
    gop_size = args.gop_size
    i_frame_weight = args.i_weight
    diff_threshold = args.diff_threshold
    codebook_size_8x8 = args.codebook_8x8
    codebook_size_8x4 = args.codebook_8x4
    codebook_size_4x4 = args.codebook_4x4
    codebook_size_4x2 = args.codebook_4x2
    coverage_radius_8x8 = args.coverage_radius_8x8
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
            # 提取8x8块用于主要编码
            blocks_8x8 = extract_yuv444_blocks_8x8(frm)
            all_frame_blocks.append(blocks_8x8)
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
        'blocks_8x8_used': 0,
        'blocks_8x4_used': 0,
        'blocks_4x4_used': 0,
        'blocks_4x2_used': 0,
        'i_frame_stats': {'blocks_8x8_used': 0, 'blocks_8x4_used': 0, 'blocks_4x4_used': 0, 'blocks_4x2_used': 0},
        'p_frame_stats': {'blocks_8x8_used': 0, 'blocks_8x4_used': 0, 'blocks_4x4_used': 0, 'blocks_4x2_used': 0}
    }

    for gop_idx in range(gop_count):
        print(f"\n处理GOP {gop_idx + 1}/{gop_count}")
        
        # 确定当前GOP的帧范围
        start_frame = gop_idx * gop_size
        end_frame = min((gop_idx + 1) * gop_size, total_frames)
        gop_frames = all_frame_blocks[start_frame:end_frame]
        
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

        gop_codebooks.append((codebook_8x8, codebook_8x4, codebook_4x4, codebook_4x2))
        
        # 编码当前GOP的所有帧
        for frame_idx, frame_blocks_8x8 in enumerate(gop_frames):
            global_frame_idx = start_frame + frame_idx
            
            if frame_idx == 0:  # I帧
                frame_data, frame_stats = encode_i_frame_multi_level_8x8(frame_blocks_8x8, codebook_8x8, codebook_8x4, codebook_4x4, codebook_4x2, coverage_radius_8x8, coverage_radius_8x4, coverage_radius_4x4)
                frame_types.append(0)  # I帧
                print(f"  I帧 {global_frame_idx}: {BLOCKS_8x8_PER_FRAME} 个8x8块 (8x8码表: {frame_stats['blocks_8x8_used']}, 8x4码表: {frame_stats['blocks_8x4_used']}, 4x4码表: {frame_stats['blocks_4x4_used']}, 4x2码表: {frame_stats['blocks_4x2_used']})")
                
                # 更新统计
                total_stats['blocks_8x8_used'] += frame_stats['blocks_8x8_used']
                total_stats['blocks_8x4_used'] += frame_stats['blocks_8x4_used']
                total_stats['blocks_4x4_used'] += frame_stats['blocks_4x4_used']
                total_stats['blocks_4x2_used'] += frame_stats['blocks_4x2_used']
                total_stats['i_frame_stats']['blocks_8x8_used'] += frame_stats['blocks_8x8_used']
                total_stats['i_frame_stats']['blocks_8x4_used'] += frame_stats['blocks_8x4_used']
                total_stats['i_frame_stats']['blocks_4x4_used'] += frame_stats['blocks_4x4_used']
                total_stats['i_frame_stats']['blocks_4x2_used'] += frame_stats['blocks_4x2_used']
            else:  # P帧
                # P帧只编码变化的块
                previous_blocks = gop_frames[frame_idx - 1]
                frame_data, frame_stats = encode_p_frame_multi_level_8x8(
                    frame_blocks_8x8, previous_blocks, codebook_8x8, codebook_8x4, codebook_4x4, codebook_4x2, 
                    diff_threshold, coverage_radius_8x8, coverage_radius_8x4, coverage_radius_4x4
                )
                frame_types.append(1)  # P帧
                # print(f"  P帧 {global_frame_idx}: 变化块 (8x8码表: {frame_stats['blocks_8x8_used']}, 8x4码表: {frame_stats['blocks_8x4_used']}, 4x4码表: {frame_stats['blocks_4x4_used']}, 4x2码表: {frame_stats['blocks_4x2_used']})")
                
                # 更新统计
                total_stats['blocks_8x8_used'] += frame_stats['blocks_8x8_used']
                total_stats['blocks_8x4_used'] += frame_stats['blocks_8x4_used']
                total_stats['blocks_4x4_used'] += frame_stats['blocks_4x4_used']
                total_stats['blocks_4x2_used'] += frame_stats['blocks_4x2_used']
                total_stats['p_frame_stats']['blocks_8x8_used'] += frame_stats['blocks_8x8_used']
                total_stats['p_frame_stats']['blocks_8x4_used'] += frame_stats['blocks_8x4_used']
                total_stats['p_frame_stats']['blocks_4x4_used'] += frame_stats['blocks_4x4_used']
                total_stats['p_frame_stats']['blocks_4x2_used'] += frame_stats['blocks_4x2_used']
            
            encoded_frames.append(frame_data)
            current_offset += len(frame_data)
            frame_offsets.append(current_offset)

    # 移除最后一个多余的偏移
    frame_offsets = frame_offsets[:-1]

    # 写入文件
    write_header(pathlib.Path(args.out).with_suffix(".h"), total_frames, gop_count, gop_size, codebook_size_8x8, codebook_size_8x4, codebook_size_4x4, codebook_size_4x2)
    write_source(pathlib.Path(args.out).with_suffix(".c"), gop_codebooks, encoded_frames, frame_offsets, frame_types, codebook_size_8x8, codebook_size_8x4, codebook_size_4x4, codebook_size_4x2)

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
    print(f"块尺寸: 8x8({BLOCKS_8x8_PER_FRAME}), 8x4({BLOCKS_8x4_PER_FRAME}), 4x4({BLOCKS_4x4_PER_FRAME}), 4x2({BLOCKS_4x2_PER_FRAME})")
    print(f"码表大小: 8x8({codebook_size_8x8}), 8x4({codebook_size_8x4}), 4x4({codebook_size_4x4}), 4x2({codebook_size_4x2})")
    print(f"覆盖半径: 8x8({coverage_radius_8x8}), 8x4({coverage_radius_8x4}), 4x4({coverage_radius_4x4})")
    
    # 码表使用统计
    print(f"\n📋 码表使用统计:")
    print(f"总计:")
    print(f"  - 8x8码表使用: {total_stats['blocks_8x8_used']:,} 个8x8块")
    print(f"  - 8x4码表使用: {total_stats['blocks_8x4_used']:,} 个8x4块")
    print(f"  - 4x4码表使用: {total_stats['blocks_4x4_used']:,} 个4x4块")
    print(f"  - 4x2码表使用: {total_stats['blocks_4x2_used']:,} 个4x2块")
    
    total_8x8_blocks = i_frame_count * BLOCKS_8x8_PER_FRAME  # I帧中所有8x8块都需要编码
    total_possible_8x4_blocks = total_8x8_blocks * 2  # 每个8x8块最多拆分为2个8x4块
    total_possible_4x4_blocks = total_possible_8x4_blocks * 2  # 每个8x4块最多拆分为2个4x4块
    total_possible_4x2_blocks = total_possible_4x4_blocks * 2  # 每个4x4块最多拆分为2个4x2块
    
    print(f"I帧统计:")
    print(f"  - 8x8码表使用: {total_stats['i_frame_stats']['blocks_8x8_used']:,} 个8x8块")
    print(f"  - 8x4码表使用: {total_stats['i_frame_stats']['blocks_8x4_used']:,} 个8x4块")
    print(f"  - 4x4码表使用: {total_stats['i_frame_stats']['blocks_4x4_used']:,} 个4x4块")
    print(f"  - 4x2码表使用: {total_stats['i_frame_stats']['blocks_4x2_used']:,} 个4x2块")
    if total_8x8_blocks > 0:
        i_8x8_ratio = total_stats['i_frame_stats']['blocks_8x8_used'] / total_8x8_blocks * 100
        print(f"  - I帧中8x8码表覆盖率: {i_8x8_ratio:.1f}%")
    
    print(f"P帧统计:")
    print(f"  - 8x8码表使用: {total_stats['p_frame_stats']['blocks_8x8_used']:,} 个8x8块")
    print(f"  - 8x4码表使用: {total_stats['p_frame_stats']['blocks_8x4_used']:,} 个8x4块")
    print(f"  - 4x4码表使用: {total_stats['p_frame_stats']['blocks_4x4_used']:,} 个4x4块")
    print(f"  - 4x2码表使用: {total_stats['p_frame_stats']['blocks_4x2_used']:,} 个4x2块")
    
    # 计算各部分大小
    # 1. 码表大小
    codebook_8x8_size_bytes = gop_count * codebook_size_8x8 * 64 * 2  # 每个8x8码字64个uint16
    codebook_8x4_size_bytes = gop_count * codebook_size_8x4 * 32 * 2  # 每个8x4码字32个uint16
    codebook_4x4_size_bytes = gop_count * codebook_size_4x4 * 16 * 2  # 每个4x4码字16个uint16
    codebook_4x2_size_bytes = gop_count * codebook_size_4x2 * 8 * 2   # 每个4x2码字8个uint16
    codebook_size_bytes = codebook_8x8_size_bytes + codebook_8x4_size_bytes + codebook_4x4_size_bytes + codebook_4x2_size_bytes
    
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
    print(f"8x8码表数据: {codebook_8x8_size_bytes:,} 字节 ({codebook_8x8_size_bytes/1024:.1f} KB)")
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
        
        # 使用4x2码表编码
        upper_indices = encode_frame_with_codebook(upper_4x2.reshape(1, -1), codebook_4x2)
        lower_indices = encode_frame_with_codebook(lower_4x2.reshape(1, -1), codebook_4x2)
        
        stats['blocks_4x2_used'] = 2
        encoding = [MARKER_4x4_BLOCK, upper_indices[0], lower_indices[0]]
        return encoding, stats

def encode_i_frame_multi_level_8x8(
    frame_blocks_8x8: np.ndarray, 
    codebook_8x8: np.ndarray,
    codebook_8x4: np.ndarray, 
    codebook_4x4: np.ndarray, 
    codebook_4x2: np.ndarray, 
    coverage_radius_8x8: float = 150.0,
    coverage_radius_8x4: float = 120.0,
    coverage_radius_4x4: float = 80.0
) -> tuple:
    """
    使用四级码表编码I帧 - 严格递归分裂：8x8→8x4→4x4→4x2
    
    新的编码格式：
    - 8x8块：8x8码字索引 (直接是索引)
    - 分裂为8x4块：MARKER_8x8_BLOCK, 上半8x4编码..., 下半8x4编码...
    - 分裂为4x4块：MARKER_8x4_BLOCK, 左半4x4编码..., 右半4x4编码...
    - 分裂为4x2块：MARKER_4x4_BLOCK, 上半4x2码字索引, 下半4x2码字索引
    
    返回格式：([总块数, 块1编码, 块2编码, ...], stats)
    
    stats格式：{
        'blocks_8x8_used': 使用8x8码表的块数,
        'blocks_8x4_used': 使用8x4码表的块数(以8x4块为单位),
        'blocks_4x4_used': 使用4x4码表的块数(以4x4块为单位),
        'blocks_4x2_used': 使用4x2码表的块数(以4x2块为单位)
    }
    """
    frame_data = [BLOCKS_8x8_PER_FRAME]  # 总块数
    
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
            coverage_radius_8x8, coverage_radius_8x4, coverage_radius_4x4
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
        return [0], total_stats  # 变化块数=0
    
    frame_data = [len(changed_indices_8x8)]  # 变化块数
    
    # 逐个递归编码变化的8x8块
    for block_pos in changed_indices_8x8:
        block_8x8 = current_blocks_8x8[block_pos]
        
        # 递归编码当前8x8块
        encoding, stats = encode_8x8_block_recursive(
            block_8x8, codebook_8x8, codebook_8x4, codebook_4x4, codebook_4x2,
            coverage_radius_8x8, coverage_radius_8x4, coverage_radius_4x4
        )
        
        # P帧格式：位置 + 编码
        frame_data.append(block_pos)
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
    coverage_radius_4x4: float
) -> tuple:
    """
    递归编码单个8x8块，严格按照8x8→8x4→4x4→4x2的分裂顺序
    
    返回: (encoding_list, stats)
    encoding_list格式:
    - 如果用8x8码表: [8x8_index]
    - 如果拆分为8x4: [MARKER_8x8_BLOCK, upper_8x4_encoding..., lower_8x4_encoding...]
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
        
        # 递归编码上8x4和下8x4
        upper_encoding, upper_stats = encode_8x4_block_recursive(
            upper_8x4, codebook_8x4, codebook_4x4, codebook_4x2,
            coverage_radius_8x4, coverage_radius_4x4
        )
        lower_encoding, lower_stats = encode_8x4_block_recursive(
            lower_8x4, codebook_8x4, codebook_4x4, codebook_4x2,
            coverage_radius_8x4, coverage_radius_4x4
        )
        
        # 合并统计
        for key in stats:
            stats[key] = upper_stats[key] + lower_stats[key]
        
        # 组装编码结果
        encoding = [MARKER_8x8_BLOCK] + upper_encoding + lower_encoding
        return encoding, stats
if __name__ == "__main__":
    main()