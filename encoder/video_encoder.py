#!/usr/bin/env python3

import argparse, cv2, numpy as np, pathlib, textwrap
import multiprocessing as mp
import time
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
# 所有码表大小限制为255，使用U8索引，0xFF保留作为分裂标志
DEFAULT_CODEBOOK_SIZE_8x8 = 255     # 8x8块码表大小
DEFAULT_CODEBOOK_SIZE_8x4 = 255     # 8x4块码表大小
DEFAULT_CODEBOOK_SIZE_4x4 = 255     # 4x4块码表大小
DEFAULT_CODEBOOK_SIZE_4x2 = 255     # 4x2块码表大小（最大255，0xFF留作分裂标志）
DEFAULT_COVERAGE_RADIUS_8x8 = 80.0  # 8x8块覆盖半径
DEFAULT_COVERAGE_RADIUS_8x4 = 80.0 # 8x4块覆盖半径
DEFAULT_COVERAGE_RADIUS_4x4 = 80.0  # 4x4块覆盖半径

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
            #define VIDEO_MARKER_SPLIT    0xFF
            
            // Zone编码相关常量
            #define VIDEO_BLOCKS_PER_ZONE   {BLOCKS_PER_ZONE}
            #define VIDEO_ZONES_8x8_COUNT   {ZONES_8x8_COUNT}
            #define VIDEO_ZONES_8x4_COUNT   {ZONES_8x4_COUNT}
            #define VIDEO_ZONES_4x4_COUNT   {ZONES_4x4_COUNT}
            #define VIDEO_ZONES_4x2_COUNT   {ZONES_4x2_COUNT}
            #define VIDEO_ZONE_8x8_PIXEL_HEIGHT {ZONE_8x8_PIXEL_HEIGHT}
            #define VIDEO_ZONE_8x4_PIXEL_HEIGHT {ZONE_8x4_PIXEL_HEIGHT}
            #define VIDEO_ZONE_4x4_PIXEL_HEIGHT {ZONE_4x4_PIXEL_HEIGHT}
            #define VIDEO_ZONE_4x2_PIXEL_HEIGHT {ZONE_4x2_PIXEL_HEIGHT}

            /* 每个GOP的8x8码表：GOP_COUNT * CODEBOOK_SIZE_8x8 * BLOCK_SIZE_8x8 个uint16 */
            extern const unsigned short video_codebooks_8x8[VIDEO_GOP_COUNT][VIDEO_CODEBOOK_SIZE_8x8][VIDEO_BLOCK_SIZE_8x8];

            /* 每个GOP的8x4码表：GOP_COUNT * CODEBOOK_SIZE_8x4 * BLOCK_SIZE_8x4 个uint16 */
            extern const unsigned short video_codebooks_8x4[VIDEO_GOP_COUNT][VIDEO_CODEBOOK_SIZE_8x4][VIDEO_BLOCK_SIZE_8x4];

            /* 每个GOP的4x4码表：GOP_COUNT * CODEBOOK_SIZE_4x4 * BLOCK_SIZE_4x4 个uint16 */
            extern const unsigned short video_codebooks_4x4[VIDEO_GOP_COUNT][VIDEO_CODEBOOK_SIZE_4x4][VIDEO_BLOCK_SIZE_4x4];

            /* 每个GOP的4x2码表：GOP_COUNT * CODEBOOK_SIZE_4x2 * BLOCK_SIZE_4x2 个uint16 */
            extern const unsigned short video_codebooks_4x2[VIDEO_GOP_COUNT][VIDEO_CODEBOOK_SIZE_4x2][VIDEO_BLOCK_SIZE_4x2];

            /* 帧数据：变长编码的块索引，使用U8索引 */
            extern const unsigned char video_frame_data[];

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
        
        # 写入帧数据（变长编码）- 使用U8格式
        f.write("const unsigned char video_frame_data[] = {\n")
        all_data = []
        for frame_data in encoded_frames:
            all_data.extend(frame_data)
        
        per_line = 16
        for i in range(0, len(all_data), per_line):
            chunk = ', '.join(f"{val:3d}" for val in all_data[i:i+per_line])
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
    pa.add_argument("--gop-size", type=int,   default=120, help="GOP大小")
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

    # 验证码表大小不超过255（预留0xFF作为分裂标志）
    if codebook_size_8x8 > 255:
        raise SystemExit(f"❌ 8x8码表大小 ({codebook_size_8x8}) 不能超过255")
    if codebook_size_8x4 > 255:
        raise SystemExit(f"❌ 8x4码表大小 ({codebook_size_8x4}) 不能超过255")
    if codebook_size_4x4 > 255:
        raise SystemExit(f"❌ 4x4码表大小 ({codebook_size_4x4}) 不能超过255")
    if codebook_size_4x2 > 255:  # 4x2特别限制为255，因为需要保留0xFF
        raise SystemExit(f"❌ 4x2码表大小 ({codebook_size_4x2}) 不能超过255（0xFF保留作为分裂标志）")

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
            frm = cv2.blur(frm, (2, 2))  
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

    # 为每个GOP生成编码数据（多进程并行）
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
        'i_frame_stats': {
            'blocks_8x8_used': 0, 'blocks_8x4_used': 0, 'blocks_4x4_used': 0, 'blocks_4x2_used': 0,
            'frame_count': 0, 'total_blocks_encoded': 0,
            'avg_blocks_per_frame': 0.0, 'codec_efficiency': 0.0,
            'total_frame_size': 0, 'avg_frame_size_kb': 0.0,
            'total_codebook_size': 0, 'avg_codebook_size_kb': 0.0
        },
        'p_frame_stats': {
            'blocks_8x8_used': 0, 'blocks_8x4_used': 0, 'blocks_4x4_used': 0, 'blocks_4x2_used': 0,
            'frame_count': 0, 'total_changed_blocks': 0, 'total_segments': 0,
            'avg_changed_blocks_per_frame': 0.0, 'avg_segments_per_frame': 0.0,
            'segment_efficiency': 0.0, 'total_frame_size': 0, 'avg_frame_size_kb': 0.0,
            'block_type_sizes': {
                '8x8': 0, '8x4': 0, '4x4': 0, '4x2': 0, 'headers': 0
            },
            'zone_stats': {
                '8x8': {'zones_used': 0, 'segments_total': 0, 'blocks_total': 0, 'avg_segments_per_zone': 0.0},
                '8x4': {'zones_used': 0, 'segments_total': 0, 'blocks_total': 0, 'avg_segments_per_zone': 0.0},
                '4x4': {'zones_used': 0, 'segments_total': 0, 'blocks_total': 0, 'avg_segments_per_zone': 0.0},
                '4x2': {'zones_used': 0, 'segments_total': 0, 'blocks_total': 0, 'avg_segments_per_zone': 0.0}
            }
        },
        'frame_detailed_stats': []  # 存储每一帧的详细统计
    }

    # 准备多进程参数
    gop_args = []
    for gop_idx in range(gop_count):
        # 确定当前GOP的帧范围
        start_frame = gop_idx * gop_size
        end_frame = min((gop_idx + 1) * gop_size, total_frames)
        gop_frames = all_frame_blocks[start_frame:end_frame]
        
        # 准备参数元组
        args_tuple = (
            gop_idx, gop_frames, gop_size, i_frame_weight, diff_threshold,
            codebook_size_8x8, codebook_size_8x4, codebook_size_4x4, codebook_size_4x2,
            coverage_radius_8x8, coverage_radius_8x4, coverage_radius_4x4
        )
        gop_args.append(args_tuple)
    
    # 使用多进程并行处理GOP
    print(f"\n🚀 启动多进程处理 {gop_count} 个GOP...")
    cpu_count = mp.cpu_count()
    process_count = min(cpu_count, gop_count)  # 不要启动比GOP数量更多的进程
    print(f"CPU核心数: {cpu_count}, 使用进程数: {process_count}")
    
    start_time = time.time()
    
    with mp.Pool(processes=process_count) as pool:
        # 提交所有GOP处理任务
        results = pool.map(process_single_gop, gop_args)
    
    parallel_time = time.time() - start_time
    print(f"✅ 多进程处理完成！总耗时: {parallel_time:.1f}s")
    
    # 按GOP索引排序结果（确保顺序正确）
    results.sort(key=lambda x: x[0])
    
    # 重新组装结果
    print("\n📦 重新组装编码结果...")
    gop_codebooks = [None] * gop_count
    
    for gop_idx, codebooks, gop_encoded_frames, gop_frame_stats_list in results:
        # 存储码表
        gop_codebooks[gop_idx] = codebooks
        
        # 处理每一帧的编码数据
        for frame_idx, ((frame_data, frame_type), frame_stats) in enumerate(zip(gop_encoded_frames, gop_frame_stats_list)):
            global_frame_idx = gop_idx * gop_size + frame_idx
            
            encoded_frames.append(frame_data)
            frame_types.append(frame_type)
            current_offset += len(frame_data)
            frame_offsets.append(current_offset)
            
            # 更新统计
            total_stats['blocks_8x8_used'] += frame_stats['blocks_8x8_used']
            total_stats['blocks_8x4_used'] += frame_stats['blocks_8x4_used']
            total_stats['blocks_4x4_used'] += frame_stats['blocks_4x4_used']
            total_stats['blocks_4x2_used'] += frame_stats['blocks_4x2_used']
            
            if frame_type == 0:  # I帧
                total_stats['i_frame_stats']['frame_count'] += 1
                total_stats['i_frame_stats']['blocks_8x8_used'] += frame_stats['blocks_8x8_used']
                total_stats['i_frame_stats']['blocks_8x4_used'] += frame_stats['blocks_8x4_used']
                total_stats['i_frame_stats']['blocks_4x4_used'] += frame_stats['blocks_4x4_used']
                total_stats['i_frame_stats']['blocks_4x2_used'] += frame_stats['blocks_4x2_used']
                
                total_blocks = (frame_stats['blocks_8x8_used'] + frame_stats['blocks_8x4_used'] + 
                               frame_stats['blocks_4x4_used'] + frame_stats['blocks_4x2_used'])
                total_stats['i_frame_stats']['total_blocks_encoded'] += total_blocks
                
                # I帧大小统计（帧数据大小，不包括码表）
                total_stats['i_frame_stats']['total_frame_size'] += len(frame_data)
                
                # I帧对应的码表大小（每个I帧分摊GOP码表大小）
                # 每个GOP有一套码表，每个I帧分摊相同的码表大小
                codebook_size_bytes = (
                    codebook_size_8x8 * 64 * 2 +   # 8x8码表：每个码字64个uint16
                    codebook_size_8x4 * 32 * 2 +   # 8x4码表：每个码字32个uint16  
                    codebook_size_4x4 * 16 * 2 +   # 4x4码表：每个码字16个uint16
                    codebook_size_4x2 * 8 * 2      # 4x2码表：每个码字8个uint16
                )
                total_stats['i_frame_stats']['total_codebook_size'] += codebook_size_bytes
                
                print(f"  I帧 {global_frame_idx}: {BLOCKS_8x8_PER_FRAME} 个8x8块 (8x8码表: {frame_stats['blocks_8x8_used']}, 8x4码表: {frame_stats['blocks_8x4_used']}, 4x4码表: {frame_stats['blocks_4x4_used']}, 4x2码表: {frame_stats['blocks_4x2_used']}) - 大小: {len(frame_data)} 字节")
            else:  # P帧
                total_stats['p_frame_stats']['frame_count'] += 1
                total_stats['p_frame_stats']['blocks_8x8_used'] += frame_stats['blocks_8x8_used']
                total_stats['p_frame_stats']['blocks_8x4_used'] += frame_stats['blocks_8x4_used']
                total_stats['p_frame_stats']['blocks_4x4_used'] += frame_stats['blocks_4x4_used']
                total_stats['p_frame_stats']['blocks_4x2_used'] += frame_stats['blocks_4x2_used']
                
                # P帧特有统计
                if 'total_changed_blocks' in frame_stats:
                    total_stats['p_frame_stats']['total_changed_blocks'] += frame_stats['total_changed_blocks']
                if 'total_segments' in frame_stats:
                    total_stats['p_frame_stats']['total_segments'] += frame_stats['total_segments']
                if 'zone_stats' in frame_stats:
                    for block_type, zone_data in frame_stats['zone_stats'].items():
                        if 'zones_used' in zone_data:
                            total_stats['p_frame_stats']['zone_stats'][block_type]['zones_used'] += zone_data['zones_used']
                        if 'segments_total' in zone_data:
                            total_stats['p_frame_stats']['zone_stats'][block_type]['segments_total'] += zone_data['segments_total']
                        if 'blocks_total' in zone_data:
                            total_stats['p_frame_stats']['zone_stats'][block_type]['blocks_total'] += zone_data['blocks_total']
                
                # P帧大小统计
                total_stats['p_frame_stats']['total_frame_size'] += len(frame_data)
                
                # P帧各级码块大小统计（简化估算）
                if 'block_type_sizes' in frame_stats:
                    for block_type, size in frame_stats['block_type_sizes'].items():
                        total_stats['p_frame_stats']['block_type_sizes'][block_type] += size
                else:
                    # 如果没有详细的大小统计，使用简化估算
                    # 每个8x8块索引 + bitmap + overhead ≈ 1字节
                    # 每个8x4块索引 + bitmap + overhead ≈ 1字节  
                    # 每个4x4块索引 + bitmap + overhead ≈ 1字节
                    # 每个4x2块索引 + bitmap + overhead ≈ 1字节
                    estimated_8x8_size = frame_stats['blocks_8x8_used'] * 1
                    estimated_8x4_size = frame_stats['blocks_8x4_used'] * 1
                    estimated_4x4_size = frame_stats['blocks_4x4_used'] * 1  
                    estimated_4x2_size = frame_stats['blocks_4x2_used'] * 1
                    estimated_header_size = len(frame_data) - (estimated_8x8_size + estimated_8x4_size + estimated_4x4_size + estimated_4x2_size)
                    
                    total_stats['p_frame_stats']['block_type_sizes']['8x8'] += estimated_8x8_size
                    total_stats['p_frame_stats']['block_type_sizes']['8x4'] += estimated_8x4_size
                    total_stats['p_frame_stats']['block_type_sizes']['4x4'] += estimated_4x4_size
                    total_stats['p_frame_stats']['block_type_sizes']['4x2'] += estimated_4x2_size
                    total_stats['p_frame_stats']['block_type_sizes']['headers'] += max(0, estimated_header_size)
                
                changed_blocks = (frame_stats['blocks_8x8_used'] + frame_stats['blocks_8x4_used'] + 
                                 frame_stats['blocks_4x4_used'] + frame_stats['blocks_4x2_used'])
                print(f"  P帧 {global_frame_idx}: {changed_blocks} 个变化块 (8x8码表: {frame_stats['blocks_8x8_used']}, 8x4码表: {frame_stats['blocks_8x4_used']}, 4x4码表: {frame_stats['blocks_4x4_used']}, 4x2码表: {frame_stats['blocks_4x2_used']}) - 大小: {len(frame_data)} 字节")
            
            # 保存每帧详细统计
            frame_detail = {
                'frame_idx': global_frame_idx,
                'frame_type': 'I' if frame_type == 0 else 'P',
                'gop_idx': gop_idx,
                'frame_size': len(frame_data),
                'blocks_8x8_used': frame_stats['blocks_8x8_used'],
                'blocks_8x4_used': frame_stats['blocks_8x4_used'],
                'blocks_4x4_used': frame_stats['blocks_4x4_used'],
                'blocks_4x2_used': frame_stats['blocks_4x2_used']
            }
            if frame_type == 1 and 'zone_stats' in frame_stats:
                frame_detail['zone_stats'] = frame_stats['zone_stats']
            
            total_stats['frame_detailed_stats'].append(frame_detail)

    # 移除最后一个多余的偏移
    frame_offsets = frame_offsets[:-1]
    
    # 计算平均统计
    if total_stats['i_frame_stats']['frame_count'] > 0:
        total_stats['i_frame_stats']['avg_blocks_per_frame'] = (
            total_stats['i_frame_stats']['total_blocks_encoded'] / total_stats['i_frame_stats']['frame_count']
        )
        total_stats['i_frame_stats']['codec_efficiency'] = (
            total_stats['i_frame_stats']['blocks_8x8_used'] / total_stats['i_frame_stats']['total_blocks_encoded'] * 100
        )
        total_stats['i_frame_stats']['avg_frame_size_kb'] = (
            total_stats['i_frame_stats']['total_frame_size'] / total_stats['i_frame_stats']['frame_count'] / 1024
        )
        total_stats['i_frame_stats']['avg_codebook_size_kb'] = (
            total_stats['i_frame_stats']['total_codebook_size'] / total_stats['i_frame_stats']['frame_count'] / 1024
        )
    
    if total_stats['p_frame_stats']['frame_count'] > 0:
        total_stats['p_frame_stats']['avg_changed_blocks_per_frame'] = (
            total_stats['p_frame_stats']['total_changed_blocks'] / total_stats['p_frame_stats']['frame_count']
        )
        total_stats['p_frame_stats']['avg_segments_per_frame'] = (
            total_stats['p_frame_stats']['total_segments'] / total_stats['p_frame_stats']['frame_count']
        )
        total_stats['p_frame_stats']['avg_frame_size_kb'] = (
            total_stats['p_frame_stats']['total_frame_size'] / total_stats['p_frame_stats']['frame_count'] / 1024
        )
        if total_stats['p_frame_stats']['total_segments'] > 0:
            total_stats['p_frame_stats']['segment_efficiency'] = (
                total_stats['p_frame_stats']['total_changed_blocks'] / total_stats['p_frame_stats']['total_segments']
            )
    
    # 计算zone平均统计
    for block_type in ['8x8', '8x4', '4x4', '4x2']:
        zone_data = total_stats['p_frame_stats']['zone_stats'][block_type]
        if zone_data['zones_used'] > 0:
            zone_data['avg_segments_per_zone'] = zone_data['segments_total'] / zone_data['zones_used']
        if zone_data['segments_total'] > 0:
            zone_data['avg_blocks_per_segment'] = zone_data['blocks_total'] / zone_data['segments_total']

    # 写入文件
    write_header(pathlib.Path(args.out).with_suffix(".h"), total_frames, gop_count, gop_size, codebook_size_8x8, codebook_size_8x4, codebook_size_4x4, codebook_size_4x2)
    write_source(pathlib.Path(args.out).with_suffix(".c"), gop_codebooks, encoded_frames, frame_offsets, frame_types, codebook_size_8x8, codebook_size_8x4, codebook_size_4x4, codebook_size_4x2)

    print(f"\n📊 帧平均大小统计 (以KB计):")
    print("="*40)
    
    # I帧平均大小统计
    if total_stats['i_frame_stats']['frame_count'] > 0:
        i_frame_avg_kb = total_stats['i_frame_stats']['avg_frame_size_kb']
        i_codebook_avg_kb = total_stats['i_frame_stats']['avg_codebook_size_kb']
        i_total_avg_kb = i_frame_avg_kb + i_codebook_avg_kb
        
        print(f"🎯 I帧平均大小:")
        print(f"  - 帧内数据: {i_frame_avg_kb:.2f} KB/帧")
        print(f"  - 码表数据: {i_codebook_avg_kb:.2f} KB/帧")
        print(f"  - 总大小: {i_total_avg_kb:.2f} KB/帧")
        print(f"  - 总帧数: {total_stats['i_frame_stats']['frame_count']} 帧")
        print(f"  - 总帧内数据: {total_stats['i_frame_stats']['total_frame_size']/1024:.1f} KB")
        print(f"  - 总码表数据: {total_stats['i_frame_stats']['total_codebook_size']/1024:.1f} KB")
        print(f"  - 总I帧数据: {(total_stats['i_frame_stats']['total_frame_size'] + total_stats['i_frame_stats']['total_codebook_size'])/1024:.1f} KB")
    else:
        print("🎯 I帧平均大小: 无I帧数据")
    
    # P帧平均大小统计
    if total_stats['p_frame_stats']['frame_count'] > 0:
        p_frame_avg_kb = total_stats['p_frame_stats']['avg_frame_size_kb']
        
        print(f"\n🚀 P帧平均大小:")
        print(f"  - 帧数据: {p_frame_avg_kb:.2f} KB/帧")
        print(f"  - 总帧数: {total_stats['p_frame_stats']['frame_count']} 帧")
        print(f"  - 总帧数据: {total_stats['p_frame_stats']['total_frame_size']/1024:.1f} KB")
        
        # P帧各级码块平均大小
        print(f"  - 各级码块平均大小:")
        total_block_size = sum(total_stats['p_frame_stats']['block_type_sizes'].values())
        for block_type, size in total_stats['p_frame_stats']['block_type_sizes'].items():
            avg_size_per_frame = size / total_stats['p_frame_stats']['frame_count'] / 1024
            percentage = size / total_block_size * 100 if total_block_size > 0 else 0
            print(f"    - {block_type}: {avg_size_per_frame:.3f} KB/帧 ({percentage:.1f}%)")
    else:
        print("🚀 P帧平均大小: 无P帧数据")
    
    # I帧与P帧大小对比
    if total_stats['i_frame_stats']['frame_count'] > 0 and total_stats['p_frame_stats']['frame_count'] > 0:
        i_total_avg_kb = total_stats['i_frame_stats']['avg_frame_size_kb'] + total_stats['i_frame_stats']['avg_codebook_size_kb']
        p_avg_kb = total_stats['p_frame_stats']['avg_frame_size_kb']
        size_ratio = i_total_avg_kb / p_avg_kb if p_avg_kb > 0 else 0
        
        print(f"\n📈 I帧与P帧大小对比:")
        print(f"  - I帧平均大小: {i_total_avg_kb:.2f} KB/帧 (含码表)")
        print(f"  - P帧平均大小: {p_avg_kb:.2f} KB/帧")
        print(f"  - 大小比例: {size_ratio:.1f}:1 (I帧:P帧)")
        print(f"  - P帧压缩效率: {100/size_ratio:.1f}% (相对I帧)")
    
    print("="*40)
    
    # 继续原有的详细统计信息
    print("\n" + "="*60)
    print("📊 详细编码统计信息")
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
    
    print(f"\n🎯 I帧详细统计 ({total_stats['i_frame_stats']['frame_count']} 帧):")
    print(f"  - 8x8码表使用: {total_stats['i_frame_stats']['blocks_8x8_used']:,} 个8x8块")
    print(f"  - 8x4码表使用: {total_stats['i_frame_stats']['blocks_8x4_used']:,} 个8x4块")
    print(f"  - 4x4码表使用: {total_stats['i_frame_stats']['blocks_4x4_used']:,} 个4x4块")
    print(f"  - 4x2码表使用: {total_stats['i_frame_stats']['blocks_4x2_used']:,} 个4x2块")
    print(f"  - 总编码块数: {total_stats['i_frame_stats']['total_blocks_encoded']:,}")
    print(f"  - 平均每帧编码块数: {total_stats['i_frame_stats']['avg_blocks_per_frame']:.1f}")
    print(f"  - 8x8码表效率: {total_stats['i_frame_stats']['codec_efficiency']:.1f}% (越高越好)")
    print(f"  📊 I帧大小统计:")
    print(f"    - 总帧内数据大小: {total_stats['i_frame_stats']['total_frame_size']:,} 字节 ({total_stats['i_frame_stats']['total_frame_size']/1024:.1f} KB)")
    print(f"    - 平均帧内数据大小: {total_stats['i_frame_stats']['avg_frame_size_kb']:.2f} KB/帧")
    print(f"    - 总码表数据大小: {total_stats['i_frame_stats']['total_codebook_size']:,} 字节 ({total_stats['i_frame_stats']['total_codebook_size']/1024:.1f} KB)")
    print(f"    - 平均码表数据大小: {total_stats['i_frame_stats']['avg_codebook_size_kb']:.2f} KB/帧")
    print(f"    - 平均I帧总大小: {total_stats['i_frame_stats']['avg_frame_size_kb'] + total_stats['i_frame_stats']['avg_codebook_size_kb']:.2f} KB/帧 (帧内+码表)")
    if total_8x8_blocks > 0:
        i_8x8_ratio = total_stats['i_frame_stats']['blocks_8x8_used'] / total_8x8_blocks * 100
        print(f"  - I帧中8x8码表覆盖率: {i_8x8_ratio:.1f}%")
    
    print(f"\n🚀 P帧详细统计 ({total_stats['p_frame_stats']['frame_count']} 帧):")
    print(f"  - 8x8码表使用: {total_stats['p_frame_stats']['blocks_8x8_used']:,} 个8x8块")
    print(f"  - 8x4码表使用: {total_stats['p_frame_stats']['blocks_8x4_used']:,} 个8x4块")
    print(f"  - 4x4码表使用: {total_stats['p_frame_stats']['blocks_4x4_used']:,} 个4x4块")
    print(f"  - 4x2码表使用: {total_stats['p_frame_stats']['blocks_4x2_used']:,} 个4x2块")
    print(f"  - 总变化块数: {total_stats['p_frame_stats']['total_changed_blocks']:,}")
    print(f"  - 总编码段数: {total_stats['p_frame_stats']['total_segments']:,}")
    print(f"  - 平均每帧变化块数: {total_stats['p_frame_stats']['avg_changed_blocks_per_frame']:.1f}")
    print(f"  - 平均每帧编码段数: {total_stats['p_frame_stats']['avg_segments_per_frame']:.1f}")
    print(f"  - 段编码效率: {total_stats['p_frame_stats']['segment_efficiency']:.1f} 块/段 (越高越好)")
    print(f"  📊 P帧大小统计:")
    print(f"    - 总帧大小: {total_stats['p_frame_stats']['total_frame_size']:,} 字节 ({total_stats['p_frame_stats']['total_frame_size']/1024:.1f} KB)")
    print(f"    - 平均帧大小: {total_stats['p_frame_stats']['avg_frame_size_kb']:.2f} KB/帧")
    print(f"    - 各级码块空间占用:")
    total_block_size = sum(total_stats['p_frame_stats']['block_type_sizes'].values())
    if total_block_size > 0:
        for block_type, size in total_stats['p_frame_stats']['block_type_sizes'].items():
            percentage = size / total_block_size * 100 if total_block_size > 0 else 0
            avg_size_per_frame = size / total_stats['p_frame_stats']['frame_count'] / 1024 if total_stats['p_frame_stats']['frame_count'] > 0 else 0
            print(f"      - {block_type}码表: {size:,} 字节 ({percentage:.1f}%, {avg_size_per_frame:.3f} KB/帧)")
    
    # P帧与I帧大小对比
    if total_stats['i_frame_stats']['frame_count'] > 0 and total_stats['p_frame_stats']['frame_count'] > 0:
        i_total_avg = total_stats['i_frame_stats']['avg_frame_size_kb'] + total_stats['i_frame_stats']['avg_codebook_size_kb']
        p_avg = total_stats['p_frame_stats']['avg_frame_size_kb']
        compression_ratio = i_total_avg / p_avg if p_avg > 0 else 0
        print(f"    - P帧相对I帧压缩比: {compression_ratio:.1f}:1 (I帧: {i_total_avg:.2f} KB, P帧: {p_avg:.2f} KB)")
    
    # P帧变化率分析
    if p_frame_count > 0:
        total_possible_p_blocks = p_frame_count * BLOCKS_8x8_PER_FRAME
        p_change_rate = total_stats['p_frame_stats']['total_changed_blocks'] / total_possible_p_blocks * 100
        print(f"  - P帧总体变化率: {p_change_rate:.1f}% ({total_stats['p_frame_stats']['total_changed_blocks']:,}/{total_possible_p_blocks:,})")
    
    print(f"\n🗺️ Zone编码效率统计:")
    for block_type in ['8x8', '8x4', '4x4', '4x2']:
        zone_data = total_stats['p_frame_stats']['zone_stats'][block_type]
        if zone_data['blocks_total'] > 0:
            print(f"  {block_type}码表:")
            print(f"    - 使用的zone数: {zone_data['zones_used']}")
            print(f"    - 编码段总数: {zone_data['segments_total']}")
            print(f"    - 变化块总数: {zone_data['blocks_total']}")
            print(f"    - 平均每zone段数: {zone_data['avg_segments_per_zone']:.1f}")
            print(f"    - 平均每段块数: {zone_data['avg_blocks_per_segment']:.1f}")
            
            # 计算zone利用率
            if block_type == "8x8":
                max_zones = ZONES_8x8_COUNT
            elif block_type == "8x4":
                max_zones = ZONES_8x4_COUNT
            elif block_type == "4x4":
                max_zones = ZONES_4x4_COUNT
            elif block_type == "4x2":
                max_zones = ZONES_4x2_COUNT
            
            zone_usage_rate = zone_data['zones_used'] / (max_zones * total_stats['p_frame_stats']['frame_count']) * 100
            print(f"    - Zone利用率: {zone_usage_rate:.1f}% ({zone_data['zones_used']}/{max_zones * total_stats['p_frame_stats']['frame_count']})")
    
    print(f"\n📊 码表利用率分析:")
    print("I帧码表利用率 (相对于码表大小):")
    if total_stats['i_frame_stats']['frame_count'] > 0:
        i_8x8_per_frame = total_stats['i_frame_stats']['blocks_8x8_used'] / total_stats['i_frame_stats']['frame_count']
        i_8x4_per_frame = total_stats['i_frame_stats']['blocks_8x4_used'] / total_stats['i_frame_stats']['frame_count']
        i_4x4_per_frame = total_stats['i_frame_stats']['blocks_4x4_used'] / total_stats['i_frame_stats']['frame_count']
        i_4x2_per_frame = total_stats['i_frame_stats']['blocks_4x2_used'] / total_stats['i_frame_stats']['frame_count']
        
        print(f"  - 8x8码表: {i_8x8_per_frame:.1f} 个/帧 (码表大小: {codebook_size_8x8}, 利用率: {i_8x8_per_frame/codebook_size_8x8*100:.1f}%)")
        print(f"  - 8x4码表: {i_8x4_per_frame:.1f} 个/帧 (码表大小: {codebook_size_8x4}, 利用率: {i_8x4_per_frame/codebook_size_8x4*100:.1f}%)")
        print(f"  - 4x4码表: {i_4x4_per_frame:.1f} 个/帧 (码表大小: {codebook_size_4x4}, 利用率: {i_4x4_per_frame/codebook_size_4x4*100:.1f}%)")
        print(f"  - 4x2码表: {i_4x2_per_frame:.1f} 个/帧 (码表大小: {codebook_size_4x2}, 利用率: {i_4x2_per_frame/codebook_size_4x2*100:.1f}%)")
    
    print("P帧码表利用率 (相对于码表大小):")
    if total_stats['p_frame_stats']['frame_count'] > 0:
        p_8x8_per_frame = total_stats['p_frame_stats']['blocks_8x8_used'] / total_stats['p_frame_stats']['frame_count']
        p_8x4_per_frame = total_stats['p_frame_stats']['blocks_8x4_used'] / total_stats['p_frame_stats']['frame_count']
        p_4x4_per_frame = total_stats['p_frame_stats']['blocks_4x4_used'] / total_stats['p_frame_stats']['frame_count']
        p_4x2_per_frame = total_stats['p_frame_stats']['blocks_4x2_used'] / total_stats['p_frame_stats']['frame_count']
        
        print(f"  - 8x8码表: {p_8x8_per_frame:.1f} 个/帧 (码表大小: {codebook_size_8x8}, 利用率: {p_8x8_per_frame/codebook_size_8x8*100:.1f}%)")
        print(f"  - 8x4码表: {p_8x4_per_frame:.1f} 个/帧 (码表大小: {codebook_size_8x4}, 利用率: {p_8x4_per_frame/codebook_size_8x4*100:.1f}%)")
        print(f"  - 4x4码表: {p_4x4_per_frame:.1f} 个/帧 (码表大小: {codebook_size_4x4}, 利用率: {p_4x4_per_frame/codebook_size_4x4*100:.1f}%)")
        print(f"  - 4x2码表: {p_4x2_per_frame:.1f} 个/帧 (码表大小: {codebook_size_4x2}, 利用率: {p_4x2_per_frame/codebook_size_4x2*100:.1f}%)")
    
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
        encoding = [MARKER_SPLIT] + left_encoding + right_encoding
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
        
        # 使用4x2码表编码
        upper_indices = encode_frame_with_codebook(upper_4x2.reshape(1, -1), codebook_4x2)
        lower_indices = encode_frame_with_codebook(lower_4x2.reshape(1, -1), codebook_4x2)
        
        stats['blocks_4x2_used'] = 2
        encoding = [MARKER_SPLIT, upper_indices[0], lower_indices[0]]
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
    coverage_radius_4x4: float
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
        encoding = [MARKER_SPLIT] + upper_encoding + lower_encoding
        return encoding, stats

def process_single_gop(args):
    """
    处理单个GOP的函数 - 用于多进程
    
    Args:
        args: 包含处理GOP所需的所有参数的元组
        
    Returns:
        tuple: (gop_idx, codebooks, encoded_frames, frame_stats)
    """
    (gop_idx, gop_frames, gop_size, i_frame_weight, diff_threshold,
     codebook_size_8x8, codebook_size_8x4, codebook_size_4x4, codebook_size_4x2,
     coverage_radius_8x8, coverage_radius_8x4, coverage_radius_4x4) = args
    
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
    
    for frame_idx, frame_blocks_8x8 in enumerate(gop_frames):
        if frame_idx == 0:  # I帧
            frame_data, frame_stats = encode_i_frame_multi_level_8x8(
                frame_blocks_8x8, codebook_8x8, codebook_8x4, codebook_4x4, codebook_4x2, 
                coverage_radius_8x8, coverage_radius_8x4, coverage_radius_4x4
            )
            frame_type = 0  # I帧
        else:  # P帧
            # P帧只编码变化的块（使用新的分层编码）
            previous_blocks = gop_frames[frame_idx - 1]
            frame_data, frame_stats = encode_p_frame_multi_level_separate(
                frame_blocks_8x8, previous_blocks, codebook_8x8, codebook_8x4, codebook_4x4, codebook_4x2, 
                diff_threshold, coverage_radius_8x8, coverage_radius_8x4, coverage_radius_4x4
            )
            frame_type = 1  # P帧
        
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
    coverage_radius_4x4: float = 80.0
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

if __name__ == "__main__":
    main()