#!/usr/bin/env python3

import pathlib
import textwrap

# 新增：导入分离的工具模块
from block_utils import *

WIDTH, HEIGHT = 240, 160

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