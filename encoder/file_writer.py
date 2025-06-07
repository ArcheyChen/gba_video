#!/usr/bin/env python3
"""
file_writer.py - 输出文件生成（.h和.c文件）
"""

import pathlib
import textwrap
from config import WIDTH, HEIGHT, CODEBOOK_SIZE


def write_header(path_h: pathlib.Path, frame_cnt: int, total_bytes: int, strip_count: int, strip_heights: list):
    """生成C头文件"""
    guard = "VIDEO_DATA_H"
    strip_heights_str = ', '.join(map(str, strip_heights))
    
    with path_h.open("w", encoding="utf-8") as f:
        f.write(textwrap.dedent(f"""\
            #ifndef {guard}
            #define {guard}

            #define VIDEO_FRAME_COUNT   {frame_cnt}
            #define VIDEO_WIDTH         {WIDTH}
            #define VIDEO_HEIGHT        {HEIGHT}
            #define VIDEO_TOTAL_BYTES   {total_bytes}
            #define VIDEO_STRIP_COUNT   {strip_count}
            #define CODEBOOK_SIZE       {CODEBOOK_SIZE}
            
            // 帧类型定义
            #define FRAME_TYPE_I        0x00
            #define FRAME_TYPE_P        0x01
            
            // 块参数
            #define BLOCK_WIDTH         2
            #define BLOCK_HEIGHT        2
            #define BYTES_PER_BLOCK     7

            // 条带高度数组
            extern const unsigned char strip_heights[VIDEO_STRIP_COUNT];
            
            extern const unsigned char video_data[VIDEO_TOTAL_BYTES];
            extern const unsigned int frame_offsets[VIDEO_FRAME_COUNT];

            #endif // {guard}
            """))


def write_source(path_c: pathlib.Path, data: bytes, frame_offsets: list, strip_heights: list):
    """生成C源文件"""
    with path_c.open("w", encoding="utf-8") as f:
        f.write('#include "video_data.h"\n\n')
        
        # 写入条带高度数组
        f.write("const unsigned char strip_heights[] = {\n")
        f.write("    " + ', '.join(map(str, strip_heights)) + "\n")
        f.write("};\n\n")
        
        # 写入帧偏移表
        f.write("const unsigned int frame_offsets[] = {\n")
        for i in range(0, len(frame_offsets), 8):
            chunk = ', '.join(f"{offset}" for offset in frame_offsets[i:i+8])
            f.write("    " + chunk + ",\n")
        f.write("};\n\n")
        
        # 写入视频数据
        f.write("const unsigned char video_data[] = {\n")
        per_line = 16
        for i in range(0, len(data), per_line):
            chunk = ', '.join(f"0x{v:02X}" for v in data[i:i+per_line])
            f.write("    " + chunk + ",\n")
        f.write("};\n")