#!/usr/bin/env python3

import pathlib
import textwrap
from core_encoder import BYTES_PER_BLOCK

def write_header(path_h: pathlib.Path, frame_cnt: int, total_bytes: int, codebook_size: int, output_fps: float):
    """生成C语言头文件"""
    guard = "VIDEO_DATA_H"
    
    # 解码器期望的BYTES_PER_BLOCK（YUV420格式）
    DECODER_BYTES_PER_BLOCK = 6  # 4Y + 1Cb + 1Cr (YUV420)
    
    with path_h.open("w", encoding="utf-8") as f:
        f.write(textwrap.dedent(f"""\
            #ifndef {guard}
            #define {guard}

            #define VIDEO_FRAME_COUNT   {frame_cnt}
            #define VIDEO_WIDTH         {240}
            #define VIDEO_HEIGHT        {160}
            #define VIDEO_TOTAL_BYTES   {total_bytes}
            #define VIDEO_FPS           {int(round(output_fps*10000))}
            #define UNIFIED_CODEBOOK_SIZE {codebook_size}
            #define EFFECTIVE_UNIFIED_CODEBOOK_SIZE {254}
            
            // 帧类型定义
            #define FRAME_TYPE_I        0x00
            #define FRAME_TYPE_P        0x01
            
            // 特殊标记
            #define COLOR_BLOCK_MARKER  0xFF
            
            // 块参数
            #define BLOCK_WIDTH         2
            #define BLOCK_HEIGHT        2
            #define BYTES_PER_BLOCK     {DECODER_BYTES_PER_BLOCK}
            
            extern const unsigned char video_data[VIDEO_TOTAL_BYTES];
            extern const unsigned int frame_offsets[VIDEO_FRAME_COUNT];

            #endif // {guard}
            """))

def write_source(path_c: pathlib.Path, data: bytes, frame_offsets: list):
    """生成C语言源文件"""
    with path_c.open("w", encoding="utf-8") as f:
        f.write('#include "video_data.h"\n\n')
        
        f.write("const unsigned int frame_offsets[] = {\n")
        for i in range(0, len(frame_offsets), 8):
            chunk = ', '.join(f"{offset}" for offset in frame_offsets[i:i+8])
            f.write("    " + chunk + ",\n")
        f.write("};\n\n")
        
        f.write("const unsigned char video_data[] = {\n")
        per_line = 16
        for i in range(0, len(data), per_line):
            chunk = ', '.join(f"0x{v:02X}" for v in data[i:i+per_line])
            f.write("    " + chunk + ",\n")
        f.write("};\n")

def extract_frames_from_video(video_path: str, duration: float, fps: int, full_duration: bool = False, dither: bool = False):
    """从视频文件中提取帧"""
    import cv2
    import numpy as np
    from core_encoder import pack_yuv444_frame
    from dither_opt import apply_dither_optimized
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit("❌ 打不开输入文件")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    # 计算实际输出FPS：如果目标FPS高于源FPS，使用源FPS
    actual_output_fps = min(fps, src_fps)
    every = int(round(src_fps / actual_output_fps))
    
    if full_duration:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        grab_max = total_frames
        actual_duration = total_frames / src_fps
        print(f"编码整个视频: {total_frames} 帧，时长 {actual_duration:.2f} 秒")
    else:
        grab_max = int(duration * src_fps)
        print(f"编码时长: {duration} 秒 ({grab_max} 帧)")

    print(f"源视频FPS: {src_fps:.2f}, 目标FPS: {fps}, 实际输出FPS: {actual_output_fps:.2f}")
    if dither:
        print(f"🎨 已启用抖动算法（蛇形扫描）")
    
    frames = []
    idx = 0
    print("正在提取帧...")
    
    while idx < grab_max:
        ret, frm = cap.read()
        if not ret:
            break
        if idx % every == 0:
            frm = cv2.resize(frm, (240, 160), cv2.INTER_AREA)
            frm = cv2.GaussianBlur(frm, (3, 3), 0.41)
            if dither:
                frm = apply_dither_optimized(frm)
            
            frame_blocks = pack_yuv444_frame(frm)
            frames.append(frame_blocks)
            
            if len(frames) % 30 == 0:
                print(f"  已提取 {len(frames)} 帧")
        idx += 1
    cap.release()

    if not frames:
        raise SystemExit("❌ 没有任何帧被采样")

    print(f"总共提取了 {len(frames)} 帧")
    return frames, actual_output_fps 