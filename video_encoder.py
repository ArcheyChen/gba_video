#!/usr/bin/env python3
"""
gba_encode.py
-------------
把输入视频 / GIF / 图片序列转为 GBA Mode3 RGB15 数据，
并生成 video_data.c / video_data.h
默认只截取前 duration 秒、按 fps 采样并缩放到 240×160。
"""

import argparse
import cv2          # pip install opencv-python
import numpy as np
import pathlib
import textwrap

WIDTH, HEIGHT = 240, 160

def bgr_to_gba565(frame_bgr: np.ndarray) -> np.ndarray:
    """
    frame_bgr:  H×W×3 uint8
    return:     (H*W) uint16 RGB555 little-endian
    """
    # OpenCV BGR → R/G/B 分量
    b = frame_bgr[:,:,0].astype(np.uint16) >> 3
    g = frame_bgr[:,:,1].astype(np.uint16) >> 3
    r = frame_bgr[:,:,2].astype(np.uint16) >> 3
    gba = (r) | (g << 5) | (b << 10)
    return gba.flatten()

def write_header(path_h: pathlib.Path, frame_count: int):
    guard = "VIDEO_DATA_H"
    with path_h.open("w", encoding="utf-8") as f:
        f.write(textwrap.dedent(f"""\
            #ifndef {guard}
            #define {guard}

            #define VIDEO_FRAME_COUNT  {frame_count}
            #define VIDEO_WIDTH        {WIDTH}
            #define VIDEO_HEIGHT       {HEIGHT}

            extern const unsigned short video_data[VIDEO_FRAME_COUNT * VIDEO_WIDTH * VIDEO_HEIGHT];

            #endif // {guard}
            """))

def write_source(path_c: pathlib.Path, data: np.ndarray):
    """把 uint16 一维数组按 12 个元素 / 行排版写入 .c"""
    with path_c.open("w", encoding="utf-8") as f:
        f.write('#include "video_data.h"\n\n')
        f.write("const unsigned short video_data[] = {\n")

        per_line = 12
        for i in range(0, len(data), per_line):
            chunk = ', '.join(f"0x{val:04X}" for val in data[i:i+per_line])
            f.write("    " + chunk + ",\n")

        f.write("};\n")

def main():
    parser = argparse.ArgumentParser(
        description="Encode video to GBA Mode3 C array")
    parser.add_argument("input", help="输入视频/动画/GIF/图片序列")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="截取时长（秒），默认 5")
    parser.add_argument("--fps", type=int, default=30,
                        help="目标帧率，默认 30")
    parser.add_argument("--out", default="video_data",
                        help="输出文件名前缀，默认 video_data")

    args = parser.parse_args()
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit("❌ 无法打开输入文件")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(round(src_fps / args.fps))
    total_frames = int(args.duration * src_fps)

    raw_pixels = []

    frame_idx = 0
    grabbed = True
    while grabbed and frame_idx < total_frames:
        grabbed, frame = cap.read()
        if not grabbed:
            break

        if frame_idx % frame_interval == 0:
            resized = cv2.resize(frame, (WIDTH, HEIGHT),
                                 interpolation=cv2.INTER_AREA)
            gba = bgr_to_gba565(resized)
            raw_pixels.append(gba)

        frame_idx += 1

    cap.release()

    if not raw_pixels:
        raise SystemExit("❌ 未得到任何帧，请检查输入 / 参数")

    pixel_array = np.concatenate(raw_pixels).astype(np.uint16)

    out_prefix = pathlib.Path(args.out)
    write_header(out_prefix.with_suffix(".h"), len(raw_pixels))
    write_source(out_prefix.with_suffix(".c"), pixel_array)

    print(f"✅ 生成完成，共 {len(raw_pixels)} 帧")
    print(f"   {out_prefix}.h")
    print(f"   {out_prefix}.c")

if __name__ == "__main__":
    main()
