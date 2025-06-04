#!/usr/bin/env python3
"""
gba_encode.py  v2  ——  把视频/图片序列转成 GBA Mode3 YUV411 数据
输出 video_data.c / video_data.h
默认 5 s @ 30 fps，可用 --duration / --fps 修改
"""

import argparse, cv2, numpy as np, pathlib, textwrap

WIDTH, HEIGHT = 240, 160

def pack_yuv411(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR → YCrCb → 打包成 [Y0 Y1 Y2 Y3 Cb Cr]×(240/4)×160"""
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)          # OpenCV 顺序 Y Cr Cb
    # flatten 行优先
    packed_rows = []
    for row in range(HEIGHT):
        yrow = Y[row]
        cbrow = Cb[row]
        crrow = Cr[row]
        line = []
        for x in range(0, WIDTH, 4):
            line.extend((
                yrow[x], yrow[x+1], yrow[x+2], yrow[x+3],
                cbrow[x], crrow[x]      # 取第一像素的 Cb/Cr 即可
            ))
        packed_rows.append(line)
    return np.array(packed_rows, dtype=np.uint8).flatten()

def write_header(path_h: pathlib.Path, frame_cnt: int, bytes_per_frame: int):
    guard = "VIDEO_DATA_H"
    with path_h.open("w", encoding="utf-8") as f:
        f.write(textwrap.dedent(f"""\
            #ifndef {guard}
            #define {guard}

            #define VIDEO_FRAME_COUNT   {frame_cnt}
            #define VIDEO_WIDTH         {WIDTH}
            #define VIDEO_HEIGHT        {HEIGHT}
            #define VIDEO_BYTES_PER_FRAME {bytes_per_frame}

            extern const unsigned char video_data[VIDEO_FRAME_COUNT * VIDEO_BYTES_PER_FRAME];

            #endif // {guard}
            """))

def write_source(path_c: pathlib.Path, data: np.ndarray):
    with path_c.open("w", encoding="utf-8") as f:
        f.write('#include "video_data.h"\n\n')
        f.write("const unsigned char video_data[] = {\n")
        per_line = 16
        for i in range(0, len(data), per_line):
            chunk = ', '.join(f"0x{v:02X}" for v in data[i:i+per_line])
            f.write("    " + chunk + ",\n")
        f.write("};\n")

def main():
    pa = argparse.ArgumentParser(description="Encode to GBA YUV411")
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

    frames = []
    idx = 0
    while idx < grab_max:
        ret, frm = cap.read()
        if not ret:
            break
        if idx % every == 0:
            frm = cv2.resize(frm, (WIDTH, HEIGHT), cv2.INTER_AREA)
            frames.append(pack_yuv411(frm))
        idx += 1
    cap.release()

    if not frames:
        raise SystemExit("❌ 没有任何帧被采样")

    all_bytes = np.concatenate(frames).astype(np.uint8)
    bpf = len(frames[0])

    write_header(pathlib.Path(args.out).with_suffix(".h"), len(frames), bpf)
    write_source(pathlib.Path(args.out).with_suffix(".c"), all_bytes)

    print(f"✅ 完成：{len(frames)} 帧, {bpf} bytes/frame")

if __name__ == "__main__":
    main()
