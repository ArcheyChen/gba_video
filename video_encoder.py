#!/usr/bin/env python3
"""
gba_encode.py  v2  ——  把视频/图片序列转成 GBA Mode3 YUV411 数据
输出 video_data.c / video_data.h
默认 5 s @ 30 fps，可用 --duration / --fps 修改
"""

import argparse, cv2, numpy as np, pathlib, textwrap

WIDTH, HEIGHT = 240, 160

Y_COEFF  = np.array([0.28571429,  0.57142857,  0.14285714])
CB_COEFF = np.array([-0.14285714, -0.28571429,  0.42857143])
CR_COEFF = np.array([ 0.35714286, -0.28571429, -0.07142857])

def pack_yuv420(frame_bgr: np.ndarray) -> np.ndarray:
    """
    BGR (240×160×3) → YUV420 block packing:
      [Y00 Y01 Y10 Y11 Cb Cr]  (2×2 像素共 6 Byte)
    """
    B = frame_bgr[:, :, 0].astype(np.float32)
    G = frame_bgr[:, :, 1].astype(np.float32)
    R = frame_bgr[:, :, 2].astype(np.float32)

    Y  = (R*Y_COEFF[0]  + G*Y_COEFF[1]  + B*Y_COEFF[2]).round()
    Cb = (R*CB_COEFF[0] + G*CB_COEFF[1] + B*CB_COEFF[2]).round() + 128.0
    Cr = (R*CR_COEFF[0] + G*CR_COEFF[1] + B*CR_COEFF[2]).round() + 128.0

    Y  = np.clip(Y,  0, 255).astype(np.uint8)
    Cb = np.clip(Cb, 0, 255).astype(np.uint8)
    Cr = np.clip(Cr, 0, 255).astype(np.uint8)

    blocks = []
    for y in range(0, HEIGHT, 2):          # 行步距 2
        for x in range(0, WIDTH, 2):       # 列步距 2
            blocks.extend((
                Y [y,   x],   Y [y,   x+1],
                Y [y+1, x],   Y [y+1, x+1],
                Cb[y,   x],   Cr[y,   x]   # 取块左上像素色差
            ))
    return np.frombuffer(bytes(blocks), dtype=np.uint8)


def pack_yuv411(frame_bgr: np.ndarray) -> np.ndarray:
    """
    输入: 240×160×3 uint8 (BGR)
    输出: (57 600) uint8, layout = [Y0 Y1 Y2 Y3 Cb Cr]×(240/4)×160
    """
    # OpenCV 默认是 BGR => 先变成 float32 的 R/G/B，方便矩阵乘
    B = frame_bgr[:, :, 0].astype(np.float32)
    G = frame_bgr[:, :, 1].astype(np.float32)
    R = frame_bgr[:, :, 2].astype(np.float32)

    # 每个像素单独算 Y/Cb/Cr
    Y  = (R * Y_COEFF[0]  + G * Y_COEFF[1]  + B * Y_COEFF[2]).round()
    Cb = (R * CB_COEFF[0] + G * CB_COEFF[1] + B * CB_COEFF[2]).round() + 128.0
    Cr = (R * CR_COEFF[0] + G * CR_COEFF[1] + B * CR_COEFF[2]).round() + 128.0

    # 裁剪到合法区间并转回 uint8
    Y  = np.clip(Y,  0, 255).astype(np.uint8)
    Cb = np.clip(Cb, 0, 255).astype(np.uint8)
    Cr = np.clip(Cr, 0, 255).astype(np.uint8)

    # ↓↓↓ 打包成 4:1:1 ↓↓↓
    packed_rows = []
    for row in range(HEIGHT):
        y_row  = Y [row]
        cb_row = Cb[row]
        cr_row = Cr[row]
        line = []
        for x in range(0, WIDTH, 4):
            line.extend((                        # 6 Byte / 4 像素
                y_row[x], y_row[x+1], y_row[x+2], y_row[x+3],
                cb_row[x], cr_row[x]             # 取块首像素的 Cb / Cr
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
            # frames.append(pack_yuv411(frm))
            frames.append(pack_yuv420(frm))
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
