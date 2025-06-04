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
BLOCK_W, BLOCK_H = 4, 4
BYTES_PER_BLOCK  = 18                                   # 16Y + Cb + Cr

def pack_yuv9(frame_bgr: np.ndarray) -> np.ndarray:
    """
    把 240×160×3 BGR → YUV9：每 4×4 像素 18 Byte
    布局按行优先：Y00..Y03 Y10..Y13 Y20..Y23 Y30..Y33 Cb Cr
    """
    B = frame_bgr[:, :, 0].astype(np.float32)
    G = frame_bgr[:, :, 1].astype(np.float32)
    R = frame_bgr[:, :, 2].astype(np.float32)

    Y  = (R*Y_COEFF[0]  + G*Y_COEFF[1]  + B*Y_COEFF[2]).round()
    Cb = (R*CB_COEFF[0] + G*CB_COEFF[1] + B*CB_COEFF[2]).round()
    Cr = (R*CR_COEFF[0] + G*CR_COEFF[1] + B*CR_COEFF[2]).round()

    Y  = np.clip(Y,  0, 255).astype(np.uint8)
    # Cb = np.clip(Cb, -128,127).astype(np.uint8)
    # Cr = np.clip(Cr, -128,127).astype(np.uint8)

    blocks = bytearray()
    for y in range(0, HEIGHT, BLOCK_H):
        for x in range(0, WIDTH, BLOCK_W):
            # 16 Y：四行各 4 像素
            blocks.extend(Y [y:y+4, x:x+4].flatten())
            # # 1 Cb / 1 Cr：取块左上像素即可
            # blocks.append(Cb[y, x])
            # blocks.append(Cr[y, x])
            blocks.append(Cb[y:y+4, x:x+4].mean().round().astype(np.uint8))
            blocks.append(Cr[y:y+4, x:x+4].mean().round().astype(np.uint8))
    return np.frombuffer(blocks, dtype=np.uint8)

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
            frames.append(pack_yuv9(frm))
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
