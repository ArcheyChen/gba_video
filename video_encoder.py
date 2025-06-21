#!/usr/bin/env python3

import argparse
import pathlib

from video_encoder_core import VideoEncoderCore
from video_encoder_utils import extract_frames_from_video
from core_encoder import DEFAULT_UNIFIED_CODEBOOK_SIZE, DEFAULT_ENABLED_SEGMENTS_BITMAP, DEFAULT_ENABLED_MEDIUM_SEGMENTS_BITMAP

def main():
    pa = argparse.ArgumentParser(description="Encode to GBA YUV9 with unified codebook")
    pa.add_argument("input")
    pa.add_argument("--duration", type=float, default=5.0)
    pa.add_argument("--full-duration", action="store_true")
    pa.add_argument("--fps", type=int, default=30)
    pa.add_argument("--out", default="video_data")
    pa.add_argument("--i-frame-interval", type=int, default=60)
    pa.add_argument("--diff-threshold", type=float, default=2.0)
    pa.add_argument("--force-i-threshold", type=float, default=0.7)
    pa.add_argument("--variance-threshold", type=float, default=5.0,
                   help="方差阈值，用于区分纯色块和纹理块（默认5.0）")
    pa.add_argument("--codebook-size", type=int, default=DEFAULT_UNIFIED_CODEBOOK_SIZE,
                   help=f"统一码本大小（默认{DEFAULT_UNIFIED_CODEBOOK_SIZE}）")
    pa.add_argument("--kmeans-max-iter", type=int, default=200)
    pa.add_argument("--threads", type=int, default=None)
    pa.add_argument("--i-frame-weight", type=int, default=3,
                   help="I帧块在聚类中的权重倍数（默认3）")
    pa.add_argument("--max-workers", type=int, default=None,
                   help="GOP处理的最大进程数（默认为CPU核心数-1）")
    pa.add_argument("--dither", action="store_true",
                   help="启用Floyd-Steinberg抖动算法提升画质")
    pa.add_argument("--enabled-segments-bitmap", type=int, default=DEFAULT_ENABLED_SEGMENTS_BITMAP,
                   help=f"启用段的bitmap，每位表示对应段是否启用小码表模式（默认0x{DEFAULT_ENABLED_SEGMENTS_BITMAP:04X}）")
    pa.add_argument("--enabled-medium-segments-bitmap", type=int, default=DEFAULT_ENABLED_MEDIUM_SEGMENTS_BITMAP,
                   help=f"启用中码表段的bitmap，每位表示对应段是否启用中码表模式（默认0x{DEFAULT_ENABLED_MEDIUM_SEGMENTS_BITMAP:02X}）")
    pa.add_argument("--no-parallel", action="store_true",
                   help="禁用并行处理，使用串行模式")
    pa.add_argument("--audio-sample-rate", type=int, default=16000,
                   help="音频采样率（默认16000 Hz）")
    pa.add_argument("--no-audio", action="store_true",
                   help="禁用音频提取")
    args = pa.parse_args()

    # 提取帧
    frames, actual_output_fps = extract_frames_from_video(
        args.input, args.duration, args.fps, args.full_duration, args.dither
    )

    # 音频处理
    audio_data = None
    if not args.no_audio:
        from audio_encoder import AudioEncoder
        audio_encoder = AudioEncoder(sample_rate=args.audio_sample_rate)
        
        # 计算实际音频时长
        if args.full_duration:
            import cv2
            cap = cv2.VideoCapture(args.input)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            audio_duration = total_frames / src_fps
            cap.release()
        else:
            audio_duration = args.duration
        
        audio_data = audio_encoder.extract_audio_from_video(args.input, audio_duration)
        if audio_data is None:
            print("⚠️ 音频提取失败，继续处理视频...")

    # 创建编码器核心
    encoder = VideoEncoderCore()
    
    # 编码视频
    encoder.encode_video(
        frames=frames,
        output_path=args.out,
        i_frame_interval=args.i_frame_interval,
        diff_threshold=args.diff_threshold,
        force_i_threshold=args.force_i_threshold,
        variance_threshold=args.variance_threshold,
        codebook_size=args.codebook_size,
        kmeans_max_iter=args.kmeans_max_iter,
        i_frame_weight=args.i_frame_weight,
        max_workers=args.max_workers,
        enabled_segments_bitmap=args.enabled_segments_bitmap,
        enabled_medium_segments_bitmap=args.enabled_medium_segments_bitmap,
        use_parallel=not args.no_parallel
    )
    
    # 生成音频文件
    if audio_data is not None:
        output_base = pathlib.Path(args.out)
        audio_header_path = output_base.parent / f"audio_data.h"
        audio_source_path = output_base.parent / f"audio_data.c"
        
        audio_encoder.write_audio_header(audio_header_path, audio_data, audio_duration)
        audio_encoder.write_audio_source(audio_source_path, audio_data)
        print(f"✓ 已生成音频文件: {audio_header_path.name} / {audio_source_path.name}")

if __name__ == "__main__":
    main()
