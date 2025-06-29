#!/usr/bin/env python3

import argparse
import pathlib

from video_encoder_core import VideoEncoderCore
from video_encoder_utils import extract_frames_from_video
from core_encoder import DEFAULT_UNIFIED_CODEBOOK_SIZE, DEFAULT_ENABLED_SEGMENTS_BITMAP, DEFAULT_ENABLED_MEDIUM_SEGMENTS_BITMAP
from motion_compensation import DEFAULT_UPDATE_THRESHOLD

def main():
    pa = argparse.ArgumentParser(description="Encode to GBA YUV9 with unified codebook")
    pa.add_argument("input")
    pa.add_argument("--duration", type=float, default=5.0)
    pa.add_argument("--full-duration", action="store_true")
    pa.add_argument("--fps", type=int, default=30)
    pa.add_argument("--out", default="video_data")
    pa.add_argument("--i-frame-interval", type=int, default=60)
    pa.add_argument("--diff-threshold", type=float, default=2.5)
    pa.add_argument("--force-i-threshold", type=float, default=0.7)
    pa.add_argument("--variance-threshold", type=float, default=10,
                   help="方差阈值，用于区分纯色块和纹理块（默认5.0）")
    pa.add_argument("--color-fallback-threshold", type=float, default=10,
                   help="色块回退距离阈值，当色块与码本项距离超过此值时回退到纹理块模式（默认10.0）")
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
    pa.add_argument("--motion-compensation", action="store_true", default=True,
                   help="启用运动补偿（默认启用）")
    pa.add_argument("--no-motion-compensation", action="store_true",
                   help="禁用运动补偿")
    pa.add_argument("--motion-update-threshold", type=int, default=DEFAULT_UPDATE_THRESHOLD,
                   help="运动补偿更新阈值：8×8块内允许的最大更新2×2块数（默认8，即50%）")
    pa.add_argument("--audio-sample-rate", type=int, default=16384,
                   help="音频采样率（默认16000 Hz）")
    pa.add_argument("--no-audio", action="store_true",
                   help="禁用音频提取")
    pa.add_argument("--audio-only", action="store_true", help="只导出音频文件，不处理视频")
    pa.add_argument("--volume", type=float, default=100.0, help="音频音量百分比（100为原始，200为放大一倍，50为减半）")
    args = pa.parse_args()

    # audio_only分支提前，直接return
    if args.audio_only:
        from audio_encoder import AudioEncoder
        audio_encoder = AudioEncoder(sample_rate=args.audio_sample_rate)
        frames, actual_output_fps = extract_frames_from_video(
            args.input, args.duration, args.fps, args.full_duration, args.dither
        )
        frame_count = len(frames)
        audio_duration = frame_count / actual_output_fps
        audio_data, i_frame_audio_offsets, frame_audio_offsets = audio_encoder.extract_audio_from_video(
            args.input, audio_duration, frame_count=frame_count, volume_percent=args.volume)
        output_base = pathlib.Path(args.out)
        audio_header_path = output_base.parent / f"audio_data.h"
        audio_source_path = output_base.parent / f"audio_data.c"
        audio_encoder.write_audio_header(audio_header_path, audio_data, audio_duration, i_frame_audio_offsets, frame_audio_offsets)
        audio_encoder.write_audio_source(audio_source_path, audio_data, i_frame_audio_offsets, frame_audio_offsets)
        print(f"✓ 已生成音频文件: {audio_header_path.name} / {audio_source_path.name}")
        return

    # 提取帧
    frames, actual_output_fps = extract_frames_from_video(
        args.input, args.duration, args.fps, args.full_duration, args.dither
    )

    # 检查fps是否被修正
    if args.fps > actual_output_fps:
        print(f"⚠️ 用户输入的FPS({args.fps})高于实际视频FPS({actual_output_fps:.2f})，将自动使用实际FPS进行编码，避免加速。")
    used_fps = actual_output_fps

    # 音频处理
    audio_data = None
    audio_duration = len(frames) / used_fps  # 音频时长与实际导出帧严格一致
    if not args.no_audio:
        from audio_encoder import AudioEncoder
        audio_encoder = AudioEncoder(sample_rate=args.audio_sample_rate)
        audio_data, i_frame_audio_offsets, frame_audio_offsets = audio_encoder.extract_audio_from_video(
            args.input, audio_duration, frame_count=len(frames), volume_percent=args.volume)
        if audio_data is None:
            print("⚠️ 音频提取失败，继续处理视频...")

    # 处理运动补偿选项
    motion_compensation_enabled = args.motion_compensation and not args.no_motion_compensation
    
    # 创建编码器核心
    encoder = VideoEncoderCore()
    
    # 编码视频
    result = encoder.encode_video(
        frames=frames,
        output_path=args.out,
        i_frame_interval=args.i_frame_interval,
        diff_threshold=args.diff_threshold,
        force_i_threshold=args.force_i_threshold,
        variance_threshold=args.variance_threshold,
        color_fallback_threshold=args.color_fallback_threshold,
        codebook_size=args.codebook_size,
        kmeans_max_iter=args.kmeans_max_iter,
        i_frame_weight=args.i_frame_weight,
        max_workers=args.max_workers,
        enabled_segments_bitmap=args.enabled_segments_bitmap,
        enabled_medium_segments_bitmap=args.enabled_medium_segments_bitmap,
        fps=used_fps,
        motion_compensation_enabled=motion_compensation_enabled,
        motion_update_threshold=args.motion_update_threshold
    )
    
    # 解包编码结果
    if len(result) == 3:
        all_data, frame_offsets, i_frame_timestamps = result
    else:
        # 兼容旧版本
        all_data, frame_offsets = result
        i_frame_timestamps = None
    
    # 生成音频文件
    if audio_data is not None:
        output_base = pathlib.Path(args.out)
        audio_header_path = output_base.parent / f"audio_data.h"
        audio_source_path = output_base.parent / f"audio_data.c"
        audio_encoder.write_audio_header(audio_header_path, audio_data, audio_duration, i_frame_audio_offsets, frame_audio_offsets)
        audio_encoder.write_audio_source(audio_source_path, audio_data, i_frame_audio_offsets, frame_audio_offsets)
        print(f"✓ 已生成音频文件: {audio_header_path.name} / {audio_source_path.name}")

if __name__ == "__main__":
    main()
