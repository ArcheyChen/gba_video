#!/usr/bin/env python3
"""
main.py - GBA视频编码器主程序
把视频/图片序列转成 GBA Mode3 YUV9 数据（支持条带帧间差分 + 向量量化）
输出 video_data.c / video_data.h
"""

import argparse
import cv2
import numpy as np
import pathlib
import struct
import concurrent.futures

from config import (WIDTH, HEIGHT, DEFAULT_STRIP_COUNT, DEFAULT_FPS, DEFAULT_DURATION,
                   DEFAULT_I_FRAME_INTERVAL, DEFAULT_DIFF_THRESHOLD, 
                   DEFAULT_FORCE_I_THRESHOLD, DEFAULT_KMEANS_MAX_ITER,
                   BLOCK_W, BLOCK_H, BYTES_PER_BLOCK, CODEBOOK_SIZE)
from utils import calculate_strip_heights, get_current_codebooks
from yuv_converter import pack_yuv420_strip
from strip_processor import (generate_gop_codebooks, process_strip_parallel_with_gop)
from file_writer import write_header, write_source


def main():
    pa = argparse.ArgumentParser(description="Encode to GBA YUV9 with strip-based inter-frame compression and vector quantization")
    pa.add_argument("input")
    pa.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                   help=f"视频编码时长，单位秒（默认{DEFAULT_DURATION}）。与--full-duration互斥")
    pa.add_argument("--full-duration", action="store_true",
                   help="编码整个视频的完整时长，忽略--duration参数")
    pa.add_argument("--fps", type=int, default=DEFAULT_FPS)
    pa.add_argument("--out", default="video_data")
    pa.add_argument("--strip-count", type=int, default=DEFAULT_STRIP_COUNT,
                   help=f"条带数量（默认{DEFAULT_STRIP_COUNT}）")
    pa.add_argument("--i-frame-interval", type=int, default=DEFAULT_I_FRAME_INTERVAL, 
                   help=f"间隔多少帧插入一个I帧（默认{DEFAULT_I_FRAME_INTERVAL}）")
    pa.add_argument("--diff-threshold", type=float, default=DEFAULT_DIFF_THRESHOLD,
                   help=f"差异阈值，超过此值的块将被更新（默认{DEFAULT_DIFF_THRESHOLD}，Y通道平均差值）")
    pa.add_argument("--force-i-threshold", type=float, default=DEFAULT_FORCE_I_THRESHOLD,
                   help=f"当需要更新的块比例超过此值时，强制生成I帧（默认{DEFAULT_FORCE_I_THRESHOLD}）")
    pa.add_argument("--kmeans-max-iter", type=int, default=DEFAULT_KMEANS_MAX_ITER,
                   help=f"K-Means聚类最大迭代次数（默认{DEFAULT_KMEANS_MAX_ITER}）")
    pa.add_argument("--threads", type=int, default=None,
                   help="并行处理线程数（默认为CPU核心数）")
    args = pa.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit("❌ 打不开输入文件")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    every = int(round(src_fps / args.fps))
    
    # 根据--full-duration参数决定是否编码整个视频
    if args.full_duration:
        # 获取视频总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        grab_max = total_frames
        actual_duration = total_frames / src_fps
        print(f"编码整个视频: {total_frames} 帧，时长 {actual_duration:.2f} 秒")
    else:
        grab_max = int(args.duration * src_fps)
        print(f"编码时长: {args.duration} 秒 ({grab_max} 帧)")

    # 计算条带高度
    strip_heights = calculate_strip_heights(HEIGHT, args.strip_count)
    print(f"条带配置: {args.strip_count} 个条带，高度分别为: {strip_heights}")

    frames = []
    idx = 0
    print("正在提取帧...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        while idx < grab_max:
            ret, frm = cap.read()
            if not ret:
                break
            if idx % every == 0:
                frm = cv2.resize(frm, (WIDTH, HEIGHT), cv2.INTER_AREA)
                # 多线程处理每个条带
                strip_y_list = []
                y = 0
                for strip_height in strip_heights:
                    strip_y_list.append((frm, y, strip_height))
                    y += strip_height
                # 提交任务
                future_to_idx = {
                    executor.submit(pack_yuv420_strip, *args): i
                    for i, args in enumerate(strip_y_list)
                }
                frame_strips = [None] * len(strip_y_list)
                for future in concurrent.futures.as_completed(future_to_idx):
                    i = future_to_idx[future]
                    frame_strips[i] = future.result()
                frames.append(frame_strips)
                
                if len(frames) % 30 == 0:
                    print(f"  已提取 {len(frames)} 帧")
            idx += 1
    cap.release()

    if not frames:
        raise SystemExit("❌ 没有任何帧被采样")

    print(f"总共提取了 {len(frames)} 帧")

    # 为每个GOP生成码表
    gop_codebooks = generate_gop_codebooks(frames, args.strip_count, args.i_frame_interval, 
                                          args.kmeans_max_iter, args.threads)

    # 编码所有帧
    print("正在编码帧...")
    encoded_frames = []
    frame_offsets = []
    current_offset = 0
    prev_strips = [None] * args.strip_count
    i_frame_count = [0] * args.strip_count
    p_frame_count = [0] * args.strip_count
    
    # 计算总的码表统计
    all_codebook_stats = []
    for gop_start, gop_data in gop_codebooks.items():
        all_codebook_stats.extend(gop_data)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        for frame_idx, current_strips in enumerate(frames):
            frame_offsets.append(current_offset)
            
            # 获取当前帧应该使用的码表
            current_codebooks = get_current_codebooks(frame_idx, gop_codebooks, args.i_frame_interval)
            
            # 准备并行编码任务
            tasks = []
            for strip_idx, current_strip in enumerate(current_strips):
                task_args = (
                    current_strip, prev_strips[strip_idx], current_codebooks[strip_idx],
                    frame_idx, args.i_frame_interval, args.diff_threshold, 
                    args.force_i_threshold, frame_idx == 0
                )
                tasks.append(task_args)
            
            # 并行编码所有条带
            future_to_strip = {
                executor.submit(process_strip_parallel_with_gop, task): strip_idx
                for strip_idx, task in enumerate(tasks)
            }
            
            frame_data = bytearray()
            strip_results = [None] * args.strip_count
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_strip):
                strip_idx = future_to_strip[future]
                strip_data, is_i_frame = future.result()
                strip_results[strip_idx] = (strip_data, is_i_frame)
                
                if is_i_frame:
                    i_frame_count[strip_idx] += 1
                else:
                    p_frame_count[strip_idx] += 1
            
            # 按顺序组装帧数据
            for strip_idx, (strip_data, _) in enumerate(strip_results):
                # 存储条带长度（2字节）+ 条带数据
                frame_data.extend(struct.pack('<H', len(strip_data)))
                frame_data.extend(strip_data)
                
                # 更新参考条带
                prev_strips[strip_idx] = current_strips[strip_idx].copy() if current_strips[strip_idx].size > 0 else None
            
            encoded_frames.append(bytes(frame_data))
            current_offset += len(frame_data)
            
            if frame_idx % 30 == 0 or frame_idx == len(frames) - 1:
                print(f"  已编码 {frame_idx + 1}/{len(frames)} 帧")
    
    # 合并所有数据
    all_data = b''.join(encoded_frames)
    
    # 写入文件
    write_header(pathlib.Path(args.out).with_suffix(".h"), len(frames), len(all_data), 
                args.strip_count, strip_heights)
    write_source(pathlib.Path(args.out).with_suffix(".c"), all_data, frame_offsets, strip_heights)
    
    # 统计信息
    total_original_size = 0
    for strip_height in strip_heights:
        blocks_in_strip = (strip_height // BLOCK_H) * (WIDTH // BLOCK_W)
        total_original_size += blocks_in_strip * BYTES_PER_BLOCK
    
    original_size = len(frames) * total_original_size
    compressed_size = len(all_data)
    compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
    
    # 计算平均码表利用率
    avg_utilization = np.mean([stat['utilization'] for stat in all_codebook_stats]) * 100
    avg_effective_size = np.mean([stat['effective_size'] for stat in all_codebook_stats])
    
    print(f"\n✅ 编码完成：")
    print(f"   总帧数: {len(frames)}")
    print(f"   条带数: {args.strip_count}")
    print(f"   条带高度: {strip_heights}")
    print(f"   码表大小: {CODEBOOK_SIZE}")
    print(f"   GOP数量: {len(gop_codebooks)}")
    print(f"   平均有效码字数: {avg_effective_size:.1f}")
    print(f"   平均码表利用率: {avg_utilization:.1f}%")
    print(f"   K-Means迭代次数: {args.kmeans_max_iter}")
    
    for strip_idx in range(args.strip_count):
        print(f"   条带{strip_idx}: I帧{i_frame_count[strip_idx]}, P帧{p_frame_count[strip_idx]}")
    
    print(f"   原始大小: {original_size:,} bytes")
    print(f"   压缩后大小: {compressed_size:,} bytes")
    print(f"   压缩比: {compression_ratio:.2f}x")
    print(f"   I帧间隔: {args.i_frame_interval}")
    print(f"   差异阈值: {args.diff_threshold}")


if __name__ == "__main__":
    main()