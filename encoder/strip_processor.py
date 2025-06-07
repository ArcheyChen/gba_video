#!/usr/bin/env python3
"""
strip_processor.py - 条带处理和并行化
"""

import concurrent.futures
import numpy as np
from config import CODEBOOK_SIZE, BYTES_PER_BLOCK
from codebook import generate_codebook
from frame_encoder import (encode_strip_i_frame_vq, encode_strip_differential_vq)


def process_strip_parallel(args):
    """并行处理单个条带的编码任务"""
    (strip_blocks, prev_strip_blocks, strip_codebook, frame_idx, 
     i_frame_interval, diff_threshold, force_i_threshold, is_first_frame) = args
    
    # 决定是否强制I帧
    force_i_frame = (frame_idx % i_frame_interval == 0) or is_first_frame
    
    if force_i_frame or prev_strip_blocks is None:
        # 编码为I帧
        strip_data = encode_strip_i_frame_vq(strip_blocks, strip_codebook)
        is_i_frame = True
    else:
        # 尝试差分编码
        strip_data, is_i_frame = encode_strip_differential_vq(
            strip_blocks, prev_strip_blocks, strip_codebook,
            diff_threshold, force_i_threshold
        )
    
    return strip_data, is_i_frame


def generate_codebook_parallel(args):
    """并行生成单个条带的码表"""
    gop_idx, strip_idx, strip_blocks_samples, kmeans_max_iter = args
    
    # 合并所有样本
    if strip_blocks_samples:
        all_blocks = np.vstack(strip_blocks_samples)
        codebook, effective_size = generate_codebook(all_blocks, CODEBOOK_SIZE, kmeans_max_iter)
        total_samples = len(all_blocks)
    else:
        codebook = np.zeros((CODEBOOK_SIZE, BYTES_PER_BLOCK), dtype=np.uint8)
        effective_size = 0
        total_samples = 0
    
    return {
        'gop_idx': gop_idx,
        'strip_idx': strip_idx,
        'codebook': codebook,
        'total_samples': total_samples,
        'effective_size': effective_size,
        'utilization': effective_size / CODEBOOK_SIZE if CODEBOOK_SIZE > 0 else 0
    }


def generate_gop_codebooks(frames: list, strip_count: int, i_frame_interval: int, 
                          kmeans_max_iter: int = 100, max_workers: int = None) -> dict:
    """为每个GOP（Group of Pictures）的每个条带生成码表（并行优化版本）"""
    print("正在为每个GOP生成条带码表...")
    
    gop_codebooks = {}  # {gop_start_frame: [strip_codebooks]}
    
    # 确定所有I帧的位置
    i_frame_positions = []
    for frame_idx in range(len(frames)):
        if frame_idx % i_frame_interval == 0:
            i_frame_positions.append(frame_idx)
    
    # 准备所有码表生成任务
    codebook_tasks = []
    gop_info = []  # 存储GOP信息用于后续组装
    
    for gop_idx, gop_start in enumerate(i_frame_positions):
        # 确定GOP的结束位置
        if gop_idx + 1 < len(i_frame_positions):
            gop_end = i_frame_positions[gop_idx + 1]
        else:
            gop_end = len(frames)
        
        print(f"  准备GOP {gop_idx}: 帧 {gop_start} 到 {gop_end-1}")
        gop_info.append((gop_start, gop_end))
        
        # 为GOP中的每个条带准备任务
        for strip_idx in range(strip_count):
            # 收集该GOP中该条带的所有块数据
            strip_blocks_samples = []
            
            for frame_idx in range(gop_start, gop_end):
                strip_blocks = frames[frame_idx][strip_idx]
                if strip_blocks.size > 0:
                    blocks_flat = strip_blocks.reshape(-1, BYTES_PER_BLOCK)
                    strip_blocks_samples.append(blocks_flat)
            
            # 添加码表生成任务
            task_args = (gop_idx, strip_idx, strip_blocks_samples, kmeans_max_iter)
            codebook_tasks.append(task_args)
    
    # 并行生成所有码表
    print(f"  并行生成 {len(codebook_tasks)} 个码表...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(generate_codebook_parallel, task): task
            for task in codebook_tasks
        }
        
        # 收集结果
        codebook_results = {}
        for future in concurrent.futures.as_completed(future_to_task):
            result = future.result()
            gop_idx = result['gop_idx']
            strip_idx = result['strip_idx']
            
            if gop_idx not in codebook_results:
                codebook_results[gop_idx] = {}
            codebook_results[gop_idx][strip_idx] = result
    
    # 组装最终结果
    for gop_idx, (gop_start, gop_end) in enumerate(gop_info):
        gop_codebooks[gop_start] = []
        
        for strip_idx in range(strip_count):
            result = codebook_results[gop_idx][strip_idx]
            gop_codebooks[gop_start].append({
                'codebook': result['codebook'],
                'total_samples': result['total_samples'],
                'effective_size': result['effective_size'],
                'utilization': result['utilization']
            })
    
    return gop_codebooks


def process_strip_parallel_with_gop(args):
    """并行处理单个条带的编码任务（使用GOP码表）"""
    (strip_blocks, prev_strip_blocks, strip_codebook, frame_idx, 
     i_frame_interval, diff_threshold, force_i_threshold, is_first_frame) = args
    
    # 决定是否强制I帧
    force_i_frame = (frame_idx % i_frame_interval == 0) or is_first_frame
    
    if force_i_frame or prev_strip_blocks is None:
        # 编码为I帧
        strip_data = encode_strip_i_frame_vq(strip_blocks, strip_codebook)
        is_i_frame = True
    else:
        # 尝试差分编码
        strip_data, is_i_frame = encode_strip_differential_vq(
            strip_blocks, prev_strip_blocks, strip_codebook,
            diff_threshold, force_i_threshold
        )
    
    return strip_data, is_i_frame