#!/usr/bin/env python3

import numpy as np
import multiprocessing as mp
from core_encoder import (
    classify_4x4_blocks_unified, generate_unified_codebook, 
    identify_updated_big_blocks, DEFAULT_UNIFIED_CODEBOOK_SIZE, 
    BYTES_PER_BLOCK, calculate_2x2_block_variance
)

def extract_effective_blocks_from_big_blocks(blocks: np.ndarray, big_block_positions: set,
                                           variance_threshold: float = 5.0) -> list:
    """从指定的4x4大块位置提取有效的2x2块"""
    blocks_h, blocks_w = blocks.shape[:2]
    effective_blocks = []
    
    for big_by, big_bx in big_block_positions:
        # 收集4x4大块内的4个2x2小块
        blocks_4x4 = []
        for sub_by in range(2):
            for sub_bx in range(2):
                by = big_by * 2 + sub_by
                bx = big_bx * 2 + sub_bx
                if by < blocks_h and bx < blocks_w:
                    blocks_4x4.append(blocks[by, bx])
                else:
                    blocks_4x4.append(np.zeros(BYTES_PER_BLOCK, dtype=np.uint8))
        
        # 检查是否为色块（每个2x2子块内部一致）
        all_2x2_blocks_are_uniform = True
        for block in blocks_4x4:
            if calculate_2x2_block_variance(block) > variance_threshold:
                all_2x2_blocks_are_uniform = False
                break
        
        if all_2x2_blocks_are_uniform:
            # 色块：生成下采样的2x2块
            downsampled_block = np.zeros(BYTES_PER_BLOCK, dtype=np.uint8)
            
            y_values = []
            d_r_values = []
            d_g_values = []
            d_b_values = []
            
            for block in blocks_4x4:
                # 每个2x2块内部一致，取其内部平均值
                avg_y = np.mean(block[:4])
                y_values.append(int(avg_y))
                d_r_values.append(block[4].view(np.int8))
                d_g_values.append(block[5].view(np.int8))
                d_b_values.append(block[6].view(np.int8))
            
            # 用4个2x2子块的平均值构成下采样块
            downsampled_block[:4] = np.array(y_values, dtype=np.uint8)
            downsampled_block[4] = np.clip(np.mean(d_r_values), -128, 127).astype(np.int8).view(np.uint8)
            downsampled_block[5] = np.clip(np.mean(d_g_values), -128, 127).astype(np.int8).view(np.uint8)
            downsampled_block[6] = np.clip(np.mean(d_b_values), -128, 127).astype(np.int8).view(np.uint8)
            
            effective_blocks.append(downsampled_block)
        else:
            # 纹理块：添加所有4个2x2块
            effective_blocks.extend(blocks_4x4)
    
    return effective_blocks

def process_single_gop_frame(args_tuple):
    """处理单个GOP的单帧 - 用于多进程"""
    (gop_start, gop_end, frame_data_list, variance_threshold, 
     diff_threshold, codebook_size, kmeans_max_iter, i_frame_weight) = args_tuple
    
    try:
        effective_blocks = []
        block_types_list = []
        
        # 处理GOP中的每一帧
        prev_blocks = None
        
        for frame_idx in range(gop_start, gop_end):
            relative_frame_idx = frame_idx - gop_start
            if relative_frame_idx >= len(frame_data_list):
                break
                
            frame_blocks = frame_data_list[relative_frame_idx]
            if frame_blocks.size == 0:
                continue
            
            # 确定帧类型和需要更新的大块
            is_i_frame = (frame_idx == gop_start)  # GOP第一帧是I帧
            
            if is_i_frame:
                # I帧：所有大块都有效
                blocks_h, blocks_w = frame_blocks.shape[:2]
                big_blocks_h = blocks_h // 2
                big_blocks_w = blocks_w // 2
                updated_big_blocks = {(big_by, big_bx) for big_by in range(big_blocks_h) for big_bx in range(big_blocks_w)}
            else:
                # P帧：只有更新的大块有效
                updated_big_blocks = identify_updated_big_blocks(frame_blocks, prev_blocks, diff_threshold)
            
            # 从有效大块中提取2x2块
            frame_effective_blocks = extract_effective_blocks_from_big_blocks(
                frame_blocks, updated_big_blocks, variance_threshold)
            
            # I帧块加权：复制多次以增加在聚类中的影响力
            if is_i_frame:
                weighted_blocks = frame_effective_blocks * i_frame_weight  # 复制i_frame_weight次
                effective_blocks.extend(weighted_blocks)
            else:
                effective_blocks.extend(frame_effective_blocks)
            
            # 生成完整的block_types用于编码（所有大块，不只是有效的）
            frame_blocks_list, block_types = classify_4x4_blocks_unified(frame_blocks, variance_threshold)
            block_types_list.append((frame_idx, block_types))
            
            prev_blocks = frame_blocks.copy()
        
        # 使用有效块生成统一码本
        unified_codebook = generate_unified_codebook(effective_blocks, codebook_size, kmeans_max_iter)
        
        return {
            'gop_start': gop_start,
            'unified_codebook': unified_codebook,
            'block_types_list': block_types_list,
            'total_blocks_count': len(effective_blocks),
            'success': True
        }
        
    except Exception as e:
        return {
            'gop_start': gop_start,
            'error': str(e),
            'success': False
        }

def generate_gop_unified_codebooks(frames: list, i_frame_interval: int,
                                  variance_threshold: float, diff_threshold: float,
                                  codebook_size: int = DEFAULT_UNIFIED_CODEBOOK_SIZE,
                                  kmeans_max_iter: int = 100, i_frame_weight: int = 3,
                                  max_workers: int = None) -> dict:
    """为每个GOP生成统一码本（多进程版本）"""
    print("正在为每个GOP生成统一码本（多进程，基于有效块，I帧加权）...")
    
    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 1)  # 留一个核心给系统
    
    print(f"使用 {max_workers} 个进程并行处理")
    
    gop_codebooks = {}
    
    # 确定I帧位置
    i_frame_positions = []
    for frame_idx in range(len(frames)):
        if frame_idx % i_frame_interval == 0:
            i_frame_positions.append(frame_idx)
    
    # 准备任务参数
    tasks = []
    for gop_idx, gop_start in enumerate(i_frame_positions):
        if gop_idx + 1 < len(i_frame_positions):
            gop_end = i_frame_positions[gop_idx + 1]
        else:
            gop_end = len(frames)
        
        # 提取该GOP的所有帧数据
        frame_data_list = []
        for frame_idx in range(gop_start, gop_end):
            if frame_idx < len(frames):
                frame_data_list.append(frames[frame_idx])
            else:
                break
        
        if frame_data_list:  # 只有当有数据时才添加任务
            task_args = (
                gop_start, gop_end, frame_data_list,
                variance_threshold, diff_threshold, codebook_size,
                kmeans_max_iter, i_frame_weight
            )
            tasks.append(task_args)
    
    total_tasks = len(tasks)
    print(f"总共 {len(i_frame_positions)} 个GOP，{total_tasks} 个处理任务")
    
    # 使用多进程处理
    completed_tasks = 0
    
    # 设置进程启动方法（避免某些平台的问题）
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 如果已经设置过就跳过
    
    with mp.Pool(processes=max_workers) as pool:
        # 提交所有任务
        results = []
        for task_args in tasks:
            result = pool.apply_async(process_single_gop_frame, (task_args,))
            results.append(result)
        
        # 收集结果并显示进度
        processed_results = []
        for i, result in enumerate(results):
            try:
                task_result = result.get(timeout=300)  # 5分钟超时
                processed_results.append(task_result)
                completed_tasks += 1
                
                if completed_tasks % max(1, total_tasks // 20) == 0 or completed_tasks == total_tasks:
                    progress = completed_tasks / total_tasks * 100
                    print(f"  进度: {completed_tasks}/{total_tasks} ({progress:.1f}%)")
                    
            except Exception as e:
                print(f"  ⚠️ 任务 {i} 处理失败: {e}")
                # 创建一个失败的结果
                task_args = tasks[i]
                processed_results.append({
                    'gop_start': task_args[0],
                    'success': False,
                    'error': str(e)
                })
    
    # 组织结果
    failed_count = 0
    for result in processed_results:
        if not result['success']:
            print(f"  ❌ GOP {result['gop_start']} 处理失败: {result.get('error', '未知错误')}")
            failed_count += 1
            continue
        
        gop_start = result['gop_start']
        gop_codebooks[gop_start] = {
            'unified_codebook': result['unified_codebook'],
            'block_types_list': result['block_types_list'],
            'total_blocks_count': result['total_blocks_count']
        }
    
    if failed_count > 0:
        print(f"  ⚠️ 共有 {failed_count} 个任务处理失败")
    else:
        print(f"  ✅ 所有 {total_tasks} 个任务处理完成")
    
    # 验证结果完整性
    for gop_start in i_frame_positions:
        if gop_start not in gop_codebooks:
            print(f"  ⚠️ GOP {gop_start} 缺少数据，使用默认码本")
            # 创建默认码本
            default_codebook = np.zeros((codebook_size, BYTES_PER_BLOCK), dtype=np.uint8)
            gop_codebooks[gop_start] = {
                'unified_codebook': default_codebook,
                'block_types_list': [],
                'total_blocks_count': 0
            }
    
    return gop_codebooks
