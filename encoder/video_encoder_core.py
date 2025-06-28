#!/usr/bin/env python3

import numpy as np
import pathlib
from collections import defaultdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import pickle
from numba import njit

from core_encoder import *
from gop_processor import generate_gop_unified_codebooks
from video_encoder_stats import EncodingStats
from video_encoder_utils import write_header, write_source
from motion_compensation import motion_stats

# 全局函数，用于多进程
def sort_single_gop_worker(args):
    """批量GOP排序的worker函数"""
    gop_batch, frames, i_frame_interval, diff_threshold = args
    results = []
    for gop_start, gop_data in gop_batch:
        codebook = gop_data['unified_codebook'].copy()
        counts = np.zeros(len(codebook), dtype=int)
        gop_end = min(gop_start + i_frame_interval, len(frames))
        for fid in range(gop_start + 1, gop_end):
            cur = frames[fid]
            prev = frames[fid - 1]
            updated = identify_updated_big_blocks(cur, prev, diff_threshold)
            bt_map = None
            for fno, bt in gop_data['block_types_list']:
                if fno == fid:
                    bt_map = bt; break
            for by, bx in updated:
                is_color = bt_map and bt_map.get((by, bx), ('detail',))[0] == 'color'
                if not is_color:
                    for sy in (0,1):
                        for sx in (0,1):
                            y, x = by*2+sy, bx*2+sx
                            if y < cur.shape[0] and x < cur.shape[1]:
                                b = cur[y, x]
                                idx = quantize_blocks_unified(b.reshape(1, -1), codebook)[0]
                                counts[idx] += 1
        max_count = counts.max()
        min_count = counts.min()
        total_usage = counts.sum()
        if max_count > min_count and total_usage > 0:
            print(f"  GOP {gop_start}: 最大使用频次 {max_count}, 最小使用频次 {min_count}, 总使用次数 {total_usage}")
            order = np.argsort(-counts, kind='stable')
            gop_data['unified_codebook'] = codebook[order]
            index_mapping = np.zeros(len(codebook), dtype=int)
            for new_idx, old_idx in enumerate(order):
                index_mapping[old_idx] = new_idx
            for fid, bt in gop_data['block_types_list']:
                if bt is not None:
                    for (big_by, big_bx), (block_type, block_indices) in bt.items():
                        if block_type == 'detail':
                            new_indices = []
                            for old_idx in block_indices:
                                if old_idx < len(index_mapping):
                                    new_indices.append(index_mapping[old_idx])
                                else:
                                    new_indices.append(old_idx)
                            bt[(big_by, big_bx)] = (block_type, new_indices)
        results.append((gop_start, gop_data))
    return results

# 简化的统计数据结构，用于多进程
class SimpleStats:
    """简化的统计数据结构，用于多进程传输"""
    def __init__(self):
        self.total_frames = 0
        self.i_frames = 0
        self.p_frames = 0
        self.forced_i_frames = 0
        self.threshold_i_frames = 0
        self.i_frame_bytes = 0
        self.p_frame_bytes = 0
        self.codebook_bytes = 0
        self.index_bytes = 0
        self.p_overhead_bytes = 0
        self.small_updates = 0
        self.medium_updates = 0
        self.full_updates = 0
        self.small_bytes = 0
        self.medium_bytes = 0
        self.full_bytes = 0
        self.small_segments = {}
        self.medium_segments = {}
        self.small_blocks_per_update = []
        self.medium_blocks_per_update = []
        self.full_blocks_per_update = []
        # 新增：块数分布统计
        self.small_blocks_distribution = {1: 0, 2: 0, 3: 0, 4: 0}
        self.medium_blocks_distribution = {1: 0, 2: 0, 3: 0, 4: 0}
        self.full_blocks_distribution = {1: 0, 2: 0, 3: 0, 4: 0}
        # 运动补偿统计信息
        self.motion_stats_dict = None

def encode_frame_chunk_worker(args):
    """帧编码chunk的worker函数"""
    start_idx, end_idx, frames_chunk, gop_codebooks_chunk, i_frame_interval, diff_threshold, force_i_threshold, enabled_segments_bitmap, enabled_medium_segments_bitmap, codebook_size, color_fallback_threshold, motion_compensation_enabled, motion_update_threshold = args
    
    # 重置当前进程的运动补偿统计
    motion_stats.reset()
    
    encoded_frames = []
    frame_offsets = []
    current_offset = 0
    prev_frame = None
    
    # 使用简化的统计对象
    local_stats = SimpleStats()
    
    for frame_idx in range(start_idx, end_idx):
        if frame_idx >= len(frames_chunk) + start_idx:
            break
            
        current_frame = frames_chunk[frame_idx - start_idx]
        frame_offsets.append(current_offset)
        
        # 找到当前GOP
        gop_start = (frame_idx // i_frame_interval) * i_frame_interval
        gop_data = gop_codebooks_chunk.get(gop_start)
        
        if gop_data is None:
            # 如果找不到GOP数据，使用默认处理
            print(f"  ⚠️ 帧 {frame_idx} 找不到GOP {gop_start} 的码本数据")
            # 创建一个默认的码本和block_types
            unified_codebook = np.zeros((codebook_size, 7), dtype=np.uint8)
            _, block_types = classify_4x4_blocks_unified(current_frame, 5.0)
        else:
            unified_codebook = gop_data['unified_codebook']
            
            # 找到当前帧的block_types，处理缺失的情况
            block_types = None
            for fid, bt in gop_data['block_types_list']:
                if fid == frame_idx:
                    block_types = bt
                    break
            
            # 如果block_types仍然为None，生成默认的block_types
            if block_types is None:
                # 临时生成block_types
                _, block_types = classify_4x4_blocks_unified(current_frame, 5.0)
        
        force_i_frame = (frame_idx % i_frame_interval == 0) or frame_idx == 0
        
        if force_i_frame or prev_frame is None:
            frame_data = encode_i_frame_unified(
                current_frame, unified_codebook, block_types, color_fallback_threshold
            )
            is_i_frame = True
            
            # 计算码本和索引大小（使用输出的YUV420格式）
            OUTPUT_BYTES_PER_BLOCK = 6  # YUV420格式：4Y + 1Cb + 1Cr
            codebook_size_bytes = codebook_size * OUTPUT_BYTES_PER_BLOCK
            index_size = len(frame_data) - 1 - codebook_size_bytes
            
            # 更新统计
            local_stats.total_frames += 1
            local_stats.i_frames += 1
            if force_i_frame:
                local_stats.forced_i_frames += 1
            else:
                local_stats.threshold_i_frames += 1
            local_stats.i_frame_bytes += len(frame_data)
            local_stats.codebook_bytes += codebook_size_bytes
            local_stats.index_bytes += max(0, index_size)
        else:
            # P帧编码前，先将未更新的块从前一帧累积到当前帧，避免渐变残影
            accumulated_frame = accumulate_unchanged_blocks(current_frame, prev_frame, diff_threshold)
            
            frame_data, is_i_frame, used_zones, color_updates, detail_updates, small_updates, medium_updates, full_updates, small_bytes, medium_bytes, full_bytes, small_segments, medium_segments, small_blocks_per_update, medium_blocks_per_update, full_blocks_per_update = encode_p_frame_unified(
                accumulated_frame, prev_frame,
                unified_codebook, block_types,
                diff_threshold, force_i_threshold, enabled_segments_bitmap,
                enabled_medium_segments_bitmap, color_fallback_threshold,
                motion_compensation_enabled, motion_update_threshold
            )
            
            if is_i_frame:
                OUTPUT_BYTES_PER_BLOCK = 6  # YUV420格式：4Y + 1Cb + 1Cr
                codebook_size_bytes = codebook_size * OUTPUT_BYTES_PER_BLOCK
                index_size = len(frame_data) - 1 - codebook_size_bytes
                
                # 更新统计
                local_stats.total_frames += 1
                local_stats.i_frames += 1
                local_stats.threshold_i_frames += 1
                local_stats.i_frame_bytes += len(frame_data)
                local_stats.codebook_bytes += codebook_size_bytes
                local_stats.index_bytes += max(0, index_size)
            else:
                # 更新统计
                local_stats.total_frames += 1
                local_stats.p_frames += 1
                local_stats.p_frame_bytes += len(frame_data)
                local_stats.small_updates += small_updates
                local_stats.medium_updates += medium_updates
                local_stats.full_updates += full_updates
                local_stats.small_bytes += small_bytes
                local_stats.medium_bytes += medium_bytes
                local_stats.full_bytes += full_bytes
                
                # 合并段使用统计
                for seg_idx, count in small_segments.items():
                    local_stats.small_segments[seg_idx] = local_stats.small_segments.get(seg_idx, 0) + count
                for seg_idx, count in medium_segments.items():
                    local_stats.medium_segments[seg_idx] = local_stats.medium_segments.get(seg_idx, 0) + count
                
                # 合并块数统计
                local_stats.small_blocks_per_update.extend(small_blocks_per_update)
                local_stats.medium_blocks_per_update.extend(medium_blocks_per_update)
                local_stats.full_blocks_per_update.extend(full_blocks_per_update)
                
                # 统计块数分布
                for block_count in small_blocks_per_update:
                    if 1 <= block_count <= 4:
                        local_stats.small_blocks_distribution[block_count] += 1
                for block_count in medium_blocks_per_update:
                    if 1 <= block_count <= 4:
                        local_stats.medium_blocks_distribution[block_count] += 1
                for block_count in full_blocks_per_update:
                    if 1 <= block_count <= 4:
                        local_stats.full_blocks_distribution[block_count] += 1
        
        encoded_frames.append(frame_data)
        current_offset += len(frame_data)
        
        # 更新前一帧引用：I帧直接使用当前帧，P帧使用累积后的帧
        if force_i_frame or prev_frame is None:
            prev_frame = current_frame.copy() if current_frame.size > 0 else None
        else:
            # P帧使用累积后的帧作为下一帧的参考，以便累积渐变差异
            prev_frame = accumulated_frame.copy() if accumulated_frame.size > 0 else None
        
        if frame_idx % 30 == 0 or frame_idx == end_idx - 1:
            print(f"  已编码 {frame_idx + 1}/{end_idx} 帧")
    
    # 收集运动补偿统计信息
    local_stats.motion_stats_dict = motion_stats.get_stats_dict()
    
    # 返回编码结果和统计信息
    return encoded_frames, frame_offsets, start_idx, local_stats

class VideoEncoderCore:
    """视频编码器核心类"""
    
    def __init__(self):
        self.encoding_stats = EncodingStats()
    
    def encode_video(self, frames, output_path, i_frame_interval=60, diff_threshold=2.0, 
                    force_i_threshold=0.7, variance_threshold=5.0, color_fallback_threshold=50.0, codebook_size=256,
                    kmeans_max_iter=200, i_frame_weight=3, max_workers=None,
                    enabled_segments_bitmap=0xFFFF, enabled_medium_segments_bitmap=0x0F,
                    fps=30.0, motion_compensation_enabled=True, motion_update_threshold=8):
        """编码视频的主要流程"""
        
        print(f"码本配置: 统一码本{codebook_size}项")
        
        # 生成统一码本（传入max_workers参数）
        gop_codebooks = generate_gop_unified_codebooks(
            frames, i_frame_interval, 
            variance_threshold, diff_threshold, codebook_size, 
            kmeans_max_iter, i_frame_weight, max_workers
        )

        # 并行执行码本排序和帧编码
        print("正在并行执行码本排序和帧编码...")
        start_time = time.time()
        
        # 使用进程池并行执行
        if max_workers is None:
            max_workers = max(1, mp.cpu_count() - 1)
        
        # 第一阶段：并行码本排序
        print("正在根据使用频次对码本进行排序...")
        sorted_gop_codebooks = self._parallel_sort_codebooks(
            frames, gop_codebooks, i_frame_interval, diff_threshold, max_workers
        )
        
        # 第二阶段：并行帧编码
        print("正在编码帧...")
        encoded_frames, frame_offsets = self._parallel_encode_frames(
            frames, sorted_gop_codebooks, i_frame_interval, diff_threshold, force_i_threshold,
            enabled_segments_bitmap, enabled_medium_segments_bitmap, codebook_size, color_fallback_threshold, 
            motion_compensation_enabled, motion_update_threshold, max_workers
        )
        
        end_time = time.time()
        print(f"执行完成，耗时: {end_time - start_time:.2f}秒")
        
        all_data = b''.join(encoded_frames)
        
        # 计算I帧时间戳（用实际fps）
        i_frame_timestamps = []
        for frame_idx in range(len(frames)):
            if frame_idx % i_frame_interval == 0 or frame_idx == 0:
                timestamp = frame_idx / fps
                i_frame_timestamps.append(timestamp)
        
        # 写入文件
        write_header(pathlib.Path(output_path).with_suffix(".h"), len(frames), len(all_data), 
                    codebook_size, fps)
        write_source(pathlib.Path(output_path).with_suffix(".c"), all_data, frame_offsets)
        
        # 打印详细统计
        self.encoding_stats.print_summary(len(frames), len(all_data))
        
        return all_data, frame_offsets, i_frame_timestamps
    
    def _parallel_sort_codebooks(self, frames, gop_codebooks, i_frame_interval, diff_threshold, max_workers):
        """并行码本排序（每个进程处理多个GOP，减少通讯）"""
        gop_items = list(gop_codebooks.items())
        total_gops = len(gop_items)
        # 计算每个进程分配多少GOP
        num_workers = min(max_workers, total_gops)
        gops_per_worker = (total_gops + num_workers - 1) // num_workers
        worker_args = []
        for i in range(0, total_gops, gops_per_worker):
            batch = gop_items[i:i+gops_per_worker]
            args = (batch, frames, i_frame_interval, diff_threshold)
            worker_args.append(args)
        sorted_gop_codebooks = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(sort_single_gop_worker, args) for args in worker_args]
            for future in as_completed(futures):
                batch_result = future.result()
                for gop_start, gop_data in batch_result:
                    sorted_gop_codebooks[gop_start] = gop_data
        return sorted_gop_codebooks
    
    def _parallel_encode_frames(self, frames, gop_codebooks, i_frame_interval, diff_threshold, 
                               force_i_threshold, enabled_segments_bitmap, enabled_medium_segments_bitmap, 
                               codebook_size, color_fallback_threshold, motion_compensation_enabled, 
                               motion_update_threshold, max_workers):
        """并行帧编码"""
        # 优化分块策略：按GOP分组，减少通信开销
        num_frames = len(frames)
        num_gops = (num_frames + i_frame_interval - 1) // i_frame_interval
        
        # 如果GOP数量少于进程数，按GOP分组
        if num_gops <= max_workers:
            # 每个进程处理一个或多个完整GOP
            gop_per_worker = max(1, num_gops // max_workers)
            chunk_data_list = []
            
            for i in range(0, num_gops, gop_per_worker):
                end_gop = min(i + gop_per_worker, num_gops)
                start_frame = i * i_frame_interval
                end_frame = min(end_gop * i_frame_interval, num_frames)
                
                # 只传递该GOP范围内的帧和码本
                frames_chunk = frames[start_frame:end_frame]
                gop_codebooks_chunk = {}
                for gop_start in range(i * i_frame_interval, end_gop * i_frame_interval, i_frame_interval):
                    if gop_start in gop_codebooks:
                        gop_codebooks_chunk[gop_start] = gop_codebooks[gop_start]
                
                chunk_data = (start_frame, end_frame, frames_chunk, gop_codebooks_chunk, i_frame_interval, 
                            diff_threshold, force_i_threshold, enabled_segments_bitmap, 
                            enabled_medium_segments_bitmap, codebook_size, color_fallback_threshold,
                            motion_compensation_enabled, motion_update_threshold)
                chunk_data_list.append(chunk_data)
        else:
            # 如果GOP数量很多，按帧数分组，但确保每个分块包含完整的GOP
            frames_per_worker = max(i_frame_interval, num_frames // max_workers)
            # 确保分块边界对齐到GOP边界
            frames_per_worker = ((frames_per_worker + i_frame_interval - 1) // i_frame_interval) * i_frame_interval
            
            chunk_data_list = []
            for i in range(0, num_frames, frames_per_worker):
                end_frame = min(i + frames_per_worker, num_frames)
                frames_chunk = frames[i:end_frame]
                
                # 计算该分块涉及的GOP范围
                start_gop = i // i_frame_interval
                end_gop = (end_frame + i_frame_interval - 1) // i_frame_interval
                
                # 只传递相关的码本
                gop_codebooks_chunk = {}
                for gop_start in range(start_gop * i_frame_interval, end_gop * i_frame_interval, i_frame_interval):
                    if gop_start in gop_codebooks:
                        gop_codebooks_chunk[gop_start] = gop_codebooks[gop_start]
                
                chunk_data = (i, end_frame, frames_chunk, gop_codebooks_chunk, i_frame_interval, 
                            diff_threshold, force_i_threshold, enabled_segments_bitmap, 
                            enabled_medium_segments_bitmap, codebook_size, color_fallback_threshold,
                            motion_compensation_enabled, motion_update_threshold)
                chunk_data_list.append(chunk_data)
        
        print(f"  并行编码：{len(chunk_data_list)}个分块，每个分块约{len(frames) // len(chunk_data_list)}帧")
        
        # 使用进程池并行处理所有分块
        with ProcessPoolExecutor(max_workers=min(len(chunk_data_list), max_workers)) as executor:
            futures = [executor.submit(encode_frame_chunk_worker, chunk_data) for chunk_data in chunk_data_list]
            
            # 收集结果并按顺序重组
            results = []
            for future in as_completed(futures):
                results.append(future.result())
            
            # 按start_idx排序
            results.sort(key=lambda x: x[2])
            
            # 合并结果和统计信息
            all_encoded_frames = []
            all_frame_offsets = []
            current_offset = 0
            
            # 合并所有进程的统计信息
            for encoded_frames, frame_offsets, _, local_stats in results:
                all_encoded_frames.extend(encoded_frames)
                # 调整偏移量
                adjusted_offsets = [current_offset + offset for offset in frame_offsets]
                all_frame_offsets.extend(adjusted_offsets)
                current_offset += sum(len(frame_data) for frame_data in encoded_frames)
                
                # 合并简化的统计信息
                self.encoding_stats.total_frames_processed += local_stats.total_frames
                self.encoding_stats.total_i_frames += local_stats.i_frames
                self.encoding_stats.total_p_frames += local_stats.p_frames
                self.encoding_stats.forced_i_frames += local_stats.forced_i_frames
                self.encoding_stats.threshold_i_frames += local_stats.threshold_i_frames
                self.encoding_stats.total_i_frame_bytes += local_stats.i_frame_bytes
                self.encoding_stats.total_p_frame_bytes += local_stats.p_frame_bytes
                self.encoding_stats.total_codebook_bytes += local_stats.codebook_bytes
                self.encoding_stats.total_index_bytes += local_stats.index_bytes
                self.encoding_stats.small_codebook_updates += local_stats.small_updates
                self.encoding_stats.medium_codebook_updates += local_stats.medium_updates
                self.encoding_stats.full_codebook_updates += local_stats.full_updates
                self.encoding_stats.small_codebook_bytes += local_stats.small_bytes
                self.encoding_stats.medium_codebook_bytes += local_stats.medium_bytes
                self.encoding_stats.full_codebook_bytes += local_stats.full_bytes
                
                # 合并段使用统计
                for seg_idx, count in local_stats.small_segments.items():
                    self.encoding_stats.small_segment_usage[seg_idx] += count
                for seg_idx, count in local_stats.medium_segments.items():
                    self.encoding_stats.medium_segment_usage[seg_idx] += count
                
                # 合并块数统计
                self.encoding_stats.small_codebook_blocks_per_update.extend(local_stats.small_blocks_per_update)
                self.encoding_stats.medium_codebook_blocks_per_update.extend(local_stats.medium_blocks_per_update)
                self.encoding_stats.full_codebook_blocks_per_update.extend(local_stats.full_blocks_per_update)
                
                # 合并块数分布统计
                for block_count in [1, 2, 3, 4]:
                    self.encoding_stats.small_blocks_distribution[block_count] += local_stats.small_blocks_distribution.get(block_count, 0)
                    self.encoding_stats.medium_blocks_distribution[block_count] += local_stats.medium_blocks_distribution.get(block_count, 0)
                    self.encoding_stats.full_blocks_distribution[block_count] += local_stats.full_blocks_distribution.get(block_count, 0)
                
                # 合并运动补偿统计
                if local_stats.motion_stats_dict is not None:
                    self.encoding_stats.merge_motion_compensation_stats(local_stats.motion_stats_dict)
        
        return all_encoded_frames, all_frame_offsets 