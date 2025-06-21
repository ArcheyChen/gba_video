#!/usr/bin/env python3

import numpy as np
import pathlib
from collections import defaultdict

from core_encoder import *
from gop_processor import generate_gop_unified_codebooks
from video_encoder_stats import EncodingStats
from video_encoder_utils import write_header, write_source

class VideoEncoderCore:
    """视频编码器核心类"""
    
    def __init__(self):
        self.encoding_stats = EncodingStats()
    
    def encode_video(self, frames, output_path, i_frame_interval=60, diff_threshold=2.0, 
                    force_i_threshold=0.7, variance_threshold=5.0, codebook_size=256,
                    kmeans_max_iter=200, i_frame_weight=3, max_workers=None,
                    enabled_segments_bitmap=0xFFFF, enabled_medium_segments_bitmap=0x0F):
        """编码视频的主要流程"""
        
        print(f"码本配置: 统一码本{codebook_size}项")
        
        # 生成统一码本（传入max_workers参数）
        gop_codebooks = generate_gop_unified_codebooks(
            frames, i_frame_interval, 
            variance_threshold, diff_threshold, codebook_size, 
            kmeans_max_iter, i_frame_weight, max_workers
        )

        # 基于 GOP 内 P 帧纹理块使用频次，对每个码本项降序重排
        print("正在根据使用频次对码本进行排序...")
        self._sort_codebooks_by_usage(frames, gop_codebooks, i_frame_interval, diff_threshold)
        
        # 编码所有帧
        print("正在编码帧...")
        encoded_frames, frame_offsets = self._encode_all_frames(
            frames, gop_codebooks, i_frame_interval, diff_threshold, force_i_threshold,
            enabled_segments_bitmap, enabled_medium_segments_bitmap, codebook_size
        )
        
        all_data = b''.join(encoded_frames)
        
        # 写入文件
        write_header(pathlib.Path(output_path).with_suffix(".h"), len(frames), len(all_data), 
                    codebook_size, 30.0)  # 假设30fps
        write_source(pathlib.Path(output_path).with_suffix(".c"), all_data, frame_offsets)
        
        # 打印详细统计
        self.encoding_stats.print_summary(len(frames), len(all_data))
        
        return all_data, frame_offsets
    
    def _sort_codebooks_by_usage(self, frames, gop_codebooks, i_frame_interval, diff_threshold):
        """根据使用频次对码本进行排序"""
        for gop_start, gop_data in gop_codebooks.items():
            codebook = gop_data['unified_codebook']
            counts = np.zeros(len(codebook), dtype=int)
            
            # GOP 范围：起始帧下一个到下一个 I 帧
            gop_end = min(gop_start + i_frame_interval, len(frames))
            
            # 统计每个码本项的使用频次
            for fid in range(gop_start + 1, gop_end):
                cur = frames[fid]
                prev = frames[fid - 1]
                # 识别更新的大块
                updated = identify_updated_big_blocks(cur, prev, diff_threshold)
                # 取出该帧的 block_types
                bt_map = None
                for fno, bt in gop_data['block_types_list']:
                    if fno == fid:
                        bt_map = bt; break
                # 累加每个纹理子块的索引使用次数
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
            
            # 检查是否有使用频次差异
            max_count = counts.max()
            min_count = counts.min()
            total_usage = counts.sum()
            
            if max_count > min_count and total_usage > 0:
                print(f"  GOP {gop_start}: 最大使用频次 {max_count}, 最小使用频次 {min_count}, 总使用次数 {total_usage}")
                
                # 根据 counts 降序排序，stable 保持相同频次项原序
                order = np.argsort(-counts, kind='stable')
                gop_data['unified_codebook'] = codebook[order]
                
                # 创建索引映射表（旧索引 -> 新索引）
                index_mapping = np.zeros(len(codebook), dtype=int)
                for new_idx, old_idx in enumerate(order):
                    index_mapping[old_idx] = new_idx
                
                # 更新 block_types 中的索引
                for fid, bt in gop_data['block_types_list']:
                    if bt is not None:
                        for (big_by, big_bx), (block_type, block_indices) in bt.items():
                            if block_type == 'detail':
                                # 更新纹理块的索引
                                new_indices = []
                                for old_idx in block_indices:
                                    if old_idx < len(index_mapping):
                                        new_indices.append(index_mapping[old_idx])
                                    else:
                                        new_indices.append(old_idx)
                                bt[(big_by, big_bx)] = (block_type, new_indices)
    
    def _encode_all_frames(self, frames, gop_codebooks, i_frame_interval, diff_threshold, 
                          force_i_threshold, enabled_segments_bitmap, enabled_medium_segments_bitmap, codebook_size):
        """编码所有帧"""
        encoded_frames = []
        frame_offsets = []
        current_offset = 0
        prev_frame = None
        
        for frame_idx, current_frame in enumerate(frames):
            frame_offsets.append(current_offset)
            
            # 找到当前GOP
            gop_start = (frame_idx // i_frame_interval) * i_frame_interval
            gop_data = gop_codebooks[gop_start]
            
            unified_codebook = gop_data['unified_codebook']
            
            # 找到当前帧的block_types，处理缺失的情况
            block_types = None
            for fid, bt in gop_data['block_types_list']:
                if fid == frame_idx:
                    block_types = bt
                    break
            
            # 如果block_types仍然为None，生成默认的block_types
            if block_types is None:
                print(f"  ⚠️ 帧 {frame_idx} 缺少block_types，使用默认分类")
                # 临时生成block_types
                _, block_types = classify_4x4_blocks_unified(current_frame, 5.0)
            
            force_i_frame = (frame_idx % i_frame_interval == 0) or frame_idx == 0
            
            if force_i_frame or prev_frame is None:
                frame_data = encode_i_frame_unified(
                    current_frame, unified_codebook, block_types
                )
                is_i_frame = True
                
                # 计算码本和索引大小
                codebook_size_bytes = codebook_size * 7  # BYTES_PER_BLOCK
                index_size = len(frame_data) - 1 - codebook_size_bytes
                
                self.encoding_stats.add_i_frame(
                    len(frame_data), 
                    is_forced=force_i_frame,
                    codebook_size=codebook_size_bytes,
                    index_size=max(0, index_size)
                )
            else:
                frame_data, is_i_frame, used_zones, color_updates, detail_updates, small_updates, medium_updates, full_updates, small_bytes, medium_bytes, full_bytes, small_segments, medium_segments, small_blocks_per_update, medium_blocks_per_update, full_blocks_per_update = encode_p_frame_unified(
                    current_frame, prev_frame,
                    unified_codebook, block_types,
                    diff_threshold, force_i_threshold, enabled_segments_bitmap,
                    enabled_medium_segments_bitmap
                )
                
                if is_i_frame:
                    codebook_size_bytes = codebook_size * 7  # BYTES_PER_BLOCK
                    index_size = len(frame_data) - 1 - codebook_size_bytes
                    
                    self.encoding_stats.add_i_frame(
                        len(frame_data), 
                        is_forced=False,
                        codebook_size=codebook_size_bytes,
                        index_size=max(0, index_size)
                    )
                else:
                    total_updates = color_updates + detail_updates
                    
                    self.encoding_stats.add_p_frame(
                        len(frame_data), total_updates, used_zones,
                        color_updates, detail_updates,
                        small_updates, medium_updates, full_updates,
                        small_bytes, medium_bytes, full_bytes,
                        small_segments, medium_segments,
                        small_blocks_per_update, medium_blocks_per_update, full_blocks_per_update
                    )
            
            encoded_frames.append(frame_data)
            current_offset += len(frame_data)
            
            prev_frame = current_frame.copy() if current_frame.size > 0 else None
            
            if frame_idx % 30 == 0 or frame_idx == len(frames) - 1:
                print(f"  已编码 {frame_idx + 1}/{len(frames)} 帧")
        
        return encoded_frames, frame_offsets 