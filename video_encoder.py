#!/usr/bin/env python3

import argparse, cv2, numpy as np, pathlib, textwrap
import statistics
from collections import defaultdict

from core_encoder import *
from gop_processor import generate_gop_unified_codebooks
from dither_opt import apply_dither_optimized

class EncodingStats:
    """编码统计类"""
    def __init__(self):
        # 帧统计
        self.total_frames_processed = 0
        self.total_i_frames = 0
        self.forced_i_frames = 0  # 强制I帧（GOP开始）
        self.threshold_i_frames = 0  # 超阈值I帧
        self.total_p_frames = 0
        
        # 大小统计
        self.total_i_frame_bytes = 0
        self.total_p_frame_bytes = 0
        self.total_codebook_bytes = 0  # 只计算I帧中的码本数据
        self.total_index_bytes = 0     # 只计算I帧中的索引数据
        self.total_p_overhead_bytes = 0  # P帧的开销数据（bitmap等）
        
        # P帧块更新统计
        self.p_frame_updates = []  # 每个P帧的更新块数
        self.zone_usage = defaultdict(int)  # 区域使用次数
        
        # 细节统计
        self.color_block_bytes = 0
        self.detail_block_bytes = 0
        self.color_update_count = 0
        self.detail_update_count = 0
        
        # 码表使用统计
        self.small_codebook_updates = 0      # 小码表更新次数
        self.medium_codebook_updates = 0     # 中码表更新次数
        self.full_codebook_updates = 0       # 大码表更新次数
        self.small_codebook_bytes = 0        # 小码表数据大小
        self.medium_codebook_bytes = 0       # 中码表数据大小
        self.full_codebook_bytes = 0         # 大码表数据大小
        
        # 码表段使用统计
        self.small_segment_usage = defaultdict(int)  # 小码表各段使用次数
        self.medium_segment_usage = defaultdict(int) # 中码表各段使用次数
        
        # 码表效率统计
        self.small_codebook_blocks_per_update = []  # 每次小码表更新的块数
        self.medium_codebook_blocks_per_update = [] # 每次中码表更新的块数
        self.full_codebook_blocks_per_update = []   # 每次大码表更新的块数
    
    def add_i_frame(self, size_bytes, is_forced=True, codebook_size=0, index_size=0):
        self.total_frames_processed += 1
        self.total_i_frames += 1
        if is_forced:
            self.forced_i_frames += 1
        else:
            self.threshold_i_frames += 1
        
        self.total_i_frame_bytes += size_bytes
        self.total_codebook_bytes += codebook_size
        self.total_index_bytes += index_size
    
    def add_p_frame(self, size_bytes, updates_count, zone_count, 
                   color_updates=0, detail_updates=0,
                   small_updates=0, medium_updates=0, full_updates=0,
                   small_bytes=0, medium_bytes=0, full_bytes=0,
                   small_segments=None, medium_segments=None,
                   small_blocks_per_update=None, medium_blocks_per_update=None, full_blocks_per_update=None):
        self.total_frames_processed += 1
        self.total_p_frames += 1
        self.total_p_frame_bytes += size_bytes
        self.p_frame_updates.append(updates_count)
        self.zone_usage[zone_count] += 1
        
        # P帧开销：帧类型(1) + bitmap(2) + 每个区域的计数(2*zones)
        overhead = 3 + zone_count * 2
        self.total_p_overhead_bytes += overhead
        
        self.color_update_count += color_updates
        self.detail_update_count += detail_updates
        
        # 码表使用统计
        self.small_codebook_updates += small_updates
        self.medium_codebook_updates += medium_updates
        self.full_codebook_updates += full_updates
        self.small_codebook_bytes += small_bytes
        self.medium_codebook_bytes += medium_bytes
        self.full_codebook_bytes += full_bytes
        
        # 段使用统计
        if small_segments:
            for seg_idx, count in small_segments.items():
                self.small_segment_usage[seg_idx] += count
        if medium_segments:
            for seg_idx, count in medium_segments.items():
                self.medium_segment_usage[seg_idx] += count
        
        # 效率统计
        if small_blocks_per_update:
            self.small_codebook_blocks_per_update.extend(small_blocks_per_update)
        if medium_blocks_per_update:
            self.medium_codebook_blocks_per_update.extend(medium_blocks_per_update)
        if full_blocks_per_update:
            self.full_codebook_blocks_per_update.extend(full_blocks_per_update)
    
    def print_summary(self, total_frames, total_bytes):
        print(f"\n📊 编码统计报告")
        print(f"=" * 60)
        
        # 基本统计
        print(f"🎬 帧统计:")
        print(f"   视频帧数: {total_frames}")
        print(f"   I帧: {self.total_i_frames} ({self.total_i_frames/total_frames*100:.1f}%)")
        print(f"     - 强制I帧: {self.forced_i_frames}")
        print(f"     - 超阈值I帧: {self.threshold_i_frames}")
        print(f"   P帧: {self.total_p_frames} ({self.total_p_frames/total_frames*100:.1f}%)")
        
        # 大小统计
        print(f"\n💾 空间占用:")
        print(f"   总大小: {total_bytes:,} bytes ({total_bytes/1024:.1f} KB)")
        print(f"   I帧数据: {self.total_i_frame_bytes:,} bytes ({self.total_i_frame_bytes/total_bytes*100:.1f}%)")
        print(f"   P帧数据: {self.total_p_frame_bytes:,} bytes ({self.total_p_frame_bytes/total_bytes*100:.1f}%)")
        
        if self.total_i_frames > 0:
            print(f"   平均I帧大小: {self.total_i_frame_bytes/self.total_i_frames:.1f} bytes")
        if self.total_p_frames > 0:
            print(f"   平均P帧大小: {self.total_p_frame_bytes/self.total_p_frames:.1f} bytes")
        
        # 数据构成统计
        print(f"\n🎨 数据构成:")
        print(f"   码本数据: {self.total_codebook_bytes:,} bytes ({self.total_codebook_bytes/total_bytes*100:.1f}%)")
        print(f"   I帧索引: {self.total_index_bytes:,} bytes ({self.total_index_bytes/total_bytes*100:.1f}%)")
        
        # P帧数据构成
        p_frame_data_bytes = self.total_p_frame_bytes - self.total_p_overhead_bytes
        print(f"   P帧更新数据: {p_frame_data_bytes:,} bytes ({p_frame_data_bytes/total_bytes*100:.1f}%)")
        print(f"   P帧开销: {self.total_p_overhead_bytes:,} bytes ({self.total_p_overhead_bytes/total_bytes*100:.1f}%)")
        
        # 码表使用统计
        total_detail_updates = self.small_codebook_updates + self.medium_codebook_updates + self.full_codebook_updates
        if total_detail_updates > 0:
            print(f"\n📚 码表使用分析:")
            print(f"   小码表更新: {self.small_codebook_updates:,} 次 ({self.small_codebook_updates/total_detail_updates*100:.1f}%)")
            print(f"   中码表更新: {self.medium_codebook_updates:,} 次 ({self.medium_codebook_updates/total_detail_updates*100:.1f}%)")
            print(f"   大码表更新: {self.full_codebook_updates:,} 次 ({self.full_codebook_updates/total_detail_updates*100:.1f}%)")
            
            print(f"\n💾 码表数据大小:")
            total_codebook_data = self.small_codebook_bytes + self.medium_codebook_bytes + self.full_codebook_bytes
            if total_codebook_data > 0:
                print(f"   小码表数据: {self.small_codebook_bytes:,} bytes ({self.small_codebook_bytes/total_codebook_data*100:.1f}%)")
                print(f"   中码表数据: {self.medium_codebook_bytes:,} bytes ({self.medium_codebook_bytes/total_codebook_data*100:.1f}%)")
                print(f"   大码表数据: {self.full_codebook_bytes:,} bytes ({self.full_codebook_bytes/total_codebook_data*100:.1f}%)")
            
            # 码表效率统计
            if self.small_codebook_blocks_per_update:
                avg_small_blocks = statistics.mean(self.small_codebook_blocks_per_update)
                print(f"   小码表平均每次更新块数: {avg_small_blocks:.1f}")
            if self.medium_codebook_blocks_per_update:
                avg_medium_blocks = statistics.mean(self.medium_codebook_blocks_per_update)
                print(f"   中码表平均每次更新块数: {avg_medium_blocks:.1f}")
            if self.full_codebook_blocks_per_update:
                avg_full_blocks = statistics.mean(self.full_codebook_blocks_per_update)
                print(f"   大码表平均每次更新块数: {avg_full_blocks:.1f}")
        
        # 段使用统计
        if self.small_segment_usage:
            print(f"\n🔢 小码表段使用分布:")
            total_small_updates = sum(self.small_segment_usage.values())
            for seg_idx in sorted(self.small_segment_usage.keys()):
                usage_count = self.small_segment_usage[seg_idx]
                if self.small_codebook_updates > 0:
                    print(f"   段{seg_idx}: {usage_count}次 ({usage_count/self.small_codebook_updates*100:.1f}%)")
            
            # 验证排序效果：前4段应该占大部分使用
            if total_small_updates > 0:
                first_4_segments_usage = sum(self.small_segment_usage.get(i, 0) for i in range(4))
                first_4_percentage = first_4_segments_usage / total_small_updates * 100
                print(f"   前4段使用率: {first_4_percentage:.1f}% (排序效果指标)")
        
        if self.medium_segment_usage:
            print(f"\n🔢 中码表段使用分布:")
            total_medium_updates = sum(self.medium_segment_usage.values())
            for seg_idx in sorted(self.medium_segment_usage.keys()):
                usage_count = self.medium_segment_usage[seg_idx]
                if self.medium_codebook_updates > 0:
                    print(f"   段{seg_idx}: {usage_count}次 ({usage_count/self.medium_codebook_updates*100:.1f}%)")
            
            # 验证排序效果：前2段应该占大部分使用
            if total_medium_updates > 0:
                first_2_segments_usage = sum(self.medium_segment_usage.get(i, 0) for i in range(2))
                first_2_percentage = first_2_segments_usage / total_medium_updates * 100
                print(f"   前2段使用率: {first_2_percentage:.1f}% (排序效果指标)")
        
        # P帧更新统计
        if self.p_frame_updates:
            avg_updates = statistics.mean(self.p_frame_updates)
            median_updates = statistics.median(self.p_frame_updates)
            max_updates = max(self.p_frame_updates)
            min_updates = min(self.p_frame_updates)
            
            print(f"\n⚡ P帧更新分析:")
            print(f"   平均更新块数: {avg_updates:.1f}")
            print(f"   中位数更新块数: {median_updates}")
            print(f"   最大更新块数: {max_updates}")
            print(f"   最小更新块数: {min_updates}")
            print(f"   色块更新总数: {self.color_update_count:,}")
            print(f"   纹理块更新总数: {self.detail_update_count:,}")
        
        # 区域使用统计
        if self.zone_usage:
            print(f"\n🗺️  区域使用分布:")
            for zone_count in sorted(self.zone_usage.keys()):
                frames_count = self.zone_usage[zone_count]
                if self.total_p_frames > 0:
                    print(f"   {zone_count}个区域: {frames_count}次 ({frames_count/self.total_p_frames*100:.1f}%)")
        
        # 压缩效率
        raw_size = total_frames * WIDTH * HEIGHT * 2  # 假设16位像素
        compression_ratio = raw_size / total_bytes if total_bytes > 0 else 0
        print(f"\n📈 压缩效率:")
        print(f"   原始大小估算: {raw_size:,} bytes ({raw_size/1024/1024:.1f} MB)")
        print(f"   压缩比: {compression_ratio:.1f}:1")
        print(f"   压缩率: {(1-total_bytes/raw_size)*100:.1f}%")

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
    args = pa.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit("❌ 打不开输入文件")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    # 计算实际输出FPS：如果目标FPS高于源FPS，使用源FPS
    actual_output_fps = min(args.fps, src_fps)
    every = int(round(src_fps / actual_output_fps))
    
    if args.full_duration:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        grab_max = total_frames
        actual_duration = total_frames / src_fps
        print(f"编码整个视频: {total_frames} 帧，时长 {actual_duration:.2f} 秒")
    else:
        grab_max = int(args.duration * src_fps)
        print(f"编码时长: {args.duration} 秒 ({grab_max} 帧)")

    print(f"源视频FPS: {src_fps:.2f}, 目标FPS: {args.fps}, 实际输出FPS: {actual_output_fps:.2f}")
    print(f"码本配置: 统一码本{args.codebook_size}项")
    if args.dither:
        print(f"🎨 已启用抖动算法（蛇形扫描）")
    
    frames = []
    idx = 0
    print("正在提取帧...")
    
    while idx < grab_max:
        ret, frm = cap.read()
        if not ret:
            break
        if idx % every == 0:
            frm = cv2.resize(frm, (WIDTH, HEIGHT), cv2.INTER_AREA)
            frm = cv2.GaussianBlur(frm, (3, 3), 0.41)
            if args.dither:
                frm = apply_dither_optimized(frm)
            
            frame_blocks = pack_yuv420_frame(frm)
            frames.append(frame_blocks)
            
            if len(frames) % 30 == 0:
                print(f"  已提取 {len(frames)} 帧")
        idx += 1
    cap.release()

    if not frames:
        raise SystemExit("❌ 没有任何帧被采样")

    print(f"总共提取了 {len(frames)} 帧")

    # 生成统一码本（传入max_workers参数）
    gop_codebooks = generate_gop_unified_codebooks(
        frames, args.i_frame_interval, 
        args.variance_threshold, args.diff_threshold, args.codebook_size, 
        args.kmeans_max_iter, args.i_frame_weight, args.max_workers
    )

    # 基于 GOP 内 P 帧纹理块使用频次，对每个码本项降序重排
    import numpy as _np
    print("正在根据使用频次对码本进行排序...")
    
    for gop_start, gop_data in gop_codebooks.items():
        codebook = gop_data['unified_codebook']
        counts = _np.zeros(len(codebook), dtype=int)
        
        # GOP 范围：起始帧下一个到下一个 I 帧
        gop_end = min(gop_start + args.i_frame_interval, len(frames))
        
        # 统计每个码本项的使用频次
        for fid in range(gop_start + 1, gop_end):
            cur = frames[fid]
            prev = frames[fid - 1]
            # 识别更新的大块
            updated = identify_updated_big_blocks(cur, prev, args.diff_threshold)
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
            
            # 显示排序前的前10个最常用项
            top_indices_before = _np.argsort(-counts)[:10]
            # print(f"    排序前前10个最常用项: {top_indices_before.tolist()}")
            # print(f"    对应使用频次: {counts[top_indices_before].tolist()}")
            
            # 根据 counts 降序排序，stable 保持相同频次项原序
            order = _np.argsort(-counts, kind='stable')
            gop_data['unified_codebook'] = codebook[order]
            
            # 创建索引映射表（旧索引 -> 新索引）
            index_mapping = _np.zeros(len(codebook), dtype=int)
            for new_idx, old_idx in enumerate(order):
                index_mapping[old_idx] = new_idx
            
            # 显示排序后的前10个项（应该对应原来的最常用项）
            # print(f"    排序后前10个项对应原索引: {order[:10].tolist()}")
            
            # # 验证排序效果：检查排序后前几个索引的使用情况
            # print(f"    排序后前15个索引的使用频次: {counts[order[:15]].tolist()}")
            # print(f"    排序后前15个索引的段分布: {[i//15 for i in range(15)]}")
            
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
        else:
            # print(f"  GOP {gop_start}: 所有码本项使用频次相同或总使用次数为0，跳过排序")
    
    # 编码所有帧
    print("正在编码帧...")
    encoded_frames = []
    frame_offsets = []
    current_offset = 0
    prev_frame = None
    
    for frame_idx, current_frame in enumerate(frames):
        frame_offsets.append(current_offset)
        
        # 找到当前GOP
        gop_start = (frame_idx // args.i_frame_interval) * args.i_frame_interval
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
            _, block_types = classify_4x4_blocks_unified(current_frame, args.variance_threshold)
        
        force_i_frame = (frame_idx % args.i_frame_interval == 0) or frame_idx == 0
        
        if force_i_frame or prev_frame is None:
            frame_data = encode_i_frame_unified(
                current_frame, unified_codebook, block_types
            )
            is_i_frame = True
            
            # 计算码本和索引大小
            codebook_size = args.codebook_size * BYTES_PER_BLOCK
            index_size = len(frame_data) - 1 - codebook_size
            
            encoding_stats.add_i_frame(
                len(frame_data), 
                is_forced=force_i_frame,
                codebook_size=codebook_size,
                index_size=max(0, index_size)
            )
        else:
            frame_data, is_i_frame, used_zones, color_updates, detail_updates, small_updates, medium_updates, full_updates, small_bytes, medium_bytes, full_bytes, small_segments, medium_segments, small_blocks_per_update, medium_blocks_per_update, full_blocks_per_update = encode_p_frame_unified(
                current_frame, prev_frame,
                unified_codebook, block_types,
                args.diff_threshold, args.force_i_threshold, args.enabled_segments_bitmap,
                args.enabled_medium_segments_bitmap
            )
            
            if is_i_frame:
                codebook_size = args.codebook_size * BYTES_PER_BLOCK
                index_size = len(frame_data) - 1 - codebook_size
                
                encoding_stats.add_i_frame(
                    len(frame_data), 
                    is_forced=False,
                    codebook_size=codebook_size,
                    index_size=max(0, index_size)
                )
            else:
                total_updates = color_updates + detail_updates
                
                encoding_stats.add_p_frame(
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
    
    all_data = b''.join(encoded_frames)
    
    write_header(pathlib.Path(args.out).with_suffix(".h"), len(frames), len(all_data), 
                args.codebook_size, actual_output_fps)
    write_source(pathlib.Path(args.out).with_suffix(".c"), all_data, frame_offsets)
    
    # 打印详细统计
    encoding_stats.print_summary(len(frames), len(all_data))

def write_header(path_h: pathlib.Path, frame_cnt: int, total_bytes: int, codebook_size: int, output_fps: float):
    guard = "VIDEO_DATA_H"
    
    with path_h.open("w", encoding="utf-8") as f:
        f.write(textwrap.dedent(f"""\
            #ifndef {guard}
            #define {guard}

            #define VIDEO_FRAME_COUNT   {frame_cnt}
            #define VIDEO_WIDTH         {WIDTH}
            #define VIDEO_HEIGHT        {HEIGHT}
            #define VIDEO_TOTAL_BYTES   {total_bytes}
            #define VIDEO_FPS           {int(round(output_fps*10000))}
            #define UNIFIED_CODEBOOK_SIZE {codebook_size}
            #define EFFECTIVE_UNIFIED_CODEBOOK_SIZE {EFFECTIVE_UNIFIED_CODEBOOK_SIZE}
            
            // 帧类型定义
            #define FRAME_TYPE_I        0x00
            #define FRAME_TYPE_P        0x01
            
            // 特殊标记
            #define COLOR_BLOCK_MARKER  0xFF
            
            // 块参数
            #define BLOCK_WIDTH         2
            #define BLOCK_HEIGHT        2
            #define BYTES_PER_BLOCK     7
            
            extern const unsigned char video_data[VIDEO_TOTAL_BYTES];
            extern const unsigned int frame_offsets[VIDEO_FRAME_COUNT];

            #endif // {guard}
            """))

encoding_stats = EncodingStats()
def write_source(path_c: pathlib.Path, data: bytes, frame_offsets: list):
    with path_c.open("w", encoding="utf-8") as f:
        f.write('#include "video_data.h"\n\n')
        
        f.write("const unsigned int frame_offsets[] = {\n")
        for i in range(0, len(frame_offsets), 8):
            chunk = ', '.join(f"{offset}" for offset in frame_offsets[i:i+8])
            f.write("    " + chunk + ",\n")
        f.write("};\n\n")
        
        f.write("const unsigned char video_data[] = {\n")
        per_line = 16
        for i in range(0, len(data), per_line):
            chunk = ', '.join(f"0x{v:02X}" for v in data[i:i+per_line])
            f.write("    " + chunk + ",\n")
        f.write("};\n")

if __name__ == "__main__":
    main()
