#!/usr/bin/env python3

import statistics
from collections import defaultdict

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
        raw_size = total_frames * 240 * 160 * 2  # 假设16位像素
        compression_ratio = raw_size / total_bytes if total_bytes > 0 else 0
        print(f"\n📈 压缩效率:")
        print(f"   原始大小估算: {raw_size:,} bytes ({raw_size/1024/1024:.1f} MB)")
        print(f"   压缩比: {compression_ratio:.1f}:1")
        print(f"   压缩率: {(1-total_bytes/raw_size)*100:.1f}%") 