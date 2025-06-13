

from collections import defaultdict
import statistics
from const_def import *


class EncodingStats:
    """编码统计类 - 修复统计问题"""
    def __init__(self):
        # 帧统计
        self.total_frames_processed = 0
        self.total_i_frames = 0
        self.forced_i_frames = 0
        self.threshold_i_frames = 0
        self.total_p_frames = 0
        
        # 大小统计
        self.total_i_frame_bytes = 0
        self.total_p_frame_bytes = 0
        self.total_4x4_codebook_bytes = 0
        self.total_2x2_codebook_bytes = 0
        self.total_index_bytes = 0
        self.total_p_overhead_bytes = 0
        
        # 块类型统计 - 修复
        self.block_4x4_count = 0
        self.block_2x2_count = 0
        
        # P帧块更新统计 - 新增详细统计
        self.p_frame_updates = []
        self.zone_usage = defaultdict(int)
        self.detail_update_count = 0
        self.block_4x4_update_count = 0
        self.block_2x2_update_count = 0
        self.detail_update_bytes = 0  # 纹理块更新字节数
        self.block_4x4_update_bytes = 0  # 大块更新字节数
        
        # 条带统计
        self.strip_stats = defaultdict(lambda: {
            'i_frames': 0, 'p_frames': 0, 
            'i_bytes': 0, 'p_bytes': 0
        })
    
    def add_i_frame(self, strip_idx, size_bytes, is_forced=True, codebook_size=0, index_size=0):
        self.total_frames_processed += 1
        self.total_i_frames += 1
        if is_forced:
            self.forced_i_frames += 1
        else:
            self.threshold_i_frames += 1
        
        self.total_i_frame_bytes += size_bytes
        
        # 修复码本统计 - 分别计算4x4和2x2码本
        codebook_4x4_bytes = DEFAULT_4X4_CODEBOOK_SIZE * BYTES_PER_4X4_BLOCK
        codebook_2x2_bytes = EFFECTIVE_UNIFIED_CODEBOOK_SIZE * BYTES_PER_2X2_BLOCK
        self.total_4x4_codebook_bytes += codebook_4x4_bytes
        self.total_2x2_codebook_bytes += codebook_2x2_bytes
        
        # 索引大小 = 总大小 - 帧类型标记 - 两个码本大小
        actual_index_size = size_bytes - 1 - codebook_4x4_bytes - codebook_2x2_bytes
        self.total_index_bytes += max(0, actual_index_size)
        
        self.strip_stats[strip_idx]['i_frames'] += 1
        self.strip_stats[strip_idx]['i_bytes'] += size_bytes
    
    def add_p_frame(self, strip_idx, size_bytes, updates_count, zone_count, 
               updates_4x4=0, updates_2x2=0):  # 修改参数名和顺序
        self.total_frames_processed += 1
        self.total_p_frames += 1
        self.total_p_frame_bytes += size_bytes
        self.p_frame_updates.append(updates_count)
        self.zone_usage[zone_count] += 1
        
        # P帧开销：帧类型(1) + bitmap(2) + 每个区域的计数(2*zones)
        overhead = 3 + zone_count * 2  # 现在只有2种块类型
        self.total_p_overhead_bytes += overhead
        
        # 详细更新统计
        self.detail_update_count += updates_2x2
        self.block_4x4_update_count += updates_4x4  # 直接使用传入的值
        
        # 计算更新数据字节数
        detail_bytes = updates_2x2 * 17  # 1字节位置 + 16字节索引
        block_4x4_bytes = updates_4x4 * 5  # 1字节位置 + 4字节索引
        self.detail_update_bytes += detail_bytes
        self.block_4x4_update_bytes += block_4x4_bytes
        
        self.strip_stats[strip_idx]['p_frames'] += 1
        self.strip_stats[strip_idx]['p_bytes'] += size_bytes
    
    def add_block_type_stats(self, block_4x4s, block_2x2s):
        self.block_4x4_count += block_4x4s
        self.block_2x2_count += block_2x2s
    
    def print_summary(self, total_frames, total_bytes):
        print(f"\n📊 编码统计报告")
        print(f"=" * 60)
        
        # 计算条带级别的统计
        strip_count = len(self.strip_stats) if self.strip_stats else 1
        
        # 基本统计
        print(f"🎬 帧统计:")
        print(f"   视频帧数: {total_frames}")
        print(f"   条带总数: {strip_count}")
        print(f"   处理的条带帧: {self.total_frames_processed}")
        print(f"   I帧条带: {self.total_i_frames} ({self.total_i_frames/self.total_frames_processed*100:.1f}%)")
        print(f"     - 强制I帧: {self.forced_i_frames}")
        print(f"     - 超阈值I帧: {self.threshold_i_frames}")
        print(f"   P帧条带: {self.total_p_frames} ({self.total_p_frames/self.total_frames_processed*100:.1f}%)")
        
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
        print(f"   4x4块码本数据: {self.total_4x4_codebook_bytes:,} bytes ({self.total_4x4_codebook_bytes/total_bytes*100:.1f}%)")
        print(f"   2x2块码本数据: {self.total_2x2_codebook_bytes:,} bytes ({self.total_2x2_codebook_bytes/total_bytes*100:.1f}%)")
        print(f"   I帧索引: {self.total_index_bytes:,} bytes ({self.total_index_bytes/total_bytes*100:.1f}%)")
        
        # P帧数据构成
        p_frame_data_bytes = self.total_p_frame_bytes - self.total_p_overhead_bytes
        print(f"   P帧更新数据: {p_frame_data_bytes:,} bytes ({p_frame_data_bytes/total_bytes*100:.1f}%)")
        print(f"     - 2x2块更新: {self.detail_update_bytes:,} bytes ({self.detail_update_bytes/total_bytes*100:.1f}%)")
        print(f"     - 4x4块更新: {self.block_4x4_update_bytes:,} bytes ({self.block_4x4_update_bytes/total_bytes*100:.1f}%)")
        print(f"   P帧开销: {self.total_p_overhead_bytes:,} bytes ({self.total_p_overhead_bytes/total_bytes*100:.1f}%)")
        
        # 块类型统计
        print(f"\n🧩 块类型分布:")
        total_block_types = self.block_4x4_count + self.block_2x2_count
        if total_block_types > 0:
            print(f"   4x4块: {self.block_4x4_count} 个 ({self.block_4x4_count/total_block_types*100:.1f}%)")
            print(f"   2x2块: {self.block_2x2_count} 个 ({self.block_2x2_count/total_block_types*100:.1f}%)")
        
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
            print(f"   2x2块更新总数: {self.detail_update_count:,}")
            print(f"   4x4块更新总数: {self.block_4x4_update_count:,}")
        
        # 区域使用统计
        if self.zone_usage:
            print(f"\n🗺️  区域使用分布:")
            for zone_count in sorted(self.zone_usage.keys()):
                frames_count = self.zone_usage[zone_count]
                if self.total_p_frames > 0:
                    print(f"   {zone_count}个区域: {frames_count}次 ({frames_count/self.total_p_frames*100:.1f}%)")
        
        # 条带统计
        print(f"\n📏 条带统计:")
        for strip_idx in sorted(self.strip_stats.keys()):
            stats = self.strip_stats[strip_idx]
            total_strip_frames = stats['i_frames'] + stats['p_frames']
            total_strip_bytes = stats['i_bytes'] + stats['p_bytes']
            if total_strip_frames > 0:
                print(f"   条带{strip_idx}: {total_strip_frames}帧, {total_strip_bytes:,}bytes, "
                      f"平均{total_strip_bytes/total_strip_frames:.1f}bytes/帧")
        
        # 压缩效率
        raw_size = total_frames * WIDTH * HEIGHT * 2
        compression_ratio = raw_size / total_bytes if total_bytes > 0 else 0
        print(f"\n📈 压缩效率:")
        print(f"   原始大小估算: {raw_size:,} bytes ({raw_size/1024/1024:.1f} MB)")
        print(f"   压缩比: {compression_ratio:.1f}:1")
        print(f"   压缩率: {(1-total_bytes/raw_size)*100:.1f}%")