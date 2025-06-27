#!/usr/bin/env python3

import statistics
from collections import defaultdict
from motion_compensation import motion_stats

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
        
        # 新增：码表块数分布统计
        self.small_blocks_distribution = {1: 0, 2: 0, 3: 0, 4: 0}  # 小码表：更新1/2/3/4块的个数
        self.medium_blocks_distribution = {1: 0, 2: 0, 3: 0, 4: 0} # 中码表：更新1/2/3/4块的个数
        self.full_blocks_distribution = {1: 0, 2: 0, 3: 0, 4: 0}   # 大码表：更新1/2/3/4块的个数
        
        # 运动补偿统计（汇总所有多进程的统计）
        self.motion_compensation_stats = None
    
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
            # 统计块数分布
            for block_count in small_blocks_per_update:
                if 1 <= block_count <= 4:
                    self.small_blocks_distribution[block_count] += 1
        if medium_blocks_per_update:
            # 统计块数分布
            for block_count in medium_blocks_per_update:
                if 1 <= block_count <= 4:
                    self.medium_blocks_distribution[block_count] += 1
        if full_blocks_per_update:
            # 统计块数分布
            for block_count in full_blocks_per_update:
                if 1 <= block_count <= 4:
                    self.full_blocks_distribution[block_count] += 1
    
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
            
            # 新增：码表块数分布统计
            print(f"\n📊 码表块数分布:")
            if self.small_codebook_updates > 0:
                print(f"   小码表块数分布:")
                for block_count in [1, 2, 3, 4]:
                    count = self.small_blocks_distribution[block_count]
                    percentage = count / self.small_codebook_updates * 100
                    print(f"     {block_count}块: {count}次 ({percentage:.1f}%)")
            
            if self.medium_codebook_updates > 0:
                print(f"   中码表块数分布:")
                for block_count in [1, 2, 3, 4]:
                    count = self.medium_blocks_distribution[block_count]
                    percentage = count / self.medium_codebook_updates * 100
                    print(f"     {block_count}块: {count}次 ({percentage:.1f}%)")
            
            if self.full_codebook_updates > 0:
                print(f"   大码表块数分布:")
                for block_count in [1, 2, 3, 4]:
                    count = self.full_blocks_distribution[block_count]
                    percentage = count / self.full_codebook_updates * 100
                    print(f"     {block_count}块: {count}次 ({percentage:.1f}%)")
        
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
            print(f"   中位数更新块数: {median_updates:.1f}")
            print(f"   最大更新块数: {max_updates}")
            print(f"   最小更新块数: {min_updates}")
            print(f"   色块更新总数: {self.color_update_count:,}")
            print(f"   纹理块更新总数: {self.detail_update_count:,}")
        
        # 区域使用统计
        if self.zone_usage:
            print(f"\n🗺️ 区域使用分布:")
            total_zones_used = sum(self.zone_usage.values())
            for zone_count in sorted(self.zone_usage.keys()):
                usage_count = self.zone_usage[zone_count]
                print(f"   {zone_count}个区域: {usage_count}次 ({usage_count/total_zones_used*100:.1f}%)")
        
        # 压缩效率
        original_size = total_frames * 240 * 160 * 3  # 假设原始BGR格式
        compression_ratio = original_size / total_bytes
        compression_rate = (1 - total_bytes / original_size) * 100
        
        print(f"\n📈 压缩效率:")
        print(f"   原始大小估算: {original_size:,} bytes ({original_size/1024/1024:.1f} MB)")
        print(f"   压缩比: {compression_ratio:.1f}:1")
        print(f"   压缩率: {compression_rate:.1f}%")
        
        # 运动补偿统计
        if self.motion_compensation_stats is not None:
            self.finalize_motion_compensation_stats()
            self._print_motion_compensation_stats(self.motion_compensation_stats)
    
    def _print_motion_compensation_stats(self, mc_stats):
        """打印运动补偿统计信息"""
        print(f"\n🎯 运动补偿统计:")
        print(f"============================================================")
        
        # 帧级统计
        frames = mc_stats['frames']
        print(f"📺 帧统计:")
        print(f"   处理帧数: {frames['total_processed']}")
        print(f"   使用运动补偿帧: {frames['with_motion_compensation']} ({frames['motion_compensation_frame_ratio']*100:.1f}%)")
        
        # 块级统计
        blocks = mc_stats['blocks']
        print(f"\n🔲 块统计:")
        print(f"   评估的8×8块: {blocks['total_8x8_evaluated']:,}")
        print(f"   运动补偿的8×8块: {blocks['total_8x8_compensated']:,} ({blocks['motion_compensation_ratio']*100:.1f}%)")
        print(f"   节省的2×2块: {blocks['total_2x2_blocks_saved']:,}")
        if blocks['total_8x8_compensated'] > 0:
            print(f"   平均每个补偿块节省: {blocks['average_blocks_saved_per_compensation']:.1f} 个2×2块")
        
        # 运动向量统计
        motion_vectors = mc_stats['motion_vectors']
        if motion_vectors['top_vectors']:
            print(f"\n🎯 运动向量分析:")
            print(f"   最常用运动向量:")
            for i, ((dx, dy), count) in enumerate(motion_vectors['top_vectors'][:5]):
                print(f"     {i+1}. ({dx:+3d}, {dy:+3d}): {count:,} 次")
        
        # 运动幅度分布
        if motion_vectors['magnitude_distribution']:
            print(f"   运动幅度分布:")
            for magnitude, count in sorted(motion_vectors['magnitude_distribution'].items())[:8]:
                print(f"     {magnitude}像素: {count:,} 次")
        
        # Zone使用统计
        zones = mc_stats['zones']
        if zones['usage_count']:
            print(f"\n🗺️  Zone使用统计:")
            for zone_idx, count in sorted(zones['usage_count'].items()):
                print(f"   Zone {zone_idx}: {count:,} 次")
        
        # 数据大小统计
        data_size = mc_stats['data_size']
        if data_size['motion_data_bytes'] > 0:
            print(f"\n💾 运动补偿数据:")
            print(f"   运动数据大小: {data_size['motion_data_bytes']:,} bytes ({data_size['motion_data_bytes']/1024:.1f} KB)")
            if self.total_p_frame_bytes > 0:
                motion_ratio = data_size['motion_data_bytes'] / self.total_p_frame_bytes * 100
                print(f"   占P帧数据比例: {motion_ratio:.1f}%")
    
    def merge_stats(self, other_stats):
        """合并另一个统计对象的数据"""
        # 帧统计
        self.total_frames_processed += other_stats.total_frames_processed
        self.total_i_frames += other_stats.total_i_frames
        self.forced_i_frames += other_stats.forced_i_frames
        self.threshold_i_frames += other_stats.threshold_i_frames
        self.total_p_frames += other_stats.total_p_frames
        
        # 大小统计
        self.total_i_frame_bytes += other_stats.total_i_frame_bytes
        self.total_p_frame_bytes += other_stats.total_p_frame_bytes
        self.total_codebook_bytes += other_stats.total_codebook_bytes
        self.total_index_bytes += other_stats.total_index_bytes
        self.total_p_overhead_bytes += other_stats.total_p_overhead_bytes
        
        # P帧块更新统计
        self.p_frame_updates.extend(other_stats.p_frame_updates)
        for zone_count, usage_count in other_stats.zone_usage.items():
            self.zone_usage[zone_count] += usage_count
        
        # 细节统计
        self.color_block_bytes += other_stats.color_block_bytes
        self.detail_block_bytes += other_stats.detail_block_bytes
        self.color_update_count += other_stats.color_update_count
        self.detail_update_count += other_stats.detail_update_count
        
        # 码表使用统计
        self.small_codebook_updates += other_stats.small_codebook_updates
        self.medium_codebook_updates += other_stats.medium_codebook_updates
        self.full_codebook_updates += other_stats.full_codebook_updates
        self.small_codebook_bytes += other_stats.small_codebook_bytes
        self.medium_codebook_bytes += other_stats.medium_codebook_bytes
        self.full_codebook_bytes += other_stats.full_codebook_bytes
        
        # 码表段使用统计
        for seg_idx, count in other_stats.small_segment_usage.items():
            self.small_segment_usage[seg_idx] += count
        for seg_idx, count in other_stats.medium_segment_usage.items():
            self.medium_segment_usage[seg_idx] += count
        
        # 码表效率统计
        self.small_codebook_blocks_per_update.extend(other_stats.small_codebook_blocks_per_update)
        self.medium_codebook_blocks_per_update.extend(other_stats.medium_codebook_blocks_per_update)
        self.full_codebook_blocks_per_update.extend(other_stats.full_codebook_blocks_per_update)
        
        # 合并块数分布统计
        for block_count in [1, 2, 3, 4]:
            self.small_blocks_distribution[block_count] += other_stats.small_blocks_distribution.get(block_count, 0)
            self.medium_blocks_distribution[block_count] += other_stats.medium_blocks_distribution.get(block_count, 0)
            self.full_blocks_distribution[block_count] += other_stats.full_blocks_distribution.get(block_count, 0)
    
    def merge_motion_compensation_stats(self, motion_stats_dict):
        """合并运动补偿统计信息"""
        if motion_stats_dict is None:
            return
            
        if self.motion_compensation_stats is None:
            # 初始化
            self.motion_compensation_stats = {
                'frames': {
                    'total_processed': 0,
                    'with_motion_compensation': 0,
                    'motion_compensation_frame_ratio': 0.0
                },
                'blocks': {
                    'total_8x8_evaluated': 0,
                    'total_8x8_compensated': 0,
                    'motion_compensation_ratio': 0.0,
                    'total_2x2_blocks_saved': 0,
                    'average_blocks_saved_per_compensation': 0.0
                },
                'motion_vectors': {
                    'top_vectors': [],
                    'magnitude_distribution': defaultdict(int)
                },
                'zones': {
                    'usage_count': defaultdict(int)
                },
                'data_size': {
                    'motion_data_bytes': 0,
                    'motion_data_ratio': 0.0
                }
            }
        
        # 合并帧统计
        frames = motion_stats_dict['frames']
        self.motion_compensation_stats['frames']['total_processed'] += frames['total_processed']
        self.motion_compensation_stats['frames']['with_motion_compensation'] += frames['with_motion_compensation']
        
        # 合并块统计
        blocks = motion_stats_dict['blocks']
        self.motion_compensation_stats['blocks']['total_8x8_evaluated'] += blocks['total_8x8_evaluated']
        self.motion_compensation_stats['blocks']['total_8x8_compensated'] += blocks['total_8x8_compensated']
        self.motion_compensation_stats['blocks']['total_2x2_blocks_saved'] += blocks['total_2x2_blocks_saved']
        
        # 合并运动向量分布
        motion_vectors = motion_stats_dict['motion_vectors']
        for (dx, dy), count in motion_vectors['top_vectors']:
            # 将运动向量添加到字典中进行合并
            found = False
            for i, ((existing_dx, existing_dy), existing_count) in enumerate(self.motion_compensation_stats['motion_vectors']['top_vectors']):
                if existing_dx == dx and existing_dy == dy:
                    self.motion_compensation_stats['motion_vectors']['top_vectors'][i] = ((dx, dy), existing_count + count)
                    found = True
                    break
            if not found:
                self.motion_compensation_stats['motion_vectors']['top_vectors'].append(((dx, dy), count))
        
        # 合并幅度分布
        for magnitude, count in motion_vectors['magnitude_distribution'].items():
            self.motion_compensation_stats['motion_vectors']['magnitude_distribution'][magnitude] += count
        
        # 合并zone统计
        zones = motion_stats_dict['zones']
        for zone_idx, count in zones['usage_count'].items():
            self.motion_compensation_stats['zones']['usage_count'][zone_idx] += count
        
        # 合并数据大小统计
        data_size = motion_stats_dict['data_size']
        self.motion_compensation_stats['data_size']['motion_data_bytes'] += data_size['motion_data_bytes']
    
    def finalize_motion_compensation_stats(self):
        """计算运动补偿统计的最终数据"""
        if self.motion_compensation_stats is None:
            return
            
        frames = self.motion_compensation_stats['frames']
        if frames['total_processed'] > 0:
            frames['motion_compensation_frame_ratio'] = frames['with_motion_compensation'] / frames['total_processed']
        
        blocks = self.motion_compensation_stats['blocks']
        if blocks['total_8x8_evaluated'] > 0:
            blocks['motion_compensation_ratio'] = blocks['total_8x8_compensated'] / blocks['total_8x8_evaluated']
        if blocks['total_8x8_compensated'] > 0:
            blocks['average_blocks_saved_per_compensation'] = blocks['total_2x2_blocks_saved'] / blocks['total_8x8_compensated']
        
        # 排序运动向量
        self.motion_compensation_stats['motion_vectors']['top_vectors'].sort(key=lambda x: x[1], reverse=True)
        self.motion_compensation_stats['motion_vectors']['top_vectors'] = self.motion_compensation_stats['motion_vectors']['top_vectors'][:5]
        
        # 数据大小比例
        if self.total_p_frame_bytes > 0:
            self.motion_compensation_stats['data_size']['motion_data_ratio'] = self.motion_compensation_stats['data_size']['motion_data_bytes'] / self.total_p_frame_bytes