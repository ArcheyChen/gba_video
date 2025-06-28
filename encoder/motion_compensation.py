#!/usr/bin/env python3

import numpy as np
from numba import jit, njit
from typing import List, Tuple, Dict
from collections import defaultdict

# 运动补偿常量
SCREEN_WIDTH = 240
SCREEN_HEIGHT = 160
BLOCK_8X8_SIZE = 8
BLOCK_2X2_SIZE = 2

# 8×8块数量：30×20
BLOCKS_8X8_WIDTH = SCREEN_WIDTH // BLOCK_8X8_SIZE  # 30
BLOCKS_8X8_HEIGHT = SCREEN_HEIGHT // BLOCK_8X8_SIZE  # 20

# Zone配置：每个zone包含30×8个8×8块
BLOCKS_8X8_PER_ZONE_ROW = BLOCKS_8X8_WIDTH  # 30
BLOCKS_8X8_PER_ZONE_HEIGHT = 8  # 8行8×8块
BLOCKS_8X8_PER_ZONE = BLOCKS_8X8_PER_ZONE_ROW * BLOCKS_8X8_PER_ZONE_HEIGHT  # 240
TOTAL_ZONES = (BLOCKS_8X8_HEIGHT + BLOCKS_8X8_PER_ZONE_HEIGHT - 1) // BLOCKS_8X8_PER_ZONE_HEIGHT  # 3

# 运动向量范围：±7像素
MOTION_RANGE = 7

# 损失函数参数
DEFAULT_UPDATE_THRESHOLD = 12  # 8×8块内需要更新的2×2块数量阈值（总共16个2×2块的75%）

@njit
def encode_motion_vector(dx: int, dy: int) -> int:
    """
    将运动向量编码为u8
    dx, dy: ±7像素范围
    返回: 高4位=dy+7, 低4位=dx+7
    """
    return ((dy + MOTION_RANGE) << 4) | (dx + MOTION_RANGE)

@njit
def decode_motion_vector(encoded: int) -> Tuple[int, int]:
    """
    解码运动向量
    返回: (dx, dy)
    """
    dx = (encoded & 0x0F) - MOTION_RANGE
    dy = ((encoded >> 4) & 0x0F) - MOTION_RANGE
    return dx, dy

@njit
def get_8x8_block_zone_info(block_8x8_idx: int) -> Tuple[int, int]:
    """
    获取8×8块的zone信息
    返回: (zone_idx, zone_relative_idx)
    """
    block_row = block_8x8_idx // BLOCKS_8X8_WIDTH
    block_col = block_8x8_idx % BLOCKS_8X8_WIDTH
    
    zone_idx = block_row // BLOCKS_8X8_PER_ZONE_HEIGHT
    zone_relative_row = block_row % BLOCKS_8X8_PER_ZONE_HEIGHT
    zone_relative_idx = zone_relative_row * BLOCKS_8X8_PER_ZONE_ROW + block_col
    
    return zone_idx, zone_relative_idx

@njit
def calculate_sad_8x8_blocks(current_blocks: np.ndarray, prev_blocks: np.ndarray, 
                            cur_start_y: int, cur_start_x: int,
                            ref_start_y: int, ref_start_x: int) -> float:
    """
    计算两个8×8块区域的SAD（Sum of Absolute Differences）
    使用YUV444格式的Y分量进行比较
    """
    sad = 0.0
    blocks_h, blocks_w = current_blocks.shape[:2]
    
    # 8×8像素 = 4×4个2×2块
    for dy in range(4):
        for dx in range(4):
            cur_y = cur_start_y + dy
            cur_x = cur_start_x + dx
            ref_y = ref_start_y + dy
            ref_x = ref_start_x + dx
            
            # 边界检查
            if (cur_y >= blocks_h or cur_x >= blocks_w or 
                ref_y >= blocks_h or ref_x >= blocks_w or
                ref_y < 0 or ref_x < 0):
                sad += 1000.0  # 边界外的大惩罚
                continue
            
            # 计算2×2块的Y分量差异
            for i in range(4):  # Y分量的4个像素
                cur_val = float(current_blocks[cur_y, cur_x, i])
                ref_val = float(prev_blocks[ref_y, ref_x, i])
                sad += abs(cur_val - ref_val)
    
    return sad

@njit
def count_updated_2x2_blocks_after_motion(current_blocks: np.ndarray, prev_blocks: np.ndarray,
                                        cur_start_y: int, cur_start_x: int,
                                        dx: int, dy: int, diff_threshold: float) -> int:
    """
    计算运动补偿后还需要更新的2×2块数量
    """
    updated_count = 0
    blocks_h, blocks_w = current_blocks.shape[:2]
    
    # 运动向量从像素转换为2×2块坐标
    ref_start_y = cur_start_y + dy // 2
    ref_start_x = cur_start_x + dx // 2
    
    # 检查4×4个2×2块
    for dy_block in range(4):
        for dx_block in range(4):
            cur_y = cur_start_y + dy_block
            cur_x = cur_start_x + dx_block
            ref_y = ref_start_y + dy_block
            ref_x = ref_start_x + dx_block
            
            # 边界检查
            if (cur_y >= blocks_h or cur_x >= blocks_w or 
                ref_y >= blocks_h or ref_x >= blocks_w or
                ref_y < 0 or ref_x < 0):
                updated_count += 1
                continue
            
            # 计算2×2块的Y分量差异
            y_diff_sum = 0.0
            for i in range(4):  # Y分量的4个像素
                cur_val = float(current_blocks[cur_y, cur_x, i])
                ref_val = float(prev_blocks[ref_y, ref_x, i])
                y_diff_sum += abs(cur_val - ref_val)
            
            avg_diff = y_diff_sum / 4.0
            if avg_diff > diff_threshold:
                updated_count += 1
    
    return updated_count

def hierarchical_diamond_search(current_blocks: np.ndarray, prev_blocks: np.ndarray,
                               block_8x8_y: int, block_8x8_x: int, 
                               diff_threshold: float) -> Tuple[Tuple[int, int], float, int]:
    """
    分层钻石搜索算法
    返回: (最佳运动向量(dx, dy), SAD值, 更新块数量)
    """
    # 当前8×8块在2×2块坐标系中的起始位置
    cur_start_y = block_8x8_y * 4
    cur_start_x = block_8x8_x * 4
    
    best_mv = (0, 0)
    best_sad = float('inf')
    best_updates = 16  # 最坏情况：所有块都需要更新
    
    # 第一层：大钻石搜索（步长4）
    large_diamond = [(0, 4), (4, 0), (0, -4), (-4, 0), (0, 0)]
    for dx, dy in large_diamond:
        if abs(dx) <= MOTION_RANGE and abs(dy) <= MOTION_RANGE:
            ref_start_y = cur_start_y + dy // 2
            ref_start_x = cur_start_x + dx // 2
            
            sad = calculate_sad_8x8_blocks(current_blocks, prev_blocks,
                                         cur_start_y, cur_start_x,
                                         ref_start_y, ref_start_x)
            
            if sad < best_sad:
                best_sad = sad
                best_mv = (dx, dy)
    
    # 第二层：小钻石搜索（步长1）
    center_x, center_y = best_mv
    small_diamond = [
        (0, 1), (1, 0), (0, -1), (-1, 0),  # 上下左右
        (1, 1), (1, -1), (-1, 1), (-1, -1)  # 对角线
    ]
    
    for dx_offset, dy_offset in small_diamond:
        dx = center_x + dx_offset
        dy = center_y + dy_offset
        
        if abs(dx) <= MOTION_RANGE and abs(dy) <= MOTION_RANGE:
            ref_start_y = cur_start_y + dy // 2
            ref_start_x = cur_start_x + dx // 2
            
            sad = calculate_sad_8x8_blocks(current_blocks, prev_blocks,
                                         cur_start_y, cur_start_x,
                                         ref_start_y, ref_start_x)
            
            if sad < best_sad:
                best_sad = sad
                best_mv = (dx, dy)
    
    # 计算最佳运动向量下需要更新的块数
    dx, dy = best_mv
    updates_needed = count_updated_2x2_blocks_after_motion(
        current_blocks, prev_blocks, cur_start_y, cur_start_x, dx, dy, diff_threshold
    )
    
    return best_mv, best_sad, updates_needed

def detect_motion_compensation_candidates(current_blocks: np.ndarray, prev_blocks: np.ndarray,
                                        diff_threshold: float, 
                                        update_threshold: int = DEFAULT_UPDATE_THRESHOLD) -> Dict:
    """
    检测可以进行运动补偿的8×8块
    返回按zone组织的运动补偿候选
    """
    motion_candidates = {}
    total_8x8_blocks = BLOCKS_8X8_WIDTH * BLOCKS_8X8_HEIGHT
    
    # 遍历所有8×8块
    for block_8x8_idx in range(total_8x8_blocks):
        block_8x8_y = block_8x8_idx // BLOCKS_8X8_WIDTH
        block_8x8_x = block_8x8_idx % BLOCKS_8X8_WIDTH
        
        # 进行运动搜索
        best_mv, sad, updates_needed = hierarchical_diamond_search(
            current_blocks, prev_blocks, block_8x8_y, block_8x8_x, diff_threshold
        )
        
        # 如果更新块数少于阈值且有真实运动（非零向量），标记为候选
        dx, dy = best_mv
        if updates_needed <= update_threshold and (dx != 0 or dy != 0):
            zone_idx, zone_relative_idx = get_8x8_block_zone_info(block_8x8_idx)
            
            if zone_idx not in motion_candidates:
                motion_candidates[zone_idx] = []
            
            motion_candidates[zone_idx].append({
                'zone_relative_idx': zone_relative_idx,
                'motion_vector': best_mv,
                'updates_needed': updates_needed,
                'sad': sad,
                'block_8x8_pos': (block_8x8_y, block_8x8_x)
            })
    
    # 更新统计信息
    motion_stats.update_frame_stats(motion_candidates, total_8x8_blocks)
        # 调试信息
    # total_compensated_blocks = sum(len(candidates) for candidates in motion_candidates.values())
    # if total_compensated_blocks > 0:
    #     print(f"  运动补偿: {total_compensated_blocks}/{total_8x8_blocks} 个8x8块")
    

    return motion_candidates

def apply_motion_compensation_to_blocks(current_blocks: np.ndarray, prev_blocks: np.ndarray,
                                      motion_candidates: Dict) -> np.ndarray:
    """
    将运动补偿应用到块数据，生成新的参考帧
    从prev_blocks开始，应用运动补偿生成A'帧
    返回: 应用运动补偿后的块数据
    """
    # 复制前一帧作为基础（A帧 -> A'帧）
    compensated_blocks = prev_blocks.copy()
    
    for zone_idx, candidates in motion_candidates.items():
        for candidate in candidates:
            block_8x8_y, block_8x8_x = candidate['block_8x8_pos']
            dx, dy = candidate['motion_vector']
            
            # 当前8×8块在2×2块坐标系中的起始位置
            cur_start_y = block_8x8_y * 4
            cur_start_x = block_8x8_x * 4
            
            # 运动补偿的源位置（在prev_blocks中）
            ref_start_y = cur_start_y + dy // 2
            ref_start_x = cur_start_x + dx // 2
            
            # 复制4×4个2×2块（从prev_blocks的源位置复制到compensated_blocks的目标位置）
            blocks_h, blocks_w = prev_blocks.shape[:2]
            for dy_block in range(4):
                for dx_block in range(4):
                    cur_y = cur_start_y + dy_block
                    cur_x = cur_start_x + dx_block
                    ref_y = ref_start_y + dy_block
                    ref_x = ref_start_x + dx_block
                    
                    # 边界检查
                    if (cur_y < blocks_h and cur_x < blocks_w and 
                        ref_y >= 0 and ref_y < blocks_h and ref_x >= 0 and ref_x < blocks_w):
                        # 从prev_blocks的源位置复制到compensated_blocks的目标位置
                        compensated_blocks[cur_y, cur_x] = prev_blocks[ref_y, ref_x]
    
    return compensated_blocks

def merge_consecutive_motion_blocks(candidates: List[Dict]) -> List[Dict]:
    """
    合并同一zone内运动向量相同且连续的8x8块
    约束：连续块只能在同一行内，不能跨行
    返回：合并后的运动补偿条带列表，格式为{'zone_relative_idx': int, 'motion_vector': tuple, 'count': int}
    """
    if not candidates:
        return []
    
    # 按zone内相对索引排序
    sorted_candidates = sorted(candidates, key=lambda x: x['zone_relative_idx'])
    
    merged_strips = []
    current_strip = None
    
    for candidate in sorted_candidates:
        zone_relative_idx = candidate['zone_relative_idx']
        motion_vector = candidate['motion_vector']
        
        # 将zone内相对索引转换为8x8块的全局坐标
        block_8x8_y, block_8x8_x = candidate['block_8x8_pos']
        
        if current_strip is None:
            # 开始新的条带
            current_strip = {
                'zone_relative_idx': zone_relative_idx,
                'motion_vector': motion_vector,
                'count': 1,
                'start_x': block_8x8_x,
                'start_y': block_8x8_y
            }
        else:
            # 检查是否可以合并到当前条带  
            can_merge = (
                motion_vector == current_strip['motion_vector'] and  # 运动向量相同
                block_8x8_y == current_strip['start_y'] and  # 在同一行
                block_8x8_x == current_strip['start_x'] + current_strip['count'] and  # 连续
                block_8x8_x < BLOCKS_8X8_WIDTH  # 当前块不超出行边界
            )
            
            if can_merge:
                # 合并到当前条带
                current_strip['count'] += 1
            else:
                # 结束当前条带，开始新条带
                # 移除临时字段
                strip_data = {
                    'zone_relative_idx': current_strip['zone_relative_idx'],
                    'motion_vector': current_strip['motion_vector'],
                    'count': current_strip['count']
                }
                merged_strips.append(strip_data)
                
                current_strip = {
                    'zone_relative_idx': zone_relative_idx,
                    'motion_vector': motion_vector,
                    'count': 1,
                    'start_x': block_8x8_x,
                    'start_y': block_8x8_y
                }
    
    # 添加最后一个条带
    if current_strip is not None:
        strip_data = {
            'zone_relative_idx': current_strip['zone_relative_idx'],
            'motion_vector': current_strip['motion_vector'],
            'count': current_strip['count']
        }
        merged_strips.append(strip_data)
    
    return merged_strips

def encode_motion_compensation_data(motion_candidates: Dict) -> bytes:
    """
    编码运动补偿数据
    格式：zone_bitmap + 各zone的数据
    新格式支持连续块合并：u8 offset + u8 motion_vector + u8 count
    """
    data = bytearray()
    
    # Zone bitmap：哪些zone有运动补偿数据（用足够的字节表示）
    zone_bitmap = 0
    for zone_idx in motion_candidates.keys():
        zone_bitmap |= (1 << zone_idx)
    
    # 写入zone bitmap（用1个字节就足够，最多3个zone）
    data.append(zone_bitmap)
    
    # 写入各zone的数据
    total_strips = 0
    total_blocks_before_merge = 0
    total_blocks_after_merge = 0
    
    for zone_idx in range(TOTAL_ZONES):
        if zone_idx in motion_candidates:
            candidates = motion_candidates[zone_idx]
            total_blocks_before_merge += len(candidates)
            
            # 合并连续的运动补偿块
            merged_strips = merge_consecutive_motion_blocks(candidates)
            total_strips += len(merged_strips)
            
            # 计算合并后的总块数（用于验证）
            for strip in merged_strips:
                total_blocks_after_merge += strip['count']
            
            # 写入该zone的条带数量
            data.append(len(merged_strips))
            
            # 写入每个条带的数据
            for strip in merged_strips:
                # zone内相对索引（u8）
                data.append(strip['zone_relative_idx'])
                
                # 运动向量（u8）
                dx, dy = strip['motion_vector']
                encoded_mv = encode_motion_vector(dx, dy)
                data.append(encoded_mv)
                
                # 连续块数量（u8）
                data.append(strip['count'])
    
    motion_data = bytes(data)
    
    # 更新统计信息
    motion_stats.motion_data_bytes += len(motion_data)
    motion_stats.total_strips += total_strips
    motion_stats.blocks_before_merge += total_blocks_before_merge
    motion_stats.blocks_after_merge += total_blocks_after_merge
    
    return motion_data

class MotionCompensationStats:
    """运动补偿统计信息"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置统计信息"""
        # 基本统计
        self.total_frames_processed = 0
        self.frames_with_motion_compensation = 0
        
        # 8x8块统计
        self.total_8x8_blocks_evaluated = 0
        self.total_8x8_blocks_compensated = 0
        self.motion_compensation_ratio = 0.0
        
        # 运动向量统计
        self.motion_vector_distribution = defaultdict(int)  # 按(dx, dy)统计
        self.motion_magnitude_histogram = defaultdict(int)  # 按幅度统计
        
        # 效果统计
        self.total_2x2_blocks_saved = 0  # 通过运动补偿节省的2x2块数
        self.average_blocks_saved_per_compensation = 0.0
        
        # 合并统计
        self.total_strips = 0  # 合并后的条带总数
        self.blocks_before_merge = 0  # 合并前的块数
        self.blocks_after_merge = 0  # 合并后的块数（用于验证）
        self.merge_efficiency = 0.0  # 合并效率：(blocks_before - strips) / blocks_before
        
        # Zone统计
        self.zone_usage_count = defaultdict(int)  # 每个zone被使用的次数
        
        # 数据大小统计
        self.motion_data_bytes = 0
        self.motion_data_ratio = 0.0  # 占总数据的比例
    
    def update_frame_stats(self, motion_candidates: Dict, total_8x8_blocks: int):
        """更新单帧统计"""
        self.total_frames_processed += 1
        self.total_8x8_blocks_evaluated += total_8x8_blocks
        
        if motion_candidates:
            self.frames_with_motion_compensation += 1
            
            # 统计运动补偿块数
            compensated_blocks = 0
            blocks_saved = 0
            
            for zone_idx, candidates in motion_candidates.items():
                self.zone_usage_count[zone_idx] += 1
                compensated_blocks += len(candidates)
                
                for candidate in candidates:
                    # 统计运动向量分布
                    dx, dy = candidate['motion_vector']
                    self.motion_vector_distribution[(dx, dy)] += 1
                    
                    # 统计运动幅度
                    magnitude = int(np.sqrt(dx*dx + dy*dy))
                    self.motion_magnitude_histogram[magnitude] += 1
                    
                    # 统计节省的块数（16个2x2块减去需要更新的块数）
                    blocks_saved += (16 - candidate['updates_needed'])
            
            self.total_8x8_blocks_compensated += compensated_blocks
            self.total_2x2_blocks_saved += blocks_saved
    
    def update_data_size(self, motion_data_size: int, total_frame_size: int):
        """更新数据大小统计"""
        self.motion_data_bytes += motion_data_size
        if total_frame_size > 0:
            self.motion_data_ratio = motion_data_size / total_frame_size
    
    def finalize_stats(self):
        """计算最终统计数据"""
        if self.total_8x8_blocks_evaluated > 0:
            self.motion_compensation_ratio = self.total_8x8_blocks_compensated / self.total_8x8_blocks_evaluated
        
        if self.total_8x8_blocks_compensated > 0:
            self.average_blocks_saved_per_compensation = self.total_2x2_blocks_saved / self.total_8x8_blocks_compensated
        
        # 计算合并效率
        if self.blocks_before_merge > 0:
            self.merge_efficiency = (self.blocks_before_merge - self.total_strips) / self.blocks_before_merge
    
    def get_stats_dict(self) -> Dict:
        """获取统计信息字典"""
        self.finalize_stats()
        
        # 计算最常用的运动向量
        top_motion_vectors = sorted(self.motion_vector_distribution.items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
        
        # 计算运动幅度分布
        magnitude_stats = dict(sorted(self.motion_magnitude_histogram.items()))
        
        return {
            'frames': {
                'total_processed': self.total_frames_processed,
                'with_motion_compensation': self.frames_with_motion_compensation,
                'motion_compensation_frame_ratio': self.frames_with_motion_compensation / max(1, self.total_frames_processed)
            },
            'blocks': {
                'total_8x8_evaluated': self.total_8x8_blocks_evaluated,
                'total_8x8_compensated': self.total_8x8_blocks_compensated,
                'motion_compensation_ratio': self.motion_compensation_ratio,
                'total_2x2_blocks_saved': self.total_2x2_blocks_saved,
                'average_blocks_saved_per_compensation': self.average_blocks_saved_per_compensation
            },
            'motion_vectors': {
                'top_vectors': top_motion_vectors,
                'magnitude_distribution': magnitude_stats
            },
            'zones': {
                'usage_count': dict(self.zone_usage_count)
            },
            'merging': {
                'total_strips': self.total_strips,
                'blocks_before_merge': self.blocks_before_merge,
                'blocks_after_merge': self.blocks_after_merge,
                'merge_efficiency': self.merge_efficiency
            },
            'data_size': {
                'motion_data_bytes': self.motion_data_bytes,
                'motion_data_ratio': self.motion_data_ratio
            }
        }

# 全局统计实例
motion_stats = MotionCompensationStats()
