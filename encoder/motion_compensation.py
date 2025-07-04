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
DEFAULT_UPDATE_THRESHOLD = 6  
DEFAULT_MIN_IMPROVEMENT_THRESHOLD = 2  # 运动补偿相比不运动至少要减少的块数

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
def calculate_mse_8x8_blocks(current_blocks: np.ndarray, prev_blocks: np.ndarray, 
                            cur_start_y: int, cur_start_x: int,
                            ref_start_y: int, ref_start_x: int) -> float:
    """
    计算两个8×8块区域的MSE（Mean Squared Error）
    使用YUV444格式的Y分量进行比较
    """
    mse = 0.0
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
                mse += 10000.0  # 边界外的大惩罚（相当于100的平方差）
                continue
            
            # 计算2×2块的Y分量差异
            for i in range(4):  # Y分量的4个像素
                cur_val = float(current_blocks[cur_y, cur_x, i])
                ref_val = float(prev_blocks[ref_y, ref_x, i])
                diff = cur_val - ref_val
                mse += diff * diff  # 使用平方差
    
    return mse / 16.0  # 除以总像素数得到平均值

@njit
def count_updated_2x2_blocks_after_motion(current_blocks: np.ndarray, prev_blocks: np.ndarray,
                                        cur_start_y: int, cur_start_x: int,
                                        dx: int, dy: int, diff_threshold: float) -> int:
    """
    计算运动补偿后还需要更新的2×2块数量
    使用统一的块差异计算方法
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
            
            # 使用统一的块差异计算
            diff = calculate_2x2_block_difference_unified(
                current_blocks[cur_y, cur_x], prev_blocks[ref_y, ref_x]
            )
            
            if diff > diff_threshold:
                updated_count += 1
    
    return updated_count

@njit
def count_updated_2x2_blocks_no_motion(current_blocks: np.ndarray, prev_blocks: np.ndarray,
                                      cur_start_y: int, cur_start_x: int,
                                      diff_threshold: float) -> int:
    """
    计算不进行运动补偿时需要更新的2×2块数量
    直接比较当前帧和前一帧对应位置的8×8块
    使用统一的块差异计算方法
    """
    updated_count = 0
    blocks_h, blocks_w = current_blocks.shape[:2]
    
    # 检查4×4个2×2块
    for dy_block in range(4):
        for dx_block in range(4):
            cur_y = cur_start_y + dy_block
            cur_x = cur_start_x + dx_block
            
            # 边界检查
            if cur_y >= blocks_h or cur_x >= blocks_w:
                updated_count += 1
                continue
            
            # 使用统一的块差异计算
            diff = calculate_2x2_block_difference_unified(
                current_blocks[cur_y, cur_x], prev_blocks[cur_y, cur_x]
            )
            
            if diff > diff_threshold:
                updated_count += 1
    
    return updated_count

@njit
def diamond_search_iteration(current_blocks: np.ndarray, prev_blocks: np.ndarray,
                           cur_start_y: int, cur_start_x: int,
                           best_mv: Tuple[int, int], diamond_pattern: List[Tuple[int, int]],
                           best_mse: float) -> Tuple[Tuple[int, int], float, bool]:
    """
    执行一次钻石搜索迭代
    返回: (新的最佳运动向量, 新的最佳MSE, 是否有更新)
    """
    current_best_mv = (0, 0)
    current_best_mse = best_mse
    have_update = False
    
    for dx, dy in diamond_pattern:
        new_dx = best_mv[0] + dx
        new_dy = best_mv[1] + dy
        
        if abs(new_dx) <= MOTION_RANGE and abs(new_dy) <= MOTION_RANGE:
            ref_start_y = cur_start_y + new_dy
            ref_start_x = cur_start_x + new_dx
            
            mse = calculate_mse_8x8_blocks(current_blocks, prev_blocks,
                                        cur_start_y, cur_start_x,
                                        ref_start_y, ref_start_x)
            
            if mse < current_best_mse:
                current_best_mse = mse
                current_best_mv = (new_dx, new_dy)
                have_update = True
    
    return current_best_mv, current_best_mse, have_update

def hierarchical_diamond_search(current_blocks: np.ndarray, prev_blocks: np.ndarray,
                               block_8x8_y: int, block_8x8_x: int, 
                               diff_threshold: float) -> Tuple[Tuple[int, int], float, int, int]:
    """
    分层钻石搜索算法
    返回: (最佳运动向量(dx, dy), MSE值, 运动后更新块数量, 不运动时更新块数量)
    """
    # 当前8×8块在2×2块坐标系中的起始位置
    cur_start_y = block_8x8_y * 4
    cur_start_x = block_8x8_x * 4
    
    # 首先计算不运动时需要更新的块数（基准）
    no_motion_updates = count_updated_2x2_blocks_no_motion(
        current_blocks, prev_blocks, cur_start_y, cur_start_x, diff_threshold
    )
    
    best_mv = (0, 0)
    best_mse = float('inf')
    
    # 第一层：大钻石搜索（步长4）
    large_diamond = [(0,0), (0, 4), (4, 0), (0, -4), (-4, 0), (2, 2), (2, -2), (-2, 2), (-2, -2)]
    
    # 初始搜索
    best_mv, best_mse, _ = diamond_search_iteration(current_blocks, prev_blocks,
                                                   cur_start_y, cur_start_x,
                                                   best_mv, large_diamond, best_mse)
    
    # 继续大钻石搜索直到没有改进
    while True:
        new_mv, new_mse, have_update = diamond_search_iteration(current_blocks, prev_blocks,
                                                               cur_start_y, cur_start_x,
                                                               best_mv, large_diamond, best_mse)
        if not have_update:
            break
        best_mv, best_mse = new_mv, new_mse

    # 第二层：中钻石搜索（步长2）
    small_diamond = [(0, 2), (2, 0), (0, -2), (-2, 0),  # 上下左右
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]  # 对角线
    
    # 继续小钻石搜索直到没有改进
    while True:
        new_mv, new_mse, have_update = diamond_search_iteration(current_blocks, prev_blocks,
                                                               cur_start_y, cur_start_x,
                                                               best_mv, small_diamond, best_mse)
        if not have_update:
            break
        best_mv, best_mse = new_mv, new_mse
    
    # 第三层：小钻石搜索（步长1）
    small_diamond = [(0, 1), (1, 0), (0, -1), (-1, 0),  # 上下左右
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]  # 对角线
    
    # 继续小钻石搜索直到没有改进
    while True:
        new_mv, new_mse, have_update = diamond_search_iteration(current_blocks, prev_blocks,
                                                               cur_start_y, cur_start_x,
                                                               best_mv, small_diamond, best_mse)
        if not have_update:
            break
        best_mv, best_mse = new_mv, new_mse
    
    # 计算最佳运动向量下需要更新的块数
    dx, dy = best_mv
    updates_needed = count_updated_2x2_blocks_after_motion(
        current_blocks, prev_blocks, cur_start_y, cur_start_x, dx, dy, diff_threshold
    )
    
    return best_mv, best_mse, updates_needed, no_motion_updates

def detect_motion_compensation_candidates(current_blocks: np.ndarray, prev_blocks: np.ndarray,
                                        diff_threshold: float, 
                                        update_threshold: int = DEFAULT_UPDATE_THRESHOLD,
                                        min_improvement_threshold: int = DEFAULT_MIN_IMPROVEMENT_THRESHOLD) -> Dict:
    """
    检测可以进行运动补偿的8×8块
    增加了运动补偿效果判断：只有当运动补偿显著减少更新块数时才采用
    增加了早期退出优化：如果块本身就很好则跳过运动搜索
    
    参数:
    - min_improvement_threshold: 运动补偿相比不运动至少要减少的块数（默认2个）
    
    返回按zone组织的运动补偿候选
    """
    motion_candidates = {}
    total_8x8_blocks = BLOCKS_8X8_WIDTH * BLOCKS_8X8_HEIGHT
    
    # 统计信息
    motion_rejected_count = 0  # 因效果不明显被拒绝的运动补偿数量
    early_skip_count = 0  # 因块本身就很好而跳过搜索的数量
    
    # 遍历所有8×8块
    for block_8x8_idx in range(total_8x8_blocks):
        block_8x8_y = block_8x8_idx // BLOCKS_8X8_WIDTH
        block_8x8_x = block_8x8_idx % BLOCKS_8X8_WIDTH
        
        # 首先计算不运动时需要更新的块数
        cur_start_y = block_8x8_y * 4
        cur_start_x = block_8x8_x * 4
        no_motion_updates = count_updated_2x2_blocks_no_motion(
            current_blocks, prev_blocks, cur_start_y, cur_start_x, diff_threshold
        )
        
        # 早期退出优化：如果块本身就很好（需要更新的块数很少），跳过运动搜索
        if no_motion_updates < min_improvement_threshold:
            early_skip_count += 1
            continue
        
        # 进行运动搜索
        best_mv, mse, updates_needed, _ = hierarchical_diamond_search(
            current_blocks, prev_blocks, block_8x8_y, block_8x8_x, diff_threshold
        )
        
        # 计算运动补偿的改进效果
        improvement = no_motion_updates - updates_needed
        
        # 判断是否采用运动补偿的条件：
        # 1. 更新块数少于阈值
        # 2. 有真实运动（非零向量）
        # 3. 运动补偿相比不运动有明显改进
        dx, dy = best_mv
        is_real_motion = (dx != 0 or dy != 0)
        has_significant_improvement = improvement >= min_improvement_threshold
        
        if (updates_needed <= update_threshold and 
            is_real_motion and 
            has_significant_improvement):
            
            zone_idx, zone_relative_idx = get_8x8_block_zone_info(block_8x8_idx)
            
            if zone_idx not in motion_candidates:
                motion_candidates[zone_idx] = []
            
            motion_candidates[zone_idx].append({
                'zone_relative_idx': zone_relative_idx,
                'motion_vector': best_mv,
                'updates_needed': updates_needed,
                'no_motion_updates': no_motion_updates,
                'improvement': improvement,
                'mse': mse,
                'block_8x8_pos': (block_8x8_y, block_8x8_x)
            })
        elif is_real_motion and not has_significant_improvement:
            # 统计因效果不明显被拒绝的运动补偿
            motion_rejected_count += 1
    
    # 更新统计信息
    motion_stats.update_frame_stats(motion_candidates, total_8x8_blocks)
    motion_stats.motion_rejected_count += motion_rejected_count
    motion_stats.early_skip_count += early_skip_count
    
    # 调试信息（注释掉，改为在最后汇总时输出）
    # total_compensated_blocks = sum(len(candidates) for candidates in motion_candidates.values())
    # if total_compensated_blocks > 0 or motion_rejected_count > 0:
    #     print(f"  运动补偿: {total_compensated_blocks}/{total_8x8_blocks} 个8x8块采用, {motion_rejected_count} 个因效果不明显被拒绝")

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
        
        # 运动补偿效果统计
        self.motion_rejected_count = 0  # 因效果不明显被拒绝的运动补偿数量
        self.motion_rejection_ratio = 0.0  # 拒绝率
        self.early_skip_count = 0  # 因块本身就很好而跳过搜索的数量
        self.early_skip_ratio = 0.0  # 早期跳过率
        
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
            self.early_skip_ratio = self.early_skip_count / self.total_8x8_blocks_evaluated
        
        if self.total_8x8_blocks_compensated > 0:
            self.average_blocks_saved_per_compensation = self.total_2x2_blocks_saved / self.total_8x8_blocks_compensated
        
        # 计算运动补偿拒绝率
        total_motion_attempts = self.total_8x8_blocks_compensated + self.motion_rejected_count
        if total_motion_attempts > 0:
            self.motion_rejection_ratio = self.motion_rejected_count / total_motion_attempts
        
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
                'motion_rejected_count': self.motion_rejected_count,
                'motion_rejection_ratio': self.motion_rejection_ratio,
                'early_skip_count': self.early_skip_count,
                'early_skip_ratio': self.early_skip_ratio,
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

@njit
def calculate_2x2_block_difference_unified(current_block: np.ndarray, prev_block: np.ndarray) -> float:
    """
    统一的2×2块差异计算函数，与core_encoder保持一致
    计算Y分量的平均平方差（MSE）
    """
    y_diff_sum = 0.0
    for i in range(4):  # Y分量的4个像素
        current_val = float(current_block[i])
        prev_val = float(prev_block[i])
        diff = current_val - prev_val
        y_diff_sum += diff * diff  # 使用平方差（MSE）
    return y_diff_sum / 4.0
