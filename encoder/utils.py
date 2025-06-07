#!/usr/bin/env python3
"""
utils.py - GBA视频编码器工具函数
"""

from config import BLOCK_H


def calculate_strip_heights(height: int, strip_count: int) -> list:
    """计算每个条带的高度，确保每个条带高度都是4的倍数"""
    # 确保总高度能被4整除
    if height % 4 != 0:
        raise ValueError(f"视频高度 {height} 必须是4的倍数")
    
    # 计算每个条带的基础高度（必须是4的倍数）
    base_height = (height // strip_count // 4) * 4
    
    # 计算剩余的高度
    remaining_height = height - (base_height * strip_count)
    
    strip_heights = []
    for i in range(strip_count):
        current_height = base_height
        # 将剩余高度以4的倍数分配给前面的条带
        if remaining_height >= 4:
            current_height += 4
            remaining_height -= 4
        strip_heights.append(current_height)
    
    # 验证总高度
    if sum(strip_heights) != height:
        raise ValueError(f"条带高度分配错误: {strip_heights} 总和 {sum(strip_heights)} != {height}")
    
    # 验证每个条带高度都是4的倍数
    for i, h in enumerate(strip_heights):
        if h % 4 != 0:
            raise ValueError(f"条带 {i} 高度 {h} 不是4的倍数")
    
    return strip_heights


def calculate_block_diff(block1, block2) -> float:
    """计算两个块的差异度（使用Y通道的平均绝对差值）"""
    import numpy as np
    # 只比较Y通道（前4个字节）
    y_diff = np.abs(block1[:4].astype(np.int16) - block2[:4].astype(np.int16))
    return y_diff.mean()  # 使用平均差值，更敏感


def get_current_codebooks(frame_idx: int, gop_codebooks: dict, i_frame_interval: int) -> list:
    """获取当前帧应该使用的码表"""
    # 找到当前帧所属的GOP起始位置
    gop_start = (frame_idx // i_frame_interval) * i_frame_interval
    
    if gop_start in gop_codebooks:
        return [strip_data['codebook'] for strip_data in gop_codebooks[gop_start]]
    else:
        # 如果找不到，使用第一个GOP的码表
        first_gop = min(gop_codebooks.keys())
        return [strip_data['codebook'] for strip_data in gop_codebooks[first_gop]]