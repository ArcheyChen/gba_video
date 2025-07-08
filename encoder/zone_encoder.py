#!/usr/bin/env python3
"""
Zone编码器 - 基于240块为一个zone的优化编码方案
"""
import numpy as np

# Zone配置 - 每240个块为一个zone  
BLOCKS_PER_ZONE = 240

# Zone数量和bitmap类型
ZONES_8x8_COUNT = 3   # (600 + 239) // 240 = 3
ZONES_8x4_COUNT = 5   # (1200 + 239) // 240 = 5
ZONES_4x4_COUNT = 10  # (2400 + 239) // 240 = 10
ZONES_4x2_COUNT = 20  # (4800 + 239) // 240 = 20

def encode_zone_bitmap_and_data(changed_blocks_dict, block_type):
    """
    基于zone的编码：bitmap + 每个zone的连续段数据
    
    Args:
        changed_blocks_dict: {block_pos: codeword_idx} 字典
        block_type: "8x8", "8x4", "4x4", "4x2"
    
    Returns:
        bytes: 编码后的数据
    """
    if block_type == "8x8":
        zones_count = ZONES_8x8_COUNT
        bitmap_bytes = 1  # u8
    elif block_type == "8x4":
        zones_count = ZONES_8x4_COUNT
        bitmap_bytes = 1  # u8
    elif block_type == "4x4":
        zones_count = ZONES_4x4_COUNT
        bitmap_bytes = 2  # u16
    elif block_type == "4x2":
        zones_count = ZONES_4x2_COUNT
        bitmap_bytes = 4  # u32
    else:
        raise ValueError(f"未知块类型: {block_type}")
    
    # 1. 按zone分组变化的块
    zones_data = {}
    for block_pos, codeword_idx in changed_blocks_dict.items():
        zone_id = block_pos // BLOCKS_PER_ZONE
        zone_offset = block_pos % BLOCKS_PER_ZONE
        if zone_id not in zones_data:
            zones_data[zone_id] = []
        zones_data[zone_id].append((zone_offset, codeword_idx))
    
    # 2. 生成bitmap
    bitmap = 0
    for zone_id in zones_data.keys():
        if zone_id < zones_count:  # 防止越界
            bitmap |= (1 << zone_id)
    
    # 3. 编码bitmap
    encoded_data = []
    if bitmap_bytes == 1:
        encoded_data.append(bitmap & 0xFF)
    elif bitmap_bytes == 2:
        encoded_data.extend([(bitmap >> 0) & 0xFF, (bitmap >> 8) & 0xFF])
    elif bitmap_bytes == 4:
        encoded_data.extend([
            (bitmap >> 0) & 0xFF, (bitmap >> 8) & 0xFF,
            (bitmap >> 16) & 0xFF, (bitmap >> 24) & 0xFF
        ])
    
    # 4. 对每个有变化的zone编码数据
    for zone_id in sorted(zones_data.keys()):
        if zone_id >= zones_count:
            continue
            
        zone_blocks = zones_data[zone_id]
        zone_blocks.sort(key=lambda x: x[0])  # 按zone内偏移排序
        
        # 生成连续段
        segments = []
        current_segment = None
        
        for zone_offset, codeword_idx in zone_blocks:
            if current_segment is None:
                current_segment = [zone_offset, zone_offset, [codeword_idx]]
            elif zone_offset == current_segment[1] + 1:
                # 连续块
                current_segment[1] = zone_offset
                current_segment[2].append(codeword_idx)
            else:
                # 非连续，开始新段
                segments.append(current_segment)
                current_segment = [zone_offset, zone_offset, [codeword_idx]]
        
        if current_segment is not None:
            segments.append(current_segment)
        
        # 编码段数量（u8）
        encoded_data.append(len(segments) & 0xFF)
        
        # 编码每个段
        for start_offset, end_offset, codewords in segments:
            run_length = end_offset - start_offset + 1
            # u8位置偏移 + u8连续数量 + 码表项序列
            encoded_data.append(start_offset & 0xFF)
            encoded_data.append(run_length & 0xFF)
            encoded_data.extend(codewords)
    
    return bytes(encoded_data)

def encode_blocks_with_zone_optimization(blocks_list):
    """
    使用zone优化的块编码（替代之前的run-length编码）
    
    Args:
        blocks_list: 四个列表 [8x8_blocks, 8x4_blocks, 4x4_blocks, 4x2_blocks]
                    每个列表包含 (block_pos, codeword_idx) 元组
    
    Returns:
        bytes: 编码后的数据
    """
    blocks_8x8, blocks_8x4, blocks_4x4, blocks_4x2 = blocks_list
    
    # 转换为字典格式
    changed_8x8 = {pos: idx for pos, idx in blocks_8x8}
    changed_8x4 = {pos: idx for pos, idx in blocks_8x4}
    changed_4x4 = {pos: idx for pos, idx in blocks_4x4}
    changed_4x2 = {pos: idx for pos, idx in blocks_4x2}
    
    # 分别编码四种块类型
    data_8x8 = encode_zone_bitmap_and_data(changed_8x8, "8x8")
    data_8x4 = encode_zone_bitmap_and_data(changed_8x4, "8x4")
    data_4x4 = encode_zone_bitmap_and_data(changed_4x4, "4x4")
    data_4x2 = encode_zone_bitmap_and_data(changed_4x2, "4x2")
    
    # 组合所有数据
    return data_8x8 + data_8x4 + data_4x4 + data_4x2
