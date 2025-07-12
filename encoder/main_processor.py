#!/usr/bin/env python3

import argparse
import cv2
import numpy as np
import pathlib
import time
import multiprocessing as mp

# 导入分离的模块
from encoding_algorithms import *
from output_writer import *

# 新增：导入分离的工具模块
from block_utils import *
from codebook import *

WIDTH, HEIGHT = 240, 160

# 多级码表配置（默认值，将被命令行参数覆盖）
# 所有码表大小限制为255，使用U8索引，0xFF保留作为分裂标志
DEFAULT_CODEBOOK_SIZE_8x8 = 255     # 8x8块码表大小
DEFAULT_CODEBOOK_SIZE_8x4 = 255     # 8x4块码表大小
DEFAULT_CODEBOOK_SIZE_4x4 = 255     # 4x4块码表大小
DEFAULT_CODEBOOK_SIZE_4x2 = 255     # 4x2块码表大小（最大255，0xFF留作分裂标志）
DEFAULT_COVERAGE_RADIUS_8x8 = 70.0  # 8x8块覆盖半径
DEFAULT_COVERAGE_RADIUS_8x4 = 65.0 # 8x4块覆盖半径
DEFAULT_COVERAGE_RADIUS_4x4 = 60.0  # 4x4块覆盖半径

# 块尺寸定义
BLOCK_8x8_W, BLOCK_8x8_H = 8, 8   # 8x8块
BLOCK_8x4_W, BLOCK_8x4_H = 8, 4   # 8x4块
BLOCK_4x4_W, BLOCK_4x4_H = 4, 4   # 4x4块
BLOCK_4x2_W, BLOCK_4x2_H = 4, 2   # 4x2块

PIXELS_PER_8x8_BLOCK = BLOCK_8x8_W * BLOCK_8x8_H  # 64
PIXELS_PER_8x4_BLOCK = BLOCK_8x4_W * BLOCK_8x4_H  # 32
PIXELS_PER_4x4_BLOCK = BLOCK_4x4_W * BLOCK_4x4_H  # 16
PIXELS_PER_4x2_BLOCK = BLOCK_4x2_W * BLOCK_4x2_H  # 8

# 8x8块数量（用于I帧主编码）
BLOCKS_8x8_PER_FRAME = (WIDTH // BLOCK_8x8_W) * (HEIGHT // BLOCK_8x8_H)  # 30 * 20 = 600
# 8x4块数量（用于分裂编码）
BLOCKS_8x4_PER_FRAME = (WIDTH // BLOCK_8x4_W) * (HEIGHT // BLOCK_8x4_H)  # 30 * 40 = 1200
# 4x4块数量（用于分裂编码）
BLOCKS_4x4_PER_FRAME = (WIDTH // BLOCK_4x4_W) * (HEIGHT // BLOCK_4x4_H)  # 60 * 40 = 2400
# 4x2块数量（用于细分编码）
BLOCKS_4x2_PER_FRAME = (WIDTH // BLOCK_4x2_W) * (HEIGHT // BLOCK_4x2_H)  # 60 * 80 = 4800

# Zone配置 - 每240个块为一个zone
BLOCKS_PER_ZONE = 240

# Zone数量和bitmap类型
ZONES_8x8_COUNT = (BLOCKS_8x8_PER_FRAME + BLOCKS_PER_ZONE - 1) // BLOCKS_PER_ZONE  # 3个zone
ZONES_8x4_COUNT = (BLOCKS_8x4_PER_FRAME + BLOCKS_PER_ZONE - 1) // BLOCKS_PER_ZONE  # 5个zone  
ZONES_4x4_COUNT = (BLOCKS_4x4_PER_FRAME + BLOCKS_PER_ZONE - 1) // BLOCKS_PER_ZONE  # 10个zone
ZONES_4x2_COUNT = (BLOCKS_4x2_PER_FRAME + BLOCKS_PER_ZONE - 1) // BLOCKS_PER_ZONE  # 20个zone

def main():
    pa = argparse.ArgumentParser(description="Encode to GBA IP-Frame with Multi-Level Codebook")
    pa.add_argument("input")
    pa.add_argument("--duration", type=float, default=5.0)
    pa.add_argument("--fps",      type=int,   default=30)
    pa.add_argument("--gop-size", type=int,   default=120, help="GOP大小")
    pa.add_argument("--i-weight", type=int,   default=3, help="I帧权重")
    pa.add_argument("--diff-threshold", type=float, default=100, help="P帧块差异阈值")
    pa.add_argument("--codebook-8x8", type=int, default=DEFAULT_CODEBOOK_SIZE_8x8, help="8x8码表大小")
    pa.add_argument("--codebook-8x4", type=int, default=DEFAULT_CODEBOOK_SIZE_8x4, help="8x4码表大小")
    pa.add_argument("--codebook-4x4", type=int, default=DEFAULT_CODEBOOK_SIZE_4x4, help="4x4码表大小")
    pa.add_argument("--codebook-4x2", type=int, default=DEFAULT_CODEBOOK_SIZE_4x2, help="4x2码表大小")
    pa.add_argument("--coverage-radius-8x8", type=float, default=DEFAULT_COVERAGE_RADIUS_8x8, help="8x8块覆盖半径")
    pa.add_argument("--coverage-radius-8x4", type=float, default=DEFAULT_COVERAGE_RADIUS_8x4, help="8x4块覆盖半径")
    pa.add_argument("--coverage-radius-4x4", type=float, default=DEFAULT_COVERAGE_RADIUS_4x4, help="4x4块覆盖半径")
    pa.add_argument("--out", default="video_data")
    args = pa.parse_args()

    # 使用局部变量而不是修改全局变量
    gop_size = args.gop_size
    i_frame_weight = args.i_weight
    diff_threshold = args.diff_threshold
    codebook_size_8x8 = args.codebook_8x8
    codebook_size_8x4 = args.codebook_8x4
    codebook_size_4x4 = args.codebook_4x4
    codebook_size_4x2 = args.codebook_4x2
    coverage_radius_8x8 = args.coverage_radius_8x8
    coverage_radius_8x4 = args.coverage_radius_8x4
    coverage_radius_4x4 = args.coverage_radius_4x4

    # 验证码表大小不超过255（预留0xFF作为分裂标志）
    if codebook_size_8x8 > 255:
        raise SystemExit(f"❌ 8x8码表大小 ({codebook_size_8x8}) 不能超过255")
    if codebook_size_8x4 > 255:
        raise SystemExit(f"❌ 8x4码表大小 ({codebook_size_8x4}) 不能超过255")
    if codebook_size_4x4 > 255:
        raise SystemExit(f"❌ 4x4码表大小 ({codebook_size_4x4}) 不能超过255")
    if codebook_size_4x2 > 255:  # 4x2特别限制为255，因为需要保留0xFF
        raise SystemExit(f"❌ 4x2码表大小 ({codebook_size_4x2}) 不能超过255（0xFF保留作为分裂标志）")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit("❌ 打不开输入文件")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    every = int(round(src_fps / args.fps))
    grab_max = int(args.duration * src_fps)

    # 读取所有帧
    print("读取视频帧...")
    all_frame_blocks = []
    idx = 0
    while idx < grab_max:
        ret, frm = cap.read()
        if not ret:
            break
        if idx % every == 0:
            frm = cv2.resize(frm, (WIDTH, HEIGHT), cv2.INTER_AREA)
            # frm = cv2.blur(frm, (2, 2))  
            # 提取8x8块用于主要编码
            blocks_8x8 = extract_yuv444_blocks_8x8(frm)
            all_frame_blocks.append(blocks_8x8)
        idx += 1
    cap.release()

    if not all_frame_blocks:
        raise SystemExit("❌ 没有任何帧被采样")

    total_frames = len(all_frame_blocks)
    gop_count = (total_frames + gop_size - 1) // gop_size
    print(f"总帧数: {total_frames}, GOP数量: {gop_count}, GOP大小: {gop_size}")

    # 为每个GOP生成编码数据（多进程并行）
    gop_codebooks = []
    encoded_frames = []
    frame_offsets = [0]  # 第一帧从0开始
    frame_types = []
    current_offset = 0
    
    # 统计信息
    total_stats = {
        'blocks_8x8_used': 0,
        'blocks_8x4_used': 0,
        'blocks_4x4_used': 0,
        'blocks_4x2_used': 0,
        'i_frame_stats': {
            'blocks_8x8_used': 0, 'blocks_8x4_used': 0, 'blocks_4x4_used': 0, 'blocks_4x2_used': 0,
            'frame_count': 0, 'total_blocks_encoded': 0,
            'avg_blocks_per_frame': 0.0, 'codec_efficiency': 0.0,
            'total_frame_size': 0, 'avg_frame_size_kb': 0.0,
            'total_codebook_size': 0, 'avg_codebook_size_kb': 0.0
        },
        'p_frame_stats': {
            'blocks_8x8_used': 0, 'blocks_8x4_used': 0, 'blocks_4x4_used': 0, 'blocks_4x2_used': 0,
            'frame_count': 0, 'total_changed_blocks': 0, 'total_segments': 0,
            'avg_changed_blocks_per_frame': 0.0, 'avg_segments_per_frame': 0.0,
            'segment_efficiency': 0.0, 'total_frame_size': 0, 'avg_frame_size_kb': 0.0,
            'block_type_sizes': {
                '8x8': 0, '8x4': 0, '4x4': 0, '4x2': 0, 'headers': 0
            },
            'zone_stats': {
                '8x8': {'zones_used': 0, 'segments_total': 0, 'blocks_total': 0, 'avg_segments_per_zone': 0.0},
                '8x4': {'zones_used': 0, 'segments_total': 0, 'blocks_total': 0, 'avg_segments_per_zone': 0.0},
                '4x4': {'zones_used': 0, 'segments_total': 0, 'blocks_total': 0, 'avg_segments_per_zone': 0.0},
                '4x2': {'zones_used': 0, 'segments_total': 0, 'blocks_total': 0, 'avg_segments_per_zone': 0.0}
            }
        },
        'frame_detailed_stats': []  # 存储每一帧的详细统计
    }

    # 准备多进程参数
    gop_args = []
    for gop_idx in range(gop_count):
        # 确定当前GOP的帧范围
        start_frame = gop_idx * gop_size
        end_frame = min((gop_idx + 1) * gop_size, total_frames)
        gop_frames = all_frame_blocks[start_frame:end_frame]
        
        # 准备参数元组
        args_tuple = (
            gop_idx, gop_frames, gop_size, i_frame_weight, diff_threshold,
            codebook_size_8x8, codebook_size_8x4, codebook_size_4x4, codebook_size_4x2,
            coverage_radius_8x8, coverage_radius_8x4, coverage_radius_4x4
        )
        gop_args.append(args_tuple)
    
    # 使用多进程并行处理GOP
    print(f"\n🚀 启动多进程处理 {gop_count} 个GOP...")
    cpu_count = mp.cpu_count()
    process_count = min(cpu_count, gop_count)  # 不要启动比GOP数量更多的进程
    print(f"CPU核心数: {cpu_count}, 使用进程数: {process_count}")
    
    start_time = time.time()
    
    with mp.Pool(processes=process_count) as pool:
        # 提交所有GOP处理任务
        results = pool.map(process_single_gop, gop_args)
    
    parallel_time = time.time() - start_time
    print(f"✅ 多进程处理完成！总耗时: {parallel_time:.1f}s")
    
    # 按GOP索引排序结果（确保顺序正确）
    results.sort(key=lambda x: x[0])
    
    # 重新组装结果
    print("\n📦 重新组装编码结果...")
    gop_codebooks = [None] * gop_count
    
    for gop_idx, codebooks, gop_encoded_frames, gop_frame_stats_list in results:
        # 存储码表
        gop_codebooks[gop_idx] = codebooks
        
        # 处理每一帧的编码数据
        for frame_idx, ((frame_data, frame_type), frame_stats) in enumerate(zip(gop_encoded_frames, gop_frame_stats_list)):
            global_frame_idx = gop_idx * gop_size + frame_idx
            
            encoded_frames.append(frame_data)
            frame_types.append(frame_type)
            current_offset += len(frame_data)
            frame_offsets.append(current_offset)
            
            # 更新统计
            total_stats['blocks_8x8_used'] += frame_stats['blocks_8x8_used']
            total_stats['blocks_8x4_used'] += frame_stats['blocks_8x4_used']
            total_stats['blocks_4x4_used'] += frame_stats['blocks_4x4_used']
            total_stats['blocks_4x2_used'] += frame_stats['blocks_4x2_used']
            
            if frame_type == 0:  # I帧
                total_stats['i_frame_stats']['frame_count'] += 1
                total_stats['i_frame_stats']['blocks_8x8_used'] += frame_stats['blocks_8x8_used']
                total_stats['i_frame_stats']['blocks_8x4_used'] += frame_stats['blocks_8x4_used']
                total_stats['i_frame_stats']['blocks_4x4_used'] += frame_stats['blocks_4x4_used']
                total_stats['i_frame_stats']['blocks_4x2_used'] += frame_stats['blocks_4x2_used']
                
                total_blocks = (frame_stats['blocks_8x8_used'] + frame_stats['blocks_8x4_used'] + 
                               frame_stats['blocks_4x4_used'] + frame_stats['blocks_4x2_used'])
                total_stats['i_frame_stats']['total_blocks_encoded'] += total_blocks
                
                # I帧大小统计（帧数据大小，不包括码表）
                total_stats['i_frame_stats']['total_frame_size'] += len(frame_data)
                
                # I帧对应的码表大小（每个I帧分摊GOP码表大小）
                # 每个GOP有一套码表，每个I帧分摊相同的码表大小
                codebook_size_bytes = (
                    codebook_size_8x8 * 64 * 2 +   # 8x8码表：每个码字64个uint16
                    codebook_size_8x4 * 32 * 2 +   # 8x4码表：每个码字32个uint16  
                    codebook_size_4x4 * 16 * 2 +   # 4x4码表：每个码字16个uint16
                    codebook_size_4x2 * 8 * 2      # 4x2码表：每个码字8个uint16
                )
                total_stats['i_frame_stats']['total_codebook_size'] += codebook_size_bytes
                
                print(f"  I帧 {global_frame_idx}: {BLOCKS_8x8_PER_FRAME} 个8x8块 (8x8码表: {frame_stats['blocks_8x8_used']}, 8x4码表: {frame_stats['blocks_8x4_used']}, 4x4码表: {frame_stats['blocks_4x4_used']}, 4x2码表: {frame_stats['blocks_4x2_used']}) - 大小: {len(frame_data)} 字节")
            else:  # P帧
                total_stats['p_frame_stats']['frame_count'] += 1
                total_stats['p_frame_stats']['blocks_8x8_used'] += frame_stats['blocks_8x8_used']
                total_stats['p_frame_stats']['blocks_8x4_used'] += frame_stats['blocks_8x4_used']
                total_stats['p_frame_stats']['blocks_4x4_used'] += frame_stats['blocks_4x4_used']
                total_stats['p_frame_stats']['blocks_4x2_used'] += frame_stats['blocks_4x2_used']
                
                # P帧特有统计
                if 'total_changed_blocks' in frame_stats:
                    total_stats['p_frame_stats']['total_changed_blocks'] += frame_stats['total_changed_blocks']
                if 'total_segments' in frame_stats:
                    total_stats['p_frame_stats']['total_segments'] += frame_stats['total_segments']
                if 'zone_stats' in frame_stats:
                    for block_type, zone_data in frame_stats['zone_stats'].items():
                        if 'zones_used' in zone_data:
                            total_stats['p_frame_stats']['zone_stats'][block_type]['zones_used'] += zone_data['zones_used']
                        if 'segments_total' in zone_data:
                            total_stats['p_frame_stats']['zone_stats'][block_type]['segments_total'] += zone_data['segments_total']
                        if 'blocks_total' in zone_data:
                            total_stats['p_frame_stats']['zone_stats'][block_type]['blocks_total'] += zone_data['blocks_total']
                
                # P帧大小统计
                total_stats['p_frame_stats']['total_frame_size'] += len(frame_data)
                
                # P帧各级码块大小统计（简化估算）
                if 'block_type_sizes' in frame_stats:
                    for block_type, size in frame_stats['block_type_sizes'].items():
                        total_stats['p_frame_stats']['block_type_sizes'][block_type] += size
                else:
                    # 如果没有详细的大小统计，使用简化估算
                    # 每个8x8块索引 + bitmap + overhead ≈ 1字节
                    # 每个8x4块索引 + bitmap + overhead ≈ 1字节  
                    # 每个4x4块索引 + bitmap + overhead ≈ 1字节
                    # 每个4x2块索引 + bitmap + overhead ≈ 1字节
                    estimated_8x8_size = frame_stats['blocks_8x8_used'] * 1
                    estimated_8x4_size = frame_stats['blocks_8x4_used'] * 1
                    estimated_4x4_size = frame_stats['blocks_4x4_used'] * 1  
                    estimated_4x2_size = frame_stats['blocks_4x2_used'] * 1
                    estimated_header_size = len(frame_data) - (estimated_8x8_size + estimated_8x4_size + estimated_4x4_size + estimated_4x2_size)
                    
                    total_stats['p_frame_stats']['block_type_sizes']['8x8'] += estimated_8x8_size
                    total_stats['p_frame_stats']['block_type_sizes']['8x4'] += estimated_8x4_size
                    total_stats['p_frame_stats']['block_type_sizes']['4x4'] += estimated_4x4_size
                    total_stats['p_frame_stats']['block_type_sizes']['4x2'] += estimated_4x2_size
                    total_stats['p_frame_stats']['block_type_sizes']['headers'] += max(0, estimated_header_size)
                
                changed_blocks = (frame_stats['blocks_8x8_used'] + frame_stats['blocks_8x4_used'] + 
                                 frame_stats['blocks_4x4_used'] + frame_stats['blocks_4x2_used'])
                print(f"  P帧 {global_frame_idx}: {changed_blocks} 个变化块 (8x8码表: {frame_stats['blocks_8x8_used']}, 8x4码表: {frame_stats['blocks_8x4_used']}, 4x4码表: {frame_stats['blocks_4x4_used']}, 4x2码表: {frame_stats['blocks_4x2_used']}) - 大小: {len(frame_data)} 字节")
            
            # 保存每帧详细统计
            frame_detail = {
                'frame_idx': global_frame_idx,
                'frame_type': 'I' if frame_type == 0 else 'P',
                'gop_idx': gop_idx,
                'frame_size': len(frame_data),
                'blocks_8x8_used': frame_stats['blocks_8x8_used'],
                'blocks_8x4_used': frame_stats['blocks_8x4_used'],
                'blocks_4x4_used': frame_stats['blocks_4x4_used'],
                'blocks_4x2_used': frame_stats['blocks_4x2_used']
            }
            if frame_type == 1 and 'zone_stats' in frame_stats:
                frame_detail['zone_stats'] = frame_stats['zone_stats']
            
            total_stats['frame_detailed_stats'].append(frame_detail)

    # 移除最后一个多余的偏移
    frame_offsets = frame_offsets[:-1]
    
    # 计算平均统计
    if total_stats['i_frame_stats']['frame_count'] > 0:
        total_stats['i_frame_stats']['avg_blocks_per_frame'] = (
            total_stats['i_frame_stats']['total_blocks_encoded'] / total_stats['i_frame_stats']['frame_count']
        )
        total_stats['i_frame_stats']['codec_efficiency'] = (
            total_stats['i_frame_stats']['blocks_8x8_used'] / total_stats['i_frame_stats']['total_blocks_encoded'] * 100
        )
        total_stats['i_frame_stats']['avg_frame_size_kb'] = (
            total_stats['i_frame_stats']['total_frame_size'] / total_stats['i_frame_stats']['frame_count'] / 1024
        )
        total_stats['i_frame_stats']['avg_codebook_size_kb'] = (
            total_stats['i_frame_stats']['total_codebook_size'] / total_stats['i_frame_stats']['frame_count'] / 1024
        )
    
    if total_stats['p_frame_stats']['frame_count'] > 0:
        total_stats['p_frame_stats']['avg_changed_blocks_per_frame'] = (
            total_stats['p_frame_stats']['total_changed_blocks'] / total_stats['p_frame_stats']['frame_count']
        )
        total_stats['p_frame_stats']['avg_segments_per_frame'] = (
            total_stats['p_frame_stats']['total_segments'] / total_stats['p_frame_stats']['frame_count']
        )
        total_stats['p_frame_stats']['avg_frame_size_kb'] = (
            total_stats['p_frame_stats']['total_frame_size'] / total_stats['p_frame_stats']['frame_count'] / 1024
        )
        if total_stats['p_frame_stats']['total_segments'] > 0:
            total_stats['p_frame_stats']['segment_efficiency'] = (
                total_stats['p_frame_stats']['total_changed_blocks'] / total_stats['p_frame_stats']['total_segments']
            )
    
    # 计算zone平均统计
    for block_type in ['8x8', '8x4', '4x4', '4x2']:
        zone_data = total_stats['p_frame_stats']['zone_stats'][block_type]
        if zone_data['zones_used'] > 0:
            zone_data['avg_segments_per_zone'] = zone_data['segments_total'] / zone_data['zones_used']
        if zone_data['segments_total'] > 0:
            zone_data['avg_blocks_per_segment'] = zone_data['blocks_total'] / zone_data['segments_total']

    # 写入文件
    write_header(pathlib.Path(args.out).with_suffix(".h"), total_frames, gop_count, gop_size, codebook_size_8x8, codebook_size_8x4, codebook_size_4x4, codebook_size_4x2)
    write_source(pathlib.Path(args.out).with_suffix(".c"), gop_codebooks, encoded_frames, frame_offsets, frame_types, codebook_size_8x8, codebook_size_8x4, codebook_size_4x4, codebook_size_4x2)

    print(f"\n📊 帧平均大小统计 (以KB计):")
    print("="*40)
    
    # I帧平均大小统计
    if total_stats['i_frame_stats']['frame_count'] > 0:
        i_frame_avg_kb = total_stats['i_frame_stats']['avg_frame_size_kb']
        i_codebook_avg_kb = total_stats['i_frame_stats']['avg_codebook_size_kb']
        i_total_avg_kb = i_frame_avg_kb + i_codebook_avg_kb
        
        print(f"🎯 I帧平均大小:")
        print(f"  - 帧内数据: {i_frame_avg_kb:.2f} KB/帧")
        print(f"  - 码表数据: {i_codebook_avg_kb:.2f} KB/帧")
        print(f"  - 总大小: {i_total_avg_kb:.2f} KB/帧")
        print(f"  - 总帧数: {total_stats['i_frame_stats']['frame_count']} 帧")
        print(f"  - 总帧内数据: {total_stats['i_frame_stats']['total_frame_size']/1024:.1f} KB")
        print(f"  - 总码表数据: {total_stats['i_frame_stats']['total_codebook_size']/1024:.1f} KB")
        print(f"  - 总I帧数据: {(total_stats['i_frame_stats']['total_frame_size'] + total_stats['i_frame_stats']['total_codebook_size'])/1024:.1f} KB")
    else:
        print("🎯 I帧平均大小: 无I帧数据")
    
    # P帧平均大小统计
    if total_stats['p_frame_stats']['frame_count'] > 0:
        p_frame_avg_kb = total_stats['p_frame_stats']['avg_frame_size_kb']
        
        print(f"\n🚀 P帧平均大小:")
        print(f"  - 帧数据: {p_frame_avg_kb:.2f} KB/帧")
        print(f"  - 总帧数: {total_stats['p_frame_stats']['frame_count']} 帧")
        print(f"  - 总帧数据: {total_stats['p_frame_stats']['total_frame_size']/1024:.1f} KB")
        
        # P帧各级码块平均大小
        print(f"  - 各级码块平均大小:")
        total_block_size = sum(total_stats['p_frame_stats']['block_type_sizes'].values())
        for block_type, size in total_stats['p_frame_stats']['block_type_sizes'].items():
            avg_size_per_frame = size / total_stats['p_frame_stats']['frame_count'] / 1024
            percentage = size / total_block_size * 100 if total_block_size > 0 else 0
            print(f"    - {block_type}: {avg_size_per_frame:.3f} KB/帧 ({percentage:.1f}%)")
    else:
        print("🚀 P帧平均大小: 无P帧数据")
    
    # I帧与P帧大小对比
    if total_stats['i_frame_stats']['frame_count'] > 0 and total_stats['p_frame_stats']['frame_count'] > 0:
        i_total_avg_kb = total_stats['i_frame_stats']['avg_frame_size_kb'] + total_stats['i_frame_stats']['avg_codebook_size_kb']
        p_avg_kb = total_stats['p_frame_stats']['avg_frame_size_kb']
        size_ratio = i_total_avg_kb / p_avg_kb if p_avg_kb > 0 else 0
        
        print(f"\n📈 I帧与P帧大小对比:")
        print(f"  - I帧平均大小: {i_total_avg_kb:.2f} KB/帧 (含码表)")
        print(f"  - P帧平均大小: {p_avg_kb:.2f} KB/帧")
        print(f"  - 大小比例: {size_ratio:.1f}:1 (I帧:P帧)")
        print(f"  - P帧压缩效率: {100/size_ratio:.1f}% (相对I帧)")
    
    print("="*40)
    
    # 继续原有的详细统计信息
    print("\n" + "="*60)
    print("📊 详细编码统计信息")
    print("="*60)
    
    # 基本信息
    total_data_size = sum(len(frame_data) for frame_data in encoded_frames)
    i_frame_count = sum(1 for ft in frame_types if ft == 0)
    p_frame_count = sum(1 for ft in frame_types if ft == 1)
    
    print(f"总帧数: {total_frames}")
    print(f"  - I帧: {i_frame_count} 帧")
    print(f"  - P帧: {p_frame_count} 帧")
    print(f"GOP数量: {gop_count}, GOP大小: {gop_size}")
    print(f"块尺寸: 8x8({BLOCKS_8x8_PER_FRAME}), 8x4({BLOCKS_8x4_PER_FRAME}), 4x4({BLOCKS_4x4_PER_FRAME}), 4x2({BLOCKS_4x2_PER_FRAME})")
    print(f"码表大小: 8x8({codebook_size_8x8}), 8x4({codebook_size_8x4}), 4x4({codebook_size_4x4}), 4x2({codebook_size_4x2})")
    print(f"覆盖半径: 8x8({coverage_radius_8x8}), 8x4({coverage_radius_8x4}), 4x4({coverage_radius_4x4})")
    
    # 码表使用统计
    print(f"\n📋 码表使用统计:")
    print(f"总计:")
    print(f"  - 8x8码表使用: {total_stats['blocks_8x8_used']:,} 个8x8块")
    print(f"  - 8x4码表使用: {total_stats['blocks_8x4_used']:,} 个8x4块")
    print(f"  - 4x4码表使用: {total_stats['blocks_4x4_used']:,} 个4x4块")
    print(f"  - 4x2码表使用: {total_stats['blocks_4x2_used']:,} 个4x2块")
    
    total_8x8_blocks = i_frame_count * BLOCKS_8x8_PER_FRAME  # I帧中所有8x8块都需要编码
    total_possible_8x4_blocks = total_8x8_blocks * 2  # 每个8x8块最多拆分为2个8x4块
    total_possible_4x4_blocks = total_possible_8x4_blocks * 2  # 每个8x4块最多拆分为2个4x4块
    total_possible_4x2_blocks = total_possible_4x4_blocks * 2  # 每个4x4块最多拆分为2个4x2块
    
    print(f"\n🎯 I帧详细统计 ({total_stats['i_frame_stats']['frame_count']} 帧):")
    print(f"  - 8x8码表使用: {total_stats['i_frame_stats']['blocks_8x8_used']:,} 个8x8块")
    print(f"  - 8x4码表使用: {total_stats['i_frame_stats']['blocks_8x4_used']:,} 个8x4块")
    print(f"  - 4x4码表使用: {total_stats['i_frame_stats']['blocks_4x4_used']:,} 个4x4块")
    print(f"  - 4x2码表使用: {total_stats['i_frame_stats']['blocks_4x2_used']:,} 个4x2块")
    print(f"  - 总编码块数: {total_stats['i_frame_stats']['total_blocks_encoded']:,}")
    print(f"  - 平均每帧编码块数: {total_stats['i_frame_stats']['avg_blocks_per_frame']:.1f}")
    print(f"  - 8x8码表效率: {total_stats['i_frame_stats']['codec_efficiency']:.1f}% (越高越好)")
    print(f"  📊 I帧大小统计:")
    print(f"    - 总帧内数据大小: {total_stats['i_frame_stats']['total_frame_size']:,} 字节 ({total_stats['i_frame_stats']['total_frame_size']/1024:.1f} KB)")
    print(f"    - 平均帧内数据大小: {total_stats['i_frame_stats']['avg_frame_size_kb']:.2f} KB/帧")
    print(f"    - 总码表数据大小: {total_stats['i_frame_stats']['total_codebook_size']:,} 字节 ({total_stats['i_frame_stats']['total_codebook_size']/1024:.1f} KB)")
    print(f"    - 平均码表数据大小: {total_stats['i_frame_stats']['avg_codebook_size_kb']:.2f} KB/帧")
    print(f"    - 平均I帧总大小: {total_stats['i_frame_stats']['avg_frame_size_kb'] + total_stats['i_frame_stats']['avg_codebook_size_kb']:.2f} KB/帧 (帧内+码表)")
    if total_8x8_blocks > 0:
        i_8x8_ratio = total_stats['i_frame_stats']['blocks_8x8_used'] / total_8x8_blocks * 100
        print(f"  - I帧中8x8码表覆盖率: {i_8x8_ratio:.1f}%")
    
    print(f"\n🚀 P帧详细统计 ({total_stats['p_frame_stats']['frame_count']} 帧):")
    print(f"  - 8x8码表使用: {total_stats['p_frame_stats']['blocks_8x8_used']:,} 个8x8块")
    print(f"  - 8x4码表使用: {total_stats['p_frame_stats']['blocks_8x4_used']:,} 个8x4块")
    print(f"  - 4x4码表使用: {total_stats['p_frame_stats']['blocks_4x4_used']:,} 个4x4块")
    print(f"  - 4x2码表使用: {total_stats['p_frame_stats']['blocks_4x2_used']:,} 个4x2块")
    print(f"  - 总变化块数: {total_stats['p_frame_stats']['total_changed_blocks']:,}")
    print(f"  - 总编码段数: {total_stats['p_frame_stats']['total_segments']:,}")
    print(f"  - 平均每帧变化块数: {total_stats['p_frame_stats']['avg_changed_blocks_per_frame']:.1f}")
    print(f"  - 平均每帧编码段数: {total_stats['p_frame_stats']['avg_segments_per_frame']:.1f}")
    print(f"  - 段编码效率: {total_stats['p_frame_stats']['segment_efficiency']:.1f} 块/段 (越高越好)")
    print(f"  📊 P帧大小统计:")
    print(f"    - 总帧大小: {total_stats['p_frame_stats']['total_frame_size']:,} 字节 ({total_stats['p_frame_stats']['total_frame_size']/1024:.1f} KB)")
    print(f"    - 平均帧大小: {total_stats['p_frame_stats']['avg_frame_size_kb']:.2f} KB/帧")
    print(f"    - 各级码块空间占用:")
    total_block_size = sum(total_stats['p_frame_stats']['block_type_sizes'].values())
    if total_block_size > 0:
        for block_type, size in total_stats['p_frame_stats']['block_type_sizes'].items():
            percentage = size / total_block_size * 100 if total_block_size > 0 else 0
            avg_size_per_frame = size / total_stats['p_frame_stats']['frame_count'] / 1024 if total_stats['p_frame_stats']['frame_count'] > 0 else 0
            print(f"      - {block_type}码表: {size:,} 字节 ({percentage:.1f}%, {avg_size_per_frame:.3f} KB/帧)")
    
    # P帧与I帧大小对比
    if total_stats['i_frame_stats']['frame_count'] > 0 and total_stats['p_frame_stats']['frame_count'] > 0:
        i_total_avg = total_stats['i_frame_stats']['avg_frame_size_kb'] + total_stats['i_frame_stats']['avg_codebook_size_kb']
        p_avg = total_stats['p_frame_stats']['avg_frame_size_kb']
        compression_ratio = i_total_avg / p_avg if p_avg > 0 else 0
        print(f"    - P帧相对I帧压缩比: {compression_ratio:.1f}:1 (I帧: {i_total_avg:.2f} KB, P帧: {p_avg:.2f} KB)")
    
    # P帧变化率分析
    if p_frame_count > 0:
        total_possible_p_blocks = p_frame_count * BLOCKS_8x8_PER_FRAME
        p_change_rate = total_stats['p_frame_stats']['total_changed_blocks'] / total_possible_p_blocks * 100
        print(f"  - P帧总体变化率: {p_change_rate:.1f}% ({total_stats['p_frame_stats']['total_changed_blocks']:,}/{total_possible_p_blocks:,})")
    
    print(f"\n🗺️ Zone编码效率统计:")
    for block_type in ['8x8', '8x4', '4x4', '4x2']:
        zone_data = total_stats['p_frame_stats']['zone_stats'][block_type]
        if zone_data['blocks_total'] > 0:
            print(f"  {block_type}码表:")
            print(f"    - 使用的zone数: {zone_data['zones_used']}")
            print(f"    - 编码段总数: {zone_data['segments_total']}")
            print(f"    - 变化块总数: {zone_data['blocks_total']}")
            print(f"    - 平均每zone段数: {zone_data['avg_segments_per_zone']:.1f}")
            print(f"    - 平均每段块数: {zone_data['avg_blocks_per_segment']:.1f}")
            
            # 计算zone利用率
            if block_type == "8x8":
                max_zones = ZONES_8x8_COUNT
            elif block_type == "8x4":
                max_zones = ZONES_8x4_COUNT
            elif block_type == "4x4":
                max_zones = ZONES_4x4_COUNT
            elif block_type == "4x2":
                max_zones = ZONES_4x2_COUNT
            
            zone_usage_rate = zone_data['zones_used'] / (max_zones * total_stats['p_frame_stats']['frame_count']) * 100
            print(f"    - Zone利用率: {zone_usage_rate:.1f}% ({zone_data['zones_used']}/{max_zones * total_stats['p_frame_stats']['frame_count']})")
    
    print(f"\n📊 码表利用率分析:")
    print("I帧码表利用率 (相对于码表大小):")
    if total_stats['i_frame_stats']['frame_count'] > 0:
        i_8x8_per_frame = total_stats['i_frame_stats']['blocks_8x8_used'] / total_stats['i_frame_stats']['frame_count']
        i_8x4_per_frame = total_stats['i_frame_stats']['blocks_8x4_used'] / total_stats['i_frame_stats']['frame_count']
        i_4x4_per_frame = total_stats['i_frame_stats']['blocks_4x4_used'] / total_stats['i_frame_stats']['frame_count']
        i_4x2_per_frame = total_stats['i_frame_stats']['blocks_4x2_used'] / total_stats['i_frame_stats']['frame_count']
        
        print(f"  - 8x8码表: {i_8x8_per_frame:.1f} 个/帧 (码表大小: {codebook_size_8x8}, 利用率: {i_8x8_per_frame/codebook_size_8x8*100:.1f}%)")
        print(f"  - 8x4码表: {i_8x4_per_frame:.1f} 个/帧 (码表大小: {codebook_size_8x4}, 利用率: {i_8x4_per_frame/codebook_size_8x4*100:.1f}%)")
        print(f"  - 4x4码表: {i_4x4_per_frame:.1f} 个/帧 (码表大小: {codebook_size_4x4}, 利用率: {i_4x4_per_frame/codebook_size_4x4*100:.1f}%)")
        print(f"  - 4x2码表: {i_4x2_per_frame:.1f} 个/帧 (码表大小: {codebook_size_4x2}, 利用率: {i_4x2_per_frame/codebook_size_4x2*100:.1f}%)")
    
    print("P帧码表利用率 (相对于码表大小):")
    if total_stats['p_frame_stats']['frame_count'] > 0:
        p_8x8_per_frame = total_stats['p_frame_stats']['blocks_8x8_used'] / total_stats['p_frame_stats']['frame_count']
        p_8x4_per_frame = total_stats['p_frame_stats']['blocks_8x4_used'] / total_stats['p_frame_stats']['frame_count']
        p_4x4_per_frame = total_stats['p_frame_stats']['blocks_4x4_used'] / total_stats['p_frame_stats']['frame_count']
        p_4x2_per_frame = total_stats['p_frame_stats']['blocks_4x2_used'] / total_stats['p_frame_stats']['frame_count']
        
        print(f"  - 8x8码表: {p_8x8_per_frame:.1f} 个/帧 (码表大小: {codebook_size_8x8}, 利用率: {p_8x8_per_frame/codebook_size_8x8*100:.1f}%)")
        print(f"  - 8x4码表: {p_8x4_per_frame:.1f} 个/帧 (码表大小: {codebook_size_8x4}, 利用率: {p_8x4_per_frame/codebook_size_8x4*100:.1f}%)")
        print(f"  - 4x4码表: {p_4x4_per_frame:.1f} 个/帧 (码表大小: {codebook_size_4x4}, 利用率: {p_4x4_per_frame/codebook_size_4x4*100:.1f}%)")
        print(f"  - 4x2码表: {p_4x2_per_frame:.1f} 个/帧 (码表大小: {codebook_size_4x2}, 利用率: {p_4x2_per_frame/codebook_size_4x2*100:.1f}%)")
    
    # 计算各部分大小
    # 1. 码表大小
    codebook_8x8_size_bytes = gop_count * codebook_size_8x8 * 64 * 2  # 每个8x8码字64个uint16
    codebook_8x4_size_bytes = gop_count * codebook_size_8x4 * 32 * 2  # 每个8x4码字32个uint16
    codebook_4x4_size_bytes = gop_count * codebook_size_4x4 * 16 * 2  # 每个4x4码字16个uint16
    codebook_4x2_size_bytes = gop_count * codebook_size_4x2 * 8 * 2   # 每个4x2码字8个uint16
    codebook_size_bytes = codebook_8x8_size_bytes + codebook_8x4_size_bytes + codebook_4x4_size_bytes + codebook_4x2_size_bytes
    
    # 2. 帧数据大小
    frame_data_size_bytes = total_data_size * 2  # 每个u16是2字节
    
    # 3. 偏移表大小
    offsets_size_bytes = len(frame_offsets) * 4  # 每个u32是4字节
    
    # 4. 帧类型表大小
    frame_types_size_bytes = len(frame_types) * 1  # 每个u8是1字节
    
    # 5. I帧和P帧数据分析
    i_frame_data_size = 0
    p_frame_data_size = 0
    
    for i, (frame_data, frame_type) in enumerate(zip(encoded_frames, frame_types)):
        if frame_type == 0:  # I帧
            i_frame_data_size += len(frame_data)
        else:  # P帧
            p_frame_data_size += len(frame_data)
    
    i_frame_data_bytes = i_frame_data_size * 2
    p_frame_data_bytes = p_frame_data_size * 2
    
    # 总文件大小
    total_file_size = codebook_size_bytes + frame_data_size_bytes + offsets_size_bytes + frame_types_size_bytes
    
    print("\n💾 内存使用分析:")
    print(f"8x8码表数据: {codebook_8x8_size_bytes:,} 字节 ({codebook_8x8_size_bytes/1024:.1f} KB)")
    print(f"8x4码表数据: {codebook_8x4_size_bytes:,} 字节 ({codebook_8x4_size_bytes/1024:.1f} KB)")
    print(f"4x4码表数据: {codebook_4x4_size_bytes:,} 字节 ({codebook_4x4_size_bytes/1024:.1f} KB)")
    print(f"4x2码表数据: {codebook_4x2_size_bytes:,} 字节 ({codebook_4x2_size_bytes/1024:.1f} KB)")
    print(f"总码表数据: {codebook_size_bytes:,} 字节 ({codebook_size_bytes/1024:.1f} KB)")
    print(f"I帧数据: {i_frame_data_bytes:,} 字节 ({i_frame_data_bytes/1024:.1f} KB)")
    print(f"P帧数据: {p_frame_data_bytes:,} 字节 ({p_frame_data_bytes/1024:.1f} KB)")
    print(f"偏移表: {offsets_size_bytes:,} 字节 ({offsets_size_bytes/1024:.1f} KB)")
    print(f"帧类型表: {frame_types_size_bytes:,} 字节")
    print(f"总大小: {total_file_size:,} 字节 ({total_file_size/1024:.1f} KB)")
    
    print(f"\n📈 压缩效率:")
    original_size = total_frames * WIDTH * HEIGHT * 2  # 原始BGR555大小
    compression_ratio = original_size / total_file_size
    print(f"原始大小: {original_size:,} 字节 ({original_size/1024/1024:.1f} MB)")
    print(f"压缩后大小: {total_file_size:,} 字节 ({total_file_size/1024:.1f} KB)")
    print(f"压缩比: {compression_ratio:.1f}:1 ({100/compression_ratio:.1f}%)")
    
    print(f"✅ 编码完成！输出文件: {args.out}.h, {args.out}.c")

if __name__ == "__main__":
    main()