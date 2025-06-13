#!/usr/bin/env python3

import argparse, cv2, numpy as np, pathlib, textwrap
import struct
import concurrent.futures
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from collections import defaultdict
import statistics
from numba import jit, njit, types
from numba.typed import List

from dither_opt import apply_dither_optimized
from encode_stat import EncodingStats
from pack_yuv_strip import pack_yuv420_strip
from const_def import *
from frame_encoder import *
from block_classifier import *
from block_quantizer import *
from codebook_generator import *
from file_writer import *

# 全局统计对象
encoding_stats = EncodingStats()

def calculate_strip_heights(height: int, strip_count: int) -> list:
    """计算每个条带的高度，确保每个条带高度都是4的倍数"""
    if height % 4 != 0:
        raise ValueError(f"视频高度 {height} 必须是4的倍数")
    
    base_height = (height // strip_count // 4) * 4
    remaining_height = height - (base_height * strip_count)
    
    strip_heights = []
    for i in range(strip_count):
        current_height = base_height
        if remaining_height >= 4:
            current_height += 4
            remaining_height -= 4
        strip_heights.append(current_height)
    
    if sum(strip_heights) != height:
        raise ValueError(f"条带高度分配错误: {strip_heights} 总和 {sum(strip_heights)} != {height}")
    
    for i, h in enumerate(strip_heights):
        if h % 4 != 0:
            raise ValueError(f"条带 {i} 高度 {h} 不是4的倍数")
    
    return strip_heights




def generate_gop_codebooks_with_4x4_blocks(frames: list, strip_count: int, i_frame_interval: int,
                                         variance_threshold: float, diff_threshold: float,
                                         distortion_threshold: float = 10.0,
                                         codebook_4x4_size: int = DEFAULT_4X4_CODEBOOK_SIZE,
                                         codebook_2x2_size: int = EFFECTIVE_UNIFIED_CODEBOOK_SIZE,
                                         kmeans_max_iter: int = 100, i_frame_weight: int = 3) -> dict:
    """为每个GOP生成4x4块码表和2x2块码表"""
    print("正在为每个GOP生成4x4块码表和2x2块码表...")
    
    gop_codebooks = {}
    
    i_frame_positions = []
    for frame_idx in range(len(frames)):
        if frame_idx % i_frame_interval == 0:
            i_frame_positions.append(frame_idx)
    
    for gop_idx, gop_start in enumerate(i_frame_positions):
        if gop_idx + 1 < len(i_frame_positions):
            gop_end = i_frame_positions[gop_idx + 1]
        else:
            gop_end = len(frames)
        
        print(f"  处理GOP {gop_idx}: 帧 {gop_start} 到 {gop_end-1}")
        
        gop_codebooks[gop_start] = []
        
        for strip_idx in range(strip_count):
            all_4x4_blocks = []
            all_2x2_blocks = []
            block_types_list = []
            
            # 处理GOP中的每一帧
            prev_strip_blocks = None
            
            for frame_idx in range(gop_start, gop_end):
                strip_blocks = frames[frame_idx][strip_idx]
                if strip_blocks.size == 0:
                    continue
                
                # 确定需要处理的8x8超级块
                is_i_frame = (frame_idx == gop_start)
                
                if is_i_frame:
                    blocks_h, blocks_w = strip_blocks.shape[:2]
                    super_blocks_h = blocks_h // 4
                    super_blocks_w = blocks_w // 4
                    updated_super_blocks = {(super_by, super_bx) for super_by in range(super_blocks_h) for super_bx in range(super_blocks_w)}
                else:
                    updated_super_blocks = identify_updated_8x8_super_blocks(strip_blocks, prev_strip_blocks, diff_threshold)
                
                # 从有效8x8超级块中提取数据用于训练码表
                for super_by, super_bx in updated_super_blocks:
                    blocks_8x8 = []
                    for sub_by in range(4):
                        for sub_bx in range(4):
                            by = super_by * 4 + sub_by
                            bx = super_bx * 4 + sub_bx
                            if by < strip_blocks.shape[0] and bx < strip_blocks.shape[1]:
                                blocks_8x8.append(strip_blocks[by, bx])
                            else:
                                blocks_8x8.append(np.zeros(BYTES_PER_2X2_BLOCK, dtype=np.uint8))
                    
                    # 将16个2x2块重组为4个4x4块
                    blocks_4x4_in_super = []
                    for quad_idx in range(4):
                        quad_by = quad_idx // 2
                        quad_bx = quad_idx % 2
                        blocks_2x2_in_4x4 = []
                        for sub_by in range(2):
                            for sub_bx in range(2):
                                block_idx = (quad_by * 2 + sub_by) * 4 + (quad_bx * 2 + sub_bx)
                                blocks_2x2_in_4x4.append(blocks_8x8[block_idx])
                        block_4x4 = pack_4x4_block_from_2x2_blocks(blocks_2x2_in_4x4)
                        blocks_4x4_in_super.append(block_4x4)
                    
                    # 添加到训练数据
                    if is_i_frame:
                        all_4x4_blocks.extend(blocks_4x4_in_super * i_frame_weight)
                        all_2x2_blocks.extend(blocks_8x8 * i_frame_weight)
                    else:
                        all_4x4_blocks.extend(blocks_4x4_in_super)
                        all_2x2_blocks.extend(blocks_8x8)
                
                prev_strip_blocks = strip_blocks.copy()
            
            # 生成码表
            codebook_4x4 = generate_4x4_codebook(all_4x4_blocks, codebook_4x4_size, kmeans_max_iter)
            codebook_2x2 = generate_unified_codebook_simplified(
                all_2x2_blocks, codebook_2x2_size, kmeans_max_iter)
            
            # 为每一帧生成分类信息
            for frame_idx in range(gop_start, gop_end):
                strip_blocks = frames[frame_idx][strip_idx]
                if strip_blocks.size == 0:
                    continue
                
                block_4x4_indices, _, block_types = classify_8x8_super_blocks_with_4x4_codebook(
                    strip_blocks, codebook_4x4, variance_threshold, distortion_threshold)
                block_types_list.append((frame_idx, block_types, block_4x4_indices))
            
            gop_codebooks[gop_start].append({
                'codebook_4x4': codebook_4x4,
                'codebook_2x2': codebook_2x2,
                'block_types_list': block_types_list,
                'distortion_threshold': distortion_threshold
            })
            
            print(f"    条带{strip_idx}: 4x4块{len(all_4x4_blocks)}个, 2x2块{len(all_2x2_blocks)}个")
    
    return gop_codebooks







def identify_updated_4x4_blocks(current_blocks: np.ndarray, prev_blocks: np.ndarray,
                               diff_threshold: float) -> dict:
    """识别需要更新的4x4块（以4x4块为单位）"""
    if prev_blocks is None or current_blocks.shape != prev_blocks.shape:
        # 如果没有前一帧，所有4x4块都需要更新
        blocks_h, blocks_w = current_blocks.shape[:2]
        super_blocks_h = blocks_h // 4
        super_blocks_w = blocks_w // 4
        updated_4x4_blocks = {}
        for super_by in range(super_blocks_h):
            for super_bx in range(super_blocks_w):
                updated_4x4_blocks[(super_by, super_bx)] = [0, 1, 2, 3]  # 所有4个4x4块都更新
        return updated_4x4_blocks
    
    blocks_h, blocks_w = current_blocks.shape[:2]
    super_blocks_h = blocks_h // 4
    super_blocks_w = blocks_w // 4
    updated_4x4_blocks = {}
    
    for super_by in range(super_blocks_h):
        for super_bx in range(super_blocks_w):
            updated_quads = []
            
            # 检查4个4x4子块
            for quad_idx in range(4):
                quad_by = quad_idx // 2
                quad_bx = quad_idx % 2
                
                # 计算该4x4块内4个2x2块的差异
                total_diff = 0.0
                for sub_by in range(2):
                    for sub_bx in range(2):
                        by = super_by * 4 + quad_by * 2 + sub_by
                        bx = super_bx * 4 + quad_bx * 2 + sub_bx
                        
                        if by < blocks_h and bx < blocks_w:
                            current_block = current_blocks[by, bx]
                            prev_block = prev_blocks[by, bx]
                            
                            # 计算Y分量差异
                            y_diff = np.mean(np.abs(current_block[:4].astype(np.float32) - 
                                                   prev_block[:4].astype(np.float32)))
                            total_diff += y_diff
                
                avg_diff = total_diff / 4.0  # 4个2x2块的平均差异
                if avg_diff > diff_threshold:
                    updated_quads.append(quad_idx)
            
            if updated_quads:
                updated_4x4_blocks[(super_by, super_bx)] = updated_quads
    
    return updated_4x4_blocks

def encode_strip_i_frame_mixed(blocks: np.ndarray, codebook_4x4: np.ndarray,
                              codebook_2x2: np.ndarray, distortion_threshold: float = 10.0) -> bytes:
    """编码I帧条带（混编模式）"""
    data = bytearray()
    data.append(FRAME_TYPE_I)
    
    if blocks.size > 0:
        blocks_h, blocks_w = blocks.shape[:2]
        super_blocks_h = blocks_h // 4
        super_blocks_w = blocks_w // 4
        
        # 存储4x4块码表
        data.extend(codebook_4x4.flatten().tobytes())
        
        # 存储2x2块码表
        data.extend(codebook_2x2.flatten().tobytes())
        
        # 按8x8超级块的顺序编码
        for super_by in range(super_blocks_h):
            for super_bx in range(super_blocks_w):
                # 收集当前8x8超级块的16个2x2块
                blocks_8x8 = []
                for sub_by in range(4):
                    for sub_bx in range(4):
                        by = super_by * 4 + sub_by
                        bx = super_bx * 4 + sub_bx
                        if by < blocks_h and bx < blocks_w:
                            blocks_8x8.append(blocks[by, bx])
                        else:
                            blocks_8x8.append(np.zeros(BYTES_PER_2X2_BLOCK, dtype=np.uint8))
                
                # 对4个4x4子块进行分类
                block_4x4_usage = classify_4x4_blocks_in_8x8_super_block(
                    blocks_8x8, codebook_4x4, codebook_2x2, distortion_threshold)
                
                # 编码4个4x4子块
                for quad_idx in range(4):
                    if quad_idx in block_4x4_usage:
                        mode, indices = block_4x4_usage[quad_idx]
                        if mode == '4x4':
                            # 4x4块：0xFF + 1个4x4块码表索引
                            data.append(BLOCK_4X4_MARKER)
                            data.append(indices)
                        else:  # mode == '2x2'
                            # 2x2块：4个2x2块码表索引
                            for idx in indices:
                                data.append(idx)
                    else:
                        # 出错情况，用全0填充
                        data.extend([0] * 4)
    
    return bytes(data)

def encode_strip_p_frame_mixed(current_blocks: np.ndarray, prev_blocks: np.ndarray,
                              codebook_4x4: np.ndarray, codebook_2x2: np.ndarray,
                              diff_threshold: float, force_i_threshold: float = 0.7,
                              distortion_threshold: float = 10.0) -> tuple:
    """编码P帧条带（混编模式 + 4x4块跳过）"""
    if prev_blocks is None or current_blocks.shape != prev_blocks.shape:
        i_frame_data = encode_strip_i_frame_mixed(
            current_blocks, codebook_4x4, codebook_2x2, distortion_threshold)
        return i_frame_data, True, 0, 0, 0
    
    blocks_h, blocks_w = current_blocks.shape[:2]
    super_blocks_h = blocks_h // 4
    super_blocks_w = blocks_w // 4
    total_4x4_blocks = super_blocks_h * super_blocks_w * 4
    
    if total_4x4_blocks == 0:
        return b'', True, 0, 0, 0
    
    # 识别需要更新的4x4块
    updated_4x4_blocks = identify_updated_4x4_blocks(current_blocks, prev_blocks, diff_threshold)
    
    # 计算更新比例（以4x4块为单位）
    total_updated_4x4 = sum(len(quads) for quads in updated_4x4_blocks.values())
    update_ratio = total_updated_4x4 / total_4x4_blocks if total_4x4_blocks > 0 else 0
    
    if update_ratio > force_i_threshold:
        i_frame_data = encode_strip_i_frame_mixed(
            current_blocks, codebook_4x4, codebook_2x2, distortion_threshold)
        return i_frame_data, True, 0, 0, 0
    
    # 编码P帧
    data = bytearray()
    data.append(FRAME_TYPE_P)
    
    # 计算区域数量
    zones_count = (super_blocks_h + ZONE_HEIGHT_SUPER_BLOCKS - 1) // ZONE_HEIGHT_SUPER_BLOCKS
    
    # 按区域组织更新
    zone_updates = [[] for _ in range(zones_count)]
    
    for (super_by, super_bx), updated_quads in updated_4x4_blocks.items():
        # 计算属于哪个区域
        zone_idx = min(super_by // ZONE_HEIGHT_SUPER_BLOCKS, zones_count - 1)
        zone_relative_by = super_by % ZONE_HEIGHT_SUPER_BLOCKS
        zone_relative_idx = zone_relative_by * super_blocks_w + super_bx
        
        # 收集当前8x8超级块的16个2x2块
        blocks_8x8 = []
        for sub_by in range(4):
            for sub_bx in range(4):
                by = super_by * 4 + sub_by
                bx = super_bx * 4 + sub_bx
                if by < blocks_h and bx < blocks_w:
                    blocks_8x8.append(current_blocks[by, bx])
                else:
                    blocks_8x8.append(np.zeros(BYTES_PER_2X2_BLOCK, dtype=np.uint8))
        
        # 对需要更新的4x4子块进行分类
        block_4x4_usage = classify_4x4_blocks_in_8x8_super_block(
            blocks_8x8, codebook_4x4, codebook_2x2, distortion_threshold)
        
        # 构建更新数据
        update_data = []
        for quad_idx in range(4):
            if quad_idx in updated_quads:
                mode, indices = block_4x4_usage[quad_idx]
                if mode == '4x4':
                    update_data.extend([BLOCK_4X4_MARKER, indices])
                else:  # mode == '2x2'
                    update_data.extend(indices)
            else:
                # 跳过该4x4块
                update_data.append(BLOCK_SKIP_MARKER)
        
        zone_updates[zone_idx].append((zone_relative_idx, update_data))
    
    # 生成区域bitmap
    zone_bitmap = 0
    used_zones = 0
    total_updates = 0
    
    for zone_idx in range(zones_count):
        if zone_updates[zone_idx]:
            zone_bitmap |= (1 << zone_idx)
            used_zones += 1
            total_updates += len(zone_updates[zone_idx])
    
    data.extend(struct.pack('<H', zone_bitmap))
    
    # 按区域编码更新
    for zone_idx in range(zones_count):
        if zone_bitmap & (1 << zone_idx):
            updates = zone_updates[zone_idx]
            data.append(len(updates))
            
            for relative_idx, update_data in updates:
                data.append(relative_idx)
                for byte_val in update_data:
                    data.append(byte_val)
    
    return bytes(data), False, used_zones, total_updates, 0


def generate_codebook(blocks_data: np.ndarray, codebook_size: int, max_iter: int = 100) -> tuple:
    """使用K-Means聚类生成码表"""
    if len(blocks_data) == 0:
        return np.zeros((codebook_size, BYTES_PER_2X2_BLOCK), dtype=np.uint8), 0
    
    if blocks_data.ndim > 2:
        blocks_data = blocks_data.reshape(-1, BYTES_PER_2X2_BLOCK)
    
    # 移除去重操作，直接使用原始数据进行聚类
    # 这样K-Means可以基于数据的真实分布（包括频次）进行更好的聚类
    effective_size = min(len(blocks_data), codebook_size)
    
    if len(blocks_data) <= codebook_size:
        # 如果数据量小于码本大小，需要去重避免重复
        blocks_as_tuples = [tuple(block) for block in blocks_data]
        unique_tuples = list(set(blocks_as_tuples))
        unique_blocks = np.array(unique_tuples, dtype=np.uint8)
        
        codebook = np.zeros((codebook_size, BYTES_PER_2X2_BLOCK), dtype=np.uint8)
        codebook[:len(unique_blocks)] = unique_blocks
        if len(unique_blocks) > 0:
            for i in range(len(unique_blocks), codebook_size):
                codebook[i] = unique_blocks[-1]
        return codebook, len(unique_blocks)
    
    # 对于大数据集，直接进行K-Means聚类
    kmeans = MiniBatchKMeans(
        n_clusters=codebook_size,
        random_state=42,
        batch_size=min(1000, len(blocks_data)),
        max_iter=max_iter,
        n_init=3
    )
    blocks_for_clustering = convert_blocks_for_clustering(blocks_data)
    kmeans.fit(blocks_for_clustering)
    codebook = convert_codebook_from_clustering(kmeans.cluster_centers_)
    
    return codebook, codebook_size


def convert_codebook_from_clustering(codebook_float: np.ndarray) -> np.ndarray:
    """将聚类结果转换回正确的块格式"""
    codebook = np.zeros_like(codebook_float, dtype=np.uint8)
    
    codebook[:, 0:4] = np.clip(codebook_float[:, 0:4].round(), 0, 255).astype(np.uint8)
    
    for i in range(4, BYTES_PER_2X2_BLOCK):
        clipped_values = np.clip(codebook_float[:, i].round(), -128, 127).astype(np.int8)
        codebook[:, i] = clipped_values.view(np.uint8)
    
    return codebook



def generate_unified_codebook_simplified(small_blocks: list, 
                                       codebook_size: int = EFFECTIVE_UNIFIED_CODEBOOK_SIZE,
                                       kmeans_max_iter: int = 100) -> np.ndarray:
    """生成2x2小块的统一码表（254项，避免0xFE）"""
    if small_blocks:
        blocks_array = np.array(small_blocks)
        codebook, _ = generate_codebook(blocks_array, codebook_size, kmeans_max_iter)
        
        # 创建254项码表
        full_codebook = np.zeros((codebook_size, BYTES_PER_2X2_BLOCK), dtype=np.uint8)
        actual_size = min(len(codebook), codebook_size)
        full_codebook[:actual_size] = codebook[:actual_size]
        
        # 填充剩余项
        if actual_size > 0:
            for i in range(actual_size, codebook_size):
                full_codebook[i] = full_codebook[actual_size - 1]
    else:
        full_codebook = np.zeros((codebook_size, BYTES_PER_2X2_BLOCK), dtype=np.uint8)
    
    return full_codebook



def main():
    pa = argparse.ArgumentParser(description="Encode to GBA YUV9 with mixed 4x4/2x2 block codebook")
    pa.add_argument("input")
    pa.add_argument("--duration", type=float, default=5.0)
    pa.add_argument("--full-duration", action="store_true")
    pa.add_argument("--fps", type=int, default=30)
    pa.add_argument("--out", default="video_data")
    pa.add_argument("--strip-count", type=int, default=DEFAULT_STRIP_COUNT)
    pa.add_argument("--i-frame-interval", type=int, default=60)
    pa.add_argument("--diff-threshold", type=float, default=2.0)
    pa.add_argument("--force-i-threshold", type=float, default=0.7)
    pa.add_argument("--variance-threshold", type=float, default=5.0)
    pa.add_argument("--distortion-threshold", type=float, default=10.0,
                   help="失真阈值，用于决定是否使用4x4块码表（默认10.0）")
    pa.add_argument("--codebook-4x4-size", type=int, default=DEFAULT_4X4_CODEBOOK_SIZE)
    pa.add_argument("--codebook-2x2-size", type=int, default=EFFECTIVE_UNIFIED_CODEBOOK_SIZE)
    pa.add_argument("--kmeans-max-iter", type=int, default=200)
    pa.add_argument("--threads", type=int, default=None)
    pa.add_argument("--i-frame-weight", type=int, default=3)
    pa.add_argument("--dither", action="store_true")

    args = pa.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit("❌ 打不开输入文件")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    every = int(round(src_fps / args.fps))
    
    if args.full_duration:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        grab_max = total_frames
        actual_duration = total_frames / src_fps
        print(f"编码整个视频: {total_frames} 帧，时长 {actual_duration:.2f} 秒")
    else:
        grab_max = int(args.duration * src_fps)
        print(f"编码时长: {args.duration} 秒 ({grab_max} 帧)")

    strip_heights = calculate_strip_heights(HEIGHT, args.strip_count)
    print(f"条带配置: {args.strip_count} 个条带，高度分别为: {strip_heights}")
    print(f"码本配置: 4x4块码表{args.codebook_4x4_size}项, 2x2块码表{args.codebook_2x2_size}项")
    print(f"🔄 已启用混编模式（4x4+2x2混合编码）")
    if args.dither:
        print(f"🎨 已启用抖动算法（蛇形扫描）")
    
    frames = []
    idx = 0
    print("正在提取帧...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        while idx < grab_max:
            ret, frm = cap.read()
            if not ret:
                break
            if idx % every == 0:
                frm = cv2.resize(frm, (WIDTH, HEIGHT), cv2.INTER_AREA)
                # if args.dither:
                #     frm = apply_dither_optimized(frm)
                strip_y_list = []
                y = 0
                for strip_height in strip_heights:
                    strip_y_list.append((frm, y, strip_height))
                    y += strip_height
                future_to_idx = {
                    executor.submit(pack_yuv420_strip, *args): i
                    for i, args in enumerate(strip_y_list)
                }
                frame_strips = [None] * len(strip_y_list)
                for future in concurrent.futures.as_completed(future_to_idx):
                    i = future_to_idx[future]
                    frame_strips[i] = future.result()
                frames.append(frame_strips)
                
                if len(frames) % 30 == 0:
                    print(f"  已提取 {len(frames)} 帧")
            idx += 1
    cap.release()

    if not frames:
        raise SystemExit("❌ 没有任何帧被采样")

    print(f"总共提取了 {len(frames)} 帧")

    # 生成码表
    gop_codebooks = generate_gop_codebooks_with_4x4_blocks(
        frames, args.strip_count, args.i_frame_interval, 
        args.variance_threshold, args.diff_threshold, args.distortion_threshold,
        args.codebook_4x4_size, args.codebook_2x2_size,
        args.kmeans_max_iter, args.i_frame_weight
    )

    # 编码所有帧
    print("正在编码帧...")
    encoded_frames = []
    frame_offsets = []
    current_offset = 0
    prev_strips = [None] * args.strip_count
    
    for frame_idx, current_strips in enumerate(frames):
        frame_offsets.append(current_offset)
        
        # 找到当前GOP
        gop_start = (frame_idx // args.i_frame_interval) * args.i_frame_interval
        gop_data = gop_codebooks[gop_start]
        
        frame_data = bytearray()
        
        for strip_idx, current_strip in enumerate(current_strips):
            strip_gop_data = gop_data[strip_idx]
            codebook_4x4 = strip_gop_data['codebook_4x4']
            codebook_2x2 = strip_gop_data['codebook_2x2']
            distortion_threshold = strip_gop_data['distortion_threshold']
            
            force_i_frame = (frame_idx % args.i_frame_interval == 0) or frame_idx == 0
            
            if force_i_frame or prev_strips[strip_idx] is None:
                # 使用新的混编I帧编码
                strip_data = encode_strip_i_frame_mixed(
                    current_strip, codebook_4x4, codebook_2x2, distortion_threshold
                )
                is_i_frame = True
                
                # 计算码本和索引大小
                codebook_4x4_size = args.codebook_4x4_size * BYTES_PER_4X4_BLOCK
                codebook_2x2_size = args.codebook_2x2_size * BYTES_PER_2X2_BLOCK
                index_size = len(strip_data) - 1 - codebook_4x4_size - codebook_2x2_size
                
                encoding_stats.add_i_frame(
                    strip_idx, len(strip_data), 
                    is_forced=force_i_frame,
                    codebook_size=codebook_4x4_size + codebook_2x2_size,
                    index_size=max(0, index_size)
                )
            else:
                # 使用新的混编P帧编码
                strip_data, is_i_frame, used_zones, total_updates, _ = encode_strip_p_frame_mixed(
                    current_strip, prev_strips[strip_idx],
                    codebook_4x4, codebook_2x2,
                    args.diff_threshold, args.force_i_threshold, distortion_threshold
                )
                
                if is_i_frame:
                    codebook_4x4_size = args.codebook_4x4_size * BYTES_PER_4X4_BLOCK
                    codebook_2x2_size = args.codebook_2x2_size * BYTES_PER_2X2_BLOCK
                    index_size = len(strip_data) - 1 - codebook_4x4_size - codebook_2x2_size
                    
                    encoding_stats.add_i_frame(
                        strip_idx, len(strip_data), 
                        is_forced=False,
                        codebook_size=codebook_4x4_size + codebook_2x2_size,
                        index_size=max(0, index_size)
                    )
                else:
                    encoding_stats.add_p_frame(
                        strip_idx, len(strip_data), total_updates, used_zones,
                        0, 0  # 这些统计在混编模式下不再适用
                    )
            
            frame_data.extend(struct.pack('<H', len(strip_data)))
            frame_data.extend(strip_data)
            
            prev_strips[strip_idx] = current_strip.copy() if current_strip.size > 0 else None
        
        encoded_frames.append(bytes(frame_data))
        current_offset += len(frame_data)
        
        if frame_idx % 30 == 0 or frame_idx == len(frames) - 1:
            print(f"  已编码 {frame_idx + 1}/{len(frames)} 帧")
    
    all_data = b''.join(encoded_frames)
    
    write_header(pathlib.Path(args.out).with_suffix(".h"), len(frames), len(all_data), 
                args.strip_count, strip_heights, args.codebook_4x4_size, args.codebook_2x2_size)
    write_source(pathlib.Path(args.out).with_suffix(".c"), all_data, frame_offsets, strip_heights)
    
    # 打印详细统计
    encoding_stats.print_summary(len(frames), len(all_data))

if __name__ == "__main__":
    main()