#!/usr/bin/env python3
"""
yuv_converter.py - YUV颜色空间转换和块打包
"""

import numpy as np
from config import (Y_COEFF, CB_COEFF, CR_COEFF, BLOCK_W, BLOCK_H, BYTES_PER_BLOCK)


def pack_yuv420_strip(frame_bgr: np.ndarray, strip_y: int, strip_height: int) -> np.ndarray:
    """
    向量化实现，把指定条带的 240×strip_height×3 BGR → YUV420：每 2×2 像素 7 Byte
    布局按行优先：(Y>>1) (Y>>1) (Y>>1) (Y>>1) d_r d_g d_b
    返回形状为 (strip_blocks_h, blocks_w, 7) 的数组，每个元素是一个2x2块
    """
    strip_bgr = frame_bgr[strip_y:strip_y + strip_height, :, :]
    B = strip_bgr[:, :, 0].astype(np.float32)
    G = strip_bgr[:, :, 1].astype(np.float32)
    R = strip_bgr[:, :, 2].astype(np.float32)

    Y  = (R*Y_COEFF[0]  + G*Y_COEFF[1]  + B*Y_COEFF[2]).round()
    Cb = (R*CB_COEFF[0] + G*CB_COEFF[1] + B*CB_COEFF[2]).round()
    Cr = (R*CR_COEFF[0] + G*CR_COEFF[1] + B*CR_COEFF[2]).round()

    Y  = np.clip(Y,  0, 255).astype(np.uint8)
    Cb = np.clip(Cb, -128, 127).astype(np.int16)
    Cr = np.clip(Cr, -128, 127).astype(np.int16)

    h, w = strip_bgr.shape[:2]
    blocks_h = h // BLOCK_H
    blocks_w = w // BLOCK_W

    # reshape为块结构: (blocks_h, 2, blocks_w, 2)
    Y_blocks  = Y.reshape(blocks_h, BLOCK_H, blocks_w, BLOCK_W)
    Cb_blocks = Cb.reshape(blocks_h, BLOCK_H, blocks_w, BLOCK_W)
    Cr_blocks = Cr.reshape(blocks_h, BLOCK_H, blocks_w, BLOCK_W)

    # 预处理Y值：右移1位
    y_flat = (Y_blocks.transpose(0,2,1,3).reshape(blocks_h, blocks_w, 4) >> 1).astype(np.uint8)
    
    # Cb/Cr平均
    cb_mean = np.clip(Cb_blocks.mean(axis=(1,3)).round(), -128, 127).astype(np.int16)
    cr_mean = np.clip(Cr_blocks.mean(axis=(1,3)).round(), -128, 127).astype(np.int16)
    
    # 预计算差值并右移1位: d_r = Cr>>1, d_g = (-(Cb>>1)-Cr)>>1, d_b = Cb>>1
    d_r = np.clip(cr_mean, -128, 127).astype(np.int8)  # Cr>>1
    d_g = np.clip((-(cb_mean >> 1) - cr_mean) >> 1, -128, 127).astype(np.int8)  # (-(Cb>>1)-Cr)>>1
    d_b = np.clip(cb_mean, -128, 127).astype(np.int8)  # Cb>>1

    # 合并：4个Y值(>>1) + d_r + d_g + d_b
    block_array = np.zeros((blocks_h, blocks_w, BYTES_PER_BLOCK), dtype=np.uint8)
    block_array[..., 0:4] = y_flat
    block_array[..., 4] = d_r.view(np.uint8)  # d_r
    block_array[..., 5] = d_g.view(np.uint8)  # d_g
    block_array[..., 6] = d_b.view(np.uint8)  # d_b
    
    return block_array