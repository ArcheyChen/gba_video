#!/usr/bin/env python3
"""
config.py - GBA视频编码器配置和常量定义
"""

import numpy as np

# 视频参数
WIDTH, HEIGHT = 240, 160
DEFAULT_STRIP_COUNT = 4
CODEBOOK_SIZE = 256

# 块参数
BLOCK_W, BLOCK_H = 2, 2
BYTES_PER_BLOCK = 7  # 4Y + d_r + d_g + d_b

# YUV转换系数矩阵
# 特殊的系数矩阵，可以很方便地转换成RGB
Y_COEFF  = np.array([0.28571429,  0.57142857,  0.14285714])
CB_COEFF = np.array([-0.14285714, -0.28571429,  0.42857143])
CR_COEFF = np.array([ 0.35714286, -0.28571429, -0.07142857])

# 帧类型标识
FRAME_TYPE_I = 0x00  # I帧（关键帧）
FRAME_TYPE_P = 0x01  # P帧（差分帧）

# 默认编码参数
DEFAULT_FPS = 30
DEFAULT_DURATION = 5.0
DEFAULT_I_FRAME_INTERVAL = 60
DEFAULT_DIFF_THRESHOLD = 2.0
DEFAULT_FORCE_I_THRESHOLD = 0.7
DEFAULT_KMEANS_MAX_ITER = 200