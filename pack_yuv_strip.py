from numba import njit
import numpy as np
from const_def import *
@njit
def clip_value(value, min_val, max_val):
    """Numba兼容的clip函数"""
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    else:
        return value

@njit
def pack_yuv420_strip_numba(bgr_strip, strip_height, width):
    """Numba加速的YUV420转换"""
    blocks_h = strip_height // BLOCK_H
    blocks_w = width // BLOCK_W
    
    block_array = np.zeros((blocks_h, blocks_w, BYTES_PER_2X2_BLOCK), dtype=np.uint8)
    
    for by in range(blocks_h):
        for bx in range(blocks_w):
            # 提取2x2块
            y_start = by * BLOCK_H
            x_start = bx * BLOCK_W
            
            # BGR to YUV conversion for 2x2 block
            cb_sum = 0.0
            cr_sum = 0.0
            y_values = np.zeros(4, dtype=np.uint8)
            
            idx = 0
            for dy in range(BLOCK_H):
                for dx in range(BLOCK_W):
                    if y_start + dy < strip_height and x_start + dx < width:
                        b = float(bgr_strip[y_start + dy, x_start + dx, 0])
                        g = float(bgr_strip[y_start + dy, x_start + dx, 1])  
                        r = float(bgr_strip[y_start + dy, x_start + dx, 2])
                        
                        y = r * 0.28571429 + g * 0.57142857 + b * 0.14285714
                        cb = r * (-0.14285714) + g * (-0.28571429) + b * 0.42857143
                        cr = r * 0.35714286 + g * (-0.28571429) + b * (-0.07142857)
                        
                        y_values[idx] = np.uint8(clip_value(y / 2.0, 0.0, 255.0))
                        cb_sum += cb
                        cr_sum += cr
                        idx += 1
            
            # Store Y values
            block_array[by, bx, 0:4] = y_values
            
            # Compute and store chroma
            cb_mean = cb_sum / 4.0
            cr_mean = cr_sum / 4.0
            
            d_r = clip_value(cr_mean, -128.0, 127.0)
            d_g = clip_value((-(cb_mean/2.0) - cr_mean) / 2.0, -128.0, 127.0)  
            d_b = clip_value(cb_mean, -128.0, 127.0)
            
            # 将有符号值转换为无符号字节存储
            block_array[by, bx, 4] = np.uint8(np.int8(d_r).view(np.uint8))
            block_array[by, bx, 5] = np.uint8(np.int8(d_g).view(np.uint8))
            block_array[by, bx, 6] = np.uint8(np.int8(d_b).view(np.uint8))
    
    return block_array

def pack_yuv420_strip(frame_bgr: np.ndarray, strip_y: int, strip_height: int) -> np.ndarray:
    """使用Numba加速的YUV转换包装函数"""
    strip_bgr = frame_bgr[strip_y:strip_y + strip_height, :, :]
    return pack_yuv420_strip_numba(strip_bgr, strip_height, WIDTH)