import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def floyd_steinberg_jit(img):
    """
    使用Numba JIT优化的Floyd-Steinberg抖动算法（蛇形扫描）
    img: H×W×3 float32 array
    """
    h, w, c = img.shape
    
    for y in range(h):
        # 蛇形扫描：偶数行从左到右，奇数行从右到左
        if y % 2 == 0:
            # 从左到右扫描
            for x in range(w):
                for channel in range(c):
                    old_pixel = img[y, x, channel]
                    
                    # 量化到RGB555精度
                    new_pixel = np.float32((round(old_pixel) >> 3) << 3)
                    img[y, x, channel] = new_pixel
                    
                    # 计算量化误差
                    quant_error = old_pixel - new_pixel
                    
                    # 误差扩散（从左到右）
                    # 右 (7/16)
                    if x + 1 < w:
                        img[y, x + 1, channel] += quant_error * 0.4375
                    
                    # 下一行的像素
                    if y + 1 < h:
                        # 左下 (3/16)
                        if x > 0:
                            img[y + 1, x - 1, channel] += quant_error * 0.1875
                        # 正下 (5/16)
                        img[y + 1, x, channel] += quant_error * 0.3125
                        # 右下 (1/16)
                        if x + 1 < w:
                            img[y + 1, x + 1, channel] += quant_error * 0.0625
        else:
            # 从右到左扫描
            for x in range(w - 1, -1, -1):
                for channel in range(c):
                    old_pixel = img[y, x, channel]
                    
                    # 量化到RGB555精度
                    new_pixel = np.float32((round(old_pixel) >> 3) << 3)
                    img[y, x, channel] = new_pixel
                    
                    # 计算量化误差
                    quant_error = old_pixel - new_pixel
                    
                    # 误差扩散（从右到左）
                    # 左 (7/16)
                    if x - 1 >= 0:
                        img[y, x - 1, channel] += quant_error * 0.4375
                    
                    # 下一行的像素
                    if y + 1 < h:
                        # 右下 (3/16)
                        if x + 1 < w:
                            img[y + 1, x + 1, channel] += quant_error * 0.1875
                        # 正下 (5/16)
                        img[y + 1, x, channel] += quant_error * 0.3125
                        # 左下 (1/16)
                        if x - 1 >= 0:
                            img[y + 1, x - 1, channel] += quant_error * 0.0625
                        
    return img

def apply_dither_optimized(image_bgr: np.ndarray) -> np.ndarray:
    """
    优化的Floyd-Steinberg抖动算法，专门针对RGB555
    image_bgr: H×W×3 uint8
    return: H×W×3 uint8 (已应用抖动的图像)
    """
    # 转换为float32以提高计算速度
    img = image_bgr.astype(np.float32)
    img = floyd_steinberg_jit(img)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img