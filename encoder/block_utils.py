 
from numba import jit, prange
import numpy as np 

WIDTH, HEIGHT = 240, 160
BLOCK_8x4_W, BLOCK_8x4_H = 8, 4
BLOCK_4x4_W, BLOCK_4x4_H = 4, 4
BLOCK_4x2_W, BLOCK_4x2_H = 4, 2

@jit(nopython=True, cache=True)
def convert_bgr_to_yuv(B, G, R):
    Y  = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B
    return Y, Cb, Cr

@jit(nopython=True, cache=True)
def extract_blocks_from_yuv(Y, Cb, Cr, height, width, block_h, block_w):
    num_blocks_y = height // block_h
    num_blocks_x = width // block_w
    total_blocks = num_blocks_y * num_blocks_x
    blocks = np.zeros((total_blocks, 24), dtype=np.uint8)
    block_idx = 0
    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            y_start = by * block_h
            x_start = bx * block_w
            for py in range(block_h):
                for px in range(block_w):
                    blocks[block_idx, py * block_w + px] = Y[y_start + py, x_start + px]
            for py in range(block_h):
                for px in range(block_w):
                    blocks[block_idx, 8 + py * block_w + px] = Cb[y_start + py, x_start + px]
            for py in range(block_h):
                for px in range(block_w):
                    blocks[block_idx, 16 + py * block_w + px] = Cr[y_start + py, x_start + px]
            block_idx += 1
    return blocks

@jit(nopython=True, cache=True)
def extract_blocks_from_yuv_8x4(Y, Cb, Cr, height, width, block_h, block_w):
    num_blocks_y = height // block_h
    num_blocks_x = width // block_w
    total_blocks = num_blocks_y * num_blocks_x
    blocks = np.zeros((total_blocks, 96), dtype=np.uint8)
    block_idx = 0
    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            y_start, y_end = by * block_h, (by + 1) * block_h
            x_start, x_end = bx * block_w, (bx + 1) * block_w
            y_block = Y[y_start:y_end, x_start:x_end].flatten()
            blocks[block_idx, :32] = y_block
            cb_block = Cb[y_start:y_end, x_start:x_end].flatten()
            blocks[block_idx, 32:64] = cb_block
            cr_block = Cr[y_start:y_end, x_start:x_end].flatten()
            blocks[block_idx, 64:96] = cr_block
            block_idx += 1
    return blocks

@jit(nopython=True, cache=True)
def extract_blocks_from_yuv_4x4(Y, Cb, Cr, height, width, block_h, block_w):
    num_blocks_y = height // block_h
    num_blocks_x = width // block_w
    total_blocks = num_blocks_y * num_blocks_x
    blocks = np.zeros((total_blocks, 48), dtype=np.uint8)
    block_idx = 0
    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            y_start = by * block_h
            x_start = bx * block_w
            for py in range(block_h):
                for px in range(block_w):
                    blocks[block_idx, py * block_w + px] = Y[y_start + py, x_start + px]
            for py in range(block_h):
                for px in range(block_w):
                    blocks[block_idx, 16 + py * block_w + px] = Cb[y_start + py, x_start + px]
            for py in range(block_h):
                for px in range(block_w):
                    blocks[block_idx, 32 + py * block_w + px] = Cr[y_start + py, x_start + px]
            block_idx += 1
    return blocks

def extract_yuv444_blocks_4x2(frame_bgr: np.ndarray) -> np.ndarray:
    B = frame_bgr[:, :, 0].astype(np.float32)
    G = frame_bgr[:, :, 1].astype(np.float32)
    R = frame_bgr[:, :, 2].astype(np.float32)
    Y, Cb, Cr = convert_bgr_to_yuv(B, G, R)
    Y  = np.clip(np.round(Y), 0, 255).astype(np.uint8)
    Cb = np.clip(np.round(Cb + 128), 0, 255).astype(np.uint8)
    Cr = np.clip(np.round(Cr + 128), 0, 255).astype(np.uint8)
    blocks = extract_blocks_from_yuv(Y, Cb, Cr, HEIGHT, WIDTH, BLOCK_4x2_H, BLOCK_4x2_W)
    return blocks

def extract_yuv444_blocks_8x4(frame_bgr: np.ndarray) -> np.ndarray:
    B = frame_bgr[:, :, 0].astype(np.float32)
    G = frame_bgr[:, :, 1].astype(np.float32)
    R = frame_bgr[:, :, 2].astype(np.float32)
    Y, Cb, Cr = convert_bgr_to_yuv(B, G, R)
    Y  = np.clip(np.round(Y), 0, 255).astype(np.uint8)
    Cb = np.clip(np.round(Cb + 128), 0, 255).astype(np.uint8)
    Cr = np.clip(np.round(Cr + 128), 0, 255).astype(np.uint8)
    blocks = extract_blocks_from_yuv_8x4(Y, Cb, Cr, HEIGHT, WIDTH, BLOCK_8x4_H, BLOCK_8x4_W)
    return blocks

def extract_yuv444_blocks_4x4(frame_bgr: np.ndarray) -> np.ndarray:
    B = frame_bgr[:, :, 0].astype(np.float32)
    G = frame_bgr[:, :, 1].astype(np.float32)
    R = frame_bgr[:, :, 2].astype(np.float32)
    Y, Cb, Cr = convert_bgr_to_yuv(B, G, R)
    Y  = np.clip(np.round(Y), 0, 255).astype(np.uint8)
    Cb = np.clip(np.round(Cb + 128), 0, 255).astype(np.uint8)
    Cr = np.clip(np.round(Cr + 128), 0, 255).astype(np.uint8)
    blocks = extract_blocks_from_yuv_4x4(Y, Cb, Cr, HEIGHT, WIDTH, BLOCK_4x4_H, BLOCK_4x4_W)
    return blocks

@jit(nopython=True, cache=True)
def yuv444_to_bgr555_jit(yuv444_block):
    y_values = yuv444_block[:8].astype(np.float32)
    cb_values = yuv444_block[8:16].astype(np.float32) - 128
    cr_values = yuv444_block[16:24].astype(np.float32) - 128
    bgr555_values = np.zeros(8, dtype=np.uint16)
    for i in range(8):
        Y = y_values[i]
        Cb = cb_values[i]
        Cr = cr_values[i]
        R = Y + 1.402 * Cr
        G = Y - 0.344136 * Cb - 0.714136 * Cr
        B = Y + 1.772 * Cb
        R = max(0.0, min(255.0, R))
        G = max(0.0, min(255.0, G))
        B = max(0.0, min(255.0, B))
        R5 = int(R * 31 / 255)
        G5 = int(G * 31 / 255)
        B5 = int(B * 31 / 255)
        bgr555_values[i] = (B5 << 10) | (G5 << 5) | R5
    return bgr555_values

def yuv444_to_bgr555(yuv444_block: np.ndarray) -> np.ndarray:
    return yuv444_to_bgr555_jit(yuv444_block)

@jit(nopython=True, cache=True)
def yuv444_to_bgr555_4x4_jit(yuv444_block):
    y_values = yuv444_block[:16].astype(np.float32)
    cb_values = yuv444_block[16:32].astype(np.float32) - 128
    cr_values = yuv444_block[32:48].astype(np.float32) - 128
    bgr555_values = np.zeros(16, dtype=np.uint16)
    for i in range(16):
        y = y_values[i]
        cb = cb_values[i]
        cr = cr_values[i]
        R = y + 1.402 * cr
        G = y - 0.344136 * cb - 0.714136 * cr
        B = y + 1.772 * cb
        R = max(0, min(255, R))
        G = max(0, min(255, G))
        B = max(0, min(255, B))
        R5 = int(R * 31 / 255)
        G5 = int(G * 31 / 255)
        B5 = int(B * 31 / 255)
        bgr555_values[i] = (B5 << 10) | (G5 << 5) | R5
    return bgr555_values

def yuv444_to_bgr555_4x4(yuv444_block: np.ndarray) -> np.ndarray:
    return yuv444_to_bgr555_4x4_jit(yuv444_block)

@jit(nopython=True, cache=True)
def yuv444_to_bgr555_8x4_jit(yuv444_block):
    y_values = yuv444_block[:32].astype(np.float32)
    cb_values = yuv444_block[32:64].astype(np.float32) - 128
    cr_values = yuv444_block[64:96].astype(np.float32) - 128
    bgr555_values = np.zeros(32, dtype=np.uint16)
    for i in range(32):
        y = y_values[i]
        cb = cb_values[i]
        cr = cr_values[i]
        R = y + 1.402 * cr
        G = y - 0.344136 * cb - 0.714136 * cr
        B = y + 1.772 * cb
        R = max(0, min(255, R))
        G = max(0, min(255, G))
        B = max(0, min(255, B))
        R5 = int(R * 31 / 255)
        G5 = int(G * 31 / 255)
        B5 = int(B * 31 / 255)
        bgr555_values[i] = (B5 << 10) | (G5 << 5) | R5
    return bgr555_values

def yuv444_to_bgr555_8x4(yuv444_block: np.ndarray) -> np.ndarray:
    return yuv444_to_bgr555_8x4_jit(yuv444_block)

@jit(nopython=True, cache=True)
def calculate_block_difference(block1, block2):
    diff = 0.0
    for i in range(24):
        d = float(block1[i]) - float(block2[i])
        diff += d * d
    return diff

@jit(nopython=True, cache=True)
def calculate_block_difference_8x4(block1, block2):
    diff = 0.0
    for i in range(96):
        diff += (float(block1[i]) - block2[i]) ** 2
    return diff

@jit(nopython=True, cache=True)
def calculate_block_difference_4x4(block1, block2):
    diff = 0.0
    for i in range(48):
        d = float(block1[i]) - float(block2[i])
        diff += d * d
    return diff

@jit(nopython=True, cache=True)
def find_changed_blocks(current_blocks, previous_blocks, threshold):
    num_blocks = current_blocks.shape[0]
    temp_indices = np.zeros(num_blocks, dtype=np.int32)
    count = 0
    for i in range(num_blocks):
        diff = calculate_block_difference(current_blocks[i], previous_blocks[i])
        if diff > threshold:
            temp_indices[count] = i
            count += 1
    if count > 0:
        return temp_indices[:count].copy()
    else:
        return np.zeros(0, dtype=np.int32)

@jit(nopython=True, cache=True)
def find_changed_blocks_8x4(current_blocks, previous_blocks, threshold):
    num_blocks = current_blocks.shape[0]
    temp_indices = np.zeros(num_blocks, dtype=np.int32)
    count = 0
    for i in range(num_blocks):
        diff = calculate_block_difference_8x4(current_blocks[i], previous_blocks[i])
        if diff > threshold:
            temp_indices[count] = i
            count += 1
    if count > 0:
        return temp_indices[:count]
    else:
        return np.zeros(0, dtype=np.int32)

@jit(nopython=True, cache=True)
def find_changed_blocks_4x4(current_blocks, previous_blocks, threshold):
    num_blocks = current_blocks.shape[0]
    temp_indices = np.zeros(num_blocks, dtype=np.int32)
    count = 0
    for i in range(num_blocks):
        diff = calculate_block_difference_4x4(current_blocks[i], previous_blocks[i])
        if diff > threshold:
            temp_indices[count] = i
            count += 1
    if count > 0:
        return temp_indices[:count].copy()
    else:
        return np.zeros(0, dtype=np.int32)

@jit(nopython=True, cache=True, parallel=True)
def compute_distances_jit(blocks, codebook):
    n_blocks = blocks.shape[0]
    n_codewords = codebook.shape[0]
    indices = np.zeros(n_blocks, dtype=np.uint16)
    for i in prange(n_blocks):
        min_dist = np.inf
        min_idx = 0
        for j in range(n_codewords):
            dist = 0.0
            for k in range(blocks.shape[1]):
                diff = float(blocks[i, k]) - codebook[j, k]
                dist += diff * diff
            if dist < min_dist:
                min_dist = dist
                min_idx = j
        indices[i] = min_idx
    return indices

def encode_frame_with_codebook(blocks: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    indices = compute_distances_jit(blocks, codebook.astype(np.float32))
    return indices
