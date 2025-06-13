
import numpy as np
from const_def import *
from block_quantizer import quantize_blocks_unified

def encode_strip_i_frame_with_4x4_blocks(blocks: np.ndarray, codebook_4x4: np.ndarray,
                                        codebook_2x2: np.ndarray, block_types: dict,
                                        block_4x4_indices: dict) -> bytes:
    """编码I帧条带"""
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
                if (super_by, super_bx) in block_types:
                    block_type = block_types[(super_by, super_bx)]
                    
                    if block_type == '4x4_blocks':
                        # 4x4块：0xFF + 4个4x4块码表索引
                        data.append(BLOCK_4X4_MARKER)
                        indices_4x4 = block_4x4_indices[(super_by, super_bx)]
                        for idx in indices_4x4:
                            data.append(idx)
                        
                    else:  # 2x2_blocks
                        # 纹理块：16个2x2块码表索引，按行优先顺序
                        for sub_by in range(4):
                            for sub_bx in range(4):
                                by = super_by * 4 + sub_by
                                bx = super_bx * 4 + sub_bx
                                if by < blocks_h and bx < blocks_w:
                                    block = blocks[by, bx]
                                    idx_2x2 = quantize_blocks_unified(block.reshape(1, -1), codebook_2x2)[0]
                                    data.append(idx_2x2)
                                else:
                                    data.append(0)
    
    return bytes(data)