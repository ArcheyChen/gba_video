// gba_video_player.cpp  v3
// Mode 3 单缓冲 + YUV9 → RGB555 + 帧间差分解码

#include <gba_systemcalls.h>
#include <gba_video.h>
#include <gba_dma.h>
#include <gba_interrupt.h>
#include <gba_input.h>
#include <cstring>

#include "video_data.h"

constexpr int PIXELS_PER_FRAME = SCREEN_WIDTH * SCREEN_HEIGHT;

// EWRAM 单缓冲
EWRAM_BSS u16 ewramBuffer[PIXELS_PER_FRAME];
IWRAM_DATA static u8 clip_table_raw[768];
u8* lookup_table = clip_table_raw + 256; //预先添加偏移，这样查表的时候，遇到负数也直接查

void init_table(){
    for(int i=-256;i<768-256;i++){
        u8 raw_val;
        if(i<=0)
            raw_val = 0; // 小于等于0的值都裁剪为0
        else if(i>=255)
            raw_val = 255; // 大于等于255的值都裁剪为255
        else
            raw_val = static_cast<u8>(i); // 其他值直接赋值
        lookup_table[i] = raw_val>>3; // 填充查找表
    }
}

IWRAM_CODE inline u16 yuv_to_rgb555(u8 y, s16 d_r, s16 d_g, s16 d_b)
{
    // 近似整数 YUV → RGB
    // R = Y + (Cr << 1)
    // G = Y - (Cb >> 1) - Cr  
    // B = Y + (Cb << 1)
    u32 result = lookup_table[y + d_r];
    result |= (lookup_table[y + d_g] << 5);
    return result | (lookup_table[y + d_b] << 10);
}

// 解码单个4x4块到指定位置
IWRAM_CODE inline void decode_block(const u8* src, u16* dst_base, int block_x, int block_y)
{
    s8  Cb = static_cast<s8>(src[16]);
    s8  Cr = static_cast<s8>(src[17]);
    
    s16 d_r = Cr << 1;           // 2*Cr
    s16 d_g = -(Cb >> 1) - Cr;   // -Cb/2 - Cr
    s16 d_b = Cb << 1;           // 2*Cb
    
    // 计算块在帧缓冲中的起始位置
    u16* dst = dst_base + (block_y * 4 * SCREEN_WIDTH) + (block_x * 4);
    
    // 解码4x4像素
    for(int row = 0; row < 4; row++) {
        u16* dst_row = dst + row * SCREEN_WIDTH;
        const u8* y_row = src + row * 4;
        
        dst_row[0] = yuv_to_rgb555(y_row[0], d_r, d_g, d_b);
        dst_row[1] = yuv_to_rgb555(y_row[1], d_r, d_g, d_b);
        dst_row[2] = yuv_to_rgb555(y_row[2], d_r, d_g, d_b);
        dst_row[3] = yuv_to_rgb555(y_row[3], d_r, d_g, d_b);
    }
}

IWRAM_CODE void decode_i_frame(const u8* src, u16* dst)
{
    // 跳过帧类型标记
    src++;
    
    // 解码所有块
    for (int by = 0; by < BLOCKS_PER_COL; by++) {
        for (int bx = 0; bx < BLOCKS_PER_ROW; bx++) {
            decode_block(src, dst, bx, by);
            src += BYTES_PER_BLOCK;
        }
    }
}

IWRAM_CODE void decode_p_frame(const u8* src, u16* dst)
{
    // 跳过帧类型标记
    src++;
    
    // 读取需要更新的块数（小端序）
    u16 blocks_to_update = src[0] | (src[1] << 8);
    src += 2;
    
    // 防止越界
    if (blocks_to_update > TOTAL_BLOCKS) {
        blocks_to_update = TOTAL_BLOCKS;
    }
    
    // 处理每个需要更新的块
    for (u16 i = 0; i < blocks_to_update; i++) {
        // 读取块索引（小端序）
        u16 block_idx = src[0] | (src[1] << 8);
        src += 2;
        
        // 防止越界
        if (block_idx >= TOTAL_BLOCKS) {
            src += BYTES_PER_BLOCK;
            continue;
        }
        
        // 计算块坐标
        int bx = block_idx % BLOCKS_PER_ROW;
        int by = block_idx / BLOCKS_PER_ROW;
        
        // 解码这个块
        decode_block(src, dst, bx, by);
        src += BYTES_PER_BLOCK;
    }
}

// 添加调试用的帧计数器（可选）
#ifdef DEBUG
EWRAM_DATA int debug_i_frames = 0;
EWRAM_DATA int debug_p_frames = 0;
#endif

IWRAM_CODE void decode_frame(const u8* src, u16* dst)
{
    // 检查帧类型
    u8 frame_type = src[0];
    
    if (frame_type == FRAME_TYPE_I) {
        #ifdef DEBUG
        debug_i_frames++;
        #endif
        decode_i_frame(src, dst);
    } else if (frame_type == FRAME_TYPE_P) {
        #ifdef DEBUG
        debug_p_frames++;
        #endif
        decode_p_frame(src, dst);
    }
}

static volatile u32 vbl = 0;
void isr_vbl() { ++vbl; REG_IF = IRQ_VBLANK; }

int main()
{
    REG_DISPCNT = MODE_3 | BG2_ENABLE;

    irqInit();
    irqSet(IRQ_VBLANK, isr_vbl);
    irqEnable(IRQ_VBLANK);

    init_table();
    
    // 清空缓冲区
    memset(ewramBuffer, 0, PIXELS_PER_FRAME * sizeof(u16));
    VBlankIntrWait();
    DMA3COPY(ewramBuffer, VRAM, PIXELS_PER_FRAME | DMA16);

    int frame = 0;
    
    while (1)
    {
        // 使用偏移表获取当前帧的数据位置
        const unsigned char* frame_data = video_data + frame_offsets[frame];
        
        // 解码当前帧（I帧或P帧）
        decode_frame(frame_data, ewramBuffer);

        // 等待垂直同步并复制到VRAM
        VBlankIntrWait();
        DMA3COPY(ewramBuffer, VRAM, PIXELS_PER_FRAME | DMA16);

        frame++;
        if(frame >= VIDEO_FRAME_COUNT) {
            frame = 0; // 循环播放
        }

        // 按键检测（可选）
        scanKeys();
        u16 keys = keysDown();
        
        if (keys & KEY_START) {
            // 暂停功能
            while (!(keysDown() & KEY_START)) {
                scanKeys();
                VBlankIntrWait();
            }
        }
        
        if (keys & KEY_A) {
            // 快进：跳过5帧
            frame += 5;
            if (frame >= VIDEO_FRAME_COUNT) frame = 0;
        }
    }
}