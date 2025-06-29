// gba_video_player.cpp  v3
// Mode 3 单缓冲 + 码表解码 I帧

#include <gba_systemcalls.h>
#include <gba_video.h>
#include <gba_dma.h>
#include <gba_interrupt.h>
#include <gba_input.h>

#include "video_data.h"

constexpr int PIXELS_PER_FRAME = SCREEN_WIDTH * SCREEN_HEIGHT;

// EWRAM 单缓冲
EWRAM_BSS u16 ewramBuffer[PIXELS_PER_FRAME];

// 裁切查找表
IWRAM_DATA static u8 clip_table_raw[1024];
u8* clip_lookup_table = clip_table_raw + 256;

void init_clip_table(){
    for(int i = -256; i < 1024 - 256; i++){
        u8 raw_val;
        if(i <= 0)
            raw_val = 0;
        else if(i >= 255)
            raw_val = 255;
        else
            raw_val = static_cast<u8>(i);
        clip_lookup_table[i] = raw_val >> 3;
    }
}
// 从码表中解码一个4x4块
IWRAM_CODE void decode_block_from_codebook(const s16* codeword, u16* dst, int dst_stride)
{
    // 从码字中提取数据：16个Y + 1个Cb + 1个Cr
    u8 Y[16];
    for(int i = 0; i < 16; i++) {
        Y[i] = static_cast<u8>(codeword[i]);
    }
    s8 Cb = static_cast<s8>(codeword[16]);
    s8 Cr = static_cast<s8>(codeword[17]);

    // 计算色度偏移
    s16 d_r = Cr << 1;           // 2*Cr
    s16 d_g = -(Cb >> 1) - Cr;   // -Cb/2 - Cr
    s16 d_b = Cb << 1;           // 2*Cb

    // 填充4x4块
    for(int y = 0; y < 4; y++) {
        u16* row = dst + y * dst_stride;
        for(int x = 0; x < 4; x++) {
            u8 luma = Y[y * 4 + x];
            
            s16 r = clip_lookup_table[luma + d_r];
            s16 g = clip_lookup_table[luma + d_g]; 
            s16 b = clip_lookup_table[luma + d_b];

            row[x] = (r) | ((g) << 5) | ((b) << 10);  // RGB555
        }
    }
}


IWRAM_CODE void decode_frame(const u16* frame_indices, u16* dst)
{
    const s16* codebook = video_codebook;
    
    int block_idx = 0;
    for (int y = 0; y < SCREEN_HEIGHT; y += 4)
    {
        for (int x = 0; x < SCREEN_WIDTH; x += 4)
        {
            // 获取当前块的码字索引
            u16 codeword_idx = frame_indices[block_idx++];
            
            // 从码表中获取码字 (18个s16值)
            const s16* codeword = codebook + codeword_idx * VIDEO_BLOCK_SIZE;
            
            // 解码4x4块到目标位置
            u16* block_dst = dst + y * SCREEN_WIDTH + x;
            decode_block_from_codebook(codeword, block_dst, SCREEN_WIDTH);
        }
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

    const u16* frame_indices = video_frame_indices;
    const int frames_total = VIDEO_FRAME_COUNT;

    int frame = 0;
    init_clip_table();
    
    while (1)
    {
        // 计算当前帧的索引数据起始位置
        const u16* current_frame_indices = frame_indices + frame * VIDEO_BLOCKS_PER_FRAME;

        decode_frame(current_frame_indices, ewramBuffer);

        DMA3COPY(ewramBuffer, VRAM, PIXELS_PER_FRAME | DMA16);

        frame++;
        if(frame >= frames_total) // 循环播放
        {
            frame = 0; // 重置帧计数
        }

        // 想加暂停 / 退出可自行检测按键
        scanKeys();
        u16 keys = keysDown();
        if (keys & (KEY_START | KEY_A)) {
            // 暂停功能
            while (!(keysDown() & (KEY_START | KEY_A))) {
                scanKeys();
                VBlankIntrWait();
            }
        }
    }
}
