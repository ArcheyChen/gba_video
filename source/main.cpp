// gba_video_player.cpp  v2
// Mode 3 单缓冲 + YUV411 → RGB555 整数近似解码

#include <gba_systemcalls.h>
#include <gba_video.h>
#include <gba_dma.h>
#include <gba_interrupt.h>
#include <gba_input.h>

#include "video_data.h"

constexpr int PIXELS_PER_FRAME = SCREEN_WIDTH * SCREEN_HEIGHT;

// EWRAM 单缓冲
EWRAM_BSS u16 ewramBuffer[PIXELS_PER_FRAME];


IWRAM_DATA static u8 clip_table_raw[1024];

u8* clip_lookup_table = clip_table_raw + 256;

void init_clip_table(){
    for(int i=-256;i<1024-256;i++){
        u8 raw_val;
        if(i<=0)
            raw_val = 0;
        else if(i>=255)
            raw_val = 255;
        else
            raw_val = static_cast<u8>(i);
        clip_lookup_table[i] = raw_val>>3;
    }
}
// 近似整数 YUV→RGB (Y:0-255, Cb/Cr:-128..127) → 5-bit packed
inline u16 yuv_to_rgb555(u8 y   ,s16 d_r
                                ,s16 d_g
                                ,s16 d_b)
{
    s16 r = clip_lookup_table[y + d_r];                // Y + d_r
    s16 g = clip_lookup_table[y + d_g];                // Y + d_g
    s16 b = clip_lookup_table[y + d_b];                // Y + d_b

    return  (r)          |
           ((g) << 5)    |
           ((b) << 10);  // RGB555
}


IWRAM_CODE void decode_frame(const u8* src, u16* dst)
{
    for (int y = 0; y < SCREEN_HEIGHT; y += 4)
    {
        // 当前 4 行首指针
        u16* row0 = dst + y * SCREEN_WIDTH;
        u16* row1 = row0 + SCREEN_WIDTH;
        u16* row2 = row1 + SCREEN_WIDTH;
        u16* row3 = row2 + SCREEN_WIDTH;

        for (int x = 0; x < SCREEN_WIDTH; x += 4)
        {
            // 取 16×Y
            u8  Y00 = src[ 0]; u8 Y01 = src[ 1]; u8 Y02 = src[ 2]; u8 Y03 = src[ 3];
            u8  Y10 = src[ 4]; u8 Y11 = src[ 5]; u8 Y12 = src[ 6]; u8 Y13 = src[ 7];
            u8  Y20 = src[ 8]; u8 Y21 = src[ 9]; u8 Y22 = src[10]; u8 Y23 = src[11];
            u8  Y30 = src[12]; u8 Y31 = src[13]; u8 Y32 = src[14]; u8 Y33 = src[15];
            s8  Cb  = static_cast<s8>(src[16]);
            s8  Cr  = static_cast<s8>(src[17]);
            src += 18;

            s16 d_r = Cr << 1;           // 2*Cr
            s16 d_g = -(Cb >> 1) - Cr;   // -Cb/2 - Cr
            s16 d_b = Cb << 1;           // 2*Cb

            // 写 4×4 像素
            row0[x]   = yuv_to_rgb555(Y00, d_r, d_g, d_b);
            row0[x+1] = yuv_to_rgb555(Y01, d_r, d_g, d_b);
            row0[x+2] = yuv_to_rgb555(Y02, d_r, d_g, d_b);
            row0[x+3] = yuv_to_rgb555(Y03, d_r, d_g, d_b);

            row1[x]   = yuv_to_rgb555(Y10, d_r, d_g, d_b);
            row1[x+1] = yuv_to_rgb555(Y11, d_r, d_g, d_b);
            row1[x+2] = yuv_to_rgb555(Y12, d_r, d_g, d_b);
            row1[x+3] = yuv_to_rgb555(Y13, d_r, d_g, d_b);

            row2[x]   = yuv_to_rgb555(Y20, d_r, d_g, d_b);
            row2[x+1] = yuv_to_rgb555(Y21, d_r, d_g, d_b);
            row2[x+2] = yuv_to_rgb555(Y22, d_r, d_g, d_b);
            row2[x+3] = yuv_to_rgb555(Y23, d_r, d_g, d_b);

            row3[x]   = yuv_to_rgb555(Y30, d_r, d_g, d_b);
            row3[x+1] = yuv_to_rgb555(Y31, d_r, d_g, d_b);
            row3[x+2] = yuv_to_rgb555(Y32, d_r, d_g, d_b);
            row3[x+3] = yuv_to_rgb555(Y33, d_r, d_g, d_b);
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

    const unsigned char* movie = video_data;
    const int frames_total = VIDEO_FRAME_COUNT;
    const int stride       = VIDEO_BYTES_PER_FRAME;

    int frame = 0;
    const unsigned char *vdata_ptr = movie;
    init_clip_table();
    while (1)
    {
        // const unsigned char* src = movie + frame * stride;

        decode_frame(vdata_ptr, ewramBuffer);

        VBlankIntrWait();
        DMA3COPY(ewramBuffer, VRAM, PIXELS_PER_FRAME | DMA16);

        frame++;
        if(frame >= frames_total) // 循环播放
        {
            frame = 0; // 重置帧计数
            vdata_ptr = movie; // 重置指针
        }
        else
        {
            vdata_ptr += stride; // 移动到下一帧
        }

        // 想加暂停 / 退出可自行检测按键
    }
}
