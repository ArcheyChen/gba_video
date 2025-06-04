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

// 近似整数 YUV→RGB (Y:0-255, Cb/Cr:-128..127) → 5-bit packed
inline u16 yuv_to_rgb555(u8 y   ,s16 d_r
                                ,s16 d_g
                                ,s16 d_b)
{
    s16 r = y + d_r;                // Y + d_r
    s16 g = y + d_g;                // Y + d_g
    s16 b = y + d_b;                // Y + d_b

    // // // 裁剪至 0-255
    if (r < 0) r = 0; else if (r > 255) r = 255;
    if (g < 0) g = 0; else if (g > 255) g = 255;
    if (b < 0) b = 0; else if (b > 255) b = 255;


    return  (r >> 3)          |
           ((g >> 3) << 5)    |
           ((b >> 3) << 10);  // RGB555
}


IWRAM_CODE void decode_frame(const unsigned char* src, u16* dst)
{
    for (int y = 0; y < SCREEN_HEIGHT; y += 2)          // 每次处理 2 行
    {
        u16* row0 = dst + y * SCREEN_WIDTH;             // 当前行指针
        u16* row1 = row0 + SCREEN_WIDTH;                // 下一行指针

        for (int x = 0; x < SCREEN_WIDTH; x += 2)       // 2×2 块
        {
            u8 Y00 = src[0];  // Y00
            u8 Y01 = src[1];  // Y01
            u8 Y10 = src[2];  // Y10
            u8 Y11 = src[3];  // Y11
            s8 Cb  = (s8)(src[4]); // Cb
            s8 Cr  = (s8)(src[5]); // Cr

            s16 d_r = (Cr << 1); // Cr * 2;
            s16 d_g = -(Cb >> 1) - Cr; // -Cb/2 - Cr;
            s16 d_b = (Cb << 1); // Cb * 2;
            // 写入 4 像素
            *row0++ = yuv_to_rgb555(Y00, d_r, d_g, d_b);
            *row0++ = yuv_to_rgb555(Y01, d_r, d_g, d_b);
            *row1++ = yuv_to_rgb555(Y10, d_r, d_g, d_b);
            *row1++ = yuv_to_rgb555(Y11, d_r, d_g, d_b);

            src += 6; // 移动到下一个 2x2 块
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
    while (1)
    {
        const unsigned char* src = movie + frame * stride;

        decode_frame(src, ewramBuffer);

        VBlankIntrWait();
        DMA3COPY(ewramBuffer, VRAM, PIXELS_PER_FRAME | DMA16);

        frame = (frame + 1) % frames_total;
        if(frame == 0)
        {
            vdata_ptr = movie; // 重置指针
        }
        else
        {
            vdata_ptr += stride; // 移动到下一帧
        }

        // 想加暂停 / 退出可自行检测按键
    }
}
