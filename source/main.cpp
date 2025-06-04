// gba_video_player.cpp
// 简单的 Mode3 单缓冲视频播放器（EWRAM 缓冲 + DMA 复制到 VRAM）

#include <gba_systemcalls.h>
#include <gba_video.h>
#include <gba_dma.h>
#include <gba_interrupt.h>
#include <gba_input.h>

#include "video_data.h"      // 由 python 脚本生成

// 方便：一些常量
constexpr int PIXELS_PER_FRAME = SCREEN_WIDTH * SCREEN_HEIGHT;


EWRAM_BSS u16 ewramBuffer[PIXELS_PER_FRAME];

static volatile u32 vbl = 0;// 垂直同步计数器，以后显示FPS可能有用
void isr_vbl(void){ ++vbl; REG_IF=IRQ_VBLANK; }
int main()
{
    // 设置 Mode 3 单缓冲
    REG_DISPCNT = MODE_3 | BG2_ENABLE;

    irqInit(); irqSet(IRQ_VBLANK, isr_vbl); irqEnable(IRQ_VBLANK);


    const u16* movie = video_data;               // ROM 中的数据
    const int  frames_total = VIDEO_FRAME_COUNT; // 由头文件给出

    int frame = 0;
    while (1)
    {
        DMA3COPY(movie + frame * PIXELS_PER_FRAME,
                ewramBuffer,
                PIXELS_PER_FRAME | DMA16 );

        //垂直同步（VSync）等待，确保在 VBlank 时复制数据
        VBlankIntrWait();           // 等待垂直同步中断（VBlank）
        DMA3COPY(ewramBuffer,
                VRAM,
                PIXELS_PER_FRAME | DMA16 );

        // 循环播放
        frame = (frame + 1) % frames_total;
    }
}
