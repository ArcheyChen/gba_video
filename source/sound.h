#ifndef SOUND_H
#define SOUND_H
#include <gba.h>
#include <gba_types.h>
#include "audio_data.h"

IWRAM_CODE void inline sound_stop(void) {
    REG_DMA1CNT = 0;
    REG_TM0CNT_H = 0;
    REG_FIFO_A = 0;  // 清空 FIFO
} 
#define TIMER_FREQ 16777216  // GBA 主频 = 16.777216 MHz

constexpr static inline u16 timer_reload(u32 sample_rate) {
    return (u16)(0x10000 - (TIMER_FREQ / sample_rate));
}


// 音频播放函数声明
void sound_init(void);
IWRAM_CODE void inline sound_play(const u8 *data) {
    sound_stop();  // 关闭之前的播放

    // 设置 Timer0 控制采样频率
    constexpr u16 timer_reload_val = timer_reload(SAMPLE_RATE);
    REG_TM0CNT_L = timer_reload_val;
    REG_TM0CNT_H = TIMER_START;

    DMA1COPY(data, &REG_FIFO_A, 
        DMA_DST_FIXED |
        DMA_SRC_INC |
        DMA_REPEAT |
        DMA32 |
        DMA_SPECIAL |
        DMA_ENABLE);
}

#endif // SOUND_H 