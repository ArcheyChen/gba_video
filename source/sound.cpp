#include <gba.h>
#include "sound.h"

#define TIMER_FREQ 16777216  // GBA 主频 = 16.777216 MHz

static inline u16 timer_reload(u32 sample_rate) {
    return (u16)(0x10000 - (TIMER_FREQ / sample_rate));
}

void sound_init(void) {
    REG_SOUNDCNT_X = SNDSTAT_ENABLE;  // 开启总音频电源

    REG_SOUNDCNT_H =
        SNDA_VOL_100 | SNDA_R_ENABLE | SNDA_L_ENABLE | SNDA_RESET_FIFO |
        SNDA_VOL_100;  // DirectSound A 左右声道、满音量
}

void sound_play(const u8 *data, u32 sample_rate, bool loop) {
    sound_stop();  // 关闭之前的播放

    // 设置 Timer0 控制采样频率
    REG_TM0CNT_L = timer_reload(sample_rate);
    REG_TM0CNT_H = TIMER_START;

    DMA1COPY(data, &REG_FIFO_A, 
        DMA_DST_FIXED |
        DMA_SRC_INC |
        DMA_REPEAT |
        DMA32 |
        DMA_SPECIAL |
        DMA_ENABLE);

    if (!loop)
        REG_DMA1CNT &= ~DMA_REPEAT;  // 只播一次
}

void sound_stop(void) {
    REG_DMA1CNT = 0;
    REG_TM0CNT_H = 0;
    REG_FIFO_A = 0;  // 清空 FIFO
} 