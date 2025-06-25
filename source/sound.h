#ifndef SOUND_H
#define SOUND_H
#include <gba.h>
#include <gba_types.h>

IWRAM_CODE void inline sound_stop(void) {
    REG_DMA1CNT = 0;
    REG_TM0CNT_H = 0;
    REG_FIFO_A = 0;  // 清空 FIFO
} 

// 音频播放函数声明
void sound_init(void);
IWRAM_CODE void inline sound_play(const u8 *data) {
    // sound_stop();  // 关闭之前的播放

    // 设置 Timer0 控制采样频率
    // constexpr u16 timer_reload_val = timer_reload(SAMPLE_RATE);
    constexpr u16 timer_reload_val = 64612;
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