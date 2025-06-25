#include <gba.h>
#include "sound.h"
#include "audio_data.h"

#define TIMER_FREQ 16777216  // GBA 主频 = 16.777216 MHz

constexpr static inline u16 timer_reload(u32 sample_rate) {
    return (u16)(0x10000 - (TIMER_FREQ / sample_rate));
}

void sound_init(void) {
    REG_SOUNDCNT_X = SNDSTAT_ENABLE;  // 开启总音频电源

    REG_SOUNDCNT_H =
        SNDA_VOL_100 | SNDA_R_ENABLE | SNDA_L_ENABLE | SNDA_RESET_FIFO |
        SNDA_VOL_100;  // DirectSound A 左右声道、满音量
}

