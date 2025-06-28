#include <gba.h>
#include "sound.h"
#include "audio_data.h"



void sound_init(void) {
    REG_SOUNDCNT_X = SNDSTAT_ENABLE;  // 开启总音频电源

    REG_SOUNDCNT_H =
        SNDA_VOL_100 | SNDA_R_ENABLE | SNDA_L_ENABLE | SNDA_RESET_FIFO |
        SNDA_VOL_100;  // DirectSound A 左右声道、满音量
}

