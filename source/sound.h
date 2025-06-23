#ifndef SOUND_H
#define SOUND_H

#include <gba_types.h>

// 音频播放函数声明
void sound_init(void);
void sound_play(const u8 *data);
void sound_stop(void);

#endif // SOUND_H 