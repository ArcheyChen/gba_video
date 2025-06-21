// gba_video_player.cpp  v8
// Mode 3 单缓冲 + YUV9 → RGB555 + 条带帧间差分解码 + 统一码本向量量化 + 单条带模式

#include <gba_systemcalls.h>
#include <gba_video.h>
#include <gba_dma.h>
#include <gba_interrupt.h>
#include <gba_input.h>
#include <cstring>

#include "video_data.h"
#include "video_renderer.h"
#include "video_decoder.h"
#include "audio_data.h"  // 音频数据头文件
#include "sound.h"       // 音频播放头文件

static volatile u32 vbl = 0;
static volatile u32 acc = 0;
static volatile bool should_copy = false;
static volatile bool audio_playing = false;  // 音频播放状态
#define LCD_FPS 597275
//这个是乘了10000后的FPS，这样更精确
IWRAM_CODE void isr_vbl() { 
    ++vbl; 
    acc += VIDEO_FPS;  // 使用头文件中定义的FPS

    if(acc >= LCD_FPS) {
        should_copy = true;
        acc -= LCD_FPS;
    }
    REG_IF = IRQ_VBLANK; 
}

IWRAM_CODE void doit(){
    int frame = 0;
    
    // 初始化音频播放
    if (audio_data_len > 0) {
        sound_init();
        sound_play((const u8*)audio_data, audio_data_len, SAMPLE_RATE, true);  // 循环播放
        audio_playing = true;
    }
    
    while (1)
    {
        const unsigned char* frame_data = video_data + frame_offsets[frame];
        
        // VBlankIntrWait(); // 注释掉以提高性能，让DMA自动等待

        
        // 渲染帧
        int i_frame_index = VideoRenderer::render_frame(frame_data);
        
        while(!should_copy) {
            VideoDecoder::preload_codebook(frame_data);
            // 预加载码本，以消耗掉空闲时间
            // 这个函数里面如果没事做，会等待VBlank，因此不用担心跑满CPU
        }
        should_copy = false;
        
        // 拷贝到VRAM
        DMA3COPY(VideoRenderer::ewramBuffer, VRAM, (SCREEN_WIDTH * SCREEN_HEIGHT >> 1) | DMA32);
        
        // 音画同步：如果是I帧，重新同步音频播放
        #ifdef I_FRAME_AUDIO_OFFSET_COUNT
        if (i_frame_index >= 0 && audio_playing && i_frame_index < I_FRAME_AUDIO_OFFSET_COUNT) {
            // 停止当前音频播放
            sound_stop();
            
            // 从I帧对应的音频偏移处重新开始播放
            const u8* audio_offset = (const u8*)audio_data + i_frame_audio_offsets[i_frame_index];
            u32 remaining_audio_size = audio_data_len - i_frame_audio_offsets[i_frame_index];
            
            if (remaining_audio_size > 0) {
                sound_play(audio_offset, remaining_audio_size, SAMPLE_RATE, true);
            } else {
                // 如果剩余音频不足，重新开始播放
                sound_play((const u8*)audio_data, audio_data_len, SAMPLE_RATE, true);
            }
        }
        #endif
        
        frame++;
        if(frame >= VIDEO_FRAME_COUNT) {
            frame = 0;
            // 重置I帧计数器
            VideoRenderer::reset_i_frame_counter();
            // 视频循环时，音频也应该重新开始
            if (audio_playing) {
                sound_stop();
                sound_play((const u8*)audio_data, audio_data_len, SAMPLE_RATE, true);
            }
        }
        
    }
}

int main()
{
    REG_DISPCNT = MODE_3 | BG2_ENABLE;

    irqInit();
    irqSet(IRQ_VBLANK, isr_vbl);
    irqEnable(IRQ_VBLANK);

    // 初始化视频渲染器
    VideoRenderer::init();

    doit();
}