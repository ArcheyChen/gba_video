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
static bool force_sound_sync = true;
static bool free_play_mode = false;

static int frame = 0;
static u16 delayed_frame = 0; // 用于延迟帧处理
#define LCD_FPS 597275
//这个是乘了10000后的FPS，这样更精确
IWRAM_CODE void isr_vbl() { 
    ++vbl; 
    acc += VIDEO_FPS;  // 使用头文件中定义的FPS

    if(acc >= LCD_FPS) {
        delayed_frame += should_copy & 1;//如果之前的拷贝操作没有完成，延迟帧计数器加1
        should_copy = true;
        acc -= LCD_FPS;
    }

    REG_IF = IRQ_VBLANK; 
}

IWRAM_CODE void doit(){
    
    // *(vu32*)(0x4000800) = 0x0E000020;//EWRAM Overclock
    // u32 wait_state = 0x00000000;
    // wait_state |= (2)<<2;
    // wait_state |= 1<<4;

    // *(vu32*)(0x4000204) = wait_state;//wait_state Overclock
    
    // 初始化音频播放   
    sound_init();
    VideoDecoder::reset_codebook();
    force_sound_sync = true;
    
    while (1)//main display loop
    {
        const unsigned char* frame_data = video_data + frame_offsets[frame];

        // 渲染帧
        VideoRenderer::render_frame(frame_data);
        
        while(!should_copy && !free_play_mode) {
            if(VideoDecoder::rgb555_codebook_preloaded){
                VBlankIntrWait();
                continue;;
            }
            if(VideoDecoder::next_i_frame == -1){
                VideoDecoder::find_next_i_frame(video_data, frame+1);
                if(!should_copy)
                    VBlankIntrWait();
                continue;
            }
            VideoDecoder::preload_codebook(video_data + frame_offsets[VideoDecoder::next_i_frame]);

        }
        should_copy = false;
        
        // 拷贝到VRAM
        DMA3COPY(VideoRenderer::ewramBuffer, VRAM, (SCREEN_WIDTH * SCREEN_HEIGHT >> 1) | DMA32);
        #ifdef I_FRAME_AUDIO_OFFSET_COUNT
        if (delayed_frame || force_sound_sync) {
            // 从I帧对应的音频偏移处重新开始播放
            const u8* audio_offset = (const u8*)audio_data + frame_audio_offsets[frame];
            sound_play(audio_offset);
            force_sound_sync = false;
            delayed_frame = 0;
        }
        #endif
        // 音画同步：如果是I帧，重新同步音频播放

        
        frame++;
        if(frame >= VIDEO_FRAME_COUNT) {
            frame = 0;
            sound_stop(); // 重置音频播放
            VideoDecoder::reset_codebook();
            // 视频循环时，音频也应该重新开始
            force_sound_sync = true; // 强制音频同步
        }
        scanKeys();
        u16 keys = keysDown();
        if (keys & (KEY_START | KEY_A)) {

            sound_stop();
            force_sound_sync = true;
            // 暂停功能
            while (!(keysDown() & (KEY_START | KEY_A))) {
                scanKeys();
                VBlankIntrWait();
            }
            free_play_mode = false; // 暂停时退出自由播放模式
        }
        if(keys & KEY_SELECT){
            free_play_mode = !free_play_mode; // 切换自由播放模式
        }
        if (keys & (KEY_RIGHT)){
            sound_stop();//防止爆音
            frame += 3 * VIDEO_FPS/10000;//快进3秒，这个VIDEO_FPS是乘了10000的
            if (frame >= VIDEO_FRAME_COUNT) {
                frame = 0;
            }
            while(!VideoDecoder::is_i_frame(video_data + frame_offsets[frame])) {
                frame++;
                if (frame >= VIDEO_FRAME_COUNT) {
                    frame = 0;
                }
            }
            VideoDecoder::reset_codebook();
            force_sound_sync = true; // 强制音频同步
            continue;
        }

        if (keys & KEY_LEFT){
            sound_stop();//防止爆音
            frame -= 3 * VIDEO_FPS/10000;//快进3秒，这个VIDEO_FPS是乘了10000的
            if (frame <= 0) {
                frame = 0;
            }
            while(!VideoDecoder::is_i_frame(video_data + frame_offsets[frame])) {
                frame--;
                if (frame <= 0) {
                    frame = 0;
                }
            }
            VideoDecoder::reset_codebook();
            force_sound_sync = true; // 强制音频同步
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