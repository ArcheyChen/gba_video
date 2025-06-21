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

static volatile u32 vbl = 0;
static volatile u32 acc = 0;
static volatile bool should_copy = false;
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
    
    while (1)
    {
        const unsigned char* frame_data = video_data + frame_offsets[frame];
        
        // VBlankIntrWait(); // 注释掉以提高性能，让DMA自动等待
        while(!should_copy) {
            VideoDecoder::preload_codebook(frame_data);
            // 预加载码本，以消耗掉空闲时间
            // 这个函数里面如果没事做，会等待VBlank，因此不用担心跑满CPU
        }
        should_copy = false;
        
        // 渲染帧
        VideoRenderer::render_frame(frame_data);

        frame++;
        if(frame >= VIDEO_FRAME_COUNT) {
            frame = 0;
        }
        scanKeys();
        u16 keys = keysDown();
        
        if (keys & KEY_START) {
            // 暂停功能
            while (!(keysDown() & KEY_START)) {
                scanKeys();
                VBlankIntrWait();
            }
        }
        
        if (keys & KEY_A) {
            // 快进：跳过5帧
            frame += 5;
            if (frame >= VIDEO_FRAME_COUNT) frame = 0;
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