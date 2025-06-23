#include <gba_video.h>
#include <gba_dma.h>
#include <cstring>
#include "video_renderer.h"
#include "video_decoder.h"

// 静态成员变量定义
EWRAM_BSS u16 VideoRenderer::ewramBuffer[SCREEN_WIDTH * SCREEN_HEIGHT];

void VideoRenderer::init() {
    // 初始化视频解码器
    VideoDecoder::init();
    
    // 清除缓冲区
    clear_buffer();
    
    
    // 初始显示
    DMA3COPY(ewramBuffer, VRAM, (SCREEN_WIDTH * SCREEN_HEIGHT >> 1) | DMA32);
}

int VideoRenderer::render_frame(const u8* frame_data) {
    // 检查是否为I帧
    bool is_i_frame = VideoDecoder::is_i_frame(frame_data);
    
    // 解码帧到缓冲区
    VideoDecoder::decode_frame(frame_data, ewramBuffer);
    
    // 如果是I帧，增加计数器并返回I帧编号
    if (is_i_frame) {
        return 1;
    } else {
        return -1;  // P帧
    }
}

void VideoRenderer::clear_buffer() {
    memset(ewramBuffer, 0, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(u16));
} 