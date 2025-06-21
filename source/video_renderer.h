#ifndef VIDEO_RENDERER_H
#define VIDEO_RENDERER_H

#include <gba_types.h>

// 视频渲染器类
class VideoRenderer {
private:
    // EWRAM 单缓冲
    static u16 ewramBuffer[SCREEN_WIDTH * SCREEN_HEIGHT];
    
public:
    // 初始化函数
    static void init();
    
    // 渲染函数
    static void render_frame(const u8* frame_data);
    
    // 获取缓冲区指针
    static u16* get_buffer() { return ewramBuffer; }
    
    // 清除缓冲区
    static void clear_buffer();
};

#endif // VIDEO_RENDERER_H 