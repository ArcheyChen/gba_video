// gba_video_player.cpp  v4
// Mode 3 单缓冲 + IP帧码表解码

#include <gba_systemcalls.h>
#include <gba_video.h>
#include <gba_dma.h>
#include <gba_interrupt.h>
#include <gba_input.h>

#include "video_data.h"

constexpr int PIXELS_PER_FRAME = SCREEN_WIDTH * SCREEN_HEIGHT;

struct YUV_Struct{
    s8 cb;      // Cb 色度分量 (-128~127)
    s8 cr;      // Cr 色度分量 (-128~127)
    u8 y[8];    // Y 亮度分量 (0~255)
} __attribute__((packed));
// EWRAM 单缓冲和RGB555码表
EWRAM_BSS u16 ewramBuffer[PIXELS_PER_FRAME];

union RGB555_Struct
{
    u16 rgb[2][4];//y,x的访问，因为内存中是这么排布的
    u16 rgb_array[8];  // 直接访问4x2块的RGB555值
    u32 rgb_u32[2][2]; //u32访问更快速
}__attribute__((packed));

EWRAM_BSS RGB555_Struct rgb555_codebook[VIDEO_CODEBOOK_SIZE];  // 当前GOP的预解码RGB555码表，每个码字8个像素

// 裁切查找表
IWRAM_DATA static u8 clip_table_raw[1024];
u8* clip_lookup_table = clip_table_raw + 256;

void init_clip_table(){
    for(int i = -256; i < 1024 - 256; i++){
        u8 raw_val;
        if(i <= 0)
            raw_val = 0;
        else if(i >= 255)
            raw_val = 255;
        else
            raw_val = static_cast<u8>(i);
        clip_lookup_table[i] = raw_val >> 3;
    }
}

// 预解码指定GOP的YUV码表到RGB555格式
IWRAM_CODE void precompute_rgb555_codebook(int gop_index)
{
    const YUV_Struct* codebook = (YUV_Struct*)video_codebooks[gop_index];
    
    for(int codeword_idx = 0; codeword_idx < VIDEO_CODEBOOK_SIZE; codeword_idx++)
    {
        YUV_Struct yuv_data = codebook[codeword_idx];
        
        // 计算色度偏移
        s16 cr = yuv_data.cr;
        s16 cb = yuv_data.cb;
        s16 d_r = cr << 1;           // 2*Cr
        s16 d_g = -(cb >> 1) - cr;   // -Cb/2 - Cr
        s16 d_b = cb << 1;           // 2*Cb
        
        // 预计算4x2块的所有RGB555值
        RGB555_Struct* rgb555_block = rgb555_codebook + codeword_idx;
        for(int i = 0; i < 8; i++)
        {
            u8 luma = yuv_data.y[i];
            
            s16 r = clip_lookup_table[luma + d_r];
            s16 g = clip_lookup_table[luma + d_g]; 
            s16 b = clip_lookup_table[luma + d_b];

            rgb555_block->rgb_array[i] = (r) | ((g) << 5) | ((b) << 10);  // RGB555
        }
    }
}
// 从预解码的RGB555码表中解码一个4x2块
IWRAM_CODE void decode_block_from_rgb555_codebook(u16 codeword_idx, u16* dst, int dst_stride)
{
    const RGB555_Struct &rgb555_block = rgb555_codebook[codeword_idx];

    // 直接复制预解码的RGB555数据
    u16* row = dst;
    for(int y = 0; y < 2; y++) {
        ((u32*)(row))[0] = rgb555_block.rgb_u32[y][0]; // 每次复制2个像素
        ((u32*)(row))[1] = rgb555_block.rgb_u32[y][1]; // 每次复制2个像素
        row += dst_stride;
    }
}


IWRAM_CODE void decode_i_frame(const u16* frame_data, u16* dst)
{
    // I帧格式：[块数量, 码字1, 码字2, ..., 码字N]
    u16 block_count = frame_data[0];
    const u16* indices = frame_data + 1;
    
    int block_idx = 0;
    for (int y = 0; y < SCREEN_HEIGHT; y += 2)
    {
        for (int x = 0; x < SCREEN_WIDTH; x += 4)
        {
            // 获取当前块的码字索引
            u16 codeword_idx = indices[block_idx++];
            
            // 解码4x2块到目标位置
            u16* block_dst = dst + y * SCREEN_WIDTH + x;
            decode_block_from_rgb555_codebook(codeword_idx, block_dst, SCREEN_WIDTH);
        }
    }
}

IWRAM_CODE void decode_p_frame(const u16* frame_data, u16* dst)
{
    // P帧格式：[块数量, 位置1, 码字1, 位置2, 码字2, ...]
    u16 changed_block_count = frame_data[0];
    const u16* data = frame_data + 1;
    
    for (int i = 0; i < changed_block_count; i++)
    {
        u16 block_pos = data[i * 2];     // 块位置（线性索引）
        u16 codeword_idx = data[i * 2 + 1]; // 码字索引
        
        // 将线性块位置转换为2D坐标
        int block_y = (block_pos / (SCREEN_WIDTH / 4)) * 2;  // 每行60个块，每块高度2
        int block_x = (block_pos % (SCREEN_WIDTH / 4)) * 4;  // 每块宽度4
        
        // 解码4x2块到目标位置
        u16* block_dst = dst + block_y * SCREEN_WIDTH + block_x;
        decode_block_from_rgb555_codebook(codeword_idx, block_dst, SCREEN_WIDTH);
    }
}

static volatile u32 vbl = 0;
void isr_vbl() { ++vbl; REG_IF = IRQ_VBLANK; }

int main()
{
    REG_DISPCNT = MODE_3 | BG2_ENABLE;

    irqInit();
    irqSet(IRQ_VBLANK, isr_vbl);
    irqEnable(IRQ_VBLANK);

    const int frames_total = VIDEO_FRAME_COUNT;
    int frame = 0;
    int current_gop = -1;  // 当前GOP索引
    
    init_clip_table();
    
    while (1)
    {
        // 检查是否需要切换GOP
        int required_gop = frame / VIDEO_GOP_SIZE;
        if (required_gop != current_gop) {
            current_gop = required_gop;
            // 预解码新GOP的RGB555码表
            precompute_rgb555_codebook(current_gop);
        }
        
        // 获取当前帧的数据
        u32 frame_offset = video_frame_offsets[frame];
        const u16* frame_data = video_frame_data + frame_offset;
        u8 frame_type = video_frame_types[frame];
        
        // 根据帧类型解码
        if (frame_type == 0) {
            // I帧：完全重绘
            decode_i_frame(frame_data, ewramBuffer);
        } else {
            // P帧：增量更新
            decode_p_frame(frame_data, ewramBuffer);
        }

        // 复制到VRAM
        DMA3COPY(ewramBuffer, VRAM, PIXELS_PER_FRAME | DMA16);

        frame++;
        if(frame >= frames_total) // 循环播放
        {
            frame = 0; // 重置帧计数
            current_gop = -1; // 强制重新加载GOP
        }

        // 暂停/退出控制
        scanKeys();
        u16 keys = keysDown();
        if (keys & (KEY_START | KEY_A)) {
            // 暂停功能
            while (!(keysDown() & (KEY_START | KEY_A))) {
                scanKeys();
                VBlankIntrWait();
            }
        }
        
        VBlankIntrWait();
    }
}
