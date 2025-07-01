// gba_video_player.cpp  v4
// Mode 3 单缓冲 + IP帧码表解码

#include <gba_systemcalls.h>
#include <gba_video.h>
#include <gba_dma.h>
#include <gba_interrupt.h>
#include <gba_input.h>

#include "video_data.h"

constexpr int PIXELS_PER_FRAME = SCREEN_WIDTH * SCREEN_HEIGHT;

// EWRAM 单缓冲和BGR555码表
EWRAM_BSS u16 ewramBuffer[PIXELS_PER_FRAME];

union BGR555_Struct
{
    u16 bgr[2][4];     // y,x的访问，因为内存中是这么排布的
    u16 bgr_array[8];  // 直接访问4x2块的BGR555值
    u32 bgr_u32[2][2]; // u32访问更快速
}__attribute__((packed));

EWRAM_BSS BGR555_Struct bgr555_buffer[VIDEO_CODEBOOK_SIZE];  // 当前GOP的BGR555码表缓冲区，每个码字8个像素

// 块位置到内存偏移的查找表
EWRAM_DATA static u32 block_offset_table[VIDEO_BLOCKS_PER_FRAME];

// 预计算块位置到内存偏移的查找表
void init_block_offset_table(){
    int block_pos = 0;
    for (int y = 0; y < SCREEN_HEIGHT; y += 2) {
        for (int x = 0; x < SCREEN_WIDTH; x += 4) {
            block_offset_table[block_pos] = y * SCREEN_WIDTH + x;
            block_pos++;
        }
    }
}

// 加载指定GOP的BGR555码表到缓冲区
IWRAM_CODE void load_bgr555_codebook(int gop_index)
{
    const u16* codebook = (u16*)video_codebooks[gop_index];
    
    for(int codeword_idx = 0; codeword_idx < VIDEO_CODEBOOK_SIZE; codeword_idx++)
    {
        // 直接复制BGR555数据，每个码字8个u16值
        const u16* src = codebook + codeword_idx * 8;
        BGR555_Struct* dst = bgr555_buffer + codeword_idx;
        
        // 使用DMA快速复制
        DMA3COPY(src, dst->bgr_array, 8 | DMA16);
    }
}

// 从BGR555码表缓冲区解码一个4x2块
IWRAM_CODE void decode_block_from_bgr555_buffer(u16 codeword_idx, u16* dst, int dst_stride)
{
    const BGR555_Struct &bgr555_block = bgr555_buffer[codeword_idx];

    // 直接复制BGR555数据，每行4像素
    u16* row = dst;
    for(int y = 0; y < 2; y++) {
        ((u32*)(row))[0] = bgr555_block.bgr_u32[y][0]; // 每次复制2个像素
        ((u32*)(row))[1] = bgr555_block.bgr_u32[y][1]; // 每次复制2个像素
        row += dst_stride;
    }
}


IWRAM_CODE void decode_i_frame(const u16* frame_data, u16* dst)
{
    // I帧格式：[块数量, 码字1, 码字2, ..., 码字N]
    u16 block_count = frame_data[0];
    const u16* indices = frame_data + 1;
    
    for (int block_pos = 0; block_pos < VIDEO_BLOCKS_PER_FRAME; block_pos++)
    {
        // 获取当前块的码字索引
        u16 codeword_idx = indices[block_pos];
        
        // 使用查找表直接获取目标地址
        u16* block_dst = dst + block_offset_table[block_pos];
        decode_block_from_bgr555_buffer(codeword_idx, block_dst, SCREEN_WIDTH);
    }
}

IWRAM_CODE void decode_p_frame(const u16* frame_data, u16* dst)
{
    // P帧格式：[块数量, 位置1, 码字1, 位置2, 码字2, ...]
    u16 changed_block_count = frame_data[0];
    const u16* data = frame_data + 1;
    
    for (int i = 0; i < changed_block_count; i++)
    {
        u16 block_pos = data[i * 2];         // 块位置（线性索引）
        u16 codeword_idx = data[i * 2 + 1];  // 码字索引
        
        // 使用查找表直接获取目标地址，避免乘除法
        u16* block_dst = dst + block_offset_table[block_pos];
        decode_block_from_bgr555_buffer(codeword_idx, block_dst, SCREEN_WIDTH);
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
    
    init_block_offset_table();  // 初始化块位置查找表
    
    while (1)
    {
        // 检查是否需要切换GOP
        int required_gop = frame / VIDEO_GOP_SIZE;
        if (required_gop != current_gop) {
            current_gop = required_gop;
            // 加载新GOP的BGR555码表到缓冲区
            load_bgr555_codebook(current_gop);
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
