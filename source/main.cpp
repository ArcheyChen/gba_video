// gba_video_player.cpp  v5
// Mode 3 单缓冲 + 多级IP帧码表解码 (4x4 + 4x2)

#include <gba_systemcalls.h>
#include <gba_video.h>
#include <gba_dma.h>
#include <gba_interrupt.h>
#include <gba_input.h>

#include "video_data.h"

constexpr int PIXELS_PER_FRAME = SCREEN_WIDTH * SCREEN_HEIGHT;

// EWRAM 单缓冲
EWRAM_BSS u16 ewramBuffer[PIXELS_PER_FRAME];

// 4x4块的BGR555结构体
union BGR555_4x4_Struct
{
    u16 bgr[4][4];     // y,x的访问
    u16 bgr_array[16]; // 直接访问4x4块的BGR555值
    u32 bgr_u32[4][2]; // u32访问更快速
}__attribute__((packed));

// 4x2块的BGR555结构体
union BGR555_4x2_Struct
{
    u16 bgr[2][4];     // y,x的访问
    u16 bgr_array[8];  // 直接访问4x2块的BGR555值
    u32 bgr_u32[2][2]; // u32访问更快速
}__attribute__((packed));

// 当前GOP的BGR555码表缓冲区
EWRAM_BSS BGR555_4x4_Struct bgr555_buffer_4x4[VIDEO_CODEBOOK_SIZE_4x4];  // 4x4码表
EWRAM_BSS BGR555_4x2_Struct bgr555_buffer_4x2[VIDEO_CODEBOOK_SIZE_4x2];  // 4x2码表

// 块位置到内存偏移的查找表
EWRAM_DATA static u32 block_4x4_offset_table[VIDEO_BLOCKS_4x4_PER_FRAME];
EWRAM_DATA static u32 block_4x2_offset_table[VIDEO_BLOCKS_4x2_PER_FRAME];

// 预计算4x4块位置到内存偏移的查找表
void init_block_4x4_offset_table(){
    int block_pos = 0;
    for (int y = 0; y < SCREEN_HEIGHT; y += 4) {
        for (int x = 0; x < SCREEN_WIDTH; x += 4) {
            block_4x4_offset_table[block_pos] = y * SCREEN_WIDTH + x;
            block_pos++;
        }
    }
}

// 预计算4x2块位置到内存偏移的查找表
void init_block_4x2_offset_table(){
    int block_pos = 0;
    for (int y = 0; y < SCREEN_HEIGHT; y += 2) {
        for (int x = 0; x < SCREEN_WIDTH; x += 4) {
            block_4x2_offset_table[block_pos] = y * SCREEN_WIDTH + x;
            block_pos++;
        }
    }
}

// 加载指定GOP的4x4和4x2 BGR555码表到缓冲区
IWRAM_CODE void load_bgr555_codebooks(int gop_index)
{
    // 加载4x4码表
    const u16* codebook_4x4 = (u16*)video_codebooks_4x4[gop_index];
    for(int codeword_idx = 0; codeword_idx < VIDEO_CODEBOOK_SIZE_4x4; codeword_idx++)
    {
        // 直接复制BGR555数据，每个4x4码字16个u16值
        const u16* src = codebook_4x4 + codeword_idx * 16;
        BGR555_4x4_Struct* dst = bgr555_buffer_4x4 + codeword_idx;
        
        // 使用DMA快速复制
        DMA3COPY(src, dst->bgr_array, 16 | DMA16);
    }
    
    // 加载4x2码表
    const u16* codebook_4x2 = (u16*)video_codebooks_4x2[gop_index];
    for(int codeword_idx = 0; codeword_idx < VIDEO_CODEBOOK_SIZE_4x2; codeword_idx++)
    {
        // 直接复制BGR555数据，每个4x2码字8个u16值
        const u16* src = codebook_4x2 + codeword_idx * 8;
        BGR555_4x2_Struct* dst = bgr555_buffer_4x2 + codeword_idx;
        
        // 使用DMA快速复制
        DMA3COPY(src, dst->bgr_array, 8 | DMA16);
    }
}

// 从BGR555码表缓冲区解码一个4x4块
IWRAM_CODE void decode_4x4_block_from_bgr555_buffer(u16 codeword_idx, u16* dst, int dst_stride)
{
    const BGR555_4x4_Struct &bgr555_block = bgr555_buffer_4x4[codeword_idx];

    // 直接复制BGR555数据，每行4像素
    u16* row = dst;
    for(int y = 0; y < 4; y++) {
        ((u32*)(row))[0] = bgr555_block.bgr_u32[y][0]; // 每次复制2个像素
        ((u32*)(row))[1] = bgr555_block.bgr_u32[y][1]; // 每次复制2个像素
        row += dst_stride;
    }
}

// 从BGR555码表缓冲区解码一个4x2块
IWRAM_CODE void decode_4x2_block_from_bgr555_buffer(u16 codeword_idx, u16* dst, int dst_stride)
{
    const BGR555_4x2_Struct &bgr555_block = bgr555_buffer_4x2[codeword_idx];

    // 直接复制BGR555数据，每行4像素
    u16* row = dst;
    for(int y = 0; y < 2; y++) {
        ((u32*)(row))[0] = bgr555_block.bgr_u32[y][0]; // 每次复制2个像素
        ((u32*)(row))[1] = bgr555_block.bgr_u32[y][1]; // 每次复制2个像素
        row += dst_stride;
    }
}


IWRAM_CODE void decode_i_frame_multi_level(const u16* frame_data, u16* dst)
{
    // I帧格式：[总块数, 块1编码, 块2编码, ...]
    // 其中块编码为：
    // - 4x4块：MARKER_4x4_BLOCK, 码字索引
    // - 4x2块：上半码字索引, 下半码字索引
    
    u16 total_blocks = frame_data[0];
    const u16* data = frame_data + 1;
    int data_idx = 0;
    
    for (int block_4x4_pos = 0; block_4x4_pos < VIDEO_BLOCKS_4x4_PER_FRAME; block_4x4_pos++)
    {
        u16* block_4x4_dst = dst + block_4x4_offset_table[block_4x4_pos];
        
        if (data[data_idx] == VIDEO_MARKER_4x4) {
            // 使用4x4码表
            u16 codeword_idx = data[data_idx + 1];
            decode_4x4_block_from_bgr555_buffer(codeword_idx, block_4x4_dst, SCREEN_WIDTH);
            data_idx += 2;
        } else {
            // 拆分为两个4x2块
            u16 upper_codeword = data[data_idx];
            u16 lower_codeword = data[data_idx + 1];
            
            // 解码上半部分 (前2行)
            decode_4x2_block_from_bgr555_buffer(upper_codeword, block_4x4_dst, SCREEN_WIDTH);
            
            // 解码下半部分 (后2行)
            u16* lower_dst = block_4x4_dst + 2 * SCREEN_WIDTH;
            decode_4x2_block_from_bgr555_buffer(lower_codeword, lower_dst, SCREEN_WIDTH);
            
            data_idx += 2;
        }
    }
}

IWRAM_CODE void decode_p_frame_multi_level(const u16* frame_data, u16* dst)
{
    // P帧格式：[4x4变化块数, 4x4块编码..., 4x2变化块数, 4x2块编码...]
    
    const u16* data = frame_data;
    int data_idx = 0;
    
    // 第一部分：解码4x4块
    u16 changed_4x4_count = data[data_idx++];
    
    for (int i = 0; i < changed_4x4_count; i++)
    {
        u16 block_4x4_pos = data[data_idx++];     // 4x4块位置
        u16 marker = data[data_idx++];            // 应该是MARKER_4x4_BLOCK
        u16 codeword_idx = data[data_idx++];      // 4x4码字索引
        
        u16* block_4x4_dst = dst + block_4x4_offset_table[block_4x4_pos];
        decode_4x4_block_from_bgr555_buffer(codeword_idx, block_4x4_dst, SCREEN_WIDTH);
    }
    
    // 第二部分：解码需要拆分为4x2的块
    u16 changed_4x2_blocks_count = data[data_idx++];
    
    for (int i = 0; i < changed_4x2_blocks_count; i++)
    {
        u16 block_4x4_pos = data[data_idx++];     // 原4x4块位置
        u16 upper_codeword = data[data_idx++];    // 上半4x2码字索引
        u16 lower_codeword = data[data_idx++];    // 下半4x2码字索引
        
        u16* block_4x4_dst = dst + block_4x4_offset_table[block_4x4_pos];
        
        // 解码上半部分 (前2行)
        decode_4x2_block_from_bgr555_buffer(upper_codeword, block_4x4_dst, SCREEN_WIDTH);
        
        // 解码下半部分 (后2行)
        u16* lower_dst = block_4x4_dst + 2 * SCREEN_WIDTH;
        decode_4x2_block_from_bgr555_buffer(lower_codeword, lower_dst, SCREEN_WIDTH);
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
    
    // 初始化块位置查找表
    init_block_4x4_offset_table();
    init_block_4x2_offset_table();
    
    while (1)
    {
        // 检查是否需要切换GOP
        int required_gop = frame / VIDEO_GOP_SIZE;
        if (required_gop != current_gop) {
            current_gop = required_gop;
            // 加载新GOP的多级BGR555码表到缓冲区
            load_bgr555_codebooks(current_gop);
        }
        
        // 获取当前帧的数据
        u32 frame_offset = video_frame_offsets[frame];
        const u16* frame_data = video_frame_data + frame_offset;
        u8 frame_type = video_frame_types[frame];
        
        // 根据帧类型解码
        if (frame_type == 0) {
            // I帧：完全重绘
            decode_i_frame_multi_level(frame_data, ewramBuffer);
        } else {
            // P帧：增量更新
            decode_p_frame_multi_level(frame_data, ewramBuffer);
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
