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

// 8x4块的BGR555结构体
union BGR555_8x4_Struct
{
    u16 bgr[4][8];     // y,x的访问
    u16 bgr_array[32]; // 直接访问8x4块的BGR555值
    u32 bgr_u32[4][4]; // u32访问更快速
}__attribute__((packed));

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
EWRAM_BSS BGR555_8x4_Struct bgr555_buffer_8x4[VIDEO_CODEBOOK_SIZE_8x4];  // 8x4码表
EWRAM_BSS BGR555_4x4_Struct bgr555_buffer_4x4[VIDEO_CODEBOOK_SIZE_4x4];  // 4x4码表
EWRAM_BSS BGR555_4x2_Struct bgr555_buffer_4x2[VIDEO_CODEBOOK_SIZE_4x2];  // 4x2码表

// 块位置到内存偏移的查找表
EWRAM_DATA static u32 block_8x4_offset_table[VIDEO_BLOCKS_8x4_PER_FRAME];
EWRAM_DATA static u32 block_4x4_offset_table[VIDEO_BLOCKS_4x4_PER_FRAME];
EWRAM_DATA static u32 block_4x2_offset_table[VIDEO_BLOCKS_4x2_PER_FRAME];

// 预计算8x4块位置到内存偏移的查找表
void init_block_8x4_offset_table(){
    int block_pos = 0;
    for (int y = 0; y < SCREEN_HEIGHT; y += 4) {
        for (int x = 0; x < SCREEN_WIDTH; x += 8) {
            block_8x4_offset_table[block_pos] = y * SCREEN_WIDTH + x;
            block_pos++;
        }
    }
}

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

// 加载指定GOP的8x4、4x4和4x2 BGR555码表到缓冲区
IWRAM_CODE void load_bgr555_codebooks(int gop_index)
{
    // 加载8x4码表
    const u16* codebook_8x4 = (u16*)video_codebooks_8x4[gop_index];
    for(int codeword_idx = 0; codeword_idx < VIDEO_CODEBOOK_SIZE_8x4; codeword_idx++)
    {
        // 直接复制BGR555数据，每个8x4码字32个u16值
        const u16* src = codebook_8x4 + codeword_idx * 32;
        BGR555_8x4_Struct* dst = bgr555_buffer_8x4 + codeword_idx;
        
        // 使用DMA快速复制
        DMA3COPY(src, dst->bgr_array, 32 | DMA16);
    }
    
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

// 从BGR555码表缓冲区解码一个8x4块
IWRAM_CODE void decode_8x4_block_from_bgr555_buffer(u16 codeword_idx, u16* dst, int dst_stride)
{
    const BGR555_8x4_Struct &bgr555_block = bgr555_buffer_8x4[codeword_idx];

    // 直接复制BGR555数据，每行8像素
    u16* row = dst;
    for(int y = 0; y < 4; y++) {
        ((u32*)(row))[0] = bgr555_block.bgr_u32[y][0]; // 每次复制2个像素
        ((u32*)(row))[1] = bgr555_block.bgr_u32[y][1]; // 每次复制2个像素
        ((u32*)(row))[2] = bgr555_block.bgr_u32[y][2]; // 每次复制2个像素
        ((u32*)(row))[3] = bgr555_block.bgr_u32[y][3]; // 每次复制2个像素
        row += dst_stride;
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


// 递归解码4x2块编码数据（最小块，不再分裂）
IWRAM_CODE int decode_4x2_recursive(const u16* data, int data_idx, u16* dst, int dst_stride)
{
    // 4x2是最小块，不会再分裂，直接使用4x2码表解码
    u16 codeword_idx = data[data_idx];
    decode_4x2_block_from_bgr555_buffer(codeword_idx, dst, dst_stride);
    return 1; // 消耗1个u16
}

// 递归解码4x4块编码数据
IWRAM_CODE int decode_4x4_recursive(const u16* data, int data_idx, u16* dst, int dst_stride)
{
    u16 first_element = data[data_idx];
    
    if (first_element == VIDEO_MARKER_4x4) {
        // 4x4分裂：需要拆分为两个4x2块
        data_idx++; // 跳过MARKER_4x4
        
        // 递归解码上半4x2块 (前2行)
        int upper_consumed = decode_4x2_recursive(data, data_idx, dst, dst_stride);
        data_idx += upper_consumed;
        
        // 递归解码下半4x2块 (后2行)
        u16* lower_dst = dst + 2 * dst_stride;
        int lower_consumed = decode_4x2_recursive(data, data_idx, lower_dst, dst_stride);
        data_idx += lower_consumed;
        
        return 1 + upper_consumed + lower_consumed; // MARKER + 上编码 + 下编码
    } else {
        // 直接4x4码字：使用4x4码表
        u16 codeword_idx = first_element;
        decode_4x4_block_from_bgr555_buffer(codeword_idx, dst, dst_stride);
        return 1; // 消耗1个u16
    }
}

// 递归解码8x4块编码数据
IWRAM_CODE int decode_8x4_recursive(const u16* data, int data_idx, u16* dst, int dst_stride)
{
    u16 first_element = data[data_idx];
    
    if (first_element == VIDEO_MARKER_8x4) {
        // 8x4分裂：需要拆分为两个4x4块
        data_idx++; // 跳过MARKER_8x4
        
        // 递归解码左半4x4块
        int left_consumed = decode_4x4_recursive(data, data_idx, dst, dst_stride);
        data_idx += left_consumed;
        
        // 递归解码右半4x4块 (偏移4列)
        u16* right_dst = dst + 4;
        int right_consumed = decode_4x4_recursive(data, data_idx, right_dst, dst_stride);
        data_idx += right_consumed;
        
        return 1 + left_consumed + right_consumed; // MARKER + 左编码 + 右编码
    } else {
        // 直接8x4码字：使用8x4码表
        u16 codeword_idx = first_element;
        decode_8x4_block_from_bgr555_buffer(codeword_idx, dst, dst_stride);
        return 1; // 消耗1个u16
    }
}

IWRAM_CODE void decode_i_frame_multi_level(const u16* frame_data, u16* dst)
{
    // I帧新递归格式：[总块数, 块1编码..., 块2编码..., ...]
    // 其中每个块编码是递归的：
    // - 8x4块：8x4码字索引 (直接是索引)
    // - 分裂为4x4块：MARKER_8x4, 左半4x4编码..., 右半4x4编码...
    // - 分裂为4x2块：MARKER_4x4, 上半4x2码字索引, 下半4x2码字索引
    
    // u16 total_blocks = frame_data[0]; // 暂时未使用
    int data_idx = 1;
    
    for (int block_8x4_pos = 0; block_8x4_pos < VIDEO_BLOCKS_8x4_PER_FRAME; block_8x4_pos++)
    {
        u16* block_8x4_dst = dst + block_8x4_offset_table[block_8x4_pos];
        
        // 递归解码当前8x4块
        int consumed = decode_8x4_recursive(frame_data, data_idx, block_8x4_dst, SCREEN_WIDTH);
        data_idx += consumed;
    }
}

IWRAM_CODE void decode_p_frame_multi_level(const u16* frame_data, u16* dst)
{
    // P帧新递归格式：[变化块数, 位置1, 块1编码..., 位置2, 块2编码..., ...]
    // 其中每个块编码是递归的，与I帧相同
    
    int data_idx = 0;
    
    // 读取变化块数
    u16 changed_count = frame_data[data_idx++];
    
    for (int i = 0; i < changed_count; i++)
    {
        u16 block_8x4_pos = frame_data[data_idx++];     // 8x4块位置
        u16* block_8x4_dst = dst + block_8x4_offset_table[block_8x4_pos];
        
        // 递归解码当前8x4块
        int consumed = decode_8x4_recursive(frame_data, data_idx, block_8x4_dst, SCREEN_WIDTH);
        data_idx += consumed;
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
    init_block_8x4_offset_table();
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
