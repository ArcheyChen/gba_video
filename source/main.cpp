// gba_video_player.cpp  v6
// Mode 3 单缓冲 + 四级IP帧码表解码 (8x8 + 8x4 + 4x4 + 4x2)

#include <gba_systemcalls.h>
#include <gba_video.h>
#include <gba_dma.h>
#include <gba_interrupt.h>
#include <gba_input.h>
#include <string.h>

#include "video_data.h"

constexpr int PIXELS_PER_FRAME = SCREEN_WIDTH * SCREEN_HEIGHT;

IWRAM_CODE int decode_8x4_recursive(const u8* data, int data_idx, u16* dst, int dst_stride);

// 从 u8 数组中读取 u32 值（小端序）
IWRAM_CODE inline u32 read_u32_from_u8(const u8* data, int idx) {
    return data[idx] | (data[idx + 1] << 8) | (data[idx + 2] << 16) | (data[idx + 3] << 24);
}

// 从 u8 数组中读取 u16 值（小端序）
IWRAM_CODE inline u16 read_u16_from_u8(const u8* data, int idx) {
    return data[idx] | (data[idx + 1] << 8);
}
// EWRAM 单缓冲
EWRAM_BSS u16 ewramBuffer[PIXELS_PER_FRAME];

// 8x8块的BGR555结构体
union BGR555_8x8_Struct
{
    u16 bgr[8][8];     // y,x的访问
    u16 bgr_array[64]; // 直接访问8x8块的BGR555值
    u32 bgr_u32[8][4]; // u32访问更快速
}__attribute__((packed));

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
EWRAM_BSS BGR555_8x8_Struct bgr555_buffer_8x8[VIDEO_CODEBOOK_SIZE_8x8];  // 8x8码表
EWRAM_BSS BGR555_8x4_Struct bgr555_buffer_8x4[VIDEO_CODEBOOK_SIZE_8x4];  // 8x4码表
EWRAM_BSS BGR555_4x4_Struct bgr555_buffer_4x4[VIDEO_CODEBOOK_SIZE_4x4];  // 4x4码表
IWRAM_DATA BGR555_4x2_Struct bgr555_buffer_4x2[VIDEO_CODEBOOK_SIZE_4x2];  // 4x2码表

// 块位置到内存偏移的查找表
EWRAM_DATA static u16 block_8x8_offset_table[VIDEO_BLOCKS_8x8_PER_FRAME];
EWRAM_DATA static u16 block_8x4_offset_table[VIDEO_BLOCKS_8x4_PER_FRAME];
EWRAM_DATA static u16 block_4x4_offset_table[VIDEO_BLOCKS_4x4_PER_FRAME];
EWRAM_DATA static u16 block_4x2_offset_table[VIDEO_BLOCKS_4x2_PER_FRAME];

// 预计算8x8块位置到内存偏移的查找表
void init_block_8x8_offset_table(){
    int block_pos = 0;
    for (int y = 0; y < SCREEN_HEIGHT; y += 8) {
        for (int x = 0; x < SCREEN_WIDTH; x += 8) {
            block_8x8_offset_table[block_pos] = y * SCREEN_WIDTH + x;
            block_pos++;
        }
    }
}

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

// 加载指定GOP的8x8、8x4、4x4和4x2 BGR555码表到缓冲区
IWRAM_CODE void load_bgr555_codebooks(int gop_index)
{
    // 加载8x8码表
    const u16* codebook_8x8 = (u16*)video_codebooks_8x8[gop_index];
    for(int codeword_idx = 0; codeword_idx < VIDEO_CODEBOOK_SIZE_8x8; codeword_idx++)
    {
        // 直接复制BGR555数据，每个8x8码字64个u16值
        const u16* src = codebook_8x8 + codeword_idx * 64;
        BGR555_8x8_Struct* dst = bgr555_buffer_8x8 + codeword_idx;
        
        // 使用 memcpy 复制数据
        memcpy(dst->bgr_array, src, 64 * sizeof(u16));
    }
    
    // 加载8x4码表
    const u16* codebook_8x4 = (u16*)video_codebooks_8x4[gop_index];
    for(int codeword_idx = 0; codeword_idx < VIDEO_CODEBOOK_SIZE_8x4; codeword_idx++)
    {
        // 直接复制BGR555数据，每个8x4码字32个u16值
        const u16* src = codebook_8x4 + codeword_idx * 32;
        BGR555_8x4_Struct* dst = bgr555_buffer_8x4 + codeword_idx;
        
        // 使用 memcpy 复制数据
        memcpy(dst->bgr_array, src, 32 * sizeof(u16));
    }
    
    // 加载4x4码表
    const u16* codebook_4x4 = (u16*)video_codebooks_4x4[gop_index];
    for(int codeword_idx = 0; codeword_idx < VIDEO_CODEBOOK_SIZE_4x4; codeword_idx++)
    {
        // 直接复制BGR555数据，每个4x4码字16个u16值
        const u16* src = codebook_4x4 + codeword_idx * 16;
        BGR555_4x4_Struct* dst = bgr555_buffer_4x4 + codeword_idx;
        
        // 使用 memcpy 复制数据
        memcpy(dst->bgr_array, src, 16 * sizeof(u16));
    }
    
    // 加载4x2码表
    const u16* codebook_4x2 = (u16*)video_codebooks_4x2[gop_index];
    for(int codeword_idx = 0; codeword_idx < VIDEO_CODEBOOK_SIZE_4x2; codeword_idx++)
    {
        // 直接复制BGR555数据，每个4x2码字8个u16值
        const u16* src = codebook_4x2 + codeword_idx * 8;
        BGR555_4x2_Struct* dst = bgr555_buffer_4x2 + codeword_idx;
        
        // 使用 memcpy 复制数据
        memcpy(dst->bgr_array, src, 8 * sizeof(u16));
    }
}

// 从BGR555码表缓冲区解码一个8x8块
IWRAM_CODE void decode_8x8_block_from_bgr555_buffer(u16 codeword_idx, u16* dst, int dst_stride)
{
    const BGR555_8x8_Struct &bgr555_block = bgr555_buffer_8x8[codeword_idx];

    // 直接复制BGR555数据，每行8像素
    u16* row = dst;
    for(int y = 0; y < 8; y++) {
        ((u32*)(row))[0] = bgr555_block.bgr_u32[y][0]; // 每次复制2个像素
        ((u32*)(row))[1] = bgr555_block.bgr_u32[y][1]; // 每次复制2个像素
        ((u32*)(row))[2] = bgr555_block.bgr_u32[y][2]; // 每次复制2个像素
        ((u32*)(row))[3] = bgr555_block.bgr_u32[y][3]; // 每次复制2个像素
        row += dst_stride;
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


// 递归解码8x8块编码数据
IWRAM_CODE int decode_8x8_recursive(const u8* data, int data_idx, u16* dst, int dst_stride)
{
    u8 first_element = data[data_idx];
    
    if (first_element == VIDEO_MARKER_SPLIT) {
        // 8x8分裂：需要拆分为两个8x4块
        data_idx++; // 跳过MARKER_SPLIT
        
        // 递归解码上半8x4块 (前4行)
        int upper_consumed = decode_8x4_recursive(data, data_idx, dst, dst_stride);
        data_idx += upper_consumed;
        
        // 递归解码下半8x4块 (后4行)
        u16* lower_dst = dst + 4 * dst_stride;
        int lower_consumed = decode_8x4_recursive(data, data_idx, lower_dst, dst_stride);
        data_idx += lower_consumed;
        
        return 1 + upper_consumed + lower_consumed; // MARKER + 上编码 + 下编码
    } else {
        // 直接8x8码字：使用8x8码表
        u8 codeword_idx = first_element;
        decode_8x8_block_from_bgr555_buffer(codeword_idx, dst, dst_stride);
        return 1; // 消耗1个u8
    }
}

// 递归解码4x2块编码数据（最小块，不再分裂）
IWRAM_CODE int decode_4x2_recursive(const u8* data, int data_idx, u16* dst, int dst_stride)
{
    // 4x2是最小块，不会再分裂，直接使用4x2码表解码
    u8 codeword_idx = data[data_idx];
    decode_4x2_block_from_bgr555_buffer(codeword_idx, dst, dst_stride);
    return 1; // 消耗1个u8
}

// 递归解码4x4块编码数据
IWRAM_CODE int decode_4x4_recursive(const u8* data, int data_idx, u16* dst, int dst_stride)
{
    u8 first_element = data[data_idx];
    
    if (first_element == VIDEO_MARKER_SPLIT) {
        // 4x4分裂：需要拆分为两个4x2块
        data_idx++; // 跳过MARKER_SPLIT
        
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
        u8 codeword_idx = first_element;
        decode_4x4_block_from_bgr555_buffer(codeword_idx, dst, dst_stride);
        return 1; // 消耗1个u8
    }
}

// 递归解码8x4块编码数据
IWRAM_CODE int decode_8x4_recursive(const u8* data, int data_idx, u16* dst, int dst_stride)
{
    u8 first_element = data[data_idx];
    
    if (first_element == VIDEO_MARKER_SPLIT) {
        // 8x4分裂：需要拆分为两个4x4块
        data_idx++; // 跳过MARKER_SPLIT
        
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
        u8 codeword_idx = first_element;
        decode_8x4_block_from_bgr555_buffer(codeword_idx, dst, dst_stride);
        return 1; // 消耗1个u8
    }
}

IWRAM_CODE void decode_i_frame_multi_level(const u8* frame_data, u16* dst)
{
    // I帧新递归格式：[总块数(2字节), 块1编码..., 块2编码..., ...]
    // 其中每个块编码是递归的：
    // - 8x8块：8x8码字索引 (直接是索引)
    // - 分裂为8x4块：MARKER_SPLIT, 上半8x4编码..., 下半8x4编码...
    // - 分裂为4x4块：MARKER_SPLIT, 左半4x4编码..., 右半4x4编码...
    // - 分裂为4x2块：MARKER_SPLIT, 上半4x2码字索引, 下半4x2码字索引
    
    // 读取总块数（2字节，小端序）
    // u16 total_blocks = read_u16_from_u8(frame_data, 0);
    int data_idx = 2;  // 跳过2字节的总块数
    
    for (int block_8x8_pos = 0; block_8x8_pos < VIDEO_BLOCKS_8x8_PER_FRAME; block_8x8_pos++)
    {
        u16* block_8x8_dst = dst + block_8x8_offset_table[block_8x8_pos];
        
        // 递归解码当前8x8块
        int consumed = decode_8x8_recursive(frame_data, data_idx, block_8x8_dst, SCREEN_WIDTH);
        data_idx += consumed;
    }
}

IWRAM_CODE void decode_p_frame_multi_level(const u8* frame_data, u16* dst)
{
    // P帧新递归格式：[变化块数(2字节), 位置1(2字节), 块1编码..., 位置2(2字节), 块2编码..., ...]
    // 其中每个块编码是递归的，与I帧相同
    
    int data_idx = 0;
    
    // 读取变化块数（2字节，小端序）
    u16 changed_count = read_u16_from_u8(frame_data, data_idx);
    data_idx += 2;
    
    for (int i = 0; i < changed_count; i++)
    {
        u16 block_8x8_pos = read_u16_from_u8(frame_data, data_idx);     // 8x8块位置
        data_idx += 2;
        u16* block_8x8_dst = dst + block_8x8_offset_table[block_8x8_pos];
        
        // 递归解码当前8x8块
        int consumed = decode_8x8_recursive(frame_data, data_idx, block_8x8_dst, SCREEN_WIDTH);
        data_idx += consumed;
    }
}

IWRAM_CODE int decode_zone_bitmap_and_data(const u8* data, int data_idx, u16* dst, 
                                          int block_type_id, const u16* offset_table)
{
    int original_data_idx = data_idx;
    
    // 1. 读取bitmap
    u32 bitmap = 0;
    int zones_count = 0;
    
    switch (block_type_id) {
        case 0: // 8x8
            zones_count = VIDEO_ZONES_8x8_COUNT;
            bitmap = data[data_idx];
            data_idx += 1;
            break;
        case 1: // 8x4
            zones_count = VIDEO_ZONES_8x4_COUNT;
            bitmap = data[data_idx];
            data_idx += 1;
            break;
        case 2: // 4x4
            zones_count = VIDEO_ZONES_4x4_COUNT;
            bitmap = read_u16_from_u8(data, data_idx);
            data_idx += 2;
            break;
        case 3: // 4x2
            zones_count = VIDEO_ZONES_4x2_COUNT;
            bitmap = read_u32_from_u8(data, data_idx);
            data_idx += 4;
            break;
    }
    
    // 2. 处理每个活跃的zone
    for (int zone_id = 0; zone_id < zones_count; zone_id++) {
        if (!(bitmap & (1 << zone_id))) {
            continue; // 此zone没有变化
        }
        
        // 读取此zone的段数量
        u8 segment_count = data[data_idx++];
        
        // 处理每个段
        for (int seg = 0; seg < segment_count; seg++) {
            u8 zone_offset = data[data_idx++];  // zone内偏移
            u8 run_length = data[data_idx++];   // 连续块数量
            
            // 解码连续的块
            for (int i = 0; i < run_length; i++) {
                u8 codeword_idx = data[data_idx++]; // 码表索引
                
                // 计算全局块位置
                int global_block_pos = zone_id * VIDEO_BLOCKS_PER_ZONE + zone_offset + i;
                
                // 获取目标像素位置
                u16* block_dst = dst + offset_table[global_block_pos];
                
                // 根据块类型解码
                switch (block_type_id) {
                    case 0: // 8x8
                        decode_8x8_block_from_bgr555_buffer(codeword_idx, block_dst, SCREEN_WIDTH);
                        break;
                    case 1: // 8x4
                        decode_8x4_block_from_bgr555_buffer(codeword_idx, block_dst, SCREEN_WIDTH);
                        break;
                    case 2: // 4x4
                        decode_4x4_block_from_bgr555_buffer(codeword_idx, block_dst, SCREEN_WIDTH);
                        break;
                    case 3: // 4x2
                        decode_4x2_block_from_bgr555_buffer(codeword_idx, block_dst, SCREEN_WIDTH);
                        break;
                }
            }
        }
    }
    
    return data_idx - original_data_idx; // 返回消耗的字节数
}

IWRAM_CODE void decode_p_frame_zone_optimization(const u8* frame_data, u16* dst)
{
    // P帧zone编码格式：
    // [8x8 zone数据] + [8x4 zone数据] + [4x4 zone数据] + [4x2 zone数据]
    // 每个zone数据格式：bitmap + 每个活跃zone的段数据
    
    int data_idx = 0;
    
    // 1. 解码8x8块 (block_type_id = 0)
    data_idx += decode_zone_bitmap_and_data(frame_data, data_idx, dst, 0, block_8x8_offset_table);
    
    // 2. 解码8x4块 (block_type_id = 1)
    data_idx += decode_zone_bitmap_and_data(frame_data, data_idx, dst, 1, block_8x4_offset_table);
    
    // 3. 解码4x4块 (block_type_id = 2)
    data_idx += decode_zone_bitmap_and_data(frame_data, data_idx, dst, 2, block_4x4_offset_table);
    
    // 4. 解码4x2块 (block_type_id = 3)
    data_idx += decode_zone_bitmap_and_data(frame_data, data_idx, dst, 3, block_4x2_offset_table);
}

static volatile u32 vbl = 0;
static volatile u32 played_frames = 0;
static volatile u32 FPS = 0;
u16 FPS_DISPLAY_LEN = 0;
void isr_vbl() { 
    ++vbl; 
    if(vbl >= 60){
        vbl = 0;
        FPS = played_frames;
        played_frames = 0;
        FPS_DISPLAY_LEN = SCREEN_WIDTH/30 * FPS;
    }
    REG_IF = IRQ_VBLANK; 
}

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
    init_block_8x8_offset_table();
    init_block_8x4_offset_table();
    init_block_4x4_offset_table();
    init_block_4x2_offset_table();
    
    while (1)
    {
        // 检查是否需要切换GOP
        int required_gop = frame / VIDEO_GOP_SIZE;
        if (required_gop != current_gop) {
            current_gop = required_gop;
            // 加载新GOP的四级BGR555码表到缓冲区
            load_bgr555_codebooks(current_gop);
        }
        
        // 获取当前帧的数据
        u32 frame_offset = video_frame_offsets[frame];
        const u8* frame_data = video_frame_data + frame_offset;
        u8 frame_type = video_frame_types[frame];
        
        // 根据帧类型解码
        if (frame_type == 0) {
            // I帧：完全重绘
            decode_i_frame_multi_level(frame_data, ewramBuffer);
        } else {
            // P帧：增量更新（使用zone优化解码）
            decode_p_frame_zone_optimization(frame_data, ewramBuffer);
        }

        // 复制到VRAM
        played_frames++;
        u16 FPS_DISPLAY_COLOR = RGB5(31, 31, 31); // 白色
        // 在屏幕上显示FPS,条状显示，节省资源
        DMA3COPY(ewramBuffer, VRAM, PIXELS_PER_FRAME | DMA16);
        if(FPS_DISPLAY_LEN > 0)
            DMA3COPY(&FPS_DISPLAY_COLOR, VRAM, FPS_DISPLAY_LEN | DMA16 | DMA_SRC_FIXED);

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
