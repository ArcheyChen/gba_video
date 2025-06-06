// gba_video_player.cpp  v6
// Mode 3 单缓冲 + YUV9 → RGB555 + 条带帧间差分解码 + 向量量化

#include <gba_systemcalls.h>
#include <gba_video.h>
#include <gba_dma.h>
#include <gba_interrupt.h>
#include <gba_input.h>
#include <cstring>

#include "video_data.h"
#include <tuple>

constexpr int PIXELS_PER_FRAME = SCREEN_WIDTH * SCREEN_HEIGHT;

// EWRAM 单缓冲
EWRAM_BSS u16 ewramBuffer[PIXELS_PER_FRAME];
IWRAM_DATA static u8 clip_table_raw[512];//u8+s8最大范围是在[512]大小内的
u8* clip_lookup_table = clip_table_raw + 128; //s8最小值是-128，预先添加偏移，这样查表的时候，遇到负数也直接查
struct YUV_Struct{
    u8 y[2][2];
    s8 d_r;    // 预计算的 Cr
    s8 d_g;    // 预计算的 (-(Cb>>1)-Cr)>>1
    s8 d_b;    // 预计算的 Cb
} __attribute__((packed));

// 每个条带的码表存储（在EWRAM中）
IWRAM_DATA u8 strip_codebooks[VIDEO_STRIP_COUNT][CODEBOOK_SIZE][BYTES_PER_BLOCK];

void init_table(){
    for(int i=-128;i<512-128;i++){
        u8 raw_val;
        if(i<=0)
            raw_val = 0; // 小于等于0的值都裁剪为0
        else if(i>=255)
            raw_val = 255; // 大于等于255的值都裁剪为255
        else
            raw_val = static_cast<u8>(i); // 其他值直接赋值
        clip_lookup_table[i] = raw_val>>2; // 填充查找表，改为>>2因为码表值已预先>>1
    }
}

IWRAM_CODE inline u16 yuv_to_rgb555(u8 y, s8 d_r, s8 d_g, s8 d_b)
{
    // 使用预计算的查找表进行转换    
    u32 result = clip_lookup_table[y + d_r];
    result |= (clip_lookup_table[y + d_g] << 5);
    return result | (clip_lookup_table[y + d_b] << 10);
}


// 条带信息结构
struct StripInfo {
    u16 start_y;       // 条带起始Y坐标
    u16 height;        // 条带高度
    u16 blocks_per_row; // 每行块数
    u16 blocks_per_col; // 每列块数
    u16 total_blocks;   // 总块数
    u16 buffer_offset; // 条带在缓冲区中的起始偏移
};

IWRAM_DATA StripInfo strip_info[VIDEO_STRIP_COUNT];
// 通用的块相对偏移表，所有条带共用
IWRAM_DATA u16 block_relative_offsets[240/2*80/2]; // 2x2块数

void init_strip_info(){
    u16 current_y = 0;

    // 首先计算最大的条带块数，用于初始化通用偏移表
    u16 max_blocks_per_row = VIDEO_WIDTH / BLOCK_WIDTH;
    u16 max_blocks_per_col = 0;

    for(int strip_idx = 0; strip_idx < VIDEO_STRIP_COUNT; strip_idx++){
        u16 strip_blocks_per_col = strip_heights[strip_idx] / BLOCK_HEIGHT;
        if(strip_blocks_per_col > max_blocks_per_col) {
            max_blocks_per_col = strip_blocks_per_col;
        }
    }

    // 初始化通用的块相对偏移表
    for(int block_idx = 0; block_idx < max_blocks_per_row * max_blocks_per_col; block_idx++){
        u16 bx = block_idx % max_blocks_per_row;
        u16 by = block_idx / max_blocks_per_row;
        block_relative_offsets[block_idx] = (by * BLOCK_HEIGHT * SCREEN_WIDTH) + (bx * BLOCK_WIDTH);
    }

    // 初始化各条带信息
    for(int strip_idx = 0; strip_idx < VIDEO_STRIP_COUNT; strip_idx++){
        strip_info[strip_idx].start_y = current_y;
        strip_info[strip_idx].height = strip_heights[strip_idx];
        strip_info[strip_idx].blocks_per_row = VIDEO_WIDTH / BLOCK_WIDTH;
        strip_info[strip_idx].blocks_per_col = strip_heights[strip_idx] / BLOCK_HEIGHT;
        strip_info[strip_idx].total_blocks = strip_info[strip_idx].blocks_per_row * strip_info[strip_idx].blocks_per_col;
        strip_info[strip_idx].buffer_offset = current_y * SCREEN_WIDTH;
        current_y += strip_heights[strip_idx];
    }
}

// 解码单个4x4块到指定位置
IWRAM_CODE inline void decode_block(const u8* src, u16* dst)
{
    YUV_Struct *yuv_data = (YUV_Struct*)src;
    s8 d_r = yuv_data->d_r;
    s8 d_g = yuv_data->d_g;
    s8 d_b = yuv_data->d_b;

    // 解码2x2像素
    u16* dst_row = dst;
    for(int row = 0; row < 2; row++) {
        dst_row[0] = yuv_to_rgb555(yuv_data->y[row][0], d_r, d_g, d_b);
        dst_row[1] = yuv_to_rgb555(yuv_data->y[row][1], d_r, d_g, d_b);
        dst_row += SCREEN_WIDTH;
    }
}

IWRAM_CODE void decode_strip_i_frame(int strip_idx, const u8* src, u16* dst)
{
    u16 strip_base_offset = strip_info[strip_idx].buffer_offset;
    
    // 读取码表大小（应该是CODEBOOK_SIZE）
    u16 codebook_size = src[0] | (src[1] << 8);
    src += 2;
    
    // 读取码表数据
    memcpy(strip_codebooks[strip_idx], src, codebook_size * BYTES_PER_BLOCK);
    src += codebook_size * BYTES_PER_BLOCK;
    
    // 解码条带内所有块（使用量化索引）
    for (int block_idx = 0; block_idx < strip_info[strip_idx].total_blocks; block_idx++) {
        u8 quant_idx = *src++;
        // 从码表中获取块数据并解码
        decode_block(strip_codebooks[strip_idx][quant_idx], 
                    dst + strip_base_offset + block_relative_offsets[block_idx]);
    }
}

IWRAM_CODE void decode_strip_p_frame(int strip_idx, const u8* src, u16* dst)
{
    u16 strip_base_offset = strip_info[strip_idx].buffer_offset;
    
    // 读取需要更新的块数（小端序）
    u16 blocks_to_update = src[0] | (src[1] << 8);
    src += 2;
    
    // 处理每个需要更新的块
    for (u16 i = 0; i < blocks_to_update; i++) {
        // 读取块索引（小端序）
        u16 block_idx = src[0] | (src[1] << 8);
        src += 2;
        
        // 读取量化索引
        u8 quant_idx = *src++;
        
        // 确保块索引在有效范围内
        if (block_idx < strip_info[strip_idx].total_blocks) {
            // 从码表中获取块数据并解码
            decode_block(strip_codebooks[strip_idx][quant_idx], 
                        dst + strip_base_offset + block_relative_offsets[block_idx]);
        }
    }
}

IWRAM_CODE void decode_strip(int strip_idx, const u8* src, u16 strip_data_size, u16* dst)
{
    if (strip_data_size == 0) return;
    
    // 检查帧类型
    u8 frame_type = *src++;
    
    if (frame_type == FRAME_TYPE_I) {
        decode_strip_i_frame(strip_idx, src, dst);
    } else if (frame_type == FRAME_TYPE_P) {
        decode_strip_p_frame(strip_idx, src, dst);
    }
}

IWRAM_CODE void decode_frame(const u8* frame_data, u16* dst)
{
    const u8* src = frame_data;
    
    // 解码每个条带
    for (int strip_idx = 0; strip_idx < VIDEO_STRIP_COUNT; strip_idx++) {
        // 读取条带数据长度（小端序）
        u16 strip_data_size = src[0] | (src[1] << 8);
        src += 2;
        
        // 解码当前条带
        decode_strip(strip_idx, src, strip_data_size, dst);
        
        // 移动到下一个条带
        src += strip_data_size;
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

    init_table();
    init_strip_info();
    
    // 清空缓冲区
    memset(ewramBuffer, 0, PIXELS_PER_FRAME * sizeof(u16));
    // VBlankIntrWait();
    DMA3COPY(ewramBuffer, VRAM, PIXELS_PER_FRAME | DMA16);

    int frame = 0;
    
    while (1)
    {
        // 使用偏移表获取当前帧的数据位置
        const unsigned char* frame_data = video_data + frame_offsets[frame];
        
        // 解码当前帧（按条带处理I帧或P帧）
        decode_frame(frame_data, ewramBuffer);

        // 等待垂直同步并复制到VRAM
        // VBlankIntrWait();
        DMA3COPY(ewramBuffer, VRAM, PIXELS_PER_FRAME | DMA16);

        frame++;
        if(frame >= VIDEO_FRAME_COUNT) {
            frame = 0; // 循环播放
        }

        // 按键检测（可选）
        // scanKeys();
        // u16 keys = keysDown();
        
        // if (keys & KEY_START) {
        //     // 暂停功能
        //     while (!(keysDown() & KEY_START)) {
        //         scanKeys();
        //         VBlankIntrWait();
        //     }
        // }
        
        // if (keys & KEY_A) {
        //     // 快进：跳过5帧
        //     frame += 5;
        //     if (frame >= VIDEO_FRAME_COUNT) frame = 0;
        // }
    }
}