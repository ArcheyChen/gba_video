// gba_video_player.cpp  v7
// Mode 3 单缓冲 + YUV9 → RGB555 + 条带帧间差分解码 + 双码本向量量化

#include <gba_systemcalls.h>
#include <gba_video.h>
#include <gba_dma.h>
#include <gba_interrupt.h>
#include <gba_input.h>
#include <cstring>

#include "video_data.h"
#include <tuple>

constexpr int PIXELS_PER_FRAME = SCREEN_WIDTH * SCREEN_HEIGHT;

// 新增常量定义
#define ZONE_HEIGHT_PIXELS 16  // 每个区域的像素高度
#define ZONE_HEIGHT_BIG_BLOCKS (ZONE_HEIGHT_PIXELS / (BLOCK_HEIGHT * 2))  // 每个区域的4x4大块行数

// EWRAM 单缓冲
EWRAM_BSS u16 ewramBuffer[PIXELS_PER_FRAME];
IWRAM_DATA static u8 clip_table_raw[512];
u8* clip_lookup_table = clip_table_raw + 128;

struct YUV_Struct{
    u8 y[2][2];
    s8 d_r;    // 预计算的 Cr
    s8 d_g;    // 预计算的 (-(Cb>>1)-Cr)>>1
    s8 d_b;    // 预计算的 Cb
} __attribute__((packed));

// 统一码本存储（在IWRAM中）
IWRAM_DATA u8 strip_unified_codebooks_raw[VIDEO_STRIP_COUNT][UNIFIED_CODEBOOK_SIZE*sizeof(YUV_Struct)+4]__attribute__((aligned(32)));
IWRAM_DATA YUV_Struct *strip_unified_codebooks[VIDEO_STRIP_COUNT];

// 大块索引码表存储（在IWRAM中）
IWRAM_DATA u8 strip_big_block_codebooks_raw[VIDEO_STRIP_COUNT][BIG_BLOCK_CODEBOOK_SIZE*4+4]__attribute__((aligned(32)));
IWRAM_DATA u8 (*strip_big_block_codebooks[VIDEO_STRIP_COUNT])[4];

void init_table(){
    for(int i = 0; i < VIDEO_STRIP_COUNT; i++) {
        strip_unified_codebooks[i] = (YUV_Struct*)strip_unified_codebooks_raw[i];
        strip_big_block_codebooks[i] = (u8(*)[4])strip_big_block_codebooks_raw[i];
    }
    for(int i=-128;i<512-128;i++){
        u8 raw_val;
        if(i<=0)
            raw_val = 0;
        else if(i>=255)
            raw_val = 255;
        else
            raw_val = static_cast<u8>(i);
        clip_lookup_table[i] = raw_val>>2;
    }
}

IWRAM_CODE inline u32 yuv_to_rgb555_2pix(const u8 y[2], s8 d_r, s8 d_g, s8 d_b)
{
    u8 _y = y[0];
    u32 result = clip_lookup_table[_y + d_r];
    result |= (clip_lookup_table[_y + d_g] << 5);
    result |= (clip_lookup_table[_y + d_b] << 10);

    _y = y[1];
    result |= (clip_lookup_table[_y + d_r] << 16);
    result |= (clip_lookup_table[_y + d_g] << 21);
    result |= (clip_lookup_table[_y + d_b] << 26);
    
    return result;
}

// 条带信息结构
struct StripInfo {
    u16 start_y;
    u16 height;
    u16 blocks_per_row;
    u16 blocks_per_col;
    u16 total_blocks;
    u16 buffer_offset;
};

IWRAM_DATA StripInfo strip_info[VIDEO_STRIP_COUNT];
IWRAM_DATA u16 big_block_relative_offsets[240/4*80/4];

void init_strip_info(){
    u16 current_y = 0;

    u16 max_big_blocks_per_row = VIDEO_WIDTH / (BLOCK_WIDTH * 2);
    u16 max_big_blocks_per_col = 0;

    for(int strip_idx = 0; strip_idx < VIDEO_STRIP_COUNT; strip_idx++){
        u16 strip_big_blocks_per_col = strip_heights[strip_idx] / (BLOCK_HEIGHT * 2);
        if(strip_big_blocks_per_col > max_big_blocks_per_col) {
            max_big_blocks_per_col = strip_big_blocks_per_col;
        }
    }

    for(int big_block_idx = 0; big_block_idx < max_big_blocks_per_row * max_big_blocks_per_col; big_block_idx++){
        u16 big_bx = big_block_idx % max_big_blocks_per_row;
        u16 big_by = big_block_idx / max_big_blocks_per_row;
        big_block_relative_offsets[big_block_idx] = (big_by * BLOCK_HEIGHT * 2 * SCREEN_WIDTH) + (big_bx * BLOCK_WIDTH * 2);
    }

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

// 解码单个2x2块
IWRAM_CODE inline void decode_block(const YUV_Struct &yuv_data, u16* dst)
{
    const s8 &d_r = yuv_data.d_r;
    const s8 &d_g = yuv_data.d_g;
    const s8 &d_b = yuv_data.d_b;

    u32* dst_row = (u32*)dst;
    *dst_row = yuv_to_rgb555_2pix(
        yuv_data.y[0], d_r, d_g, d_b);
    *(dst_row + SCREEN_WIDTH/2) = yuv_to_rgb555_2pix(
        yuv_data.y[1], d_r, d_g, d_b);
}

// 解码色块（2x2上采样到4x4）
IWRAM_CODE inline void decode_color_block(const YUV_Struct &yuv_data, u16* dst)
{
    const s8 &d_r = yuv_data.d_r;
    const s8 &d_g = yuv_data.d_g;
    const s8 &d_b = yuv_data.d_b;

    // 将2x2的Y值上采样到4x4
    // 原始: 12/34 -> 目标: 1122/1122/3344/3344
    // y[0][0]=1, y[0][1]=2, y[1][0]=3, y[1][1]=4
    
    // 为每个重复的Y值对创建数组
    u8 y_11[2] = {yuv_data.y[0][0], yuv_data.y[0][0]}; // 1,1
    u8 y_22[2] = {yuv_data.y[0][1], yuv_data.y[0][1]}; // 2,2
    u8 y_33[2] = {yuv_data.y[1][0], yuv_data.y[1][0]}; // 3,3
    u8 y_44[2] = {yuv_data.y[1][1], yuv_data.y[1][1]}; // 4,4
    
    // 生成4个2像素组合
    u32 pix_11 = yuv_to_rgb555_2pix(y_11, d_r, d_g, d_b);
    u32 pix_22 = yuv_to_rgb555_2pix(y_22, d_r, d_g, d_b);
    u32 pix_33 = yuv_to_rgb555_2pix(y_33, d_r, d_g, d_b);
    u32 pix_44 = yuv_to_rgb555_2pix(y_44, d_r, d_g, d_b);

    u32* dst_row = (u32*)dst;
    
    // 第一行：1122
    *dst_row = pix_11;
    *(dst_row + 1) = pix_22;
    
    // 第二行：1122
    dst_row += SCREEN_WIDTH/2;
    *dst_row = pix_11;
    *(dst_row + 1) = pix_22;
    
    // 第三行：3344
    dst_row += SCREEN_WIDTH/2;
    *dst_row = pix_33;
    *(dst_row + 1) = pix_44;
    
    // 第四行：3344
    dst_row += SCREEN_WIDTH/2;
    *dst_row = pix_33;
    *(dst_row + 1) = pix_44;
}

// 通用的4x4大块解码函数（纹理块）
IWRAM_CODE inline void decode_big_block(const YUV_Struct* codebook, const u8 quant_indices[4], u16* big_block_dst)
{
    decode_block(codebook[quant_indices[0]], big_block_dst);
    decode_block(codebook[quant_indices[1]], big_block_dst + 2);
    decode_block(codebook[quant_indices[2]], big_block_dst + SCREEN_WIDTH * 2);
    decode_block(codebook[quant_indices[3]], big_block_dst + SCREEN_WIDTH * 2 + 2);
}

// DMA拷贝码本的辅助函数
IWRAM_CODE void copy_unified_codebook(u8* dst_raw, const u8* src, YUV_Struct** codebook_ptr, int codebook_size)
{
    u8* copy_raw_ptr = dst_raw + 4; // 跳过对齐填充的4字节
    *codebook_ptr = (YUV_Struct*)copy_raw_ptr;
    int remain_copy = codebook_size * BYTES_PER_BLOCK;
    
    if(((u32)src) & 1){
        u8 data = *src++;
        copy_raw_ptr[-1] = copy_raw_ptr[-3] = data;
        *codebook_ptr = (YUV_Struct*)((u32)*codebook_ptr - 1);
        remain_copy -= 1;
    }
    if(((u32)src) & 2){
        copy_raw_ptr[-2] = *src++;
        copy_raw_ptr[-1] = *src++;
        *codebook_ptr = (YUV_Struct*)((u32)*codebook_ptr - 2);
        remain_copy -= 2;
    }

    DMA3COPY(src, copy_raw_ptr, (remain_copy>>2) | DMA32);
    int tail = remain_copy & 3;
    int body = remain_copy - tail;
    src += body;
    copy_raw_ptr += body;
    while(tail--) {
        *copy_raw_ptr++ = *src++;
    }
}

// DMA拷贝大块索引码表的辅助函数
IWRAM_CODE void copy_big_block_codebook(u8* dst_raw, const u8* src, u8 (** codebook_ptr)[4], int codebook_size)
{
    u8* copy_raw_ptr = dst_raw + 4; // 跳过对齐填充的4字节
    *codebook_ptr = (u8(*)[4])copy_raw_ptr;
    int remain_copy = codebook_size * 4; // 每项4字节
    
    if(((u32)src) & 1){
        u8 data = *src++;
        copy_raw_ptr[-1] = copy_raw_ptr[-3] = data;
        *codebook_ptr = (u8(*)[4])((u32)*codebook_ptr - 1);
        remain_copy -= 1;
    }
    if(((u32)src) & 2){
        copy_raw_ptr[-2] = *src++;
        copy_raw_ptr[-1] = *src++;
        *codebook_ptr = (u8(*)[4])((u32)*codebook_ptr - 2);
        remain_copy -= 2;
    }

    DMA3COPY(src, copy_raw_ptr, (remain_copy>>2) | DMA32);
    int tail = remain_copy & 3;
    int body = remain_copy - tail;
    src += body;
    copy_raw_ptr += body;
    while(tail--) {
        *copy_raw_ptr++ = *src++;
    }
}

IWRAM_CODE void decode_strip_i_frame_unified_with_big_block(int strip_idx, const u8* src, u16* dst)
{
    u16 strip_base_offset = strip_info[strip_idx].buffer_offset;
    
    // 拷贝统一码本
    copy_unified_codebook(strip_unified_codebooks_raw[strip_idx], src, 
                         &strip_unified_codebooks[strip_idx], UNIFIED_CODEBOOK_SIZE);
    src += UNIFIED_CODEBOOK_SIZE * BYTES_PER_BLOCK;
    
    // 拷贝大块索引码表
    copy_big_block_codebook(strip_big_block_codebooks_raw[strip_idx], src,
                           &strip_big_block_codebooks[strip_idx], BIG_BLOCK_CODEBOOK_SIZE);
    src += BIG_BLOCK_CODEBOOK_SIZE * 4;
    
    auto &strip = strip_info[strip_idx];
    auto &unified_codebook = strip_unified_codebooks[strip_idx];
    auto &big_block_codebook = strip_big_block_codebooks[strip_idx];
    
    u16 tot_big_blocks = strip.total_blocks >> 2;
    u16* strip_dst = dst + strip_base_offset;
    
    // 解码所有4x4大块
    for (int big_block_idx = 0; big_block_idx < tot_big_blocks; big_block_idx++) {
        u16* big_block_dst = strip_dst + big_block_relative_offsets[big_block_idx];
        
        // 读取第一个索引/标记
        u8 first_byte = *src++;
        
        if (first_byte == COLOR_BLOCK_MARKER) {
            // 色块：读取统一码本索引
            u8 unified_idx = *src++;
            decode_color_block(unified_codebook[unified_idx], big_block_dst);
        } else if (first_byte == COMPLEX_TEXTURE_MARKER) {
            // 复杂纹理块：读取4个统一码本索引
            u8 quant_indices[4];
            quant_indices[0] = *src++;
            quant_indices[1] = *src++;
            quant_indices[2] = *src++;
            quant_indices[3] = *src++;
            decode_big_block(unified_codebook, quant_indices, big_block_dst);
        } else {
            // 大块索引：使用大块索引码表
            u8 big_block_idx_val = first_byte;
            if (big_block_idx_val < EFFECTIVE_BIG_BLOCK_CODEBOOK_SIZE) {
                // 从大块索引码表获取4个统一码本索引
                decode_big_block(unified_codebook, big_block_codebook[big_block_idx_val], big_block_dst);
            } else {
                // 错误情况，使用默认处理
                u8 quant_indices[4] = {0, 0, 0, 0};
                decode_big_block(unified_codebook, quant_indices, big_block_dst);
            }
        }
    }
}

IWRAM_CODE void decode_strip_p_frame_unified_with_big_block(int strip_idx, const u8* src, u16* dst)
{
    u16 strip_base_offset = strip_info[strip_idx].buffer_offset;
    
    // 读取区域bitmap
    u8 zone_bitmap = *src++;
    
    auto &unified_codebook = strip_unified_codebooks[strip_idx];
    auto &big_block_codebook = strip_big_block_codebooks[strip_idx];
    
    // 计算该条带的大块布局
    u16 strip_big_blocks_w = VIDEO_WIDTH / (BLOCK_WIDTH * 2);
    
    // 使用bitmap右移优化处理每个有效区域
    u8 zone_idx = 0;
    while (zone_bitmap) {
        if (zone_bitmap & 1) {
            // 读取纹理块更新数量
            u8 detail_blocks_to_update = *src++;
            
            // 读取色块更新数量
            u8 color_blocks_to_update = *src++;
            
            // 计算区域在条带中的起始大块行
            u16 zone_start_big_by = zone_idx * ZONE_HEIGHT_BIG_BLOCKS;
            
            // 处理纹理块更新
            for (u8 i = 0; i < detail_blocks_to_update; i++) {
                u8 zone_relative_idx = *src++;
                
                // 读取第一个字节判断编码类型
                u8 first_byte = *src++;
                
                u8 quant_indices[4];
                
                if (first_byte == COMPLEX_TEXTURE_MARKER) {
                    // 复杂纹理模式：读取4个统一码本索引
                    quant_indices[0] = *src++;
                    quant_indices[1] = *src++;
                    quant_indices[2] = *src++;
                    quant_indices[3] = *src++;
                } else {
                    // 大块索引模式：从大块索引码表获取4个索引
                    u8 big_block_idx_val = first_byte;
                    if (big_block_idx_val < EFFECTIVE_BIG_BLOCK_CODEBOOK_SIZE) {
                        quant_indices[0] = big_block_codebook[big_block_idx_val][0];
                        quant_indices[1] = big_block_codebook[big_block_idx_val][1];
                        quant_indices[2] = big_block_codebook[big_block_idx_val][2];
                        quant_indices[3] = big_block_codebook[big_block_idx_val][3];
                    } else {
                        // 错误情况，使用默认值
                        quant_indices[0] = quant_indices[1] = quant_indices[2] = quant_indices[3] = 0;
                    }
                }
                
                // 将区域相对坐标转换为条带内的绝对坐标
                u16 relative_big_by = zone_relative_idx / strip_big_blocks_w;
                u16 relative_big_bx = zone_relative_idx % strip_big_blocks_w;
                u16 absolute_big_by = zone_start_big_by + relative_big_by;
                u16 big_block_idx = absolute_big_by * strip_big_blocks_w + relative_big_bx;
                
                u16* big_block_dst = dst + strip_base_offset + big_block_relative_offsets[big_block_idx];
                decode_big_block(unified_codebook, quant_indices, big_block_dst);
            }
            
            // 处理色块更新
            for (u8 i = 0; i < color_blocks_to_update; i++) {
                u8 zone_relative_idx = *src++;
                
                // 跳过COLOR_BLOCK_MARKER（已在编码时写入）
                u8 marker = *src++;
                if (marker != COLOR_BLOCK_MARKER) {
                    // 错误处理：如果不是预期的标记，回退一步
                    src--;
                }
                
                u8 unified_idx = *src++;
                
                // 将区域相对坐标转换为条带内的绝对坐标
                u16 relative_big_by = zone_relative_idx / strip_big_blocks_w;
                u16 relative_big_bx = zone_relative_idx % strip_big_blocks_w;
                u16 absolute_big_by = zone_start_big_by + relative_big_by;
                u16 big_block_idx = absolute_big_by * strip_big_blocks_w + relative_big_bx;
                
                u16* big_block_dst = dst + strip_base_offset + big_block_relative_offsets[big_block_idx];
                decode_color_block(unified_codebook[unified_idx], big_block_dst);
            }
        }
        zone_bitmap >>= 1;
        zone_idx++;
    }
}

IWRAM_CODE void decode_strip_i_frame_unified(int strip_idx, const u8* src, u16* dst)
{
    // 兼容性：检查是否包含大块索引码表
    // 简单的启发式检查：如果数据长度足够长，认为包含大块索引码表
    decode_strip_i_frame_unified_with_big_block(strip_idx, src, dst);
}

IWRAM_CODE void decode_strip_p_frame_unified(int strip_idx, const u8* src, u16* dst)
{
    // 使用新的大块索引支持的P帧解码
    decode_strip_p_frame_unified_with_big_block(strip_idx, src, dst);
}

// 函数声明
IWRAM_CODE void decode_strip(int strip_idx, const u8* src, u16 strip_data_size, u16* dst);

IWRAM_CODE void decode_strip(int strip_idx, const u8* src, u16 strip_data_size, u16* dst)
{
    u8 frame_type = *src++;
    
    if (frame_type == FRAME_TYPE_I) {
        decode_strip_i_frame_unified(strip_idx, src, dst);
    } else if (frame_type == FRAME_TYPE_P) {
        decode_strip_p_frame_unified(strip_idx, src, dst);
    }
}

IWRAM_CODE void decode_frame(const u8* frame_data, u16* dst)
{
    const u8* src = frame_data;
    
    for (int strip_idx = 0; strip_idx < VIDEO_STRIP_COUNT; strip_idx++) {
        u16 strip_data_size = src[0] | (src[1] << 8);
        src += 2;
        
        decode_strip(strip_idx, src, strip_data_size, dst);
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
    
    memset(ewramBuffer, 0, PIXELS_PER_FRAME * sizeof(u16));
    DMA3COPY(ewramBuffer, VRAM, PIXELS_PER_FRAME | DMA16);

    int frame = 0;
    
    while (1)
    {
        const unsigned char* frame_data = video_data + frame_offsets[frame];
        decode_frame(frame_data, ewramBuffer);
        
        // VBlankIntrWait(); // 注释掉以提高性能，让DMA自动等待
        DMA3COPY(ewramBuffer, VRAM, PIXELS_PER_FRAME | DMA16);

        frame++;
        if(frame >= VIDEO_FRAME_COUNT) {
            frame = 0;
        }
    }
}