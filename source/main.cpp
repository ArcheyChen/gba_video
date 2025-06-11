// gba_video_player.cpp  v7
// Mode 3 单缓冲 + YUV9 → RGB555 + 条带帧间差分解码 + 统一码本向量量化

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
#define ZONE_HEIGHT_BIG_BLOCKS (ZONE_HEIGHT_PIXELS / 4)  // 每个区域的4x4大块行数 (16像素/4 = 4行)
#define BIG_BLOCK_MARKER 0xFE  // 4x4大块标记
#define BYTES_PER_BIG_BLOCK 28  // 16Y + 4*(d_r + d_g + d_b)

// EWRAM 单缓冲
EWRAM_BSS u16 ewramBuffer[PIXELS_PER_FRAME];
IWRAM_DATA static u8 clip_table_raw[512];
u8* clip_lookup_table = clip_table_raw + 128;
IWRAM_DATA static s8 bayer_bias_4_2x2[4][4] =
//  {};
{
    {-2, 0, 1, -1},
    {-1, 0, 1, 0},
    {0, -1, 0,1},
    {0, -1, 1, -2}
};
// IWRAM_DATA const  s8 bayer_bias1d[16] = {
//         -4,  0, -3,  1,
//          2, -2,  3, -1,
//         -3,  1, -4,  0,
//          4,  0,  3, -2
//     };

struct YUV_Struct{
    u8 y[2][2];
    s8 d_r;    // 预计算的 Cr
    s8 d_g;    // 预计算的 (-(Cb>>1)-Cr)>>1
    s8 d_b;    // 预计算的 Cb
} __attribute__((packed));

struct BigYUV_Struct{
    u8 y[16];     // 16个Y值
    s8 d_r[4];    // 4个d_r值
    s8 d_g[4];    // 4个d_g值  
    s8 d_b[4];    // 4个d_b值
} __attribute__((packed));


EWRAM_DATA u8 strip_big_block_codebooks_raw[VIDEO_STRIP_COUNT][BIG_BLOCK_CODEBOOK_SIZE*sizeof(BigYUV_Struct)+4]__attribute__((aligned(32)));
EWRAM_DATA BigYUV_Struct *strip_big_block_codebooks[VIDEO_STRIP_COUNT];

EWRAM_DATA u8 strip_small_block_codebooks_raw[VIDEO_STRIP_COUNT][SMALL_BLOCK_CODEBOOK_SIZE*sizeof(YUV_Struct)+4]__attribute__((aligned(32)));
EWRAM_DATA YUV_Struct *strip_small_block_codebooks[VIDEO_STRIP_COUNT];

void init_table(){
    for(int i = 0; i < VIDEO_STRIP_COUNT; i++) {
        // strip_unified_codebooks[i] = (YUV_Struct*)strip_unified_codebooks_raw[i];
        strip_big_block_codebooks[i] = (BigYUV_Struct*)strip_big_block_codebooks_raw[i];
        strip_small_block_codebooks[i] = (YUV_Struct*)strip_small_block_codebooks_raw[i];
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

IWRAM_CODE inline u32 yuv_to_rgb555_2pix(const u8 y[2], s8 d_r, s8 d_g, s8 d_b, const s8* bayer_bias)
{
    s16 _y = y[0] + bayer_bias[0];
    u32 result = clip_lookup_table[_y + d_r];
    result |= (clip_lookup_table[_y + d_g] << 5);
    result |= (clip_lookup_table[_y + d_b] << 10);

    _y = y[1] + bayer_bias[1];
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

    // 计算最大的4x4大块数量，用于预分配偏移数组
    u16 max_big_blocks_per_row = VIDEO_WIDTH / (BLOCK_WIDTH * 2);  // 240/4 = 60
    u16 max_big_blocks_per_col = 0;

    // 找到最大的条带4x4大块列数
    for(int strip_idx = 0; strip_idx < VIDEO_STRIP_COUNT; strip_idx++){
        u16 strip_big_blocks_per_col = strip_heights[strip_idx] / (BLOCK_HEIGHT * 2);  // 高度/4
        if(strip_big_blocks_per_col > max_big_blocks_per_col) {
            max_big_blocks_per_col = strip_big_blocks_per_col;
        }
    }

    // 预计算4x4大块的相对偏移（相对于条带起始位置）
    for(int big_block_idx = 0; big_block_idx < max_big_blocks_per_row * max_big_blocks_per_col; big_block_idx++){
        u16 big_bx = big_block_idx % max_big_blocks_per_row;
        u16 big_by = big_block_idx / max_big_blocks_per_row;
        // 4x4大块 = 4x4像素，所以偏移是 big_by*4*屏幕宽度 + big_bx*4
        big_block_relative_offsets[big_block_idx] = (big_by * 4 * SCREEN_WIDTH) + (big_bx * 4);
    }

    // 初始化每个条带的信息
    for(int strip_idx = 0; strip_idx < VIDEO_STRIP_COUNT; strip_idx++){
        strip_info[strip_idx].start_y = current_y;
        strip_info[strip_idx].height = strip_heights[strip_idx];
        strip_info[strip_idx].blocks_per_row = VIDEO_WIDTH / BLOCK_WIDTH;  // 2x2块每行数量
        strip_info[strip_idx].blocks_per_col = strip_heights[strip_idx] / BLOCK_HEIGHT;  // 2x2块每列数量
        strip_info[strip_idx].total_blocks = strip_info[strip_idx].blocks_per_row * strip_info[strip_idx].blocks_per_col;
        strip_info[strip_idx].buffer_offset = current_y * SCREEN_WIDTH;  // 像素偏移
        current_y += strip_heights[strip_idx];
    }
}



// 解码单个2x2块 - 简化Bayer索引计算
IWRAM_CODE inline void decode_block(const YUV_Struct &yuv_data, u16* dst, u16 screen_y, u16 screen_x)
{
    const s8 &d_r = yuv_data.d_r;
    const s8 &d_g = yuv_data.d_g;
    const s8 &d_b = yuv_data.d_b;

    u32* dst_row = (u32*)dst;
    
    // 简化的Bayer索引计算
    u8 bayer_y0 = screen_y & 3;
    u8 bayer_y1 = (screen_y + 1) & 3;
    u8 bayer_x0 = screen_x & 3;
    u8 bayer_x1 = (screen_x + 1) & 3;
    
    // 第一行两个像素
    u8 y_pair_0[2] = {yuv_data.y[0][0], yuv_data.y[0][1]};
    s8 bayer_bias_0[2] = {
        bayer_bias_4_2x2[bayer_y0][bayer_x0],
        bayer_bias_4_2x2[bayer_y0][bayer_x1]
    };
    *dst_row = yuv_to_rgb555_2pix(y_pair_0, d_r, d_g, d_b, bayer_bias_0);
    
    // 第二行两个像素
    u8 y_pair_1[2] = {yuv_data.y[1][0], yuv_data.y[1][1]};
    s8 bayer_bias_1[2] = {
        bayer_bias_4_2x2[bayer_y1][bayer_x0],
        bayer_bias_4_2x2[bayer_y1][bayer_x1]
    };
    *(dst_row + SCREEN_WIDTH/2) = yuv_to_rgb555_2pix(y_pair_1, d_r, d_g, d_b, bayer_bias_1);
}

// 解码色块（2x2上采样到4x4） - 简化Bayer索引
IWRAM_CODE inline void decode_color_block(const YUV_Struct &yuv_data, u16* dst, u16 screen_y, u16 screen_x)
{
    const s8 &d_r = yuv_data.d_r;
    const s8 &d_g = yuv_data.d_g;
    const s8 &d_b = yuv_data.d_b;

    // 为每个重复的Y值对创建数组
    u8 y_11[2] = {yuv_data.y[0][0], yuv_data.y[0][0]}; // 1,1
    u8 y_22[2] = {yuv_data.y[0][1], yuv_data.y[0][1]}; // 2,2
    u8 y_33[2] = {yuv_data.y[1][0], yuv_data.y[1][0]}; // 3,3
    u8 y_44[2] = {yuv_data.y[1][1], yuv_data.y[1][1]}; // 4,4
    
    u32* dst_row = (u32*)dst;
    
    // 为4x4块的每一行计算正确的Bayer索引
    for(int row = 0; row < 4; row++) {
        u8 current_y = (screen_y + row) & 3;
        u8 x0 = screen_x & 3;
        u8 x1 = (screen_x + 1) & 3;
        u8 x2 = (screen_x + 2) & 3;
        u8 x3 = (screen_x + 3) & 3;
        
        if(row < 2) {
            // 前两行使用y_11和y_22
            s8 bayer_bias_left[2] = {
                bayer_bias_4_2x2[current_y][x0],
                bayer_bias_4_2x2[current_y][x1]
            };
            s8 bayer_bias_right[2] = {
                bayer_bias_4_2x2[current_y][x2],
                bayer_bias_4_2x2[current_y][x3]
            };
            *dst_row = yuv_to_rgb555_2pix(y_11, d_r, d_g, d_b, bayer_bias_left);
            *(dst_row + 1) = yuv_to_rgb555_2pix(y_22, d_r, d_g, d_b, bayer_bias_right);
        } else {
            // 后两行使用y_33和y_44
            s8 bayer_bias_left[2] = {
                bayer_bias_4_2x2[current_y][x0],
                bayer_bias_4_2x2[current_y][x1]
            };
            s8 bayer_bias_right[2] = {
                bayer_bias_4_2x2[current_y][x2],
                bayer_bias_4_2x2[current_y][x3]
            };
            *dst_row = yuv_to_rgb555_2pix(y_33, d_r, d_g, d_b, bayer_bias_left);
            *(dst_row + 1) = yuv_to_rgb555_2pix(y_44, d_r, d_g, d_b, bayer_bias_right);
        }
        
        dst_row += SCREEN_WIDTH/2;
    }
}

// 通用的4x4大块解码函数（纹理块） - 修复Bayer索引
IWRAM_CODE void decode_big_block(const YUV_Struct* codebook, const u8 quant_indices[4], u16* big_block_dst, u16 screen_y, u16 screen_x)
{
    decode_block(codebook[quant_indices[0]], big_block_dst, screen_y, screen_x);
    decode_block(codebook[quant_indices[1]], big_block_dst + 2, screen_y, screen_x + 2);
    decode_block(codebook[quant_indices[2]], big_block_dst + SCREEN_WIDTH * 2, screen_y + 2, screen_x);
    decode_block(codebook[quant_indices[3]], big_block_dst + SCREEN_WIDTH * 2 + 2, screen_y + 2, screen_x + 2);
}

// 简化的码本拷贝函数 - 使用memcpy
IWRAM_CODE void copy_unified_codebook_simple(u8* dst_raw, const u8* src, YUV_Struct** codebook_ptr, int codebook_size)
{
    *codebook_ptr = (YUV_Struct*)(dst_raw + 4); // 跳过对齐填充
    int copy_size = codebook_size * BYTES_PER_BLOCK;
    memcpy(*codebook_ptr, src, copy_size);
}

// 简化的大块码本拷贝函数 - 使用memcpy
IWRAM_CODE void copy_big_block_codebook_simple(u8* dst_raw, const u8* src, BigYUV_Struct** codebook_ptr, int codebook_size)
{
    *codebook_ptr = (BigYUV_Struct*)(dst_raw + 4); // 跳过对齐填充
    int copy_size = codebook_size * BYTES_PER_BIG_BLOCK;
    memcpy(*codebook_ptr, src, copy_size);
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

// DMA拷贝大块码本的辅助函数
IWRAM_CODE void copy_big_block_codebook(u8* dst_raw, const u8* src, BigYUV_Struct** codebook_ptr, int codebook_size)
{
    u8* copy_raw_ptr = dst_raw + 4; // 跳过对齐填充的4字节
    *codebook_ptr = (BigYUV_Struct*)copy_raw_ptr;
    int remain_copy = codebook_size * BYTES_PER_BIG_BLOCK;
    
    if(((u32)src) & 1){
        u8 data = *src++;
        copy_raw_ptr[-1] = copy_raw_ptr[-3] = data;
        *codebook_ptr = (BigYUV_Struct*)((u32)*codebook_ptr - 1);
        remain_copy -= 1;
    }
    if(((u32)src) & 2){
        copy_raw_ptr[-2] = *src++;
        copy_raw_ptr[-1] = *src++;
        *codebook_ptr = (BigYUV_Struct*)((u32)*codebook_ptr - 2);
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

// 移除旧的统一码表I帧解码函数
// IWRAM_CODE void decode_strip_i_frame_unified(int strip_idx, const u8* src, u16* dst)

// 移除旧的统一码表P帧解码函数  
// IWRAM_CODE void decode_strip_p_frame_unified(int strip_idx, const u8* src, u16* dst)

// 解码4x4大块 - 修复Bayer索引
IWRAM_CODE void decode_big_block_direct(const BigYUV_Struct &big_yuv_data, u16* big_block_dst, u16 screen_y, u16 screen_x)
{
    // 直接从4x4大块数据解码，每个位置使用对应的Y值和色度组
    for(int sub_by = 0; sub_by < 2; sub_by++) {
        for(int sub_bx = 0; sub_bx < 2; sub_bx++) {
            int chroma_idx = sub_by * 2 + sub_bx;  // 0,1,2,3对应4个色度组
            
            // 提取当前2x2区域的Y值和色度
            u8 y_block[2][2];
            for(int dy = 0; dy < 2; dy++) {
                for(int dx = 0; dx < 2; dx++) {
                    int y_idx = (sub_by * 2 + dy) * 4 + (sub_bx * 2 + dx);
                    y_block[dy][dx] = big_yuv_data.y[y_idx];
                }
            }
            
            s8 d_r = big_yuv_data.d_r[chroma_idx];
            s8 d_g = big_yuv_data.d_g[chroma_idx];
            s8 d_b = big_yuv_data.d_b[chroma_idx];
            
            // 计算目标位置和屏幕坐标
            u16* block_dst = big_block_dst + (sub_by * SCREEN_WIDTH * 2) + (sub_bx * 2);
            u16 block_screen_y = screen_y + sub_by * 2;
            u16 block_screen_x = screen_x + sub_bx * 2;
            
            // 临时创建YUV结构用于解码
            YUV_Struct temp_yuv;
            temp_yuv.y[0][0] = y_block[0][0];
            temp_yuv.y[0][1] = y_block[0][1];
            temp_yuv.y[1][0] = y_block[1][0];
            temp_yuv.y[1][1] = y_block[1][1];
            temp_yuv.d_r = d_r;
            temp_yuv.d_g = d_g;
            temp_yuv.d_b = d_b;
            
            decode_block(temp_yuv, block_dst, block_screen_y, block_screen_x);
        }
    }
}

// 临时禁用Bayer抖动的解码函数
IWRAM_CODE inline void decode_block_no_dither(const YUV_Struct &yuv_data, u16* dst)
{
    const s8 &d_r = yuv_data.d_r;
    const s8 &d_g = yuv_data.d_g;
    const s8 &d_b = yuv_data.d_b;

    u32* dst_row = (u32*)dst;
    
    // 不使用Bayer抖动，偏置为0
    s8 zero_bias[2] = {0, 0};
    
    // 第一行两个像素
    u8 y_pair_0[2] = {yuv_data.y[0][0], yuv_data.y[0][1]};
    *dst_row = yuv_to_rgb555_2pix(y_pair_0, d_r, d_g, d_b, zero_bias);
    
    // 第二行两个像素
    u8 y_pair_1[2] = {yuv_data.y[1][0], yuv_data.y[1][1]};
    *(dst_row + SCREEN_WIDTH/2) = yuv_to_rgb555_2pix(y_pair_1, d_r, d_g, d_b, zero_bias);
}

// 临时禁用Bayer抖动的色块解码
IWRAM_CODE inline void decode_color_block_no_dither(const YUV_Struct &yuv_data, u16* dst)
{
    const s8 &d_r = yuv_data.d_r;
    const s8 &d_g = yuv_data.d_g;
    const s8 &d_b = yuv_data.d_b;

    // 为每个重复的Y值对创建数组
    u8 y_11[2] = {yuv_data.y[0][0], yuv_data.y[0][0]}; // 1,1
    u8 y_22[2] = {yuv_data.y[0][1], yuv_data.y[0][1]}; // 2,2
    u8 y_33[2] = {yuv_data.y[1][0], yuv_data.y[1][0]}; // 3,3
    u8 y_44[2] = {yuv_data.y[1][1], yuv_data.y[1][1]}; // 4,4
    
    u32* dst_row = (u32*)dst;
    s8 zero_bias[2] = {0, 0};
    
    // 第一行：1122
    *dst_row = yuv_to_rgb555_2pix(y_11, d_r, d_g, d_b, zero_bias);
    *(dst_row + 1) = yuv_to_rgb555_2pix(y_22, d_r, d_g, d_b, zero_bias);
    
    // 第二行：1122
    dst_row += SCREEN_WIDTH/2;
    *dst_row = yuv_to_rgb555_2pix(y_11, d_r, d_g, d_b, zero_bias);
    *(dst_row + 1) = yuv_to_rgb555_2pix(y_22, d_r, d_g, d_b, zero_bias);
    
    // 第三行：3344
    dst_row += SCREEN_WIDTH/2;
    *dst_row = yuv_to_rgb555_2pix(y_33, d_r, d_g, d_b, zero_bias);
    *(dst_row + 1) = yuv_to_rgb555_2pix(y_44, d_r, d_g, d_b, zero_bias);
    
    // 第四行：3344
    dst_row += SCREEN_WIDTH/2;
    *dst_row = yuv_to_rgb555_2pix(y_33, d_r, d_g, d_b, zero_bias);
    *(dst_row + 1) = yuv_to_rgb555_2pix(y_44, d_r, d_g, d_b, zero_bias);
}

// 临时禁用Bayer抖动的大块解码
IWRAM_CODE void decode_big_block_no_dither(const YUV_Struct* codebook, const u8 quant_indices[4], u16* big_block_dst)
{
    decode_block_no_dither(codebook[quant_indices[0]], big_block_dst);
    decode_block_no_dither(codebook[quant_indices[1]], big_block_dst + 2);
    decode_block_no_dither(codebook[quant_indices[2]], big_block_dst + SCREEN_WIDTH * 2);
    decode_block_no_dither(codebook[quant_indices[3]], big_block_dst + SCREEN_WIDTH * 2 + 2);
}

// 临时禁用Bayer抖动的直接大块解码
IWRAM_CODE void decode_big_block_direct_no_dither(const BigYUV_Struct &big_yuv_data, u16* big_block_dst)
{
    // 直接从4x4大块数据解码，每个位置使用对应的Y值和色度组
    for(int sub_by = 0; sub_by < 2; sub_by++) {
        for(int sub_bx = 0; sub_bx < 2; sub_bx++) {
            int chroma_idx = sub_by * 2 + sub_bx;  // 0,1,2,3对应4个色度组
            
            // 提取当前2x2区域的Y值和色度
            u8 y_block[2][2];
            for(int dy = 0; dy < 2; dy++) {
                for(int dx = 0; dx < 2; dx++) {
                    int y_idx = (sub_by * 2 + dy) * 4 + (sub_bx * 2 + dx);
                    y_block[dy][dx] = big_yuv_data.y[y_idx];
                }
            }
            
            s8 d_r = big_yuv_data.d_r[chroma_idx];
            s8 d_g = big_yuv_data.d_g[chroma_idx];
            s8 d_b = big_yuv_data.d_b[chroma_idx];
            
            // 计算目标位置
            u16* block_dst = big_block_dst + (sub_by * SCREEN_WIDTH * 2) + (sub_bx * 2);
            
            // 临时创建YUV结构用于解码
            YUV_Struct temp_yuv;
            temp_yuv.y[0][0] = y_block[0][0];
            temp_yuv.y[0][1] = y_block[0][1];
            temp_yuv.y[1][0] = y_block[1][0];
            temp_yuv.y[1][1] = y_block[1][1];
            temp_yuv.d_r = d_r;
            temp_yuv.d_g = d_g;
            temp_yuv.d_b = d_b;
            
            decode_block_no_dither(temp_yuv, block_dst);
        }
    }
}

IWRAM_CODE void decode_strip_i_frame_with_big_blocks(int strip_idx, const u8* src, u16* dst)
{
    u16 strip_base_offset = strip_info[strip_idx].buffer_offset;
    u16 strip_start_y = strip_info[strip_idx].start_y;  // 添加条带起始Y坐标
    
    // 使用简化的拷贝函数
    copy_big_block_codebook_simple(strip_big_block_codebooks_raw[strip_idx], src, 
                                  &strip_big_block_codebooks[strip_idx], BIG_BLOCK_CODEBOOK_SIZE);
    src += BIG_BLOCK_CODEBOOK_SIZE * BYTES_PER_BIG_BLOCK;
    
    copy_unified_codebook_simple(strip_small_block_codebooks_raw[strip_idx], src, 
                                &strip_small_block_codebooks[strip_idx], SMALL_BLOCK_CODEBOOK_SIZE);
    src += SMALL_BLOCK_CODEBOOK_SIZE * BYTES_PER_BLOCK;
    
    auto &strip = strip_info[strip_idx];
    auto &big_block_codebook = strip_big_block_codebooks[strip_idx];
    auto &small_block_codebook = strip_small_block_codebooks[strip_idx];
    
    // 计算当前条带的4x4大块数量
    u16 strip_big_blocks_w = VIDEO_WIDTH / 4;  // 每行4x4大块数量 = 240/4 = 60
    u16 strip_big_blocks_h = strip.height / 4;  // 每列4x4大块数量 = 条带高度/4
    u16 tot_big_blocks = strip_big_blocks_w * strip_big_blocks_h;
    
    u16* strip_dst = dst + strip_base_offset;
    
    // 解码所有4x4大块
    for (int big_block_idx = 0; big_block_idx < tot_big_blocks; big_block_idx++) {
        // 计算4x4大块在当前条带内的偏移
        u16 big_bx = big_block_idx % strip_big_blocks_w;
        u16 big_by = big_block_idx / strip_big_blocks_w;
        u16 block_offset = big_by * 4 * SCREEN_WIDTH + big_bx * 4;
        u16* big_block_dst = strip_dst + block_offset;
        
        // 读取第一个索引/标记
        u8 first_byte = *src++;
        
        if (first_byte == COLOR_BLOCK_MARKER) {
            // 色块：读取小块码表索引
            u8 small_idx = *src++;
            decode_color_block_no_dither(small_block_codebook[small_idx], big_block_dst);
        } else if (first_byte == BIG_BLOCK_MARKER) {
            // 4x4大块：读取大块码表索引
            u8 big_idx = *src++;
            decode_big_block_direct_no_dither(big_block_codebook[big_idx], big_block_dst);
        } else {
            // 纹理块：当前字节是第一个小块码表索引，继续读取3个
            u8 quant_indices[4];
            quant_indices[0] = first_byte;
            quant_indices[1] = *src++;
            quant_indices[2] = *src++;
            quant_indices[3] = *src++;
            decode_big_block_no_dither(small_block_codebook, quant_indices, big_block_dst);
        }
    }
}

IWRAM_CODE void decode_strip_p_frame_with_big_blocks(int strip_idx, const u8* src, u16* dst)
{
    u16 strip_base_offset = strip_info[strip_idx].buffer_offset;
    u16 strip_start_y = strip_info[strip_idx].start_y;  // 添加条带起始Y坐标
    
    // 读取区域bitmap
    u8 zone_bitmap = *src++;
    
    auto &big_block_codebook = strip_big_block_codebooks[strip_idx];
    auto &small_block_codebook = strip_small_block_codebooks[strip_idx];
    
    // 计算该条带的大块布局
    u16 strip_big_blocks_w = VIDEO_WIDTH / 4;  // 每行4x4大块数量 = 240/4 = 60
    u16 strip_big_blocks_h = strip_info[strip_idx].height / 4;  // 每列4x4大块数量
    
    // 使用bitmap右移优化处理每个有效区域
    u8 zone_idx = 0;
    while (zone_bitmap) {
        if (zone_bitmap & 1) {
            // 读取三种类型的更新数量
            u8 detail_blocks_to_update = *src++;
            u8 color_blocks_to_update = *src++;
            u8 big_blocks_to_update = *src++;
            
            // 计算区域在条带中的起始大块行 - 直接用4x4大块坐标
            u16 zone_start_big_by = zone_idx * ZONE_HEIGHT_BIG_BLOCKS;
            
            // 处理纹理块更新（4个索引）
            for (u8 i = 0; i < detail_blocks_to_update; i++) {
                u8 zone_relative_idx = *src++;
                
                u8 quant_indices[4];
                quant_indices[0] = *src++;
                quant_indices[1] = *src++;
                quant_indices[2] = *src++;
                quant_indices[3] = *src++;
                
                // 将区域相对坐标转换为条带内的绝对坐标（4x4大块坐标）
                u16 relative_big_by = zone_relative_idx / strip_big_blocks_w;
                u16 relative_big_bx = zone_relative_idx % strip_big_blocks_w;
                u16 absolute_big_by = zone_start_big_by + relative_big_by;
                
                // 确保不越界
                if (absolute_big_by < strip_big_blocks_h && relative_big_bx < strip_big_blocks_w) {
                    // 计算在条带内的像素偏移
                    u16 block_offset = absolute_big_by * 4 * SCREEN_WIDTH + relative_big_bx * 4;
                    u16* big_block_dst = dst + strip_base_offset + block_offset;
                    decode_big_block_no_dither(small_block_codebook, quant_indices, big_block_dst);
                }
            }
            
            // 处理色块更新（1个小块码表索引）
            for (u8 i = 0; i < color_blocks_to_update; i++) {
                u8 zone_relative_idx = *src++;
                u8 small_idx = *src++;
                
                // 将区域相对坐标转换为条带内的绝对坐标（4x4大块坐标）
                u16 relative_big_by = zone_relative_idx / strip_big_blocks_w;
                u16 relative_big_bx = zone_relative_idx % strip_big_blocks_w;
                u16 absolute_big_by = zone_start_big_by + relative_big_by;
                
                // 确保不越界
                if (absolute_big_by < strip_big_blocks_h && relative_big_bx < strip_big_blocks_w) {
                    // 计算在条带内的像素偏移
                    u16 block_offset = absolute_big_by * 4 * SCREEN_WIDTH + relative_big_bx * 4;
                    u16* big_block_dst = dst + strip_base_offset + block_offset;
                    decode_color_block_no_dither(small_block_codebook[small_idx], big_block_dst);
                }
            }
            
            // 处理4x4大块更新（1个大块码表索引）
            for (u8 i = 0; i < big_blocks_to_update; i++) {
                u8 zone_relative_idx = *src++;
                u8 big_idx = *src++;
                
                // 将区域相对坐标转换为条带内的绝对坐标（4x4大块坐标）
                u16 relative_big_by = zone_relative_idx / strip_big_blocks_w;
                u16 relative_big_bx = zone_relative_idx % strip_big_blocks_w;
                u16 absolute_big_by = zone_start_big_by + relative_big_by;
                
                // 确保不越界
                if (absolute_big_by < strip_big_blocks_h && relative_big_bx < strip_big_blocks_w) {
                    // 计算在条带内的像素偏移
                    u16 block_offset = absolute_big_by * 4 * SCREEN_WIDTH + relative_big_bx * 4;
                    u16* big_block_dst = dst + strip_base_offset + block_offset;
                    decode_big_block_direct_no_dither(big_block_codebook[big_idx], big_block_dst);
                }
            }
        }
        zone_bitmap >>= 1;
        zone_idx++;
    }
}

IWRAM_CODE void decode_strip(int strip_idx, const u8* src, u16 strip_data_size, u16* dst)
{
    u8 frame_type = *src++;
    
    if (frame_type == FRAME_TYPE_I) {
        decode_strip_i_frame_with_big_blocks(strip_idx, src, dst);
    } else if (frame_type == FRAME_TYPE_P) {
        decode_strip_p_frame_with_big_blocks(strip_idx, src, dst);
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