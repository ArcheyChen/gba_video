// gba_video_player.cpp  v8
// Mode 3 单缓冲 + YUV9 → RGB555 + 条带帧间差分解码 + 统一码本向量量化 + 单条带模式

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
IWRAM_DATA static s8 bayer_bias_4_2x2[4][4] =
{
    {-2, 0, 1, -1},
    {-1, 0, 1, 0},
    {0, -1, 0,1},
    {0, -1, 1, -2}
};

struct YUV_Struct{
    u8 y[2][2];
    s8 d_r;    // 预计算的 Cr
    s8 d_g;    // 预计算的 (-(Cb>>1)-Cr)>>1
    s8 d_b;    // 预计算的 Cb
} __attribute__((packed));

// 统一码本存储（在IWRAM中）
IWRAM_DATA u8 unified_codebook_raw[UNIFIED_CODEBOOK_SIZE*sizeof(YUV_Struct)+4]__attribute__((aligned(32)));
IWRAM_DATA YUV_Struct *unified_codebook;

void init_table(){
    unified_codebook = (YUV_Struct*)unified_codebook_raw;
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

IWRAM_CODE inline u32 yuv_to_rgb555_2pix(u8 _y0, u8 _y1, s8 d_r, s8 d_g, s8 d_b, const s8* bayer_bias)
{
    s8 y0 = _y0 + bayer_bias[0];
    s8 y1 = _y1 + bayer_bias[1];
    u32 result = clip_lookup_table[y0 + d_r];
    result |= (clip_lookup_table[y0 + d_g] << 5);
    result |= (clip_lookup_table[y0 + d_b] << 10);

    result |= (clip_lookup_table[y1 + d_r] << 16);
    result |= (clip_lookup_table[y1 + d_g] << 21);
    result |= (clip_lookup_table[y1 + d_b] << 26);
    
    return result;
}

IWRAM_DATA u16 big_block_relative_offsets[240/4*160/4];
// zone内block相对偏移查找表
IWRAM_DATA u16 zone_block_relative_offsets[240];

void init_block_offsets(){
    u16 max_big_blocks_per_row = VIDEO_WIDTH / (BLOCK_WIDTH * 2);
    u16 max_big_blocks_per_col = VIDEO_HEIGHT / (BLOCK_HEIGHT * 2);

    for(int big_block_idx = 0; big_block_idx < max_big_blocks_per_row * max_big_blocks_per_col; big_block_idx++){
        u16 big_bx = big_block_idx % max_big_blocks_per_row;
        u16 big_by = big_block_idx / max_big_blocks_per_row;
        big_block_relative_offsets[big_block_idx] = (big_by * BLOCK_HEIGHT * 2 * SCREEN_WIDTH) + (big_bx * BLOCK_WIDTH * 2);
    }

    // 初始化zone内block相对偏移查找表
    u16 zone_big_blocks_w = VIDEO_WIDTH / (BLOCK_WIDTH * 2);  // 每个zone行内的大块数
    for(int zone_block_idx = 0; zone_block_idx < 240; zone_block_idx++){
        u16 zone_big_bx = zone_block_idx % zone_big_blocks_w;
        u16 zone_big_by = zone_block_idx / zone_big_blocks_w;
        zone_block_relative_offsets[zone_block_idx] = 
            (zone_big_by * BLOCK_HEIGHT * 2 * SCREEN_WIDTH) + (zone_big_bx * BLOCK_WIDTH * 2);
    }
}

// 解码单个2x2块
IWRAM_CODE inline void decode_block(const YUV_Struct &yuv_data, u16* dst, u8 bayer_idx)
{
    const s8 &d_r = yuv_data.d_r;
    const s8 &d_g = yuv_data.d_g;
    const s8 &d_b = yuv_data.d_b;

    u32* dst_row = (u32*)dst;
    auto const &y = yuv_data.y;
    *dst_row = yuv_to_rgb555_2pix(
        y[0][0],y[0][1], d_r, d_g, d_b, bayer_bias_4_2x2[bayer_idx]);
    *(dst_row + SCREEN_WIDTH/2) = yuv_to_rgb555_2pix(
        y[1][0],y[1][1], d_r, d_g, d_b,bayer_bias_4_2x2[bayer_idx]+2);
}

// 解码色块（2x2上采样到4x4）
IWRAM_CODE inline void decode_color_block(const YUV_Struct &yuv_data, u16* dst)
{
    const s8 &d_r = yuv_data.d_r;
    const s8 &d_g = yuv_data.d_g;
    const s8 &d_b = yuv_data.d_b;
    auto &y = yuv_data.y;
    
    u32* dst_row = (u32*)dst;
    
    // 第一行：1122
    *dst_row = yuv_to_rgb555_2pix(y[0][0],y[0][0], d_r, d_g, d_b,bayer_bias_4_2x2[0]);;
    *(dst_row + 1) = yuv_to_rgb555_2pix(y[0][1],y[0][1], d_r, d_g, d_b,bayer_bias_4_2x2[1]);
    
    // 第二行：1122
    dst_row += SCREEN_WIDTH/2;
    *dst_row = yuv_to_rgb555_2pix(y[0][0],y[0][0], d_r, d_g, d_b,bayer_bias_4_2x2[2]);;
    *(dst_row + 1) = yuv_to_rgb555_2pix(y[0][1],y[0][1], d_r, d_g, d_b,bayer_bias_4_2x2[3]);
    
    // 第三行：3344
    dst_row += SCREEN_WIDTH/2;
    *dst_row = yuv_to_rgb555_2pix(y[1][0],y[1][0], d_r, d_g, d_b,bayer_bias_4_2x2[0]);
    *(dst_row + 1) = yuv_to_rgb555_2pix(y[1][1],y[1][1], d_r, d_g, d_b,bayer_bias_4_2x2[1]);
    
    // 第四行：3344
    dst_row += SCREEN_WIDTH/2;
    *dst_row = yuv_to_rgb555_2pix(y[1][0],y[1][0], d_r, d_g, d_b,bayer_bias_4_2x2[2]);
    *(dst_row + 1) = yuv_to_rgb555_2pix(y[1][1],y[1][1], d_r, d_g, d_b,bayer_bias_4_2x2[3]);
}

// 通用的4x4大块解码函数（纹理块）
IWRAM_CODE void decode_big_block(const YUV_Struct* codebook, const u8 quant_indices[4], u16* big_block_dst)
{
    decode_block(codebook[quant_indices[0]], big_block_dst ,0);
    decode_block(codebook[quant_indices[1]], big_block_dst + 2,1);
    decode_block(codebook[quant_indices[2]], big_block_dst + SCREEN_WIDTH * 2,2);
    decode_block(codebook[quant_indices[3]], big_block_dst + SCREEN_WIDTH * 2 + 2,3);
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

bool code_book_preloaded = false;
IWRAM_CODE void preload_codebook(const u8* src)
{
    u8 frame_type = src[0];
    if(frame_type != FRAME_TYPE_I || code_book_preloaded) {
        VBlankIntrWait(); // 等待VBlank，防止CPU空跑跑满
        return; // 只在I帧中预加载码本
    }
    copy_unified_codebook(unified_codebook_raw, src + 1, &unified_codebook, UNIFIED_CODEBOOK_SIZE);
    code_book_preloaded = true;
}

IWRAM_CODE void decode_i_frame_unified(const u8* src, u16* dst)
{
    // 拷贝统一码本
    if(!code_book_preloaded){
        copy_unified_codebook(unified_codebook_raw, src, 
                         &unified_codebook, UNIFIED_CODEBOOK_SIZE);
    }
    code_book_preloaded = false;//清空标记，不然下一次的预载可能就不管用了
    src += UNIFIED_CODEBOOK_SIZE * BYTES_PER_BLOCK;
    
    u16 tot_big_blocks = (VIDEO_WIDTH / (BLOCK_WIDTH * 2)) * (VIDEO_HEIGHT / (BLOCK_HEIGHT * 2));
    
    // 解码所有4x4大块
    for (int big_block_idx = 0; big_block_idx < tot_big_blocks; big_block_idx++) {
        u16* big_block_dst = dst + big_block_relative_offsets[big_block_idx];
        
        // 读取第一个索引/标记
        u8 first_byte = *src++;
        
        if (first_byte == COLOR_BLOCK_MARKER) {
            // 色块：读取统一码本索引
            u8 unified_idx = *src++;
            decode_color_block(unified_codebook[unified_idx], big_block_dst);
        } else {
            // 纹理块：当前字节是第一个统一码本索引，继续读取3个
            u8 quant_indices[4];
            quant_indices[0] = first_byte;
            quant_indices[1] = *src++;
            quant_indices[2] = *src++;
            quant_indices[3] = *src++;
            decode_big_block(unified_codebook, quant_indices, big_block_dst);
        }
    }
}

IWRAM_CODE void decode_p_frame_unified(const u8* src, u16* dst)
{
    // 读取区域bitmap
    u16 zone_bitmap = src[0] | (src[1] << 8);
    src += 2; // 跳过bitmap的两个字节
    
    // 使用bitmap右移优化处理每个有效区域
    u8 zone_idx = 0;
    while (zone_bitmap) {
        if (zone_bitmap & 1) {
            // 读取两种类型的更新数量
            u8 detail_blocks_to_update = *src++;
            u8 color_blocks_to_update = *src++;
            
            // 计算zone在整个屏幕中的基址偏移
            u16 zone_base_offset = zone_idx * ZONE_HEIGHT_PIXELS * SCREEN_WIDTH;
            u16* zone_dst = dst + zone_base_offset;
            
            // 处理纹理块更新（4个索引）
            for (u8 i = 0; i < detail_blocks_to_update; i++) {
                u8 zone_relative_idx = *src++;
                
                u8 quant_indices[4];
                quant_indices[0] = *src++;
                quant_indices[1] = *src++;
                quant_indices[2] = *src++;
                quant_indices[3] = *src++;
                
                // 直接使用查找表获取相对偏移
                u16* big_block_dst = zone_dst + zone_block_relative_offsets[zone_relative_idx];
                decode_big_block(unified_codebook, quant_indices, big_block_dst);
            }
            
            // 处理色块更新（1个统一码本索引）
            for (u8 i = 0; i < color_blocks_to_update; i++) {
                u8 zone_relative_idx = *src++;
                u8 unified_idx = *src++;
                
                // 直接使用查找表获取相对偏移
                u16* big_block_dst = zone_dst + zone_block_relative_offsets[zone_relative_idx];
                decode_color_block(unified_codebook[unified_idx], big_block_dst);
            }
        }
        zone_bitmap >>= 1;
        zone_idx++;
    }
}

IWRAM_CODE void decode_frame(const u8* frame_data, u16* dst)
{
    u8 frame_type = *frame_data++;
    
    if (frame_type == FRAME_TYPE_I) {
        decode_i_frame_unified(frame_data, dst);
    } else if (frame_type == FRAME_TYPE_P) {
        decode_p_frame_unified(frame_data, dst);
    }
}

static volatile u32 vbl = 0;
static volatile u32 acc = 0;
static volatile bool should_copy = false;
IWRAM_CODE void isr_vbl() { 
    ++vbl; 
    acc += VIDEO_FPS;  // 使用头文件中定义的FPS
    if(acc >= 60) {
        should_copy = true;
        acc -= 60;
    }
    REG_IF = IRQ_VBLANK; 
}

int main()
{
    REG_DISPCNT = MODE_3 | BG2_ENABLE;

    irqInit();
    irqSet(IRQ_VBLANK, isr_vbl);
    irqEnable(IRQ_VBLANK);

    init_table();
    init_block_offsets();
    
    memset(ewramBuffer, 0, PIXELS_PER_FRAME * sizeof(u16));
    DMA3COPY(ewramBuffer, VRAM, PIXELS_PER_FRAME | DMA16);

    int frame = 0;
    
    while (1)
    {
        const unsigned char* frame_data = video_data + frame_offsets[frame];
        decode_frame(frame_data, ewramBuffer);
        
        // VBlankIntrWait(); // 注释掉以提高性能，让DMA自动等待
        while(!should_copy) {
            preload_codebook(frame_data);
            // 预加载码本，以消耗掉空闲时间
            // 这个函数里面如果没事做，会等待VBlank，因此不用担心跑满CPU
        }
        should_copy = false;
        DMA3COPY(ewramBuffer, VRAM, (PIXELS_PER_FRAME>>1) | DMA32);

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