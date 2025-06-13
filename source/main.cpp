// gba_video_player.cpp  v8
// Mode 3 单缓冲 + YUV9 → RGB555 + 条带帧间差分解码 + 8x8超级块编码

#include <gba_systemcalls.h>
#include <gba_video.h>
#include <gba_dma.h>
#include <gba_interrupt.h>
#include <gba_input.h>
#include <cstring>

#include "video_data.h"
#include <tuple>

constexpr int PIXELS_PER_FRAME = SCREEN_WIDTH * SCREEN_HEIGHT;

// 新增常量定义 - 8x8超级块
#define ZONE_HEIGHT_PIXELS 16  // 每个区域的像素高度
#define ZONE_HEIGHT_SUPER_BLOCKS (ZONE_HEIGHT_PIXELS / SUPER_BLOCK_SIZE)  // 每个区域的8x8超级块行数 (16像素/8 = 2行)
#define BYTES_PER_4X4_BLOCK 28  // 16Y + 4*(d_r + d_g + d_b)

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

struct Block4x4_Struct{
    YUV_Struct blocks_2x2[4];  // 4个2x2块，每个块包含2x2的Y值和色度差
} __attribute__((packed));

IWRAM_DATA u8 strip_4x4_codebooks_raw[VIDEO_STRIP_COUNT][CODEBOOK_4X4_SIZE*sizeof(Block4x4_Struct)+4]__attribute__((aligned(32)));
IWRAM_DATA Block4x4_Struct *strip_4x4_codebooks[VIDEO_STRIP_COUNT];

IWRAM_DATA u8 strip_2x2_codebooks_raw[VIDEO_STRIP_COUNT][CODEBOOK_2X2_SIZE*sizeof(YUV_Struct)+4]__attribute__((aligned(32)));
IWRAM_DATA YUV_Struct *strip_2x2_codebooks[VIDEO_STRIP_COUNT];

void init_table(){
    for(int i = 0; i < VIDEO_STRIP_COUNT; i++) {
        strip_4x4_codebooks[i] = (Block4x4_Struct*)strip_4x4_codebooks_raw[i];
        strip_2x2_codebooks[i] = (YUV_Struct*)strip_2x2_codebooks_raw[i];
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

// 简化的YUV转RGB函数，不使用Bayer抖动
IWRAM_CODE inline u32 yuv_to_rgb555_2pix(const u8 y[2], s8 d_r, s8 d_g, s8 d_b)
{
    u32 result = clip_lookup_table[y[0] + d_r];
    result |= (clip_lookup_table[y[0] + d_g] << 5);
    result |= (clip_lookup_table[y[0] + d_b] << 10);

    result |= (clip_lookup_table[y[1] + d_r] << 16);
    result |= (clip_lookup_table[y[1] + d_g] << 21);
    result |= (clip_lookup_table[y[1] + d_b] << 26);
    
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
IWRAM_DATA u16 super_block_relative_offsets[VIDEO_WIDTH * VIDEO_HEIGHT / 64];  // 预计算的8x8超级块相对偏移

void init_strip_info(){
    u16 current_y = 0;

    // 计算最大的8x8超级块数量，用于预分配偏移数组
    u16 max_super_blocks_per_row = VIDEO_WIDTH / SUPER_BLOCK_SIZE;  // 240/8 = 30
    u16 max_super_blocks_per_col = 0;

    // 找到最大的条带8x8超级块列数
    for(int strip_idx = 0; strip_idx < VIDEO_STRIP_COUNT; strip_idx++){
        u16 strip_super_blocks_per_col = strip_heights[strip_idx] / SUPER_BLOCK_SIZE;  // 高度/8
        if(strip_super_blocks_per_col > max_super_blocks_per_col) {
            max_super_blocks_per_col = strip_super_blocks_per_col;
        }
    }

    // 预计算8x8超级块的相对偏移（相对于条带起始位置）
    for(int super_block_idx = 0; super_block_idx < max_super_blocks_per_row * max_super_blocks_per_col; super_block_idx++){
        u16 super_bx = super_block_idx % max_super_blocks_per_row;
        u16 super_by = super_block_idx / max_super_blocks_per_row;
        // 8x8超级块 = 8x8像素，所以偏移是 super_by*8*屏幕宽度 + super_bx*8
        super_block_relative_offsets[super_block_idx] = (super_by * SUPER_BLOCK_SIZE * SCREEN_WIDTH) + (super_bx * SUPER_BLOCK_SIZE);
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
IWRAM_CODE inline void decode_block(const YUV_Struct &yuv_data, u16* dst)
{
    const s8 &d_r = yuv_data.d_r;
    const s8 &d_g = yuv_data.d_g;
    const s8 &d_b = yuv_data.d_b;

    u32* dst_row = (u32*)dst;
    
    // 第一行两个像素
    u8 y_pair_0[2] = {yuv_data.y[0][0], yuv_data.y[0][1]};
    *dst_row = yuv_to_rgb555_2pix(y_pair_0, d_r, d_g, d_b);
    
    // 第二行两个像素
    u8 y_pair_1[2] = {yuv_data.y[1][0], yuv_data.y[1][1]};
    *(dst_row + SCREEN_WIDTH/2) = yuv_to_rgb555_2pix(y_pair_1, d_r, d_g, d_b);
}

// 简化的4x4块解码函数 - 现在解码4个2x2块
IWRAM_CODE void decode_4x4_block(const YUV_Struct* codebook_2x2, const u8 quant_indices[4], u16* block_4x4_dst)
{
    decode_block(codebook_2x2[quant_indices[0]], block_4x4_dst);
    decode_block(codebook_2x2[quant_indices[1]], block_4x4_dst + 2);
    decode_block(codebook_2x2[quant_indices[2]], block_4x4_dst + SCREEN_WIDTH * 2);
    decode_block(codebook_2x2[quant_indices[3]], block_4x4_dst + SCREEN_WIDTH * 2 + 2);
}

// 简化的4x4码本拷贝函数 - 使用memcpy
IWRAM_CODE void copy_4x4_codebook_simple(u8* dst_raw, const u8* src, Block4x4_Struct** codebook_ptr, int codebook_size)
{
    *codebook_ptr = (Block4x4_Struct*)(dst_raw + 4); // 跳过对齐填充
    int copy_size = codebook_size * BYTES_PER_4X4_BLOCK;
    memcpy(*codebook_ptr, src, copy_size);
}

// 简化的2x2码本拷贝函数 - 使用memcpy
IWRAM_CODE void copy_2x2_codebook_simple(u8* dst_raw, const u8* src, YUV_Struct** codebook_ptr, int codebook_size)
{
    *codebook_ptr = (YUV_Struct*)(dst_raw + 4); // 跳过对齐填充
    int copy_size = codebook_size * BYTES_PER_2X2_BLOCK;
    memcpy(*codebook_ptr, src, copy_size);
}

// 直接解码4x4块
IWRAM_CODE void decode_4x4_block_direct(const Block4x4_Struct &block_4x4_data, u16* block_4x4_dst)
{
    // 新结构：直接包含4个YUV_Struct，按行优先顺序
    // block_4x4_data.blocks_2x2[0] = 左上 2x2块
    // block_4x4_data.blocks_2x2[1] = 右上 2x2块  
    // block_4x4_data.blocks_2x2[2] = 左下 2x2块
    // block_4x4_data.blocks_2x2[3] = 右下 2x2块
    
    // 解码左上块 (0,0)
    decode_block(block_4x4_data.blocks_2x2[0], block_4x4_dst);
    
    // 解码右上块 (0,2)
    decode_block(block_4x4_data.blocks_2x2[1], block_4x4_dst + 2);
    
    // 解码左下块 (2,0)  
    decode_block(block_4x4_data.blocks_2x2[2], block_4x4_dst + SCREEN_WIDTH * 2);
    
    // 解码右下块 (2,2)
    decode_block(block_4x4_data.blocks_2x2[3], block_4x4_dst + SCREEN_WIDTH * 2 + 2);
}

// 解码8x8超级块 - 使用4x4块码表
IWRAM_CODE void decode_8x8_super_block_with_4x4(const Block4x4_Struct* codebook_4x4, const u8 indices_4x4[4], u16* super_block_dst)
{
    // 解码4个4x4块：左上、右上、左下、右下
    decode_4x4_block_direct(codebook_4x4[indices_4x4[0]], super_block_dst);                           // 左上4x4
    decode_4x4_block_direct(codebook_4x4[indices_4x4[1]], super_block_dst + 4);                       // 右上4x4
    decode_4x4_block_direct(codebook_4x4[indices_4x4[2]], super_block_dst + SCREEN_WIDTH * 4);       // 左下4x4
    decode_4x4_block_direct(codebook_4x4[indices_4x4[3]], super_block_dst + SCREEN_WIDTH * 4 + 4);   // 右下4x4
}

// 解码8x8超级块 - 使用2x2块码表
IWRAM_CODE void decode_8x8_super_block_with_2x2(const YUV_Struct* codebook_2x2, const u8 indices_2x2[16], u16* super_block_dst)
{
    // 解码16个2x2块，按行优先顺序
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            int idx = row * 4 + col;
            u16* block_2x2_dst = super_block_dst + row * 2 * SCREEN_WIDTH + col * 2;
            decode_block(codebook_2x2[indices_2x2[idx]], block_2x2_dst);
        }
    }
}

IWRAM_CODE void decode_strip_i_frame_with_8x8_super_blocks(int strip_idx, const u8* src, u16* dst)
{
    u16 strip_base_offset = strip_info[strip_idx].buffer_offset;
    
    // 使用简化的拷贝函数
    copy_4x4_codebook_simple(strip_4x4_codebooks_raw[strip_idx], src, 
                             &strip_4x4_codebooks[strip_idx], CODEBOOK_4X4_SIZE);
    src += CODEBOOK_4X4_SIZE * BYTES_PER_4X4_BLOCK;
    
    copy_2x2_codebook_simple(strip_2x2_codebooks_raw[strip_idx], src, 
                            &strip_2x2_codebooks[strip_idx], CODEBOOK_2X2_SIZE);
    src += CODEBOOK_2X2_SIZE * BYTES_PER_2X2_BLOCK;
    
    auto &strip = strip_info[strip_idx];
    auto &codebook_4x4 = strip_4x4_codebooks[strip_idx];
    auto &codebook_2x2 = strip_2x2_codebooks[strip_idx];
    
    // 计算当前条带的8x8超级块数量
    u16 strip_super_blocks_w = VIDEO_WIDTH / SUPER_BLOCK_SIZE;
    u16 strip_super_blocks_h = strip.height / SUPER_BLOCK_SIZE;
    u16 tot_super_blocks = strip_super_blocks_w * strip_super_blocks_h;
    
    u16* strip_dst = dst + strip_base_offset;
    
    // 解码所有8x8超级块
    u16 super_bx = 0, super_by = 0;
    for (int super_block_idx = 0; super_block_idx < tot_super_blocks; super_block_idx++) {
        u16 block_offset = super_by * SUPER_BLOCK_SIZE * SCREEN_WIDTH + super_bx * SUPER_BLOCK_SIZE;
        u16* super_block_dst = strip_dst + block_offset;
        
        // 读取第一个索引/标记
        u8 first_byte = *src++;
        
        if (first_byte == BLOCK_4X4_MARKER) {
            // 4x4块模式：读取4个4x4块码表索引
            u8 indices_4x4[4];
            indices_4x4[0] = *src++;
            indices_4x4[1] = *src++;
            indices_4x4[2] = *src++;
            indices_4x4[3] = *src++;
            decode_8x8_super_block_with_4x4(codebook_4x4, indices_4x4, super_block_dst);
        } else {
            // 2x2块模式：当前字节是第一个2x2块码表索引，继续读取15个
            u8 indices_2x2[16];
            indices_2x2[0] = first_byte;
            for (int i = 1; i < 16; i++) {
                indices_2x2[i] = *src++;
            }
            decode_8x8_super_block_with_2x2(codebook_2x2, indices_2x2, super_block_dst);
        }
        super_bx++;
        if (super_bx >= strip_super_blocks_w) {
            super_bx = 0;
            super_by++;
        }
    }
}

IWRAM_CODE void decode_strip_p_frame_with_8x8_super_blocks(int strip_idx, const u8* src, u16* dst)
{
    u16 strip_base_offset = strip_info[strip_idx].buffer_offset;
    
    // 读取区域bitmap
    u16 zone_bitmap = src[0] | (src[1] << 8);
    src += 2;  // 跳过区域bitmap

    auto &codebook_4x4 = strip_4x4_codebooks[strip_idx];
    auto &codebook_2x2 = strip_2x2_codebooks[strip_idx];
    
    // 计算该条带的8x8超级块布局
    u16 strip_super_blocks_w = VIDEO_WIDTH / SUPER_BLOCK_SIZE;
    u16 strip_super_blocks_h = strip_info[strip_idx].height / SUPER_BLOCK_SIZE;
    
    // 使用bitmap右移优化处理每个有效区域
    u8 zone_idx = 0;
    while (zone_bitmap) {
        if (zone_bitmap & 1) {
            // 读取两种类型的更新数量
            u8 blocks_2x2_to_update = *src++;
            u8 blocks_4x4_to_update = *src++;
            
            // 计算区域在条带中的起始超级块行
            u16 zone_start_super_by = zone_idx * ZONE_HEIGHT_SUPER_BLOCKS;
            
            // 处理2x2块更新（16个索引）
            for (u8 i = 0; i < blocks_2x2_to_update; i++) {
                u8 zone_relative_idx = *src++;
                
                u8 indices_2x2[16];
                for (int j = 0; j < 16; j++) {
                    indices_2x2[j] = *src++;
                }
                
                u16 relative_super_by = zone_relative_idx / strip_super_blocks_w;
                u16 relative_super_bx = zone_relative_idx % strip_super_blocks_w;
                u16 absolute_super_by = zone_start_super_by + relative_super_by;
                
                u16 block_offset = absolute_super_by * SUPER_BLOCK_SIZE * SCREEN_WIDTH + relative_super_bx * SUPER_BLOCK_SIZE;
                u16* super_block_dst = dst + strip_base_offset + block_offset;
                decode_8x8_super_block_with_2x2(codebook_2x2, indices_2x2, super_block_dst);
            }
            
            // 处理4x4块更新（4个4x4块码表索引）
            for (u8 i = 0; i < blocks_4x4_to_update; i++) {
                u8 zone_relative_idx = *src++;
                u8 indices_4x4[4];
                indices_4x4[0] = *src++;
                indices_4x4[1] = *src++;
                indices_4x4[2] = *src++;
                indices_4x4[3] = *src++;
                
                u16 relative_super_by = zone_relative_idx / strip_super_blocks_w;
                u16 relative_super_bx = zone_relative_idx % strip_super_blocks_w;
                u16 absolute_super_by = zone_start_super_by + relative_super_by;
                
                u16 block_offset = (absolute_super_by * SCREEN_WIDTH + relative_super_bx) * SUPER_BLOCK_SIZE;
                u16* super_block_dst = dst + strip_base_offset + block_offset;
                decode_8x8_super_block_with_4x4(codebook_4x4, indices_4x4, super_block_dst);
            }
        }
        zone_bitmap >>= 1;
        zone_idx++;
    }
}

#define BLOCK_SKIP_MARKER 0xFE  // 新增：跳过4x4块的标记

IWRAM_CODE void decode_strip_i_frame_mixed(int strip_idx, const u8* src, u16* dst)
{
    u16 strip_base_offset = strip_info[strip_idx].buffer_offset;
    
    // 加载码表
    copy_4x4_codebook_simple(strip_4x4_codebooks_raw[strip_idx], src, 
                             &strip_4x4_codebooks[strip_idx], CODEBOOK_4X4_SIZE);
    src += CODEBOOK_4X4_SIZE * BYTES_PER_4X4_BLOCK;
    
    copy_2x2_codebook_simple(strip_2x2_codebooks_raw[strip_idx], src, 
                            &strip_2x2_codebooks[strip_idx], CODEBOOK_2X2_SIZE);
    src += CODEBOOK_2X2_SIZE * BYTES_PER_2X2_BLOCK;
    
    auto &strip = strip_info[strip_idx];
    auto &codebook_4x4 = strip_4x4_codebooks[strip_idx];
    auto &codebook_2x2 = strip_2x2_codebooks[strip_idx];
    
    // 计算当前条带的8x8超级块数量
    u16 strip_super_blocks_w = VIDEO_WIDTH / SUPER_BLOCK_SIZE;
    u16 strip_super_blocks_h = strip.height / SUPER_BLOCK_SIZE;
    u16 tot_super_blocks = strip_super_blocks_w * strip_super_blocks_h;
    
    u16* strip_dst = dst + strip_base_offset;
    
    // 解码所有8x8超级块
    u16 super_bx = 0, super_by = 0;
    for (int super_block_idx = 0; super_block_idx < tot_super_blocks; super_block_idx++) {
        u16 block_offset = super_by * SUPER_BLOCK_SIZE * SCREEN_WIDTH + super_bx * SUPER_BLOCK_SIZE;
        u16* super_block_dst = strip_dst + block_offset;
        
        // 解码4个4x4子块
        for (int quad_idx = 0; quad_idx < 4; quad_idx++) {
            u16 quad_by = quad_idx / 2;
            u16 quad_bx = quad_idx % 2;
            u16* quad_dst = super_block_dst + quad_by * 4 * SCREEN_WIDTH + quad_bx * 4;
            
            u8 first_byte = *src++;
            
            if (first_byte == BLOCK_4X4_MARKER) {
                // 4x4块模式：读取1个4x4块码表索引
                u8 index_4x4 = *src++;
                decode_4x4_block_direct(codebook_4x4[index_4x4], quad_dst);
            } else {
                // 2x2块模式：当前字节是第一个2x2块码表索引，继续读取3个
                u8 indices_2x2[4];
                indices_2x2[0] = first_byte;
                indices_2x2[1] = *src++;
                indices_2x2[2] = *src++;
                indices_2x2[3] = *src++;
                decode_4x4_block(codebook_2x2, indices_2x2, quad_dst);
            }
        }
        
        super_bx++;
        if (super_bx >= strip_super_blocks_w) {
            super_bx = 0;
            super_by++;
        }
    }
}

IWRAM_CODE void decode_strip_p_frame_mixed(int strip_idx, const u8* src, u16* dst)
{
    u16 strip_base_offset = strip_info[strip_idx].buffer_offset;
    
    // 读取区域bitmap
    u16 zone_bitmap = src[0] | (src[1] << 8);
    src += 2;

    auto &codebook_4x4 = strip_4x4_codebooks[strip_idx];
    auto &codebook_2x2 = strip_2x2_codebooks[strip_idx];
    
    // 计算该条带的8x8超级块布局
    u16 strip_super_blocks_w = VIDEO_WIDTH / SUPER_BLOCK_SIZE;
    u16 strip_super_blocks_h = strip_info[strip_idx].height / SUPER_BLOCK_SIZE;
    
    // 处理每个有效区域
    u8 zone_idx = 0;
    while (zone_bitmap) {
        if (zone_bitmap & 1) {
            u8 updates_count = *src++;
            
            // 计算区域在条带中的起始超级块行
            u16 zone_start_super_by = zone_idx * ZONE_HEIGHT_SUPER_BLOCKS;
            
            // 处理该区域的所有更新
            for (u8 i = 0; i < updates_count; i++) {
                u8 zone_relative_idx = *src++;
                
                u16 relative_super_by = zone_relative_idx / strip_super_blocks_w;
                u16 relative_super_bx = zone_relative_idx % strip_super_blocks_w;
                u16 absolute_super_by = zone_start_super_by + relative_super_by;
                
                u16 block_offset = absolute_super_by * SUPER_BLOCK_SIZE * SCREEN_WIDTH + relative_super_bx * SUPER_BLOCK_SIZE;
                u16* super_block_dst = dst + strip_base_offset + block_offset;
                
                // 解码4个4x4子块
                for (int quad_idx = 0; quad_idx < 4; quad_idx++) {
                    u16 quad_by = quad_idx / 2;
                    u16 quad_bx = quad_idx % 2;
                    u16* quad_dst = super_block_dst + quad_by * 4 * SCREEN_WIDTH + quad_bx * 4;
                    
                    u8 first_byte = *src++;
                    
                    if (first_byte == BLOCK_SKIP_MARKER) {
                        // 跳过该4x4块，不做任何操作
                        continue;
                    } else if (first_byte == BLOCK_4X4_MARKER) {
                        // 4x4块模式
                        u8 index_4x4 = *src++;
                        decode_4x4_block_direct(codebook_4x4[index_4x4], quad_dst);
                    } else {
                        // 2x2块模式
                        u8 indices_2x2[4];
                        indices_2x2[0] = first_byte;
                        indices_2x2[1] = *src++;
                        indices_2x2[2] = *src++;
                        indices_2x2[3] = *src++;
                        decode_4x4_block(codebook_2x2, indices_2x2, quad_dst);
                    }
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
        decode_strip_i_frame_mixed(strip_idx, src, dst);
    } else if (frame_type == FRAME_TYPE_P) {
        decode_strip_p_frame_mixed(strip_idx, src, dst);
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