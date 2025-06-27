#include <gba_systemcalls.h>
#include <gba_video.h>
#include <gba_dma.h>
#include <gba_interrupt.h>
#include <cstring>
#include "video_decoder.h"

constexpr int PIXELS_PER_FRAME = SCREEN_WIDTH * SCREEN_HEIGHT;

// 静态成员变量定义
u8 VideoDecoder::unified_codebook_raw[UNIFIED_CODEBOOK_SIZE*sizeof(YUV_Struct)+4]__attribute__((aligned(32)));
YUV_Struct* VideoDecoder::unified_codebook;
bool VideoDecoder::code_book_preloaded = false;
int VideoDecoder::last_check_frame = -1;
int VideoDecoder::next_i_frame = -1; //-1代表没找到下一个iframe

// RGB555码本存储
RGB555_Struct VideoDecoder::rgb555_codebook_buf[2][UNIFIED_CODEBOOK_SIZE];
int VideoDecoder::current_rgb555_codebook_index = 1;  // 当前使用的RGB555码本索引
RGB555_Struct* VideoDecoder::rgb555_codebook = VideoDecoder::rgb555_codebook_buf[1];
bool VideoDecoder::rgb555_codebook_preloaded = false;

// 查找表定义
u8 VideoDecoder::clip_lookup_table_raw[2048];
u8* VideoDecoder::clip_lookup_table = nullptr;
u16 VideoDecoder::big_block_relative_offsets[240/4*160/4];
u16 VideoDecoder::zone_block_relative_offsets[240];

void VideoDecoder::init() {
    // 初始化查找表
    int table_size = sizeof(clip_lookup_table_raw) / sizeof(clip_lookup_table_raw[0]);
    int offset = 1024;
    clip_lookup_table = clip_lookup_table_raw + offset;
    for(int i=-offset;i<table_size-offset;i++){
        u8 raw_val;
        if(i<=0)
            raw_val = 0;
        else if(i>=255)
            raw_val = 255;
        else
            raw_val = static_cast<u8>(i);
        clip_lookup_table[i] = raw_val>>3;
    }
    
    // 初始化块偏移查找表
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

// RGB555码本解码函数 - 直接使用预计算的RGB555值
IWRAM_CODE void VideoDecoder::decode_block_rgb555(const RGB555_Struct &rgb555_data, u16* dst)
{
    u16* dst_row = dst;
    // 直接拷贝预计算的RGB555值
    dst_row[0] = rgb555_data.rgb[0][0];
    dst_row[1] = rgb555_data.rgb[0][1];
    dst_row[SCREEN_WIDTH] = rgb555_data.rgb[1][0];
    dst_row[SCREEN_WIDTH + 1] = rgb555_data.rgb[1][1];
}

// RGB555码本解码色块（2x2上采样到4x4）
IWRAM_CODE void VideoDecoder::decode_color_block_rgb555(const RGB555_Struct &rgb555_data, u16* dst)
{
    u16* dst_row = dst;
    
    // 第一行：1122
    dst_row[0] = rgb555_data.rgb[0][0];
    dst_row[1] = rgb555_data.rgb[0][0];
    dst_row[2] = rgb555_data.rgb[0][1];
    dst_row[3] = rgb555_data.rgb[0][1];
    
    // 第二行：1122
    dst_row += SCREEN_WIDTH;
    dst_row[0] = rgb555_data.rgb[0][0];
    dst_row[1] = rgb555_data.rgb[0][0];
    dst_row[2] = rgb555_data.rgb[0][1];
    dst_row[3] = rgb555_data.rgb[0][1];
    
    // 第三行：3344
    dst_row += SCREEN_WIDTH;
    dst_row[0] = rgb555_data.rgb[1][0];
    dst_row[1] = rgb555_data.rgb[1][0];
    dst_row[2] = rgb555_data.rgb[1][1];
    dst_row[3] = rgb555_data.rgb[1][1];
    
    // 第四行：3344
    dst_row += SCREEN_WIDTH;
    dst_row[0] = rgb555_data.rgb[1][0];
    dst_row[1] = rgb555_data.rgb[1][0];
    dst_row[2] = rgb555_data.rgb[1][1];
    dst_row[3] = rgb555_data.rgb[1][1];
}
IWRAM_CODE void VideoDecoder::decode_segment_rgb555(u8 CODE_BOOK_SIZE,u8 INDEX_BIT_LEN,u8 INDEX_BIT_MASK ,u16 seg_idx, const u8** src, u16* zone_dst, 
                                            const RGB555_Struct* unified_codebook)
{
    const RGB555_Struct* codebook = unified_codebook;
    codebook += seg_idx<<INDEX_BIT_LEN;
    
    u8 num_blocks = *(*src)++;
    
    // 记录bitmap和位置数据的起始位置
    const u8* bitmap_and_indices_ptr = *src;
    
    // 跳过bitmap和位置数据
    *src += (num_blocks >> 1) * 3;  // 每2个块用3字节
    if (num_blocks & 1) {
        *src += 2;  // 最后一个奇数块用2字节
    }
    
    // 现在src指向bitstream开始位置
    BitReader reader(src,INDEX_BIT_LEN);
    
    // 解码每2个块为一组
    const u8* bitmap_ptr = bitmap_and_indices_ptr;
    for (u8 i = 0; i < (num_blocks >> 1); i++) {
        u8 valid_bitmap = *bitmap_ptr++;
        u8 zone_relative_idx1 = *bitmap_ptr++;
        u8 zone_relative_idx2 = *bitmap_ptr++;
        
        // 解码第一个4x4块
        u16* big_block_dst = zone_dst + zone_block_relative_offsets[zone_relative_idx1];
        decode_normal_4x4_block(valid_bitmap, reader, big_block_dst, codebook);
        
        // 解码第二个4x4块
        big_block_dst = zone_dst + zone_block_relative_offsets[zone_relative_idx2];
        decode_normal_4x4_block(valid_bitmap, reader, big_block_dst, codebook);
    }
    
    // 处理最后一个奇数块（如果存在）
    if (num_blocks & 1) {
        u8 valid_bitmap = *bitmap_ptr++;
        u8 zone_relative_idx1 = *bitmap_ptr++;
        
        u16* big_block_dst = zone_dst + zone_block_relative_offsets[zone_relative_idx1];
        decode_normal_4x4_block(valid_bitmap, reader, big_block_dst, codebook);
    }
    // BitReader析构时会自动更新src指针
}
// RGB555码本通用的4x4大块解码函数（纹理块）
IWRAM_CODE void VideoDecoder::decode_big_block_rgb555(const RGB555_Struct* codebook, const u8 quant_indices[4], u16* big_block_dst)
{
    decode_block_rgb555(codebook[quant_indices[0]], big_block_dst);
    decode_block_rgb555(codebook[quant_indices[1]], big_block_dst + 2);
    decode_block_rgb555(codebook[quant_indices[2]], big_block_dst + SCREEN_WIDTH * 2);
    decode_block_rgb555(codebook[quant_indices[3]], big_block_dst + SCREEN_WIDTH * 2 + 2);
}


// DMA拷贝码本的辅助函数
IWRAM_CODE void VideoDecoder::copy_unified_codebook(u8* dst_raw, const u8* src, YUV_Struct** codebook_ptr, int codebook_size)
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

// 将YUV码本转换为RGB555码本
IWRAM_CODE void VideoDecoder::convert_yuv_to_rgb555_codebook(const YUV_Struct* yuv_codebook, RGB555_Struct* rgb555_codebook, int codebook_size)
{
    for(int i = 0; i < codebook_size; i++) {
        const YUV_Struct& yuv = yuv_codebook[i];
        RGB555_Struct& rgb555 = rgb555_codebook[i];
        
        auto lookup_table = clip_lookup_table_raw + 1024;
        // 转换2x2块的每个像素
        for(int y = 0; y < 2; y++) {
            for(int x = 0; x < 2; x++) {
                u8 y_val = yuv.y[y][x];
                s8 cb = yuv.cb;
                s8 cr = yuv.cr;
                
                // 使用您的快速YUV到RGB转换矩阵：
                // R = Y + (Cr << 1)
                // G = Y - (Cb >> 1) - Cr  
                // B = Y + (Cb << 1)
                u16 r = lookup_table[y_val + (cr << 1)];
                u16 g = lookup_table[y_val - (cb >> 1) - cr];
                u16 b = lookup_table[y_val + (cb << 1)];
                
                rgb555.rgb[y][x] = r | (g << 5) | (b << 10);
            }
        }
    }
}

IWRAM_CODE void VideoDecoder::decode_normal_4x4_block(u8 &valid_bitmap, BitReader &reader, u16* big_block_dst, const RGB555_Struct * codebook){
        for (u8 sub_idx = 0; sub_idx < 4; sub_idx++) {
            if (valid_bitmap & 1) {
                u8 codebook_idx = reader.read();
                u16* subblock_dst = big_block_dst;
                switch (sub_idx) {
                    case 0: /* 左上 */ break;
                    case 1: subblock_dst += 2; break; /* 右上 */
                    case 2: subblock_dst += SCREEN_WIDTH * 2; break; /* 左下 */
                    case 3: subblock_dst += SCREEN_WIDTH * 2 + 2; break; /* 右下 */
                }
                decode_block_rgb555(codebook[codebook_idx], subblock_dst);
            }
            valid_bitmap >>= 1;
        }
    }

// 运动补偿相关函数

// 解码运动向量
IWRAM_CODE void VideoDecoder::decode_motion_vector(u8 encoded, int* dx, int* dy)
{
    *dx = (encoded & 0x0F) - MOTION_RANGE;
    *dy = ((encoded >> 4) & 0x0F) - MOTION_RANGE;
}

// 应用8x8块的运动补偿
IWRAM_CODE void VideoDecoder::apply_motion_compensation_8x8_block(u16* dst, u16* vram_src,
                                                                 int block_8x8_y, int block_8x8_x,
                                                                 int motion_dx, int motion_dy)
{
    // 计算8x8块在屏幕中的起始位置
    int dst_start_y = block_8x8_y * MOTION_BLOCK_8X8_SIZE;
    int dst_start_x = block_8x8_x * MOTION_BLOCK_8X8_SIZE;
    
    // 计算运动补偿后的源位置
    int src_start_y = dst_start_y + motion_dy;
    int src_start_x = dst_start_x + motion_dx;
    
    // 复制8x8像素块（从VRAM复制到buffer）
    for (int y = 0; y < MOTION_BLOCK_8X8_SIZE; y++) {
        for (int x = 0; x < MOTION_BLOCK_8X8_SIZE; x++) {
            int dst_y = dst_start_y + y;
            int dst_x = dst_start_x + x;
            int src_y = src_start_y + y;
            int src_x = src_start_x + x;
            
            // 边界检查
            if (dst_y >= 0 && dst_y < SCREEN_HEIGHT && dst_x >= 0 && dst_x < SCREEN_WIDTH &&
                src_y >= 0 && src_y < SCREEN_HEIGHT && src_x >= 0 && src_x < SCREEN_WIDTH) {
                // 从VRAM的源位置复制像素到buffer的目标位置
                dst[dst_y * SCREEN_WIDTH + dst_x] = vram_src[src_y * SCREEN_WIDTH + src_x];
            }
        }
    }
}

// 解码运动补偿数据
IWRAM_CODE void VideoDecoder::decode_motion_compensation_data(const u8** src, u16* dst, u16* vram_src)
{
    // 读取zone bitmap
    u8 zone_bitmap = *(*src)++;
    
    // 遍历每个zone
    for (int zone_idx = 0; zone_idx < MOTION_TOTAL_ZONES; zone_idx++) {
        if (zone_bitmap & (1 << zone_idx)) {
            // 读取该zone的运动补偿块数量
            u8 motion_blocks_count = *(*src)++;
            
            // 解码每个运动补偿块
            for (u8 i = 0; i < motion_blocks_count; i++) {
                // 读取zone内相对索引
                u8 zone_relative_idx = *(*src)++;
                
                // 读取编码的运动向量
                u8 encoded_mv = *(*src)++;
                
                // 解码运动向量
                int motion_dx, motion_dy;
                decode_motion_vector(encoded_mv, &motion_dx, &motion_dy);
                
                // 计算8x8块的全局坐标
                int zone_start_8x8_row = zone_idx * MOTION_BLOCKS_8X8_PER_ZONE_HEIGHT;
                int relative_8x8_row = zone_relative_idx / MOTION_BLOCKS_8X8_PER_ZONE_ROW;
                int relative_8x8_col = zone_relative_idx % MOTION_BLOCKS_8X8_PER_ZONE_ROW;
                
                int block_8x8_y = zone_start_8x8_row + relative_8x8_row;
                int block_8x8_x = relative_8x8_col;
                
                // 应用运动补偿
                apply_motion_compensation_8x8_block(dst, vram_src, block_8x8_y, block_8x8_x, 
                                                  motion_dx, motion_dy);
            }
        }
    }
}

IWRAM_CODE void VideoDecoder::load_codebook_and_convert(const u8* src)
{
    // 预加载码本
    if(!code_book_preloaded)
        copy_unified_codebook(unified_codebook_raw, src, &unified_codebook, UNIFIED_CODEBOOK_SIZE);
    
    // 转换为RGB555码本
    if(!rgb555_codebook_preloaded)
        convert_yuv_to_rgb555_codebook(unified_codebook, rgb555_codebook_buf[current_rgb555_codebook_index^1], UNIFIED_CODEBOOK_SIZE);
}

IWRAM_CODE void VideoDecoder::preload_codebook(const u8* src)
{
    u8 frame_type = src[0];
    if(frame_type != FRAME_TYPE_I || rgb555_codebook_preloaded) {
        VBlankIntrWait(); // 等待VBlank，防止CPU空跑跑满
        return; // 只在I帧中预加载码本
    }
    if(!code_book_preloaded){
        // 预加载统一码本
        copy_unified_codebook(unified_codebook_raw, src + 1, &unified_codebook, UNIFIED_CODEBOOK_SIZE);
        code_book_preloaded = true;
        return;
    }
    if(!rgb555_codebook_preloaded){
        // 预加载RGB555码本
        convert_yuv_to_rgb555_codebook(unified_codebook, rgb555_codebook_buf[current_rgb555_codebook_index^1], UNIFIED_CODEBOOK_SIZE);
        rgb555_codebook_preloaded = true;
    }
}

IWRAM_CODE void VideoDecoder::decode_i_frame_unified(const u8* src, u16* dst)
{
    // 拷贝统一码本
    if(!rgb555_codebook_preloaded){
        load_codebook_and_convert(src);
    }
    reset_codebook();

    src += UNIFIED_CODEBOOK_SIZE * BYTES_PER_BLOCK;
    current_rgb555_codebook_index ^= 1; // 切换到另一个RGB555码本缓冲区
    rgb555_codebook = rgb555_codebook_buf[current_rgb555_codebook_index];
    
    u16 tot_big_blocks = (VIDEO_WIDTH / (BLOCK_WIDTH * 2)) * (VIDEO_HEIGHT / (BLOCK_HEIGHT * 2));
    
    // 解码所有4x4大块
    for (int big_block_idx = 0; big_block_idx < tot_big_blocks; big_block_idx++) {
        u16* big_block_dst = dst + big_block_relative_offsets[big_block_idx];
        
        // 读取第一个索引/标记
        u8 first_byte = *src++;
        
        if (first_byte == COLOR_BLOCK_MARKER) {
            // 色块：读取统一码本索引
            u8 unified_idx = *src++;
            decode_color_block_rgb555(rgb555_codebook[unified_idx], big_block_dst);
        } else {
            // 纹理块：当前字节是第一个统一码本索引，继续读取3个
            u8 quant_indices[4];
            quant_indices[0] = first_byte;
            quant_indices[1] = *src++;
            quant_indices[2] = *src++;
            quant_indices[3] = *src++;
            decode_big_block_rgb555(rgb555_codebook, quant_indices, big_block_dst);
        }
    }
}

// RGB555版本的分段解码函数
IWRAM_CODE void VideoDecoder::decode_small_codebook_segment_rgb555(u16 seg_idx, const u8** src, u16* zone_dst, 
                                            const RGB555_Struct* unified_codebook)
{
    decode_segment_rgb555(MINI_CODEBOOK_SIZE,4,0xF,seg_idx,src,zone_dst,unified_codebook);
}

IWRAM_CODE void VideoDecoder::decode_medium_codebook_segment_rgb555(u8 seg_idx, const u8** src, u16* zone_dst, 
                                             const RGB555_Struct* unified_codebook)
{
    decode_segment_rgb555(MEDIUM_CODEBOOK_SIZE,6,0x3F,seg_idx,src,zone_dst,unified_codebook);
}

IWRAM_CODE void VideoDecoder::decode_full_index_segment_rgb555(const u8** src, u16* zone_dst, 
                                        const RGB555_Struct* unified_codebook)
{
    //只有一大段，因此CODEBOOKSIZE什么的，都填0即可
    decode_segment_rgb555(0,8,0xFF,0,src,zone_dst,unified_codebook);
}

// RGB555版本的P帧解码函数
IWRAM_CODE void VideoDecoder::decode_p_frame_unified_rgb555(const u8* src, u16* dst)
{
    // 首先将VRAM中的前一帧复制到buffer
    memcpy(dst, (u16*)VRAM, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(u16));
    
    // 然后解码运动补偿数据，从VRAM的其他位置复制到buffer的目标位置
    decode_motion_compensation_data(&src, dst, (u16*)VRAM);
    
    // 读取两个区域bitmap：纹理块和色块
    u16 detail_zone_bitmap = src[0] | (src[1] << 8);
    u16 color_zone_bitmap = src[2] | (src[3] << 8);
    src += 4; // 跳过两个bitmap的四个字节
    
    // 处理纹理块更新（新的bitmap+bitstream格式）
    u8 zone_idx = 0;
    u16 temp_bitmap = detail_zone_bitmap;
    while (temp_bitmap) {
        if (temp_bitmap & 1) {
            // 计算zone在整个屏幕中的基址偏移
            u16 zone_base_offset = zone_idx * ZONE_HEIGHT_PIXELS * SCREEN_WIDTH;
            u16* zone_dst = dst + zone_base_offset;
            
            // 读取小码表启用段bitmap
            u16 enabled_segments_bitmap = src[0] | (src[1] << 8);
            src += 2; // 跳过启用段bitmap的两个字节
            
            // 处理启用的小码表段
            for (u16 seg_idx = 0; seg_idx < 16; seg_idx++) {
                if (enabled_segments_bitmap & (1 << seg_idx)) {
                    decode_segment_rgb555(MINI_CODEBOOK_SIZE,4,0xF,seg_idx,&src,zone_dst,rgb555_codebook);
                }
            }
            
            // 读取中码表启用段bitmap
            u8 enabled_medium_segments_bitmap = *src++;
            
            // 处理启用的中码表段
            for (u8 seg_idx = 0; seg_idx < 4; seg_idx++) {
                if (enabled_medium_segments_bitmap & (1 << seg_idx)) {
                    decode_segment_rgb555(MEDIUM_CODEBOOK_SIZE,6,0x3F,seg_idx,&src,zone_dst,rgb555_codebook);
                }
            }
            
            // 处理剩余更新（完整索引）
            decode_full_index_segment_rgb555(&src, zone_dst, rgb555_codebook);
        }
        temp_bitmap >>= 1;
        zone_idx++;
    }
    
    // 处理色块更新（逻辑不变）
    zone_idx = 0;
    temp_bitmap = color_zone_bitmap;
    while (temp_bitmap) {
        if (temp_bitmap & 1) {
            // 读取色块更新数量
            u8 color_blocks_to_update = *src++;
            
            // 计算zone在整个屏幕中的基址偏移
            u16 zone_base_offset = zone_idx * ZONE_HEIGHT_PIXELS * SCREEN_WIDTH;
            u16* zone_dst = dst + zone_base_offset;
            
            // 处理色块更新（1个统一码本索引）
            for (u8 i = 0; i < color_blocks_to_update; i++) {
                u8 zone_relative_idx = *src++;
                u8 unified_idx = *src++;
                
                // 直接使用查找表获取相对偏移
                u16* big_block_dst = zone_dst + zone_block_relative_offsets[zone_relative_idx];
                decode_color_block_rgb555(rgb555_codebook[unified_idx], big_block_dst);
            }
        }
        temp_bitmap >>= 1;
        zone_idx++;
    }
}

IWRAM_CODE void VideoDecoder::decode_frame(const u8* frame_data, u16* dst)
{
    u8 frame_type = *frame_data++;
    
    if (frame_type == FRAME_TYPE_I) {
        decode_i_frame_unified(frame_data, dst);
    } else if (frame_type == FRAME_TYPE_P) {
        decode_p_frame_unified_rgb555(frame_data, dst);
    }
}

IWRAM_CODE bool VideoDecoder::is_i_frame(const u8* frame_data)
{
    u8 frame_type = *frame_data;
    return frame_type == FRAME_TYPE_I;
}
