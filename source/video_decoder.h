#ifndef VIDEO_DECODER_H
#define VIDEO_DECODER_H

#include <gba_types.h>
#include "video_data.h"
#include "bit_reader.h"

// 新增常量定义
#define ZONE_HEIGHT_PIXELS 16  // 每个区域的像素高度
#define ZONE_HEIGHT_BIG_BLOCKS (ZONE_HEIGHT_PIXELS / (BLOCK_HEIGHT * 2))  // 每个区域的4x4大块行数
#define MINI_CODEBOOK_SIZE 16  // 每个分段码本的大小
#define MEDIUM_CODEBOOK_SIZE 64  // 每个中等码本的大小
#define SKIP_MARKER_4BIT 0xF   // 4bit跳过标记
#define SKIP_MARKER_6BIT 0x3F  // 6bit跳过标记

// YUV数据结构
struct YUV_Struct{
    u8 y[2][2];
    s8 d_r;    // 预计算的 Cr
    s8 d_g;    // 预计算的 (-(Cb>>1)-Cr)>>1
    s8 d_b;    // 预计算的 Cb
} __attribute__((packed));

// 解码后的RGB555数据结构（直接存储RGB555值，避免实时转换）
struct RGB555_Struct{
    u16 rgb[2][2];  // 直接存储RGB555值
} __attribute__((packed));

// 视频解码器类
class VideoDecoder {
private:
    // 统一码本存储（在IWRAM中）
    static u8 unified_codebook_raw[UNIFIED_CODEBOOK_SIZE*sizeof(YUV_Struct)+4]__attribute__((aligned(32)));
    static YUV_Struct *unified_codebook;
    static bool code_book_preloaded;
    
    // 解码后的RGB555码本存储（在IWRAM中）
    static u8 rgb555_codebook_raw[UNIFIED_CODEBOOK_SIZE*sizeof(RGB555_Struct)+4]__attribute__((aligned(32)));
    static RGB555_Struct *rgb555_codebook;
    static bool rgb555_codebook_preloaded;
    
    // 查找表
    static u16 big_block_relative_offsets[240/4*160/4];
    static u16 zone_block_relative_offsets[240];
    
    // 私有解码函数
    static void decode_color_block(const YUV_Struct &yuv_data, u16* dst);
    static void decode_big_block(const YUV_Struct* codebook, const u8 quant_indices[4], u16* big_block_dst);
    
    // 新增：RGB555码本解码函数
    static void decode_color_block_rgb555(const RGB555_Struct &rgb555_data, u16* dst);
    static void decode_big_block_rgb555(const RGB555_Struct* codebook, const u8 quant_indices[4], u16* big_block_dst);
    static void decode_block_rgb555(const RGB555_Struct &rgb555_data, u16* dst);
    
    // 码本相关函数
    static void copy_unified_codebook(u8* dst_raw, const u8* src, YUV_Struct** codebook_ptr, int codebook_size);
    static void convert_yuv_to_rgb555_codebook(const YUV_Struct* yuv_codebook, RGB555_Struct* rgb555_codebook, int codebook_size);
    
    // 分段解码函数
    static void decode_small_codebook_4x4_block(u8 &valid_bitmap, BitReader &reader, u16* big_block_dst, const YUV_Struct * mini_codebook);
    static void decode_medium_codebook_4x4_block(u8 &valid_bitmap, BitReader &reader, u16* big_block_dst, const YUV_Struct* medium_codebook);
    static void decode_full_index_4x4_block(u8 &valid_bitmap, BitReader &reader, u16* big_block_dst, const YUV_Struct* unified_codebook);
    
    // 新增：RGB555分段解码函数
    static void decode_small_codebook_4x4_block_rgb555(u8 &valid_bitmap, BitReader &reader, u16* big_block_dst, const RGB555_Struct * mini_codebook);
    static void decode_medium_codebook_4x4_block_rgb555(u8 &valid_bitmap, BitReader &reader, u16* big_block_dst, const RGB555_Struct* medium_codebook);
    static void decode_full_index_4x4_block_rgb555(u8 &valid_bitmap, BitReader &reader, u16* big_block_dst, const RGB555_Struct* unified_codebook);
    
    static void decode_small_codebook_segment(u16 seg_idx, const u8** src, u16* zone_dst, const YUV_Struct* unified_codebook);
    static void decode_medium_codebook_segment(u8 seg_idx, const u8** src, u16* zone_dst, const YUV_Struct* unified_codebook);
    static void decode_full_index_segment(const u8** src, u16* zone_dst, const YUV_Struct* unified_codebook);
    
    // 新增：RGB555分段解码函数
    static void decode_small_codebook_segment_rgb555(u16 seg_idx, const u8** src, u16* zone_dst, const RGB555_Struct* unified_codebook);
    static void decode_medium_codebook_segment_rgb555(u8 seg_idx, const u8** src, u16* zone_dst, const RGB555_Struct* unified_codebook);
    static void decode_full_index_segment_rgb555(const u8** src, u16* zone_dst, const RGB555_Struct* unified_codebook);
    
    // 帧解码函数
    static void decode_i_frame_unified(const u8* src, u16* dst);
    static void decode_p_frame_unified(const u8* src, u16* dst);
    
    // 新增：RGB555帧解码函数
    static void decode_i_frame_unified_rgb555(const u8* src, u16* dst);
    static void decode_p_frame_unified_rgb555(const u8* src, u16* dst);

public:
    // 查找表（移到public以便外部函数访问）
    static u8 clip_lookup_table[512];
    
    // 初始化函数
    static void init();
    
    // 主要解码接口
    static void decode_frame(const u8* frame_data, u16* dst);
    static void preload_codebook(const u8* src);
    
    // 检查是否为I帧
    static bool is_i_frame(const u8* frame_data);
    
    // 获取码本状态
    static bool is_codebook_preloaded() { return code_book_preloaded; }
    static bool is_rgb555_codebook_preloaded() { return rgb555_codebook_preloaded; }

    static void decode_block(const YUV_Struct &yuv_data, u16* dst);
};

#endif // VIDEO_DECODER_H 