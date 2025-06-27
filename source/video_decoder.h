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

// 运动补偿相关常量
#define MOTION_BLOCK_8X8_SIZE 8  // 运动补偿的8x8块大小
#define MOTION_BLOCKS_8X8_WIDTH (SCREEN_WIDTH / MOTION_BLOCK_8X8_SIZE)  // 30
#define MOTION_BLOCKS_8X8_HEIGHT (SCREEN_HEIGHT / MOTION_BLOCK_8X8_SIZE)  // 20
#define MOTION_BLOCKS_8X8_PER_ZONE_ROW MOTION_BLOCKS_8X8_WIDTH  // 30
#define MOTION_BLOCKS_8X8_PER_ZONE_HEIGHT 8  // 每个zone包含8行8x8块
#define MOTION_TOTAL_ZONES ((MOTION_BLOCKS_8X8_HEIGHT + MOTION_BLOCKS_8X8_PER_ZONE_HEIGHT - 1) / MOTION_BLOCKS_8X8_PER_ZONE_HEIGHT)  // 3
#define MOTION_RANGE 7  // 运动向量范围：±7像素

// YUV数据结构
struct YUV_Struct{
    u8 y[2][2];
    s8 cb;     // Cb 色度分量
    s8 cr;     // Cr 色度分量
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
    
    // 解码后的RGB555码本存储（在IWRAM中）
    static RGB555_Struct rgb555_codebook_buf[2][UNIFIED_CODEBOOK_SIZE];
    static RGB555_Struct *rgb555_codebook;
    static int current_rgb555_codebook_index;  // 当前使用的RGB555码本索引
    
    // 查找表
    static u16 big_block_relative_offsets[240/4*160/4];
    static u16 zone_block_relative_offsets[240];
    static u16 zone_motion_block_relative_offsets[240];

    // 新增：RGB555码本解码函数
    static void decode_color_block_rgb555(const RGB555_Struct &rgb555_data, u16* dst);
    static void decode_big_block_rgb555(const RGB555_Struct* codebook, const u8 quant_indices[4], u16* big_block_dst);
    static void decode_block_rgb555(const RGB555_Struct &rgb555_data, u16* dst);
    
    // 码本相关函数
    static void copy_unified_codebook(u8* dst_raw, const u8* src, YUV_Struct** codebook_ptr, int codebook_size);
    static void convert_yuv_to_rgb555_codebook(const YUV_Struct* yuv_codebook, RGB555_Struct* rgb555_codebook, int codebook_size);
    
    static void decode_normal_4x4_block(u8 &valid_bitmap, BitReader &reader, u16* big_block_dst, const RGB555_Struct * codebook);

    
    static void decode_small_codebook_segment(u16 seg_idx, const u8** src, u16* zone_dst, const YUV_Struct* unified_codebook);
    static void decode_medium_codebook_segment(u8 seg_idx, const u8** src, u16* zone_dst, const YUV_Struct* unified_codebook);
    static void decode_full_index_segment(const u8** src, u16* zone_dst, const YUV_Struct* unified_codebook);
    
    // 新增：RGB555分段解码函数
    static void decode_small_codebook_segment_rgb555(u16 seg_idx, const u8** src, u16* zone_dst, const RGB555_Struct* unified_codebook);
    static void decode_medium_codebook_segment_rgb555(u8 seg_idx, const u8** src, u16* zone_dst, const RGB555_Struct* unified_codebook);
    static void decode_full_index_segment_rgb555(const u8** src, u16* zone_dst, const RGB555_Struct* unified_codebook);
    
    static void decode_segment_rgb555(u8 CODE_BOOK_SIZE,u8 INDEX_BIT_LEN,u8 INDEX_BIT_MASK ,u16 seg_idx, const u8** src, u16* zone_dst, 
                                            const RGB555_Struct* unified_codebook);
    
    // 运动补偿相关函数
    static void decode_motion_compensation_data(const u8* &sr, u16* dst, u16* vram_src);
    static void apply_motion_compensation_8x8_block(u16* dst, u16* vram_src, 
                                                  int block_8x8_y, int block_8x8_x, 
                                                  int motion_dx, int motion_dy);
    static void decode_motion_vector(u8 encoded, int* dx, int* dy);
    // 帧解码函数
    static void decode_i_frame_unified(const u8* src, u16* dst);
    static void decode_p_frame_unified(const u8* src, u16* dst);
    
    // 新增：RGB555帧解码函数
    static void decode_i_frame_unified_rgb555(const u8* src, u16* dst);
    static void decode_p_frame_unified_rgb555(const u8* src, u16* dst);

public:
    static int next_i_frame;//用于预载
    static bool code_book_preloaded;
    static bool rgb555_codebook_preloaded;
    static int last_check_frame;
    // 查找表（移到public以便外部函数访问）
    static u8 clip_lookup_table_raw[2048];
    static u8 *clip_lookup_table;
    static void find_next_i_frame(const u8* video_data,int start_frame){
        constexpr int max_find_count = 30; // 一次最多查找30帧
        if(last_check_frame == -1){
            last_check_frame = start_frame;
        }
        for(int i=0;i<max_find_count;i++){
            if(is_i_frame(video_data + frame_offsets[last_check_frame])){
                next_i_frame = last_check_frame;
                last_check_frame = -1; // 重置偏移
                return;
            }
            last_check_frame++;
        }
    }
    
    // 初始化函数
    static void init();
    
    // 主要解码接口
    static void decode_frame(const u8* frame_data, u16* dst);
    static void preload_codebook(const u8* src);
    static void load_codebook_and_convert(const u8* src);
    
    // 检查是否为I帧
    static bool is_i_frame(const u8* frame_data);
    
    // 获取码本状态
    static bool is_codebook_preloaded() { return code_book_preloaded; }
    static bool is_rgb555_codebook_preloaded() { return rgb555_codebook_preloaded; }

    static void decode_block(const YUV_Struct &yuv_data, u16* dst);
    static void reset_codebook() {
        code_book_preloaded = false;
        rgb555_codebook_preloaded = false;
        last_check_frame = -1;
        next_i_frame = -1; // 重置下一个I帧索引
    }
};

#endif // VIDEO_DECODER_H 