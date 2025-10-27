// gba_video_player.cpp  v4
// Mode 3 单缓冲 + IP帧码表解码

#include <gba_systemcalls.h>
#include <gba_video.h>
#include <gba_dma.h>
#include <gba_interrupt.h>
#include <gba_input.h>

#include "video_data.h"

constexpr int PIXELS_PER_FRAME = SCREEN_WIDTH * SCREEN_HEIGHT;

struct YUV_Struct{
    s8 cb;      // Cb 残差 (-128~127)
    s8 cr;      // Cr 残差 (-128~127)
    s8 y[16];   // Y 残差 (-128~127) - 4x4块需要16个像素
} __attribute__((packed));

// YUV9累积缓冲区（残差编码模式）
EWRAM_BSS u8 yuv_y_buffer[SCREEN_HEIGHT][SCREEN_WIDTH];    // 37.5 KB - Y全分辨率，0-255
IWRAM_DATA u8 yuv_cb_buffer[SCREEN_HEIGHT/4][SCREEN_WIDTH/4]; // 2.4 KB - Cb按4x4块，0-255
IWRAM_DATA u8 yuv_cr_buffer[SCREEN_HEIGHT/4][SCREEN_WIDTH/4]; // 2.4 KB - Cr按4x4块，0-255

// RGB缓冲区（EWRAM中的双缓冲，用于预转换YUV到RGB555）
EWRAM_DATA u16 rgb_buffer[SCREEN_HEIGHT][SCREEN_WIDTH];    // 76.8 KB - RGB555格式

// 变化块队列（记录需要更新到VRAM的块）
IWRAM_DATA u16 dirty_block_queue[VIDEO_BLOCKS_PER_FRAME];  // 4.8 KB - 最多2400个块索引
IWRAM_DATA u16 dirty_block_count = 0;  // 当前队列中的块数量


// 块位置到VRAM左上角偏移的查找表（优化：消除乘法）
EWRAM_DATA static u32 block_vram_offset_table[VIDEO_BLOCKS_PER_FRAME];

// 块位置到X/Y坐标的查找表
IWRAM_DATA static u8 block_x_table[VIDEO_BLOCKS_PER_FRAME];
IWRAM_DATA static u8 block_y_table[VIDEO_BLOCKS_PER_FRAME];

// 当前GOP索引（用于码表访问）
static int current_gop_index = 0;

// YUV到RGB转换的裁切查找表
IWRAM_DATA static u8 clip_table_raw[1024];
u8* clip_lookup_table = clip_table_raw + 256;

// 通用u8裁剪表：输入范围 -256~511 → 输出 0~255
IWRAM_DATA static u8 clamp_to_u8_table_raw[768];
u8* clamp_to_u8 = clamp_to_u8_table_raw + 256;  // 偏移后可直接用负数索引

// 通用s8裁剪表：输入范围 -256~255 → 输出 -128~127
IWRAM_DATA static s8 clamp_to_s8_table_raw[512];
s8* clamp_to_s8 = clamp_to_s8_table_raw + 256;  // 偏移后可直接用负数索引

// Bayer抖动矩阵 (4x4) - 预先计算好除以31的结果
// 使用const而不是IWRAM_DATA来避免冲突
static const u8 bayer_pattern_4x4[4][4] = {
    {0, 6, 1, 7},   // 15/31=0, 195/31=6, 60/31=1, 240/31=7
    {4, 2, 5, 3},   // 135/31=4, 75/31=2, 180/31=5, 120/31=3
    {1, 7, 0, 6},   // 45/31=1, 225/31=7, 30/31=0, 210/31=6
    {5, 3, 4, 2}    // 165/31=5, 105/31=3, 150/31=4, 90/31=2
};

// YUV到RGB555的转换函数（使用裁剪查找表和Bayer抖动）
IWRAM_CODE inline u16 yuv_to_rgb555(u8 y8, u8 cb8, u8 cr8, int x, int y)
{
    // 获取Bayer抖动值（已预先除以31）
    u8 dither = bayer_pattern_4x4[y & 3][x & 3];

    // 移除Cb/Cr的128偏移
    s16 cb = cb8 - 128;
    s16 cr = cr8 - 128;

    // YUV到RGB转换的差分值
    s16 d_r = cr << 1;           // 2*Cr
    s16 d_g = -(cb >> 1) - cr;   // -Cb/2 - Cr
    s16 d_b = cb << 1;           // 2*Cb

    // 使用裁剪查找表进行RGB555转换
    u16 r = clip_lookup_table[dither + y8 + d_r];
    u16 g = clip_lookup_table[dither + y8 + d_g];
    u16 b = clip_lookup_table[dither + y8 + d_b];

    // 组合为RGB555
    return (r) | ((g) << 5) | ((b) << 10);
}

void init_clip_table(){
    for(int i = -256; i < 1024 - 256; i++){
        u8 raw_val;
        if(i <= 0)
            raw_val = 0;
        else if(i >= 255)
            raw_val = 255;
        else
            raw_val = static_cast<u8>(i);
        clip_lookup_table[i] = raw_val >> 3;  // 输出0-31 (5位)
    }
}

void init_clamp_tables(){
    // 初始化u8裁剪表：-256~511 → 0~255
    for(int i = -256; i < 512; i++){
        if(i < 0)
            clamp_to_u8[i] = 0;
        else if(i > 255)
            clamp_to_u8[i] = 255;
        else
            clamp_to_u8[i] = (u8)i;
    }

    // 初始化s8裁剪表：-256~255 → -128~127
    for(int i = -256; i < 256; i++){
        if(i < -128)
            clamp_to_s8[i] = -128;
        else if(i > 127)
            clamp_to_s8[i] = 127;
        else
            clamp_to_s8[i] = (s8)i;
    }
}


// 预计算块位置到VRAM左上角偏移的查找表（优化：消除convert中的乘法）
void init_block_vram_offset_table(){
    int block_pos = 0;
    for (int y = 0; y < SCREEN_HEIGHT; y += 4) {
        for (int x = 0; x < SCREEN_WIDTH; x += 4) {
            // 块左上角在VRAM中的偏移 = y * SCREEN_WIDTH + x
            block_vram_offset_table[block_pos] = y * SCREEN_WIDTH + x;
            block_pos++;
        }
    }
}

// 预计算块位置到X/Y坐标的查找表
void init_block_xy_tables(){
    int block_pos = 0;
    for (int y = 0; y < SCREEN_HEIGHT / 4; y++) {
        for (int x = 0; x < SCREEN_WIDTH / 4; x++) {
            block_x_table[block_pos] = x;
            block_y_table[block_pos] = y;
            block_pos++;
        }
    }
}

// 将一个YUV残差块累加到YUV缓冲区并同时更新RGB缓冲区（残差模式）
IWRAM_CODE void apply_residual_to_buffer(u16 codeword_idx, int block_pos, u8 y_mean, u8 cb_mean, u8 cr_mean)
{
    const YUV_Struct* codebook = (YUV_Struct*)video_codebooks[current_gop_index];
    const YUV_Struct& residual_block = codebook[codeword_idx];

    // 使用预计算的查找表获取块的X/Y坐标
    int block_x = block_x_table[block_pos];
    int block_y = block_y_table[block_pos];

    // 累加Cb/Cr残差到均值（均值现在是0~255 u8格式）
    // 残差是-128~127 s8格式，需要先转换为s16计算，再裁剪回u8
    s16 cb_final = (s16)cb_mean + (s16)residual_block.cb;
    s16 cr_final = (s16)cr_mean + (s16)residual_block.cr;

    u8 cb_final_u8 = clamp_to_u8[cb_final];
    u8 cr_final_u8 = clamp_to_u8[cr_final];

    yuv_cb_buffer[block_y][block_x] = cb_final_u8;
    yuv_cr_buffer[block_y][block_x] = cr_final_u8;

    // 累加Y残差到均值（16个像素）并直接转换到RGB555
    int base_y = block_y * 4;
    int base_x = block_x * 4;
    for (int py = 0; py < 4; py++) {
        int y = base_y + py;
        for (int px = 0; px < 4; px++) {
            int x = base_x + px;
            s16 y_final = (s16)y_mean + (s16)residual_block.y[py * 4 + px];
            u8 y_final_u8 = clamp_to_u8[y_final];

            // 使用公式转换YUV到RGB555并写入RGB缓冲区（带Bayer抖动）
            rgb_buffer[y][x] = yuv_to_rgb555(y_final_u8, cb_final_u8, cr_final_u8, x, y);

            // 同时更新YUV缓冲区（保留用于P帧比较）
            yuv_y_buffer[y][x] = y_final_u8;
        }
    }
}


IWRAM_CODE void decode_i_frame(const u16* frame_data)
{
    // I帧新格式：[块数量, Y均值×2400, Cb均值×2400, Cr均值×2400, 残差索引×2400]
    u16 block_count = frame_data[0];

    // 解析均值数据
    const u16* y_means = frame_data + 1;                    // Y均值起始
    const u16* cb_means = frame_data + 1 + block_count;      // Cb均值起始
    const u16* cr_means = frame_data + 1 + block_count * 2;  // Cr均值起始
    const u16* residual_indices = frame_data + 1 + block_count * 3;  // 残差索引起始

    // 清空队列并准备记录全部块
    dirty_block_count = 0;

    for (int block_pos = 0; block_pos < VIDEO_BLOCKS_PER_FRAME; block_pos++)
    {
        // 获取当前块的均值（Cb/Cr现在直接是0~255 u8格式）
        u8 y_mean = (u8)y_means[block_pos];
        u8 cb_mean = (u8)cb_means[block_pos];
        u8 cr_mean = (u8)cr_means[block_pos];

        // 获取残差码字索引
        u16 residual_idx = residual_indices[block_pos];

        // 应用残差到YUV缓冲区
        apply_residual_to_buffer(residual_idx, block_pos, y_mean, cb_mean, cr_mean);

        // I帧：记录所有块到队列
        dirty_block_queue[dirty_block_count++] = block_pos;
    }
}

IWRAM_CODE void decode_p_frame(const u16* frame_data)
{
    // P帧新格式：[段数量, 段数据..., 单块数量, 单块数据...]
    // 段数据: [起始, 长度, Y均值×N, Cb均值×N, Cr均值×N, 残差索引×N]
    // 单块数据: [位置, Y均值, Cb均值, Cr均值, 残差索引]
    u16 segment_count = frame_data[0];

    if (segment_count == 0) {
        // 检查是否有单块数据
        if (frame_data[1] == 0) {
            // 完全无变化的P帧，队列保持为空（不清空，沿用上一帧）
            return;
        }
    }

    // P帧开始时清空队列
    dirty_block_count = 0;

    const u16* data = frame_data + 1;
    u16 data_offset = 0;

    // 处理连续段（长度>=2）
    for (int seg = 0; seg < segment_count; seg++)
    {
        u16 seg_start = data[data_offset++];     // 段起始位置
        u16 seg_count = data[data_offset++];     // 段块数量

        // 解析均值数据
        const u16* y_means = data + data_offset;
        const u16* cb_means = data + data_offset + seg_count;
        const u16* cr_means = data + data_offset + seg_count * 2;
        data_offset += seg_count * 3;  // 跳过均值数据

        // 解析残差索引并解码
        for (int i = 0; i < seg_count; i++)
        {
            u16 block_pos = seg_start + i;
            u16 residual_idx = data[data_offset++];

            // 获取均值（Cb/Cr现在直接是0~255 u8格式）
            u8 y_mean = (u8)y_means[i];
            u8 cb_mean = (u8)cb_means[i];
            u8 cr_mean = (u8)cr_means[i];

            // 应用残差到YUV缓冲区
            apply_residual_to_buffer(residual_idx, block_pos, y_mean, cb_mean, cr_mean);

            // P帧：仅记录变化的块到队列
            dirty_block_queue[dirty_block_count++] = block_pos;
        }
    }

    // 处理单块（长度=1）
    u16 single_count = data[data_offset++];

    // 解码所有单块
    for (int i = 0; i < single_count; i++)
    {
        u16 block_pos = data[data_offset++];        // 块位置
        u8 y_mean = (u8)data[data_offset++];        // Y均值
        u8 cb_mean = (u8)data[data_offset++];       // Cb均值（直接是0~255 u8格式）
        u8 cr_mean = (u8)data[data_offset++];       // Cr均值（直接是0~255 u8格式）
        u16 residual_idx = data[data_offset++];     // 残差索引

        // 应用残差到YUV缓冲区
        apply_residual_to_buffer(residual_idx, block_pos, y_mean, cb_mean, cr_mean);

        // P帧：仅记录变化的块到队列
        dirty_block_queue[dirty_block_count++] = block_pos;
    }
}

// 将RGB缓冲区中的变化块拷贝到VRAM（VBlank期间调用）
IWRAM_CODE void copy_rgb_buffer_to_vram()
{
    u16* vram = (u16*)VRAM;

    // 遍历队列中的所有变化块
    for (int q = 0; q < dirty_block_count; q++)
    {
        u16 block_pos = dirty_block_queue[q];

        // 使用预计算的查找表获取块的X/Y坐标
        int block_x = block_x_table[block_pos];
        int block_y = block_y_table[block_pos];

        // 使用预计算的VRAM偏移表（消除乘法）
        u32 vram_idx = block_vram_offset_table[block_pos];  // 块左上角VRAM偏移
        int base_y = block_y * 4;
        int base_x = block_x * 4;

        // 直接从RGB缓冲区拷贝4x4块到VRAM（优化：批量内存拷贝）
        for (int py = 0; py < 4; py++) {
            int y = base_y + py;
            // 使用DMA或手动拷贝一行4个像素
            vram[vram_idx + 0] = rgb_buffer[y][base_x + 0];
            vram[vram_idx + 1] = rgb_buffer[y][base_x + 1];
            vram[vram_idx + 2] = rgb_buffer[y][base_x + 2];
            vram[vram_idx + 3] = rgb_buffer[y][base_x + 3];
            vram_idx += SCREEN_WIDTH;  // 移动到下一行
        }
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

    init_clip_table();
    init_clamp_tables();        // 初始化裁剪查找表
    init_block_vram_offset_table();  // 初始化块VRAM偏移查找表
    init_block_xy_tables();     // 初始化块X/Y坐标查找表

    // 初始化YUV缓冲区为0
    for (int y = 0; y < SCREEN_HEIGHT; y++) {
        for (int x = 0; x < SCREEN_WIDTH; x++) {
            yuv_y_buffer[y][x] = 0;
        }
    }
    for (int y = 0; y < SCREEN_HEIGHT / 4; y++) {
        for (int x = 0; x < SCREEN_WIDTH / 4; x++) {
            yuv_cb_buffer[y][x] = 0;
            yuv_cr_buffer[y][x] = 0;
        }
    }

    // 初始化RGB缓冲区为黑色
    for (int y = 0; y < SCREEN_HEIGHT; y++) {
        for (int x = 0; x < SCREEN_WIDTH; x++) {
            rgb_buffer[y][x] = 0;  // RGB555黑色
        }
    }

    while (1)
    {
        // 检查是否需要切换GOP
        int required_gop = frame / VIDEO_GOP_SIZE;
        if (required_gop != current_gop_index) {
            current_gop_index = required_gop;
        }

        // 获取当前帧的数据
        u32 frame_offset = video_frame_offsets[frame];
        const u16* frame_data = video_frame_data + frame_offset;
        u8 frame_type = video_frame_types[frame];

        // 根据帧类型解码到YUV缓冲区和RGB缓冲区
        if (frame_type == 0) {
            // I帧：完全重写YUV缓冲和RGB缓冲
            decode_i_frame(frame_data);
        } else {
            // P帧：更新变化的块
            decode_p_frame(frame_data);
        }

        // 等待VBlank并拷贝RGB缓冲区到VRAM
        VBlankIntrWait();
        copy_rgb_buffer_to_vram();

        frame++;
        if(frame >= frames_total) // 循环播放
        {
            frame = 0; // 重置帧计数
            current_gop_index = -1; // 强制重新加载GOP
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

    }
}
