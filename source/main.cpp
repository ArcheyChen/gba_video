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
EWRAM_BSS u8 yuv_cb_buffer[SCREEN_HEIGHT/4][SCREEN_WIDTH/4]; // 2.4 KB - Cb按4x4块，0-255
EWRAM_BSS u8 yuv_cr_buffer[SCREEN_HEIGHT/4][SCREEN_WIDTH/4]; // 2.4 KB - Cr按4x4块，0-255

// VRAM输出缓冲
EWRAM_BSS u16 vram_buffer[PIXELS_PER_FRAME];  // 75 KB - RGB555输出缓冲

// YUV555到RGB555的查找表（2^15 = 32K条目 × 2字节 = 64KB）
EWRAM_DATA static u16 yuv555_to_rgb555_table[32768];

// 块位置到内存偏移的查找表
EWRAM_DATA static u32 block_offset_table[VIDEO_BLOCKS_PER_FRAME];

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

void init_clip_table(){
    for(int i = -256; i < 1024 - 256; i++){
        u8 raw_val;
        if(i <= 0)
            raw_val = 0;
        else if(i >= 255)
            raw_val = 255;
        else
            raw_val = static_cast<u8>(i);
        clip_lookup_table[i] = raw_val >> 3;
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

void init_yuv555_to_rgb555_table(){
    // 遍历所有可能的YUV555组合（5+5+5 = 15位）
    for(int yuv555 = 0; yuv555 < 32768; yuv555++){
        // 解包YUV555：Y(5位) + Cb(5位) + Cr(5位)
        int y5  = (yuv555 >> 0) & 0x1F;   // 低5位：Y
        int cb5 = (yuv555 >> 5) & 0x1F;   // 中5位：Cb
        int cr5 = (yuv555 >> 10) & 0x1F;  // 高5位：Cr

        // 扩展到8位（左移3位，相当于 ×8）
        int y8  = y5 << 3;              // 0~248
        int cb8 = (cb5 << 3) - 128;     // -128~120
        int cr8 = (cr5 << 3) - 128;     // -128~120

        // YUV到RGB转换（使用简化系数）
        int r = y8 + (cr8 << 1);           // Y + 2*Cr
        int g = y8 - (cb8 >> 1) - cr8;     // Y - Cb/2 - Cr
        int b = y8 + (cb8 << 1);           // Y + 2*Cb

        // 裁剪到0~255
        if(r < 0) r = 0; else if(r > 255) r = 255;
        if(g < 0) g = 0; else if(g > 255) g = 255;
        if(b < 0) b = 0; else if(b > 255) b = 255;

        // 转换为RGB555（右移3位）
        int r5 = r >> 3;
        int g5 = g >> 3;
        int b5 = b >> 3;

        // 打包为RGB555
        yuv555_to_rgb555_table[yuv555] = (u16)((r5) | (g5 << 5) | (b5 << 10));
    }
}

// 预计算块位置到内存偏移的查找表
void init_block_offset_table(){
    int block_pos = 0;
    for (int y = 0; y < SCREEN_HEIGHT; y += 4) {
        for (int x = 0; x < SCREEN_WIDTH; x += 4) {
            block_offset_table[block_pos] = y * SCREEN_WIDTH + x;
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

// 将一个YUV残差块累加到YUV缓冲区（残差模式）
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

    yuv_cb_buffer[block_y][block_x] = clamp_to_u8[cb_final];
    yuv_cr_buffer[block_y][block_x] = clamp_to_u8[cr_final];

    // 累加Y残差到均值（16个像素），使用查找表裁剪
    int base_y = block_y * 4;
    int base_x = block_x * 4;
    for (int py = 0; py < 4; py++) {
        for (int px = 0; px < 4; px++) {
            s16 y_final = (s16)y_mean + (s16)residual_block.y[py * 4 + px];
            yuv_y_buffer[base_y + py][base_x + px] = clamp_to_u8[y_final];
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
            // 完全无变化的P帧，什么都不做
            return;
        }
    }

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
    }
}

// 将整个YUV缓冲区转换为RGB555并写入VRAM缓冲（优化：使用YUV555查找表）
IWRAM_CODE void convert_yuv_to_rgb555()
{
    int vram_idx = 0;

    // 按4x4块遍历（60块宽 × 40块高）
    for (int block_y = 0; block_y < SCREEN_HEIGHT / 4; block_y++) {
        for (int block_x = 0; block_x < SCREEN_WIDTH / 4; block_x++) {

            // 获取当前块的Cb/Cr（现在直接是0~255 u8格式）并转换为5位
            u16 cb8 = yuv_cb_buffer[block_y][block_x];  // 0~255
            u16 cr8 = yuv_cr_buffer[block_y][block_x];  // 0~255

            // 直接右移3位得到5位
            u16 cb5 = (cb8 >> 3);  // 5位Cb
            u16 cr5 = (cr8 >> 3);  // 5位Cr

            // 处理4x4块内的16个像素
            int base_y = block_y * 4;
            int base_x = block_x * 4;

            for (int py = 0; py < 4; py++) {
                int y = base_y + py;
                vram_idx = y * SCREEN_WIDTH + base_x;

                for (int px = 0; px < 4; px++) {
                    int x = base_x + px;

                    // 获取Y亮度并转换为5位
                    u16 y8 = yuv_y_buffer[y][x];     // 0~255
                    u16 y5 = (y8 >> 3);       // 5位Y

                    // 组合YUV555索引：Y(低5位) + Cb(中5位) + Cr(高5位)
                    u16 yuv555 = y5 | (cb5 << 5) | (cr5 << 10);

                    // 直接查表得到RGB555
                    vram_buffer[vram_idx++] = yuv555_to_rgb555_table[yuv555];
                }
            }
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
    init_yuv555_to_rgb555_table();  // 初始化YUV555到RGB555查找表
    init_block_offset_table();  // 初始化块位置查找表
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

        // 根据帧类型解码到YUV缓冲区
        if (frame_type == 0) {
            // I帧：完全重写YUV缓冲
            decode_i_frame(frame_data);
        } else {
            // P帧：更新变化的块
            decode_p_frame(frame_data);
        }

        // 转换YUV到RGB555
        convert_yuv_to_rgb555();

        // 复制到VRAM
        DMA3COPY(vram_buffer, VRAM, PIXELS_PER_FRAME | DMA16);

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

        VBlankIntrWait();
    }
}
