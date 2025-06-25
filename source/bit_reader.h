#ifndef BIT_READER_H
#define BIT_READER_H

#include <gba_types.h>

class BitReader {
private:
    const u8** src_ptr;     // 指向外部src指针的指针
    const u8* inner_src;    // 内部的，减少二重指针开销
    u32 bit_buffer;         // 32位缓冲区，存储当前的位数据
    u8 bits_in_buffer;      // 缓冲区中剩余的有效位数
    u8 read_bit_len;//一次读的长度
    u8 read_bit_mask;
    
    // 内联函数：填充缓冲区
    inline void fill_buffer() {
        // 当缓冲区位数少于24位时，尝试填充
        while (bits_in_buffer <= 24) {
            bit_buffer |= (static_cast<u32>(*inner_src) << bits_in_buffer);
            inner_src++;
            bits_in_buffer += 8;
        }
    }
    
public:
    // 构造函数：接受指向src指针的指针 - 移除IWRAM_CODE避免section冲突
    inline BitReader(const u8** src,int bits) : src_ptr(src),inner_src(*src), bit_buffer(0), bits_in_buffer(0) {
        read_bit_len = bits;
        read_bit_mask = (1<<bits) - 1;
        if(read_bit_len < 8){
            // 预填充缓冲区
            fill_buffer();
        }
    }
    
    // 读取指定位数的数据
    IWRAM_CODE inline u8 read() {
        if(read_bit_len == 8){
            return *inner_src++;
        }
        // 确保缓冲区有足够的位
        if (bits_in_buffer < read_bit_len) {
            fill_buffer();
        }
        
        // 提取所需的位
        u8 result = bit_buffer & (read_bit_mask);
        
        // 更新缓冲区
        bit_buffer >>= read_bit_len;
        bits_in_buffer -= read_bit_len;
        
        return result;
    }
    
    // 析构函数：确保src指针正确对齐到字节边界 - 移除IWRAM_CODE避免section冲突
    inline ~BitReader() {
        // 如果还有未消耗的位，需要调整src指针
        if (bits_in_buffer > 0) {
            // 回退到正确的位置
            inner_src -= (bits_in_buffer >> 3);
        }
        *src_ptr = inner_src;
    }
};

#endif // BIT_READER_H
