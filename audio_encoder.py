#!/usr/bin/env python3

import subprocess
import tempfile
import os
import struct
from pathlib import Path
from pydub import AudioSegment
import numpy as np

class AudioEncoder:
    """音频编码器，从视频文件中提取音频并转换为GBA格式"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.bytes_per_second = sample_rate * 1 * 1  # 采样率 * 声道数 * 字节宽度
    
    def extract_audio_from_video(self, video_path: str, duration: float, start_time: float = 0.0, i_frame_timestamps=None, frame_count=None, volume_percent=100):
        """
        从视频文件中提取音频，并可调整音量（百分比）
        """
        print(f"正在从视频中提取音频...")
        print(f"  音频采样率: {self.sample_rate} Hz")
        print(f"  音频时长: {duration:.2f} 秒")
        print(f"  音量调整: {volume_percent}%")
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        try:
            # 使用ffmpeg提取音频
            cmd = ["ffmpeg", "-i", video_path]
            if start_time > 0:
                cmd.extend(["-ss", str(start_time)])
            if duration > 0:
                cmd.extend(["-t", str(duration)])
            cmd.extend([
                "-vn",                   # 不包含视频
                "-acodec", "pcm_s16le",  # 16位PCM编码
                "-ar", "44100",          # 先用44.1kHz提取
                "-ac", "1",              # 单声道
                "-y",                    # 覆盖输出文件
                temp_audio_path
            ])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ ffmpeg提取音频失败: {result.stderr}")
                return None
            
            # 使用pydub加载和处理音频
            sound = AudioSegment.from_file(temp_audio_path)
            sound = sound.set_channels(1).set_frame_rate(self.sample_rate).set_sample_width(1)
            # 音量调整
            if volume_percent != 100:
                sound = sound + (20 * np.log10(volume_percent / 100.0))
            # 获取实际音频时长
            actual_audio_duration = sound.duration_seconds
            if actual_audio_duration < duration:
                print(f"⚠️ 实际音频时长({actual_audio_duration:.2f}s)小于请求时长({duration:.2f}s)，自动截断到实际时长。")
                duration = actual_audio_duration
            
            # 获取原始PCM数据
            raw_data = sound.raw_data
            pcm8 = list(raw_data)
            
            # 确保4字节对齐
            while len(pcm8) % 4 != 0:
                pcm8.append(0x80)  # GBA静音中心值
            
            # 计算每帧音频偏移
            frame_audio_offsets = None
            if frame_count is not None:
                frame_audio_offsets = []
                for i in range(frame_count):
                    timestamp = i / frame_count * duration
                    offset = int(timestamp * self.bytes_per_second)
                    if offset < len(pcm8):
                        frame_audio_offsets.append(offset)
                    else:
                        frame_audio_offsets.append(len(pcm8) - 1)
            
            # 计算I帧音频偏移
            i_frame_audio_offsets = []
            if i_frame_timestamps:
                for timestamp in i_frame_timestamps:
                    # 将时间戳转换为音频字节偏移
                    offset = int(timestamp * self.bytes_per_second)
                    if offset < len(pcm8):
                        i_frame_audio_offsets.append(offset)
                    else:
                        i_frame_audio_offsets.append(len(pcm8) - 1)  # 防止越界
            
            print(f"✓ 音频提取完成: {len(pcm8)} 字节")
            print(f"✓ I帧音频偏移: {len(i_frame_audio_offsets)} 个偏移点")
            
            return bytes(pcm8), i_frame_audio_offsets, frame_audio_offsets
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
    
    def write_audio_header(self, path_h: Path, audio_data: bytes, duration: float, i_frame_audio_offsets=None, frame_audio_offsets=None):
        """生成音频头文件"""
        guard = "AUDIO_DATA_H"
        
        with path_h.open("w", encoding="utf-8") as f:
            f.write(f"""#ifndef {guard}
#define {guard}

#include <gba_types.h>

// 音频参数
#define SAMPLE_RATE {self.sample_rate}
#define AUDIO_BYTES_PER_SECOND {self.bytes_per_second}
#define AUDIO_DURATION_MS {int(duration * 1000)}
#define audio_data_len {len(audio_data)}

// I帧音频偏移
#define I_FRAME_AUDIO_OFFSET_COUNT {len(i_frame_audio_offsets) if i_frame_audio_offsets else 0}

// 音频数据声明
extern const unsigned char audio_data[audio_data_len];
""")
            
            if i_frame_audio_offsets:
                f.write("extern const unsigned int i_frame_audio_offsets[I_FRAME_AUDIO_OFFSET_COUNT];\n")
            
            if frame_audio_offsets:
                f.write(f"#define FRAME_AUDIO_OFFSET_COUNT {len(frame_audio_offsets)}\n")
                f.write(f"extern const unsigned int frame_audio_offsets[FRAME_AUDIO_OFFSET_COUNT];\n")
            
            f.write(f"\n#endif // {guard}\n")
    
    def write_audio_source(self, path_c: Path, audio_data: bytes, i_frame_audio_offsets=None, frame_audio_offsets=None):
        """生成音频源文件"""
        with path_c.open("w", encoding="utf-8") as f:
            f.write('#include "audio_data.h"\n\n')
            
            # 转换为32位字（小端序）
            words = []
            for i in range(0, len(audio_data), 4):
                word = (audio_data[i] | 
                       (audio_data[i+1] << 8) | 
                       (audio_data[i+2] << 16) | 
                       (audio_data[i+3] << 24))
                words.append(word)
            
            f.write("const unsigned char audio_data[] = {\n")
            per_line = 16
            for i in range(0, len(audio_data), per_line):
                chunk = ', '.join(f"0x{v:02X}" for v in audio_data[i:i+per_line])
                f.write("    " + chunk + ",\n")
            f.write("};\n")
            
            # 写入I帧音频偏移数组
            if i_frame_audio_offsets:
                f.write("\nconst unsigned int i_frame_audio_offsets[] = {\n")
                per_line = 8
                for i in range(0, len(i_frame_audio_offsets), per_line):
                    chunk = ', '.join(f"{offset}" for offset in i_frame_audio_offsets[i:i+per_line])
                    f.write("    " + chunk + ",\n")
                f.write("};\n")
            
            # 写入每帧音频偏移数组
            if frame_audio_offsets:
                f.write("\nconst unsigned int frame_audio_offsets[] = {\n")
                per_line = 8
                for i in range(0, len(frame_audio_offsets), per_line):
                    chunk = ', '.join(f"{offset}" for offset in frame_audio_offsets[i:i+per_line])
                    f.write("    " + chunk + ",\n")
                f.write("};\n") 