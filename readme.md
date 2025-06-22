## Ausar's GBA Video Player

这是Ausar编写的GBA视频播放器/转码器。

使用该项目的衍生项目必须开源。
未经本人同意，禁止用于商业用途。

目前这个还比较的粗糙，希望有能人志士完善这个播放器。

基本用法：
0.安装devkitpro libgba,以及python和相关库
1.用video_encoder.py对输入视频进行转码，各项转码的参数可以用`-h`选项来查看。
2.得到了`audio/vidio_data.c/h`这四个文件后，放到source目录编译即可

基本性能：可实现30FPS流畅播放，一分钟的纯视频在2.3MB-5.5MB不等，添加上音频后，由于采用的是PCM编码，因此体积增加直接等同于你的采样率。如16Khz的采样率，一秒的体积会增加16KB。

采用的基本技术：YUV420采样，VQ量化，I/P帧编码，频率编码，平坦区域上采样。


-----

给开发者的内部编码结构详解：

1.首先，输入视频会被切分为若干个2x2的像素块进行YUV420采样。即，得到4个Y值，一个U值和一个V值。（代码中叫Y,Cr,Cb）。由于GBA性能比较弱，因此，采用的是一个特殊的矩阵，这样可以用位运算来进行YUV到RGB的反向转换。而开发的后期，为了进一步加速解码，甚至采用的是预先运算的d_r,d_g,d_b值，只要和Y加在一起即可还原RGB

2.采用的是向量量化（VQ）技术进行编码，即，2x2块的YUV值会被当成一个向量，将许多块的值集合起来，采用Kmeans算法获得最具有代表性的256个值（实现中是255，留出一个值当标记位），被称为码表，然后画面中的所有2x2块均用码表中的值代替。

3.采用了I/P帧编码技术，I帧，指的是这一帧会传输所有的像素，而P帧，只传输与上一帧不同的像素，从而减少整体编码所需要的数据量。I帧，与随后的P帧（直到下一个I帧为止），被称为一个GOP（图片组），每个图片组中需要传输的2x2块像素，会进行单独的码表训练。训练得到的码表会随着I帧一起传输。

4.内部编码的方式如下：
通用的：
- 按照4X4大块为粒度进行编码。
- 一个4x4大块，可能由4个2x2的小块组成，即需要存储4个码表中的索引，这种情况称为纹理块。
- 一个4x4的大块，如果其可以由一个2x2的小块上采样得到，即将对应的像素复制4次，这种情况称为色块。
I帧：
- 按照从左到右，从上到下的顺序，依次编码所有的4x4块。
- 默认是纹理块模式，即，连续的4个uint8代表4个2x2小块在255项码表中的值。
- 但是，如果第一个u8的值是0xFF，即代表这是一个色块，此时紧随其后的一个u8，代表码表中的一个2x2像素，然后将这个2x2块上采样即得到4x4块。

P帧：
- 将240x160的屏幕划分成若干个240项的4x4块，这一个区域代码中称为zone。用一个bitmap来代表哪一个zone存在需要更新的数据。
- 由于每一个zone内最多只有240个4x4块，因此可以用u8来表示其位置索引。
- zone开始的时候，会有一个bitmap，表示了各个4x4块中，有几个2x2块需要更新。因此一个4x4块用4bit表示。为了方便解码，将两个4x4大块做一组编码。因此是一个u8的bitmap和两个u8的block_place_idx为一组。
- 每个4x4大块的1-4个codebook_idx，其编码可能是u4,u6或者u8（后面解释）。因此有一个bitreader类，专门用于处理这些bit流的读取。并且这些bitstream单独连续存放，以减少碎片。
- 基于这样一个观察：在码表中，各项的使用并不均匀，往往有几项占了绝大部分的使用率，因此我们可以考虑基于频率的编码。
    - 每个GOP的码表，会根据使用频率降序排列
    - 我引入了两种新的码表索引模式，分别称为"小码表"和"中码表"模式，小码表模式只能访问16项码表，中码表只能访问64项码表。因此，其索引分别是u4和u6
    - 小码表模式时，编码器会试图尝试使用0-15,16-31....这样的码表项来编码像素块。中码表模式时也同理。
    - 小码表用一个16bit的bitmap来表示，在哪几段里面存在需要更新的数据，而中码表用8bit来表示。
    - 只有一个4x4块中的4个2x2码块都能用同一个段内的小码表或者中码表表示时，才能进入对应的模式，否则用255项的全码表来兜底。

解码器的一些细节：
- 单缓冲设计：数据会先在EWRAM的buffer上准备好，再用DMA拷贝至
- 码表预解码：码表是YUV格式，但是解码器加载码表后，会先转换成u16的RGB555格式存储，这样解码的时候，只需要读码表的数据，并填入缓冲区即可，极大地加速了解码效率
- 码表预加载：当一帧解码完成，但是还没到显示的时机时，会试图搜索下一个I帧的位置，如果成功找到下一个I帧，那么会提前加载其码表数据。以减少I帧到来时，既要加载码表，又要解码码表，还需要渲染I帧的负担。
- 音频同步：每间隔64帧同步一次音频。采用PCM模式的音频播放。



----
TODO（希望有人能帮我改善的地方）：

- 音频：目前有低噪，我对音频基本一无所知，这方面有待优化。
- 目前编码器里面，没有加上对I帧索引的记录，造成需要在解码器里面手动搜索I帧位置，对性能有影响。
- 基本的播放控制功能，如暂停，前进，后退等。
- 解码器运行时抖动算法：由于GBA颜色太少，因此颜色显示有断层，需要增加抖动算法来缓解。但是，目前的预解码方式好像不好增加bayer抖动算法的样子，希望有人能优化。
- 奇怪的颜色问题：在白色边缘处，容易出现青色或者其他颜色的奇怪颜色，需要修正。
- 压缩率：现在的压缩率还有待提升，一个32MB的卡带，只能塞下一个7分钟左右的视频。
- 基本的播放控制功能，如暂停，前进，后退等。

---

## Ausar's GBA Video Player (English Translation)

This is a GBA video player/transcoder written by Ausar.

Derivative projects using this project must be open source.
Commercial use is prohibited without my permission.

This project is still quite rough at the moment. I hope talented individuals can help improve this player.

### Basic Usage:
0. Install devkitpro libgba, as well as Python and related libraries.
1. Use `video_encoder.py` to transcode the input video. You can view all transcoding parameters with the `-h` option.
2. After obtaining the four files `audio/video_data.c/h`, place them in the `source` directory and compile.

### Basic Performance:
- Achieves smooth playback at 30FPS.
- One minute of pure video takes about 2.3MB-5.5MB. After adding audio (PCM encoding), the size increases directly according to your sample rate. For example, at 16KHz, each second adds 16KB.

### Core Technologies Used:
- YUV420 sampling
- VQ quantization
- I/P frame encoding
- Frequency encoding
- Upsampling for flat regions

-----

### Internal Encoding Structure for Developers:

1. The input video is first divided into 2x2 pixel blocks for YUV420 sampling, resulting in 4 Y values and one U and V value (called Y, Cr, Cb in the code). Due to the GBA's limited performance, a special matrix is used so that YUV to RGB conversion can be done with bitwise operations. Later, for further decoding speed, precomputed d_r, d_g, d_b values are used, so RGB can be restored by simply adding them to Y.

2. Vector Quantization (VQ) is used for encoding: the YUV values of each 2x2 block are treated as a vector, and many such blocks are clustered using Kmeans to obtain the 256 most representative values (actually 255 in the implementation, reserving one as a marker), called the codebook. All 2x2 blocks in the frame are replaced by values from the codebook.

3. I/P frame encoding is used. I-frames transmit all pixels, while P-frames only transmit pixels that differ from the previous frame, reducing the overall data size. An I-frame and its following P-frames (until the next I-frame) form a GOP (Group of Pictures). Each GOP trains its own codebook for the 2x2 blocks that need to be transmitted. The codebook is transmitted together with the I-frame.

4. Internal encoding details:
- Uses 4x4 macroblocks as the encoding unit.
- A 4x4 macroblock may consist of four 2x2 sub-blocks, each needing to store a codebook index. This is called a texture block.
- If a 4x4 macroblock can be upsampled from a single 2x2 block (i.e., all pixels are the same), it is called a color block. The corresponding 2x2 block is upsampled to form the 4x4 block.

I-frames:
- Encode all 4x4 blocks from left to right, top to bottom.
- Default is texture block mode: four consecutive uint8s represent the indices of the four 2x2 sub-blocks in the 255-entry codebook.
- If the first u8 is 0xFF, it indicates a color block. The following u8 gives the codebook index for the 2x2 block, which is then upsampled to a 4x4 block.

P-frames:
- The 240x160 screen is divided into zones of 240 4x4 blocks each. Each zone is represented by a bitmap indicating which 4x4 blocks need updating.
- Since each zone has at most 240 4x4 blocks, their positions can be represented by u8 indices.
- At the start of each zone, a bitmap indicates which 4x4 blocks have 2x2 sub-blocks needing updates. Each 4x4 block uses 4 bits. For decoding convenience, two 4x4 blocks are grouped together, so each group uses a u8 bitmap and two u8 block_place_idx.
- Each 4x4 block's 1-4 codebook indices may be encoded as u4, u6, or u8 (see below). A bitreader class is used to handle these bitstreams, which are stored contiguously to reduce fragmentation.
- Observation: codebook usage is highly skewed, with a few entries used most frequently. Thus, frequency-based encoding is used:
    - Each GOP's codebook is sorted by usage frequency in descending order.
    - Two new codebook index modes are introduced: "small codebook" (16 entries, u4) and "medium codebook" (64 entries, u6). The encoder tries to use 0-15, 16-31, etc., for small/medium codebook modes.
    - Small codebook mode uses a 16-bit bitmap to indicate which segments have updates; medium codebook uses 8 bits.
    - Only if all four 2x2 sub-blocks in a 4x4 block can be represented within the same segment can the corresponding mode be used; otherwise, the full 255-entry codebook (u8) is used as fallback.

Decoder details:
- Single buffering: data is prepared in EWRAM buffer, then copied to VRAM via DMA.
- Codebook pre-decoding: codebooks are in YUV format, but after loading, are converted to u16 RGB555 for fast decoding.
- Codebook preloading: after decoding a frame but before display, the decoder tries to find the next I-frame and preload its codebook to reduce I-frame decoding/display lag.
- Audio sync: audio is synchronized every 64 frames. PCM audio playback is used.

----

### TODO (Areas for Improvement):

- Audio: Currently there is some noise; I have little knowledge of audio, so this needs optimization.
- The encoder does not record I-frame indices, so the decoder must search for I-frames manually, affecting performance.
- Decoder runtime dithering: due to GBA's limited colors, color banding occurs. Dithering should be added, but the current pre-decoding method makes Bayer dithering hard to implement.
- Strange color issues: cyan or other odd colors may appear at white edges; needs fixing.
- Compression ratio: still needs improvement. A 32MB cartridge can only hold about 7 minutes of video.
- Basic playback control functions, such as pause, forward, rewind, etc.