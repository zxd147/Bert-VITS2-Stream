import gc
import os
import subprocess
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from queue import Queue

from utils.log_utils import logger
import numpy as np
import pydub
import pyloudnorm as pylon
import soundfile as sf
import torch
from scipy.signal import resample

# 支持的音频格式
supported_formats = {'wav', 'mp3', 'pcm', 'flac', 'ogg', 'aac', 'wma', 'm4a', 'raw', 'bytes', 'opus', 'aiff'}
# 设置输入格式的映射
subtype_format_map = {
    np.float32: 'f32le',  # 输入格式为 32 位浮点小端格式 PCM
    np.float16: 'f16le',
    np.int16: 's16le',
    np.int32: 's32le',
    np.uint8: 'u8',
}
# ffmpeg 命令中的参数, 根据格式匹配音频编码器和输出文件的容器格式
codec_format_map = {
    'wav': {'codec': 'pcm_s16le', 'format': 'wav'},
    'mp3': {'codec': 'libmp3lame', 'format': 'mp3'},
    'pcm': {'codec': 'pcm_s16le', 'format': 'raw'},
    'flac': {'codec': 'flac', 'format': 'flac'},
    'ogg': {'codec': 'libvorbis', 'format': 'ogg'},
    'aac': {'codec': 'aac', 'format': 'adts'},
    'wma': {'codec': 'wmav2', 'format': 'asf'},  # wma或者asf
    'm4a': {'codec': 'aac', 'format': 'mp4'},
    'opus': {'codec': 'libopus', 'format': 'ogg'},  # Opus 格式
    'aiff': {'codec': 'pcm_s16be', 'format': 'aiff'},  # AIFF 格式, usually use at Apple.
    # You can use the format ipod to export to m4a (see original answer) ['matroska', 'mp4', 'ipod']
    'raw': {'codec': 'pcm_s16le', 'format': 'raw'},
    'bytes': {'codec': 'pcm_s16le', 'format': 'raw'},
}
# see original answer: https://stackoverflow.com/questions/62598172/m4a-mp4-audio-file-encoded-with-pydubffmpeg
# -doesnt-play-on-android
# 定义音频格式与其MIME类型的映射
mime_types_map = {
    'wav': 'audio/wav',
    'mp3': 'audio/mpeg',  # or 'audio/mp3'
    'pcm': 'audio/pcm',  # 通常PCM格式的MIME类型
    'flac': 'audio/flac',
    'ogg': 'audio/ogg',
    'aac': 'audio/aac',
    'wma': 'audio/wma',  # WMA格式的MIME类型, 'x-ms-wma'
    'm4a': 'audio/m4a',  # M4A格式的MIME类型
    'opus': 'audio/opus',  # Opus格式的MIME类型
    'aiff': 'audio/aiff',  # AIFF格式的MIME类型
    'raw': 'audio/raw',  # PCM格式通常使用audio/pcm，但具体MIME类型可能取决于PCM数据的字节序和位深 ['pcm', 'raw']
    'bytes': 'audio/pcm',  # 通常PCM格式的MIME类型
}

# 使用线程解决在调用 soundfile 库时可能遇到的堆栈溢出问题
# 通常只需要在程序启动时设置一次，之后的线程池中的线程将使用这个设置。
stack_size = 4096 * 4096
threading.stack_size(stack_size)


# 垃圾回收操作执行垃圾回收和 CUDA 缓存清空
def torch_gc():
    """释放内存"""
    # Prior inference run might have large variables not cleaned up due to exception during the run.
    # Free up as much memory as possible to allow this run to be successful.
    gc.collect()
    if torch.cuda.is_available():  # 检查是否可用CUDA
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.ipc_collect()  # 收集CUDA内存碎片


def write_audio_data(audio_data, default_samplerate, sampling_rate, audio_format, write_mode, read_stream,
                     normalize=True):
    audio_data = audio_data.astype(np.float32)
    audio_format = audio_format.replace('.', '')
    # 如果当前采样率与默认采样率不一致，进行重采样
    if sampling_rate != default_samplerate and np.any(audio_data):
        # 计算重采样后的样本数量
        num_samples = int(float(len(audio_data)) * float(sampling_rate) / float(default_samplerate))
        audio_data = resample(audio_data, num_samples)
    # normalize（规范化）, 目的是确保音频文件在播放时具有适当的响度，并且不会因为振幅过大而导致失真。
    if normalize and np.any(audio_data):
        meter = pylon.Meter(sampling_rate, block_size=0.020)  # 时间= 44100Hz/1024样本≈0.0232秒
        loudness = meter.integrated_loudness(audio_data)
        audio_data = pylon.normalize.loudness(audio_data, loudness, -18.0)
        audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
    # See: https://docs.python.org/3/library/threading.html
    # The stack size of this thread is at least 32768
    # If stack overflow error still occurs, just modify the `stack_size`.
    # stack_size = n * 4096, where n should be a positive integer.
    # Here we chose n = 4096.
    # 同步缓冲区，不需要 await
    # 使用上下文管理器来管理其生命周期。
    audio_data = (audio_data * 2 ** 15).astype(np.int16)
    with BytesIO() as buffer:
        # buffer.truncate(0)  # BytesIO 的实例在创建时默认是空的
        # buffer.seek(0)  # 指针已经在开始位置。
        try:
            start = time.process_time()
            if audio_format not in supported_formats:
                raise ValueError("Unsupported file format")
            elif audio_format == 'bytes':
                # 处理 bytes 格式
                if read_stream:
                    buffer.write(audio_data.tobytes())
                    buffer.seek(0)  # 读取前，确保缓冲区指针在开始位置
                    for audio_content in iter(lambda: buffer.read(1024), b''):
                        yield audio_content
                else:
                    buffer.write(audio_data.tobytes())
                    buffer.seek(0)  # 读取前，确保缓冲区指针在开始位置
                    audio_content = buffer.getvalue()
                    yield audio_content
            else:
                is_ffmpeg = 'ffmpeg' in write_mode
                write_map = {0: {0: direct_package,
                                 1: direct_ffmpeg},
                             1: {0: stream_package,
                                 1: stream_ffmpeg}}
                yield from write_map[read_stream][is_ffmpeg](buffer, audio_data, sampling_rate, audio_format)
            end = time.process_time()
            logger.debug(f"write_mode: {write_mode}, process_time: {end - start}")
        except RuntimeError as e:
            # If changing the thread stack size is unsupported, a RuntimeError is raised.
            logger.error(f"RuntimeError: {e}")
            # logger.info("Please changing the thread stack size is unsupported.")
        except ValueError as e:
            # If the specified stack size is invalid, a ValueError is raised and the stack size is unmodified.
            logger.error(f"ValueError: The specified stack size is invalid or other value error: {e}.")


def stream_ffmpeg(_, audio_data, sampling_rate, audio_format):
    # "ffmpeg_with_stream", "ffmpeg_without_stream_with_stream"
    # if read_stream and 'ffmpeg' in write_mode:
    ffmpeg_args = write_with_ffmpeg(audio_data, sampling_rate, audio_format)
    # 使用子进程调用 ffmpeg
    # logger.info(f"process_args: {ffmpeg_args}")
    process = subprocess.Popen(ffmpeg_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    # 将 audio_data 转换为字节流
    audio_bytes = audio_data.tobytes()

    def write_input(audio_chunk):
        try:
            process.stdin.write(audio_chunk)
        finally:
            process.stdin.close()

    def stream_output(ffmpeg_process, chunk_size=1024):
        while True:
            audio = ffmpeg_process.stdout.read(chunk_size)
            if not audio:
                break
            yield audio

    input_thread = threading.Thread(target=write_input, args=(audio_bytes,))
    input_thread.start()
    try:
        for audio_data in stream_output(process):
            yield audio_data
    finally:
        process.stdout.close()
        process.stderr.close()
        process.wait()
        input_thread.join()
    process.wait()  # 等待进程完成
    # 检查错误信息
    if process.returncode != 0:
        error_data = process.stderr.read().decode()
        raise RuntimeError(f'FFmpeg error: {error_data}')
    # try:
    #     # 读取输出数据并流式生成, iter 函数将 process.stdout 包装为一个迭代器
    #     for audio_content in iter(lambda: process.stdout.read(1024), b''):
    #         yield audio_content  # 确保返回字节数据

    # out, err = process.communicate(input=audio_bytes)
    # # 写入缓冲区
    # if process.returncode == 0:
    #     buffer.write(out)
    # else:
    #     logger.error(f"Error occurred: {err.decode()}")
    # buffer.seek(0)  # 读取前，确保缓冲区指针在开始位置
    # audio_content = buffer.getvalue()
    # yield audio_content


def stream_package(buffer, audio_data, sampling_rate, audio_format):
    # if read_stream and 'package' in write_mode:
    # if all(keyword in write_mode for keyword in ["stream", "soundfile"]):
    with ThreadPoolExecutor(max_workers=10) as executor:  # 限制最大线程数
        future = executor.submit(write_with_soundfile, buffer, audio_data, sampling_rate, audio_format)
        future.result()  # 等待线程完成
        buffer.seek(0)  # 回到缓冲区的起始位置，准备读取
        # 使用 for 循环逐块读取缓冲区数据并返回，每块大小为 1024 字节
        # yield from buffer.read(1024)
        for audio_content in iter(lambda: buffer.read(1024), b''):
            yield audio_content
        # buffer.seek(0)  # 回到缓冲区的起始位置，准备读取
        # yield buffer


def direct_package(buffer, audio_data, sampling_rate, audio_format):
    # if 'package' in write_mode:
    with ThreadPoolExecutor(max_workers=10) as executor:  # 限制最大线程数
        future = executor.submit(write_with_soundfile, buffer, audio_data, sampling_rate,
                                 audio_format)
        future.result()  # 等待线程完成
        buffer.seek(0)  # 读取前，确保缓冲区指针在开始位置
        audio_content = buffer.getvalue()
        yield audio_content


def direct_ffmpeg(_, audio_data, sampling_rate, audio_format):
    with ThreadPoolExecutor(max_workers=10) as executor:  # 限制最大线程数
        args_queue = Queue()
        pack_ogg_thread = threading.Thread(target=write_with_ffmpeg,
                                           args=(
                                               audio_data, sampling_rate, audio_format, args_queue))
        pack_ogg_thread.start()
        pack_ogg_thread.join()
        ffmpeg_args = args_queue.get()
        process = subprocess.Popen(ffmpeg_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        # 将 audio_data 转换为字节流
        audio_bytes = audio_data.tobytes()
        audio_content, stderr = process.communicate(input=audio_bytes)
        # # 读取输出
        # output_data = process.stdout.read()
        # process.stdout.close()
        # # 等待输入线程完成
        # input_thread.join()
        process.stdout.close()  # 确保关闭标准输出流
        process.stderr.close()
        yield audio_content  # 返回所有输出数据
        process.wait()  # 等待进程完成
        # 检查错误信息
        if process.returncode != 0:
            error_data = process.stderr.read().decode()
            raise RuntimeError(f'FFmpeg error: {error_data}')

        # # 写入缓冲区
        # if process.returncode == 0:
        #     buffer.write(out)
        # else:
        #     logger.error(f"Error occurred: {err.decode()}")
    # buffer.seek(0)  # 读取前，确保缓冲区指针在开始位置
    # audio_content = buffer.getvalue()
    # buffer.truncate(0)  # BytesIO 的实例在创建时默认是空的
    # buffer.seek(0)
    # # 不需要显式调用 buffer.close()
    # buffer.close()  # 关闭对象
    # yield audio_content


def write_with_soundfile(buffer, audio_data, sampling_rate, audio_format):
    codec = codec_format_map[audio_format]['codec'] if audio_format != 'wav' else None
    out_format = codec_format_map[audio_format]['format']
    # MP3 格式，应该不需要设置 subtype。因为 MP3 是有损格式，而 PCM 是无损格式，所以它们不兼容。
    subtype, _ = get_subtype_format(audio_data) if audio_format not in ['mp3', 'ogg'] else (None, None)
    if audio_format == 'wav':
        # 打开 WAV 文件进行写入
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)  # 设置声道数，1 表示单声道
            wf.setsampwidth(2)  # 设置样本宽度，2 字节 = 16 位 / 8
            wf.setframerate(sampling_rate)  # 设置采样率
            wf.setnframes(len(audio_data) // 2)  # 设置总帧数（假设是 16 位单声道）
            wf.writeframes(audio_data.tobytes())  # 写入音频数据
    elif audio_format in ['wav', 'mp3', 'ogg', 'flac']:
        # 支持的数据类型有 float64、float32、int32 和 int16。
        sf.write(buffer, audio_data, sampling_rate, format=audio_format, subtype=subtype)
    elif audio_format in ['pcm', 'raw']:
        sf.write(buffer, audio_data, sampling_rate, format='raw', subtype=subtype)
    # pydub 库背后实际使用 ffmpeg 来执行实际的格式转换和处理工作。
    elif audio_format in ['wav', 'mp3', 'aac', 'wma', 'm4a', 'opus', 'aiff', '...']:  # 使用列表后续可扩展
        audio = pydub.AudioSegment(audio_data.tobytes(), sample_width=2, frame_rate=sampling_rate, channels=1)
        audio.export(buffer, format=out_format, codec=codec, bitrate="192k")
    elif audio_format == 'bytes':
        # 处理 bytes 格式
        buffer.write(audio_data.tobytes())
    else:
        raise ValueError("Unsupported file format")
    # BytesIO 是可变对象，引用同一个 buffer 对象。在子线程中对 buffer 的修改会影响主线程中的 buffer，主线程能正确获取到写入的结果。
    return buffer


def write_with_ffmpeg(audio_data, sampling_rate, audio_format, args_queue=None):
    # 针对不同的 audio_format 设置输出参数
    # {'codec': 'pcm_s16le', 'format': 'wav'},
    codec = codec_format_map[audio_format]['codec']
    out_format = codec_format_map[audio_format]['format']
    # audio_data_dtype = audio_data.dtype
    # input_format = subtype_format_map.get(np.dtype(audio_data_dtype), 'f32le')
    _, input_format = get_subtype_format(audio_data)
    os.environ['FFMPEG_BUFFER_SIZE'] = str(1024 * 1024 * 3)  # 设置为1MB
    # 设置 ffmpeg 的输入参数
    process_args = [
        'ffmpeg',
        '-y',
        '-f', input_format,
        '-ar', str(sampling_rate),  # 设置采样率
        '-ac', '1',  # 单声道
        '-i', 'pipe:0',  # 从标准输入读取
        '-b:a', '192k',  # 比特率
        '-vn',  # 不处理视频
        '-map_metadata', '-1',  # 禁用元数据
        # '-metadata', 'title=""',  # 额外确保无标题
        # '-metadata', 'artist=""',  # 额外确保无艺术家信息
        # '-metadata', 'album=""',  # 额外确保无专辑信息
        # '-movflags',
        # '+faststart',
    ]
    # 对于acc, AC编码器需要输出流的元数据（例如文件头），而在管道流中，FFmpeg无法提前确定输出文件的大小和结构，从而导致"muxer does not support non seekable output"错误。
    if audio_format == 'wav':
        process_args.extend(['-fflags', '+bitexact']),  # 标准WAV（44个字节的头部）
    process_args.extend(['-c:a', codec])
    process_args.extend(['-f', out_format])
    process_args.append('pipe:1')  # 输出到标准输出
    if args_queue:
        # 将 process_args 放入队列
        args_queue.put(process_args)
    return process_args


def get_subtype_format(audio_data):
    if np.issubdtype(audio_data.dtype, np.floating):
        if audio_data.dtype == np.float32:
            return 'FLOAT', 'f32le'
        elif audio_data.dtype == np.float64:
            return 'FLOAT', 'f64le'  # 返回对应的subtype
    elif np.issubdtype(audio_data.dtype, np.integer):
        if audio_data.dtype == np.int16:
            return 'PCM_16', 's16le'
        elif audio_data.dtype == np.int32:
            return 'PCM_32', 's32le'
    elif audio_data.dtype == np.uint8:
        return 'PCM_U8', 'u8'
    return 'FLOAT', 'f32le'  # 默认格式及其subtype


def get_mime_type(audio_format='.wav'):
    audio_format = audio_format.replace('.', '')
    mime_type = mime_types_map.get(audio_format.lower(), 'audio/wav')  # 默认为wav
    return mime_type


def get_data_url(audio_format):
    # 根据音频格式生成MIME类型
    mime_type = get_mime_type(audio_format)
    # 生成Data URL
    data_url = f"data:{mime_type};base64"
    return data_url
