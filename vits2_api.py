import asyncio
import base64
import copy
import gc
import json
import logging
import logging.handlers
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional, Union, Literal

import aiofiles
import torch
import uvicorn
from fastapi import FastAPI
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, confloat
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from config import config
from infer_tts import generate, get_models
from utils.audio_utils import write_audio_data, get_mime_type, get_data_url
from utils.log_utils import logger


async def run_in_threadpool(func, *args):
    # 使用*符号后，所有多余的位置参数会被收集到一个元组中。
    # 使用**符号后，所有多余的关键字参数会被收集到一个字典中。
    # 异步包装器函数，用于运行同步代码
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, func, *args)
    # return status, audio_content
    return result


def torch_gc():
    """释放内存"""
    # 垃圾回收操作执行垃圾回收和 CUDA 缓存清空
    # Prior inference run might have large variables not cleaned up due to exception during the run.
    # Free up as much memory as possible to allow this run to be successful.
    gc.collect()
    if torch.cuda.is_available():  # 检查是否可用CUDA
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.ipc_collect()  # 收集CUDA内存碎片


def init_app():
    refresh_models(current_speaker, current_language)
    if torch.cuda.is_available():
        log = f'本次加载模型的设备为GPU: {torch.cuda.get_device_name(0)}'
    else:
        log = '本次加载模型的设备为CPU.'
    vits_logger.info(log)
    log = f"Service started!"
    vits_logger.info(log)


def configure_logging():
    log_file = 'logs/api.log'
    logger = logging.getLogger('vits')
    logger.setLevel(logging.INFO)
    handel_format = '%(asctime)s - %(levelname)s - %(message)s'
    # 设置 propagate 为 False
    # propagate 用于控制日志消息的传播行为，如果设置为 True（默认值），那么该 logger 记录的消息会向上层的 logger 传播，导致记录两次日志。
    logger.propagate = False
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    # 设置日志格式
    formatter = logging.Formatter(handel_format)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # 添加处理器到日志记录器
    logger.addHandler(console_handler)
    # logger.addHandler(file_handler)
    return logger


def get_current_time():
    current_time = datetime.now()
    return current_time


def analyze_path(default_dir, file_path, filee_name, file_format):
    # final_path = None
    # 检查路径是否为完整路径
    if os.path.isabs(file_path):
        # 检查是否有文件后缀
        if os.path.splitext(file_path)[1] and os.path.exists(os.path.dirname(file_path)):
            # 这是一个完整路径且父目录存在
            final_path = file_path
        elif os.path.isdir(file_path):
            # 这是一个存在的完整目录
            final_path = os.path.join(file_path, filee_name + file_format)
        else:
            # 路径不存在或者不合法
            logs = f"Invalid file_path provided, {file_path}"
            vits_logger.error(logs)
            raise ValueError(logs)
    elif file_path.find('/') == -1:
        # 不存在路径分隔符
        if os.path.splitext(file_path)[1]:
            # 这是一个带后缀的文件名
            final_path = os.path.join(default_dir, file_path)
        elif file_path == '':
            final_path = os.path.join(default_dir, filee_name + file_format)
        else:
            # 没有后缀名的文件名
            final_path = os.path.join(default_dir, file_path + file_format)
    else:
        if '/' in file_path and os.path.isdir(
                os.path.dirname(file_path) if os.path.splitext(file_path)[1] else file_path):
            # 不是绝对路径但包含路径分隔符且存在, 这是一个相对目录
            logs = f"Warning: {file_path}, 相对路径仅供测试, 请使用绝对路径"
            vits_logger.warning(logs)
            if os.path.splitext(file_path)[1]:
                final_path = os.path.abspath(file_path)
            else:
                file_path = os.path.join(file_path, filee_name + file_format)
                final_path = os.path.abspath(file_path)
        else:
            # 路径不存在或者不合法
            logs = f"Invalid file_path provided, {file_path}"
            vits_logger.error(logs)
            raise ValueError(logs)
    # 确保目录存在
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    return final_path


def refresh_models(speaker, language):
    global models_map, hps_map, device_map, request_count, current_speaker, current_language  # 声明全局变量
    if speaker != current_speaker or not models_map:
        current_speaker = speaker
        device_map, models_map, hps_map = get_models(models_config, speaker, language)
    elif current_language == 'mix':
        # 检查请求的语言是否不是 'mix'
        if language != 'mix':
            request_count += 1  # 增加请求计数器
            # 如果一定次数之后，语言都不是 'mix'，则重设current_language为'zh'
            if request_count > 10:  # 假设次数阈值为10
                current_language = 'zh'
                request_count = 0  # 重置请求计数器
                device_map, models_map, hps_map = get_models(models_config, speaker, language)
        else:
            request_count = 0  # 重置请求计数器
    elif language == 'mix':
        current_language = language
        device_map, models_map, hps_map = get_models(models_config, speaker, language)
    return device_map, models_map, hps_map


def get_audio_data(generator, default_samplerate, audio_samplerate, audio_format, write_mode):
    status, audio_data = list(generator)[0]
    read_stream = False
    # write_mode += 'without_stream'
    audio_contents = write_audio_data(audio_data, default_samplerate, audio_samplerate, audio_format, write_mode, read_stream)
    audio_content = list(audio_contents)[0]
    return status, audio_content


def stream_get_audio_data(generator, default_samplerate, audio_samplerate, audio_format, write_mode):
    start = time.process_time()
    read_stream = False if 'without_stream' in write_mode else True
    # write_mode += 'with_stream'
    target_len = 5 * 1024 * 1024  # 5M
    # 预设目标总长度（这里设置为 5MB）
    large_len = b'\xff\xff\xff\xff'  # 预设一个无限大的文件大小
    file_len = (target_len - 8).to_bytes(4, 'little')  # 4位小端序
    data_len = (target_len - 44).to_bytes(4, 'little')
    for index, audio_data in enumerate(generator):
        vits_logger.debug(f'---write_audio_data: {index}...')
        audio_format = '.bytes' if audio_format == '.wav' and index != 0 else audio_format
        # wav文件头为44位，每个文件头固定了数据块大小, raw不带文件头
        audio_contents = write_audio_data(audio_data, default_samplerate, audio_samplerate, audio_format, write_mode, read_stream)
        for idx, audio_content in enumerate(audio_contents):
            # vits_logger.debug(f'===write_audio_data: {idx}...')
            # if audio_format == '.wav' and write_mode != 'ffmpeg':
            if audio_format == '.wav':  # wav文件头为44位，每个文件头固定了数据块大小
                if idx == 0:
                    if index == 0:
                        # header_size = 44
                        # audio_content = audio_content[:header_size - 4] + audio_content[header_size:] if index == 0 else audio_content[header_size:]
                        wav_header = audio_content[:44]
                        audio_content = audio_content[44:]
                        # 创建新的文件头，替换文件大小ChunkSize字段和音频数据 Subchunk2Size 字段
                        # 复制 BIFF, fmt 子块（从第 8 字节到第 36 字节）, + data, 更新 Subchunk2Size
                        # audio_content = wav_header[:40] + audio_content
                        audio_content = wav_header[:4] + large_len + wav_header[8:40] + large_len + audio_content
                    else:
                        # header_size = 44
                        # audio_content = audio_content[:header_size - 4] + audio_content[header_size:] if index == 0 else audio_content[header_size:]
                        # wav_header = audio_content[:44]
                        audio_content = audio_content[44:]
                        # 创建新的文件头，替换文件大小ChunkSize字段和音频数据 Subchunk2Size 字段
                        # 复制 BIFF, fmt 子块（从第 8 字节到第 36 字节）, + data, 更新 Subchunk2Size
                        # audio_content = wav_header[:40] + audio_content
                        # audio_content = wav_header[:4] + large_len + wav_header[8:40] + large_len + audio_content
                else:
                    audio_content = audio_content
            yield audio_content
    end = time.process_time()
    vits_logger.info(f"time_all: {end - start}\n")


class GenerateRequest(BaseModel):
    sno: Union[int, str] = Field(default_factory=lambda: int(time.time() * 100))  # 动态生成时间戳
    uid: Union[int, str] = 'admin'
    text: str
    stream: Optional[bool] = None
    speaker: Literal['jt', 'huang', 'li'] = 'jt'
    language: Literal['zh', 'en', 'ja', 'mix'] = 'zh'
    speech_rate: confloat(ge=0.0, le=2.0) = 1.1  # speech_rate 必须是 0.0 到 2.0 之间的浮点数
    cut_by_sent: bool = False
    audio_format: str = ".wav"  # 默认值 ".wav"
    audio_path: Optional[str] = '/mnt/digital_service/audio'  # 音频路径或目录
    audio_samplerate: int = 44100
    write_mode: Literal['ffmpeg', 'package', 'ffmpeg_without_stream', 'package_without_stream'] = 'package'
    return_base64: Optional[bool] = Field(default=None, description="Whether to return base64 encoded audio")


class GenerateResponse(BaseModel):
    code: int
    sno: Optional[Union[int, str]] = None
    messages: str
    audio_path: Optional[str] = None  # 音频文件路径
    audio_base64: Optional[str] = None  # 音频 base64 编码


# 身份验证中间件
class BasicAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, secret_key: str):
        super().__init__(app)
        self.required_credentials = secret_key

    async def dispatch(self, request: Request, call_next):
        authorization: str = request.headers.get('Authorization')
        if authorization and authorization.startswith('Bearer '):
            provided_credentials = authorization.split(' ')[1]
            # 比较提供的令牌和所需的令牌
            if provided_credentials == self.required_credentials:
                return await call_next(request)
        # 返回一个带有自定义消息的JSON响应
        return JSONResponse(
            status_code=400,
            content={"detail": "Unauthorized: Invalid or missing credentials"},
            headers={'WWW-Authenticate': 'Bearer realm="Secure Area"'}
        )


# vits_logger = configure_logging()
vits_logger = logger
vits_app = FastAPI()
secret_key = os.getenv('VITS2-SECRET-KEY', 'sk-vits2')
vits_app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'], )
# vits_app.add_middleware(BasicAuthMiddleware, secret_key=secret_key)

# 创建一个线程池
executor = ThreadPoolExecutor(max_workers=3)
models_map = None
hps_map = None
device_map = None
request_count = 0  # 初始化请求计数器
current_speaker = 'jt'
current_language = 'zh'
models_config = config.api_config.models


@vits_app.get("/")
async def root():
    service_name = """
        <html> <head> <title>vits2_service</title> </head>
            <body style="display: flex; justify-content: center;"> <h1>vits2_service</h1></body> </html>
        """
    return HTMLResponse(content=service_name, status_code=200)


@vits_app.get("/health")
async def health():
    """Health check."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    health_data = {"status": "healthy", "timestamp": timestamp}
    # 返回JSON格式的响应
    return JSONResponse(content=health_data, status_code=200)


@vits_app.post("/v1/tts")
async def post_generate_audio(request: Request):
    try:
        # 判断请求的内容类型
        if request.headers.get('content-type') == 'application/json':
            json_data = await request.json()
            request = GenerateRequest(**json_data)
        else:
            # 解析表单数据
            form_data = await request.form()
            request = GenerateRequest(**form_data)
        start = time.process_time()
        text = request.text
        if not text:
            raise ValueError("The text must be provided and not empty.")
        request_data = request.model_dump()
        logs = f"TTS request param: {request_data}"
        vits_logger.info(logs)
        default_dir = '/mnt/digital_service/audio'
        default_samplerate = config.resample_config.sampling_rate
        audio_name = str(uuid.uuid4().hex[:8])
        sno = request.sno
        audio_path = request.audio_path
        audio_format = request.audio_format
        audio_samplerate = request.audio_samplerate
        stream = request.stream or False
        speaker = request.speaker
        language = request.language
        speech_rate = request.speech_rate
        write_mode = request.write_mode
        cut_by_sent = request.cut_by_sent
        return_base64 = request.return_base64
        sdp_ratio = 0.5
        noise_scale = 0.6
        noise_scale_w = 0.9
        para_interval = 0.5
        sent_interval = 0.3
        length_scale = 2.0 - speech_rate
        # length_scale = min(max(0.0, (2.0 - speech_rate)), 2.0)
        audio_path = analyze_path(default_dir, audio_path, audio_name, audio_format)
        logs = f"使用的音频路径为{audio_path}"  # 不一定会保存，如果流式
        refresh_models(speaker, language)
        vits_logger.info(logs)
        with torch.no_grad():
            # # 将GPU推理任务放入线程池中以实现异步处理
            # audio_content = await run_in_threadpool(generate, text, speaker, language, sdp_ratio, noise_scale,
            #                                 noise_scale_w, length_scale, cut_by_sent, para_interval, sent_interval)
            generator = generate(text, speaker, language, stream, audio_samplerate, hps_map, models_map,
                                 device_map, sdp_ratio, noise_scale, noise_scale_w,
                                 length_scale, cut_by_sent, para_interval, sent_interval)
            if not stream:
                end1 = time.process_time()
                vits_logger.debug(f"time_1: {end1 - start}")
                # 将流式生成器放入线程池
                status, audio_content = await run_in_threadpool(get_audio_data, generator, default_samplerate, audio_samplerate, audio_format, write_mode)
                # status, audio_content = await get_audio_data(generator, audio_data, default_samplerate, audio_samplerate, audio_format, write_mode)
                # status, audio_data = list(generator)[0]
                end2 = time.process_time()
                vits_logger.debug(f"time_2: {end2 - end1}")
                # audio_content = await write_audio_data(audio_data, default_samplerate, audio_samplerate, audio_format, write_mode)
                end3 = time.process_time()
                vits_logger.debug(f"time_3: {end3 - end2}")
                # 写入到文件
                async with aiofiles.open(audio_path, mode='wb') as file:
                    await file.write(audio_content)
                # if request.return_base64:
                    # 将音频内容编码为 base64
                data_url = get_data_url(audio_format)
                audio_base64 = base64.b64encode(audio_content).decode('utf-8')
                audio_base64 = f'{data_url},{audio_base64}'
                # len('data:audio/wav;base64,'): 22
                audio_base64_log = audio_base64[:30] + "..." + audio_base64[-20:]  # 只记录前30个字符
                if not return_base64:
                    audio_base64 = audio_base64_log
            else:
                # 将流式生成器放入线程池
                stream_generator = await run_in_threadpool(stream_get_audio_data, generator, default_samplerate,
                                                           audio_samplerate, audio_format, write_mode)
                # stream_generator = stream_write_audio_data(generator, default_samplerate, audio_samplerate, audio_format, write_mode)
                media_type = get_mime_type(audio_format) if audio_format.replace('.', '') != 'mp3' else 'audio/mp3'
                end = time.process_time()
                vits_logger.info(f"time_all: {end - start}")
                return StreamingResponse(stream_generator, media_type=media_type)
            end4 = time.process_time()
            vits_logger.debug(f"time_4: {end4-end3}")
        # torch_gc()
        code = 0 if status != 'Failed' else -1
        messages = f"Generate audio status is {status}!"
        results = GenerateResponse(
            code=code,
            sno=sno,
            messages=messages,
            audio_path=audio_path,
            audio_base64=audio_base64
        )
        results_log = copy.deepcopy(results)
        if audio_base64:
            results_log.audio_base64 = audio_base64_log
        logs = f"TTS response results: {results_log}"
        vits_logger.info(logs)
        end = time.process_time()
        vits_logger.debug(f"time_5: {end - end4}")
        vits_logger.info(f"time_all: {end - start}\n")
        return JSONResponse(status_code=200, content=results.model_dump())
    except json.JSONDecodeError as je:
        error_message = GenerateResponse(
            code=-1,
            messages=f"JSONDecodeError, Invalid JSON format: {str(je)} "
        )
        logs = f"Generate audio  error: {error_message}\n"
        vits_logger.error(logs)
        return JSONResponse(status_code=400, content=error_message.model_dump())
    # except Exception as e:
    #     error_message = GenerateResponse(
    #         code=-1,
    #         messages=f"Exception: {str(e)} "
    #     )
    #     logs = f"Generate audio  error: {error_message}\n"
    #     vits_logger.error(logs)
    #     return JSONResponse(status_code=500, content=error_message.model_dump())


@vits_app.get("/v1/tts")
# async def generate_audio(request: GenerateRequest,  # GET 请求没有请求体, 无法直接使用GenerateRequest
#                          text: str = Query(..., min_length=2, description="The text to synthesize")):
#     try:
#         text = request.text or text
#         request_data = request.model_dump()
#         logs = f"TTS request param: {request_data}"
#         vits_logger.info(logs)
#         uid = request.uid
#         sno = request.sno
#         audio_path = request.audio_path
#         audio_format = request.audio_format
#         audio_samplerate = request.audio_samplerate
#         stream = request.stream or False
#         speaker = request.speaker
#         language = request.language
#         speech_rate = request.speech_rate
#         write_mode = request.write_mode
#         cut_by_sent = request.cut_by_sent
#         return_base64 = request.return_base64
async def get_generate_audio(
        sno: Union[int, str] = 1024,
        uid: Union[int, str] = 'admin',
        text: str = Query(..., min_length=2, description="The text to synthesize"),  # 必需参数，没有默认值
        stream: Optional[bool] = None,
        speaker: Literal['jt', 'huang', 'li'] = 'jt',
        language: Literal['zh', 'en', 'ja', 'mix'] = 'zh',
        speech_rate: confloat(ge=0.0, le=2.0) = 1.1,  # speech_rate 必须是 0.0 到 2.0 之间的浮点数
        cut_by_sent: bool = False,
        audio_format: str = ".wav",  # 默认值 ".wav"
        audio_path: Optional[str] = '/mnt/digital_service/audio',  # 音频路径或目录
        audio_samplerate: int = 44100,
        write_mode: Literal['ffmpeg', 'package', 'ffmpeg_without_stream', 'package_without_stream'] = 'package',
        return_base64: Optional[bool] = None
):
    try:
        start = time.process_time()
        sdp_ratio = 0.5
        noise_scale = 0.6
        noise_scale_w = 0.9
        para_interval = 0.5
        sent_interval = 0.3
        length_scale = 2.0 - speech_rate
        stream = stream or True
        if not text:
            raise ValueError("The text must be provided and not empty.")
        # 创建 request_data 字典
        request_data = {
            "sno": sno,
            "uid": uid,
            "text": text,
            "stream": stream,
            "speaker": speaker,
            "language": language,
            "speech_rate": speech_rate,
            "cut_by_sent": cut_by_sent,
            "audio_format": audio_format,
            "audio_path": audio_path,
            "audio_samplerate": audio_samplerate,
            "write_mode": write_mode,
            "return_base64": return_base64
        }

        logs = f"TTS request param: {request_data}"
        vits_logger.info(logs)
        default_dir = '/mnt/digital_service/audio'
        default_samplerate = config.resample_config.sampling_rate
        audio_name = str(uuid.uuid4().hex[:8])
        audio_path = analyze_path(default_dir, audio_path, audio_name, audio_format)
        logs = f"使用的音频路径为{audio_path}"  # 不一定会保存，如果流式
        refresh_models(speaker, language)
        vits_logger.info(logs)

        with torch.no_grad():
            generator = generate(text, speaker, language, stream, audio_samplerate, hps_map, models_map,
                                 device_map, sdp_ratio, noise_scale, noise_scale_w,
                                 length_scale, cut_by_sent, para_interval, sent_interval)
            if not stream:
                # 将流式生成器放入线程池
                status, audio_content = await run_in_threadpool(get_audio_data, generator, default_samplerate,
                                                                audio_samplerate, audio_format, write_mode)
                # 写入到文件
                async with aiofiles.open(audio_path, mode='wb') as file:
                    await file.write(audio_content)
                # 将音频内容编码为 base64
                data_url = get_data_url(audio_format)
                audio_base64 = base64.b64encode(audio_content).decode('utf-8')
                audio_base64 = f'{data_url},{audio_base64}'
                audio_base64_log = audio_base64[:30] + "..."  # 只记录前30个字符
                if not return_base64:
                    audio_base64 = audio_base64_log
            else:
                # 流式生成
                stream_generator = await run_in_threadpool(stream_get_audio_data, generator, default_samplerate,
                                                           audio_samplerate, audio_format, write_mode)
                media_type = get_mime_type(audio_format) if audio_format.replace('.', '') != 'mp3' else 'audio/mp3'
                # end = time.process_time()
                # vits_logger.info(f"time_all: {end - start}\n")
                return StreamingResponse(stream_generator, media_type=media_type)

        code = 0 if status != 'Failed' else -1
        messages = f"Generate audio status is {status}!"
        results = GenerateResponse(
            code=code,
            sno=sno,
            messages=messages,
            audio_path=audio_path,
            audio_base64=audio_base64
        )

        logs = f"TTS response results: {results}"
        vits_logger.info(logs)
        end = time.process_time()
        vits_logger.info(f"time_all: {end - start}\n")
        return JSONResponse(status_code=200, content=results.model_dump())
    except json.JSONDecodeError as je:
        error_message = GenerateResponse(
            code=-1,
            messages=f"JSONDecodeError, Invalid JSON format: {str(je)} "
        )
        logs = f"Generate audio error: {error_message}\n"
        vits_logger.error(logs)
        return JSONResponse(status_code=400, content=error_message.model_dump())
    # except Exception as e:
    #     error_message = GenerateResponse(
    #         code=-1,
    #         messages=f"Exception: {str(e)} "
    #     )
    #     logs = f"Generate audio error: {error_message}\n"
    #     vits_logger.error(logs)
    #     return JSONResponse(status_code=500, content=error_message.model_dump())


@vits_app.get('/audio/generate', response_class=HTMLResponse)
async def convert_audio(
        request: Request,
):
    with open("./audio_generate.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    init_app()
    host = config.api_config.host
    port = config.api_config.port
    uvicorn.run(vits_app, host=host, port=8031)


