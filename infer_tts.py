import gc
import os
import re
import time

import gradio as gr
import numpy as np
import torch
from cn2an import an2cn

from config import config
from infer import get_model, infer
from tools.sentence import split_by_language
from utils.log_utils import logger
from utils.match_utils import split_sentences, split_paragraphs
from utils.model_utils import get_hparams_from_file

# 模型加载和设备设置
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
model_instances = {}


# 垃圾回收操作执行垃圾回收和 CUDA 缓存清空
def torch_gc():
    """释放内存"""
    # Prior inference run might have large variables not cleaned up due to exception during the run.
    # Free up as much memory as possible to allow this run to be successful.
    gc.collect()
    if torch.cuda.is_available():  # 检查是否可用CUDA
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.ipc_collect()  # 收集CUDA内存碎片


def get_models(models_config, speaker, language):
    """加载模型和超参数"""
    # 构建一个字典，将 speaker 和 language 映射到模型配置
    global model_instances
    hps_map = {}
    models_map = {}
    device_map = {}
    temp_model_instances = {}
    speaker_models_config = models_config[speaker]
    languages = [language] if language != 'mix' else speaker_models_config.keys()
    for lang in languages:
        save_one_model(speaker_models_config, lang, temp_model_instances, device_map, models_map, hps_map)
    model_instances = temp_model_instances
    del temp_model_instances
    torch_gc()
    return device_map, models_map, hps_map


def save_one_model(speaker_models_config, language, temp_model_instances, device_map, models_map, hps_map):
    model_path = speaker_models_config[language]['model_path']
    config_path = speaker_models_config[language]['config_path']
    device = speaker_models_config[language]['device']
    # 自动选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    # 获取超参数
    hps = get_hparams_from_file(config_path)
    # 加载模型，如果模型已经存在则复用
    model = model_instances.get(model_path, get_model(model_path=model_path, device=device, hps=hps))
    models_map[language] = temp_model_instances[model_path] = model
    hps_map[language] = hps
    device_map[language] = device


def process_text(ori_text):
    # 替换符号前后的数字并进行处理
    def replace_symbol(match):
        left, symbol, right = match.groups()
        # 判断符号两边是否是数字
        if left.isdigit() and right.isdigit():
            left_num = int(left)
            right_num = int(right)
            # 判断数字大小来决定替换的符号
            if symbol == '-':
                if left_num < right_num:
                    return f"{left}至{right}"  # 当左边数字小于右边数字时，替换为"至"
                else:
                    return f"{left}减{right}"  # 否则，替换为"减"
            elif symbol == '+':
                return f"{left}加{right}"  # + 直接替换为 "加"
            elif symbol == '=':
                return f"{left}等于{right}"  # = 直接替换为 "等于"
        return match.group(0)  # 如果不符合条件，则保持原样

    # 正则表达式匹配整数和小数
    def process_num(match):
        num = match.group()  # 获取匹配的数字
        is_year = '年' in ori_text[match.start():min(len(ori_text), match.end() + 2)]  # 检查后面是否跟有“年”
        if is_year:  # 如果是年份
            pro_text = an2cn(int(num), mode='direct')  # 将年份处理为二零二四的形式
        else:
            parts = num.split('.')
            chinese_parts = [an2cn(int(part), mode='low') for part in parts]  # 列表推导式直接转换
            pro_text = ''.join(['', '点'][len(parts) > 1].join(chinese_parts))
        return pro_text

    clean_text = re.sub(r'[|<>\[\]]', '', ori_text)
    # 使用正则表达式匹配符号前后的数字并进行替换
    symbol_pattern = r'(\d+)([+-=])(\d+)'
    translated_text = re.sub(symbol_pattern, replace_symbol, clean_text)
    # symbol_replacements = {'-': '减', '+': '加', '=': '等于'}
    symbol_replacements = {'+': '加', '=': '等于'}
    # 使用str.translate()和str.maketrans()来创建一个转换表，然后一次性替换所有指定的字符。
    translated_text = translated_text.translate(str.maketrans(symbol_replacements))
    # 使用 re.sub 替换文本中的数字
    converted_text = re.sub(r'\d+(\.\d+)?', process_num, translated_text)
    return converted_text


def gradio_infer(text, speaker, language, rate, sdp, scale, noise, length):
    stream = False
    device_map, models_map, hps_map = get_models(models, speaker, language)
    generator = generate(text, speaker, language, stream, rate, hps_map,
                         models_map, device_map, sdp, scale, noise, length)
    status, audio_concat = list(generator)[0]
    # audio_concat = (audio_concat / np.abs(audio_concat).max() * 32767).astype(np.int16)
    # 归一化到 [-1.0, 1.0] 范围
    audio_data = audio_concat / np.max(np.abs(audio_concat), axis=0)  # 确保最大值为 1.0
    audio_data = audio_data.astype(np.float32)  # 转为 float32 类型
    print(f"生成的音频数据大小: {audio_data.shape}")  # 输出生成音频数据的形状
    final_audio = (rate, audio_concat)
    return status, final_audio


def generate(full_text, speaker, language, stream, sampling_rate, hps_map, models_map, device_map, sdp_ratio,
             noise_scale, noise_scale_w, length_scale, cut_by_sent=False, para_interval=0.5, sent_interval=0.3):
    """文本转语音 API"""
    start = time.process_time()
    if not full_text:
        raise ValueError("The text must be provided and not empty.")
    processed_text = process_text(full_text)
    para_list = split_paragraphs(processed_text)
    para_list = [para for para in para_list if para.strip()]
    # 初始化 all_audios 为一个空的 NumPy 数组
    all_audios = np.array([], dtype=np.float32)
    sent_silence = np.zeros(int(sampling_rate * sent_interval), dtype=np.float32)
    para_silence = np.zeros(int(sampling_rate * (para_interval - sent_interval)), dtype=np.float32)
    for paragraph in para_list:
        if cut_by_sent:
            text_list = split_sentences(paragraph)
            sent_list = [sentence for sentence in text_list if sentence.strip()]  # 去除空字符串
        else:
            sent_list = [paragraph]
        for sentence in sent_list:
            # 简单防止纯符号引发参考音频泄露, sent.isalnum()是否都是字母或数字(即不包含标点符号或空格等)。sent.isalpha()是否都是字母(包括中文、英文等非拉丁字母。)。
            if not any(sent.isalnum() or sent.isalpha() for sent in sentence):
                continue
            audios = go_infer(sentence, speaker, language, sampling_rate, hps_map, models_map,
                              device_map, sdp_ratio, noise_scale, noise_scale_w, length_scale)
            if stream:
                for audio_date in audios:
                    yield audio_date
                yield sent_silence
            else:
                for audio in audios:
                    all_audios = np.concatenate((all_audios, audio))
                all_audios = np.concatenate((all_audios, sent_silence))
        if stream:
            yield para_silence
        else:
            all_audios = np.concatenate((all_audios, para_silence))
    torch_gc()  # 大概会消耗0.2s时间，看情况使用
    end = time.process_time()
    logger.debug(f"infer_time: {end - start}")
    # 如果没有这个判断，会在流式模式下尝试返回一个音频数组，这样会破坏生成器的特性，并可能引发错误。
    if stream:
        # 结束流式传输时，返回一个空数组
        logger.success('done!')
        yield np.array([])  # 返回空数组以通知流结束
    else:
        audio_date = all_audios
        if audio_date.size != 0:
            if np.any(audio_date):  # 检查 audio_concat 是否为空
                status = 'Success'
                logger.success('音频合成完成。')
            # 检查列表中的所有音频数据是否都是静音
            else:
                status = 'Silent'
                logger.warning('所有音频数据均为静音。')
        else:
            # 如果 audio_concat 为空，可以选择返回一个空数组，或者抛出一个自定义异常
            status = 'Failed'
            audio_date = np.zeros(sampling_rate // 2, dtype=np.float32)
            logger.error('音频合成失败。')
        # 生成器不应返回值。即使在非流式模式下，go_infer 的返回值也不会被使用。
        yield status, audio_date


def go_infer(sentence, speaker, language, sampling_rate, hps_map, models_map, device_map, sdp_ratio,
             noise_scale, noise_scale_w, length_scale):
    """区分语言文本转语音 API"""
    with torch.no_grad():
        if language == "mix":
            """多语言，根据语言切分"""
            sentences_list = split_by_language(sentence, target_languages=["zh", "ja", "en"])
            # [('你好你是谁, ', 'zh'), ('hello，how are you，', 'en'), ('我很好', 'zh')]
        else:
            """单语言，直接打包成元组"""
            sentences_list = [(sentence, language)]
        for one_text, one_language in sentences_list:
            logger.info(f'{one_text}, {one_language}, {speaker}')
            hps = hps_map[one_language]
            model = models_map[one_language]
            device = device_map[one_language]
            audio = infer(one_text, speaker, one_language, hps, model, device, sdp_ratio,
                          noise_scale, noise_scale_w, length_scale)
            audio = np.zeros(sampling_rate, dtype=np.float32) if audio.size == 0 else audio

            # frame_size = 1024
            # num_frames = len(audio) // frame_size
            # for i in range(num_frames):
            #     yield audio[i * frame_size: (i + 1) * frame_size]
            # # 如果音频长度不是 frame_size 的整数倍，返回剩余的音频数据
            # remainder = audio[num_frames * frame_size:]
            # if remainder.size > 0:
            #     yield remainder
            # # # 使用 np.array_split 分割音频
            # # for frame in np.array_split(audio, np.ceil(len(audio) / frame_size)):
            # #     yield frame
            yield audio


# #Gradio UI 调试部分
if __name__ == "__main__":
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                # all_speakers = ["jt", "huang", "li"]
                all_languages = ["zh", "ja", "en", "mix", "auto"]
                web_device = config.webui_config.device
                web_config_path = config.webui_config.config_path
                web_hps = get_hparams_from_file(web_config_path)
                models = config.webui_config.models
                web_model_path = config.webui_config.model
                web_model = get_model(model_path=web_model_path, device=web_device, hps=web_hps)
                num_speakers = web_model.n_speakers
                speaker_ids = web_hps.data.spk2id
                # 获取所有的 speaker 名称
                all_speakers = list(speaker_ids.keys())
                web_text = gr.TextArea(label="输入文本内容")
                web_speaker = gr.Dropdown(choices=all_speakers, value=all_speakers[0], label="Speaker")
                web_language = gr.Dropdown(choices=all_languages, value=all_languages[0], label="Language")
                web_sdp_ratio = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.1, label="SDP Ratio")
                web_noise_scale = gr.Slider(minimum=0.1, maximum=2, value=0.6, step=0.1, label="Noise")
                web_noise_scale_w = gr.Slider(minimum=0.1, maximum=2, value=0.9, step=0.1, label="Noise_W")
                web_length_scale = gr.Slider(minimum=0.1, maximum=2, value=1.0, step=0.1, label="Length")
                web_sampling_rate = gr.Slider(minimum=10000, maximum=50000, value=44100, step=100,
                                              label="sampling_rate")
                btn = gr.Button("生成音频！", variant="primary")
                # 创建State组件来存储模型字典和其他数据
                explain_image = gr.Image(label="参数解释信息", show_label=True,
                                         value=os.path.abspath("./temp/img/参数说明.png"))
            with gr.Column():
                text_output = gr.Textbox(label="状态信息")
                audio_output = gr.Audio(type="numpy", label="输出音频")
                with gr.Row():
                    Kami_sato_Ayaka_image = gr.Image(label="神里绫华", show_label=True,
                                                     value=os.path.abspath("./temp/img/神里绫华.png"))
                    Yuyu_image = gr.Image(label="yuyu", show_label=True,
                                          value=os.path.abspath("./temp/img/yuyu.png"))
                    Na_xi_da_image = gr.Image(label="纳西妲", show_label=True,
                                              value=os.path.abspath("./temp/img/纳西妲.png"))
                    Xiao_Gong_image = gr.Image(label="宵宫", show_label=True,
                                               value=os.path.abspath("./temp/img/宵宫.png"))
        btn.click(
            fn=gradio_infer,
            inputs=[web_text, web_speaker, web_language, web_sampling_rate, web_sdp_ratio, web_noise_scale,
                    web_noise_scale_w, web_length_scale],
            outputs=[text_output, audio_output])
    logger.info("推理页面已开启!")
    app.launch(share=config.webui_config.share, server_name='0.0.0.0', server_port=config.webui_config.port)


