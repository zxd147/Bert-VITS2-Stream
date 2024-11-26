import argparse
import json
import os
import re

import torch
import whisper
import yaml

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一层目录
parent_dir = os.path.dirname(current_dir)
# 切换到上一层目录
os.chdir(parent_dir)
from config import Config

yml_path = 'config.yml'
config = Config(yml_path)
with open(yml_path, mode="r", encoding="utf-8") as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)
speaker_path = config_yaml["dataset_path"]
speaker = os.path.basename(speaker_path)
transcription_path = config.preprocess_text_config.transcription_path
transcription_dir = os.path.dirname(transcription_path)
os.makedirs(transcription_dir, exist_ok=True)


lang2token = {
    'zh': "ZH",
    'ja': "JP",
    "en": "EN",
}


def transcribe_one(audio_path):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    language = max(probs, key=probs.get)
    # decode the audio
    prompt = ""
    options = whisper.DecodingOptions(beam_size=5, prompt=prompt)
    result = whisper.decode(model, mel, options)
    asr_text = result.text
    # print the recognized text
    print(asr_text)
    return language, asr_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="CJE")
    parser.add_argument("--whisper_size", default="large-v2")
    args = parser.parse_args()
    if args.languages == "CJE":
        lang2token = {
            'zh': "ZH",
            'ja': "JP",
            "en": "EN",
        }
    elif args.languages == "CJ":
        lang2token = {
            'zh': "ZH",
            'ja': "JP",
        }
    elif args.languages == "C":
        lang2token = {
            'zh': "ZH",
        }
    assert (torch.cuda.is_available()), "Please enable GPU in order to run Whisper!"
    model_path = f"/home/zxd/code/ASR/whisper/all_models/models_pt/medium.pt"
    print("=====加载模型...=====")
    model = whisper.load_model(model_path)
    print("=====模型加载成功=====")
    parent_dir = config.resample_config.out_path
    print(f"输入目录：{parent_dir}")
    speaker_annotations = []
    # os.walk() 返回一个生成器，每次迭代生成三个值：目录路径，子目录列表，子文件列表
    # total_files = sum([len(files) for r, d, files in os.walk(parent_dir)])
    # resample audios
    with open(config.train_ms_config.config_path, 'r', encoding='utf-8') as f:
        hps = json.load(f)
    target_sr = hps['data']['sampling_rate']
    processed_files = 0

    # 获取所有音频文件
    file_list = list(os.walk(parent_dir))[0][2]
    total_files = len(file_list)  # 统计总文件数
    # 使用 sorted 进行自然排序，只考虑文件名中的数字部分
    file_list = sorted(file_list, key=lambda file: int(re.search(r'(\d+)', file).group()))
    print("=====开始转录=====\n")
    for i, wavfile in enumerate(file_list):
        # try to load file as audio
        try:
            # transcribe text
            in_wav_path = f"{parent_dir}/{wavfile}"
            lang, text = transcribe_one(in_wav_path)
            if lang not in list(lang2token.keys()):
                print(f"{lang} not supported, ignoring\n")
                continue
            # text = "ZH|" + text + "\n"
            # text = f"{in_wav_path}|{speaker}|{lang2token[lang]}|{text}\n"
            text = f"{in_wav_path}|{speaker}|{lang}|{text}\n"
            speaker_annotations.append(text)

            processed_files += 1
            print(f"Processed: {processed_files}/{total_files}\n")
        except Exception as e:
            print(e)
            continue

    # write into annotation
    if len(speaker_annotations) == 0:
        print(
            "Warning: no short audios found, "
            "this is expected if you have only uploaded long audios, videos or video links."
            "This is not expected if you have uploaded a zip file of short audios. "
            "Please check your file structure or make sure your audio language is supported.")
    with open(transcription_path, 'w', encoding='utf-8') as f:
        for line in speaker_annotations:
            f.write(line)
print(f"转录完成，标注文件位于：{transcription_path}")
