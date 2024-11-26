import os
import json
from collections import defaultdict
from random import shuffle
from typing import Optional

import click
from tqdm import tqdm

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一层目录
parent_dir = os.path.dirname(current_dir)
# 切换到上一层目录
os.chdir(parent_dir)
from config import config
from text.cleaner import text_to_phonemes

preprocess_text_config = config.preprocess_text_config


@click.command()
@click.option(
    "--transcription-path",
    default=preprocess_text_config.transcription_path,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--cleaned-path", default=preprocess_text_config.cleaned_path)
@click.option("--train-path", default=preprocess_text_config.train_path)
@click.option("--val-path", default=preprocess_text_config.val_path)
@click.option(
    "--config-path",
    default=preprocess_text_config.config_path,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--val-per-lang", default=preprocess_text_config.val_per_lang)
@click.option("--max-val-total", default=preprocess_text_config.max_val_total)
@click.option("--clean/--no-clean", default=preprocess_text_config.clean)
@click.option("-y", "--yml_config")
def preprocess(
        transcription_path: str,
        cleaned_path: Optional[str],
        train_path: str,
        val_path: str,
        config_path: str,
        val_per_lang: int,
        max_val_total: int,
        clean: bool,
        yml_config: str,  # 这个不要删
):
    print("开始预处理...")
    if cleaned_path == "" or cleaned_path is None:
        cleaned_path = transcription_path + ".cleaned"

    if clean:
        print("数据清洗...")
        with open(cleaned_path, "w", encoding="utf-8") as out_file:
            with open(transcription_path, "r", encoding="utf-8") as trans_file:
                lines = trans_file.readlines()
                # print(lines, ' ', len(lines))
                if len(lines) != 0:
                    for line in tqdm(lines):
                        try:
                            utterance, spk, language, text = line.strip().split("|")
                            norm_text, phones, tones, word2ph = text_to_phonemes(
                                text, language
                            )
                            out_file.write(
                                "{}|{}|{}|{}|{}|{}|{}\n".format(
                                    utterance,
                                    spk,
                                    language,
                                    norm_text,
                                    " ".join(phones),
                                    " ".join([str(i) for i in tones]),
                                    " ".join([str(i) for i in word2ph]),
                                )
                            )
                        except Exception as e:
                            print(line)
                            print(f"生成训练集和验证集时发生错误！, 详细信息:\n{e}")

    print("开始处理文本数据...")
    transcription_path = cleaned_path
    spk_utt_map = defaultdict(list)
    spk_id_map = {}
    current_sid = 0

    with open(transcription_path, "r", encoding="utf-8") as f:
        audioPaths = set()
        countSame = 0
        countNotFound = 0
        for line in f.readlines():
            utterance, spk, language, text, phones, tones, word2ph = line.strip().split("|")
            if utterance in audioPaths:
                # 过滤数据集错误：相同的音频匹配多个文本，导致后续bert出问题
                print(f"重复音频文本：{line}")
                countSame += 1
                continue
            if not os.path.isfile(utterance):
                # 过滤数据集错误：不存在对应音频
                print(f"没有找到对应的音频：{utterance}")
                countNotFound += 1
                continue
            audioPaths.add(utterance)
            spk_utt_map[language].append(line)
            if spk not in spk_id_map.keys():
                spk_id_map[spk] = current_sid
                current_sid += 1
        print(f"总重复音频数：{countSame}，总未找到的音频数:{countNotFound}")

    print("生成训练集和验证集...")
    train_list = []
    val_list = []
    for spk, utterances in spk_utt_map.items():
        shuffle(utterances)
        val_list += utterances[:val_per_lang]
        train_list += utterances[val_per_lang:]

    shuffle(val_list)
    if len(val_list) > max_val_total:
        train_list += val_list[max_val_total:]
        val_list = val_list[:max_val_total]

    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)
    print(f"训练集保存到：{train_path}")

    with open(val_path, "w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)
    print(f"验证集保存到：{val_path}")

    # print("更新配置文件...")
    # json_config = json.load(open(config_path, encoding="utf-8"))
    # json_config["data"]["spk2id"] = spk_id_map
    # json_config["data"]["n_speakers"] = len(spk_id_map)
    # # 新增写入：写入训练版本、数据集路径
    # json_config["data"]["training_files"] = os.path.normpath(train_path).replace(
    #     "\\", "/"
    # )
    # json_config["data"]["validation_files"] = os.path.normpath(val_path).replace(
    #     "\\", "/"
    # )
    # with open(config_path, "w", encoding="utf-8") as f:
    #     json.dump(json_config, f, indent=2, ensure_ascii=False)
    print("训练集和验证集生成完成！")


if __name__ == "__main__":
    preprocess()
