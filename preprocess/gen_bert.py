import argparse
import os
from multiprocessing import Pool

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

import commons
from config import config
from text import check_bert_models, convert_to_ids, get_bert
from utils.model_utils import get_hparams_from_file

preprocess_text_config = config.preprocess_text_config


# 处理每一行文本数据，并生成相应的 BERT 特征
def process_line(x):
    line, add_blank = x
    device = config.bert_gen_config.device
    if config.bert_gen_config.use_multi_device:
        # 表示当前进程标识符
        rank = mp.current_process()._identity
        rank = rank[0] if len(rank) > 0 else 0
        if torch.cuda.is_available():
            gpu_id = rank % torch.cuda.device_count()
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")
    wav_path, _, language_str, text, phones, tone, word2ph = line.strip().split("|")
    phone = phones.split(" ")
    tone = [int(i) for i in tone.split(" ")]
    word2ph = [int(i) for i in word2ph.split(" ")]
    word2ph = [i for i in word2ph]
    phone, tone, language = convert_to_ids(phone, tone, language_str)

    if add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    bert_path = wav_path.replace(".WAV", ".wav").replace(".wav", ".bert.pt")

    try:
        bert = torch.load(bert_path)
        assert bert.shape[0] == 2048
    except Exception:
        print("load bert file fail, go to get bert...")
        bert = get_bert(text, word2ph, language_str, device)
        assert bert.shape[-1] == len(phone)
        torch.save(bert, bert_path)
        torch.cuda.empty_cache()
        print(f"bert file is saved to {bert_path}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default=config.bert_gen_config.config_path
    )
    parser.add_argument(
        "--num_processes", type=int, default=config.bert_gen_config.num_processes
    )
    args, _ = parser.parse_known_args()
    config_path = args.config
    # hps 是“hyperparameters”的缩写，意为“超参数”
    hps = get_hparams_from_file(config_path)
    check_bert_models()
    lines = []
    train_list = preprocess_text_config.train_path
    with open(train_list, encoding="utf-8") as f:
        lines.extend(f.readlines())

    val_list = preprocess_text_config.val_path
    with open(val_list, encoding="utf-8") as f:
        lines.extend(f.readlines())
    # 创建一个与 lines 列表长度相同的布尔值列表
    add_blank = [hps.data.add_blank] * len(lines)

    if len(lines) != 0:
        print(f"开始生成bert, 共有{len(lines)}行数据待生成bert文件!")
        num_processes = args.num_processes
        with Pool(processes=num_processes) as pool:
            for _ in tqdm(
                    pool.imap_unordered(process_line, zip(lines, add_blank)),
                    total=len(lines),
            ):
                # 这里是缩进的代码块，表示循环体
                pass  # 使用pass语句作为占位符
    bert_path = config.resample_config.out_path
    bert_files = [file for file in os.listdir(bert_path) if file.endswith(".bert.pt")]
    print(f"BERT 生成完毕!, 共有 {len(bert_files)} 个 .bert.pt 文件生成!")
