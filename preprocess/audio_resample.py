import argparse
import os
from multiprocessing import Pool, cpu_count

import librosa
import soundfile
from tqdm import tqdm

os.chdir('../')
from config import config


def process(item):
    speaker_dir, wav_name, task_args = item
    wav_path = os.path.join(args.in_path, speaker_dir, wav_name)
    if os.path.exists(wav_path) and wav_path.lower().endswith(".wav"):
        wav, sr = librosa.load(wav_path, sr=args.sr)
        soundfile.write(os.path.join(args.out_path, speaker_dir, wav_name), wav, sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sr",
        type=int,
        default=config.resample_config.sampling_rate,
        help="sampling rate",
    )
    parser.add_argument(
        "--in_path",
        type=str,
        default=config.resample_config.in_path,
        help="path to source dir",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=config.resample_config.out_path,
        help="path to target dir",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=0,
        help="cpu_processes",
    )
    args, _ = parser.parse_known_args()
    # autodl 无卡模式会识别出46个cpu
    if args.processes == 0:
        processes = cpu_count() - 2 if cpu_count() > 4 else 1
    else:
        processes = args.processes
    pool = Pool(processes=processes)

    tasks = []
    in_path: str = args.in_path
    out_path: str = args.out_path
    print("===开始音频重采样===")
    for spk_in_path, _, filenames in os.walk(in_path):
        # 子级目录
        spk_dir = os.path.relpath(spk_in_path, in_path)
        spk_out_path = os.path.join(out_path, spk_dir)
        os.makedirs(spk_out_path, exist_ok=True)
        for filename in filenames:
            if filename.lower().endswith(".wav"):  # .
                task_tuple = (spk_dir, filename, args)
                tasks.append(task_tuple)

    # 并行处理任务:
    for _ in tqdm(pool.imap_unordered(process, tasks)):
        pass

    pool.close()
    pool.join()

    print(f"音频重采样完毕!, 并移动至{out_path}")
