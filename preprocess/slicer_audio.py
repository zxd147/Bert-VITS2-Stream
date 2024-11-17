import os
import re

import librosa  # Optional. Use any library you like to read audio files.
import soundfile  # Optional. Use any library you like to write audio files.
import yaml

from config import Config
from tools.slicer import Slicer

os.chdir('../')
yml_path = 'config.yml'
config = Config(yml_path)
with open(yml_path, mode="r", encoding="utf-8") as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)

k = 0
speaker_path = config_yaml["dataset_path"]
speaker = os.path.basename(speaker_path)
audio_dir = config.resample_config.in_path
audio_files = [file for file in os.listdir(audio_dir) if file.endswith('.wav')]
audio_files = sorted(audio_files, key=lambda file: int(re.search(r'(\d+)', file).group()))
audio_paths = [os.path.join(audio_dir, file) for file in audio_files]

print(f"输入目录：{audio_dir}, 音频数量：{len(audio_paths)}")
for i, audio_path in enumerate(audio_paths):
    audio, sr = librosa.load(audio_path, sr=None,
                             mono=False)  # Load an audio file with librosa.
    slicer = Slicer(
        sr=sr,
        threshold=-40,
        min_length=2000,
        min_interval=400,
        hop_size=10,
        max_sil_kept=500
    )
    chunks = slicer.slice(audio)
    file_name = os.path.basename(audio_path).split('.')[0]
    for chunk in chunks:
        if len(chunk.shape) > 1:
            chunk = chunk.T  # Swap axes if the audio is stereo.
        # out_path = f'{audio_dir}/{speaker}_{k}.wav'
        out_path = f'{audio_dir}/{file_name}_{k}.wav'
        soundfile.write(out_path, chunk, sr)  # Save sliced audio files with soundfile.
        k += 1

    if os.path.exists(audio_path):  # 如果文件存在
        os.remove(audio_path)
all_files = os.listdir(audio_dir)
print(f"切分完成，切分后的音频数量为：{len(all_files)}")
