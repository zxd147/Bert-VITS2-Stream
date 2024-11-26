import os
import random
import re

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一层目录
parent_dir = os.path.dirname(current_dir)
# 切换到上一层目录
os.chdir(parent_dir)
from config import Config

yml_path = 'config.yml'
config = Config(yml_path)

ori_speaker = 'all'
speaker = 'jt'
k = random.randint(10 ** 3, 10 ** 5 - 1)
print(f"从{k}开始重命名")
audio_dir = config.resample_config.in_path
# audio_dir = config.resample_config.out_path
# audio_files = [file for file in os.listdir(audio_dir) if file.endswith('.wav')]
audio_files = [file for file in os.listdir(audio_dir)]
audio_files = sorted(audio_files, key=lambda file: int(re.search(r'(\d+)', file).group()))
audio_paths = [os.path.join(audio_dir, file) for file in audio_files]
print(f"输入目录：{audio_dir}，文件总数：{len(audio_paths)}")

for audio_path in audio_paths:
    # new_audio_path = f'{audio_dir}/{speaker}_{k}.wav'
    new_audio_path = audio_path.replace(ori_speaker, speaker)
    os.rename(audio_path, new_audio_path)
    k += 1

print(f"重命名完成，文件总数：{len(os.listdir(audio_dir))}")
