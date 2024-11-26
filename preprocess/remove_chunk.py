import os

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一层目录
parent_dir = os.path.dirname(current_dir)
# 切换到上一层目录
os.chdir(parent_dir)
from pydub import AudioSegment


def remove_first_100ms(file_path):
    # 检查文件是否存在
    if not os.path.isfile(file_path):
        print(f"文件 {file_path} 不存在")
        return
    # 加载音频文件
    audio = AudioSegment.from_file(file_path)
    # 获取音频的时长，去除最后100毫秒
    # 去除前 100 毫秒
    duration_ms = len(audio)  # 音频时长（毫秒）
    trimmed_audio = audio[100:]
    # trimmed_audio = audio[:duration_ms - 100]  # 去掉最后100ms
    # 保存裁剪后的音频
    trimmed_audio.export(file_path, format=os.path.splitext(file_path)[1][1:])
    print(f"音频处理完毕，已保存至 {file_path}")


if __name__ == "__main__":
    audio_file_path = 'Data/li/audios/wavs/li_03_53.wav'
    remove_first_100ms(audio_file_path)
