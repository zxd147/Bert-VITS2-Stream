import re
import os

import click

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一层目录
parent_dir = os.path.dirname(current_dir)
# 切换到上一层目录
os.chdir(parent_dir)
from config import Config

# 配置文件路径
yml_path = 'config.yml'
config = Config(yml_path)
preprocess_text_config = config.preprocess_text_config


@click.command()
@click.option("--file", default='train', type=str, help="选择排序的文件: 'train' 或 'val'")
# @click.option("--file", default='val', type=str, help="选择排序的文件: 'train' 或 'val'")
def rerank(file):
    # 根据传入参数选择排序的文件
    file_list = preprocess_text_config.train_path if file == 'train' else preprocess_text_config.val_path
    # file_list = file_list.replace('train', 'result').replace('val', 'result')

    # 读取文件内容
    with open(file_list, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # 打印读取到的文件行数
    print(f"正在重新排序文件：{file_list}，总共{len(lines)}行")

    # 按行中的数字进行排序
    sorted_lines = sorted(lines, key=lambda line: extract_numbers(line))
    # 将排序后的内容重新写回文件
    with open(file_list, "w", encoding="utf-8") as f:
        f.writelines(sorted_lines)

    # 打印处理完成信息
    print(f"重新排序完成，总共{len(sorted_lines)}行")


# 提取第一个数字和下划线 _ 后的第二个数字
def extract_numbers(line):
    # 按字母顺序原样返回行作为第一个排序关键字
    alphabetical_part = line
    # 提取第一个数字
    first_number = int(re.search(r'(\d+)', line).group())
    # 提取 _ 后的第二个数字
    second_number_match = re.search(r'_(\d+)_(\d+)', line)
    # second_number_match = re.search(r'_(\d+)', line)
    if second_number_match:
        second_number = int(second_number_match.group(2))
        # second_number = int(second_number_match.group(1))
    else:
        second_number = 0  # 如果没有匹配到第二个数字，则默认设为 0
    # return alphabetical_part, first_number, second_number
    return first_number, second_number


if __name__ == "__main__":
    rerank()
