# 使用现有的基础镜像
FROM init/python:ubuntu22.04-cuda11.8-python3.10

# 设置代理 (确保网络环境)
# ENV all_proxy=http://192.168.0.64:7890

# 更新并安装必要的依赖
RUN apt-get update \
    # 在这里安装你需要的依赖，比如 git、python 等
    && apt-get install git -y \
    && apt-get install ffmpeg -y \
    && apt-get install mecab libmecab2 libmecab-dev libbz2-dev -y \
    && apt-get clean

# 设置工作目录
WORKDIR /app/Bert-VITS2

# 设置 GPU 使用
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# 映射端口
EXPOSE 8031

# 克隆并直接重命名为 Bert-VITS2
RUN cd /app && git clone https://github.com/zxd147/Bert-VITS2-Stream.git Bert-VITS2

# 复制本地文件到容器内
COPY ./nltk_data/ /usr/local/lib/nltk_data/
COPY ./config.yml ./config.yml
COPY ./Data/ ./Data/
COPY ./bert/ ./bert/

# 安装依赖
RUN pip install -r ./requirements.txt

# 容器启动时默认执行的命令
CMD ["python", "vits2_api.py"]
