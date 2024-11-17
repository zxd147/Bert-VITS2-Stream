#!/bin/bash

bash stop_service.sh

# 获取当前时间戳
timestamp=$(date +"%Y%m%d_%H%M%S")
PORT=8031
lsof -i :$PORT | grep LISTEN | awk '{print $2}' | xargs kill -9 >/dev/null 2>&1
# 日志文件名
log_dir="logs"
output_log="output.log"
# 输出文件名
output_txt="output.txt"

# 如果日志文件存在，则重命名为带有时间戳的新文件名
if [ -f "${log_dir}/${output_log}" ]; then
    mv "${log_dir}/${output_log}" "${log_dir}/${timestamp}_${output_log}"
    echo "Existing log file renamed to ${timestamp}_${output_log}"
fi


# 启动服务并将输出重定向到 output.log
cd /home/zxd/code/TTS/Bert-VITS2
nohup /home/zxd/.conda/envs/bv2/bin/python vits2_api.py > "${log_dir}/${output_log}" 2>&1 &

# 获取启动的进程 ID 并输出启动信息
PID=$!
echo "================[${timestamp}]===============" > ${output_txt}
echo "Service started successfully with PID ${PID}." | tee -a ${output_txt}

## 保持前台运行
#while kill -0 ${PID} 2>/dev/null; do
#    sleep 10
#done


