"""
logger封装
"""

from loguru import logger
import sys


# 移除所有默认的处理器
logger.remove()

# 自定义格式并添加到标准输出
# log_format = "<g>{time:MM-DD HH:mm:ss}</g> <lvl>{level:<9}</lvl>| {file}:{line} | {message}"
log_format = "<g>{time:MM-DD HH:mm:ss}</g> <lvl>{level:<9}</lvl>| {message}"

logger.add(sys.stdout, level="INFO", format=log_format, backtrace=True, diagnose=True)
