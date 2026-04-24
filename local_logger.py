# -*- coding: utf-8 -*-
"""
日志与计时工具模块

Author: wjianxz
Date: 2025-11-13
"""
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)


def model_request_error():
    return "请求超时或发生错误，请稍后重试。"


def setup_global_logger_root(
    log_file: str = "app.log",
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    全局配置日志系统，影响所有通过 logging.getLogger(...) 获取的 logger。
    仅应在主程序入口调用一次。
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8", mode='w')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    return root_logger


class Timer:
    def __init__(
        self,
        name="代码块",
        logger=None,
        level: int = logging.INFO,
    ):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start
        logger.info(f"{self.name} 运行耗时: {elapsed:.4f} 秒\n")


def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f"----------->>>>>>>> {func.__name__}  运行时间: {end - start:.6f} 秒\n")
        return result
    return wrapper
