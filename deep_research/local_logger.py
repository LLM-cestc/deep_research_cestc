# -*- coding: utf-8 -*-
"""
# 风险模块：内容风险控制

Author: wjianxz
Date: 2025-11-13
"""
import logging
import time
from functools import wraps
import queue

from typing import Optional, Any
logger = logging.getLogger(__name__)  # 自动继承 root logger 的 handlers

# 配置日志

def check_llm_output(
    output: Any,
    *,
    function_name: str = "llm_inference",
    query: Optional[str] = None,
    raise_on_none: bool = False,
) -> bool:
    """
    检测大模型函数返回值是否为 None，并记录异常日志。

    Args:
        output: 大模型函数的返回值。
        function_name: 函数名称，用于日志标识（默认 "llm_inference"）。
        query: 可选的用户查询内容，便于排查问题。
        raise_on_none: 若为 True，当 output 为 None 时抛出 RuntimeError。

    Returns:
        bool: True 表示正常（非 None），False 表示返回值为 None。

    Example:
        result = call_llm("你好吗？")
        if not check_llm_output(result, function_name="call_qwen", query="你好吗？"):
            result = "抱歉，模型暂时无法回答。"
    """
    if output is None:
        msg = f"大模型函数 '{function_name}' 返回值为 None！"
        if query is not None:
            msg += f" 查询内容: {query!r}"
        logger.error(msg)

        if raise_on_none:
            raise RuntimeError(f"LLM function '{function_name}' returned None.")
        return False
    else:
        logger.debug(f"'{function_name}' 返回有效结果 (type: {type(output).__name__})")
        return True

def model_request_error():
    return "请求超时或发生错误，请稍后重试。"


def setup_logger(
    name: str = __name__,
    log_file: str = "app.log",
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    设置日志记录器，支持同时输出到文件和控制台。

    Args:
        name: logger 名称（建议用 __name__）
        log_file: 日志文件名（相对于当前脚本目录）
        level: 日志级别
        console: 是否输出到控制台

    Returns:
        配置好的 logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # 避免重复添加 handler（防止多次调用时日志重复）
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建格式器
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件处理器：日志写入当前目录的 log_file
    log_path = log_file
    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode='w')

    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器（可选）
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def setup_global_logger_root(
    log_file: str = "app.log",
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    全局配置日志系统，影响所有通过 logging.getLogger(...) 获取的 logger。
    
    仅应在主程序入口调用一次。
    """
    # 获取 root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 防止重复添加 handlers（重要！）
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件处理器
    log_path = log_file
    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode='w')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # 返回 root logger（或任意子 logger，它们都会继承配置）
    return root_logger

class Timer:
    def __init__(
        self,
        name="代码块",
        logger: Optional[logging.Logger] = None,
        level: int = logging.INFO,
    ):
        self.name = name
        self.logger = logger
        self.level = level
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start
        # print(f"{self.name} 耗时: {elapsed:.4f} 秒")
        logger.info(f"{self.name} 运行耗时: {elapsed:.4f} 秒\n")
        
        # if self.logger is not None:
        #     self.logger.log(self.level, elapsed)
        # else:
        #     # 回退到 print（兼容旧用法）
        #     logger.info(f"{self.name} 运行耗时: {elapsed:.4f} 秒\n")

class Thinking:
    def __init__(
        self,
        name="代码块",
        result_queue: Optional[queue.Queue] = None,
    ):
        self.name = name
        self.logger = logger
        self.result_queue = result_queue

    def __enter__(self):
        self.result_queue.put(f"{self.name} 运行耗时: {time.perf_counter() - self.start:.4f} 秒\n")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.result_queue.put(f"{self.name} 运行耗时: {time.perf_counter() - self.start:.4f} 秒\n")
        

class RealTimeFileHandler(logging.Handler):
    def __init__(self, filename, mode='a', encoding='utf-8'):
        super().__init__()
        self.filename = filename
        self.mode = mode
        self.encoding = encoding
        self.stream = None
        self._open()

    def _open(self):
        self.stream = open(self.filename, self.mode, encoding=self.encoding, buffering=1)

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            stream.flush()  # 关键：立即刷新
        except Exception:
            self.handleError(record)

    def close(self):
        if self.stream:
            self.stream.close()
        super().close()


def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f"----------->>>>>>>> {func.__name__}  运行时间: {end - start:.6f} 秒\n")
        return result
    return wrapper


# 使用示例
if __name__ == "__main__":
    # logger = setup_logger(__name__, log_file="llm_monitor.log")

    # test()
    pass
