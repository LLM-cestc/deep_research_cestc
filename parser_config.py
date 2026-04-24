# -*- coding: utf-8 -*-
"""
配置文件加载模块（精简版：仅保留写作 Deep Research 所需配置）

Author: wjianxz
Date: 2025-11-13
"""
import yaml
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


def load_yaml_config(config_path: str) -> Any:
    if not config_path:
        raise ValueError("配置文件路径不能为空")
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_file.absolute()}")
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config if config is not None else {}
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML 格式错误 in {config_file}: {e}")


class ModelConfig(BaseModel):
    name: str = "qwen-model"
    server: str = ""
    temperature: Optional[float] = Field(0.7, ge=0.0, le=1.0)
    max_tokens: int = 8192
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    score: float = Field(0.9, ge=0.0, le=1.0)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: Optional[str] = None


class HopConfig(BaseModel):
    max_hops: int = 4
    maxepochs: int = 2
    urlmaxnum: int = 0
    relativescore: float = 0.8
    secondconfscore: float = 0.85
    thirdconfscore: float = 0.85
    upspeed: bool = False
    rewrite: bool = False
    norefturn: bool = True
    turnsession: bool = False
    historynum: int = 5
    session: int = 1
    sessionthreshold: float = 0.6
    turn_on_deepresearch: bool = True
    deepresearch_max_depth: int = 1
    deepresearch_conf_score: float = 0.85
    # 用户可选：总章节数（None 表示使用默认/模型决定）
    target_total_sections: Optional[int] = None
    # 用户可选：总字数目标（None 表示使用默认/模型决定）
    target_total_words: Optional[int] = None
    # 用户可选：递归层数（None 表示使用 deepresearch_max_depth）
    target_recursion_depth: Optional[int] = None
    deepresearch_root_branching: int = 6
    deepresearch_child_branching: int = 3
    deepresearch_expand_min_chars: int = 1800
    deepresearch_max_nodes: int = 20
    deepresearch_force_expand_root: bool = True
    # 性能：全局请求最小间隔（秒）。并行章节时建议 0~0.5；网关易 503 时可调到 1~2
    min_request_interval_seconds: float = 0.35
    # 同层章节并行请求数（1=串行）。受服务端并发与限流影响，过大可能 503
    chapter_parallel_workers: int = 4
    # 章节局部质检：最小正文字符数
    chapter_min_chars: int = 1000
    # 章节局部质检失败后的最大重写次数
    chapter_rewrite_max_retries: int = 1
    # 是否启用章节局部质检与局部重写
    enable_local_chapter_check: bool = True
    # 是否启用“逐段一致性校验”
    enable_paragraph_consistency_check: bool = True
    # 参与一致性校验的最小段落长度
    paragraph_min_chars: int = 120
    # 段落一致性通过比例阈值（通过段落数 / 有效段落数）
    paragraph_min_pass_ratio: float = 0.75
    # 不一致时是否允许回退到“重生大纲”
    enable_outline_backtrack: bool = True
    # 不一致时是否允许回退到“重生主题+大纲”
    enable_topic_backtrack: bool = True
    # 快速路径：跳过「整合成一篇」的 LLM，直接拼接各章
    skip_merge_llm: bool = False
    # 快速路径：跳过置信度评分与重试
    skip_confidence_check: bool = False
    # 快速路径：跳过文末润色（agent_deep_search 中）
    skip_final_polish_llm: bool = False


class AppPattern(BaseModel):
    select_pattern: str = "deep_research"


class AppConfig(BaseModel):
    session: ModelConfig = ModelConfig()
    deepresearch: ModelConfig = ModelConfig()
    speeddeepresearch: ModelConfig = ModelConfig()
    confidence: ModelConfig = ModelConfig()
    maxhop: HopConfig = HopConfig()
    logging: LoggingConfig = LoggingConfig()
    pattern: AppPattern = AppPattern()
    formatOutput: bool = False


def load_validated_config(config_path: str) -> AppConfig:
    raw_config = load_yaml_config(config_path)
    return AppConfig(**raw_config)
