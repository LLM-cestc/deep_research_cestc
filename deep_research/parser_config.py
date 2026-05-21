# -*- coding: utf-8 -*-
"""
# 配置文件加载模块：用于安全读取 YAML 格式的配置文件。

Author: wjianxz
Date: 2025-11-13
"""
import yaml
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


def load_yaml_config(config_path: str) -> Any:
    """
    安全地从 YAML 文件加载配置。

    Args:
        config_path (str): YAML 配置文件路径

    Returns:
        Dict[str, Any]: 配置字典

    Raises:
        FileNotFoundError: 文件不存在
        yaml.YAMLError: YAML 格式错误
        ValueError: 文件路径为空或无效
    """
    if not config_path:
        raise ValueError("配置文件路径不能为空")

    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_file.absolute()}")

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        # safe_load 可能返回 None（空文件），转为空 dict 更安全
        return config if config is not None else {}
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML 格式错误 in {config_file}: {e}")


class ModelConfig(BaseModel):
    """
    模型配置类，用于定义和存储模型的各种参数

    继承自BaseModel，提供数据验证和序列化功能
    """

    name: str = "qwen-model"
    server: str = ""
    temperature: Optional[float] = Field(0.7, ge=0.0, le=1.0)
    max_tokens: int = 4096
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    score: float = Field(0.9, ge=0.0, le=1.0)


class LoggingConfig(BaseModel):
    """
    日志配置类，用于定义日志记录的基本配置参数。
    继承自BaseModel，通常用于数据验证和设置管理。
    """

    level: str = "INFO"
    file: Optional[str] = None


class HopConfig(BaseModel):
    """
    HopConfig类用于配置跳跃搜索的相关参数。
    继承自BaseModel，提供默认配置值。
    """

    max_hops: int = 4
    maxepochs: int = 5
    urlmaxnum: int = 20
    relativescore: float = 0.8
    secondconfscore: float = 0.85
    thirdconfscore: float = 0.85
    upspeed: bool = True
    rewrite: bool = False
    norefturn: bool = False
    turnsession: bool = True
    historynum: int = 5
    session: int = 1
    sessionthreshold: float = 0.6
    turn_on_deepresearch: bool = False
    deepresearch_max_depth: int = 2
    deepresearch_conf_score: float = 0.85


class AppPattern(BaseModel):
    """
    应用程序模式类，用于定义应用程序的运行模式配置。
    继承自BaseModel，表明这是一个基础模型类。
    """

    select_pattern: str = "default"


class TokenizerConfig(BaseModel):
    name: str = "qwen-tokenizer"


class RagConfig(BaseModel):

    TOP_K: int = 6
    score_threshold: float = 0.0
    reference_filenames: list[str] = ["references.jsonl"]
    # BM25 参数：k1 控制词频饱和度，b 控制长度归一化
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    # 稠密检索开关
    use_dense: bool = False
    # 稠密 embedding 模式：true=HTTP API（dense_http_url），false=本地模型（dense_model）
    use_dense_http: bool = False
    dense_http_url: str = ""
    # HTTP 请求体中的 model 名，需与服务端注册名一致
    dense_http_model: str = ""
    # 本地模式：路径或 HuggingFace id（如 BAAI/bge-small-zh-v1.5）
    dense_model: str = "BAAI/bge-small-zh-v1.5"
    dense_top_k: int = 20
    # 本地 SentenceTransformer 使用设备；cpu 可避免和 vLLM 抢显存
    dense_device: str = ""
    # 首次构建全量向量时可用多卡并行；为空则使用 dense_device 单卡/CPU
    dense_encode_devices: list[str] = []
    dense_batch_size: int = 32
    # BM25 候选预过滤：低于最高分 × 此比例的候选被丢弃（0=不过滤）
    bm25_min_score_ratio: float = 0.1
    # 加权融合权重（BM25 归一化分 × bm25_weight + dense 归一化分 × dense_weight）
    bm25_weight: float = 0.3
    dense_weight: float = 0.7
    # 仅出现在一路的文档，总分乘此系数（<1 可压低无关召回）
    single_channel_penalty: float = 0.5


class WebConfig(BaseModel):
    """可信源联网检索配置。"""

    enabled: bool = False
    # provider 兼容字段；联网检索：Bing/Baidu 为 HTML；Google 仅为 SerpAPI（见 serpapi_*）。
    provider: str = "html_scrape"
    # 单选首选引擎（bing | baidu | google）。google 仅 SerpAPI，需配置 serpapi_api_key。
    # 为空字符串时按 engine_order 多引擎降级搜索。
    search_engine: str = ""
    # 多引擎降级顺序，按数组顺序依次尝试，首个非空结果生效（search_engine 为空时生效）
    engine_order: list[str] = ["bing", "google"]
    # 是否合并多引擎结果（默认 False=首个非空即返回）
    search_merge_engines: bool = False
    # 单次搜索请求超时（秒）。未设置则回退 request_timeout。
    search_request_timeout: float | None = None
    # 可选 HTTP/HTTPS 代理：写入 Session（SerpAPI 在国外时可走代理）；Bing/Baidu 在配置了 proxy 时会按需直连。
    search_proxy: str = ""
    # 每次外部请求间随机睡眠区间（秒），防风控
    search_sleep_min: float = 0.2
    search_sleep_max: float = 0.6
    # 同一题内主引擎整轮无结果时的重试次数（含首次）
    search_round_retries: int = 2
    # 重试/切换降级引擎前的冷却（秒）
    search_round_cooldown_s: float = 4.0

    # -------- SerpAPI（Google 唯一检索路径）--------
    serpapi_key_env: str = "SERPAPI_API_KEY"
    serpapi_api_key: str = ""
    serpapi_url: str = "https://serpapi.com/search.json"
    serpapi_request_timeout: float | None = None
    serpapi_retries: int = 1
    trusted_domains: list[str] = [
        "npc.gov.cn",
        "court.gov.cn",
        "spp.gov.cn",
        "pbc.gov.cn",
        "csrc.gov.cn",
        "samr.gov.cn",
    ]
    enforce_trusted_domains: bool = False
    # 联网检索词由 agent 侧 build_retrieval_plan_prompt 一次规划，此处仅限制条数上限。
    max_queries: int = 3
    max_search_results: int = 10
    max_fetch_pages: int = 5
    max_evidence_items: int = 5
    max_chunks_per_page: int = 2
    chunk_size: int = 800
    chunk_overlap: int = 120
    request_timeout: float = 10.0
    title_score_threshold: float = 1.0
    min_content_length: int = 300
    # title_then_full=标题筛页后整页正文送模型（推荐）；chunk=按块打分；auto/full_page 同 title_then_full
    evidence_mode: str = "title_then_full"
    full_page_max_chars: int = 16000
    max_full_page_items: int = 3
    full_page_min_score: float = 0.35
    # 抓取后由 extractinfo 模型筛选重要原文、去无关并控篇幅
    page_summary_enabled: bool = False
    page_summary_max_input_chars: int = 16000
    page_summary_max_output_chars: int = 4500
    page_summary_parallel: int = 3
    page_summary_timeout: int = 120


class AppConfig(BaseModel):
    """
    应用程序配置类，继承自BaseModel，用于存储和管理应用程序的各种配置参数。
    每个配置项都是特定的配置类实例，用于控制应用程序的不同方面。
    """

    session: ModelConfig = ModelConfig(
        name="qwen-model",
        server="",
        temperature=0.7,
        max_tokens=4096,
        top_p=0.95,
        repetition_penalty=1.05,
        score=0.9,
    )
    tokenizer: TokenizerConfig = TokenizerConfig(name="qwen-model")
    safemodel: ModelConfig = ModelConfig(
        name="qwen-model",
        server="",
        temperature=0.7,
        max_tokens=4096,
        top_p=0.95,
        repetition_penalty=1.05,
        score=0.9,
    )
    deepresearch: ModelConfig = ModelConfig(
        name="qwen-model",
        server="",
        temperature=0.7,
        max_tokens=4096,
        top_p=0.95,
        repetition_penalty=1.05,
        score=0.9,
    )
    speeddeepresearch: ModelConfig = ModelConfig(
        name="qwen-model",
        server="",
        temperature=0.7,
        max_tokens=4096,
        top_p=0.95,
        repetition_penalty=1.05,
        score=0.9,
    )
    confidence: ModelConfig = ModelConfig(
        name="qwen-model",
        server="",
        temperature=0.7,
        max_tokens=4096,
        top_p=0.95,
        repetition_penalty=1.05,
        score=0.9,
    )
    subquestion: ModelConfig = ModelConfig(
        name="qwen-model",
        server="",
        temperature=0.7,
        max_tokens=4096,
        top_p=0.95,
        repetition_penalty=1.05,
        score=0.9,
    )
    rewrite: ModelConfig = ModelConfig(
        name="qwen-model",
        server="",
        temperature=0.7,
        max_tokens=4096,
        top_p=0.95,
        repetition_penalty=1.05,
        score=0.9,
    )
    beautifulformat: ModelConfig = ModelConfig(
        name="qwen-model",
        server="",
        temperature=0.7,
        max_tokens=4096,
        top_p=0.95,
        repetition_penalty=1.05,
        score=0.9,
    )
    maxhop: HopConfig = HopConfig()
    logging: LoggingConfig = LoggingConfig(level="INFO", file=None)
    pattern: AppPattern = AppPattern()
    rag: RagConfig = RagConfig()
    web: WebConfig = WebConfig()

    # 
    formatOutput: bool = False

def load_validated_config(config_path: str) -> AppConfig:
    """
    加载并验证YAML配置文件

    Args:
        config_path (str): 配置文件路径；相对路径以 deep_research_dev 项目根为基准

    Returns:
        AppConfig: 经过验证的配置对象，包含所有必要的配置信息
    """
    from deep_research.paths import resolve_project_path

    resolved = str(resolve_project_path(config_path))
    raw_config = load_yaml_config(resolved)
    return AppConfig(**raw_config)


# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # 使用
    from deep_research.paths import default_config_path

    config = load_validated_config(default_config_path())
    print("=========", config.deepresearch.name)  # 类型安全，带默认值和校验
    # from functools import partial
    # deep_search_rag_bound_fn = partial(deep_search_rag, config=config)
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.name, trust_remote_code=True)
