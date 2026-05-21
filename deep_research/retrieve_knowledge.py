# -*- coding: utf-8 -*-
"""
内部知识库检索（RAG 核心）

检索流程：
  1. BM25 关键词召回（必选）
  2. 稠密语义检索（可选，通过 config.rag.use_dense 开关）
     - use_dense_http=true  → HTTP API（OpenAI 兼容 /v1/embeddings）
     - use_dense_http=false → 本地 SentenceTransformer（如 bge-base-zh-v1.5）
  3. 加权融合 + 单通道惩罚 + 分数阈值过滤

Author: wjianxz
Date: 2025-11-13
"""
from typing import List, Dict, Any
import re
import json
import threading
from pathlib import Path
import os
import jieba  # type: ignore
from rank_bm25 import BM25Okapi  # type: ignore
try:
    from deep_research.parser_config import AppConfig
except ImportError:
    from deep_research.parser_config import AppConfig
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 中文分词 & BM25 索引文本构建
# ---------------------------------------------------------------------------

def chinese_tokenizer(text: str):
    return [word for word in jieba.cut(text) if word.strip() and len(word) > 1]


def _short_law_name(law_name: str) -> str:
    """法名简写，便于「刑法」「专利法」等短称召回。"""
    if not law_name or not isinstance(law_name, str):
        return ""
    s = law_name.strip()
    for prefix in ("中华人民共和国", "中华人民共和国 "):
        if s.startswith(prefix):
            return s[len(prefix):].strip() or s
    return s


def _article_to_digit_suffix(article: str) -> str:
    """从条款字符串中提取阿拉伯数字形式，便于「264条」类查询召回。"""
    if not article or not isinstance(article, str):
        return ""
    s = article.strip()
    digits = re.findall(r"\d+", s)
    if digits:
        return " ".join(digits)
    cn_d = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    cn_unit = {"十": 10, "百": 100, "千": 1000}
    num = 0
    cur = 0
    for c in s:
        if c in cn_d:
            cur = cur * 10 + cn_d[c]
        elif c in cn_unit:
            unit = cn_unit[c]
            if cur == 0 and c == "十":
                cur = 1
            num += cur * unit
            cur = 0
        else:
            if cur:
                num += cur
                cur = 0
    num += cur
    return str(num) if num else ""


def _build_index_text(data: Dict[str, Any]) -> str:
    """构建用于 BM25 的索引文本，加强法名与条款的权重与变体以提升召回。"""
    parts: list[str] = []
    law_name = ""
    for key in ("law_name", "category", "title"):
        value = str(data.get(key, "")).strip()
        if value:
            law_name = value
            break
    if law_name:
        parts.extend([law_name, law_name, law_name])
        short = _short_law_name(law_name)
        if short and short != law_name:
            parts.extend([short, short])
    for key in ("article", "number"):
        value = str(data.get(key, "")).strip()
        if value:
            parts.append(value)
            digit_suffix = _article_to_digit_suffix(value)
            if digit_suffix:
                parts.append(digit_suffix)
            break
    text = str(data.get("text", "")).strip()
    if text:
        parts.append(text)
    return " ".join(parts).strip()


def _build_display_text(data: Dict[str, Any]) -> str:
    law_name = str(data.get("law_name", "")).strip()
    article = str(data.get("article", "")).strip()
    text = str(data.get("text", "")).strip()
    prefix_parts: list[str] = []
    if law_name:
        prefix_parts.append(law_name)
    if article:
        prefix_parts.append(article)
    prefix = " ".join(prefix_parts).strip()
    if prefix and text:
        return f"{prefix}：{text}"
    return text or prefix


# ---------------------------------------------------------------------------
# 文件加载
# ---------------------------------------------------------------------------

DEFAULT_REFERENCE_FILENAME = "references.jsonl"


def _rag_base_dir() -> Path:
    """所有相对路径的基准：本文件所在目录（deep_research），与运行时的 cwd 无关。"""
    return Path(__file__).resolve().parent


def _resolve_local_reference_path(
    reference_filenames: list[str] | None = None,
) -> str:
    """config 中的 reference_filenames 相对 deep_research 目录解析。"""
    filenames = [
        name.strip()
        for name in (reference_filenames or [])
        if isinstance(name, str) and name.strip()
    ]
    if not filenames:
        filenames = [DEFAULT_REFERENCE_FILENAME]
    base_dir = _rag_base_dir()
    for name in [*filenames, "references.jsonl"]:
        candidate = base_dir / name
        if candidate.is_file():
            return str(candidate)
    return str(base_dir / filenames[0])


def load_texts_from_jsonl(file_path: str) -> tuple[list[str], list[Dict[str, Any]]]:
    """从 JSONL 文件中加载文本数据，同时构建索引文本和展示文本。"""
    documents: list[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                text = str(data.get("text", "")).strip()
                if text:
                    data["_index_text"] = _build_index_text(data)
                    data["_display_text"] = _build_display_text(data)
                    documents.append(data)
    texts = [
        str(doc.get("_index_text") or doc.get("text", "")).strip()
        for doc in documents
    ]
    return texts, documents


def query_chinese_tokenizer(texts: list[str]):
    """使用中文分词器对输入文本列表进行分词处理。"""
    return [chinese_tokenizer(text) for text in texts]


# ---------------------------------------------------------------------------
# BM25 索引缓存
# ---------------------------------------------------------------------------

_KB_CACHE: dict[tuple, tuple[list[Dict[str, Any]], list[list[str]], Any]] = {}
_KB_CACHE_MAX_SIZE = 4
_kb_build_locks_guard = threading.Lock()
_kb_build_locks: dict[tuple, threading.Lock] = {}


def _bm25_lock_for(cache_key: tuple) -> threading.Lock:
    with _kb_build_locks_guard:
        lock = _kb_build_locks.get(cache_key)
        if lock is None:
            lock = threading.Lock()
            _kb_build_locks[cache_key] = lock
        return lock


def _get_or_build_bm25(
    reference_path: str, config: AppConfig
) -> tuple[list[Dict[str, Any]], list[list[str]], Any] | None:
    """按 reference_path + 文件 mtime + bm25_k1/b 缓存 BM25 索引。"""
    try:
        mtime = os.path.getmtime(reference_path)
    except OSError:
        mtime = 0.0
    k1 = getattr(config.rag, "bm25_k1", 1.2)
    b = getattr(config.rag, "bm25_b", 0.75)
    cache_key = (reference_path, mtime, k1, b)
    if cache_key in _KB_CACHE:
        logger.debug("[RAG] BM25 索引缓存命中: %s", reference_path)
        return _KB_CACHE[cache_key]
    lock = _bm25_lock_for(cache_key)
    with lock:
        if cache_key in _KB_CACHE:
            logger.debug("[RAG] BM25 索引缓存命中（并发等待后）: %s", reference_path)
            return _KB_CACHE[cache_key]
        logger.info("[RAG] BM25 索引未命中缓存，正在构建: %s (k1=%s, b=%s)", reference_path, k1, b)
        texts, documents = load_texts_from_jsonl(reference_path)
        if not documents:
            return None
        tokenized_corpus = query_chinese_tokenizer(texts)
        bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
        while len(_KB_CACHE) >= _KB_CACHE_MAX_SIZE and _KB_CACHE:
            _KB_CACHE.pop(next(iter(_KB_CACHE)))
        _KB_CACHE[cache_key] = (documents, tokenized_corpus, bm25)
        return documents, tokenized_corpus, bm25


# ---------------------------------------------------------------------------
# 稠密 embedding：HTTP API / 本地 SentenceTransformer
# ---------------------------------------------------------------------------

class _HttpEmbeddingModel:
    """通过 HTTP API（OpenAI 兼容 /v1/embeddings）调用的 embedding 模型，
    接口与 SentenceTransformer.encode 一致。"""

    def __init__(self, base_url: str, model_name: str | None = None, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        if "/v1/embeddings" in self.base_url or "/embeddings" in self.base_url:
            self.embed_url = self.base_url
        else:
            self.embed_url = self.base_url + "/v1/embeddings"
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = self.base_url.rstrip("/").split("/")[-1] or "embedding"
        self.timeout = timeout
        logger.info("[Dense HTTP] embed_url=%s, model_name=%s", self.embed_url, self.model_name)

    def _post(self, body: dict) -> dict:
        """发送 HTTP 请求，优先 httpx，降级 requests。"""
        try:
            import httpx
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(self.embed_url, json=body)
                r.raise_for_status()
                return r.json()
        except ImportError:
            import requests as _req
            r = _req.post(self.embed_url, json=body, timeout=self.timeout)
            r.raise_for_status()
            return r.json()

    @staticmethod
    def _parse_embeddings(resp: dict) -> list[list[float]]:
        """从 OpenAI 兼容响应中解析 embedding 向量列表。"""
        items = resp.get("data", [])
        if not items and "embedding" in resp:
            items = [resp]
        items = sorted(items, key=lambda x: x.get("index", 0))
        return [item.get("embedding", []) for item in items]

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        max_workers: int = 4,
    ):
        import numpy as np
        import time as _time
        total = len(texts)
        num_batches = (total + batch_size - 1) // batch_size
        logger.info("[Dense HTTP encode] total=%d, batch_size=%d, batches=%d", total, batch_size, num_batches)
        t0 = _time.time()

        batches = []
        for i in range(0, total, batch_size):
            batches.append((i, texts[i: i + batch_size]))

        if max_workers > 1 and len(batches) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            results_map: dict[int, list[list[float]]] = {}

            def _encode_batch(idx_and_batch):
                idx, batch = idx_and_batch
                body = {"model": self.model_name, "input": batch}
                resp = self._post(body)
                return idx, self._parse_embeddings(resp)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_encode_batch, b): b[0] for b in batches}
                done_count = 0
                for future in as_completed(futures):
                    idx, parsed = future.result()
                    results_map[idx] = parsed
                    done_count += 1
                    if done_count % 20 == 0 or done_count == len(batches):
                        elapsed = _time.time() - t0
                        logger.info("[Dense HTTP encode] progress: %d/%d batches, %.1fs", done_count, len(batches), elapsed)

            all_embeddings: list[list[float]] = []
            for i, _ in batches:
                all_embeddings.extend(results_map[i])
        else:
            all_embeddings = []
            for i, batch in batches:
                body = {"model": self.model_name, "input": batch}
                resp = self._post(body)
                parsed = self._parse_embeddings(resp)
                all_embeddings.extend(parsed)

        elapsed = _time.time() - t0
        if not all_embeddings:
            logger.warning("[Dense HTTP encode] no embeddings returned!")
            return np.empty((0, 0), dtype="float32")
        result = np.array(all_embeddings, dtype="float32")
        logger.info("[Dense HTTP encode] shape=%s, %.1fs (%.0f texts/sec)", result.shape, elapsed, total / max(elapsed, 0.001))
        return result


# 稠密模型与索引缓存
_DENSE_MODEL_CACHE: dict[str, Any] = {}
_DENSE_INDEX_CACHE: dict[tuple[str, float, str], tuple[list[Dict[str, Any]], Any, Any]] = {}
_DENSE_CACHE_MAX_SIZE = 2
_dense_build_locks_guard = threading.Lock()
_dense_build_locks: dict[tuple[str, float, str], threading.Lock] = {}


def _dense_lock_for(cache_key: tuple[str, float, str]) -> threading.Lock:
    with _dense_build_locks_guard:
        lock = _dense_build_locks.get(cache_key)
        if lock is None:
            lock = threading.Lock()
            _dense_build_locks[cache_key] = lock
        return lock


def _resolve_dense_device(config: AppConfig | None) -> str | None:
    raw = (os.environ.get("RAG_DENSE_DEVICE") or "").strip()
    if not raw and config is not None:
        raw = (getattr(config.rag, "dense_device", "") or "").strip()
    if not raw or raw.lower() in ("auto", "default"):
        return None
    return raw


def _resolve_dense_encode_devices(config: AppConfig | None) -> list[str]:
    """全量文档编码使用的设备列表；可用 RAG_DENSE_ENCODE_DEVICES 覆盖。"""
    raw = (os.environ.get("RAG_DENSE_ENCODE_DEVICES") or "").strip()
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    if config is None:
        return []
    devices = getattr(config.rag, "dense_encode_devices", []) or []
    if isinstance(devices, str):
        return [item.strip() for item in devices.split(",") if item.strip()]
    return [str(item).strip() for item in devices if str(item).strip()]


def _load_dense_model(model_name: str, config: AppConfig | None = None) -> Any:
    """懒加载 embedding 模型：
    - http(s) URL → HTTP API（_HttpEmbeddingModel）
    - 其他 → 本地 SentenceTransformer
    加载失败返回 None，退化为仅 BM25。
    """
    if model_name in _DENSE_MODEL_CACHE:
        return _DENSE_MODEL_CACHE[model_name]

    # ---- HTTP API 模式 ----
    if model_name.startswith("http://") or model_name.startswith("https://"):
        body_model = (getattr(config.rag, "dense_http_model", "") or "").strip() if config else ""
        logger.info("[Dense] 使用 HTTP API: %s (body model: %s)", model_name, body_model or "从 URL 解析")
        try:
            wrapper = _HttpEmbeddingModel(model_name, model_name=body_model or None)
            _DENSE_MODEL_CACHE[model_name] = wrapper
            return wrapper
        except Exception as e:
            logger.warning("[Dense] HTTP 模型初始化失败（退化为仅 BM25）: %s", e)
            return None

    # ---- 本地 SentenceTransformer 模式 ----
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("[Dense] 未安装 sentence_transformers，稠密检索已禁用。pip install sentence-transformers")
        return None
    logger.info("[Dense] 加载本地模型: %s", model_name)
    paths_to_try = [model_name]
    if os.path.isabs(model_name) or model_name.startswith("/"):
        for old, new in [("bge-small-zh-v1.5", "bge-small-zh-v1___5"),
                         ("bge-small-zh-v1___5", "bge-small-zh-v1.5")]:
            if old in model_name and not os.path.isdir(model_name):
                alt = model_name.replace(old, new)
                if os.path.isdir(alt):
                    paths_to_try.append(alt)
    device_arg = _resolve_dense_device(config)
    for path in paths_to_try:
        try:
            if device_arg:
                model = SentenceTransformer(path, device=device_arg)
                logger.info("[Dense] SentenceTransformer device=%s", device_arg)
            else:
                model = SentenceTransformer(path)
            _DENSE_MODEL_CACHE[model_name] = model
            return model
        except Exception as e:
            if path != paths_to_try[-1]:
                logger.info("[Dense] 尝试备用路径: %s", path)
                continue
            logger.warning("[Dense] 模型加载失败（退化为仅 BM25）: %s", e)
            return None
    return None


def _resolve_dense_model_path(model_name: str) -> str:
    """config 中的 dense_model 相对 deep_research 目录解析。HTTP URL 原样返回。"""
    if not model_name:
        return model_name
    if model_name.startswith("http://") or model_name.startswith("https://"):
        return model_name
    if os.path.isabs(model_name) or model_name.startswith("/"):
        return model_name
    if "/" in model_name and "\\" not in model_name and len(model_name.split("/")) == 2:
        return model_name  # HuggingFace 模型 id 如 BAAI/bge-xxx
    base_dir = _rag_base_dir()
    candidate = base_dir / model_name
    if candidate.is_dir() or candidate.is_file():
        return str(candidate.resolve())
    return model_name


def _resolve_effective_dense_model(config: AppConfig) -> str:
    """根据 use_dense_http 选择 HTTP URL 或本地模型路径。"""
    use_http = bool(getattr(config.rag, "use_dense_http", False))
    http_url = (getattr(config.rag, "dense_http_url", "") or "").strip()
    if use_http and http_url:
        return http_url
    local_model = getattr(config.rag, "dense_model", "BAAI/bge-small-zh-v1.5") or "BAAI/bge-small-zh-v1.5"
    return _resolve_dense_model_path(local_model)


def _dense_disk_cache_path(reference_path: str, model_name: str, mtime: float, n_docs: int) -> Path:
    import hashlib

    cache_dir = Path(reference_path).resolve().parent / ".dense_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{Path(reference_path).name}|{model_name}|{int(mtime)}|{n_docs}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{digest}.npy"


def _get_or_build_dense_index(
    reference_path: str,
    documents: list[Dict[str, Any]],
    index_texts: list[str],
    config: AppConfig,
) -> tuple[list[Dict[str, Any]], Any, Any] | None:
    """构建或从缓存获取 FAISS 稠密索引；向量落盘，避免每次启动重新编码。"""
    model_name = _resolve_effective_dense_model(config)
    try:
        mtime = os.path.getmtime(reference_path)
    except OSError:
        mtime = 0.0
    cache_key = (reference_path, mtime, model_name)
    if cache_key in _DENSE_INDEX_CACHE:
        logger.debug("[Dense] 索引缓存命中: %s", reference_path)
        return _DENSE_INDEX_CACHE[cache_key]

    lock = _dense_lock_for(cache_key)
    with lock:
        if cache_key in _DENSE_INDEX_CACHE:
            logger.debug("[Dense] 索引缓存命中（并发等待后）: %s", reference_path)
            return _DENSE_INDEX_CACHE[cache_key]
        logger.info("[Dense] 索引缓存未命中，正在构建: %s (model=%s)", reference_path, model_name)
        try:
            import numpy as np
            import faiss
        except ImportError:
            logger.warning("[Dense] 未安装 faiss，稠密检索已禁用。pip install faiss-cpu")
            return None
        n = len(index_texts)
        if n == 0:
            logger.warning("[Dense] index_texts 为空，跳过向量索引构建")
            return None
        disk_path = _dense_disk_cache_path(reference_path, model_name, mtime, n)
        embeddings = None
        if disk_path.is_file():
            try:
                embeddings = np.load(str(disk_path)).astype("float32")
                if embeddings.shape[0] != n:
                    logger.warning("[Dense] 磁盘缓存条数不匹配，将重新编码: %s", disk_path)
                    embeddings = None
                else:
                    logger.info("[Dense] 命中磁盘向量缓存: %s shape=%s", disk_path, embeddings.shape)
            except Exception as e:
                logger.warning("[Dense] 读取磁盘向量缓存失败，将重新编码: %s", e)
                embeddings = None

        model = _load_dense_model(model_name, config)
        if model is None:
            return None

        if embeddings is None:
            batch_size = max(1, int(getattr(config.rag, "dense_batch_size", 32)))
            encode_devices = _resolve_dense_encode_devices(config)
            logger.info(
                "[Dense] 开始编码文档: total=%d, batch_size=%d, encode_devices=%s；首次会慢，之后读磁盘缓存",
                n,
                batch_size,
                encode_devices or [_resolve_dense_device(config) or "auto"],
            )
            try:
                if len(encode_devices) > 1 and hasattr(model, "start_multi_process_pool"):
                    logger.info("[Dense] 使用多卡多进程编码: %s", encode_devices)
                    pool = model.start_multi_process_pool(target_devices=encode_devices)
                    try:
                        embeddings = model.encode_multi_process(
                            index_texts,
                            pool,
                            batch_size=batch_size,
                            chunk_size=max(128, n // (len(encode_devices) * 4)),
                        )
                        embeddings = np.asarray(embeddings, dtype="float32")
                    finally:
                        model.stop_multi_process_pool(pool)
                    logger.info("[Dense] 多卡编码完成: shape=%s", embeddings.shape)
                else:
                    parts: list[Any] = []
                    total_batches = (n + batch_size - 1) // batch_size
                    log_every = max(1, total_batches // 20)
                    for batch_idx, start in enumerate(range(0, n, batch_size), 1):
                        end = min(start + batch_size, n)
                        part = model.encode(
                            index_texts[start:end],
                            batch_size=end - start,
                            show_progress_bar=False,
                        )
                        parts.append(np.asarray(part, dtype="float32"))
                        if batch_idx == 1 or batch_idx % log_every == 0 or end == n:
                            logger.info(
                                "[Dense] 编码进度: %d/%d (%.1f%%)",
                                end,
                                n,
                                100.0 * end / n,
                            )
                    embeddings = np.vstack(parts)
                np.save(str(disk_path), embeddings)
                logger.info("[Dense] 已写入磁盘向量缓存: %s shape=%s", disk_path, embeddings.shape)
            except Exception as e:
                logger.error("[Dense] 文档编码失败: %s", e)
                return None
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        while len(_DENSE_INDEX_CACHE) >= _DENSE_CACHE_MAX_SIZE and _DENSE_INDEX_CACHE:
            _DENSE_INDEX_CACHE.pop(next(iter(_DENSE_INDEX_CACHE)))
        _DENSE_INDEX_CACHE[cache_key] = (documents, index, model)
        return documents, index, model


def prebuild_indices(config: AppConfig) -> None:
    """服务启动后后台预热 BM25 / Dense 索引，不阻塞前端启动。"""
    try:
        reference_path = _resolve_local_reference_path(
            getattr(config.rag, "reference_filenames", None)
        )
        cached = _get_or_build_bm25(reference_path, config)
        if not cached:
            logger.warning("[Prebuild] BM25 预热失败")
            return
        documents, _, _ = cached
        logger.info("[Prebuild] BM25 预热完成: %d documents", len(documents))
        if getattr(config.rag, "use_dense", False):
            index_texts = [
                str(doc.get("_index_text") or doc.get("text", "")).strip()
                for doc in documents
            ]
            if _get_or_build_dense_index(reference_path, documents, index_texts, config):
                logger.info("[Prebuild] Dense 预热完成")
            else:
                logger.warning("[Prebuild] Dense 预热失败，运行时将退化为 BM25")
    except Exception as e:
        logger.error("[Prebuild] 异常: %s", e)


# ---------------------------------------------------------------------------
# 分数归一化 & 加权融合
# ---------------------------------------------------------------------------

def _normalize_scores(candidates: list[tuple[int, float]]) -> dict[int, float]:
    """将候选列表的原始分数 min-max 归一化到 [0, 1]。"""
    if not candidates:
        return {}
    scores = [sc for _, sc in candidates]
    s_min, s_max = min(scores), max(scores)
    rng = s_max - s_min
    if rng > 0:
        return {idx: (sc - s_min) / rng for idx, sc in candidates}
    else:
        return {idx: 1.0 for idx, _ in candidates}


def _weighted_merge(
    bm25_list: list[tuple[int, float]],
    dense_list: list[tuple[int, float]],
    bm25_weight: float = 0.3,
    dense_weight: float = 0.7,
    single_channel_penalty: float = 0.5,
) -> list[tuple[int, float]]:
    """加权融合 BM25 与稠密检索结果。

    对两路各自做 min-max 归一化到 [0,1]，再按权重加权求和。
    单通道惩罚：仅出现在一路的文档总分再乘以惩罚系数，压低无关召回。
    最终 score ∈ [0, 1]，可直接配合 score_threshold 过滤。
    """
    bm25_norm = _normalize_scores(bm25_list)
    dense_norm = _normalize_scores(dense_list)
    dense_raw: dict[int, float] = {idx: max(0.0, sc) for idx, sc in dense_list}
    all_ids = set(bm25_norm.keys()) | set(dense_norm.keys())
    fused: dict[int, float] = {}
    for idx in all_ids:
        sc = 0.0
        in_bm25 = idx in bm25_norm
        in_dense = idx in dense_norm
        if in_bm25:
            sc += bm25_weight * bm25_norm[idx]
        if in_dense:
            sc += dense_weight * dense_norm[idx]
        if single_channel_penalty < 1.0 and (in_bm25 != in_dense):
            if in_dense and not in_bm25:
                raw_sim = dense_raw.get(idx, 0.0)
                sc *= max(single_channel_penalty, raw_sim)
            else:
                sc *= single_channel_penalty
        fused[idx] = sc
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# 主检索函数
# ---------------------------------------------------------------------------

_rag_logic_logged = False
_dense_fallback_logged = False


def retrieve_from_knowledge_base(
    query: str, config: AppConfig, reference_filenames: list[str] | None = None
) -> List[Dict[str, Any]] | None:
    """在文档集合中搜索与查询最相关的法条。

    BM25（必选）+ 可选稠密检索（HTTP API 或本地 SentenceTransformer），
    使用加权融合合并两路结果。
    """
    try:
        filenames = (
            reference_filenames
            if reference_filenames is not None
            else getattr(config.rag, "reference_filenames", None)
        )
        reference_path = _resolve_local_reference_path(filenames)

        # 仅首次打出检索配置
        global _rag_logic_logged
        if not _rag_logic_logged:
            _rag_logic_logged = True
            use_dense = getattr(config.rag, "use_dense", False)
            dense_source = _resolve_effective_dense_model(config) if use_dense else "disabled"
            logger.info(
                "[RAG] 数据=%s | TOP_K=%s | use_dense=%s | dense=%s | threshold=%s",
                reference_path,
                getattr(config.rag, "TOP_K", 6),
                use_dense,
                dense_source,
                getattr(config.rag, "score_threshold", 0),
            )

        # ---- BM25 检索 ----
        cached = _get_or_build_bm25(reference_path, config)
        if not cached:
            logger.warning("[RAG] BM25 索引构建失败或无文档")
            return None
        documents, tokenized_corpus, bm25 = cached
        tokenized_query = chinese_tokenizer(query)
        scores = bm25.get_scores(tokenized_query)

        top_k = int(getattr(config.rag, "TOP_K", 6))
        over_k = min(max(top_k * 2, top_k + 5), len(scores))
        top_indices_bm25 = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:over_k]
        bm25_candidates = [(i, float(scores[i])) for i in top_indices_bm25 if scores[i] > 0]

        # BM25 预过滤
        bm25_min_ratio = float(getattr(config.rag, "bm25_min_score_ratio", 0.1))
        if bm25_candidates and bm25_min_ratio > 0:
            bm25_max = bm25_candidates[0][1]
            cutoff = bm25_max * bm25_min_ratio
            bm25_candidates = [(i, sc) for i, sc in bm25_candidates if sc >= cutoff]
            logger.debug("[RAG] BM25 预过滤: max=%.4f, cutoff=%.4f, 剩余 %d 候选",
                         bm25_max, cutoff, len(bm25_candidates))

        # ---- 稠密检索 + 融合 ----
        use_dense = getattr(config.rag, "use_dense", False)
        candidates: list[tuple[int, float]]

        if use_dense and documents:
            index_texts = [
                str(doc.get("_index_text") or doc.get("text", "")).strip()
                for doc in documents
            ]
            dense_cached = _get_or_build_dense_index(
                reference_path, documents, index_texts, config
            )
            if dense_cached:
                _, faiss_index, model = dense_cached
                import numpy as np
                import faiss
                q_emb = model.encode([query], show_progress_bar=False).astype("float32")
                faiss.normalize_L2(q_emb)
                dense_top_k = getattr(config.rag, "dense_top_k", 20)
                search_k = min(dense_top_k, faiss_index.ntotal)
                if search_k > 0:
                    D, I = faiss_index.search(q_emb, search_k)
                    dense_candidates = [(int(I[0][i]), float(D[0][i])) for i in range(len(I[0]))]
                    bm25_w = float(getattr(config.rag, "bm25_weight", 0.3))
                    dense_w = float(getattr(config.rag, "dense_weight", 0.7))
                    single_penalty = float(getattr(config.rag, "single_channel_penalty", 0.5))
                    merged = _weighted_merge(bm25_candidates, dense_candidates,
                                             bm25_weight=bm25_w, dense_weight=dense_w,
                                             single_channel_penalty=single_penalty)
                    candidates = merged[:top_k]
                    logger.debug("[RAG] BM25(w=%.1f) + Dense(w=%.1f) 融合, top_k=%s", bm25_w, dense_w, top_k)
                else:
                    bm25_norm = _normalize_scores(bm25_candidates)
                    candidates = [(idx, bm25_norm.get(idx, 0.0)) for idx, _ in bm25_candidates[:top_k]]
            else:
                bm25_norm = _normalize_scores(bm25_candidates)
                candidates = [(idx, bm25_norm.get(idx, 0.0)) for idx, _ in bm25_candidates[:top_k]]
                global _dense_fallback_logged
                if not _dense_fallback_logged:
                    _dense_fallback_logged = True
                    logger.info("[RAG] 稠密检索不可用，已退化为仅 BM25")
        else:
            bm25_norm = _normalize_scores(bm25_candidates)
            candidates = [(idx, bm25_norm.get(idx, 0.0)) for idx, _ in bm25_candidates[:top_k]]

        # ---- 格式化结果 ----
        results_formatted = []
        for idx, sc in candidates:
            doc = documents[idx]
            display_text = str(doc.get("_display_text") or doc.get("text", "")).strip()
            results_formatted.append(
                {
                    "law_name": doc.get("law_name"),
                    "article": doc.get("article"),
                    "category": doc.get("category"),
                    "number": doc.get("number"),
                    "text": display_text,
                    "score": round(float(sc), 4),
                }
            )

        # 分数阈值过滤
        score_threshold = float(getattr(config.rag, "score_threshold", 0))
        if score_threshold > 0:
            before = len(results_formatted)
            results_formatted = [r for r in results_formatted if r["score"] >= score_threshold]
            if len(results_formatted) < before:
                logger.debug("[RAG] 阈值过滤 %.2f: %d -> %d 条", score_threshold, before, len(results_formatted))

        # 日志：简洁的检索摘要
        logger.info("[RAG] query='%s' | 返回 %d 条 (threshold=%.2f)",
                     query[:60], len(results_formatted[:top_k]), score_threshold)

        return results_formatted[:top_k]

    except Exception as e:
        logger.error("[RAG] 检索失败: %s", e)
        return None


# ---------------------------------------------------------------------------
# 测试入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    query = "管理人由依法设立的公司或者合伙企业担任，有限合伙人可以随时退伙吗？"
    config = AppConfig(rag=AppConfig().rag)
    response = retrieve_from_knowledge_base(query, config=config)
    print(response)
    print("检索完成。总共找到", len(response) if response else 0, "条结果。")
