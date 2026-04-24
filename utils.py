# -*- coding: utf-8 -*-
"""
# 公共能力模块：各种公共抽象函数

Author: wjianxz
Date: 2025-11-13
"""
from typing import Callable, Dict, Any, Optional, Tuple, Union
import json
import re
import time
import threading
import requests
from requests import Response
import random
from collections import Counter
from deep_research.protocal import ReferenceList

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  内部工具函数
# ---------------------------------------------------------------------------

def _clean_model_output(text: str) -> str:
    """清理模型输出中的格式指令残留，如 '（至少100字）' 等。"""
    if not text:
        return text
    cleaned = re.sub(r'[（\(]\s*至少\s*\d+\s*字\s*[）\)]', '', text)
    return cleaned.strip()


_request_lock = threading.Lock()
_last_request_time = 0.0
_MIN_REQUEST_INTERVAL = 2.0


def set_min_request_interval(seconds: float) -> None:
    """设置两次 HTTP 请求之间的最小间隔（秒）。0 表示不额外等待。供 config 调优并行与限流。"""
    global _MIN_REQUEST_INTERVAL
    _MIN_REQUEST_INTERVAL = max(0.0, float(seconds))


def _throttle():
    """全局节流：确保连续请求之间至少间隔 _MIN_REQUEST_INTERVAL 秒。"""
    global _last_request_time
    if _MIN_REQUEST_INTERVAL <= 0:
        return
    with _request_lock:
        now = time.time()
        elapsed = now - _last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            wait = _MIN_REQUEST_INTERVAL - elapsed
            time.sleep(wait)
        _last_request_time = time.time()


def _normalize_api_url(api_url: str) -> str:
    """
    兼容两种配置：
    1) 直接给完整 OpenAI 接口地址：.../v1/chat/completions
    2) 只给模型网关前缀地址：.../deepseek-v32
    """
    base = (api_url or "").strip().rstrip("/")
    if not base:
        return api_url
    if base.endswith("/v1/chat/completions"):
        return base
    return f"{base}/v1/chat/completions"


# ---------------------------------------------------------------------------
#  输出质量检测
# ---------------------------------------------------------------------------

def is_invalid_output(text: str) -> bool:
    if not text:
        return True
    s = str(text).strip()
    if len(s) < 20:
        return False
    return _is_gibberish(s) or _is_repetitive(s)


def _is_gibberish(text: str) -> bool:
    total = len(text)
    if total < 50:
        return False
    zh_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    alnum_count = len(re.findall(r"[A-Za-z0-9]", text))
    punct_count = len(re.findall(r"[^\w\u4e00-\u9fff\s]", text))
    valid_ratio = (zh_count + alnum_count) / total
    punct_ratio = punct_count / total
    if valid_ratio < 0.4:
        return True
    if punct_ratio > 0.35:
        return True
    if re.search(r"(.)\1{30,}", text):
        return True
    return False


def _is_repetitive(text: str) -> bool:
    total = len(text)
    if total < 120:
        return False
    n = 10
    grams = [text[i : i + n] for i in range(0, total - n + 1)]
    if not grams:
        return False
    counts = Counter(grams)
    top_gram, top_count = counts.most_common(1)[0]
    coverage = (top_count * n) / total
    if top_count >= 5 and coverage > 0.25:
        return True
    return False


# ---------------------------------------------------------------------------
#  模型请求
# ---------------------------------------------------------------------------

def send_request_to_model_dr(
    user_query: str,
    prompt_builder: Callable[
        [str, Optional[str], Optional[str], Optional[Union[str, ReferenceList]]], str
    ],
    topic_report: Optional[Union[str, None]] = None,
    history_answer: Optional[Union[str, None]] = None,
    references: Optional[Union[str, ReferenceList, None]] = None,
    model_name: str = "qwen3",
    api_url: str = "http://localhost:8008/v1/chat/completions",
    timeout: int = 120,
    temperature: float = 0.8,
    max_tokens: int = 8192,
    top_p: float = 0.92,
    repetition_penalty: float = 1.2,
) -> Optional[str]:
    """
    向 vLLM 服务发送请求（deep research 写作管线专用，支持 topic_report 参数）。
    """
    normalized_api_url = _normalize_api_url(api_url)
    user_prompt = prompt_builder(user_query, topic_report, history_answer, references)
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "你是一位专业的写作助手，擅长撰写结构清晰、内容详实的深度文章。",
            },
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}
    proxies = None

    max_retries = 6
    last_resp: Optional[Response] = None
    last_exc: Optional[BaseException] = None
    adaptive_max_tokens = int(max_tokens)

    # requests timeout 既包含 connect 也包含 read；这里将 connect 单独缩短，避免卡在建连阶段太久
    connect_timeout_s = 10
    read_timeout_s = max(30, int(timeout))

    for attempt in range(max_retries):
        try:
            _throttle()
            payload["max_tokens"] = adaptive_max_tokens
            last_resp = requests.post(
                url=normalized_api_url,
                headers=headers,
                data=json.dumps(payload),
                proxies=proxies,
                timeout=(connect_timeout_s, read_timeout_s),
            )

            if last_resp.status_code == 503 and attempt < max_retries - 1:
                retry_after = last_resp.headers.get("Retry-After")
                if retry_after and str(retry_after).strip().isdigit():
                    wait = int(str(retry_after).strip())
                else:
                    wait = 2 ** attempt + 1
                wait = min(30, wait) + random.uniform(0.0, 0.8)
                logger.warning(
                    "收到 503，%d 秒后重试 (%d/%d) url=%s",
                    int(wait),
                    attempt + 1,
                    max_retries,
                    normalized_api_url,
                )
                time.sleep(wait)
                continue

            # 适配部分网关：当 max_tokens 过大时会 400；自动降档重试一次
            if last_resp.status_code == 400 and attempt < max_retries - 1:
                body = (last_resp.text or "")[:800]
                if any(k in body.lower() for k in ["max_tokens", "context", "length", "limit"]):
                    new_max = min(adaptive_max_tokens, 2048)
                    if new_max < adaptive_max_tokens:
                        logger.warning(
                            "收到 400 且疑似 max_tokens/上下文限制，max_tokens %d -> %d，重试 (%d/%d)",
                            adaptive_max_tokens,
                            new_max,
                            attempt + 1,
                            max_retries,
                        )
                        adaptive_max_tokens = new_max
                        continue

            last_resp.raise_for_status()

            result: Dict[str, Any] = last_resp.json()
            answer: str = result["choices"][0]["message"]["content"]
            return _clean_model_output(answer)

        except requests.exceptions.Timeout as e:
            last_exc = e
            logger.warning(
                "请求超时（>%ss）(%d/%d) url=%s",
                timeout,
                attempt + 1,
                max_retries,
                normalized_api_url,
            )
            continue
        except requests.exceptions.HTTPError as e:
            last_exc = e
            status = last_resp.status_code if last_resp is not None else "N/A"
            body = (last_resp.text or "")[:1200] if last_resp is not None else ""
            logger.error(
                "HTTP 错误 status=%s url=%s model=%s max_tokens=%s body=%s",
                status,
                normalized_api_url,
                model_name,
                adaptive_max_tokens,
                body,
            )
            continue
        except requests.exceptions.RequestException as e:
            last_exc = e
            logger.exception(
                "网络请求异常 (%d/%d) url=%s model=%s",
                attempt + 1,
                max_retries,
                normalized_api_url,
                model_name,
            )
            continue
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            last_exc = e
            raw = (last_resp.text or "")[:1200] if last_resp is not None else "N/A"
            logger.exception("响应解析失败 url=%s raw=%s", normalized_api_url, raw)
            continue

    logger.error(
        "模型请求异常（已重试 %d 次）url=%s model=%s last_exc=%r",
        max_retries,
        normalized_api_url,
        model_name,
        last_exc,
    )
    return None


def send_request_to_model(
    user_query: str,
    prompt_builder: Callable[[str, Any, Any], str],
    history_answer: Optional[list[str] | str] = None,
    references: Optional[Union[str, ReferenceList]] = None,
    model_name: str = "qwen3",
    api_url: str = "http://localhost:8008/v1/chat/completions",
    timeout: int = 120,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
) -> Optional[str]:
    """向语言模型服务发送请求并获取回复（通用版）。"""
    normalized_api_url = _normalize_api_url(api_url)
    user_prompt = prompt_builder(user_query, history_answer, references)
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "你是一位专业的写作助手，擅长撰写结构清晰、内容详实的深度文章。",
            },
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}
    proxies = None

    max_retries = 6
    last_resp: Optional[Response] = None
    last_exc: Optional[BaseException] = None
    adaptive_max_tokens = int(max_tokens)

    connect_timeout_s = 10
    read_timeout_s = max(30, int(timeout))

    for attempt in range(max_retries):
        try:
            _throttle()
            payload["max_tokens"] = adaptive_max_tokens
            last_resp = requests.post(
                url=normalized_api_url,
                headers=headers,
                data=json.dumps(payload),
                proxies=proxies,
                timeout=(connect_timeout_s, read_timeout_s),
            )
            if last_resp.status_code == 503 and attempt < max_retries - 1:
                retry_after = last_resp.headers.get("Retry-After")
                if retry_after and str(retry_after).strip().isdigit():
                    wait = int(str(retry_after).strip())
                else:
                    wait = 2 ** attempt + 1
                wait = min(30, wait) + random.uniform(0.0, 0.8)
                logger.warning(
                    "收到 503，%d 秒后重试 (%d/%d) url=%s",
                    int(wait),
                    attempt + 1,
                    max_retries,
                    normalized_api_url,
                )
                time.sleep(wait)
                continue

            if last_resp.status_code == 400 and attempt < max_retries - 1:
                body = (last_resp.text or "")[:800]
                if any(k in body.lower() for k in ["max_tokens", "context", "length", "limit"]):
                    new_max = min(adaptive_max_tokens, 2048)
                    if new_max < adaptive_max_tokens:
                        logger.warning(
                            "收到 400 且疑似 max_tokens/上下文限制，max_tokens %d -> %d，重试 (%d/%d)",
                            adaptive_max_tokens,
                            new_max,
                            attempt + 1,
                            max_retries,
                        )
                        adaptive_max_tokens = new_max
                        continue

            last_resp.raise_for_status()

            result: Dict[str, Any] = last_resp.json()
            answer: str = result["choices"][0]["message"]["content"]
            return _clean_model_output(answer)

        except requests.exceptions.Timeout as e:
            last_exc = e
            logger.warning(
                "请求超时（>%ss）(%d/%d) url=%s",
                timeout,
                attempt + 1,
                max_retries,
                normalized_api_url,
            )
            continue
        except requests.exceptions.HTTPError as e:
            last_exc = e
            status = last_resp.status_code if last_resp is not None else "N/A"
            body = (last_resp.text or "")[:1200] if last_resp is not None else ""
            logger.error(
                "HTTP 错误 status=%s url=%s model=%s max_tokens=%s body=%s",
                status,
                normalized_api_url,
                model_name,
                adaptive_max_tokens,
                body,
            )
            continue
        except requests.exceptions.RequestException as e:
            last_exc = e
            logger.exception(
                "网络请求异常 (%d/%d) url=%s model=%s",
                attempt + 1,
                max_retries,
                normalized_api_url,
                model_name,
            )
            continue
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            last_exc = e
            raw = (last_resp.text or "")[:1200] if last_resp is not None else "N/A"
            logger.exception("响应解析失败 url=%s raw=%s", normalized_api_url, raw)
            continue

    logger.error(
        "模型请求异常（已重试 %d 次）url=%s model=%s last_exc=%r",
        max_retries,
        normalized_api_url,
        model_name,
        last_exc,
    )
    return None


def send_request_to_model_streaming(
    user_query: str,
    prompt_builder: Callable[[str, Any, Any], str],
    result_queue,
    history_answer: Optional[list[str] | str] = None,
    references: Optional[Union[str, ReferenceList]] = None,
    model_name: str = "qwen3",
    api_url: str = "http://localhost:8008/v1/chat/completions",
    timeout: int = 120,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
) -> Optional[str]:
    """流式向语言模型服务发送请求，实时将内容推送到队列。"""
    normalized_api_url = _normalize_api_url(api_url)
    user_prompt = prompt_builder(user_query, history_answer, references)
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "你是一位专业的写作助手，擅长撰写结构清晰、内容详实的深度文章。",
            },
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "stream": True,
    }

    headers = {"Content-Type": "application/json"}
    proxies = None

    max_retries = 6
    last_resp: Optional[Response] = None
    last_exc: Optional[BaseException] = None
    adaptive_max_tokens = int(max_tokens)
    connect_timeout_s = 10
    read_timeout_s = max(30, int(timeout))

    for attempt in range(max_retries):
        try:
            _throttle()
            payload["max_tokens"] = adaptive_max_tokens
            response = requests.post(
                url=normalized_api_url,
                headers=headers,
                data=json.dumps(payload),
                proxies=proxies,
                timeout=(connect_timeout_s, read_timeout_s),
                stream=True,
            )
            last_resp = response
            if response.status_code == 503 and attempt < max_retries - 1:
                retry_after = response.headers.get("Retry-After")
                if retry_after and str(retry_after).strip().isdigit():
                    wait = int(str(retry_after).strip())
                else:
                    wait = 2 ** attempt + 1
                wait = min(30, wait) + random.uniform(0.0, 0.8)
                logger.warning(
                    "流式请求收到 503，%d 秒后重试 (%d/%d) url=%s",
                    int(wait),
                    attempt + 1,
                    max_retries,
                    normalized_api_url,
                )
                time.sleep(wait)
                continue
            if response.status_code == 400 and attempt < max_retries - 1:
                body = (response.text or "")[:800]
                if any(k in body.lower() for k in ["max_tokens", "context", "length", "limit"]):
                    new_max = min(adaptive_max_tokens, 2048)
                    if new_max < adaptive_max_tokens:
                        logger.warning(
                            "流式请求 400 疑似上下文限制，max_tokens %d -> %d，重试 (%d/%d)",
                            adaptive_max_tokens,
                            new_max,
                            attempt + 1,
                            max_retries,
                        )
                        adaptive_max_tokens = new_max
                        continue
            response.raise_for_status()

            full_content = ""
            stream_buffer = ""
            chunk_count = 0

            logger.info("开始流式读取响应...")
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        logger.info("流式读取完成，共 %d 个 chunk", chunk_count)
                        break
                    try:
                        chunk_data = json.loads(data_str)
                        delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if not content:
                            continue

                        chunk_count += 1
                        full_content += content
                        stream_buffer += content

                        if len(stream_buffer) >= 10:
                            result_queue.put([False, stream_buffer])
                            logger.debug("推送流式内容: %d 字符", len(stream_buffer))
                            stream_buffer = ""

                    except json.JSONDecodeError:
                        continue

            if stream_buffer:
                result_queue.put([False, stream_buffer])
                logger.debug("推送剩余流式内容: %d 字符", len(stream_buffer))

            final_answer = re.sub(
                r"<think>.*?</think>",
                "",
                full_content,
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()

            return final_answer if final_answer else full_content

        except requests.exceptions.Timeout as e:
            last_exc = e
            logger.warning(
                "流式请求超时（>%ss）(%d/%d) url=%s",
                timeout,
                attempt + 1,
                max_retries,
                normalized_api_url,
            )
            continue
        except requests.exceptions.HTTPError as e:
            last_exc = e
            status = last_resp.status_code if last_resp is not None else "N/A"
            body = (last_resp.text or "")[:1200] if last_resp is not None else ""
            logger.error(
                "流式 HTTP 错误 status=%s url=%s model=%s max_tokens=%s body=%s",
                status,
                normalized_api_url,
                model_name,
                adaptive_max_tokens,
                body,
            )
            continue
        except requests.exceptions.RequestException as e:
            last_exc = e
            logger.exception(
                "流式网络请求异常 (%d/%d) url=%s model=%s",
                attempt + 1,
                max_retries,
                normalized_api_url,
                model_name,
            )
            continue
        except Exception as e:
            last_exc = e
            logger.exception("流式处理异常 (%d/%d) url=%s", attempt + 1, max_retries, normalized_api_url)
            continue

    logger.error(
        "流式模型请求异常（已重试 %d 次）url=%s model=%s last_exc=%r",
        max_retries,
        normalized_api_url,
        model_name,
        last_exc,
    )
    return None


def send_request_to_model_dr_streaming(
    user_query: str,
    prompt_builder: Callable[
        [str, Optional[str], Optional[str], Optional[Union[str, ReferenceList]]], str
    ],
    result_queue,
    topic_report: Optional[Union[str, None]] = None,
    history_answer: Optional[Union[str, None]] = None,
    references: Optional[Union[str, ReferenceList, None]] = None,
    model_name: str = "qwen3",
    api_url: str = "http://localhost:8008/v1/chat/completions",
    timeout: int = 120,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
    *,
    progress_prefix: str = "",
) -> Optional[str]:
    """
    DeepResearch 专用的“流式”请求版本（支持 topic_report）。
    会把增量 token 推送到 result_queue（chunk[0]=False）。
    """
    normalized_api_url = _normalize_api_url(api_url)
    user_prompt = prompt_builder(user_query, topic_report, history_answer, references)
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "你是一位专业的写作助手，擅长撰写结构清晰、内容详实的长文。",
            },
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": int(max_tokens),
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}
    proxies = None

    max_retries = 6
    last_resp: Optional[Response] = None
    last_exc: Optional[BaseException] = None
    adaptive_max_tokens = int(max_tokens)

    connect_timeout_s = 10
    read_timeout_s = max(30, int(timeout))

    for attempt in range(max_retries):
        try:
            _throttle()
            payload["max_tokens"] = adaptive_max_tokens
            last_resp = requests.post(
                url=normalized_api_url,
                headers=headers,
                data=json.dumps(payload),
                proxies=proxies,
                timeout=(connect_timeout_s, read_timeout_s),
                stream=True,
            )

            if last_resp.status_code == 503 and attempt < max_retries - 1:
                retry_after = last_resp.headers.get("Retry-After")
                if retry_after and str(retry_after).strip().isdigit():
                    wait = int(str(retry_after).strip())
                else:
                    wait = 2 ** attempt + 1
                wait = min(30, wait) + random.uniform(0.0, 0.8)
                logger.warning(
                    "DR流式收到 503，%d 秒后重试 (%d/%d) url=%s",
                    int(wait),
                    attempt + 1,
                    max_retries,
                    normalized_api_url,
                )
                time.sleep(wait)
                continue

            if last_resp.status_code == 400 and attempt < max_retries - 1:
                body = (last_resp.text or "")[:800]
                if any(k in body.lower() for k in ["max_tokens", "context", "length", "limit"]):
                    new_max = min(adaptive_max_tokens, 2048)
                    if new_max < adaptive_max_tokens:
                        logger.warning(
                            "DR流式 400 疑似上下文限制，max_tokens %d -> %d，重试 (%d/%d)",
                            adaptive_max_tokens,
                            new_max,
                            attempt + 1,
                            max_retries,
                        )
                        adaptive_max_tokens = new_max
                        continue

            last_resp.raise_for_status()

            full_content = ""
            stream_buffer = ""

            for line in last_resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk_data = json.loads(data_str)
                    delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if not content:
                        continue
                    full_content += content
                    stream_buffer += content
                    if len(stream_buffer) >= 24:
                        # 推送增量，保持 UI “动起来”
                        result_queue.put([False, f"{progress_prefix}{stream_buffer}"])
                        stream_buffer = ""
                except json.JSONDecodeError:
                    continue

            if stream_buffer:
                result_queue.put([False, f"{progress_prefix}{stream_buffer}"])

            final_answer = re.sub(
                r"<think>.*?</think>",
                "",
                full_content,
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()
            return final_answer if final_answer else full_content

        except requests.exceptions.Timeout as e:
            last_exc = e
            logger.warning(
                "DR流式请求超时（>%ss）(%d/%d) url=%s",
                timeout,
                attempt + 1,
                max_retries,
                normalized_api_url,
            )
            continue
        except requests.exceptions.HTTPError as e:
            last_exc = e
            status = last_resp.status_code if last_resp is not None else "N/A"
            body = (last_resp.text or "")[:1200] if last_resp is not None else ""
            logger.error(
                "DR流式 HTTP 错误 status=%s url=%s model=%s max_tokens=%s body=%s",
                status,
                normalized_api_url,
                model_name,
                adaptive_max_tokens,
                body,
            )
            continue
        except requests.exceptions.RequestException as e:
            last_exc = e
            logger.exception(
                "DR流式网络请求异常 (%d/%d) url=%s model=%s",
                attempt + 1,
                max_retries,
                normalized_api_url,
                model_name,
            )
            continue
        except Exception as e:
            last_exc = e
            logger.exception("DR流式处理异常 (%d/%d) url=%s", attempt + 1, max_retries, normalized_api_url)
            continue

    logger.error(
        "DR流式模型请求异常（已重试 %d 次）url=%s model=%s last_exc=%r",
        max_retries,
        normalized_api_url,
        model_name,
        last_exc,
    )
    return None


# ---------------------------------------------------------------------------
#  解析工具
# ---------------------------------------------------------------------------

def parser_confidence(model_output: str, score: float = 0.85) -> Tuple[bool, float]:
    """从模型输出中解析总得分，并判断是否高于给定阈值。"""
    score_confidence = model_output.split("总得分")[-1]
    match = re.search(r"\d+\.\d+", score_confidence)
    match_score: float = 0.0
    if match:
        match_score = float(match.group())
    if match_score >= score:
        return True, match_score
    else:
        return False, match_score


def parser_sub_topic_output(text_llm_output: str) -> list[str]:
    """
    从大模型返回的文本中提取子问题/章节标题列表。
    支持换行分隔和编号标题挤在同一行两种格式。
    """
    text = text_llm_output.strip()
    if not text:
        return []

    lines = text.split("\n")
    subquestions = [line.strip() for line in lines if line.strip()]

    if len(subquestions) <= 1 and text:
        parts = re.split(r'(?=\d+[\.\、\s])', text)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 1:
            subquestions = parts

    return subquestions


def check_llm_output(
    output: Any,
    *,
    function_name: str = "llm_inference",
    query: Optional[str] = None,
    raise_on_none: bool = False,
) -> bool:
    """检测大模型函数返回值是否为 None，并记录异常日志。"""
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
