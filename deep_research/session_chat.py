# -*- coding: utf-8 -*-
"""
deep_research 的会话式入口。

对外提供「session_id → 维护历史 → 调用 deep_search_rag」的薄封装，
让调用方（LangGraph、Web、CLI、批跑脚本）只需要传 (session_id, query, reset)。

使用：
    from deep_research.session_chat import chat
    res = chat("sess-1", "<首轮问题>", reset=True)
    res2 = chat("sess-1", "<同一 session 的追问>")

环境变量：
    DEEP_RESEARCH_CONFIG    yaml 配置路径（相对 deep_research_dev 或绝对路径），默认 deep_research/config.yaml
    DEEP_CHAT_DEFAULT_MODE  默认对话模式（deep_thinking / chat / deep_research）
    DEEP_WEB_ENABLED        是否联网检索覆盖：未设置或 yaml/config/sync → 沿用
                            当前 yaml 里的 web.enabled；true/1/on 强制开，
                            false/0/off 强制关（仅在 deep_thinking 等分支生效）
    DEEP_RESEARCH_LLM_URL   （可选）运行时覆盖 config 中各业务模型的 server，
                            详见 _apply_model_runtime_overrides
    DEEP_RESEARCH_MODEL     （可选）运行时覆盖模型名 name
    DEEP_RESEARCH_MODEL_SECTIONS  （可选）逗号分隔，限制要覆盖的配置段，
                            默认覆盖 safemodel / deepresearch 等主要 LLM 段
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import logging

logger = logging.getLogger(__name__)

# 全局 session 状态：session_id -> list[HistoryMessage]
# HistoryMessage = list[QueryResult]，因此 _SESSIONS[id] 形态为 list[list[QueryResult]]
_SESSIONS: dict[str, list] = {}
_TURN_COUNT: dict[str, int] = {}
_LAST_ACTIVE: dict[str, float] = {}
_LOCK = threading.Lock()

# 配置缓存
_CONFIG_CACHE: Any | None = None
_CONFIG_PATH_USED: str | None = None


def _resolve_config_path(explicit_path: str | None = None) -> str:
    from deep_research.paths import default_config_path, resolve_project_path

    if explicit_path:
        return str(resolve_project_path(explicit_path))
    env_path = os.environ.get("DEEP_RESEARCH_CONFIG", "").strip()
    if env_path:
        return str(resolve_project_path(env_path))
    return default_config_path()


def _apply_deep_web_override(cfg: Any) -> None:
    """
    用环境变量 DEEP_WEB_ENABLED 覆盖 config.web.enabled。

    - 未设置、空串，或为 yaml/config/sync → 不改变（与配置文件一致）。
    """
    raw = os.environ.get("DEEP_WEB_ENABLED", "").strip().lower()
    if raw in ("", "yaml", "config", "sync"):
        return
    web_cfg = getattr(cfg, "web", None)
    if web_cfg is None:
        return
    if raw in ("1", "true", "yes", "on"):
        web_cfg.enabled = True
        logger.info("[session_chat] DEEP_WEB_ENABLED 覆盖联网为开启（原 yaml 可被覆盖）")
    elif raw in ("0", "false", "no", "off"):
        web_cfg.enabled = False
        logger.info("[session_chat] DEEP_WEB_ENABLED 覆盖联网为关闭（原 yaml 可被覆盖）")
    else:
        logger.warning(
            "[session_chat] 未知 DEEP_WEB_ENABLED=%r，合法：yaml/true/false/1/0/on/off，"
            "已保留配置文件中的 web.enabled=%s",
            os.environ.get("DEEP_WEB_ENABLED"),
            getattr(web_cfg, "enabled", None),
        )


def _apply_model_runtime_overrides(cfg: Any) -> None:
    """
    用 run_langgraph.py 写入的环境变量覆盖 deep_research 内部模型配置。

    这样 config.yaml 仍是默认配置；单次 LangGraph 跑批时只改 run_langgraph.py。
    """
    model_url = os.environ.get("DEEP_RESEARCH_LLM_URL", "").strip()
    model_name = os.environ.get("DEEP_RESEARCH_MODEL", "").strip()
    if not model_url and not model_name:
        return

    raw_sections = os.environ.get("DEEP_RESEARCH_MODEL_SECTIONS", "").strip()
    sections = [
        s.strip()
        for s in (
            raw_sections.split(",")
            if raw_sections
            else [
                "safemodel",
                "deepresearch",
                "speeddeepresearch",
                "confidence",
                "subquestion",
                "extractinfo",
                "beautifulformat",
                "session",
                "rewrite",
            ]
        )
        if s.strip()
    ]
    for section in sections:
        model_cfg = getattr(cfg, section, None)
        if model_cfg is None:
            continue
        if model_url:
            model_cfg.server = model_url
        if model_name:
            model_cfg.name = model_name

    logger.info(
        "[session_chat] deep_research 模型运行时覆盖: url=%s, model=%s, sections=%s",
        model_url or "<keep yaml>",
        model_name or "<keep yaml>",
        ",".join(sections),
    )


def get_config(config_path: str | None = None):
    """懒加载并缓存 AppConfig。"""
    global _CONFIG_CACHE, _CONFIG_PATH_USED
    path = _resolve_config_path(config_path)
    if _CONFIG_CACHE is not None and path == _CONFIG_PATH_USED:
        return _CONFIG_CACHE

    from deep_research.parser_config import load_validated_config

    cfg = load_validated_config(path)
    _CONFIG_CACHE = cfg
    _CONFIG_PATH_USED = path
    logger.info("[session_chat] AppConfig 已加载: %s", path)
    return cfg


def new_session_id() -> str:
    return f"sess-{uuid.uuid4().hex[:12]}"


def reset_session(session_id: str) -> None:
    with _LOCK:
        _SESSIONS.pop(session_id, None)
        _TURN_COUNT.pop(session_id, None)
        _LAST_ACTIVE.pop(session_id, None)


def list_sessions() -> list[str]:
    with _LOCK:
        return list(_SESSIONS.keys())


def gc_sessions(ttl_seconds: int = 30 * 60) -> int:
    """清理超过 ttl 未活动的 session，返回回收数量。"""
    now = time.time()
    removed = 0
    with _LOCK:
        for sid in list(_SESSIONS.keys()):
            if now - _LAST_ACTIVE.get(sid, now) > ttl_seconds:
                _SESSIONS.pop(sid, None)
                _TURN_COUNT.pop(sid, None)
                _LAST_ACTIVE.pop(sid, None)
                removed += 1
    if removed:
        logger.info("[session_chat] 已回收 %d 个空闲 session", removed)
    return removed


def _drain_queue(q: queue.Queue) -> None:
    """deep_search_rag 通过 result_queue 流式输出，非交互场景下读空丢弃即可。"""
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        return


def _safe_last_references(history: list) -> list[dict]:
    """从 history 末尾取最近一轮的引用列表（list[ReferenceItem]）。"""
    if not history:
        return []
    last_attempts = history[-1]
    if not last_attempts:
        return []
    last_qr = last_attempts[-1]
    refs_field = last_qr.get("ref") if isinstance(last_qr, dict) else None
    if not refs_field:
        return []
    last_ref_list = refs_field[-1]
    if not isinstance(last_ref_list, list):
        return []
    return list(last_ref_list)


def chat(
    session_id: str | None,
    query: str,
    *,
    reset: bool = False,
    mode: str | None = None,
    config: Any | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    """
    会话式入口。

    Args:
        session_id: 不传则自动生成；同一组追问应复用同一个 session_id。
        query: 本轮 user 输入（首问或追问皆可）。
        reset: True 时清空该 session 历史，等价于网页点「新对话」。
        mode: 'deep_thinking'（默认，走知识库 RAG）/ 'chat'（不查库）/ 'deep_research'（多跳深搜）。
              若不传，读环境变量 DEEP_CHAT_DEFAULT_MODE，默认 'deep_thinking'。
        config: 直接传入的 AppConfig；不传则自动加载 yaml。
        config_path: 仅当 config 为 None 时生效。

    Returns:
        {
          "session_id": str,
          "turn_index": int,         # 本轮在该 session 中的序号（0 起）
          "answer": str,             # 已剥离引用块的正文
          "references": [ ... ],     # 本轮命中的 ReferenceItem 列表
          "mode": str,
          "history_size": int,       # 该 session 已累计的 turn 数
        }
    """
    from deep_research.agent_deep_search import deep_search_rag

    sid = session_id or new_session_id()

    if reset:
        reset_session(sid)

    actual_mode = (
        mode
        or os.environ.get("DEEP_CHAT_DEFAULT_MODE", "").strip()
        or "deep_thinking"
    )
    cfg = config if config is not None else get_config(config_path)
    _apply_model_runtime_overrides(cfg)
    _apply_deep_web_override(cfg)
    wc = getattr(cfg, "web", None)
    if wc is not None:
        logger.info("[session_chat] 本轮 deep 联网检索 web.enabled=%s", wc.enabled)

    with _LOCK:
        history = _SESSIONS.setdefault(sid, [])
        turn_index = _TURN_COUNT.get(sid, 0)

    q: queue.Queue = queue.Queue()
    try:
        answer = deep_search_rag(
            query,
            actual_mode,
            q,
            history,
            cfg,
        )
    except Exception as exc:
        logger.exception("[session_chat] deep_search_rag 调用失败: %s", exc)
        _drain_queue(q)
        return {
            "session_id": sid,
            "turn_index": turn_index,
            "answer": "",
            "references": [],
            "mode": actual_mode,
            "history_size": len(history),
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        _drain_queue(q)

    references = _safe_last_references(history)

    with _LOCK:
        _TURN_COUNT[sid] = turn_index + 1
        _LAST_ACTIVE[sid] = time.time()

    return {
        "session_id": sid,
        "turn_index": turn_index,
        "answer": answer or "",
        "references": references,
        "mode": actual_mode,
        "history_size": len(history),
    }


__all__ = [
    "chat",
    "new_session_id",
    "reset_session",
    "list_sessions",
    "gc_sessions",
    "get_config",
]


if __name__ == "__main__":
    # 仅作手动冒烟：问题由环境变量提供，避免在代码里写死演示文案。
    # 示例：SESSION_CHAT_Q1="..." SESSION_CHAT_Q2="..." python -m deep_research.session_chat
    logging.basicConfig(level=logging.INFO)
    q1 = os.environ.get("SESSION_CHAT_Q1", "").strip()
    q2 = os.environ.get("SESSION_CHAT_Q2", "").strip()
    if not q1:
        print(
            "请设置 SESSION_CHAT_Q1（首轮问题）。可选 SESSION_CHAT_Q2（同一会话追问）。",
            file=sys.stderr,
        )
        raise SystemExit(2)
    sid = new_session_id()
    print("session:", sid)
    r1 = chat(sid, q1, reset=True)
    print("turn0 answer (prefix):", (r1.get("answer") or "")[:120])
    print("turn0 references:", len(r1.get("references") or []))
    if q2:
        r2 = chat(sid, q2)
        print("turn1 answer (prefix):", (r2.get("answer") or "")[:120])
        print("turn1 references:", len(r2.get("references") or []))
