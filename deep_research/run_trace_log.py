# -*- coding: utf-8 -*-
"""记录每次问题的 RAG 全流程产出与阶段耗时。"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_RUN_LOG_PATH: Path | None = None

DEFAULT_RUN_LOG_PATH = Path(__file__).resolve().parent / "run.log"
_MAX_TEXT = 15000
_MAX_REF_TEXT = 1200


def _truncate(text: str, max_len: int = _MAX_TEXT) -> str:
    text = text or ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... [已截断]"


def _dump(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _truncate(value)
    try:
        return _truncate(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    except TypeError:
        return _truncate(repr(value))


def init_run_log(log_path: str | Path | None = None) -> Path:
    """服务每次启动时重置 run.log。"""
    global _RUN_LOG_PATH
    path = Path(log_path) if log_path else DEFAULT_RUN_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{'=' * 80}\n"
        f"会话启动 | {datetime.now().isoformat(timespec='seconds')}\n"
        f"PID: {os.getpid()}\n"
        f"{'=' * 80}\n\n",
        encoding="utf-8",
    )
    _RUN_LOG_PATH = path
    return path


def append_run_log(text: str) -> None:
    path = _RUN_LOG_PATH or DEFAULT_RUN_LOG_PATH
    with _LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n")


def format_recall_preview(kb_results: list[dict[str, Any]] | None, max_items: int = 8) -> str:
    if not kb_results:
        return "（无召回）"
    lines: list[str] = []
    for idx, item in enumerate(kb_results[:max_items], 1):
        law = (item.get("law_name") or item.get("category") or "").strip()
        article = (item.get("article") or item.get("number") or "").strip()
        score = item.get("score")
        title = f"[{idx}] {law} {article}".strip()
        if isinstance(score, (int, float)):
            title += f" (score={round(float(score), 4)})"
        body = (item.get("text") or "").strip()
        lines.append(f"{title}\n{_truncate(body, _MAX_REF_TEXT)}")
    if len(kb_results) > max_items:
        lines.append(f"... 共 {len(kb_results)} 条，以上展示前 {max_items} 条")
    return "\n\n".join(lines)


@dataclass
class QaTrace:
    """单次提问的阶段日志。"""

    mode: str
    user_query: str
    _start: float = field(default_factory=time.perf_counter)
    _last: float = field(init=False)
    lines: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._last = self._start
        self.lines.extend(
            [
                "=" * 80,
                f"时间: {datetime.now().isoformat(timespec='seconds')}",
                f"模式: {self.mode}",
                f"用户问题: {self.user_query}",
            ]
        )

    def stage(self, name: str, *, duration_s: float | None = None, **fields: Any) -> None:
        now = time.perf_counter()
        elapsed = duration_s if duration_s is not None else now - self._last
        self._last = now
        self.lines.append(
            f"--- 阶段: {name} | 本段 {elapsed:.3f}s | 累计 {now - self._start:.3f}s ---"
        )
        for key, value in fields.items():
            if value is None or value == "":
                continue
            self.lines.append(f"{key}:\n{_dump(value)}")

    def flush(self, final_answer: str = "", *, error: str | None = None) -> None:
        total = time.perf_counter() - self._start
        self.lines.append(f"--- 总耗时 {total:.3f}s ---")
        if error:
            self.lines.append(f"错误: {error}")
        else:
            self.lines.append("--- 最终回答 ---")
            self.lines.append(_truncate(final_answer))
        append_run_log("\n".join(self.lines) + "\n")
