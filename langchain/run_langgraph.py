#!/usr/bin/env python3
# // cspell:disable
"""
LangGraph + deep_research 一键运行入口。

单条（交互或非交互）::
  python langchain/run_langgraph.py

批量（每行 JSON 的 ``input`` 作为首轮 query）示例（在 deep_research_dev 目录下执行）::

  python langchain/run_langgraph.py --jsonl ../../data/QA/test_question.jsonl --limit 10 --train-output-path langchain/agent_training_qwen2.jsonl

"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from argparse import Namespace
from typing import Any

# 本脚本在 langchain/ 下；deep_research_dev 为其上一级
_LANGCHAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_LANGCHAIN_DIR)

# 延迟导入 deep_research.paths（apply_run_config 会把 _REPO_ROOT 加入 sys.path）
def _project_path(rel: str) -> str:
    from deep_research.paths import resolve_project_path

    return str(resolve_project_path(rel))

# =============================
# 本次运行的统一参数入口（单条 / 批量共用）
# =============================
# 留空则单机模式下可在终端 `input()` 读题（见 langchain_test.prompt_legal_query_if_needed）。
LEGAL_QUERY = ""

# assistant 回答轮数：3 = 首问 + 2 次追问回答。
MAX_TURNS = 1

# 是否调用评测模型并参与 judge（False 时跳过 evaluator LLM，不产生 success_end）。
ENABLE_EVALUATOR = False

# 联网开关：
#   "yaml" — 沿用 deep_research/config.yaml 里的 web.enabled
#   True/False 或 "true"/"false" — 本次运行强制开/关，不改 yaml 文件
WEB_ENABLED = "True"

# LangGraph 三个角色模型。若三者共用一个服务，只改这里即可。
ASSISTANT_LLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
USER_SIMULATOR_LLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
EVALUATOR_LLM_URL = "http://127.0.0.1:8000/v1/chat/completions"

ASSISTANT_MODEL = "Qwen3.6-27B"
USER_SIMULATOR_MODEL = "Qwen3.6-27B"
EVALUATOR_MODEL = "Qwen3.6-27B"

# deep_research 内部多个模型节点默认仍读 config.yaml；填下面两项时会运行时覆盖
DEEP_RESEARCH_LLM_URL = ASSISTANT_LLM_URL
DEEP_RESEARCH_MODEL = ASSISTANT_MODEL

# 相对 deep_research_dev 项目根（../../data 指向 haoduo/data）
DEFAULT_QUESTIONS_JSONL = "../../data/QA/legal_question.jsonl"
DEFAULT_TRAIN_DATA_RELPATH = "langchain/agent_data.jsonl"
DEFAULT_TRAIN_DATA_FILENAME = "agent_data.jsonl"


def _run_config_dict() -> dict[str, str]:
    return {
        # ---- 业务 ----
        "LEGAL_QUERY": LEGAL_QUERY,
        # ---- LangGraph 行为（与 langchain_test 中环境变量一致）----
        "ASSISTANT_BACKEND": "deep",  # deep | vllm
        "TERMINATION_MODE": "rounds",  # judge | rounds
        "MAX_STEPS": str(MAX_TURNS),
        "ENABLE_EVALUATOR": "true" if ENABLE_EVALUATOR else "false",
        # ---- LangGraph 三个角色模型 ----
        "ASSISTANT_LLM_URL": ASSISTANT_LLM_URL,
        "USER_SIMULATOR_LLM_URL": USER_SIMULATOR_LLM_URL,
        "EVALUATOR_LLM_URL": EVALUATOR_LLM_URL,
        "ASSISTANT_MODEL": ASSISTANT_MODEL,
        "USER_SIMULATOR_MODEL": USER_SIMULATOR_MODEL,
        "EVALUATOR_MODEL": EVALUATOR_MODEL,
        # ---- deep_research ----
        "DEEP_CHAT_DEFAULT_MODE": "deep_thinking",
        "DEEP_RESEARCH_CONFIG": "deep_research/config.yaml",
        "DEEP_RESEARCH_LLM_URL": DEEP_RESEARCH_LLM_URL,
        "DEEP_RESEARCH_MODEL": DEEP_RESEARCH_MODEL,
        # ---- 落盘（相对 deep_research_dev；invoke 失败也会追加 InvokeFailed）----
        "TRAIN_DATA_PATH": DEFAULT_TRAIN_DATA_RELPATH,
        "LANGCHAIN_RUN_LOG": "langchain/agent_run.log",
        # ---- evaluator（同 langchain_test）----
        "EVALUATOR_MAX_ANSWER_CHARS": "10000",
        "EVALUATOR_MAX_TOKENS": "4096",
        "USER_SIMULATOR_MAX_TOKENS": "4096",
    }


def ensure_train_data_file_exists() -> None:
    """若 ``TRAIN_DATA_PATH`` 未指向已有文件，则创建父目录并写入空文件。"""
    raw = os.environ.get("TRAIN_DATA_PATH")
    if not raw:
        return
    path = _project_path(raw)
    os.environ["TRAIN_DATA_PATH"] = path
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not os.path.isfile(path):
        with open(path, "w", encoding="utf-8"):
            pass


def apply_run_config(overrides: dict[str, str] | None = None) -> None:
    """写入 os.environ 并把仓库根加入 sys.path，供 import deep_research。"""
    for key, value in _run_config_dict().items():
        os.environ[key] = str(value)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                os.environ[key] = str(value)

    wm = str(WEB_ENABLED if WEB_ENABLED is not None else "yaml").strip().lower()
    if wm in ("yaml", "config", "sync", ""):
        os.environ.pop("DEEP_WEB_ENABLED", None)
    elif wm in ("on", "true", "1", "yes"):
        os.environ["DEEP_WEB_ENABLED"] = "true"
    elif wm in ("off", "false", "0", "no"):
        os.environ["DEEP_WEB_ENABLED"] = "false"
    else:
        raise ValueError(
            f"WEB_ENABLED={WEB_ENABLED!r} 非法，应为 yaml/true/false 等"
        )
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    # 统一把环境变量中的相对路径解析为基于 deep_research_dev 的绝对路径
    for env_key in ("DEEP_RESEARCH_CONFIG", "TRAIN_DATA_PATH", "LANGCHAIN_RUN_LOG"):
        val = os.environ.get(env_key, "").strip()
        if val and not os.path.isabs(val):
            os.environ[env_key] = _project_path(val)


def _parse_args(argv: list[str] | None) -> Namespace:
    p = argparse.ArgumentParser(
        description="LangGraph 法律问答：单机交互/非交互，或 --jsonl 批量（input 字段）",
    )
    p.add_argument(
        "--jsonl",
        metavar="PATH",
        nargs="?",
        const=DEFAULT_QUESTIONS_JSONL,
        default=None,
        help=(
            "批量：法律问题 JSONL，每行 JSON 含字段 input；"
            f"仅写 --jsonl 不带路径时默认 {DEFAULT_QUESTIONS_JSONL}"
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="批量：在 --skip 之后最多跑多少条（强烈建议设限）",
    )
    p.add_argument(
        "--skip",
        type=int,
        default=0,
        help="批量：跳过 id 筛选后的前 N 条有效题目",
    )
    p.add_argument("--min-id", type=int, dest="min_id", default=None)
    p.add_argument("--max-id", type=int, dest="max_id", default=None)
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument(
        "--quiet",
        action="store_true",
        help="批量：单条不打印 run_once 终端摘要",
    )
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--max-turns",
        type=int,
        dest="max_turns",
        default=None,
        help="覆盖本文件顶部的 MAX_TURNS（写入 MAX_STEPS，单条/批量均生效）",
    )
    p.add_argument(
        "--legal-query",
        default=None,
        metavar="TEXT",
        help="单机：直接指定首轮问题，跳过终端 input（仍可用顶部 LEGAL_QUERY）",
    )
    ev = p.add_mutually_exclusive_group()
    ev.add_argument(
        "--enable-evaluator",
        dest="evaluator_toggle",
        action="store_const",
        const=True,
        default=None,
        help="开启评测模型与 judge（覆盖本文件顶部 ENABLE_EVALUATOR）",
    )
    ev.add_argument(
        "--disable-evaluator",
        dest="evaluator_toggle",
        action="store_const",
        const=False,
        default=None,
        help="关闭评测模型与 judge；仅占位写入 turn_logs，路由仍遵守 TERMINATION_MODE",
    )
    train_out = p.add_mutually_exclusive_group()
    train_out.add_argument(
        "--train-output-dir",
        metavar="DIR",
        default=None,
        help=(
            f"训练 JSONL 保存目录（固定文件名为 {DEFAULT_TRAIN_DATA_FILENAME}）；"
            "与 --train-output-path 二选一。"
        ),
    )
    train_out.add_argument(
        "--train-output-path",
        metavar="PATH",
        default=None,
        help=(
            "训练 JSONL 完整路径（可自定义文件名）；与 --train-output-dir 二选一。"
        ),
    )
    return p.parse_args(argv)


def _want_row(qid: object, args: Namespace) -> bool:
    if args.min_id is None and args.max_id is None:
        return True
    if qid is None:
        return False
    try:
        n = int(qid)
    except (TypeError, ValueError):
        return False
    if args.min_id is not None and n < args.min_id:
        return False
    if args.max_id is not None and n > args.max_id:
        return False
    return True


def _run_jsonl_batch(lt: Any, args: Namespace) -> None:
    log = logging.getLogger("run_langgraph.batch")
    jsonl_path = (
        args.jsonl
        if os.path.isabs(args.jsonl)
        else _project_path(args.jsonl)
    )
    if not os.path.isfile(jsonl_path):
        raise SystemExit(f"找不到 JSONL: {jsonl_path}")

    processed = 0
    scanned_lines = 0
    skipped_empty = 0
    skipped_id = 0
    skipped_offset = 0
    decode_errors = 0
    run_errors = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            scanned_lines += 1
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                decode_errors += 1
                log.error("第 %d 行 JSON 解析失败: %s", line_no, exc)
                if args.fail_fast:
                    raise
                continue

            text = row.get("input")
            if not isinstance(text, str) or not text.strip():
                skipped_empty += 1
                continue

            qid = row.get("id")
            if not _want_row(qid, args):
                skipped_id += 1
                continue

            if skipped_offset < args.skip:
                skipped_offset += 1
                continue

            if args.limit is not None and processed >= args.limit:
                log.info(
                    "已达 --limit=%d，停止读取（文件行号≈ %d）。",
                    args.limit,
                    line_no,
                )
                break

            if args.dry_run:
                processed += 1
                pe = max(1, args.progress_every)
                if processed % pe == 0:
                    log.info("dry-run 进度: %d 条", processed)
                continue

            source_meta = {"id": qid, "input": text.strip()}
            try:
                lt.run_once(
                    initial_query=text.strip(),
                    source_meta=source_meta,
                    verbose=not args.quiet,
                )
            except Exception:
                run_errors += 1
                log.exception("run_once 失败 id=%r line=%d", qid, line_no)
                if args.fail_fast:
                    raise

            processed += 1
            pe = max(1, args.progress_every)
            if processed % pe == 0:
                log.info(
                    "进度: 已完成 %d 条（累计扫描文件行=%d）",
                    processed,
                    scanned_lines,
                )

    log.info(
        "批量结束 processed=%s dry_run=%s 扫描行=%d 空input=%d id过滤=%d "
        "skip=%d json错误=%d run失败=%d",
        processed,
        args.dry_run,
        scanned_lines,
        skipped_empty,
        skipped_id,
        skipped_offset,
        decode_errors,
        run_errors,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    overrides: dict[str, str] = {}
    if args.max_turns is not None:
        overrides["MAX_STEPS"] = str(args.max_turns)
    if args.legal_query is not None:
        overrides["LEGAL_QUERY"] = args.legal_query.strip()
    if getattr(args, "evaluator_toggle", None) is not None:
        overrides["ENABLE_EVALUATOR"] = "true" if args.evaluator_toggle else "false"
    if getattr(args, "train_output_path", None):
        p = args.train_output_path.strip()
        overrides["TRAIN_DATA_PATH"] = p
    elif getattr(args, "train_output_dir", None):
        d = args.train_output_dir.strip()
        overrides["TRAIN_DATA_PATH"] = (
            os.path.join(d, DEFAULT_TRAIN_DATA_FILENAME)
            if os.path.isabs(d)
            else f"{d.rstrip('/')}/{DEFAULT_TRAIN_DATA_FILENAME}"
        )

    apply_run_config(overrides or None)
    ensure_train_data_file_exists()

    if _LANGCHAIN_DIR not in sys.path:
        sys.path.insert(0, _LANGCHAIN_DIR)

    # 必须在设置环境变量之后再导入（langchain_test 在 import 时读取 MAX_STEPS 等）
    import langchain_test as lt

    if args.jsonl:
        logging.basicConfig(
            level=os.environ.get("LANGGRAPH_BATCH_LOG_LEVEL", "INFO").upper(),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        logging.getLogger("run_langgraph").info(
            "批量模式 MAX_STEPS=%s ENABLE_EVALUATOR=%s TRAIN_DATA_PATH=%s",
            os.environ.get("MAX_STEPS"),
            os.environ.get("ENABLE_EVALUATOR"),
            os.environ.get("TRAIN_DATA_PATH"),
        )
        _run_jsonl_batch(lt, args)
        return

    if (
        args.limit is not None
        or args.skip
        or args.dry_run
        or args.min_id is not None
        or args.max_id is not None
    ):
        print(
            "提示: --limit / --skip / --dry-run / --min-id / --max-id 等仅在 --jsonl 批量模式下有效。",
            file=sys.stderr,
        )

    lt.prompt_legal_query_if_needed()
    lt.run_once()


if __name__ == "__main__":
    main()
