# -*- coding: utf-8 -*-
"""deep_research_dev 项目根目录下的相对路径解析（与运行时 cwd 无关）。"""
from __future__ import annotations

from pathlib import Path

# deep_research/ 包目录
PACKAGE_DIR = Path(__file__).resolve().parent
# deep_research_dev/
PROJECT_ROOT = PACKAGE_DIR.parent
LANGCHAIN_DIR = PROJECT_ROOT / "langchain"

# 常用相对路径（均相对 PROJECT_ROOT）
REL_CONFIG = "deep_research/config.yaml"
REL_LANGCHAIN_AGENT_DATA = "langchain/agent_data.jsonl"
REL_LANGCHAIN_RUN_LOG = "langchain/agent_run.log"
REL_DATA_QA_LEGAL_QUESTIONS = "../../data/QA/legal_question.jsonl"


def resolve_project_path(path: str | Path) -> Path:
    """将相对路径解析为基于 deep_research_dev 的绝对路径；已是绝对路径则原样 resolve。"""
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def default_config_path() -> str:
    return str(resolve_project_path(REL_CONFIG))


def ensure_on_sys_path() -> None:
    """保证 ``import deep_research`` 可用（把 deep_research_dev 加入 sys.path）。"""
    import sys

    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
