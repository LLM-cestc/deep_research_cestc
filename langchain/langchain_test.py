# // cspell:disable
import json
import logging
import os
import re
import sys
import uuid
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END

from request_model import call_role_model, call_deep_chat, get_role_model_config

_LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_MODULE_DIR)
_DEFAULT_LANGCHAIN_RUN_LOG = "langchain/agent_run.log"


def _resolve_project_path(path: str) -> str:
    """相对路径按 deep_research_dev 项目根解析。"""
    if os.path.isabs(path):
        return path
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    try:
        from deep_research.paths import resolve_project_path

        return str(resolve_project_path(path))
    except ImportError:
        return os.path.abspath(os.path.join(_REPO_ROOT, path))


class _FlushFileHandler(logging.FileHandler):
    """每条日志后立即落盘，便于 tail -f 实时查看。"""

    def emit(self, record):
        super().emit(record)
        self.flush()


def attach_langchain_run_file_logging(log_path: str | None = None) -> str:
    """
    在已有控制台 logging 基础上，追加写入文件（实时 flush）。

    环境变量 LANGCHAIN_RUN_LOG：文件路径；设为空字符串则关闭文件日志。
    """
    path = log_path if log_path is not None else os.environ.get(
        "LANGCHAIN_RUN_LOG", _DEFAULT_LANGCHAIN_RUN_LOG
    )
    path = (path or "").strip()
    if not path:
        return ""

    abs_path = _resolve_project_path(path)
    parent = os.path.dirname(abs_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format=_LOG_FMT)

    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            bf = getattr(h, "baseFilename", None)
            if bf and os.path.abspath(bf) == abs_path:
                return abs_path

    fh = _FlushFileHandler(abs_path, encoding="utf-8", mode="a")
    fh.setFormatter(logging.Formatter(_LOG_FMT))
    root.addHandler(fh)
    return abs_path


if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT)

logger = logging.getLogger("AgentDataEngineering")


def prompt_legal_query_if_needed() -> None:
    """
    若环境变量 LEGAL_QUERY 为空，则尝试录入首轮法律问题。

    常见「没出现输入框就直接跑」原因：
    1. 运行的入口是 `python langchain_test.py` 而旧版没有在 __main__ 里调本函数；
    2. Shell / IDE 已在环境中 export 了 LEGAL_QUERY（含 run_langgraph 顶部写死的常量）；
    3. stdin 非交互式终端（IDE 点 Run），无法 `input()`——请用环境变量或 LEGAL_QUERY_FILE。
    """
    q = os.environ.get("LEGAL_QUERY", "").strip()
    if q:
        logger.info("LEGAL_QUERY 已存在（跳过终端输入），字符数=%d", len(q))
        return

    qfile = (os.environ.get("LEGAL_QUERY_FILE") or "").strip()
    if qfile:
        with open(qfile, "r", encoding="utf-8") as f:
            q = f.read().strip()
        if not q:
            raise SystemExit(f"LEGAL_QUERY_FILE 内容为空: {qfile}")
        os.environ["LEGAL_QUERY"] = q
        logger.info("LEGAL_QUERY 已从 LEGAL_QUERY_FILE 读取: %s", qfile)
        return

    fallback = (os.environ.get("LEGAL_QUERY_FALLBACK") or "").strip()
    if fallback:
        os.environ["LEGAL_QUERY"] = fallback
        logger.info("LEGAL_QUERY 已使用 LEGAL_QUERY_FALLBACK")
        return

    if not sys.stdin.isatty():
        raise SystemExit(
            "LEGAL_QUERY 未设置，且当前 stdin 不是交互式终端（常见于 IDE 直接 Run），"
            "因此无法使用 input() 读题。\n"
            "任选其一：\n"
            "  export LEGAL_QUERY='你的法律问题'\n"
            "  或 export LEGAL_QUERY_FILE=/path/to/question.txt\n"
            "  或 export LEGAL_QUERY_FALLBACK='默认题目'\n"
            "  或在终端执行: python run_langgraph.py\n"
            "（若在 run_langgraph.py 顶部把 LEGAL_QUERY 写成非空字符串，也会直接使用而不再提问。）"
        )

    print(
        "输入第一轮 legal query：",
        flush=True,
    )
    query = input("> ").strip()
    if not query:
        raise SystemExit("legal query 不能为空")
    os.environ["LEGAL_QUERY"] = query
    logger.info("LEGAL_QUERY 已通过终端 stdin 录入，字符数=%d", len(query))


# 设置最大 assistant 回答轮数：
#   3 = 首问回答 + 追问1回答 + 追问2回答
MAX_STEPS = int(os.environ.get("MAX_STEPS", "3"))
if MAX_STEPS < 1:
    logger.warning("MAX_STEPS=%s 无效，已回退为 1", MAX_STEPS)
    MAX_STEPS = 1

# assistant 后端选择：
#   vllm — 直接调本地 LLM（旧行为）
#   deep  — 走 deep_research 会话式入口（多轮历史 + RAG/Web 检索都由 deep 内部完成）
ASSISTANT_BACKEND = os.environ.get("ASSISTANT_BACKEND", "deep").strip().lower()
DEEP_CHAT_MODE = os.environ.get("DEEP_CHAT_DEFAULT_MODE", "deep_thinking").strip()

# 终止策略：
#   judge     — evaluator 判定通过后立即终止；MAX_STEPS 作为兜底上限
#   rounds — 不因 evaluator 通过提前终止，固定跑满 MAX_STEPS 轮 assistant 回答
TERMINATION_MODE = os.environ.get("TERMINATION_MODE", "rounds").strip().lower()
if TERMINATION_MODE not in {"judge", "rounds"}:
    logger.warning("未知 TERMINATION_MODE=%s，已回退为 rounds", TERMINATION_MODE)
    TERMINATION_MODE = "rounds"


def _parse_enable_evaluator_env() -> bool:
    """是否调用评测模型并执行 judge；环境变量 ENABLE_EVALUATOR，默认开启。"""
    raw = os.environ.get("ENABLE_EVALUATOR")
    if raw is None or str(raw).strip() == "":
        return True
    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    logger.warning("ENABLE_EVALUATOR=%r 无法识别，默认开启评测", raw)
    return True


ENABLE_EVALUATOR = _parse_enable_evaluator_env()

# evaluator：输入过长会导致服务端截断或空输出；可调字符上限与 max_tokens
EVALUATOR_MAX_ANSWER_CHARS = int(os.environ.get("EVALUATOR_MAX_ANSWER_CHARS", "10000"))
EVALUATOR_MAX_TOKENS = int(os.environ.get("EVALUATOR_MAX_TOKENS", "4096"))
if EVALUATOR_MAX_ANSWER_CHARS < 2000:
    EVALUATOR_MAX_ANSWER_CHARS = 2000
if EVALUATOR_MAX_TOKENS < 512:
    EVALUATOR_MAX_TOKENS = 512

USER_SIMULATOR_MAX_TOKENS = int(
    os.environ.get("USER_SIMULATOR_MAX_TOKENS", "4096"),
)
if USER_SIMULATOR_MAX_TOKENS < 256:
    USER_SIMULATOR_MAX_TOKENS = 256


_DEFAULT_SIMULATOR_FOLLOW_UP = (
    "你能不能结合上一轮回答和我的具体情况，再给一些可直接照着做的建议和注意事项？"
)

_SKIPPED_EVALUATOR_RESULT: Dict[str, Any] = {
    "evaluator_skipped": True,
    "pass": False,
    "score": None,
    "scores": {},
    "problems": [],
    "improvement_suggestions": [],
    "feedback_for_follow_up": (
        "（本轮未启用自动评测）请结合上一轮回答与用户原始问题，提出一个自然、具体的后续追问。"
    ),
    "serious_problem": False,
}


# =============================
class AgentState(TypedDict, total=False):
    query: str
    history: List[str]
    current_response: str
    is_judge_fine: bool
    feedback: str
    step_count: int
    final_status: str
    # —— 新增：deep_research 会话相关 ——
    assistant_backend: str            # 'vllm' / 'deep'
    deep_session_id: str              # deep 后端的稳定 session_id
    deep_mode: str                    # deep_thinking / chat / deep_research
    references_log: List[List[Any]]   # 每轮 generator 拿到的 references 列表
    last_references: List[Any]        # 最近一轮 references（便于落盘 + 调试）
    last_deep_metadata: Dict[str, Any] # 最近一轮 deep/session 元信息
    last_used_backend: str             # 最近一轮实际使用的后端（deep/vllm/vllm_fallback）
    current_eval_result: Dict[str, Any]
    turn_logs: List[Dict[str, Any]]    # 每轮 user/answer/evaluator/followup/references/deep 记录


# =============================
def _truncate_for_evaluator(answer: str, limit: int = EVALUATOR_MAX_ANSWER_CHARS) -> str:
    if not answer or len(answer) <= limit:
        return answer
    return (
        answer[:limit]
        + "\n\n[以上 answer 已截断用于评估，原文长度="
        + str(len(answer))
        + " 字符]"
    )


def _strip_thinking_tail_for_json(text: str) -> str:
    """若模型先输出长思考再输出 JSON，取最后一个思考结束标记之后的内容。"""
    if not text:
        return text
    for sep in (
        "</think>",
        "</thinking>",
        "<|im_end|>",
    ):
        if sep in text:
            text = text.split(sep)[-1]
    return text.strip()


def _strip_model_thinking_for_export(text: str) -> str:
    """导出训练 JSONL 时移除常见推理/思考段落，不落思考过程。"""
    if not text:
        return ""
    t = text
    t = re.sub(
        r"<think>[\s\S]*?</think>",
        "",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r"<thinking>[\s\S]*?</thinking>",
        "",
        t,
        flags=re.IGNORECASE,
    )
    return _strip_thinking_tail_for_json(t).strip()


def _strip_follow_up_outer_noise(text: str) -> str:
    """去掉首尾空白与常见推理标签外壳；不改变主体分段结构。"""
    t = (text or "").strip()
    if not t:
        return ""
    t = re.sub(
        r"<think>[\s\S]*?</think>",
        "",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r"<thinking>[\s\S]*?</thinking>",
        "",
        t,
        flags=re.IGNORECASE,
    )
    return _strip_thinking_tail_for_json(t).strip()


def _is_mostly_cn_block(s: str, *, min_cjk: int) -> bool:
    """一段文字是否可作为「用户追问」主体：以汉字为主。"""
    s = (s or "").strip()
    if len(s) < 6:
        return False
    cjk = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    latin = sum(1 for ch in s if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    if cjk < min_cjk:
        return False
    if latin > max(140, int(cjk * 2.5)):
        return False
    return True


def _gather_last_cn_lines(lines: List[str]) -> str | None:
    """
    从最后一行往上收：跳过纯英文推演行，直到遇到第一段「以中文为主体」的行块；
    再向上合并同属该追问的连贯短行。
    """
    acc_rev: List[str] = []
    started = False
    for raw_ln in reversed(lines):
        ln = raw_ln.strip()
        if not ln:
            if started:
                break
            continue
        cjk = sum(1 for ch in ln if "\u4e00" <= ch <= "\u9fff")
        latin = sum(1 for ch in ln if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
        low = ln.lower()
        if (
            cjk == 0
            and latin > 36
            and any(
                k in low
                for k in (
                    "thinking process",
                    "analyze",
                    "assistant's current response",
                    "evaluation feedback",
                    "historical dialogue",
                    "original question",
                    "task:",
                    "draft",
                )
            )
        ):
            if started:
                break
            continue
        if cjk >= 8 or (
            cjk >= 5 and ("？" in ln or "?" in ln or "：" in ln)
        ):
            acc_rev.append(raw_ln.rstrip())
            started = True
            continue
        if (
            started
            and latin < 30
            and cjk >= 4
            and ("，" in ln or "；" in ln or "、" in ln)
        ):
            acc_rev.append(raw_ln.rstrip())
            continue
        if started:
            break
    if not acc_rev:
        return None
    return "\n".join(reversed(acc_rev)).strip()


def extract_tagged_followup_question(raw: str) -> str | None:
    """
    prompt 约定的外壳：只对 `{追问} ... {/追问}` 中间的正文入库与传递；
    标签外的前缀/后缀（含英文推演）一律忽略。

    兼容：全角括号、标签内首尾空白；若缺少闭合 `{/追问}` 则截取到文末或首个疑似闭合行前。
    """
    t = (raw or "").replace("｛", "{").replace("｝", "}")
    paired = re.search(
        r"\{\s*追问\s*\}\s*(.*?)\s*\{\s*/\s*追问\s*\}",
        t,
        flags=re.DOTALL | re.UNICODE,
    )
    if paired:
        inner = paired.group(1).strip().strip('"').strip("“”")
        return inner if inner else None
    lone = re.search(r"\{\s*追问\s*\}\s*", t)
    if not lone:
        return None
    rest = t[lone.end() :]
    chop = re.split(r"\{\s*/\s*追问\s*\}", rest, maxsplit=1)[0].strip()
    return chop if chop else None


def extract_tagged_eval_json_block(raw: str) -> str | None:
    """
    评估模型约定：仅解析 `{评价} ... {/评价}` 中间的 JSON；
    外侧英文/思考过程不参与 parse。
    """
    t = (raw or "").replace("｛", "{").replace("｝", "}")
    paired = re.search(
        r"\{\s*评价\s*\}\s*(.*?)\s*\{\s*/\s*评价\s*\}",
        t,
        flags=re.DOTALL | re.UNICODE,
    )
    if paired:
        inner = paired.group(1).strip()
        return inner if inner else None
    lone = re.search(r"\{\s*评价\s*\}\s*", t)
    if not lone:
        return None
    rest = t[lone.end() :]
    chop = re.split(r"\{\s*/\s*评价\s*\}", rest, maxsplit=1)[0].strip()
    return chop if chop else None


def parse_evaluator_model_text(raw: str) -> Dict[str, Any] | None:
    """先取 `{评价}` 内 JSON，若无标签再对全文 parse_json_from_text（兼容旧输出）。"""
    body = (raw or "").strip()
    candidates: list[str] = []
    inner = extract_tagged_eval_json_block(body)
    if inner and inner.strip():
        candidates.append(inner.strip())
    if body:
        candidates.append(body)

    seen: set[str] = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        obj = parse_json_from_text(cand)
        if obj:
            return obj
    return None


def pick_terminal_chinese_block(raw: str) -> str | None:
    """
    仅从完整模型输出中取「末尾一段主要中文」，用于追问 / 下一轮 user，
    不保留前置英文推演过程。
    """
    text = _strip_follow_up_outer_noise(raw)
    if not text:
        return None

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n+", text.replace("\r\n", "\n"))
        if p.strip()
    ]
    for para in reversed(paragraphs):
        p = para.strip().strip('"').strip("“”")
        p = re.sub(r"^[#*_\-`\|\d\.\s]+", "", p).strip()
        if _is_mostly_cn_block(p, min_cjk=12):
            return p.replace("\u3000", " ").strip()
        if _is_mostly_cn_block(p, min_cjk=8) and ("？" in p or "?" in p):
            return p.replace("\u3000", " ").strip()

    gl = _gather_last_cn_lines(text.splitlines())
    if gl and _is_mostly_cn_block(gl, min_cjk=8):
        return gl.replace("\u3000", " ").strip()
    return None


_RE_FU_OPEN_LINE = re.compile(r"^\{\s*追问\s*\}\s*$")
_RE_FU_CLOSE_LINE = re.compile(r"^\{\s*/\s*追问\s*\}\s*$")


def _strip_inline_md_fence(s: str) -> str:
    """去掉行首行尾的反引号/引号包裹。"""
    t = (s or "").strip()
    t = re.sub(r"^[`]+", "", t)
    t = re.sub(r"[`]+$", "", t)
    t = t.strip("“”\"'")
    return t.strip()


def _line_is_simulator_meta_or_english(line: str) -> bool:
    """英文推演行 / 元信息行：不作为追问正文。"""
    s = (line or "").strip()
    if not s:
        return True
    low = s.lower()
    meta_substrings = (
        "draft:",
        "matches all",
        "matches exactly",
        "output matches",
        "output generation",
        "self-correction",
        "verification during",
        "check constraint",
        "final check",
        "self-correction/note",
        "[output",
        "*(self",
        "proceeds.",
        "ready.",
        "one minor tweak",
        "minor tweak",
        "perfect.",
        "all good.",
    )
    if any(k in low for k in meta_substrings):
        return True
    if s.lstrip().startswith("[") and "generation" in low:
        return True
    cjk = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    latin = sum(1 for ch in s if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    if cjk < 4 and latin > 20:
        return True
    if cjk < 8 and latin > 40:
        return True
    return False


def _passes_cn_follow_up_question_heuristic(s: str) -> bool:
    """单行是否为「以汉字为主、以问句收尾」的追问。"""
    t = _strip_inline_md_fence(s)
    if not t:
        return False
    if not ("？" in t or t.endswith("?")):
        return False
    cjk = sum(1 for ch in t if "\u4e00" <= ch <= "\u9fff")
    if cjk < 8:
        return False
    latin = sum(1 for ch in t if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    if latin > max(28, int(cjk * 1.15)):
        return False
    return True


def extract_followup_three_line_format(text: str) -> str | None:
    """
    与 prompt 对齐：非空行序列中出现连续三行
    `{追问}` / 追问正文 / `{/追问}` 时，取中间一行。
    标签外的说明、思考（含英文）一律不参与抽取。
    """
    t = (text or "").replace("\r\n", "\n").replace("｛", "{").replace("｝", "}")
    t = _strip_follow_up_outer_noise(t)
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    for i in range(len(lines) - 2):
        if not _RE_FU_OPEN_LINE.match(lines[i]):
            continue
        mid = _strip_inline_md_fence(lines[i + 1])
        if not _RE_FU_CLOSE_LINE.match(lines[i + 2]):
            continue
        if not mid or _line_is_simulator_meta_or_english(mid):
            continue
        return mid
    return None


def _normalize_tagged_followup_inner(inner: str) -> str | None:
    """
    仅处理 `{追问}`…`{/追问}` 之间的正文：多行时优先取第一条合格的中文问句行，
    跳过明显英文/元信息行，避免思考过程混入 history。
    """
    s = (inner or "").strip()
    if not s:
        return None
    parts = [p.strip() for p in s.replace("\r\n", "\n").split("\n") if p.strip()]
    if not parts:
        return None
    if len(parts) == 1:
        line = _strip_inline_md_fence(parts[0])
        if _line_is_simulator_meta_or_english(line):
            return None
        if _passes_cn_follow_up_question_heuristic(line):
            return line
        if _is_mostly_cn_block(line, min_cjk=6) and ("？" in line or line.endswith("?")):
            return line
        if _is_mostly_cn_block(line, min_cjk=10):
            return line
        return None
    for p in parts:
        if _line_is_simulator_meta_or_english(p):
            continue
        if _passes_cn_follow_up_question_heuristic(p):
            return _strip_inline_md_fence(p)
        if _is_mostly_cn_block(p, min_cjk=8) and ("？" in p or p.endswith("?")):
            return _strip_inline_md_fence(p)
    for p in parts:
        if _line_is_simulator_meta_or_english(p):
            continue
        cjk = sum(1 for ch in p if "\u4e00" <= ch <= "\u9fff")
        if cjk >= 10:
            return _strip_inline_md_fence(p)
    return None


def simulated_follow_up_for_langgraph(raw: str) -> str:
    """
    user_simulator 输出 → 进入对话框的追问一句。
    优先三行格式；否则仅用标签内正文（不用全文启发式抓取，防止思考/英文外泄）。
    """
    t = _strip_follow_up_outer_noise(raw or "").strip()
    q = extract_followup_three_line_format(t)
    if q:
        return q
    inner = extract_tagged_followup_question(t)
    if inner is not None:
        q2 = _normalize_tagged_followup_inner(inner)
        if q2:
            return q2
    tail = pick_terminal_chinese_block(t)
    if tail:
        q3 = _normalize_tagged_followup_inner(tail)
        if q3:
            return q3
        p = tail.strip()
        if "\n" not in p and _is_mostly_cn_block(p, min_cjk=8) and ("？" in p or p.endswith("?")):
            return p
    return _DEFAULT_SIMULATOR_FOLLOW_UP


def dialogue_user_text_for_training_jsonl(raw: str) -> str:
    """
    JSONL dialogue.user：与 simulated_follow_up 同一抽取规则；不向题干注入默认追问。
    """
    t = _strip_follow_up_outer_noise(raw or "").strip()
    q = extract_followup_three_line_format(t)
    if q:
        return q
    inner = extract_tagged_followup_question(t)
    if inner is not None:
        q2 = _normalize_tagged_followup_inner(inner)
        if q2:
            return q2
    tail = pick_terminal_chinese_block(t)
    if tail:
        q3 = _normalize_tagged_followup_inner(tail)
        if q3:
            return q3
        p = tail.strip()
        if "\n" not in p and _is_mostly_cn_block(p, min_cjk=6) and ("？" in p or p.endswith("?")):
            return p
    return ""


def _evaluation_for_training_export(ev: Any) -> Dict[str, Any]:
    """评测结构化字段：不写 raw_text 等大块原文，避免混入思考或未解析废话。"""
    if not isinstance(ev, dict):
        return {}
    out: Dict[str, Any] = {}
    skip = frozenset({"raw_text"})
    for k, v in ev.items():
        if k in skip:
            continue
        if isinstance(v, str):
            out[k] = _strip_model_thinking_for_export(v)
        else:
            out[k] = v
    return out


# =============================
# 2. 通用辅助函数
# =============================
def build_conversation_messages(state: AgentState) -> List[Dict[str, str]]:
    """
    根据 query + history 构造完整多轮 messages。

    history 数据结构约定：
    [
        回答1,
        追问1,
        回答2,
        追问2,
        ...
    ]

    因此偶数下标是 assistant，奇数下标是 user。
    """
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "你是一名严谨、专业、通俗易懂的法律问答助手。"
                "回答法律问题时需要做到："
                "1. 先直接回答结论；"
                "2. 再说明法律依据或裁判逻辑；"
                "3. 区分一般规则、例外情况和证据要求；"
                "4. 尽量用普通人能理解的语言解释；"
                "5. 不要编造不存在的法条、案例或事实；"
                "6. 如果信息不足，要明确说明需要补充哪些事实。"
            )
        },
        {"role": "user", "content": state["query"]}
    ]

    for i, msg in enumerate(state.get("history", [])):
        role = "assistant" if i % 2 == 0 else "user"
        messages.append({"role": role, "content": msg})

    return messages


def parse_json_from_text(text: str) -> Dict[str, Any] | None:
    """
    尽量从模型输出中解析 JSON。
    兼容：
    1. 纯 JSON
    2. ```json ... ```
    3. 前后带解释文字、先思考后 JSON（截取自首个 { 的 raw_decode）
    """
    if not text:
        return None

    text = text.strip()
    text = _strip_thinking_tail_for_json(text)

    cleaned = re.sub(r"^```(?:json)?\s*", "", text)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    decoder = json.JSONDecoder()
    for candidate in (cleaned, text):
        s = candidate.strip()
        if not s:
            continue
        start = s.find("{")
        if start < 0:
            continue
        try:
            obj, _ = decoder.raw_decode(s[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    return None


def has_serious_problem(problems: List[str]) -> bool:
    """
    判断 evaluator 指出的问题中是否包含严重法律质量问题。
    """
    serious_problem_keywords = [
        "编造",
        "错误",
        "严重",
        "不准确",
        "法条错误",
        "法律依据错误",
        "明显遗漏",
        "未回答",
        "相反",
        "矛盾",
        "误导",
        "核心问题未覆盖"
    ]

    return any(
        any(keyword in str(problem) for keyword in serious_problem_keywords)
        for problem in problems
    )


# =============================
# 3. LangGraph 节点逻辑
# =============================
def _current_user_question(state: AgentState) -> str:
    """
    取本轮 generator 应该回答的「用户当前问题」。

    history 约定 [回答1, 追问1, 回答2, 追问2, ...]：
    - 长度为 0 → 首问，使用 state['query']
    - 长度为偶数 → 末尾是追问，回答它
    - 长度为奇数 → 末尾是回答（理论上不会进 generator，兜底返回 state['query']）
    """
    hist = state.get("history", [])
    if not hist:
        return state.get("query", "")
    if len(hist) % 2 == 0:
        return hist[-1]
    return state.get("query", "")


def generator_node(state: AgentState):
    """
    法律回答节点。

    根据 ASSISTANT_BACKEND 切换两种回答路径：
    - vllm：直接调单轮 chat LLM，由 LangGraph 在 prompt 里拼历史；
    - deep ：调用 deep_research 会话入口（assistant_deep），由 deep 内部维护
            自己的多轮历史、做 query rewrite + RAG/Web 检索、再生成答案。

    无论走哪条路径，本节点输出的 current_response 都只是「答案正文」，
    供 evaluator 评估，不含引用块；引用单独存到 last_references / references_log。
    """
    backend = state.get("assistant_backend") or ASSISTANT_BACKEND
    step = state.get("step_count", 0)
    logger.info(
        "进入 generator_node，当前迭代次数: %d，backend=%s",
        step,
        backend,
    )

    references: List[Any] = []
    deep_session_id = state.get("deep_session_id", "")
    deep_mode = state.get("deep_mode") or DEEP_CHAT_MODE
    used_backend = backend
    deep_metadata: Dict[str, Any] = {
        "requested_backend": backend,
        "mode": deep_mode,
    }

    if backend == "deep":
        question = _current_user_question(state)
        if not deep_session_id:
            deep_session_id = f"lg-{uuid.uuid4().hex[:12]}"
            logger.info("未携带 deep_session_id，自动生成: %s", deep_session_id)

        result = call_deep_chat(
            session_id=deep_session_id,
            query=question,
            reset=(step == 0),  # 第一次进 generator 时清空 deep 端历史
            mode=deep_mode,
        )

        if result.get("error"):
            logger.warning(
                "deep_research 调用失败，回落到 vllm 路径：%s",
                result.get("error"),
            )
            used_backend = "vllm_fallback"
            deep_metadata.update(
                {
                    "session_id": deep_session_id,
                    "used_backend": used_backend,
                    "error": result.get("error"),
                    "turn_index": result.get("turn_index"),
                    "history_size": result.get("history_size"),
                }
            )
            messages = build_conversation_messages(state)
            content = call_role_model(
                role_name="assistant",
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
            )
        else:
            content = result.get("answer", "") or ""
            references = result.get("references", []) or []
            logger.info(
                "deep_research 返回 answer 长度=%d, references=%d, turn=%s",
                len(content),
                len(references),
                result.get("turn_index"),
            )
            deep_session_id = result.get("session_id", deep_session_id)
            deep_metadata.update(
                {
                    "session_id": deep_session_id,
                    "used_backend": used_backend,
                    "turn_index": result.get("turn_index"),
                    "history_size": result.get("history_size"),
                    "mode": result.get("mode", deep_mode),
                }
            )
    else:
        used_backend = "vllm"
        deep_metadata.update({"used_backend": used_backend})
        messages = build_conversation_messages(state)
        content = call_role_model(
            role_name="assistant",
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
        )

    if not content:
        content = "模型未返回有效响应"

    new_step = step + 1
    new_refs_log = list(state.get("references_log", []) or []) + [references]

    update: Dict[str, Any] = {
        "current_response": content,
        "step_count": new_step,
        "assistant_backend": backend,
        "deep_mode": deep_mode,
        "references_log": new_refs_log,
        "last_references": references,
        "last_deep_metadata": deep_metadata,
        "last_used_backend": used_backend,
    }
    if deep_session_id:
        update["deep_session_id"] = deep_session_id
    return update


def _append_turn_log(
    state: AgentState,
    eval_result: Dict[str, Any],
    *,
    judge_fine: bool,
    feedback: str,
) -> List[Dict[str, Any]]:
    """
    追加当前 assistant 回答轮的完整日志。

    注意：simulator 生成追问发生在 evaluator 之后，因此 followup 会在
    human_simulator_node 中回填到最后一条 turn log。
    """
    step_count = int(state.get("step_count", 0) or 0)
    turn_logs = list(state.get("turn_logs", []) or [])
    turn_logs.append(
        {
            "turn_index": max(step_count - 1, 0),
            "assistant_turn": step_count,
            "user_question": _current_user_question(state),
            "answer": state.get("current_response", ""),
            "evaluator": eval_result,
            "judge_fine": judge_fine,
            "feedback": feedback,
            "followup": None,
            "references": state.get("last_references", []) or [],
            "deep": state.get("last_deep_metadata", {}) or {},
            "used_backend": state.get(
                "last_used_backend",
                state.get("assistant_backend", ASSISTANT_BACKEND),
            ),
        }
    )
    return turn_logs


def evaluator_node(state: AgentState):
    """
    质量评估节点。

    使用角色：
    - evaluator

    作用：
    - 判断当前 assistant 回答是否适合作为高质量法律问答训练数据；
    - 输出结构化 JSON；
    - 决定是否继续追问（ENABLE_EVALUATOR=False 时不调模型，is_judge_fine 恒为 False）。
    """
    if not ENABLE_EVALUATOR:
        logger.info("ENABLE_EVALUATOR=关闭，跳过评测模型与 judge，仍写入占位 turn_logs")
        stub = dict(_SKIPPED_EVALUATOR_RESULT)
        feedback = json.dumps(
            {
                "evaluator_skipped": True,
                "feedback_for_follow_up": stub["feedback_for_follow_up"],
                "note": (
                    "未调用评测模型；不设 success_end。"
                    "judge 模式下亦不会因质量提前结束；"
                    "仅按 TERMINATION_MODE（rounds / MAX_STEPS）路由。"
                ),
            },
            ensure_ascii=False,
        )
        return {
            "is_judge_fine": False,
            "feedback": feedback,
            "current_eval_result": stub,
            "turn_logs": _append_turn_log(
                state,
                stub,
                judge_fine=False,
                feedback=feedback,
            ),
            "final_status": "Processing",
        }

    logger.info("进入 evaluator_node，开始评估回答质量")

    answer_excerpt = _truncate_for_evaluator(state.get("current_response") or "")

    eval_prompt = f"""你是法律问答训练数据的质量评估员。请评估 assistant 的回答是否适合作为高质量法律问答训练数据。

【原始用户问题】
{state['query']}

【历史对话】
{json.dumps(state.get('history', []), ensure_ascii=False, indent=2)}

【当前 assistant 回答】
{answer_excerpt}

请从以下维度评分，每项 0-5 分：
1. relevance：是否正面回答用户问题；
2. legal_accuracy：法律表述是否准确，是否避免编造法条、案例和事实；
3. completeness：是否覆盖结论、依据、适用条件、例外或风险；
4. clarity：是否通俗、清晰、结构化；
5. practicality：是否给出可操作的信息，如证据、步骤、注意事项；
6. context_awareness：是否结合上下文和追问，而不是孤立回答。

【通过标准】
- 总分 >= 28；
- legal_accuracy >= 5；
- relevance >= 4；
- 没有明显编造法律依据；
- 没有严重遗漏用户核心问题。

【输出要求】
你可以在 `{{评价}}` … `{{/评价}}` 之外写任意分析或英文思考，**程序会丢弃标签外全部内容**。
**只有** `{{评价}}` 与 `{{/评价}}` 之间必须是**一个合法 JSON 对象**（UTF-8）：键名英文，pass 为小写 true/false，score 与各项分数为整数；不要用 markdown 代码围栏。

示例（外层英文可丢弃）：
Here's my scratch reasoning in English...
{{评价}}
{{
  "pass": false,
  "score": 20,
  "scores": {{
    "relevance": 4,
    "legal_accuracy": 5,
    "completeness": 4,
    "clarity": 4,
    "practicality": 3,
    "context_awareness": 4
  }},
  "problems": ["问题1"],
  "improvement_suggestions": ["改进1"],
  "feedback_for_follow_up": "一句话指出最需要补强的点"
}}
{{/评价}}
""".strip()

    system_round1 = (
        "你是一个严格的法律问答质量评估模型。"
        "你的任务不是回答用户问题，而是评估 assistant 回答质量。"
        "必须客观、严格，尤其关注法律准确性、法条依据、适用条件、证据要求和是否编造。"
        "你可以在 `{评价}` 外任意书写推演（中英文均可）；"
        "只有 `{评价}` 与 `{/评价}` 中间的 JSON 会进入评测流水线，其余输出一律忽略。"
    )

    system_round2 = (
        "最终可见结果必须可被解析为 JSON：严格使用 `{评价}` 包裹 JSON，`{/评价}` 闭合；"
        "标签外可出现思考草稿，但不会参与解析。"
    )

    eval_text = ""
    eval_result: Dict[str, Any] | None = None

    for attempt in (0, 1):
        messages = [
            {
                "role": "system",
                "content": system_round1 if attempt == 0 else system_round2,
            },
            {"role": "user", "content": eval_prompt},
        ]

        eval_text = call_role_model(
            role_name="evaluator",
            messages=messages,
            temperature=0.1 if attempt == 0 else 0.0,
            max_tokens=EVALUATOR_MAX_TOKENS,
        )
        eval_result = parse_evaluator_model_text(eval_text)
        if eval_result:
            break
        logger.warning(
            "evaluator 第 %d 次调用未能解析 JSON，返回长度=%d 前缀=%r",
            attempt + 1,
            len(eval_text or ""),
            (eval_text or "")[:400],
        )

    if not eval_result:
        logger.warning("评估结果无法解析为 JSON，判定为不合格")
        fallback_eval_result = {
            "pass": False,
            "score": 0,
            "scores": {},
            "problems": ["评估结果无法解析为 JSON（模型返回空、非 JSON 或解析失败）"],
            "improvement_suggestions": [
                "请让评估模型输出采用 `{评价}` … `{/评价}` 包裹的 JSON（标签外可无脑思考）；"
                "检查 EVALUATOR 是否在标签内写出了合法 JSON。"
                "或调大 EVALUATOR_MAX_TOKENS / 缩小 EVALUATOR_MAX_ANSWER_CHARS。"
            ],
            "feedback_for_follow_up": "请结合法律依据与案情把结论写得更可核验。",
            "raw_text": eval_text or "",
            "serious_problem": True,
            "evaluator_retries": 2,
        }
        feedback = json.dumps(fallback_eval_result, ensure_ascii=False)
        return {
            "is_judge_fine": False,
            "feedback": feedback,
            "current_eval_result": fallback_eval_result,
            "turn_logs": _append_turn_log(
                state,
                fallback_eval_result,
                judge_fine=False,
                feedback=feedback,
            ),
            "final_status": "Processing",
        }

    score = int(eval_result.get("score", 0) or 0)
    scores = eval_result.get("scores", {}) or {}
    relevance = int(scores.get("relevance", 0) or 0)
    legal_accuracy = int(scores.get("legal_accuracy", 0) or 0)
    is_pass = bool(eval_result.get("pass", False))

    problems = eval_result.get("problems", []) or []
    suggestions = eval_result.get("improvement_suggestions", []) or []
    feedback_for_follow_up = eval_result.get("feedback_for_follow_up", "")

    serious_problem = has_serious_problem(problems)

    judge_fine = (
        is_pass
        and score >= 28
        and relevance >= 4
        and legal_accuracy >= 5
        and not serious_problem
    )

    feedback = json.dumps(
        {
            "score": score,
            "scores": scores,
            "problems": problems,
            "improvement_suggestions": suggestions,
            "feedback_for_follow_up": feedback_for_follow_up,
            "serious_problem": serious_problem,
        },
        ensure_ascii=False,
    )
    eval_log = dict(eval_result)
    eval_log.update(
        {
            "score": score,
            "scores": scores,
            "problems": problems,
            "improvement_suggestions": suggestions,
            "feedback_for_follow_up": feedback_for_follow_up,
            "serious_problem": serious_problem,
        }
    )

    if judge_fine:
        logger.info("评估结果: PASS，总分: %s", score)
        return {
            "is_judge_fine": True,
            "feedback": feedback,
            "current_eval_result": eval_log,
            "turn_logs": _append_turn_log(
                state,
                eval_log,
                judge_fine=True,
                feedback=feedback,
            ),
            "final_status": "Success",
        }

    logger.info("评估结果: 不合格，总分: %s", score)
    return {
        "is_judge_fine": False,
        "feedback": feedback,
        "current_eval_result": eval_log,
        "turn_logs": _append_turn_log(
            state,
            eval_log,
            judge_fine=False,
            feedback=feedback,
        ),
        "final_status": "Processing",
    }


def human_simulator_node(state: AgentState):
    """
    用户追问模拟节点。

    使用角色：
    - user_simulator

    作用：
    - 根据当前回答和 evaluator 反馈，生成一个真实用户可能继续提出的问题。
    """
    logger.info("进入 human_simulator_node，生成追问")

    simulate_prompt = f"""
【任务】
你扮演正在咨询的普通用户。根据下列材料，写出**接下来最可能追问的一句中文问话**，用来请你面前的「法律助手」把上一轮回答补强（更具体、更可操作即可）。不要替助手作答。

【材料】
· 原始问题：{state['query']}

· 历史对话（JSON 数组，偶数位为助手、奇数位为你之前的追问）：
{json.dumps(state.get('history', []), ensure_ascii=False, indent=2)}

· 助手本轮回答：
{state['current_response']}

· 对本轮回答的质量反馈（供你决定追问方向）：
{state['feedback']}

【输出格式（唯一合法；违反则整条无效）】
整条回复只能是下面 **3 行**，不得出现第 4 行及以后，不得在首尾多写任意字符（含空格、说明、英文、草稿）：
第1行，且只能是：{{追问}}
第2行，且只能是：**一句**简体中文追问（模拟真实用户自然发言）
第3行，且只能是：{{/追问}}

【追问写什么】
围绕上一回答里仍不清楚或你想核实的一点，或者对于问题实际情况的补充即可（如要件、证据、程序、后果、例外情形等），一个问题里可以包含用逗号连接的两个小问，但仍算作**同一行**里的一句长问。

【禁止】
不允许 JSON、列表、markdown 代码块、`<think>`；`{{追问}}` 前面和 `{{/追问}}` 后面禁止任何内容。
""".strip()

    messages = [
        {
            "role": "system",
            "content": (
                "你是普通咨询用户模拟器：只根据材料写出一句追问，不替法律助手回答。\n"
                "你必须严格遵守用户消息里的三行输出格式；不得在标签外输出任何文字。"
            ),
        },
        {"role": "user", "content": simulate_prompt},
    ]

    follow_up = call_role_model(
        role_name="user_simulator",
        messages=messages,
        temperature=0.9,
        max_tokens=USER_SIMULATOR_MAX_TOKENS,
    )

    if not follow_up:
        follow_up = _DEFAULT_SIMULATOR_FOLLOW_UP

    follow_up = re.sub(r"^\s*(追问|用户追问|问题)\s*[:：]\s*", "", follow_up).strip()
    follow_up = simulated_follow_up_for_langgraph(follow_up)
    turn_logs = list(state.get("turn_logs", []) or [])
    if turn_logs:
        last_turn = dict(turn_logs[-1])
        last_turn["followup"] = follow_up
        turn_logs[-1] = last_turn

    return {
        "history": state.get("history", []) + [
            state["current_response"],
            follow_up
        ],
        "turn_logs": turn_logs,
    }


def route_after_eval(state: AgentState) -> str:
    """
    根据评估结果和当前步数，决定下一步去哪。
    """
    if TERMINATION_MODE == "rounds":
        if state["step_count"] >= MAX_STEPS:
            logger.info(
                ">>> 路由决策：rounds 模式达到目标轮数 (%d)，结束流程。",
                MAX_STEPS,
            )
            return "max_steps_end"

        logger.info(
            ">>> 路由决策：rounds 模式继续生成追问 (%d/%d)%s。",
            state["step_count"],
            MAX_STEPS,
            "" if ENABLE_EVALUATOR else "（评测已跳过）",
        )
        return "retry"

    if state["is_judge_fine"]:
        logger.info(">>> 路由决策：质量达标，结束流程。")
        return "success_end"

    if state["step_count"] >= MAX_STEPS:
        logger.info(f">>> 路由决策：达到最大重试次数 ({MAX_STEPS})，强制结束。")
        return "max_steps_end"

    logger.info(
        ">>> 路由决策：质量不达标，继续追问%s。",
        "" if ENABLE_EVALUATOR else "（评测已跳过，仅按轮数上限循环）",
    )
    return "retry"


# =============================
# 4. 构建 LangGraph 工作流
# =============================
workflow = StateGraph[AgentState, None, AgentState, AgentState](AgentState)

workflow.add_node("generator", generator_node)
workflow.add_node("evaluator", evaluator_node)
workflow.add_node("simulator", human_simulator_node)

workflow.set_entry_point("generator")

workflow.add_edge("generator", "evaluator")

workflow.add_conditional_edges(
    "evaluator",
    route_after_eval,
    {
        "success_end": END,
        "max_steps_end": END,
        "retry": "simulator"
    }
)

workflow.add_edge("simulator", "generator")

app = workflow.compile()


# =============================
# 5. 运行与保存训练数据
# =============================

def run_once(
    initial_query: str | None = None,
    *,
    source_meta: Dict[str, Any] | None = None,
    verbose: bool = True,
) -> None:
    """
    执行一轮完整 LangGraph（可被 run_langgraph.py 或批处理脚本导入调用）。

    :param initial_query: 若给出非空字符串，作为本轮用户首问，覆盖环境变量 LEGAL_QUERY。
    :param source_meta: 写入训练 JSONL 顶层的 ``source`` 字段，便于追溯来源（如 id、原始 input）。
    :param verbose: 为 False 时不打印末段摘要（批跑时减少刷屏）。
    """
    _run_log_file = attach_langchain_run_file_logging()
    if _run_log_file:
        logger.info("运行日志实时写入: %s", _run_log_file)

    deep_session_id = f"lg-{uuid.uuid4().hex[:12]}"

    default_q = "请用通俗的语言解释什么是股东未实缴出资的法律责任？"
    if initial_query is not None and str(initial_query).strip():
        query_text = str(initial_query).strip()
    else:
        query_text = os.environ.get("LEGAL_QUERY", default_q)

    initial_state: AgentState = {
        "query": query_text,
        "history": [],
        "step_count": 0,
        "is_judge_fine": False,
        "current_response": "",
        "feedback": "",
        "final_status": "Processing",
        "assistant_backend": ASSISTANT_BACKEND,
        "deep_session_id": deep_session_id,
        "deep_mode": DEEP_CHAT_MODE,
        "references_log": [],
        "last_references": [],
        "last_deep_metadata": {},
        "last_used_backend": "",
        "current_eval_result": {},
        "turn_logs": [],
    }

    logger.info(
        "ASSISTANT_BACKEND=%s, deep_session_id=%s, deep_mode=%s, "
        "TERMINATION_MODE=%s, MAX_STEPS=%s, ENABLE_EVALUATOR=%s",
        ASSISTANT_BACKEND,
        deep_session_id,
        DEEP_CHAT_MODE,
        TERMINATION_MODE,
        MAX_STEPS,
        ENABLE_EVALUATOR,
    )
    logger.info("当前角色模型配置：")
    for role, cfg in get_role_model_config().items():
        if cfg.get("type") == "chat_completions":
            logger.info(f"  {role}: model={cfg['model']}, url={cfg['url']}")
        elif cfg.get("type") == "deep_research_session":
            logger.info(
                f"  {role}: deep_research session via {cfg.get('module')} "
                f"(default_mode={cfg.get('default_mode')})"
            )
        else:
            logger.info(f"  {role}: {cfg}")

    default_train_data_path = "langchain/agent_data.jsonl"
    train_data_path = _resolve_project_path(
        os.environ.get("TRAIN_DATA_PATH", default_train_data_path)
    )
    train_parent = os.path.dirname(train_data_path)
    if train_parent:
        os.makedirs(train_parent, exist_ok=True)

    try:
        final_state = app.invoke(AgentState(initial_state))
    except Exception as invoke_exc:
        logger.exception(
            "LangGraph invoke 失败，仍写入一条占位记录到 %s 后重新抛出",
            train_data_path,
        )
        fail_record: Dict[str, Any] = {
            "schema": "legal_agent_dialog_eval/v1",
            "final_status": "InvokeFailed",
            "error": f"{type(invoke_exc).__name__}: {invoke_exc}",
            "rounds": [],
        }
        if source_meta:
            fail_record["source"] = source_meta
        with open(train_data_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(fail_record, ensure_ascii=False) + "\n")
        raise

    rounds_export: List[Dict[str, Any]] = []
    for tl in final_state.get("turn_logs") or []:
        fu_raw = tl.get("followup")
        if fu_raw is None:
            follow_export = None
        else:
            s = str(fu_raw)
            boxed = extract_tagged_followup_question(s)
            if boxed:
                t2 = boxed.strip()
                follow_export = pick_terminal_chinese_block(t2) or (
                    t2 if _is_mostly_cn_block(t2, min_cjk=6) else None
                )
            else:
                cleaned_fu = pick_terminal_chinese_block(s)
                follow_export = cleaned_fu if cleaned_fu else None

        rounds_export.append(
            {
                "assistant_round": tl.get("assistant_turn"),
                "turn_index": tl.get("turn_index"),
                "dialogue": {
                    "user": dialogue_user_text_for_training_jsonl(
                        tl.get("user_question") or "",
                    ),
                    "assistant": _strip_model_thinking_for_export(
                        tl.get("answer") or ""
                    ),
                },
                "evaluator": _evaluation_for_training_export(
                    tl.get("evaluator") or {}
                ),
                "gate_pass_this_round": tl.get("judge_fine"),
                "simulator_follow_up": follow_export,
            }
        )

    training_record: Dict[str, Any] = {
        "schema": "legal_agent_dialog_eval/v1",
        "final_status": final_state.get("final_status", "Failed_Max_Steps"),
        "rounds": rounds_export,
    }
    if source_meta:
        training_record["source"] = source_meta

    with open(train_data_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(training_record, ensure_ascii=False) + "\n")

    if verbose:
        print("\n=== 最终执行结果 ===")
        print(f"迭代次数: {final_state['step_count']}")
        print(f"最终状态: {final_state.get('final_status', 'Failed_Max_Steps')}")
        print(f"最终回答: {final_state['current_response']}")
        print(f"对话+评测精简 JSONL: {train_data_path}")
    logger.info(
        "完整 turn 链路（含引用、后端元数据）见 agent_run.log；"
        "agent_data.jsonl（或 TRAIN_DATA_PATH）仅存对话与各轮评测。"
    )


if __name__ == "__main__":
    prompt_legal_query_if_needed()
    run_once()
