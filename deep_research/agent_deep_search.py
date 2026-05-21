# -*- coding: utf-8 -*-  # noqa: UP009
"""
deepresearch agent模块

Author: wjianxz
Date: 2025-11-13
"""
import ast
import json
import re
from typing import Any, Optional
import queue
import copy
import time
import threading

from deep_research.local_logger import Timer, model_request_error
from deep_research.parser_config import load_validated_config
from deep_research.utils import (
    dedupe_repeated_answer_body,
    send_request_to_model,
    send_request_to_model_streaming,
    check_llm_output,
    parser_session_output,
    get_ref_deep_speed,
)
from deep_research.prompts import (
    build_beautiful_format_noref_prompt,
    build_beautiful_format_prompt,
    build_beautiful_format_rag_prompt,
    build_deep_research_session_rag_prompt,
    build_deep_research_session_chat_prompt,
    build_retrieval_plan_prompt,
    build_session_prompt,
    build_deep_research_beautiful_format_prompt,
)
from deep_research.protocal import (
    HistoryMessage,
    QueryResult,
    ReferenceItem,
    ReferenceList,
    DeepResearchResult,
)
from deep_research.parser_config import AppConfig
from deep_research.retrieve_knowledge import retrieve_from_knowledge_base
from deep_research.validate_safety import validate_input_safety
from deep_research.plan_pipline_search import joint_history_message
from deep_research.local_logger import timing
from deep_research.format_answer import (
    message_deepsearch_thinking,
    format_knowledge_references,
    format_web_references,
)
from deep_research.plan_pipline_search import deep_research_agent, mutil_hop_search
from deep_research.run_trace_log import QaTrace, format_recall_preview
from deep_research.web_retrieval import (
    effective_user_question_for_web,
    retrieve_web_evidence,
    web_enabled,
)

import logging

logger = logging.getLogger(__name__)  # 自动继承 root logger 的 handlers

# session 规则前置：明显延续/新话题时跳过 session 判别 LLM
_SESSION_CONTINUE_RE = re.compile(
    r"(上述|前面|刚才|上次|此前|那|这个|那个|继续|接着|在此基础上|"
    r"重新回答|重写|压缩|精简|扩写|换.{0,4}格式|翻译|举例说明|"
    r"为什么|为何|怎么理解|什么意思|再说说|补充一下)",
    re.IGNORECASE,
)
_SESSION_NEW_TOPIC_RE = re.compile(
    r"(新问题|换个话题|另一件事|另外问|不问刚才|与上文无关)",
    re.IGNORECASE,
)


def _session_rule_decision(user_query: str) -> Optional[bool]:
    """
    会话延续性规则预判。
    Returns:
        True: 明确延续，应拼接历史
        False: 明确新话题，不拼接历史
        None: 灰区，需调用 session 判别模型
    """
    q = (user_query or "").strip()
    if not q:
        return None
    if _SESSION_NEW_TOPIC_RE.search(q):
        return False
    if _SESSION_CONTINUE_RE.search(q):
        return True
    return None


def _strip_generated_reference_section(text: str) -> str:
    """最终引用依据由系统统一拼接，避免模型自带引用块与真实召回不一致。"""
    return re.split(
        r"\n\s*#{1,3}\s*(?:引用依据|参考法律法规|参考依据)\s*\n",
        text or "",
        maxsplit=1,
    )[0].strip()


def _strip_meta_analysis(text: str) -> str:
    """去掉模型误泄露到最终答案里的元分析开头。"""
    s = (text or "").strip()
    patterns = [
        r"^让我分析一下.*?(?=\n\s*#{1,3}\s*|\n\s*[一二三四五六七八九十]+[、.．]|\n\s*根据)",
        r"^用户提供了.*?(?=\n\s*#{1,3}\s*|\n\s*[一二三四五六七八九十]+[、.．]|\n\s*根据)",
    ]
    for pattern in patterns:
        s2 = re.sub(pattern, "", s, flags=re.DOTALL).strip()
        if s2 != s:
            return s2
    return s


_CITATION_GROUP_RE = re.compile(r"\[[^\]]*?(?<!W)([1-9]\d*(?:\s*[,，、]\s*(?<!W)[1-9]\d*)*)[^\]]*?\]", re.IGNORECASE)
_CITATION_NUM_RE = re.compile(r"[1-9]\d*")
# 兼容「（参见[1]）」「参见[1]」等写法
_CITATION_INLINE_RE = re.compile(
    r"(?:参见|见)\s*[\[（(]?\s*(\d{1,3})\s*[\]）)]?",
    re.IGNORECASE,
)
_WEB_CITATION_GROUP_RE = re.compile(r"\[[^\]]*?W([1-9]\d*(?:\s*[,，、]\s*W?[1-9]\d*)*)[^\]]*?\]", re.IGNORECASE)
_WEB_CITATION_NUM_RE = re.compile(r"[1-9]\d*")
_WEB_SEE_CITATION_RE = re.compile(r"（参见\[W(\d{1,3})\]）", re.IGNORECASE)
_WEB_SEE_CITATION_LINK_RE = re.compile(
    r"（参见\[\[W(\d{1,3})\]\]\(([^)]*)\)）",
    re.IGNORECASE,
)
_WEB_MARKDOWN_LINK_RE = re.compile(r"\[\[W(\d{1,3})\]\]\([^)]*\)", re.IGNORECASE)


def _collect_kb_citation_nums(answer: str, kb_count: int) -> list[int]:
    """从正文中收集本地法条引用编号（[n] / （参见[n]）等）。"""
    used: list[int] = []
    for group in _CITATION_GROUP_RE.finditer(answer):
        for raw_num in _CITATION_NUM_RE.findall(group.group(1)):
            num = int(raw_num)
            if 1 <= num <= kb_count and num not in used:
                used.append(num)
    for match in _CITATION_INLINE_RE.finditer(answer):
        num = int(match.group(1))
        if 1 <= num <= kb_count and num not in used:
            used.append(num)
    return used


def _infer_kb_citations_by_law_text(answer: str, kb_results: list[dict]) -> list[int]:
    """模型仅写法条名称未标 [n] 时，按法律名称+条款在正文中回退匹配。"""
    used: list[int] = []
    if not answer or not kb_results:
        return used
    for idx, ref in enumerate(kb_results, 1):
        law_name = (ref.get("law_name") or "").strip()
        article = (ref.get("article") or "").strip()
        title = (ref.get("title") or "").strip()
        if not article and title:
            m = re.search(r"(第[一二三四五六七八九十百千零〇\d]+条)", title)
            if m:
                article = m.group(1)
        needles: list[str] = []
        if law_name and article:
            short = re.sub(r"^中华人民共和国", "", law_name)
            for name in {law_name, short}:
                if not name:
                    continue
                needles.extend(
                    [
                        f"《{name}》{article}",
                        f"{name}{article}",
                        f"{name} {article}",
                    ]
                )
                # 正文可能省略「中华人民共和国」或仅用简称
                if len(name) >= 6:
                    needles.append(name[-6:])
        if title and len(title) >= 6:
            needles.append(title)
        if any(n and n in answer for n in needles):
            if idx not in used:
                used.append(idx)
                continue
        if article and article in answer and law_name:
            short = re.sub(r"^中华人民共和国", "", law_name)
            if (short and short in answer) or (len(law_name) >= 4 and law_name[-4:] in answer):
                if idx not in used:
                    used.append(idx)
    return used


def _select_cited_kb_results(
    answer: str, kb_results: list[dict] | None
) -> tuple[str, list[dict]]:
    """
    只保留正文中实际出现过引用标记的知识库法条，并把正文引用重排为连续编号。

    例如模型正文使用 [1] 和 [3]，则最终正文改为 [1] 和 [2]，
    引用块只拼接原第 1、3 条，避免展示未被模型采用的召回结果。
    """
    if not answer or not kb_results:
        return answer, []

    used_old_nums = _collect_kb_citation_nums(answer, len(kb_results))
    if not used_old_nums:
        used_old_nums = _infer_kb_citations_by_law_text(answer, kb_results)
        if used_old_nums:
            logger.info(
                "[citation] 正文无法条编号标记，按法条名称回退匹配: %s",
                used_old_nums,
            )

    if not used_old_nums:
        return answer, []

    old_to_new = {old_num: idx + 1 for idx, old_num in enumerate(used_old_nums)}

    def replace_group(match: re.Match) -> str:
        remapped_nums: list[str] = []
        for raw_num in _CITATION_NUM_RE.findall(match.group(1)):
            old_num = int(raw_num)
            if old_num in old_to_new:
                new_num = str(old_to_new[old_num])
                if new_num not in remapped_nums:
                    remapped_nums.append(new_num)
        if not remapped_nums:
            return match.group(0)
        return "[" + ",".join(remapped_nums) + "]"

    renumbered_answer = _CITATION_GROUP_RE.sub(replace_group, answer)
    selected_results = [kb_results[old_num - 1] for old_num in used_old_nums]
    return renumbered_answer, selected_results


def _normalize_evidence_url(url: str) -> str:
    return (url or "").strip().split("#", 1)[0].rstrip("/").lower()


def _collect_web_citation_indices(answer: str, web_evidence: list[dict]) -> list[int]:
    """收集正文中的联网引用编号：支持（参见[Wn]）、[[Wn]]、以及正文出现过的 URL。"""
    used: list[int] = []
    n = len(web_evidence)
    if not answer or not n:
        return used
    for group in _WEB_CITATION_GROUP_RE.finditer(answer):
        for raw_num in _WEB_CITATION_NUM_RE.findall(group.group(1)):
            num = int(raw_num)
            if 1 <= num <= n and num not in used:
                used.append(num)
    for m in re.finditer(r"\[\[W(\d{1,3})\]\]", answer, re.IGNORECASE):
        num = int(m.group(1))
        if 1 <= num <= n and num not in used:
            used.append(num)
    answer_norm = _normalize_evidence_url(answer)
    for idx, ev in enumerate(web_evidence, 1):
        url_key = _normalize_evidence_url(str(ev.get("url", "")))
        if url_key and len(url_key) > 12 and url_key in answer_norm and idx not in used:
            used.append(idx)
    return used


def _web_citation_link(num: int, url: str) -> str:
    url = (url or "").strip()
    if url:
        return f"（参见[[W{num}]]({url})）"
    return f"（参见[W{num}]）"


def _normalize_web_citation_markers(answer: str) -> str:
    """整理联网引用格式，保留超链接；仅合并重复包裹。"""
    answer = re.sub(
        r"（参见\s*\[\[W(\d{1,3})\]\]\(([^)]*)\)\s*）",
        lambda m: f"（参见[[W{m.group(1)}]]({m.group(2)})）",
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(
        r"（参见\s*（参见\[\[W(\d{1,3})\]\]\(([^)]*)\)）\s*）",
        r"（参见[[W\1]](\2)）",
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(
        r"（参见\s*（参见\[W(\d{1,3})\]）\s*）",
        r"（参见[W\1]）",
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(
        r"（参见\[W(\d{1,3})\]）\s*）",
        r"（参见[W\1]）",
        answer,
        flags=re.IGNORECASE,
    )
    return answer


def _inject_web_citation_links(answer: str, web_evidence: list[dict] | None) -> str:
    """将正文中的（参见[Wn]）补全为带 URL 的 Markdown 超链接。"""
    if not answer or not web_evidence:
        return answer

    def _repl_plain(match: re.Match) -> str:
        num = int(match.group(1))
        if not (1 <= num <= len(web_evidence)):
            return match.group(0)
        return _web_citation_link(num, str(web_evidence[num - 1].get("url", "")))

    return _WEB_SEE_CITATION_RE.sub(_repl_plain, answer)


def _select_cited_web_results(
    answer: str, web_evidence: list[dict] | None
) -> tuple[str, list[dict]]:
    """补全正文 [Wn] 链接；保留 W 编号与送入模型的证据序号一致（不重排）。"""
    if not answer or not web_evidence:
        return answer, []
    answer = _normalize_web_citation_markers(answer)
    used_old_nums = _collect_web_citation_indices(answer, web_evidence)
    if not used_old_nums:
        return answer, []

    def _link_for(old_num: int) -> str | None:
        if not (1 <= old_num <= len(web_evidence)):
            return None
        url = str(web_evidence[old_num - 1].get("url", ""))
        return _web_citation_link(old_num, url)

    def replace_linked(match: re.Match) -> str:
        old_num = int(match.group(1))
        linked = _link_for(old_num)
        return linked if linked else match.group(0)

    def replace_plain(match: re.Match) -> str:
        old_num = int(match.group(1))
        linked = _link_for(old_num)
        return linked if linked else match.group(0)

    linked_answer = _WEB_SEE_CITATION_LINK_RE.sub(replace_linked, answer)
    linked_answer = _WEB_SEE_CITATION_RE.sub(replace_plain, linked_answer)
    selected_results = [web_evidence[old_num - 1] for old_num in used_old_nums]
    return linked_answer, selected_results


def _append_reference_blocks(
    answer: str,
    kb_results: list[dict] | None,
    web_evidence: list[dict] | None,
    *,
    deep_thinking: bool,
) -> str:
    answer = _normalize_web_citation_markers(answer)
    answer, cited_kb_results = _select_cited_kb_results(answer, kb_results)
    answer, cited_web_results = _select_cited_web_results(answer, web_evidence)
    # 始终按原始 W 序号查 url，避免重排后正文 W1 与证据列表第 1 条错位
    answer = _inject_web_citation_links(answer, web_evidence)
    ref_block = format_knowledge_references(cited_kb_results)
    # 标题筛选全文模式：文末列出全部入选网页，避免正文只标一处 Wn 时其余来源消失
    if (
        deep_thinking
        and web_evidence
        and all(
            (e.get("delivery") or "")
            in ("title_selected_full", "title_selected_summary", "page_summary")
            for e in web_evidence
        )
    ):
        web_for_footer = web_evidence
    else:
        web_for_footer = cited_web_results
    web_block = format_web_references(web_for_footer)
    if ref_block:
        answer = answer.rstrip() + ref_block
    elif deep_thinking:
        answer = answer.rstrip() + "\n\n### 参考法律法规\n本次回答未采用检索到的法条。"
    if web_block:
        answer = answer.rstrip() + web_block
    elif deep_thinking and web_enabled_placeholder(web_evidence):
        answer = answer.rstrip() + "\n\n### 联网可信来源\n本次回答未采用联网检索证据。"
    return answer


def web_enabled_placeholder(web_evidence: list[dict] | None) -> bool:
    """仅用于判断是否展示联网来源空状态；有候选证据但未被引用时返回 True。"""
    return bool(web_evidence)


def selsect_best_answer(last_answer):

    return


def _run_kb_retrieval_with_pulse(
    result_queue: queue.Queue,
    retrieve_callable,
    pulse_interval_s: float = 10.0,
):
    """长时间知识库检索期间定期写入队列，避免界面长时间停在首条提示。"""
    stop = threading.Event()

    def _pulse_loop():
        msg = (
            "知识库检索仍在进行（首次构建 BM25 / 稠密索引、或与后台预热并行时可能较慢），"
            "请稍候…\n\n"
        )
        while not stop.wait(pulse_interval_s):
            try:
                result_queue.put([False, msg])
            except Exception:
                break

    t = threading.Thread(target=_pulse_loop, daemon=True, name="kb-retrieval-pulse")
    t.start()
    try:
        return retrieve_callable()
    finally:
        stop.set()


def _max_web_queries_from_config(config: AppConfig) -> int:
    web_cfg = getattr(config, "web", None)
    if web_cfg is None:
        return 5
    return max(1, int(getattr(web_cfg, "max_queries", 5)))


def _needs_retrieval_plan(config: AppConfig, pattern: str, user_query: str) -> bool:
    """rewrite 或 deep_thinking+联网 时，一次 LLM 规划 KB/Web 检索词。"""
    if not (user_query or "").strip():
        return False
    if config.maxhop.rewrite:
        return True
    return pattern == "deep_thinking" and web_enabled(config)


def _make_retrieval_plan_prompt_builder(max_web_queries: int, need_web_queries: bool):
    def _builder(
        user_input: str,
        history_answer: Any,
        references: Any,
    ) -> str:
        return build_retrieval_plan_prompt(
            user_input,
            history_answer,
            references,
            max_web_queries=max_web_queries,
            need_web_queries=need_web_queries,
        )

    return _builder


def _fallback_retrieval_plan(
    user_query: str, max_web_queries: int, need_web: bool
) -> dict[str, Any]:
    uq = effective_user_question_for_web(user_query)
    return {
        "display_query": uq,
        "kb_query": uq,
        "bm25_keywords": [],
        "web_queries": ([uq] if need_web else [])[:max_web_queries],
    }


def _parse_retrieval_plan_output(
    raw: str | None,
    user_query: str,
    max_web_queries: int,
    need_web: bool,
) -> dict[str, Any]:
    fallback = _fallback_retrieval_plan(user_query, max_web_queries, need_web)
    if not raw or not isinstance(raw, str):
        logger.warning("统一检索规划无有效输出，使用回退方案")
        return fallback

    text = raw.strip()
    text = re.sub(r"(?is)<thinking>.*?</thinking>", " ", text)
    text = re.sub(r"(?is)<think>.*?</think>", " ", text)
    text = text.replace("```json", "").replace("```", "").strip()

    obj: Any = None
    if "{" in text and "}" in text:
        cand = text[text.find("{") : text.rfind("}") + 1]
        cand = (
            cand.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\uff0c", ",")
            .strip()
        )
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            try:
                obj = ast.literal_eval(cand)
            except Exception:
                obj = None

    if not isinstance(obj, dict):
        logger.warning(
            "统一检索规划 JSON 解析失败，回退原问。原始输出(节选): %r",
            raw[:500],
        )
        return fallback

    display = str(obj.get("display_query") or "").strip()
    kb_q = str(obj.get("kb_query") or "").strip()
    keywords_raw = obj.get("bm25_keywords") or []
    web_raw = obj.get("web_queries") or []

    if isinstance(keywords_raw, str):
        keywords = [
            k for k in re.sub(r"[，,、；;]+", " ", keywords_raw).split() if k
        ]
    elif isinstance(keywords_raw, list):
        keywords = [str(k).strip() for k in keywords_raw if str(k).strip()]
    else:
        keywords = []

    web_queries: list[str] = []
    if isinstance(web_raw, str) and web_raw.strip():
        web_queries = [web_raw.strip()]
    elif isinstance(web_raw, list):
        web_queries = [str(q).strip() for q in web_raw if str(q).strip()]

    if not display:
        display = fallback["display_query"]
    if not kb_q:
        kb_q = display.rstrip("？?").strip() or fallback["kb_query"]
    if need_web and not web_queries:
        web_queries = list(fallback["web_queries"])

    return {
        "display_query": display,
        "kb_query": kb_q,
        "bm25_keywords": keywords[:5],
        "web_queries": web_queries[:max_web_queries],
    }


@timing
def deep_search_rag(
    user_query: str,
    mode: str,
    result_queue: queue.Queue,
    history_message: list[HistoryMessage],
    config: AppConfig,
    cancel_event=None,
) -> str:
    """
    Deep Search RAG 主流程：
    1. 安全校验
    2. 内部知识库语义检索
    3. 外部深度网络搜索
    4. 上下文融合封装

    返回结构化检索结果，供下游 LLM 生成答案使用。

    Args:
        user_query (str): 用户自然语言问题

        mode (str): 查询模式
        result_queue (queue.Queue): 结果队列，用于流式输出
        history_message (list[HistoryMessage]): 历史消息列表
        config (AppConfig): 应用配置对象
    Returns:
        Dict: 包含完整检索上下文的响应对象
    """
    logger.info("=== 进入 deep_search_rag 函数 ===")
    query_result: QueryResult = {
        "query": "",
        "rewrite": "",
        "ref": [],
        "score": [],
        "sub_query": [],
        "answer": [],
    }

    temp_dr_query_result: DeepResearchResult = {
        "depth": 0,
        "query": "",
        "topicreport": "",
        "sub_query": [],
        "subtopicreport": [],
        "answerreport": "",
        "ref": [],
        "topicreportscore": -1.0,
        "subdeepresearch": [],
    }

    deep_research_query_result: DeepResearchResult = copy.deepcopy(temp_dr_query_result)
    reference_list: ReferenceList = []
    web_evidence: list[dict] = []
    query_deepresearch_try: HistoryMessage = []

    context_message = ""
    qa_trace = QaTrace(config.pattern.select_pattern, user_query)
    logger.info("用户原始查询问题: %r", user_query)
    logger.info(
        f"config.maxhop.turn_on_deepresearch: {config.maxhop.turn_on_deepresearch}"
    )

    # session_chat / 其它调用方传入的 mode 必须与 yaml 中的 pattern 对齐。
    # 与 run_server_streaming.stream_typewriter 一致：运行时用 mode 覆盖 select_pattern。
    # 否则会出现 env 里是 deep_thinking、yaml 里是 chat，实际却始终走 chat 的情况。
    _mode_norm = (mode or "").strip()
    _allowed_patterns = frozenset(
        {"chat", "deep_thinking", "deep_research", "deep_speed"}
    )
    if _mode_norm and _mode_norm in _allowed_patterns:
        prev_pat = config.pattern.select_pattern
        if prev_pat != _mode_norm:
            logger.info(
                "运行时 mode=%r 覆盖 config.pattern.select_pattern: %r -> %r",
                _mode_norm,
                prev_pat,
                _mode_norm,
            )
            config.pattern.select_pattern = _mode_norm
    elif _mode_norm:
        logger.warning(
            "未知 mode=%r，保留 yaml 中的 select_pattern=%r（合法值：%s）",
            _mode_norm,
            config.pattern.select_pattern,
            ", ".join(sorted(_allowed_patterns)),
        )

    if config.pattern.select_pattern != "chat":
        if config.pattern.select_pattern == "deep_thinking":
            # 打开 deep_thinking 模式相关配置
            config.maxhop.norefturn = False
            config.maxhop.urlmaxnum = 8
            # 关闭 deepresearch 模式相关配置
            config.maxhop.turn_on_deepresearch = False
            logger.info(
                "pattern=deep_thinking：按深度思考分支调整 maxhop 参数（将进行知识库检索配置）"
            )
            # 关闭chat 模式相关配置
            config.maxhop.norefturn = False

        elif config.pattern.select_pattern == "deep_research":
            # 打开 deep 模式相关配置
            config.maxhop.turn_on_deepresearch = True
            config.maxhop.urlmaxnum = 5
            logger.info(
                "pattern=deep_research：按深度研究分支调整 maxhop 参数（启用 deep research 流程）"
            )
            # 关闭chat 模式相关配置
            config.maxhop.norefturn = False
    else:
        # chat 模式相关配置/关闭 deep_speed 和  deep 模式相关配置
        config.maxhop.norefturn = True
        config.maxhop.turn_on_deepresearch = False
        logger.info("当前模式为 chat 模式")

    # 单次运行/不进行session管理
    if config.maxhop.turn_on_deepresearch:

        logger.info(
            "========= 进入深度研究模块 ========, 模式为: %r, %r",
            config.pattern.select_pattern,
            config.maxhop.turn_on_deepresearch,
        )
        deep_research_agent(
            0,
            0,
            user_query,
            "",
            deep_research_query_result,
            result_queue,
            history_message,
            config,
        )
        logger.info(
            "深度研究模块deep search  研究结果: \n %s",
            json.dumps(deep_research_query_result, ensure_ascii=False),
        )
        model_output = send_request_to_model(
            user_query=deep_research_query_result["answerreport"],
            references=deep_research_query_result["ref"],
            prompt_builder=build_deep_research_beautiful_format_prompt,
            model_name=config.deepresearch.name,
            api_url=config.deepresearch.server,
        )

        result_queue.put(
            [
                False,
                "<br>"
                + "\n下面对上诉Deep research 思考过程进行最后的汇总和整理，详情如下：\n\n"
                + "<br>",
            ]
        )
        result_queue.put([True, model_output if model_output else ""])
        logger.info(
            "Deep research 本次session 查询结果: \n %s",
            json.dumps(deep_research_query_result, ensure_ascii=False),
        )
        return deep_research_query_result["answerreport"]

    # Step 0: session //session消息上下文管理
    # 需要考虑 query + history_query
    if config.maxhop.turnsession and len(history_message) > 0:
        logger.info("\n根据系统的配置进入session管理模块\n")
        session_message, history_query, total_answer = joint_history_message(
            history_message[-config.maxhop.session :]
        )
        logger.info(
            "系统配置需要考虑步长 %d session_message 历史消息为: %s",
            config.maxhop.session,
            json.dumps(session_message, ensure_ascii=False),
        )
        logger.info(
            "系统配置需要考虑步长 %d history_query 历史消息为: %s",
            config.maxhop.session,
            json.dumps(history_query, ensure_ascii=False),
        )
        rule_decision = _session_rule_decision(user_query)
        if rule_decision is True:
            logger.info("[session] 规则判定：延续，跳过 session 模型")
            intention_score = 1.0
        elif rule_decision is False:
            logger.info("[session] 规则判定：新话题，跳过 session 模型")
            intention_score = 0.0
        else:
            intention_reg = send_request_to_model(
                user_query=user_query,
                history_answer=history_query,
                references=total_answer,
                prompt_builder=build_session_prompt,
                model_name=config.session.name,
                api_url=config.session.server,
            )
            logger.info("session 判别模型输出结果为: %r", intention_reg)
            intention_score = parser_session_output(intention_reg)

        if intention_score >= config.maxhop.sessionthreshold:
            logger.info(
                "\nsession 判别模型输出结果超过阈值 %f, 需要考虑session历史消息\n",
                intention_score,
            )
            context_message = session_message
        else:
            logger.info(
                "\nsession 判别模型输出结果未超过阈值 %f, 不需要考虑session历史消息, 开启新会话\n",
                intention_score,
            )

    # 这才是用户真实的查询语句 + query_result["query"] = user_query 如何考虑
    # cur_sub_user_query = user_query

    if context_message != "":
        user_query = history_query + "\n" + user_query
    logger.info("\n用户真实查询问题: %r\n", user_query)
    qa_trace.stage("effective_query", effective_query=user_query)

    query_result["query"] = user_query  # 是否需要添加

    # Step 1: 检索规划（一次 LLM → KB 问句 + 联网检索词）
    pattern = config.pattern.select_pattern
    need_web = pattern == "deep_thinking" and web_enabled(config)
    query_rewrite = user_query
    query_for_retrieval = user_query
    planned_web_queries: list[str] = []

    if _needs_retrieval_plan(config, pattern, user_query):
        max_web_q = _max_web_queries_from_config(config)
        logger.info(
            "[retrieval_plan] 开始（rewrite=%s web=%s max_web=%d）",
            config.maxhop.rewrite,
            need_web,
            max_web_q,
        )
        plan_raw = send_request_to_model(
            user_query=user_query,
            history_answer=context_message if context_message else None,
            references=None,
            prompt_builder=_make_retrieval_plan_prompt_builder(max_web_q, need_web),
            model_name=config.rewrite.name,
            api_url=config.rewrite.server,
        )
        plan = _parse_retrieval_plan_output(
            plan_raw, user_query, max_web_q, need_web
        )
        query_rewrite = plan["display_query"]
        kb_query = plan["kb_query"]
        kw = plan["bm25_keywords"]
        query_for_retrieval = (
            f"{kb_query} {' '.join(kw)}".strip() if kw else kb_query
        )
        planned_web_queries = plan["web_queries"] if need_web else []
        query_result["rewrite"] = query_rewrite
        logger.info(
            "[retrieval_plan] display=%r kb=%r keywords=%s web=%s",
            query_rewrite,
            kb_query,
            kw,
            planned_web_queries,
        )
        print(
            f"[retrieval_plan] display_query={query_rewrite!r} kb_query={kb_query!r} "
            f"bm25_keywords={kw!r} web_queries={planned_web_queries!r}",
            flush=True,
        )
        qa_trace.stage(
            "retrieval_plan",
            display_query=query_rewrite,
            kb_query=kb_query,
            bm25_keywords=kw,
            web_queries=planned_web_queries,
            query_for_retrieval=query_for_retrieval,
            plan_raw=(plan_raw or "")[:500],
        )
    else:
        logger.info("[retrieval_plan] 跳过（rewrite=False 且未启用 deep_thinking 联网）")
        qa_trace.stage(
            "retrieval_plan",
            skipped=True,
            query_for_retrieval=query_for_retrieval,
        )

    # 安全校验
    if not validate_input_safety(query_rewrite):
        logger.error("输入内容不符合安全规范，请修改后重试")
        qa_trace.flush(error="输入内容不符合安全规范，请修改后重试")
        return "errror: 输入内容不符合安全规范，请修改后重试"

    # rag：仅 deep_thinking 走知识库检索；chat 不走知识库；其它模式不动
    if config.pattern.select_pattern == "deep_thinking":
        logger.info(
            "deep_thinking 模式：检索串=%r（展示给用户的问句=%r）",
            query_for_retrieval,
            query_rewrite,
        )
        _kb_progress = (
            f"正在基于追加问题检索知识库：{query_rewrite}\n\n"
            if len(history_message) > 0
            else f"正在基于原问题检索知识库：{query_rewrite}\n\n"
        )
        result_queue.put([False, _kb_progress])
        t_retrieval = time.perf_counter()
        with Timer(name="内部知识库检索", logger=logger, level=logging.INFO):
            kb_results = _run_kb_retrieval_with_pulse(
                result_queue,
                lambda: retrieve_from_knowledge_base(query_for_retrieval, config=config),
            )
        retrieval_elapsed = time.perf_counter() - t_retrieval
        qa_trace.stage(
            "knowledge_retrieval",
            duration_s=retrieval_elapsed,
            query=query_for_retrieval,
            recall_count=len(kb_results or []),
            recall_preview=format_recall_preview(kb_results),
        )
        result_queue.put([
            False,
            f"知识库检索完成，共召回 {len(kb_results or [])} 条，正在生成答案...\n\n",
        ])
        if kb_results is not None:
            logger.info("deep_thinking 知识库共获取到 %d 条参考文献。", len(kb_results))
    elif config.pattern.select_pattern == "chat":
        kb_results = None
        logger.info("chat 模式：不进行知识库检索")
        qa_trace.stage("knowledge_retrieval", skipped="chat 模式不检索知识库")
    else:
        if not config.maxhop.norefturn:
            logger.info(
                "开始进行 rag 外部深度搜索，检索串=%r（问句=%r）",
                query_for_retrieval,
                query_rewrite,
            )
            t_retrieval = time.perf_counter()
            with Timer(
                name="开始进行 rag 外部深度搜索，查询语句, 当前的_epochs_0，当前的_depth_0",
                logger=logger,
                level=logging.INFO,
            ):
                logger.info("开始进行 内部知识库检索，查询语句: %r", query_for_retrieval)
                kb_results = _run_kb_retrieval_with_pulse(
                    result_queue,
                    lambda: retrieve_from_knowledge_base(query_for_retrieval, config=config),
                )
            retrieval_elapsed = time.perf_counter() - t_retrieval
            qa_trace.stage(
                "knowledge_retrieval",
                duration_s=retrieval_elapsed,
                query=query_rewrite,
                recall_count=len(kb_results or []),
                recall_preview=format_recall_preview(kb_results),
            )
            if kb_results is not None:
                logger.info("首次且 外部深度搜索共获取到 %d 条参考文献。", len(kb_results))
        else:
            kb_results = None
            logger.info("首次且 不走外部搜索，走模型自动推理")
            qa_trace.stage("knowledge_retrieval", skipped="norefturn=True")
    
    # 将 kb_results 转换为 ReferenceItem 列表
    if kb_results is not None:
        for item in kb_results:
            law_name = (item.get("law_name") or item.get("category") or "").strip()
            article = (item.get("article") or item.get("number") or "").strip()
            title_parts = [part for part in [law_name, article] if part]
            title = " ".join(title_parts).strip() or "法律条文"
            content = (item.get("text") or "").strip()
            score_raw = item.get("score")
            score = round(float(score_raw), 3) if isinstance(score_raw, (int, float)) else 0.0
            reference_list.append(
                ReferenceItem(
                    {
                        "title": title,
                        "content": content,
                        "relevance_score": score,
                        "url": "",
                        "extracted_content": content,
                        "prof_depth_score": 0.0,
                    }
                )
            )
    if config.pattern.select_pattern == "deep_thinking" and web_enabled(config):
        web_cfg = getattr(config, "web", None)
        summarizing = bool(getattr(web_cfg, "page_summary_enabled", False))
        web_status = (
            "正在联网检索可信来源，并对入选网页生成摘要...\n\n"
            if summarizing
            else "正在联网检索可信来源并抽取证据...\n\n"
        )
        result_queue.put([False, web_status])
        t_web = time.perf_counter()
        try:
            web_evidence, web_trace = retrieve_web_evidence(
                query_rewrite,
                kb_results,
                config,
                planned_search_queries=planned_web_queries,
            )
        except Exception as e:
            logger.exception("联网可信源检索失败: %s", e)
            web_evidence, web_trace = [], {"error": str(e)}
        qa_trace.stage(
            "web_retrieval",
            duration_s=time.perf_counter() - t_web,
            **web_trace,
        )
        result_queue.put([
            False,
            f"联网检索完成，共生成 {len(web_evidence)} 条可信证据，正在融合回答...\n\n",
        ])
        for item in web_evidence:
            reference_list.append(
                ReferenceItem(
                    {
                        "title": item.get("title", "") or "联网可信来源",
                        "content": item.get("content", "") or item.get("quote", ""),
                        "url": item.get("url", ""),
                        "extracted_content": item.get("content", "")
                        or item.get("extracted_content", "")
                        or item.get("quote", ""),
                        "relevance_score": float(item.get("relevance_score", 0.0) or 0.0),
                        "prof_depth_score": 0.0,
                        "source_type": "web",
                        "citation_id": item.get("citation_id", ""),
                        "source": item.get("source", ""),
                        "published_at": item.get("published_at", ""),
                        "delivery": item.get("delivery", ""),
                    }
                )
            )
    logger.info(
        "首次且 内部知识库检索共获取到 %d 条参考文献。", len(reference_list)
    )
    logger.info(
        "首次且 内部知识库检索结果为: \n %s",
        json.dumps(reference_list, ensure_ascii=False),
    )
    query_result["ref"].append(reference_list)
    if cancel_event is not None and cancel_event.is_set():
        logger.info("请求已取消，跳过大模型生成。")
        return ""
    t_llm = time.perf_counter()
    with Timer(
        name="首次大模型请求计算 deep research ", logger=logger, level=logging.INFO
    ):
        # 首次可以基于小模型给出参考答案或者思考过程，后续基于大模型给出最终答案
        # 这里可以加速优化 // 由于历史答案已经非常好了，可以直接基于历史答案进行优化
        if config.pattern.select_pattern == "chat":
            build_deep_research_prompt = build_deep_research_session_chat_prompt
        else:
            build_deep_research_prompt = build_deep_research_session_rag_prompt
        
        # 使用流式请求，实时推送 think 内容到队列
        sr = config.speeddeepresearch
        answer = send_request_to_model_streaming(
            user_query=user_query,
            prompt_builder=build_deep_research_prompt,
            result_queue=result_queue,
            history_answer=context_message,
            references=reference_list,
            model_name=sr.name,
            api_url=sr.server,
            temperature=float(sr.temperature) if sr.temperature is not None else 0.7,
            max_tokens=int(sr.max_tokens),
            top_p=float(sr.top_p),
            repetition_penalty=float(sr.repetition_penalty),
            cancel_event=cancel_event,
        )
        qa_trace.stage(
            "answer_llm_streaming",
            duration_s=time.perf_counter() - t_llm,
            model=config.speeddeepresearch.name,
            api=config.speeddeepresearch.server,
            references_sent_count=len(reference_list),
            model_output=answer or "",
        )
        if not check_llm_output(
            answer, function_name="build_deep_research_prompt", query=user_query
        ):
            logger.error("首次 deepresearch 模型请求失败，终止整个流程。")
            qa_trace.flush(error=model_request_error())
            return model_request_error()
        # 流式函数已经处理了 <think> 标签的解析和推送，这里只需要确保 answer 是有效字符串
        if answer is None or not isinstance(answer, str):
            answer = ""

    # 检测返回值，如果为 None 或非字符串，则赋值为空字符串# 需要进行异常检测
    answer = answer if isinstance(answer, str) else ""
    answer = dedupe_repeated_answer_body(
        _strip_meta_analysis(_strip_generated_reference_section(answer))
    )
    query_result["answer"].append(answer)
    query_result["score"].append(0)

    # 首轮信息的完整性和有效性 严重影响和制约 整个系统的性能，除非在多轮中进行修正
    logger.info(
        "首次 进行 deepresearch 大模型请求，模型根据首次查询到的数据生成的答案: \n %r",
        answer,
    )
    epochs: int = 1
    if config.pattern.select_pattern in ["chat", "deep_thinking"]:
        # chat/deep_thinking 模式只要首答，不跑多轮 mutil_hop_search
        query_deepresearch_try.append(copy.deepcopy(query_result))
        logger.info(f"{config.pattern.select_pattern} 模式：仅使用首答，跳过多轮深度搜索")
    else:
        while True and epochs < config.maxhop.maxepochs:
            logger.info("进入多轮深度搜索环节，当前轮次: %d;", epochs)
            query_result_cp = copy.deepcopy(query_result)
            flag, query_result_cp = mutil_hop_search(
                user_query,
                query_result_cp,
                result_queue,
                1,
                config.maxhop.max_hops,
                config,
                epochs=epochs,
            )
            query_deepresearch_try.append(query_result_cp)
            if flag:
                break
            epochs = epochs + 1
            logger.info("多轮深度搜索未达到终止条件，继续下一轮搜索。当前轮次: %d;", epochs)

    # 获取最佳结果多轮搜索，这里有bug，需要修改
    logger.info(
        "query_deepresearch_try 本次session 查询结果: \n %s",
        json.dumps(query_deepresearch_try, ensure_ascii=False),
    )

    # 下面是结果的格式化输出（chat/deep_thinking 模式只用首答，不再做格式化大模型调用）
    if config.formatOutput and config.pattern.select_pattern not in ["chat", "deep_thinking"]:
        logger.info("系统配置需要对最终结果进行格式化输出")
        build_format_prompt = build_beautiful_format_prompt
        if config.pattern.select_pattern == "deep_speed":
            build_format_prompt = build_beautiful_format_prompt
        elif config.pattern.select_pattern == "deep_thinking":
            build_format_prompt = build_beautiful_format_rag_prompt
        elif config.pattern.select_pattern == "chat":
            build_format_prompt = build_beautiful_format_noref_prompt
        # 结果格式化// 可能格式化出错误// 需要修复
        logger.info(
            "最后结果的输出，格式化前:\n %r", query_deepresearch_try[-1]["answer"][-1]
        )
        model_output_format = send_request_to_model(
            user_query=query_deepresearch_try[-1]["answer"][-1],
            references=get_ref_deep_speed(query_deepresearch_try),
            prompt_builder=build_format_prompt,
            model_name=config.deepresearch.name,  # 可以加速
            api_url=config.deepresearch.server, # 可以加速
            # model_name=config.beautifulformat.name,
            # api_url=config.beautifulformat.server,
        )
        if not check_llm_output(
            model_output_format, function_name="build_beautiful_format_prompt", query=user_query
        ):
            qa_trace.flush(error=model_request_error())
            return model_request_error()
        model_output_format = model_output_format if isinstance(model_output_format, str) else ""
        model_output_format = dedupe_repeated_answer_body(
            _strip_meta_analysis(
                _strip_generated_reference_section(
                    model_output_format.split("</think>")[-1]
                )
            )
        )
        model_output_format = _append_reference_blocks(
            model_output_format,
            kb_results,
            web_evidence,
            deep_thinking=config.pattern.select_pattern == "deep_thinking",
        )
        result_queue.put([True, model_output_format])  # 流式输出测试
        qa_trace.flush(model_output_format)
        logger.info("最后结果的输出，格式化后:\n %r", model_output_format)
    else:
        logger.info(
            "系统配置不需要对最终结果进行格式化输出，直接返回原始结果"
        )
        model_output = query_deepresearch_try[-1]["answer"][-1]
        model_output = dedupe_repeated_answer_body(
            _strip_meta_analysis(_strip_generated_reference_section(model_output))
        )
        logger.info("最后结果的输出，未发生格式 :\n %r", model_output)
        model_output = _append_reference_blocks(
            model_output,
            kb_results,
            web_evidence,
            deep_thinking=config.pattern.select_pattern == "deep_thinking",
        )
        result_queue.put([True, model_output])  # 流式输出测试
        qa_trace.flush(model_output)
    # 溯源有bug//
    # new_text, new_references = replace_and_renumber_citations(model_output, ref_source)

    history_message.append(query_deepresearch_try)
    history_message = history_message[-config.maxhop.historynum :]
    logger.info("本次session 深度搜索检索结束。")
    logger.info(
        "汇总本次查询的reAct相关信息：epochs %d；query_deepresearch_try 长度为 %d；两者应该相等",
        epochs,
        len(query_deepresearch_try),
    )
    logger.info("=== 退出 deep_search_rag 函数 ===")
    logger.info("历史会话信息:  \n %s", json.dumps(history_message, ensure_ascii=False))
    return query_deepresearch_try[-1]["answer"][-1]


# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # judge_rlt = send_request_to_model(user_query=model_output.replace(" ", "").split("</think>")[-1],references=None, prompt_builder=build_quality_judgment_prompt, model_name=judge_model_name, api_url=judge_url)  # noqa: E501

    query = "什么是大模型"
    from deep_research.paths import default_config_path

    config = load_validated_config(default_config_path())

    # from deep_research.local_logger import setup_logger
    # logger = setup_logger(__name__, log_file="deepresearch.log", level=logging.DEBUG)
    # logger.info("欢迎使用由中国电子云 AI 团队开发的 deepresearch：Blue Sharp ！！！")
    # # deep_search_rag_bound_fn = partial(deep_search_rag, config=config)
    # response = deep_search_rag(query, 1, config=config, logger=logger)
    # logger.info("用户数据: %s", json.dumps(data, ensure_ascii=False))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer.name, trust_remote_code=True
    )

    # import json

    # print(json.dumps(response, indent=2, ensure_ascii=False))
