# -*- coding: utf-8 -*-  # noqa: UP009
"""
deep research 写作 agent 模块（无 RAG / 无联网）

Author: wjianxz
Date: 2025-11-13
"""
from typing import List  # used in get_total_reference_list
import copy
import queue
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from deep_research.protocal import (
    HistoryMessage,
    QueryResult,
    DeepResearchResult,
    SubQueryItem,
    ReferenceItem,
)
from deep_research.utils import (
    send_request_to_model,
    send_request_to_model_dr,
    send_request_to_model_dr_streaming,
    parser_sub_topic_output,
    parser_confidence,
    set_min_request_interval,
    is_invalid_output,
)
from deep_research.parser_config import AppConfig
from deep_research.prompts import (
    build_deep_research_report_topic_prompt,
    build_deep_research_report_subtopic_prompt,
    build_deep_research_report_prompt,
    build_deep_research_sub_report_prompt,
    build_bid_report_topic_prompt,
    build_bid_report_subtopic_prompt,
    build_bid_report_prompt,
    build_bid_sub_report_prompt,
    build_confidence_prompt,
    build_chapter_consistency_judgment_prompt,
)
from deep_research.local_logger import timing, Timer


def _bid_doc_type_hint_local(user_text: str) -> str:
    t = (user_text or "").strip()
    if any(k in t for k in ["投标", "投标文件", "投标书"]):
        return "投标文件"
    if any(k in t for k in ["评标", "评审", "评分规则", "打分规则"]):
        return "评标规则"
    return "招标文件"


def _fixed_bid_outline(doc_type: str) -> list[str]:
    if doc_type == "投标文件":
        # 与《投标模版》目录核心保持一致（可扩展，但先保证“能生成完整投标文件”）
        return [
            "投标文件封面与目录",
            "格式一：投标函",
            "格式二：开标一览表",
            "格式三：分项报价表",
            "格式四：政策适用性说明",
            "格式五：实质性响应一览表（★/▲条款逐条响应）",
            "格式六：法定代表人证明书",
            "格式七：法定代表人授权书",
            "格式八：独立承担民事责任能力证明材料",
            "格式九：承诺函（逐条复制采购文件条款并承诺）",
            "格式十：中小企业声明函（如适用）",
            "格式十一：监狱企业（如适用）",
            "格式十二：残疾人福利性单位声明函（如适用）",
            "格式十三：联合体共同投标协议书（如适用）",
            "格式十四：投标人业绩情况表（如适用）",
            "格式十五：技术和服务要求响应表（镜像对照）",
            "格式十六：商务条件响应表（具体承诺，不写“满足/符合”）",
            "格式十七：履约进度计划表",
            "格式十八：各类证明材料（占位符）",
            "格式十九：采购代理服务费支付承诺书（如适用）",
            "格式二十：需要采购人提供的附加条件（如有）",
            "格式二十一：项目实施方案、质量保证及售后服务承诺等",
            "格式二十二：附件（占位符）",
            "格式二十三：政府采购履约担保函/履约保险凭证（如适用）",
        ]
    if doc_type == "招标文件":
        # 你期望的“完整标书”章节（含合同条款与投标文件格式）
        return [
            "第一章 投标邀请",
            "第二章 采购需求",
            "第三章 投标人须知",
            "第四章 评标办法",
            "第五章 合同条款及格式",
            "第六章 投标文件格式",
        ]
    # 评标规则：给一个固定结构（后续可再细化）
    return [
        "评标原则与依据",
        "评审组织与职责",
        "评审程序与符合性审查",
        "详细评审（技术/商务/价格）与评分表",
        "无效投标情形与否决条款（★条款）",
        "定标规则与结果公示/质疑投诉处理",
    ]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  deep_research_agent: 主题规划 → 大纲拆分 → 逐章撰写 → 合并 + 质量把控
# ---------------------------------------------------------------------------


def _chapter_passes_fast_check(text: str, config: AppConfig) -> bool:
    """章节级轻量质检：长度 + 明显异常输出。"""
    body = (text or "").strip()
    if len(body) < max(200, config.maxhop.chapter_min_chars):
        return False
    if is_invalid_output(body):
        return False
    return True


def _topic_consistent_with_chapter(chapter_title: str, text: str) -> bool:
    """
    章节主题一致性校验（轻量规则）：
    - 从章节标题提取关键词（中文短语/英文词）
    - 要求正文命中至少 1~2 个关键词，避免明显跑偏
    """
    body = (text or "").strip()
    title = (chapter_title or "").strip()
    if not body or not title:
        return False

    parts = re.split(r"[，,。；;：:、（）()\[\]\s\-]+", title)
    tokens = []
    for p in parts:
        p = p.strip()
        if len(p) >= 2:
            tokens.append(p)
    if not tokens:
        return True

    hit = sum(1 for t in tokens if t in body)
    need = 1 if len(tokens) <= 2 else 2
    return hit >= need


def _paragraph_consistency_check(
    chapter_text: str,
    user_query: str,
    topic_content: str,
    chapter_title: str,
    outline_titles: list[str],
    config: AppConfig,
) -> tuple[bool, float]:
    """
    章节一致性校验（语义判定版）：
    - 调用一次模型进行语义审核
    - 仅接受输出“是/否”
    """
    if not config.maxhop.enable_paragraph_consistency_check:
        return True, 1.0
    if not (chapter_text or "").strip():
        return False, 0.0

    packed_input = (
        f"【用户原始需求】\n{user_query}\n\n"
        f"【当前主题框架】\n{topic_content}\n\n"
        f"【当前章节标题】\n{chapter_title}\n\n"
        f"【当前大纲标题】\n" + "\n".join(outline_titles) + "\n\n"
        f"【章节正文】\n{chapter_text}\n"
    )
    judge = send_request_to_model(
        user_query=packed_input,
        history_answer="",
        references=None,
        prompt_builder=build_chapter_consistency_judgment_prompt,
        model_name=config.confidence.name,
        api_url=config.confidence.server,
        temperature=0.0,
        max_tokens=16,
        top_p=1.0,
        repetition_penalty=config.confidence.repetition_penalty,
    )
    if judge is None:
        return False, 0.0

    first = judge.strip().splitlines()[0].strip() if judge.strip() else ""
    normalized = first.replace("。", "").replace(" ", "")
    ok = normalized.startswith("是")
    return ok, 1.0 if ok else 0.0


def _sanitize_outline_titles(items: list[str]) -> list[str]:
    """清理明显跑偏的元内容标题。"""
    banned = [
        "研究方法",
        "方法与特色",
        "写作进度",
        "进度建议",
        "总体构思",
        "总体写作思路",
        "写作特色",
        "结语",
    ]
    out: list[str] = []
    for it in items:
        t = (it or "").strip()
        if not t:
            continue
        # 过滤 vLLM/部分网关残留的 think 标签或碎片
        if "<think" in t.lower() or "</think>" in t.lower():
            continue
        if t in ("<think>", "</think>", "<think></think>"):
            continue
        if any(b in t for b in banned):
            continue
        out.append(t)
    return out


def _split_paragraphs_for_preview(text: str) -> list[str]:
    """用于思考区展示的段落切分。"""
    body = (text or "").strip()
    if not body:
        return []
    parts = re.split(r"\n\s*\n+", body)
    return [p.strip() for p in parts if p.strip()]


def _build_runtime_constraints(config: AppConfig, depth: int, root_sections: int) -> str:
    """将 config 中用户可配置约束拼成文本，传入 prompt。"""
    lines: list[str] = []
    if config.maxhop.target_total_words is not None and config.maxhop.target_total_words > 0:
        lines.append(f"全文目标字数：约 {config.maxhop.target_total_words} 字")
    if depth == 0 and root_sections > 0:
        lines.append(f"总章节数目标：{root_sections} 章")
    if config.maxhop.target_recursion_depth is not None and config.maxhop.target_recursion_depth > 0:
        lines.append(f"递归层数目标：{config.maxhop.target_recursion_depth} 层")
    return "\n".join(f"- {x}" for x in lines)

def deep_research_agent(
    depth: int,
    wide: int,
    user_query: str,
    far_topic: str,
    deep_research_query_result: DeepResearchResult,
    result_queue: queue.Queue,
    history_message: list[HistoryMessage],
    config: AppConfig,
    recursion_state: dict[str, int] | None = None,
) -> DeepResearchResult:
    """递归式 Deep Research 写作 agent。"""
    if recursion_state is None:
        recursion_state = {"nodes": 0}
    max_nodes = max(1, config.maxhop.deepresearch_max_nodes)
    if recursion_state["nodes"] >= max_nodes:
        logger.warning("达到递归节点上限(%d)，停止继续下钻", max_nodes)
        return deep_research_query_result
    recursion_state["nodes"] += 1

    effective_max_depth = (
        config.maxhop.target_recursion_depth
        if config.maxhop.target_recursion_depth is not None
        and config.maxhop.target_recursion_depth > 0
        else config.maxhop.deepresearch_max_depth
    )
    if depth >= effective_max_depth:
        return deep_research_query_result

    set_min_request_interval(config.maxhop.min_request_interval_seconds)

    deep_research_query_result["depth"] = depth
    deep_research_query_result["query"] = user_query

    temp_dr_query_result: DeepResearchResult = {
        "depth": depth + 1,
        "query": "",
        "topicreport": "",
        "sub_query": [],
        "subtopicreport": [],
        "answerreport": "",
        "ref": [],
        "topicreportscore": -1.0,
        "subdeepresearch": [],
    }

    logger.info("Deep research agent: depth=%d, wide=%d", depth, wide)

    is_bid_mode = config.pattern.select_pattern == "bid_generation"
    # 招投标文书是“模板填空型”任务：禁止递归下钻与重生大纲/主题（这些会把任务带偏成泛化文章）
    if is_bid_mode:
        # 仅在本章内重写，不允许改变章节结构/语义目标
        config.maxhop.deepresearch_max_depth = 1
        config.maxhop.deepresearch_force_expand_root = False
        config.maxhop.enable_outline_backtrack = False
        config.maxhop.enable_topic_backtrack = False
        # 标书写作更适合“结构/格式/字段”校验；通用语义一致性容易误判并引发无意义回退
        config.maxhop.enable_paragraph_consistency_check = False
        # 标书合并必须“零信息丢失”：禁用合并模型与最终润色（模型容易做删重/压缩）
        config.maxhop.skip_merge_llm = True
        config.maxhop.skip_final_polish_llm = True
    if is_bid_mode:
        topic_prompt_builder = build_bid_report_topic_prompt
        subtopic_prompt_builder = build_bid_report_subtopic_prompt
        chapter_prompt_builder = build_bid_sub_report_prompt
        merge_prompt_builder = build_bid_report_prompt
    else:
        topic_prompt_builder = build_deep_research_report_topic_prompt
        subtopic_prompt_builder = build_deep_research_report_subtopic_prompt
        chapter_prompt_builder = build_deep_research_sub_report_prompt
        merge_prompt_builder = build_deep_research_report_prompt

    root_sections = (
        max(1, int(config.maxhop.target_total_sections))
        if config.maxhop.target_total_sections is not None
        and config.maxhop.target_total_sections > 0
        else max(1, config.maxhop.deepresearch_root_branching)
    )
    base_constraints = _build_runtime_constraints(config, depth, root_sections)

    # ── 阶段 1: 生成写作框架 ────────────────────────────────
    if depth == 0:
        topic_content = send_request_to_model_dr(
            user_query=user_query,
            history_answer=far_topic,
            references=base_constraints,
            prompt_builder=topic_prompt_builder,
            model_name=config.session.name,
            api_url=config.session.server,
            temperature=config.session.temperature,
            max_tokens=config.session.max_tokens,
            top_p=config.session.top_p,
            repetition_penalty=config.session.repetition_penalty,
        )
        if topic_content is None:
            result_queue.put([False, "<br>模型出现异常……<br>"])
            logger.error("主题策略模型输出为空")
            return deep_research_query_result
        deep_research_query_result["topicreport"] = topic_content
        result_queue.put([False, "<br>🔍 正在分析写作主题，规划写作框架……\n<br>"])
        result_queue.put([False, topic_content])
    else:
        # 二层递归不再重做“复杂主题构思”，直接围绕当前章节进入三维拆解
        topic_content = f"围绕章节“{user_query}”拆解3个子维度"
        deep_research_query_result["topicreport"] = topic_content

    far_topic = topic_content

    # ── 阶段 2: 生成章节大纲 ────────────────────────────────
    def _generate_outline(current_topic: str) -> list[str]:
        sub_topic_content = send_request_to_model(
            user_query=user_query,
            history_answer=current_topic,
            references=base_constraints,
            prompt_builder=subtopic_prompt_builder,
            model_name=config.session.name,
            api_url=config.session.server,
            temperature=config.session.temperature,
            max_tokens=config.session.max_tokens,
            top_p=config.session.top_p,
            repetition_penalty=config.session.repetition_penalty,
        )
        if sub_topic_content is None:
            return []
        parsed = parser_sub_topic_output(sub_topic_content)
        parsed = _sanitize_outline_titles(parsed)
        if depth == 0:
            branch_limit = root_sections
        else:
            branch_limit = max(1, config.maxhop.deepresearch_child_branching)
        return parsed[:branch_limit]

    if is_bid_mode and depth == 0:
        doc_type = _bid_doc_type_hint_local(user_query)
        sub_topic_query = _fixed_bid_outline(doc_type)
        result_queue.put([False, "\n<br>📋 已按模板锁定章节结构，开始逐章生成完整标书……<br>"])
    else:
        sub_topic_query = _generate_outline(topic_content)
    if not sub_topic_query:
        logger.error("章节大纲输出为空")
        if deep_research_query_result["subtopicreport"]:
            deep_research_query_result["answerreport"] = deep_research_query_result["subtopicreport"][0]
        return deep_research_query_result
    logger.info("章节大纲: %r", sub_topic_query)

    result_queue.put([False, "\n<br>📋 写作大纲已生成，下面将逐章展开撰写……<br>"])
    for idx, sub_query in enumerate(sub_topic_query):
        result_queue.put([False, "<br>  " + str(idx + 1) + ". " + sub_query + "<br>"])

    if not sub_topic_query:
        logger.error("章节大纲为空，返回")
        return deep_research_query_result

    # ── 阶段 3: 逐章节撰写（同层可并行）──────────────────────
    def _generate_one_chapter(sub_q: str) -> str:
        chapter_constraints = base_constraints
        if (
            config.maxhop.target_total_words is not None
            and config.maxhop.target_total_words > 0
            and root_sections > 0
        ):
            per_chapter_words = max(600, int(config.maxhop.target_total_words / root_sections))
            chapter_constraints = (
                f"{base_constraints}\n- 当前章节目标字数：约 {per_chapter_words} 字"
                if base_constraints
                else f"- 当前章节目标字数：约 {per_chapter_words} 字"
            )
        # 为了让“思考区”持续滚动（而不是整章一次性出来），在串行时启用流式章节生成
        if config.maxhop.chapter_parallel_workers <= 1:
            result_queue.put([False, f"\n✍️ 正在撰写章节：{sub_q}<br>\n"])
            out = send_request_to_model_dr_streaming(
                user_query=user_query,
                topic_report=sub_q,
                history_answer=None,
                references=chapter_constraints,
                prompt_builder=chapter_prompt_builder,
                result_queue=result_queue,
                progress_prefix="",
                model_name=config.session.name,
                api_url=config.session.server,
                temperature=config.session.temperature,
                max_tokens=config.session.max_tokens,
                top_p=config.session.top_p,
                repetition_penalty=config.session.repetition_penalty,
            )
        else:
            out = send_request_to_model_dr(
                user_query=user_query,
                topic_report=sub_q,
                history_answer=None,
                references=chapter_constraints,
                prompt_builder=chapter_prompt_builder,
                model_name=config.session.name,
                api_url=config.session.server,
                temperature=config.session.temperature,
                max_tokens=config.session.max_tokens,
                top_p=config.session.top_p,
                repetition_penalty=config.session.repetition_penalty,
            )
        if out is None:
            logger.error("章节 '%s' 模型输出为空", sub_q)
            return ""
        return out

    def _rewrite_one_chapter(sub_q: str, retries: int) -> str:
        """对单章做局部重写，最多 retries 次。"""
        chapter_text = _generate_one_chapter(sub_q)
        if not config.maxhop.enable_local_chapter_check:
            return chapter_text
        for attempt in range(retries):
            if _chapter_passes_fast_check(
                chapter_text, config
            ) and _topic_consistent_with_chapter(sub_q, chapter_text):
                return chapter_text
            logger.info(
                "章节 '%s' 局部质检未通过（len=%d, topic_ok=%s），触发重写 (%d/%d)",
                sub_q,
                len((chapter_text or "").strip()),
                _topic_consistent_with_chapter(sub_q, chapter_text),
                attempt + 1,
                retries,
            )
            chapter_text = _generate_one_chapter(sub_q)
        return chapter_text

    n_ch = len(sub_topic_query)
    workers = max(1, min(n_ch, config.maxhop.chapter_parallel_workers))
    if workers > 1 and n_ch > 1:
        result_queue.put(
            [False, f"\n<br>⚡ 并行撰写 {n_ch} 个章节（并发 {workers}）……<br>"]
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            chapter_contents: list[str] = list(pool.map(_generate_one_chapter, sub_topic_query))
    else:
        chapter_contents = [_generate_one_chapter(sq) for sq in sub_topic_query]

    for idx, sub_query in enumerate(sub_topic_query):
        chapter_content = chapter_contents[idx]
        if config.maxhop.enable_local_chapter_check and (
            (not _chapter_passes_fast_check(chapter_content, config))
            or (not _topic_consistent_with_chapter(sub_query, chapter_content))
        ):
            chapter_content = _rewrite_one_chapter(
                sub_query, max(0, config.maxhop.chapter_rewrite_max_retries)
            )
            result_queue.put(
                [False, f"\n🛠️ 章节“{sub_query}”已触发局部重写以修正跑题或补全内容<br>\n"]
            )

        # 逐段一致性校验：对齐用户输入 + 主题框架 + 章节标题 + 全局大纲
        para_ok, pass_ratio = _paragraph_consistency_check(
            chapter_text=chapter_content,
            user_query=user_query,
            topic_content=topic_content,
            chapter_title=sub_query,
            outline_titles=sub_topic_query,
            config=config,
        )
        if not para_ok:
            result_queue.put(
                [
                    False,
                    f"\n⚠️ 章节“{sub_query}”语义一致性判定为“否”，触发回退修复<br>\n",
                ]
            )
            # 第一步：回退到“重生大纲”
            if config.maxhop.enable_outline_backtrack:
                new_outline = _generate_outline(topic_content)
                if new_outline:
                    sub_topic_query = new_outline
                    if idx < len(sub_topic_query):
                        sub_query = sub_topic_query[idx]
                    chapter_content = _generate_one_chapter(sub_query)
                    para_ok, pass_ratio = _paragraph_consistency_check(
                        chapter_text=chapter_content,
                        user_query=user_query,
                        topic_content=topic_content,
                        chapter_title=sub_query,
                        outline_titles=sub_topic_query,
                        config=config,
                    )
            # 第二步：仍不一致，回退到“重生主题+大纲”
            if (not para_ok) and config.maxhop.enable_topic_backtrack:
                new_topic = send_request_to_model_dr(
                    user_query=user_query,
                    history_answer=far_topic,
                    references=base_constraints,
                    prompt_builder=topic_prompt_builder,
                    model_name=config.session.name,
                    api_url=config.session.server,
                    temperature=config.session.temperature,
                    max_tokens=config.session.max_tokens,
                    top_p=config.session.top_p,
                    repetition_penalty=config.session.repetition_penalty,
                )
                if new_topic:
                    topic_content = new_topic
                    deep_research_query_result["topicreport"] = topic_content
                    new_outline = _generate_outline(topic_content)
                    if new_outline:
                        sub_topic_query = new_outline
                        if idx < len(sub_topic_query):
                            sub_query = sub_topic_query[idx]
                        chapter_content = _generate_one_chapter(sub_query)
                        para_ok, pass_ratio = _paragraph_consistency_check(
                            chapter_text=chapter_content,
                            user_query=user_query,
                            topic_content=topic_content,
                            chapter_title=sub_query,
                            outline_titles=sub_topic_query,
                            config=config,
                        )
            if not para_ok:
                logger.warning(
                    "章节 '%s' 一致性修复后模型仍判定为“否”，保留当前最好结果继续流程",
                    sub_query,
                )

        temp_query: SubQueryItem = {"sub_query": sub_query, "ref": None}
        temp_dr_query_result["query"] = sub_query
        temp_dr_query_result["ref"] = []
        temp_dr_query_result["sub_query"].append(copy.deepcopy(temp_query))
        deep_research_query_result["sub_query"].append(copy.deepcopy(temp_query))

        result_queue.put([False, "\n✍️ 正在撰写章节：" + sub_query + "<br>\n"])
        paragraphs = _split_paragraphs_for_preview(chapter_content)
        if not paragraphs:
            preview = (chapter_content or "")[:100]
            remain = max(0, len((chapter_content or "")) - len(preview))
            result_queue.put(
                [
                    False,
                    (
                        f"<br>🧩 进度：第{idx + 1}/{n_ch}章 第1/1段<br>\n"
                        f"{preview}（剩余{remain}字）<br>"
                    ),
                ]
            )
        else:
            total_para = len(paragraphs)
            for p_idx, para in enumerate(paragraphs, start=1):
                preview = para[:100]
                remain = max(0, len(para) - len(preview))
                result_queue.put(
                    [
                        False,
                        (
                            f"<br>🧩 进度：第{idx + 1}/{n_ch}章 第{p_idx}/{total_para}段<br>\n"
                            f"{preview}（剩余{remain}字）<br>"
                        ),
                    ]
                )

        temp_dr_query_result["answerreport"] = chapter_content
        deep_research_query_result["subtopicreport"].append(chapter_content)
        deep_research_query_result["subdeepresearch"].append(
            copy.deepcopy(temp_dr_query_result)
        )

    # bid_generation 模式不允许“继续深入章节”的递归，否则会跑偏成通用写作
    if (not is_bid_mode) and _should_expand_chapter(depth, chapter_content, config):
            result_queue.put([False, f"\n🔎 继续深入章节：{sub_query}<br>\n"])
            deep_research_agent(
                depth + 1,
                idx,
                sub_query,
                far_topic,
                deep_research_query_result["subdeepresearch"][idx],
                result_queue,
                history_message,
                config,
                recursion_state,
            )

    # ── 阶段 4: 合并子章节 → 生成全文 ─────────────────────
    total_merge = len(deep_research_query_result["subdeepresearch"])
    if total_merge > 0:
        result_queue.put([False, "\n🔗 正在整合各章节内容……<br>\n"])
        if is_bid_mode and config.maxhop.skip_merge_llm:
            result_queue.put(
                [
                    False,
                    "🧩 整合策略：标书模式采用“按章节顺序直接拼接（不走合并模型）”，确保不丢信息<br>",
                ]
            )
        for m_idx, ele in enumerate(deep_research_query_result["subdeepresearch"], start=1):
            merge_title = (ele.get("query", "") or f"第{m_idx}章").strip()
            result_queue.put(
                [False, f"🧩 整合进度：第{m_idx}/{total_merge}章《{merge_title}》<br>"]
            )

    history_merge, _ = joint_subquery_report(
        depth, wide, deep_research_query_result["subdeepresearch"]
    )
    total_reference_list = get_total_reference_list(
        deep_research_query_result["subdeepresearch"]
    )

    used_fallback = False
    if config.maxhop.skip_merge_llm:
        logger.info("skip_merge_llm=True，跳过整合模型，直接拼接各章")
        gen_topic_report = "\n\n".join(
            s for s in deep_research_query_result["subtopicreport"] if s
        )
        used_fallback = True
    else:
        gen_topic_report = send_request_to_model_dr(
            user_query=user_query,
            topic_report=deep_research_query_result["topicreport"],
            history_answer=history_merge,
            references=base_constraints,
            prompt_builder=merge_prompt_builder,
            model_name=config.session.name,
            api_url=config.session.server,
            temperature=config.session.temperature,
            max_tokens=config.session.max_tokens,
            top_p=config.session.top_p,
            repetition_penalty=config.session.repetition_penalty,
        )
        if not gen_topic_report:
            logger.warning("整合模型返回空，回退为直接拼接各章节内容")
            gen_topic_report = "\n\n".join(
                s for s in deep_research_query_result["subtopicreport"] if s
            )
            used_fallback = True

    # ── 质量把控（仅 depth==0 且非 fallback）──────────────
    match_score = -1.0
    if depth == 0 and not used_fallback and not config.maxhop.skip_confidence_check:
        confidence_req = send_request_to_model(
            user_query=user_query,
            history_answer=gen_topic_report,
            references=None,
            prompt_builder=build_confidence_prompt,
            model_name=config.confidence.name,
            api_url=config.confidence.server,
            temperature=config.confidence.temperature,
            max_tokens=config.confidence.max_tokens,
            top_p=config.confidence.top_p,
            repetition_penalty=config.confidence.repetition_penalty,
        )
        if confidence_req is None:
            confidence_req = ""
        confidence_flag, match_score = parser_confidence(
            confidence_req, config.maxhop.deepresearch_conf_score
        )

        retry = 0
        while (not confidence_flag or match_score < config.maxhop.deepresearch_conf_score) and retry < 2:
            logger.info("全文质量不达标 (%.2f)，重新生成 (retry=%d)", match_score, retry)
            gen_topic_report = send_request_to_model_dr(
                user_query=user_query,
                topic_report=deep_research_query_result["topicreport"],
                history_answer=history_merge,
                references=base_constraints,
                prompt_builder=merge_prompt_builder,
                model_name=config.session.name,
                api_url=config.session.server,
                temperature=config.session.temperature,
                max_tokens=config.session.max_tokens,
                top_p=config.session.top_p,
                repetition_penalty=config.session.repetition_penalty,
            )
            confidence_req = send_request_to_model(
                user_query=user_query,
                history_answer=gen_topic_report,
                references=None,
                prompt_builder=build_confidence_prompt,
                model_name=config.confidence.name,
                api_url=config.confidence.server,
                temperature=config.confidence.temperature,
                max_tokens=config.confidence.max_tokens,
                top_p=config.confidence.top_p,
                repetition_penalty=config.confidence.repetition_penalty,
            )
            if confidence_req is None:
                confidence_req = ""
            confidence_flag, match_score = parser_confidence(
                confidence_req, config.maxhop.deepresearch_conf_score
            )
            retry += 1

    deep_research_query_result["answerreport"] = gen_topic_report or ""
    deep_research_query_result["ref"] = total_reference_list
    deep_research_query_result["topicreportscore"] = match_score
    return deep_research_query_result


# ---------------------------------------------------------------------------
#  辅助函数
# ---------------------------------------------------------------------------

def get_total_reference_list(
    deep_research_query_result: list[DeepResearchResult],
) -> list[ReferenceItem]:
    total_reference_list: List[ReferenceItem] = []
    for ele in deep_research_query_result:
        total_reference_list.extend(ele["ref"])
    return total_reference_list


def _should_expand_chapter(depth: int, chapter_content: str, config: AppConfig) -> bool:
    """
    是否对当前章节继续下钻。
    - 根层可强制展开一次（便于“深入思考”）
    - 子层按章节信息量判断，内容不足则继续展开
    """
    if depth + 1 >= config.maxhop.deepresearch_max_depth:
        return False
    min_chars = max(200, config.maxhop.deepresearch_expand_min_chars)
    content_len = len((chapter_content or "").strip())
    # 根层可选“倾向下钻”，但不再强制，避免所有章节都被递归重规划导致跑偏
    if depth == 0 and config.maxhop.deepresearch_force_expand_root:
        return content_len < int(min_chars * 1.2)
    return content_len < min_chars


def joint_subquery_report(
    depth: int, wide: int, deep_research_query_result: list[DeepResearchResult]
) -> tuple[str, str]:
    history_merge = "以下是各章节的草稿内容：\n"
    for idx, ele in enumerate(deep_research_query_result):
        history_merge += (
            f"【第{idx+1}章节】：{ele['query']}\n"
            f"【内容】：{ele['answerreport']}\n\n"
        )
    return history_merge, ""


