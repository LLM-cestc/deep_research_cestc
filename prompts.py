# -*- coding: utf-8 -*-  # noqa: UP009
"""
系统 prompts 模块：定制化各种prompts

Author: wjianxz
Date: 2025-11-13
"""

from deep_research.protocal import ReferenceList
from typing import Union
import logging
import re

logger = logging.getLogger(__name__)

# 整合全文时传入的章节合并稿长度上限（字符数）。
# 历史上为适配极小上下文曾截成 1200 字，会导致合并模型只能写“摘要”，最终远短于各章之和。
# 大上下文模型可拉满；仅作防止单次请求过大的安全阀。
_MERGE_INPUT_MAX_CHARS = 120_000


def _constraints_from_references(references: Union[ReferenceList, str, None]) -> str:
    """
    从 references 中提取运行时写作约束文本。
    约定：当 references 为 str 时，作为约束文本透传给 prompt。
    """
    if isinstance(references, str):
        return references.strip()
    return ""


# ---------------------------------------------------------------------------
#  润色 / 格式化
# ---------------------------------------------------------------------------

def build_deep_research_beautiful_format_prompt(
    user_input: str,
    history_answer: Union[str, None],
    references: Union[ReferenceList, None],
) -> str:
    """
    构建后处理润色提示词，用于对模型生成的原始回答进行语言规范化、
    逻辑结构优化与格式美化。严格保持原意不变。
    """
    if not user_input or not user_input.strip():
        user_input = "（无内容）"

    if references is None:
        ref = ""
    else:
        ref = ""
        for idx, ele in enumerate(references):
            if idx >= 32:
                break
            if len(ele["title"]) > 3:
                ref += "[" + ele["title"] + "](" + ele["url"] + ")" + "\n\n"
        logger.info("build_deep_research_beautiful_format_prompt ref: %s", ref)

    return (
        "请将下面文章整理成可直接发布的 Markdown 中文长文。\n"
        "要求：只润色，不删减核心信息，不缩写，不压缩篇幅；"
        "优化标题层级、段落衔接、用词和标点，让文章更顺、更完整、更像正式成稿。\n"
        "若原文已经很长，优先保留内容完整性，再做轻量润色。"
        "不要输出说明话术，直接输出正文。\n\n"
        f"原文：\n{user_input.strip()}\n\n"
        f"参考文献：\n{ref}\n"
    )


# ---------------------------------------------------------------------------
#  Chat 对话模式
# ---------------------------------------------------------------------------

def build_deep_research_session_chat_prompt(
    user_input: str,
    history_answer: Union[str, None],
    references: Union[ReferenceList, None],
) -> str:
    """构建 Chat 模式的通用写作问答提示词。"""
    if not history_answer:
        history_text = "无相关历史回答。"
    else:
        history_text = history_answer

    prompt = (
        "你是一名专业中文写作助手，擅长解释问题、扩写观点、组织结构、输出高质量长文。\n\n"
        "请根据【对话历史】和【用户问题】，给出清晰、具体、可直接使用的回答。\n\n"
        "【回答要求】\n"
        "1. 先直接回答，再展开说明；\n"
        "2. 内容要具体，不要空话；\n"
        "3. 结构清晰，可分段或分点；\n"
        "4. 若适合扩写，就补充原因、做法、例子、影响；\n"
        "5. 不要编造事实，不要输出模板腔和套话。\n\n"
        f"【对话历史】\n{history_text}\n\n"
        f"【用户问题】\n{user_input}"
    )
    logger.info("chat mode prompt: %s", prompt)
    return prompt.strip()


def build_confidence_prompt(
    user_input: str, history_answer: str, references: str
) -> str:
    """构建用于评估答案置信度的提示词（0-1 评分）。"""
    answer_excerpt = (history_answer or "").strip()[:1200]
    bid_like = any(
        k in (user_input or "")
        for k in ["招标", "投标", "评标", "采购", "采购人", "供应商", "代理机构", "中标"]
    )
    if bid_like:
        return (
            f"【用户问题】：{user_input}\n"
            f"【系统回答】：{answer_excerpt}\n"
            "你是政府采购文书质量审查员。请按下列标准给出置信度（0-1）：\n"
            "1) 完整性检查：是否覆盖该章节/该文书应响应的核心点（含必要表格字段）。若存在明显漏项，总得分≤0.30。\n"
            "2) 实质性偏离检查：若对★条款存在负偏离或模糊承诺（仅写“满足/符合”而无具体内容），总得分=0.00。\n"
            "3) 数值一致性检查：金额、日期、比例、期限、编号等是否与输入一致。若出现编造或冲突，总得分=0.00。\n"
            "4) 格式合规性检查：标题层级、编号体系、表格列结构是否稳定。若关键结构错误，总得分≤0.20。\n"
            "5) 占位符检查：关键字段仍为{{请补充[...]}}且未给替代证据时，总得分≤0.40。\n"
            "请仅输出一行，格式为：总得分：X.XX（X.XX 为 0 到 1 之间的浮点数）。"
        ).strip()
    prompt = (
        f"【用户问题】：{user_input}\n"
        f"【系统回答】：{answer_excerpt}\n"
        "请根据以下标准评估【系统回答】的置信度：\n"
        "- 是否有明确、逻辑严谨的推理逻辑，回答思路清晰，格式完整；\n"
        "- 是否准确、完整地回答了用户的问题？如果没有回答用户问题直接输出总得分：0.00\n"
        "- 如果用户问题本身非常简单属于简单对话聊天，系统给出合理回复也认为是合理的；\n"
        "- 如果出现重复输出、乱码、多语言混合、非中文等，输出总得分：0.00\n"
        "请仅输出一行，格式为：总得分：X.XX（X.XX 为 0 到 1 之间的浮点数，例如：总得分：0.85）。"
    )
    return prompt.strip()


def build_chapter_consistency_judgment_prompt(
    user_input: str, history_answer: str, references: str
) -> str:
    """
    章节一致性语义判定 Prompt。
    只允许输出“是”或“否”。
    """
    return (
        "你是写作一致性审核器。请基于给定信息判断："
        "【章节正文】是否与【用户原始需求】、【当前主题框架】、【当前章节标题】、【当前大纲标题】语义一致。\n"
        "判定规则：\n"
        "1) 主旨是否围绕当前章节标题展开；\n"
        "2) 是否明显跑题到其他一级主题；\n"
        "3) 是否与用户需求主线冲突。\n\n"
        "如果一致，输出：是\n"
        "如果不一致，输出：否\n"
        "禁止输出任何解释、标点、额外文本。\n\n"
        f"{user_input}"
    )


# ---------------------------------------------------------------------------
#  Deep Research 写作管线（精简版，适配小上下文模型）
# ---------------------------------------------------------------------------

def build_deep_research_report_topic_prompt(
    initial_query: str,
    topic_report: Union[str, None],
    history_answer: Union[str, None],
    references: Union[ReferenceList, str, None],
) -> str:
    """阶段 1：生成写作框架。"""
    constraints = _constraints_from_references(references)
    prompt = (
        f"请为以下主题规划写作框架，给出核心维度标题。\n"
        f"输出约束：仅输出“维度标题”，每行一个短标题；不要输出“总体思路/研究方法/进度建议/写作特色/结语”等元信息。\n"
        f"根层建议 6 个维度；若是二级拆解请控制为 3 个维度。\n\n"
        f"写作主题：{initial_query}\n\n"
        f"请输出维度标题："
    )
    if constraints:
        prompt += f"\n\n【写作约束】\n{constraints}"
    return prompt.strip()


def build_deep_research_report_subtopic_prompt(
    initial_query: str, history_answer: str, references: Union[ReferenceList, None]
) -> str:
    """阶段 2：生成章节大纲标题。"""
    constraints = _constraints_from_references(references)
    prompt = (
        f"请为以下主题生成章节标题，每行一个。\n"
        f"输出约束：不要编号，不要解释，不要出现“研究方法/进度建议/总体构思/写作特色”等元内容。\n"
        f"标题要具体、有递进关系，且仅围绕当前主题，不得引入无关主线。\n\n"
        f"写作主题：{initial_query}\n\n"
        f"请输出章节标题："
    )
    if constraints:
        prompt += f"\n\n【写作约束】\n{constraints}"
    return prompt.strip()


def build_deep_research_sub_report_prompt(
    initial_query: str,
    topic_report: Union[str, None],
    history_answer: Union[str, None],
    references: Union[ReferenceList, str, None],
) -> str:
    """阶段 3：生成单个章节正文。"""
    chapter_title = _normalize_chapter_title(topic_report, initial_query or "本章")
    constraints = _constraints_from_references(references)
    prompt = (
        f"请围绕以下章节写正文，硬性要求：每章不少于1000字（建议1000-1400字），至少4段，内容要具体、充实、有分析，避免空话和重复。\n"
        f"可写定义、背景、原因、做法、案例、影响、建议，但不要写整篇文章总结。\n"
        f"主题一致性要求：每一段都必须服务于“{chapter_title}”，不要引入新的一级章节主题；"
        f"若需要扩展，只能在本章节范围内展开，不得跑题。\n"
        f"如果正文未达到1000字，请继续补写，直到满足字数要求。\n\n"
        f"文章主题：{initial_query}\n"
        f"仅允许关注这一条确切章节标题：{chapter_title}\n"
        "忽略 topic_report 中可能夹带的其他标题或框架文本。\n"
        f"本章节标题：{chapter_title}\n\n"
        f"请直接输出本章节正文，并以“## {chapter_title}”作为首行标题："
    )
    if constraints:
        prompt += f"\n\n【写作约束】\n{constraints}"
    return prompt.strip()


def build_deep_research_report_prompt(
    initial_query: str,
    topic_report: Union[str, None],
    history_answer: Union[str, None],
    references: Union[ReferenceList, str, None],
) -> str:
    """阶段 4：整合各章节生成完整文章。"""
    constraints = _constraints_from_references(references)
    if history_answer is None:
        history_answer = ""
    history_snippet = history_answer.strip()

    prompt = (
        f"请根据以下章节草稿，整合成一篇结构完整、语言流畅、逻辑连贯的中文长文。"
        f"草稿已包含各章正文，请尽量保留信息量，不要压缩成短摘要；"
        f"可做删重、过渡与总起总结，但正文篇幅应接近各章之和。\n\n"
        f"写作主题：{initial_query}\n\n"
    )
    if history_snippet:
        draft = history_snippet[:_MERGE_INPUT_MAX_CHARS]
        prompt += f"各章节草稿（约 {len(history_snippet)} 字，整合时请以全文为准）：\n{draft}\n\n"
        if len(history_snippet) > _MERGE_INPUT_MAX_CHARS:
            prompt += (
                f"（说明：草稿超过 {_MERGE_INPUT_MAX_CHARS} 字，仅截断传入；"
                f"如需全文整合请提高 prompts._MERGE_INPUT_MAX_CHARS。）\n\n"
            )
    if constraints:
        prompt += f"【写作约束】\n{constraints}\n\n"
    prompt += "请直接输出完整文章正文："
    return prompt.strip()


# ---------------------------------------------------------------------------
#  AI评标（招标书）专用写作管线
# ---------------------------------------------------------------------------


def _bid_doc_type_hint(user_text: str) -> str:
    """
    从用户输入中粗略判断文档类型：
    - 招标文件 / 投标文件 / 评标规则
    """
    t = (user_text or "").strip()
    if any(k in t for k in ["投标", "投标文件", "投标书"]):
        return "投标文件"
    if any(k in t for k in ["评标", "评审", "评分规则", "打分规则"]):
        return "评标规则"
    return "招标文件"


def _bid_common_rules() -> str:
    return (
        "【总原则】\n"
        "1) 这是“模板填空”任务：必须将内容填入标准章节与字段，不得任意改结构。\n"
        "2) 严禁编造关键事实（审批文号、金额、比例、期限、单位、资质、联系人等）。\n"
        "3) 用户未提供的字段必须输出占位符：{{请补充[字段名]}}。\n"
        "4) 输出必须采用 Markdown，表格用 Markdown 表格语法，便于后续程序写入文档模板。\n"
        "5) 除正文外，不要输出解释、备注、元评论。\n"
        "6) 数值守恒原则：金额、数量、比例、日期、期限、编号必须与输入原文一致；"
        "若原文未给确定值，只能输出 {{变量}} 或 {{请补充[字段名]}}，严禁推算或脑补。\n"
        "7) 表格列数锁定：表格列数和表头结构必须稳定；缺值用 {{请补充[字段名]}} 填单元格，"
        "严禁改列、并列、合并列。\n"
        "8) 格式锁定：严禁改写编号体系（如“一、二、三”或“1.2.3”）；"
        "严禁重构模板目录；润色仅限错别字、标点、语句顺滑。\n"
    )


def _bid_tender_chapter_rules() -> str:
    return (
        "【招标文件章节硬约束】\n"
        "必须按以下顺序输出一级章节（标题不可改）：\n"
        "第一章 投标邀请\n"
        "第二章 采购需求\n"
        "第三章 投标人须知\n"
        "第四章 评标办法\n\n"
        "第五章 合同条款及格式\n"
        "第六章 投标文件格式\n\n"
        "并满足：\n"
        "- 第一章包含：项目概述、资格要求、获取文件、提交截止、公告媒介、联系方式\n"
        "- 第二章包含：项目概况、技术规格（表格）、商务要求\n"
        "- 第三章包含：名词解释、须知前附表、电子投标要求\n"
        "- 第四章包含：评标方法、评审程序、无效投标情形\n"
        "- 第五章包含：合同协议书条款、履约期限、付款与验收、违约责任、争议解决等（缺失处用占位符）\n"
        "- 第六章包含：投标函/报价表/响应表/承诺函/资格文件清单等格式化结构（占位符）\n"
        "- 对“★”条款注明实质性要求（不满足可致无效），对“▲”条款注明影响评分\n"
    )


def _bid_response_chapter_rules() -> str:
    return (
        "【投标文件章节硬约束】\n"
        "目录需兼容《投标模版》“格式一~格式二十三”体系（如投标函、开标一览表、分项报价表、"
        "实质性响应一览表、技术和服务要求响应表、商务条件响应表、履约担保等）。\n"
        "必须按标准投标文件逻辑组织，至少包含：\n"
        "1. 投标函\n"
        "2. 开标一览表\n"
        "3. 分项报价表\n"
        "4. 资格证明文件\n"
        "5. 技术和服务要求响应表\n"
        "6. 商务条件响应表\n"
        "7. 实施方案/服务方案\n"
        "8. 承诺函与附件占位\n\n"
        "并满足：\n"
        "- 实质性条款（★）不得负偏离\n"
        "- 响应内容必须“逐条镜像”招标要求，不得泛泛而谈\n"
        "- 数值字段尽量具体；缺失值输出 {{请补充[...]}}，禁止臆造\n"
        "- 开标一览表/分项报价表应与电子投标客户端字段保持一致，避免列数错位\n"
    )


def build_bid_beautiful_format_prompt(
    user_input: str,
    history_answer: Union[str, None],
    references: Union[ReferenceList, None],
) -> str:
    if not user_input or not user_input.strip():
        user_input = "（无内容）"
    doc_type = _bid_doc_type_hint(user_input)
    chapter_rules = (
        _bid_response_chapter_rules() if doc_type == "投标文件" else _bid_tender_chapter_rules()
    )
    return (
        "你是政府采购文书终稿编辑专家。\n"
        f"目标文档类型：{doc_type}\n"
        f"{_bid_common_rules()}\n"
        f"{chapter_rules}\n"
        "任务：将以下草稿整理为最终可提交版本，统一术语、层级、编号、表格格式；\n"
        "严禁修改标题编号体系和表格结构；\n"
        "保留全部关键事实与条款，不得漏项，不得改写核心法律/商务含义。\n"
        "只输出正文。\n\n"
        f"{user_input.strip()}"
    )


def build_bid_report_topic_prompt(
    initial_query: str,
    topic_report: Union[str, None],
    history_answer: Union[str, None],
    references: Union[ReferenceList, str, None],
) -> str:
    constraints = _constraints_from_references(references)
    doc_type = _bid_doc_type_hint(initial_query)
    chapter_rules = (
        _bid_response_chapter_rules() if doc_type == "投标文件" else _bid_tender_chapter_rules()
    )
    prompt = (
        "你是政府采购文书规划助手。\n"
        f"目标文档类型：{doc_type}\n"
        f"{_bid_common_rules()}\n"
        f"{chapter_rules}\n"
        "任务：先提取关键字段，再生成一级目录。\n"
        "关键字段至少包括：项目名称、项目编号/计划编号、采购人、代理机构、预算金额、资金来源、服务期限、资格条件、评标方式。\n"
        "输出约束：\n"
        "1) 必须输出一级目录标题（每行一个）；\n"
        "2) 不得输出研究方法、进度建议、写作说明等元内容；\n"
        "3) 即使关键字段缺失，也不要中断输出：请在目录相关位置或字段处直接保留占位符 {{请补充[字段名]}}；\n"
        "4) 允许在末尾附加“待补充信息清单”（可选），但不得以清单替代正文目录。\n\n"
        f"输入材料：\n{initial_query}\n\n"
        "请输出目录（并保留缺失字段占位符）："
    )
    if constraints:
        prompt += f"\n\n【写作约束】\n{constraints}"
    return prompt.strip()


def build_bid_report_subtopic_prompt(
    initial_query: str, history_answer: str, references: Union[ReferenceList, None]
) -> str:
    constraints = _constraints_from_references(references)
    doc_type = _bid_doc_type_hint(initial_query)
    chapter_rules = (
        _bid_response_chapter_rules() if doc_type == "投标文件" else _bid_tender_chapter_rules()
    )
    prompt = (
        "你是政府采购文书结构化拆解助手。\n"
        f"目标文档类型：{doc_type}\n"
        f"{_bid_common_rules()}\n"
        f"{chapter_rules}\n"
        "请基于以下材料与目录框架，生成可直接写作的章节标题（每行一个）。\n"
        "要求：严格贴合模板，不要泛化分析标题；不要编号，不要解释。\n\n"
        f"输入材料：\n{initial_query}\n\n"
        f"目录框架：\n{history_answer}\n\n"
        "请输出章节标题："
    )
    if constraints:
        prompt += f"\n\n【写作约束】\n{constraints}"
    return prompt.strip()


def build_bid_sub_report_prompt(
    initial_query: str,
    topic_report: Union[str, None],
    history_answer: Union[str, None],
    references: Union[ReferenceList, str, None],
) -> str:
    chapter_title = _normalize_chapter_title(topic_report, "本章")
    constraints = _constraints_from_references(references)
    doc_type = _bid_doc_type_hint(initial_query)
    chapter_rules = (
        _bid_response_chapter_rules() if doc_type == "投标文件" else _bid_tender_chapter_rules()
    )
    global_constraints = _extract_bid_global_constraints(
        f"{initial_query or ''}\n{history_answer or ''}\n{constraints or ''}"
    )
    prompt = (
        f"你是{doc_type}章节撰写专家。\n"
        f"{_bid_common_rules()}\n"
        f"{chapter_rules}\n"
        "请撰写当前章节正文，要求：\n"
        "1) 逐条响应模板对应要求，避免泛泛论述；\n"
        "2) 关键信息使用明确字段值；缺失值用 {{请补充[字段名]}}；\n"
        "3) 涉及技术/商务响应时优先使用表格；\n"
        "4) 禁止跑题，不引入模板外无关章节；\n"
        "5) 请严格遵守【全局关键约束】中的数值，本章任何数值不得与其冲突。\n\n"
        f"原始招标材料：\n{initial_query}\n\n"
        f"【全局关键约束】\n{global_constraints}\n\n"
        f"仅允许关注这一条确切章节标题：{chapter_title}\n"
        "忽略输入中可能出现的其他章节标题或框架句。\n"
        f"当前章节：{chapter_title}\n\n"
        f"请直接输出章节正文，并以“## {chapter_title}”作为首行标题："
    )
    if constraints:
        prompt += f"\n\n【写作约束】\n{constraints}"
    return prompt.strip()


def build_bid_report_prompt(
    initial_query: str,
    topic_report: Union[str, None],
    history_answer: Union[str, None],
    references: Union[ReferenceList, str, None],
) -> str:
    constraints = _constraints_from_references(references)
    doc_type = _bid_doc_type_hint(initial_query)
    chapter_rules = (
        _bid_response_chapter_rules() if doc_type == "投标文件" else _bid_tender_chapter_rules()
    )
    history_snippet = (history_answer or "").strip()
    prompt = (
        f"你是{doc_type}整编专家。\n"
        f"{_bid_common_rules()}\n"
        f"{chapter_rules}\n"
        "请将以下各章节草稿整合为一份完整文书。\n"
        "要求：章节顺序严格符合模板；条款与术语统一；关键信息一致可追溯；"
        "保留必要的法律声明重复：若“投标函/资格声明/承诺函”多次出现，需保留全部实例；"
        "仅对“技术方案、背景描述”等说明性文本做语义去重，不得删减实质条款。\n\n"
        f"原始招标材料：\n{initial_query}\n\n"
    )
    if history_snippet:
        draft = history_snippet[:_MERGE_INPUT_MAX_CHARS]
        prompt += f"章节草稿：\n{draft}\n\n"
    if constraints:
        prompt += f"【写作约束】\n{constraints}\n\n"
    prompt += (
        "请直接输出完整正文。\n"
        "若发现关键字段缺失，请在正文末尾增加“待补充信息清单”并使用 {{请补充[字段名]}}。"
    )
    return prompt.strip()


def _normalize_chapter_title(raw_title: Union[str, None], fallback: str) -> str:
    """
    章节标题归一化：
    - 避免把整段大纲/多行文本当成单章标题
    - 取首个有效行并去掉常见前缀符号
    """
    text = (raw_title or "").strip()
    if not text:
        return fallback.strip() or "本章"

    for line in text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        candidate = re.sub(r"^\s*[#>\-\*\d\.\)\(（）：:]+\s*", "", candidate)
        if not candidate:
            continue
        if any(x in candidate for x in ["请输出", "写作主题", "目录框架", "输入材料"]):
            continue
        return candidate[:120].strip()

    return fallback.strip() or "本章"


def _extract_bid_global_constraints(source_text: str) -> str:
    """
    从输入材料中抽取跨章节共享的硬参数，降低并发写作时章节冲突风险。
    """
    text = (source_text or "").strip()
    if not text:
        return (
            "- 预算金额：{{请补充[预算金额]}}\n"
            "- 最高限价：{{请补充[最高限价]}}\n"
            "- 服务期限/工期：{{请补充[服务期限或工期]}}\n"
            "- 质保期：{{请补充[质保期]}}\n"
            "- 投标有效期：{{请补充[投标有效期]}}"
        )

    patterns = [
        ("项目编号", r"(项目编号[^。\n；;]{0,80})"),
        ("采购计划编号", r"(计划编号[^。\n；;]{0,80})"),
        ("预算金额", r"((预算金额|采购预算)[^。\n；;]{0,80})"),
        ("最高限价", r"(最高限价[^。\n；;]{0,80})"),
        ("资金来源", r"(资金来源[^。\n；;]{0,80})"),
        ("出资比例", r"(出资比例[^。\n；;]{0,80})"),
        ("服务期限/工期", r"((服务期限|工期|交货期|履约期限)[^。\n；;]{0,80})"),
        ("质保期", r"((质保期|保修期)[^。\n；;]{0,80})"),
        ("投标有效期", r"(投标有效期[^。\n；;]{0,80})"),
    ]

    lines = []
    for label, pattern in patterns:
        m = re.search(pattern, text)
        if m:
            value = re.sub(r"\s+", " ", m.group(1)).strip(" ：:;；，,")
            lines.append(f"- {label}：{value}")

    if not lines:
        lines = [
            "- 预算金额：{{请补充[预算金额]}}",
            "- 最高限价：{{请补充[最高限价]}}",
            "- 服务期限/工期：{{请补充[服务期限或工期]}}",
            "- 质保期：{{请补充[质保期]}}",
            "- 投标有效期：{{请补充[投标有效期]}}",
        ]
    return "\n".join(lines)
