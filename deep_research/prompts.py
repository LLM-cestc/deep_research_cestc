# -*- coding: utf-8 -*-  # noqa: UP009
"""
系统 prompts 模块：定制化各种 prompts。

**deep_thinking 首轮链路：**
- 检索：`build_retrieval_plan_prompt` → 一次 JSON（kb_query / bm25_keywords / web_queries）
- 首答：`build_deep_research_session_rag_prompt`（法条与联网同等、仔细参考 [Wn] 摘要 + 引用标注，不强制固定段落结构）
- 联网：`retrieve_web_evidence` → extract 模型筛重要段落、摘录原文、去无关/控超长 → 主答仔细阅读材料
- 联网检索词仅在 web 层做规范化，不再单独调 LLM

**当前 `deep_research` 包内实际会 import 并使用的 build_*（勿删改签名）：**
- `agent_deep_search`：build_beautiful_format_noref_prompt, build_beautiful_format_prompt,
  build_beautiful_format_rag_prompt, build_deep_research_session_chat_prompt,
  build_deep_research_session_rag_prompt, build_retrieval_plan_prompt,
  build_session_prompt, build_deep_research_beautiful_format_prompt
- `plan_pipline_search`：build_deep_research_report_topic_prompt,
  build_deep_research_report_subtopic_prompt, build_deep_research_report_prompt,
  build_deep_research_sub_report_prompt, build_sub_query_prompt, build_confidence_prompt,
  build_deep_research_prompt, build_deep_research_noref_prompt
- `assemble_rag_context`：build_extract_info_prompt
- `utils`（质量评估，惰性 import）：build_quality_judgment_prompt

Author: wjianxz
Date: 2025-11-13
"""

from deep_research.protocal import ReferenceList
from typing import Union
import logging

logger = logging.getLogger(__name__)  # 自动继承 root logger 的 handlers


def build_deep_research_beautiful_format_prompt(
    user_input: str,
    history_answer: Union[str, None],
    references: Union[ReferenceList, None],
) -> str:
    """
    构建面向大语言模型的后处理润色提示词（Post-editing Prompt），
    用于对模型生成的原始回答进行语言规范化、逻辑结构优化与格式美化。

    本函数旨在提升输出文本的可读性、专业性与结构性，同时严格约束内容保真度。
    尽管接口保留 `references` 参数以维持与检索增强生成（RAG）流水线的一致性，
    但在当前润色阶段，参考文献不参与语义校验或内容修正。

    润色过程遵循以下原则：
        1. **形式优化**：调整段落结构、规范标点使用、消除语法瑕疵、提升语句流畅度；
        2. **内容保真**：禁止增删事实性信息、不得篡改技术细节或逻辑结论；
        3. **风格统一**：采用正式书面语体，契合学术或技术文档表达规范；
        4. **结构分层**：根据内容逻辑，合理组织多级标题体系，增强可读性；
        5. **输出纯净**：返回结果应为可直接使用的最终文本，不含解释性前缀或元说明。

    Args:
        user_input (str):
            大语言模型生成的原始回答文本，作为润色对象。
        references (list[dict], optional):
            检索所得参考文献列表，每个元素应包含 "title" 和 "url" 字段。
            本阶段仅用于接口兼容，实际润色逻辑中不予使用。
            默认值为 None。

    Returns:
        str:
            符合上述规范的结构化提示词，可直接输入至润色模型以生成优化后文本。

    Notes:
        若输入为空或仅含空白字符，将自动替换为占位符“（无内容）”，
        以确保提示词结构完整性并避免模型异常行为。
    """
    if not user_input or not user_input.strip():
        user_input = "（无内容）"

    if references is None:
        ref = ""
    else:
        ### 参考文献列表
        # ref = "#### 参考文献列表"
        ref = ""
        for idx, ele in enumerate(references):
            if idx >= 32:
                break
            title = str(ele.get("title", "")).strip()
            url = str(ele.get("url", "")).strip()
            if len(title) > 3 and url:
                ref += f"[{title}]({url})\n\n"
        logger.info(f"build_deep_research_beautiful_format_prompt deep research 最后的 ref: {ref}")

    return (
        "# 文本后处理任务指令\n\n"
        "你作为专业学术编辑，请对下述由大语言模型生成的原始回答执行规范化润色。\n"
        "润色目标是在**严格保持原意不变**的前提下，提升文本的学术性、逻辑性、结构性与可读性，不要在结尾添加任何额外内容。\n\n"
        "## 处理规范\n\n"
        "请遵循以下准则进行编辑：\n\n"
        "1. **格式与语言优化**  \n"
        "   - 调整段落划分，确保语义连贯；  \n"
        "   - 规范使用中文全角标点（如，。：“”‘’）；  \n"
        "   - 修正语法错误、搭配不当及语序问题；  \n"
        "   - 提升语句流畅度与书面表达严谨性。\n\n"
        "2. **内容保真约束**  \n"
        "   - 不得新增、删除或修改任何事实性陈述、数据、法律条文或技术细节；  \n"
        "   - 禁止改变原始推论逻辑、结论或专业判断；  \n"
        "   - 不得引入外部知识、主观评价或推测性内容。\n\n"
        "3. **结构层次优化**  \n"
        "   - 若原始内容包含多个主题或逻辑模块，请使用层级化标题组织内容；  \n"
        "   - 推荐采用 Markdown 标题语法，从二级标题开始：二级标题 `##` 表示主议题，三级标题 `###` 表示子议题，四级标题 `####` 表示细分要点；\n"
        "   - 标题命名应简洁、准确，反映该部分核心内容；  \n"
        "   - 避免过度分层（一般不超过三级），确保结构清晰而不冗余。\n\n"
        "4. **输出要求**  \n"
        "   - 最终文本应为可直接用于技术报告、学位论文、法律意见书或科研文档的成品；  \n"
        "   - 不包含任何引导语（如“润色结果如下：”）、解释性语句或元注释；  \n"
        "   - 若提供参考文献列表，请在正文结束后单独列出 `#### 参考文献`，每项保持 `[标题](URL)` 格式；若未提供，则不要输出空参考文献标题。\n\n"
        "## 原始输入\n\n"
        f"{user_input.strip()}\n\n"
        f"{'#### 参考文献：' if ref.strip() else ''}\n\n"
        f"{ref}\n\n"
    )


def build_beautiful_format_noref_prompt(
    user_input: str,
    history_answer: Union[str, None],
    references: Union[ReferenceList, str, None],
) -> str:
    """
    构建面向大语言模型的后处理润色提示词（Post-editing Prompt），
    用于对模型生成的原始回答进行语言规范化、逻辑结构优化与格式美化。

    本函数旨在提升输出文本的可读性、专业性与结构性，同时严格约束内容保真度。
    尽管接口保留 `references` 参数以维持与检索增强生成（RAG）流水线的一致性，
    但在当前润色阶段，参考文献不参与语义校验或内容修正。

    润色过程遵循以下原则：
        1. **形式优化**：调整段落结构、规范标点使用、消除语法瑕疵、提升语句流畅度；
        2. **内容保真**：禁止增删事实性信息、不得篡改技术细节或逻辑结论；
        3. **风格统一**：采用正式书面语体，契合学术或技术文档表达规范；
        4. **结构分层**：根据内容逻辑，合理组织多级标题体系，增强可读性；
        5. **输出纯净**：返回结果应为可直接使用的最终文本，不含解释性前缀或元说明。

    Args:
        user_input (str):
            大语言模型生成的原始回答文本，作为润色对象。
        references (list[str], optional):
            检索所得参考文献列表。本阶段仅用于接口兼容，实际润色逻辑中不予使用。
            默认值为 None。

    Returns:
        str:
            符合上述规范的结构化提示词，可直接输入至润色模型以生成优化后文本。

    Notes:
        若输入为空或仅含空白字符，将自动替换为占位符“（无内容）”，
        以确保提示词结构完整性并避免模型异常行为。
    """
    if not user_input or not user_input.strip():
        user_input = "（无内容）"

    ref_text = ""
    if references is None:
        ref_text = ""
    elif isinstance(references, str):
        if len(references.strip()) > 5:
            # 保留完整内容，并明确编号用于引用
            ref_text = references
    logger.info(f"prompts deep research 最后的 no ref_text: {ref_text}")
    return (
        "# 文本后处理任务指令\n\n"
        "你作为专业学术编辑，请对下述由大语言模型生成的原始回答执行规范化润色。\n"
        "润色目标是在**严格保持原意不变**的前提下，提升文本的学术性、逻辑性、结构性与可读性。\n\n"
        "## 处理规范\n\n"
        "请遵循以下准则进行编辑：\n\n"
        "1. **格式与语言优化**  \n"
        "   - 调整段落划分，确保语义连贯；  \n"
        "   - 规范使用中文全角标点（如，。：“”‘’）；  \n"
        "   - 修正语法错误、搭配不当及语序问题；  \n"
        "   - 提升语句流畅度与书面表达严谨性。\n\n"
        "2. **内容保真约束**  \n"
        "   - 不得新增、删除或修改任何事实性陈述、数据、法律条文或技术细节；  \n"
        "   - 禁止改变原始推论逻辑、结论或专业判断；  \n"
        "   - 不得引入外部知识、主观评价或推测性内容。不得恶意修改原文内容。不得恶意修改原文内容。\n\n"
        "3. **结构层次优化**  \n"
        "   - 若原始内容包含多个主题或逻辑模块，请使用层级化标题组织内容；  \n"
        "   - 推荐采用 Markdown 语法，从二级标题开始：二级标题 `##` 表示主议题，三级标题 `###` 表示子议题，四级标题 `####` 表示细分要点；\n"
        "   - 不需要给出大的标题，直接子标题开始  \n"
        "   - 避免过度分层（一般不超过三级），确保结构清晰而不冗余。\n\n"
        "4. **输出要求**  \n"
        "   - 不包含任何引导语（如“润色结果如下：”）、解释性语句或元注释；  \n"
        "   - 全文使用正式书面语，保持客观、严谨、专业的学术风格。\n\n"
        "   - 不要在结尾输出参考文献标题，不要在结尾输出参考文献标题。\n\n"
        "## 原始输入\n\n"
        f"{user_input.strip()}\n\n"
    )


def build_beautiful_format_prompt(
    user_input: str,
    history_answer: Union[str, None],
    references: Union[ReferenceList, str, None],
) -> str:
    """
    构建面向大语言模型的后处理润色提示词（Post-editing Prompt），
    用于对模型生成的原始回答进行语言规范化、逻辑结构优化与格式美化。

    本函数旨在提升输出文本的可读性、专业性与结构性，同时严格约束内容保真度。
    尽管接口保留 `references` 参数以维持与检索增强生成（RAG）流水线的一致性，
    但在当前润色阶段，参考文献不参与语义校验或内容修正。

    润色过程遵循以下原则：
        1. **形式优化**：调整段落结构、规范标点使用、消除语法瑕疵、提升语句流畅度；
        2. **内容保真**：禁止增删事实性信息、不得篡改技术细节或逻辑结论；
        3. **风格统一**：采用正式书面语体，契合学术或技术文档表达规范；
        4. **结构分层**：根据内容逻辑，合理组织多级标题体系，增强可读性；
        5. **输出纯净**：返回结果应为可直接使用的最终文本，不含解释性前缀或元说明。

    Args:
        user_input (str):
            大语言模型生成的原始回答文本，作为润色对象。
        references (list[str], optional):
            检索所得参考文献列表。本阶段仅用于接口兼容，实际润色逻辑中不予使用。
            默认值为 None。

    Returns:
        str:
            符合上述规范的结构化提示词，可直接输入至润色模型以生成优化后文本。

    Notes:
        若输入为空或仅含空白字符，将自动替换为占位符“（无内容）”，
        以确保提示词结构完整性并避免模型异常行为。
    """
    if not user_input or not user_input.strip():
        user_input = "（无内容）"

    ref_text = ""
    if references is None:
        ref_text = ""
    elif isinstance(references, str):
        if len(references.strip()) > 5:
            # 保留完整内容，并明确编号用于引用
            ref_text = references
    ref_section = f"\n\n#### 参考文献\n\n{ref_text}" if ref_text.strip() else ""
    logger.info(f"prompts deep research 最后的 ref_text: {ref_text}")
    return (
        "# 文本后处理任务指令\n\n"
        "你作为专业学术编辑，请对下述由大语言模型生成的原始回答执行规范化润色。\n"
        "润色目标是在**严格保持原意不变**的前提下，提升文本的学术性、逻辑性、结构性与可读性。\n\n"
        "## 处理规范\n\n"
        "请遵循以下准则进行编辑：\n\n"
        "1. **格式与语言优化**  \n"
        "   - 调整段落划分，确保语义连贯；  \n"
        "   - 规范使用中文全角标点（如，。：“”‘’）；  \n"
        "   - 修正语法错误、搭配不当及语序问题；  \n"
        "   - 提升语句流畅度与书面表达严谨性。\n\n"
        "2. **内容保真约束**  \n"
        "   - 不得新增、删除或修改任何事实性陈述、数据、法律条文或技术细节；  \n"
        "   - 禁止改变原始推论逻辑、结论或专业判断；  \n"
        "   - 不得引入外部知识、主观评价或推测性内容。\n\n"
        "3. **结构层次优化**  \n"
        "   - 若原始内容包含多个主题或逻辑模块，请使用层级化标题组织内容；  \n"
        "   - 推荐采用 Markdown 语法，从二级标题开始：二级标题 `##` 表示主议题，三级标题 `###` 表示子议题，四级标题 `####` 表示细分要点；\n"
        "   - 不需要给出大的标题，直接子标题开始  \n"
        "   - 避免过度分层（一般不超过三级），确保结构清晰而不冗余。\n\n"
        "4. **输出要求**  \n"
        "   - 不包含任何引导语（如“润色结果如下：”）、解释性语句或元注释；  \n"
        "   - 全文使用正式书面语，保持客观、严谨、专业的学术风格；\n"
        "   - 必须保留正文中已有的引用标记（如 [1]、[W1]），不得删除、改号或改变其对应关系；\n"
        "   - 若下方提供参考文献，请原样保留其标题、URL 与编号；若未提供，不要输出“参考文献”空标题。\n\n"
        "## 原始输入\n\n"
        f"{user_input.strip()}"
        f"{ref_section}"
    )
    # "   - 如果有参考文献，在结尾输出参考文献，格式是[标题](url)不要修改参考文献内容和添加其他内容，不要修改参考文献内容和添加其他内容；如果参考文献内容为空，不要在结尾输出参考文献。\n\n"


def build_beautiful_format_rag_prompt(
    user_input: str,
    history_answer: Union[str, None],
    references: Union[ReferenceList, str, None],
) -> str:
    if not user_input or not user_input.strip():
        user_input = "（无内容）"

    ref_text = ""
    if references is None:
        ref_text = ""
    elif isinstance(references, str):
        if len(references.strip()) > 5:
            ref_text = references
    ref_section = f"\n\n#### 参考文献\n\n{ref_text}" if ref_text.strip() else ""

    logger.info(f"prompts deep research 最后的 rag ref_text: {ref_text}")

    return (
        "你是一位专业学术编辑，请对【原始输入】进行精细化润色。\n"
        "润色必须在**完全保留原意**的前提下，提升文本的逻辑性、学术表达与可读性。\n\n"
        "请严格遵循以下要求：\n\n"
        "1. **语言与格式**  \n"
        "   - 修正语法和格式错误、搭配不当或语序问题；不得改变原始推理路径、结论或专业判断；\n"
        "   - 若内容涉及多个主题或逻辑模块，请使用 Markdown 标题合理分层；从 `##` 二级标题开始组织主议题，必要时用 `###` 表示子议题；  \n"
        "   - **不要添加总标题**，直接以第一个子标题或正文开头。\n\n"
        "2. **输出要求**  \n"
        "   - 仅输出润色后的正文，**不得包含任何引导语、解释、注释、元信息或结构性预告**；  \n"
        "   - 必须保留正文已有的 [n]、[Wn] 等引用标记，不得删除、改号或移动到无关句子；\n"
        "   - **严禁出现以下内容**：\n"
        "     - 禁止以“摘要”“概要”“本文”“本研究”“本报告”“撰写目的”“以下内容”“综上所述”“引言”“背景”“结论”等开头的句子或未在原文中出现的标题；\n"
        "     - 禁止任何形式的自我指涉（如“本文将……”“本回答分析了……”）；\n"
        "     - 正文必须从第一个**实质性论述句**直接开始（例如：“人工智能正推动……”而非“本文探讨……”）；\n"
        "3. **参考文献处理规则**\n"
        "   - 若下方提供参考文献，请与原始输入中已有参考文献合并去重，保留与正文论证相关的条目；\n"
        "   - 不得改写参考文献标题、URL 或编号；若没有任何参考文献，不要输出空标题。\n"
        "## 原始输入\n\n"
        f"{user_input.strip()}"
        f"{ref_section}"
    )


def build_deep_research_noref_prompt(
    user_input: str,
    history_answer: Union[list[str], None],
    references: Union[ReferenceList, None],
) -> str:
    """
    构建企业检索 deepresearch 模型的深度研究型提示词（Prompt），
    显式要求模型在回答中按 [n] 格式引用参考文献。

    Args:
        user_input (str): 用户提出的问题。
        references (list[dict]): 从向量数据库检索到的相关法律文献片段列表，
                                 每个元素需包含 'extracted_content' 字段。

    Returns:
        str: 格式规范、指令明确的提示词字符串。
    """
    if not references:
        ref_text = ""
    else:
        # 保留完整内容，并明确编号用于引用
        ref_items = [
            f"[{i + 1}] {ref['extracted_content']}" for i, ref in enumerate(references)
        ]
        ref_text = "\n".join(ref_items)

    if not history_answer:
        history_text = "无相关历史回答。"
    else:
        history_items = [
            f"第[{i + 1}]个历史系统回答: {history_answer[i]}"
            for i, ref in enumerate(history_answer)
        ]
        history_text = "\n".join(history_items)

    prompt = (
        "你是由中电云开发的企业检索 RAG 智能助手，具备专业知识和深度分析能力。\n\n"
        "请根据以下【系统历史回答】并结合你的专业知识，严谨、全面地回答用户问题以及系统推理子问题。\n\n"
        "【回答要求】\n"
        "- 使用正式中文书面语，逻辑严密，结构清晰（建议包含 分析、结论等部分），格式可以灵活多变，**逻辑必须严谨、逻辑必须严谨、逻辑必须严谨**；\n"
        "- 若用户未明确指定回答长度或格式，请以深度研究报告形式充分作答；\n"
        "- 若用户在问题中明确提出格式、简略/详细程度等要求，请优先遵循用户指示；\n"
        "- 回答应结构清晰、语言规范、用词严谨；需要很强的逻辑性；结尾不需要给出相关的参考文献，结尾不需要给出相关的参考文献；\n"
        f"{ref_text}\n\n"
        f"【系统历史回答】\n{history_text}"
        f"【用户问题】\n{user_input}"
    )
    return prompt.strip()

def build_deep_research_session_chat_prompt(
    user_input: str,
    history_answer: Union[str, None],
    references: Union[ReferenceList, None],
) -> str:
    """
    构建法律智能问答 Chat 模式的提示词（Prompt）。
    适用于不带 RAG 检索的纯对话场景，依赖模型自身的法律知识库。

    Args:
        user_input (str): 用户提出的法律问题。
        history_answer (str): 历史对话上下文。
        references: 保留接口兼容性，Chat 模式下不使用。

    Returns:
        str: 格式规范、指令明确的法律问答提示词。
    """
    if not history_answer:
        history_text = "无相关历史回答。"
    else:
        history_text = history_answer

    prompt = (
        "你是一名专业的中国法律顾问助手，具备扎实的法学理论功底和丰富的实务经验。\n\n"
        "请根据【对话历史】和【用户问题】，提供专业、准确、实用的法律解答；如事实条件不足，应先明确关键前提，再分情形作答。\n\n"
        "【回答要求】\n"
        "1. **法律依据优先**：优先援引中国现行有效的法律法规、司法解释、指导性案例；引用时写明法律名称与条款编号。\n"
        "2. **分析结构**：按“法律规则 → 事实要件 → 法律适用 → 结论/建议”展开；存在不同情形时，应分情形说明。\n"
        "3. **实务导向**：在法律分析基础上，给出可操作建议，如主管机关、申请/投诉/诉讼路径、举证材料、期限或风险控制。\n"
        "4. **不确定性处理**：事实不足、法律存在争议或地方执行口径可能不同的，应明确说明判断边界，不得强行下结论。\n"
        "5. **格式要求**：使用中文书面语，可用 Markdown 分节或分点；用户未要求简短时应充分展开论证。\n"
        "6. **语气要求**：保持客观、冷静、专业的法律职业口吻。\n"
        "7. **禁止事项**：\n"
        "   - 不得编造不存在的法律条文或虚构案例；\n"
        "   - 不得提供违法建议或规避法律的方法；\n"
        "   - 不要在结尾添加'仅供参考'建议咨询专业律师'等套话。\n\n"
        f"【对话历史】\n{history_text}\n\n"
        f"【用户问题】\n{user_input}"
    )
    logger.info(f"chat mode prompt: {prompt}")
    return prompt.strip()


def build_deep_research_session_rag_prompt(
    user_input: str,
    history_answer: Union[str, None],
    references: Union[ReferenceList, None],
) -> str:
    """
    构建法律智能问答 Deep Thinking 模式的提示词（Prompt）。
    适用于带 RAG 检索的场景，结合知识库中的法律条文、案例、司法解释进行回答。
    """
    if not references:
        ref_text = "（未检索到相关法律文献）"
    else:
        law_items = []
        web_items = []
        law_idx = 1
        web_idx = 1
        for ref in references:
            if ref.get("source_type") == "web":
                title = (ref.get("title") or "联网可信来源").strip()
                source = (ref.get("source") or "").strip()
                published_at = (ref.get("published_at") or "").strip()
                url = (ref.get("url") or "").strip()
                content = (ref.get("extracted_content") or ref.get("content") or "").strip()
                meta_parts = [part for part in [source, published_at] if part]
                meta = f"（{'；'.join(meta_parts)}）" if meta_parts else ""
                url_line = f"\n链接：{url}" if url else ""
                delivery = (ref.get("delivery") or "").strip()
                if delivery == "title_selected_full":
                    summary_note = "\n（以下为入选网页正文原文）"
                elif delivery in ("title_selected_summary", "page_summary"):
                    summary_note = "\n（以下为 extract 模型从网页中筛选的重要原文摘录）"
                else:
                    summary_note = ""
                web_items.append(
                    f"[W{web_idx}] {title}{meta}{url_line}{summary_note}\n{content}"
                )
                web_idx += 1
            else:
                law_items.append(f"[{law_idx}] {ref['extracted_content']}")
                law_idx += 1
        sections = []
        if law_items:
            sections.append("【本地法律知识库】\n" + "\n".join(law_items))
        if web_items:
            sections.append("【联网可信证据】\n" + "\n".join(web_items))
        ref_text = "\n\n".join(sections) if sections else "（未检索到相关法律文献）"

    if not history_answer:
        history_text = "无相关历史回答。"
    else:
        history_text = history_answer

    prompt = (
        "你是一名专业的中国法律顾问助手，具备扎实的法学理论功底和丰富的实务经验。\n"
        "现已为你检索到本地法律知识库材料及（如有）联网可信证据。\n\n"
        "请根据【法律知识库检索结果】、【联网可信证据】、【对话历史】和【用户问题】，提供专业、准确、实用且有依据链条的法律解答。\n\n"
        "【回答要求】\n"
        "1. **仔细阅读材料**：作答前须通读【法律知识库检索结果】与全部 [Wn] 联网材料（含正文原文、案例表述、数字与期限），"
        "以材料为准组织论证，不得凭常识跳过或简化材料内容。\n"
        "2. **材料并用**：法条与联网证据并重；[Wn] 中已是 extract 模型筛选后的**重要原文摘录**，"
        "须据此写入具体例子、裁判口径、程序节点、举证要求并标注 [Wn]；多条角度不同须分别使用。\n"
        "3. **引用格式**：法条句末（参见[n]）；联网（参见[[Wn]](该条「链接」)），编号与 URL 须与材料包一致。"
        "材料中的案例、数据、原文表述可在正文中转述或概括，关键处注明出处。\n"
        "4. **写法**：说明法条含义及如何适用，不要只贴条文；结构自定、充分展开，事实不足或存在争议时分情形说明，必要时提示待查事实与风险边界。\n"
        "5. **禁止**：不得编造或篡改材料，不得提供违法建议；结尾不加空洞套话。\n\n"
        f"【法律知识库检索结果】\n{ref_text}\n\n"
        f"【对话历史】\n{history_text}\n\n"
        f"【用户问题】\n{user_input}"
    )
    logger.info("deep thinking rag mode prompt: len=%d", len(prompt))
    return prompt.strip()




def build_deep_research_prompt(
    user_input: str,
    history_answer: Union[list[str], None],
    references: Union[ReferenceList, None],
) -> str:

    if not references:
        ref_text = ""
    else:
        # 保留完整内容，并明确编号用于引用
        ref_items = [
            f"[{i + 1}] {ref['extracted_content']}" for i, ref in enumerate(references)
        ]
        ref_text = "\n\n".join(ref_items)

    if not history_answer:
        history_text = "无相关历史回答。"
    else:
        history_items = [
            f"第[{i + 1}]个历史系统回答: {history_answer[i]}"
            for i, ref in enumerate(history_answer)
        ]
        history_text = "\n".join(history_items)

    prompt = (
        "你是一位资深研究分析师，请根据以下【参考文献】【系统历史回答】并结合你的专业知识，严谨、全面地回答用户问题以及系统推理子问题。\n"
        "请严格依据以下三部分信息，生成一篇结构完整、逻辑严密、语言专业的回答：\n\n"
        f"{ref_text}\n\n"
        f"【系统历史回答】\n{history_text}"
        f"【用户问题】\n{user_input}"
        "【回答要求】\n"
        "1. **结构规范**：采用合理的表述架构，框架必须合理、结构清晰且逻辑性强，框架必须合理、结构清晰且逻辑性强：\n"
        "   - 确保分析深度、内容详实、逻辑连贯、层次分明"
        "2. **内容整合原则**：\n"
        "   - 回答需要将【系统历史回答】中分散但 主题或者意思 相近的信息进行归并、提炼与升华，抽取为核心观点；\n"
        "   - 所有陈述、结论与数据必须源自提供的【系统历史回答】，严禁引入外部知识或主观臆断；\n"
        "   - 每个核心的章节内容应详实且逻辑严密，以逻辑递进的形式表述。\n"
        "3. **输出格式**：\n"
        "   - 全文使用中文撰写；**严禁出现以下内容**：\n"
        "      - 以“摘要”“概要”“本文”“本研究”“本报告”“撰写目的”“以下内容”“综上所述”引言“等开头的句子；\n"
        "      - 任何形式的自我指涉（如“本文将……”“本回答分析了……”）；\n"
        "      - 正文必须从第一个**实质性论述句**直接开始（例如：“人工智能正推动……”而非“本文探讨……”）；\n"
        "   - 推荐采用 Markdown 标题语法，从二级标题开始：二级标题 `##` 表示主议题，三级标题 `###` 表示子议题，四级标题 `####` 表示细分要点；\n"
        '   - 不包含任何引导语、致谢、附录或额外说明（如"以下是报告""根据要求""XXX研究报告"等）；\n'
        "   - 回答应具备深度与完整性；\n"
        "   - 不需要给出参考文献；不需要给出参考文献；不需要给出参考文献；\n"
    )

    return prompt.strip()


def build_quality_judgment_prompt(
    user_input: str,
    history_answer: Union[str, None],
    references: Union[ReferenceList, None],
) -> str:
    """
    构建用于评估模型回答质量的系统提示词（Prompt）。

    评分维度：
    - 逻辑/法条/事实依据是否充分
    - 语义完整性
    - 回答格式清晰度
    - 内容丰富度
    - 无乱码（纯中文或合理多语言）

    Args:
        model_response (str): 待评估的模型原始回答

    Returns:
        str: 完整的评估指令 prompt
    """
    return (
        f"判断模型回答的质量，主要依据：重点校验文本内部是否有详细的逻辑依据、"
        f"语义是否完整性、模型回答格式是否清晰。信息过于简单简短（内容少）、没有逻辑分析、"
        f"缺乏事实依据、文本无格式等情况得分应较低；内容丰富、逻辑清晰、有事实依据、格式完整等应得高分。"
        f"中文和其他语言混合乱码不得分。若各方面均表现良好，最终分数应较高。\n\n"
        f"模型回答为：\n{user_input}\n\n"
        f"请输出详细的判断依据，并在最后一行以 '####' 开头输出总得分（取值范围 [0,1]，数值越大表示质量越高）。"
    )


def build_confidence_prompt(
    user_input: str, history_answer: str, references: str
) -> str:
    """
    构建用于评估答案置信度的提示（prompt），要求模型基于权威性与相关性，
    对答案给出一个 0 到 1 之间的置信度分数。

    - 0 表示无依据、未回答问题；
    - 1 表示有明确、权威证据支持且完整回答了问题；
    - 分数越高，表示答案越可靠、越贴合问题。

    参数:
        user_input (str): 用户原始提问（可用于上下文参考）。
        temp_answer (str): 模型生成的临时答案。

    返回:
        str: 格式化后的置信度评估提示。
    """
    prompt = (
        f"【用户问题】：{user_input}\n"
        f"【系统回答】：{history_answer}\n"
        "请根据以下标准评估【系统回答】的置信度：\n"
        "- 【系统回答】是否有明确、逻辑严谨的推理逻辑，回答思路清晰，格式完整；\n"
        "- 【系统回答】是否准确、完整地回答了用户的问题？【系统回答】如果没有回答用户问题直接输出总得分：0.00\n"
        "-  如果用户问题本身非常简单属于简单对话聊天，系统可以给出对应基于模型尝试回复也认为是合理的；\n"
        "-  **如果【系统回答】出现重复输出、出现复读机、乱码、多语言混合、非中文等，输出总得分：0.00**\n"
        "-  **如果【系统回答】出现重复输出、出现复读机、乱码、多语言混合、非中文等，输出总得分：0.00**\n"
        "-  **如果【系统回答】出现重复输出、出现复读机、乱码、多语言混合、非中文等，输出总得分：0.00**\n"
        "请仅输出一行，格式为：总得分：X.XX（X.XX 为 0 到 1 之间的浮点数，例如：总得分：0.85）。"
    )
    return prompt.strip()


def build_retrieval_plan_prompt(
    initial_query: str,
    history_answer: Union[str, None],
    references: Union[ReferenceList, None],
    *,
    max_web_queries: int = 5,
    need_web_queries: bool = True,
) -> str:
    """
    统一检索规划：一次输出知识库检索问句、BM25 关键词与联网检索词，保证 KB/Web 争点一致。

    输出必须为单行 JSON（无 Markdown、无解释）：
    {
      "display_query": "展示/改写后的完整法律问句，以问号结尾",
      "kb_query": "用于向量与 BM25 检索的问句（可不含问号）",
      "bm25_keywords": ["关键词1", "关键词2"],
      "web_queries": ["检索词串1", "检索词串2"]
    }
    """
    history_block = ""
    if history_answer and str(history_answer).strip():
        history_block = f"【对话历史（仅用于补全指代与背景）】\n{history_answer.strip()}\n\n"

    web_rules = ""
    if need_web_queries:
        web_rules = (
            f"4. web_queries：生成 1～{max_web_queries} 条联网检索词；**每条最多 3 个核心词**，半角空格分隔；\n"
            "   - 每条须紧扣用户问题的**案由与争点**，至少包含一个能界定案件类型的词（从用户问题或 bm25_keywords 归纳，"
            "勿凭空添加用户未涉及的新案由）；\n"
            "   - 禁止套话/空泛词：不得使用「裁判观点」「效力」「法律依据」「权威解读」「相关规定」「法律适用」等"
            "对检索无区分度的词；禁止仅写「举证责任」「赔偿范围」「索赔程序」而无案由限定；\n"
            "   - 在争点一致前提下，可从程序、举证、后果等不同角度拆分，但范围不宜过宽；可含法律名称、制度名；"
            "不要输出完整问句。\n"
        )
    else:
        web_rules = '4. web_queries：固定输出空数组 []。\n'

    return (
        "你是法律问答系统的统一检索规划器。根据用户问题（及可选对话历史），"
        "为知识库检索与联网检索生成一致的检索方案。\n\n"
        f"{history_block}"
        f"【用户问题】\n{initial_query.strip()}\n\n"
        "【任务】\n"
        "1. display_query：一句完整中文法律问句，尽量保留原文用词，只补全主体/对象/背景；必须以问号结尾。\n"
        "2. kb_query：用于语义检索与 BM25 的问句，与 display_query 争点一致，可去掉问号。\n"
        "3. bm25_keywords：2～5 个检索关键词数组，优先法律关系、行为、制度名、主体身份；\n"
        f"{web_rules}"
        "【共同规则】\n"
        "- 最小改写：不扩写、不引入原文没有的事实或新诉求；\n"
        "- 多行输入时，最后一行通常是本轮追问，前文仅用于补全背景；\n"
        "- 非法律闲聊也保持原意，不要硬改成法律问题。\n\n"
        "【输出格式】\n"
        "只输出一段合法 JSON，字段名必须齐全：display_query, kb_query, bm25_keywords, web_queries。\n"
        "bm25_keywords 与 web_queries 必须为 JSON 数组；禁止 Markdown、禁止解释、禁止多余字段。\n\n"
        "【示例】\n"
        '{"display_query":"员工被无故辞退后应如何维权？","kb_query":"员工被公司无故辞退如何维权",'
        '"bm25_keywords":["无故辞退","劳动合同","赔偿金"],'
        '"web_queries":["违法解除 劳动合同 赔偿金","劳动仲裁 劳动合同 举证","口头辞退 解除通知"]}'
    ).strip()


def build_sub_query_prompt(
    initial_query: str, history_answer: str, references: Union[ReferenceList, None]
) -> str:
    """
    构建用于生成验证性子问题的提示（prompt）。

    该提示引导模型针对当前答案中可能缺失或存疑的部分，
    提出一个具体、可搜索、聚焦于权威证据的子问题，
    用于后续检索验证（例如通过搜索引擎或知识库）。

    参数:
        initial_query (str): 用户的原始问题。
        temp_answer (str): 当前生成的答案。

    返回:
        str: 格式化后的子问题生成提示。
    """
    prompt = (
        "你是一个严谨的法律研究助手。请判断当前答案是否已经充分回答原始问题；若存在关键依据、适用条件、救济路径或风险边界缺口，请生成一个最值得继续检索的子问题。\n\n"
        f"【原始问题】：{initial_query}\n"
        f"【当前答案】：{history_answer}\n\n"
        "要求：\n"
        "- 子问题必须具体、简洁、可直接用于网络或知识库检索；\n"
        "- 优先围绕法律依据、构成要件、程序期限、主管机关、法律后果、案例口径或实务流程提问；\n"
        "- **仅输出子问题本身**，不要包含任何前缀、解释、理由或额外文本；\n"
        "- 如果【当前答案】已经充分回答【原始问题】，请输出：无子问题。\n\n"
        "示例（正确格式）：\n"
        "不予立案后发现新证据的救济程序\n"
        "监视居住违反规定的公安机关处理措施\n"
        "农民专业合作社章程法定载明事项\n\n"
        "请输出你的子问题："
    )
    return prompt.strip()


def build_web_page_summary_prompt(
    user_query: str,
    page_title: str,
    page_url: str,
    page_body: str,
) -> str:
    """
    联网入选页由 extract 模型：判断相关性 → 删无关/噪声 → 摘录重要原文 → 控制篇幅。
    """
    title = (page_title or "联网资料").strip()
    url = (page_url or "").strip()
    body = (page_body or "").strip() or "（无正文）"
    prompt = (
        "你是法律网页摘录专员。任务：阅读网页正文，判断哪些段落与用户争点**真正相关且重要**，"
        "删去无关与噪声后，将**重要部分的原文**整理输出给后续法律分析模型；你只负责摘录，不回答问题、不写法律意见。\n\n"
        f"【用户问题】\n{user_query.strip()}\n\n"
        f"【网页标题】{title}\n"
        f"【URL】{url or '（无）'}\n\n"
        f"【网页正文（上游可能已按字数截断，你仍需在可见范围内完成筛选）】\n{body}\n\n"
        "【工作流程】\n"
        "1. **相关性判断**：若网页主题与用户法律争点明显无关（仅关键词相似、案由/法律关系不同），"
        "只输出一行「【无关】」，不要输出其他任何文字。\n"
        "2. **删无关与噪声**：去掉导航、广告、推荐、评论、与争点无关的板块；不要保留套话导读。\n"
        "3. **判断重要并摘录原文**：对剩余内容，只保留与争点直接相关的段落；"
        "优先保留：法律依据、构成要件、程序/期限、举证要点、裁判/官方表述、**具体案例与示例**。\n"
        "4. **原文呈现**：关键句段尽量**照录原文**（可用引号标出）；条号、数字、日期、期限、比例不得改写成模糊概括；"
        "允许用「……」省略无关句，但不得编造网页中没有的内容。\n"
        "5. **超长处理**：正文很长时，只保留最相关的若干段原文摘录，删次要例子与重复表述，"
        "控制总篇幅约 800～4500 字；宁可少而准，也不要为凑字数保留弱相关段落。\n"
        "6. 输出为分点或分段的中文摘录，不要加 [Wn] 或 Markdown 链接。\n"
    )
    return prompt.strip()


def build_extract_info_prompt(
    initial_query: str,
    history_answer: Union[str, None],
    references: Union[ReferenceList, None],  # 保留接口但不使用
) -> str:
    """
    构建用于从网页文本中抽取主要内容（即网页的核心正文文本），并评估其与原始问题的相关性。
    """
    if history_answer is None or not history_answer.strip():
        webpage_text = "（无内容）"
    else:
        webpage_text = history_answer.strip()

    prompt = (
        "请从以下网页文本中提取其主要内容（即网页的核心正文部分），并评估核心正文内容与原始问题的相关性。\n\n"
        "**任务要求：**\n"
        "- 仅保留网页中表达核心信息的主体文本，如文章正文、报告主体、新闻正文等；\n"
        "- 过滤掉与主要内容无关的次要文本，例如：导航菜单、广告语、版权声明、页脚链接、登录提示、推荐列表、重复标语等；\n"
        "- 提取的内容应尽可能保持原文表述，不要自行总结、改写或添加解释；\n"
        f"原始问题：{initial_query}\n"
        f"网页文本：{webpage_text}\n\n"
        "**输出格式（严格按以下格式，抽取的内容应该是网页文本中相关内容，过滤掉无效乱码）：**\n"
        "抽取内容：[在此填写提取的内容]\n"
        "相关性得分：X.XX"
    )
    return prompt.strip()


def build_session_prompt(
    user_input: str, history_answer: str, references: Union[str, ReferenceList]
) -> str:
    """
    构建会话延续性判断 prompt，用于评估当前问题是否对最近一轮问答上下文构成延续性操作。
    """
    history_query = history_answer  # 注意：此处变量名可能有误，应为历史问题？见下方说明
    history_answers = references

    prompt = (
        "请严格判断【当前问题】是否构成对**最近一轮问答上下文**（包括历史问题及其回答）的延续性操作。\n\n"
        "【历史问题】：{history_query}\n"
        "【历史回答】：{history_answers}\n"
        "【当前问题】：{user_input}\n\n"
        "满足以下任一条件即视为延续性操作：\n"
        "1. 包含明确指代词（如“上述”“那”“再”“前面”“刚才”“上次”“它”“这个”等）指向历史内容；\n"
        "2. 显式要求对历史回答进行操作（如重答、重写、压缩、精简、扩写、举例、换格式、翻译等）；\n"
        "3. 请求重新回答**之前的问题**（例如：“重新回答前面的问题”）；\n"
        "4. 在同一主题下提出自然推进的子问题（例如：从“列表怎么用？”到“那字典呢？”）；\n"
        "5. 对历史回答追问原因、细节、依据、示例或澄清。\n\n"
        "以下情形**不算**延续（应判为新话题，得分 < 0.60）：\n"
        "- 仅同属一个大法律领域，但争点、主体或案由明显不同（如上一轮问劳动合同解除，本轮问股东出资责任）；\n"
        "- 无指代词、无操作指令，且问题可独立理解，不依赖上一轮才能作答。\n\n"
        "评分标准（仅输出总得分，保留两位小数）：\n"
        "- 0.80–1.00：明确延续（含操作指令、强指代或逻辑推进）；\n"
        "- 0.60–0.79：弱相关（语义模糊但可能有关联）；\n"
        "- 0.00–0.59：新话题（无指代、无操作、主题无关）。\n\n"
        " 注意：仅输出一行，格式为：总得分：X.XX\n\n"
    ).format(
        history_query=history_query,
        history_answers=history_answers,
        user_input=user_input,
    )

    logger.debug("Session Prompt: %r", prompt)
    return prompt.strip()


def build_deep_research_report_topic_prompt(
    initial_query: str,
    topic_report: Union[str, None],
    history_answer: Union[str, None],
    references: Union[ReferenceList, str, None],
) -> str:
    """
    构建用于生成深度研究报告框架与核心主题划分的提示词。
    输出应为结构化研究提纲，用于指导后续内容组织与检索规划。
    """
    # 安全处理历史回答
    if history_answer is None or not history_answer.strip():
        history_answer = "暂无已有分析。"

    prompt = (
        "你是一位资深研究顾问，正在为一项深度研究报告设计顶层框架。\n"
        "请基于用户的原始问题和当前已知信息，提出一个**结构清晰、逻辑递进的研究框架**，明确应从哪些核心主题切入，以全面、系统地回答该问题。\n\n"
        f"【用户原始问题】\n{initial_query}\n\n"
        f"【当前已知信息】\n{history_answer}\n\n"
        f"【主题信息】\n{topic_report if topic_report else '无相关信息，请根据原始问题生成。'}\n\n"
        "【输出要求】\n"
        "1. **聚焦框架设计**：不要生成具体子问题或答案，而是列出 3个左右核心研究子主题/维度；\n"
        "2. **逻辑递进**：按研究报告标准结构组织，例如：背景与定义 → 关键机制/原理 → 应用场景与案例 → 挑战与局限 → 替代方案或未来方向；\n"
        "3. **紧扣原问题**：每个主题必须直接服务于深化对用户原始问题的理解；\n"
        "4. **避免重复**：跳过当前已知信息中已充分覆盖的内容；\n"
        "5. **输出格式**：仅输出一段简洁文字，以“研究框架与主题划分”开头，后接连贯段落，**不要使用编号、项目符号或换行分隔**。\n\n"
        "【正确示例】\n"
        "研究框架与主题划分\n"
        "第一个实例：我将围绕大语言模型在医疗诊断中的应用可行性展开研究，首先界定其技术边界与伦理约束，其次分析现有临床辅助系统的架构与性能指标，接着调研真实世界中的部署案例与医生反馈，然后评估其在误诊风险与责任归属方面的挑战，最后对比传统规则引擎与多模态融合等替代路径的优劣。\n\n"
        "第二个实例：我正在着手评估通过“智能体构建 -> 智能体生成 CoT 数据采集 -> 模型训练”这一流程解决垂直领域问题的可行性与效益。我的研究将围绕三个核心方面展开：首先，深入分析当前智能体生成 CoT 数据的技术现状、质量以及面临的挑战；其次，搜寻实际案例和报告，以验证该方法的有效性；最后，探索和比较如 RAG、知识图谱集成等更优或互补的替代解决方案。\n\n"
        "请严格按上述要求输出你的研究框架与主题划分："
    )
    return prompt.strip()


def build_deep_research_report_subtopic_prompt(
    initial_query: str, history_answer: str, references: Union[ReferenceList, None]
) -> str:
    """
    构建用于生成深度研究子问题的提示词。
    子问题应支持后续检索，形成结构化研究报告。
    """
    # 安全截断历史回答，避免过长或空值
    history_snippet = (
        # history_answer[:500].strip() + ("..." if len(history_answer) > 500 else "")
        history_answer.strip()
        if history_answer.strip()
        else "暂无已有回答。"
    )

    prompt = (
        "你是一位专业研究分析师，正在围绕用户的原始问题开展深度调研。\n"
        "请严格基于以下信息，生成 **5～8个左右递进式、可检索的子问题**，如果【用户原始问题】简单生成子问题个数可以5个以内，用于驱动下一步的证据收集。\n\n"
        f"【用户原始问题】\n{initial_query}\n\n"
        f"【当前已知信息】\n{history_snippet}\n\n"
        "【生成要求】\n"
        "1. **紧扣原问题**：所有子问题必须直接服务于深化或验证用户原始问题回答；\n"
        "2. **避免重复**：不要复述当前已知信息中已明确覆盖的内容；\n"
        "3. **递进结构**：按研究报告逻辑展开——背景/定义->原理/机制->应用/案例 -> 对比/局限 -> 趋势/影响；\n"
        "4. **可检索性**：每个问题必须是具体、完整、适合直接用于网络或学术搜索短句子，**必须带有明确的搜索主体**（禁用模糊词如“吗”“啊”）；\n"
        "5. **输出格式**：仅输出子问题，每行一个，无编号、无前缀、无解释、无标点结尾以外的符号。\n\n"
        "【正确示例】\n"
        "监督微调（SFT）在大语言模型中的核心作用\n"
        "SFT 的典型训练流程包含哪些关键步骤\n"
        "有哪些开源项目实现了 SFT，其效果如何评估\n"
        "SFT 与强化学习微调（如 RLHF）的主要区别\n\n"
        "请严格按上述要求，如果【当前已知信息】已经回答了用户的问题，则不需要输出子问题，否则输出 1~5 个子问题："
    )
    return prompt.strip()


def build_deep_research_report_prompt(
    initial_query: str,
    topic_report: Union[str, None],
    history_answer: Union[str, None],
    references: Union[ReferenceList, str, None],
) -> str:
    """
    构建用于生成最终深度研究报告的提示词。
    要求模型基于子问题调研结果，整合为结构化、学术风格的综合报告，并以 Markdown 格式输出。
    """
    # 安全处理历史回答
    if history_answer is None:
        history_answer = ""
    history_snippet = (
        history_answer.strip() if history_answer.strip() else "暂无前期分析内容。"
    )

    logger.info("History Snippet for Deep Research Report: %r", history_snippet)

    prompt = (
        "你是一位资深研究分析师，正在整合多个子报告内容，撰写一份关于用户原始问题的深度综合研究报告。\n"
        "请严格依据以下三部分信息，生成一篇结构完整、逻辑严密、语言专业的综述性报告：\n\n"
        f"【原始问题】\n{initial_query}\n\n"
        f"【整体报告框架（初步提纲）】\n{topic_report}\n\n"
        f"【已有子报告内容】\n{history_snippet}\n\n"
        "【任务核心指令】\n"
        "你的首要任务是从【已有子报告内容】中识别、提炼并融合核心议题；【整体报告框架】用于限定主题边界和章节方向，最终章节应以已有材料中证据最充分、关联最紧密的内容为主。\n\n"
        "【报告构建原则】\n"
        "1. **主题提炼与框架生成**：\n"
        "   - 通读所有子报告内容，识别其中反复强调、多角度论述或数据支撑充分的关键维度；\n"
        "   - 将语义相近或逻辑关联紧密的子主题进行合并、归类与层级化，形成具有内在逻辑递进关系的章节体系；\n"
        "   - 主章节数量不超过 5 个；每个主章节包含 2～3 个子议题，确有必要时最多 5 个；每个子议题应有完整论证，避免空泛罗列。\n\n"
        "2. **内容整合与逻辑深化**：\n"
        "   - 在每个章节内，综合各子报告对该议题的论述，消除重复表述，补全逻辑断点，清晰呈现共识观点、分歧立场及证据支撑；\n"
        "   - 所有事实、判断、趋势预测或数据引用必须严格源自【已有子报告内容】，严禁引入外部知识、常识推断或主观评论；\n"
        "3. **语言与格式规范**：\n"
        "   - 采用学术性、客观、精准的专业语言，避免第一人称、宣传性措辞或模糊表达；\n"
        "   - 全文使用中文撰写，直接以 `### 摘要` 开头，不包含任何引导语、致谢、附录或元说明；\n"
        "   - 使用 Markdown 标题语法：摘要用 `### 摘要`，主议题用 `##`，子议题用 `###`，细分要点用 `####`；\n"
        "   - 在信息充分时尽量写成长文报告；若材料不足，不要强行扩写，应明确指出信息缺口。\n\n"
        "【输出要求】\n"
        "- 直接输出完整报告，从 `### 摘要` 开始连续撰写；\n"
        "- 不标注信息来源，不列出参考文献，不提及子报告编号或作者；\n"
    )
    return prompt.strip()


def build_deep_research_sub_report_prompt(
    initial_query: str,
    topic_report: Union[str, None],
    history_answer: Union[str, None],
    references: Union[ReferenceList, str, None],
) -> str:
    """
    构建用于生成最终深度研究报告的提示词。
    要求模型基于子问题调研结果，整合为结构化、学术风格的综合报告，并以 Markdown 格式输出。
    """
    # 安全处理历史回答
    if history_answer is None:
        history_answer = ""
    history_snippet = (
        history_answer.strip() if history_answer.strip() else "暂无前期分析内容。"
    )

    prompt = (
        "你是一位资深研究分析师，正在基于已完成的子问题调研，撰写一份关于用户原始问题的深度研究报告。\n\n"
        "请严格依据以下四部分信息，将分散的子问题分析内容**提炼、归纳并整合**为一篇结构严谨、逻辑连贯、语言学术化的综述性报告（使用 Markdown 格式）：\n\n"
        f"【原始问题】\n{initial_query}\n\n"
        f"【报告整体结构框架】\n{topic_report}\n\n"
        f"【子问题研究结论汇总】\n{history_snippet}\n\n"
        "【核心任务与要求】\n"
        "1. **以子问题结论为基础生成章节内容**：\n"
        "   - 报告的每一章节必须基于【子问题研究结论汇总】中的具体发现进行归纳与整合；\n"
        "   - 避免直接复制粘贴，应提炼共性、识别差异、建立逻辑关联；\n"
        "   - 在‘主要发现与研究结论’等关键章节中，明确总结各子问题的核心结论及其对原始问题的回答贡献。\n\n"
        "2. **遵循指定结构**：\n"
        "   - 严格按照【报告整体结构框架】组织内容，不得擅自增删主要章节；\n"
        # "   - 若某章节缺乏对应子问题支持，请注明：“当前信息不足，暂无法深入分析”。\n\n"
        "   - 每个章节需要逻辑清晰，表述严谨，层次结构清晰，建议使用论文或者报告层级结构”。\n\n"
        "   - 每个核心的章节内容应详实且逻辑严密，以逻辑递进的形式表述。\n\n"
        "3. **内容与语言规范**：\n"
        "   - 所有陈述必须基于提供的子问题结论或参考文献，严禁虚构、推测或引入外部知识；\n"
        "   - 使用客观、正式、学术化的语言，避免第一人称（如“我认为”）、主观副词（如“显然”“毫无疑问”）或营销式表达；\n"
        "4. **输出格式**：\n"
        "   - 全文使用 Markdown 格式；\n"
        "   - 章节标题使用 ##、### 等层级；\n"
        "   - 列表、表格等可酌情使用以提升可读性。\n"
    )
    return prompt.strip()
