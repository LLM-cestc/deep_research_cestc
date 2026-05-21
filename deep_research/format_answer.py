# -*- coding: utf-8 -*-  # noqa: UP009
"""
deepresearch agent 格式化溯源模块

Author: wjianxz
Date: 2025-11-13
"""

import re
from typing import Tuple, Union
import time 

from deep_research.protocal import ReferenceList


def replace_and_renumber_citations(
    text: str, references: Union[ReferenceList, None], max_title_length: int = 4
) -> Tuple[str, Union[ReferenceList, None]]:
    """
    将文本中的 [n] 引用重新编号为连续整数（按首次出现顺序），
    并返回新文本和按引用顺序排列的新 references 列表。

    Args:
        text: 原始文本，包含 [1], [3], [5] 等可能间断的引用。
        references: 原始参考文献列表，索引 i 对应原编号 i+1。
        max_title_length: 链接显示标题的最大长度。

    Returns:
        (new_text, new_references): 重新编号后的文本和参考文献列表。
    """
    if references is None:
        return text, references

    # 找出所有被引用的旧编号（按出现顺序，去重）
    used_old_indices = []
    for match in re.finditer(r"\[([1-9]\d*)\]", text):
        old_num = int(match.group(1))
        if old_num not in used_old_indices:
            used_old_indices.append(old_num)

    # 构建旧编号//新编号 映射
    old_to_new = {}
    new_references = []
    for new_idx, old_num in enumerate(used_old_indices):
        old_index = old_num - 1  # 转为列表索引
        if 0 <= old_index < len(references):
            old_to_new[old_num] = new_idx + 1
            new_references.append(references[old_index])
        else:
            # 超出范围的引用保留原样（不加入映射）
            pass

    # 缩写标题函数
    def abbreviate(title: str, max_len: int) -> str:
        if len(title) <= max_len:
            return title
        return title[: max_len - 1] + "…"

    # 替换文本中的 [old] 为 [缩写标题](url)
    def replace_match(match):
        old_num = int(match.group(1))
        if old_num in old_to_new:
            new_ref = new_references[old_to_new[old_num] - 1]
            url = new_ref.get("url", "#")
            raw_title = new_ref.get("title") or f"参考文献 {old_num}"
            display_title = abbreviate(raw_title, max_title_length)
            return f"[{display_title}]({url})"
        else:
            # 未映射的编号（如越界）保留原样
            return match.group(0)

    new_text = re.sub(r"\[([1-9]\d*)\]", replace_match, text)
    return new_text, new_references


def replace_citations_with_links_test(
    text: str, references: list[dict[str, str]]
) -> str:
    """
    将文本中的 [1], [2], ... 引用标记替换为 Markdown 超链接。

    Args:
        text (str): 原始文本，包含如 [1] 的引用标记。
        references (List[Dict]): 参考文献列表，索引 0 对应 [1]，
                                 每个元素需包含 'url' 字段，建议包含 'title'。

    Returns:
        str: 替换后的 Markdown 文本，[n] 变为可点击链接。
    """

    def abbreviate_title(title: str, max_len: int) -> str:
        """将标题缩写为最多 max_len 个字符，超长加省略号"""
        if len(title) <= max_len:
            return title
        return title[: max_len - 1] + "…"

    def replace_match(match):
        try:
            idx = int(match.group(1)) - 1  # [1] → index 0
            if 0 <= idx < len(references):
                url = references[idx].get("url", "#")
                title = references[idx].get("title") or f"参考文献[{idx + 1}]"
                # 返回 Markdown 链接：[1](url)
                return f'[{idx + 1}]({url} "{title}")'
            else:
                # 超出范围，保留原样
                return match.group(0)
        except (ValueError, IndexError):
            # 非法编号，保留原样
            return match.group(0)

    # 匹配 [数字]，但排除可能的误匹配（如 [123abc]）
    # 使用 \b 确保是独立的 [...] 结构
    pattern = r"\[([1-9]\d*)\]"
    result = re.sub(pattern, replace_match, text)
    return result


def _strip_leading_title(text: str, law_name_raw: str, article: str) -> str:
    """去掉 text 开头与法条标题重复的前缀，避免出现「《刑法》第二百六十四条：中华人民共和国刑法 第二百六十四条：」这种重复。"""
    if not text or not (law_name_raw or article):
        return text
    # 可能的前缀形式：带书名号、不带书名号、带/不带冒号、中间有空格等
    name_with_marks = f"《{law_name_raw.strip()}》" if law_name_raw and not law_name_raw.startswith("《") else (law_name_raw or "")
    name_no_marks = (law_name_raw or "").replace("《", "").replace("》", "").strip()
    prefixes = [
        f"{name_with_marks}{article}：",
        f"{name_with_marks}{article}:",
        f"{name_with_marks}{article}",
        f"{name_no_marks} {article}：",
        f"{name_no_marks} {article}:",
        f"{name_no_marks}{article}：",
        f"{name_no_marks}{article}:",
    ]
    for p in prefixes:
        if p and text.startswith(p):
            return text[len(p) :].lstrip("：: ")
    return text


def format_knowledge_references(
    references: list[dict[str, any]] | None
) -> str:
    """
    将知识库检索到的法条格式化为 Markdown 引用块。

    Args:
        references: 包含法条信息的列表，每个元素应包含 'law_name', 'article', 'text' 等字段。

    Returns:
        str: 格式化后的 Markdown 文本。
    """
    if not references:
        return ""

    formatted_lines = ["\n\n### 参考法律法规"]
    for idx, ref in enumerate(references, 1):
        law_name_raw = (ref.get("law_name", "") or "").strip()
        article = (ref.get("article", "") or "").strip()
        text = (ref.get("text", "") or "").strip()

        # 显示用标题：带书名号，如《中华人民共和国刑法》第二百六十四条
        law_name = f"《{law_name_raw}》" if law_name_raw and not law_name_raw.startswith("《") else law_name_raw
        title = f"{law_name}{article}"

        # 去掉正文开头与标题重复的前缀，再拼接，避免重复显示
        body = _strip_leading_title(text, law_name_raw, article)
        content = f"**{title}**：{body}" if body else f"**{title}**"

        formatted_lines.append(f"{idx}. {content}")

    return "\n\n".join(formatted_lines)


def format_web_references(references: list[dict[str, any]] | None) -> str:
    """将正文实际引用过的联网可信来源格式化为 Markdown 引用块。"""
    if not references:
        return ""

    formatted_lines = ["\n\n### 联网可信来源"]
    for idx, ref in enumerate(references, 1):
        title = (ref.get("title", "") or "").strip() or "联网资料"
        source = (ref.get("source", "") or "").strip()
        published_at = (ref.get("published_at", "") or "").strip()
        url = (ref.get("url", "") or "").strip()
        cid = (ref.get("citation_id") or "").strip()
        meta_parts = [part for part in [source, published_at] if part]
        meta = f"（{'，'.join(meta_parts)}）" if meta_parts else ""
        cid_tag = f"**[{cid}]** " if cid else ""
        if url:
            line = f"{idx}. {cid_tag}**[{title}]({url})**{meta}"
        else:
            line = f"{idx}. {cid_tag}**{title}**{meta}"
        formatted_lines.append(line)

    return "\n\n".join(formatted_lines)


async def typewriter_effect(text, delay=0.05):
    """生成器：逐字输出文本"""
    result = ""
    for char in text:
        result += char
        yield result
        time.sleep(delay)

def message_subquery():
    """模型输出答案的格式化提示"""
    message="\n<br>模型上诉的回答可能不能从多角度全面的回答用户的问题，模型经过深度思考任需要仔细回答下面的问题：<br>"
    return message

def message_deepsearch_thinking():
    """模型输出答案的格式化提示"""
    message="正在对用户的问题进行深度思考，请稍候......"
    return message

def message_deepsearch_react():
    """模型输出答案的格式化提示"""
    message="<br>模型根据上诉的回答和推理的子问题，进行深度思考内容如下......<br>"
    return message
# def message_first_input():
#     """模型输出答案的格式化提示"""
#     message="<br>请稍候......<br>\n```markdown\n"
#     return message

def message_last_ouput():
    """模型输出答案的格式化提示"""
    message="<br>下面是对上诉思考过程进行的最后汇总和整理，详情如下：<br>"
    return message

# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # 模拟模型生成的回答
    answer = "根据相关规定，企业需履行数据安全义务[1]。此外，还需定期进行风险评估[2]。"

    # 参考文献列表（索引 0 对应 [1]）
    references = [
        {"url": "https://example.com/law1", "title": "《网络安全法》"},
        {"url": "https://example.com/guide2", "title": "《数据安全评估指南》"},
    ]

    # 转换
    # linked_answer = replace_and_renumber_citations(answer, references)
    # print(linked_answer)
