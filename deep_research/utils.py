# -*- coding: utf-8 -*-
"""
# 公共能力模块：各种公共抽象函数

Author: wjianxz
Date: 2025-11-13
"""
from typing import Callable, Dict, Any, Optional, Tuple, Union
import json
import re
import requests
from collections import Counter
from deep_research.protocal import ReferenceList
from deep_research.protocal import HistoryMessage

import logging

logger = logging.getLogger(__name__)  # 自动继承 root logger 的 handlers

# 首答/评估等请求的固定 system；篇幅与结构约束放在各 task 的 user prompt 中。
LEGAL_AGENT_SYSTEM_PROMPT = (
    "你是中电云研发的法律智能体。依据用户 prompt 中的材料作答，不得编造法条或案例；"
    "材料不足或存在争议时说明判断边界，不得虚构依据。"
)

# 制表框线等：不参与「乱码/刷重复」判定，避免 Markdown 表格误伤
_BOX_DRAWING_RE = re.compile(r"[\u2500-\u257f\u2580-\u259f]")
# 《》等 CJK 符号与全角字符算正文，避免合法中文被算成「标点占比过高」
_CONTENT_CHAR_RE = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9\u3000-\u303f\uff00-\uffef]"
)


def _repair_utf8_latin1_mojibake(text: str) -> str:
    """UTF-8 被误按 latin-1 解码时的常见乱码，尝试还原（与网页抓取处逻辑一致，避免误拦截）。"""
    if not text or len(text) < 12:
        return text
    cjk0 = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    if cjk0 > len(text) * 0.06:
        return text
    try:
        fixed = text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text
    cjk1 = sum(1 for c in fixed if "\u4e00" <= c <= "\u9fff")
    if cjk1 >= cjk0 + 8 and cjk1 >= max(12, len(fixed) * 0.04):
        return fixed
    return text


def is_invalid_output(text: str) -> bool:
    if not text:
        return True
    s = _repair_utf8_latin1_mojibake(str(text).strip())
    if len(s) < 20:
        return False
    return _is_gibberish(s) or _is_repetitive(s)


def _is_gibberish(text: str) -> bool:
    total = len(text)
    if total < 50:
        return False
    t = _BOX_DRAWING_RE.sub("", text)
    total = len(t)
    if total < 50:
        return False
    content_count = len(_CONTENT_CHAR_RE.findall(t))
    ws_count = sum(1 for c in t if c.isspace())
    punct_count = max(0, total - content_count - ws_count)
    valid_ratio = content_count / total
    punct_ratio = punct_count / total
    if valid_ratio < 0.4:
        return True
    if punct_ratio > 0.35:
        return True
    # 在去掉框线后检测单字符连写，避免 ──── / ━━━ 等表格线触发
    if re.search(r"(.)\1{30,}", t):
        return True
    return False


def _is_repetitive(text: str) -> bool:
    t = _BOX_DRAWING_RE.sub("", text)
    t = re.sub(r"\s+", " ", t.strip())
    total = len(t)
    if total < 120:
        return False
    n = 10
    grams = [t[i : i + n] for i in range(0, total - n + 1)]
    if not grams:
        return False
    counts = Counter(grams)
    top_gram, top_count = counts.most_common(1)[0]
    coverage = (top_count * n) / total
    if top_count >= 5 and coverage > 0.25:
        return True
    return False


def dedupe_repeated_answer_body(text: str) -> str:
    """
    去除模型将同一段正文连续输出两遍的情况。
    - 按空行分段：若前 h 段与后 h 段完全一致，则删去第二份并保留其后内容。
    - 若整体为两半完全相同的长文（无清晰空行），则折叠为一半。
    """
    if not text or len(text.strip()) < 200:
        return text
    s = text.strip()
    parts = [p.strip() for p in re.split(r"\n\s*\n+", s) if p.strip()]
    n = len(parts)
    if n >= 4:
        max_h = n // 2
        for h in range(max_h, 1, -1):
            if parts[:h] != parts[h : 2 * h]:
                continue
            block_len = sum(len(p) for p in parts[:h])
            if block_len < 200:
                continue
            return "\n\n".join(parts[: h] + parts[2 * h :]).strip()

    if len(s) >= 500:
        half = len(s) // 2
        a, b = s[:half].strip(), s[half:].strip()
        if a == b and len(a) >= 250:
            return a
    return text


def send_request_to_model_dr(
    user_query: str,
    prompt_builder: Callable[
        [str, Optional[str], Optional[str], Optional[Union[str, ReferenceList]]], str
    ],
    topic_report: Optional[
        Union[str, None]
    ] = None,  # topic_report: Optional[str] = None,
    history_answer: Optional[Union[str, None]] = None,
    references: Optional[Union[str, ReferenceList, None]] = None,
    model_name: str = "qwen3",
    api_url: str = "http://localhost:8008/v1/chat/completions",
    timeout: int = 120,
    temperature: float = 0.8,
    max_tokens: int = 4096,
    top_p: float = 0.92,
    repetition_penalty: float = 1.2,
) -> Optional[str]:
    """
    向 vLLM 服务发送模型回答质量评估请求，并返回评估结果。

    Args:
        user_query (str): 需要被评估的模型原始回答文本
        model_name (str): 用于评估的 judge 模型名称
        api_url (str): vLLM API 的完整 endpoint
        timeout (int): 请求超时时间（秒）
        temperature (float): 采样温度
        max_tokens (int): 最大生成 token 数
        top_p (float): nucleus sampling 参数
        repetition_penalty (float): 重复惩罚系数

    Returns:
        Optional[str]: 模型返回的评估文本，若失败则返回 None
    """
    # 构造 prompt
    user_prompt = prompt_builder(user_query, topic_report, history_answer, references)
    # 构造请求体
    #     {"role": "system", "content": "你是中电云AI团队研发的 Blue shark LLM，一个由中国电子云创造的智能体助手。"},
    # "messages": [{"role": "user", "content": user_prompt}],
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": LEGAL_AGENT_SYSTEM_PROMPT,
            },
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}
    # proxies = {"http": None, "https": None}  # 禁用代理
    proxies = None  # 禁用代理
    try:
        response = requests.post(
            url=api_url,
            headers=headers,
            data=json.dumps(payload),
            proxies=proxies,
            timeout=timeout,
        )
        response.raise_for_status()  # 自动抛出 HTTP 错误

        result: Dict[str, Any] = response.json()
        answer: str = result["choices"][0]["message"]["content"]
        return answer

    except requests.exceptions.Timeout:
        logger.info(f"请求超时（>{timeout}s）: {api_url}")
    except requests.exceptions.HTTPError:
        logger.info(f"HTTP 错误 {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        logger.info(f"网络请求异常: {e}")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.info(f"响应解析失败: {e}")
        logger.info("原始响应:", response.text if "response" in locals() else "N/A")
    logger.error("模型请求异常")
    return None


def send_request_to_model(
    user_query: str,
    prompt_builder: Callable[[str, Any, Any], str],
    history_answer: Optional[list[str] | str] = None,
    references: Optional[Union[str, ReferenceList]] = None,
    model_name: str = "qwen3",
    api_url: str = "http://localhost:8008/v1/chat/completions",
    timeout: int = 120,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
) -> Optional[str]:
    """
    向语言模型服务发送请求并获取回复。

    该函数通过 HTTP POST 请求与语言模型 API 进行交互，支持自定义 prompt 构建器、历史对话和参考材料。
    包含完整的错误处理机制，在网络异常或响应解析失败时返回 None。

    Args:
        user_query (str): 用户的查询文本
        prompt_builder (Callable[[str, Any, Any], str]): 用于构建请求 prompt 的函数，接收用户查询、历史回答和参考资料作为参数
        history_answer (Optional[list[str]|str]): 历史对话记录，可以是字符串或字符串列表，默认为 None
        references (Optional[Union[str, ReferenceList]]): 参考材料，可以是字符串或 ReferenceList 对象，默认为 None
        model_name (str): 使用的模型名称，默认为 "qwen3"
        api_url (str): 模型 API 的完整 URL，默认为 "http://localhost:8008/v1/chat/completions"
        timeout (int): 请求超时时间（秒），默认为 120
        temperature (float): 控制生成随机性的采样温度，默认为 0.7
        max_tokens (int): 最大生成的 token 数量，默认为 4096
        top_p (float): nucleus sampling 参数，默认为 0.95
        repetition_penalty (float): 重复内容的惩罚系数，默认为 1.1

    Returns:
        Optional[str]: 模型生成的回复文本。如果请求失败或发生错误，返回 None

    Raises:
        requests.exceptions.Timeout: 当请求超过指定的 timeout 时间时
        requests.exceptions.HTTPError: 当收到 HTTP 错误响应时
        requests.exceptions.RequestException: 当发生其他网络请求相关错误时
        KeyError: 当响应数据格式不符合预期时
        IndexError: 当响应数据结构不正确时
        json.JSONDecodeError: 当响应无法解析为 JSON 时

    Example:
        >>> def simple_builder(query, history, refs):
        ...     return f"Query: {query}"
        >>> response = send_request_to_model(
        ...     user_query="你好",
        ...     prompt_builder=simple_builder
        ... )
        >>> print(response)
    """
    # 构造 prompt
    user_prompt = prompt_builder(user_query, history_answer, references)
    # 构造请求体
    # "messages": [{"role": "user", "content": user_prompt}],
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": LEGAL_AGENT_SYSTEM_PROMPT,
            },
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}
    # proxies = {"http": None, "https": None}  # 禁用代理
    proxies = None  # 禁用代理
    try:
        response = requests.post(
            url=api_url,
            headers=headers,
            data=json.dumps(payload),
            proxies=proxies,
            timeout=timeout,
        )
        response.raise_for_status()  # 自动抛出 HTTP 错误

        result: Dict[str, Any] = response.json()
        answer: str = result["choices"][0]["message"]["content"]
        return answer

    except requests.exceptions.Timeout:
        logger.info(f"请求超时（>{timeout}s）: {api_url}")
    except requests.exceptions.HTTPError:
        logger.info(f"HTTP 错误 {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        logger.info(f"网络请求异常: {e}")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.info(f"响应解析失败: {e}")
        logger.info("原始响应:", response.text if "response" in locals() else "N/A")
    logger.error("模型请求异常")
    return None


def send_request_to_model_streaming(
    user_query: str,
    prompt_builder: Callable[[str, Any, Any], str],
    result_queue,
    history_answer: Optional[list[str] | str] = None,
    references: Optional[Union[str, ReferenceList]] = None,
    model_name: str = "qwen3",
    api_url: str = "http://localhost:8008/v1/chat/completions",
    timeout: int = 120,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
    cancel_event=None,
) -> Optional[str]:
    """
    流式向语言模型服务发送请求，实时将推理过程推送到队列。

    - reasoning / reasoning_content：队列 [False, text]（默认进思考区）。
    - 正文 content：队列 [False, chunk, "answer"]，由前端汇总进「回答区」，与思考区分栏展示。

    Args:
        user_query (str): 用户的查询文本
        prompt_builder: 用于构建请求 prompt 的函数
        result_queue: 用于流式输出的队列
        history_answer: 历史对话记录
        references: 参考材料
        model_name (str): 使用的模型名称
        api_url (str): 模型 API 的完整 URL
        timeout (int): 请求超时时间（秒）
        temperature (float): 采样温度
        max_tokens (int): 最大生成的 token 数量
        top_p (float): nucleus sampling 参数
        repetition_penalty (float): 重复惩罚系数

    Returns:
        Optional[str]: 完整的模型响应文本（不含 <think> 标签）
    """
    user_prompt = prompt_builder(user_query, history_answer, references)
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": LEGAL_AGENT_SYSTEM_PROMPT,
            },
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "stream": True,
    }

    headers = {"Content-Type": "application/json"}
    proxies = None

    try:
        response = requests.post(
            url=api_url,
            headers=headers,
            data=json.dumps(payload),
            proxies=proxies,
            timeout=(10, min(timeout, 90)),
            stream=True,
        )
        response.raise_for_status()

        full_content = ""
        stream_buffer = ""
        chunk_count = 0
        # 正文 content 一律以第三元 "answer" 推送，由前端写入「回答区」；
        # reasoning 仍用二元 [False, text]，默认归入「思考折叠区」。

        logger.info("开始流式读取响应...")
        for line in response.iter_lines(decode_unicode=True):
            if cancel_event is not None and cancel_event.is_set():
                logger.info("流式请求收到取消信号，关闭响应。")
                response.close()
                return None
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    logger.info("流式读取完成，共 %d 个 chunk", chunk_count)
                    break
                try:
                    chunk_data = json.loads(data_str)
                    delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                    reasoning = (
                        delta.get("reasoning")
                        or delta.get("reasoning_content")
                        or ""
                    )
                    content = delta.get("content", "")
                    if not content and not reasoning:
                        continue

                    chunk_count += 1
                    if reasoning:
                        result_queue.put([False, reasoning])
                    if content:
                        full_content += content
                        stream_buffer += content
                    if len(stream_buffer) >= 10:
                        result_queue.put([False, stream_buffer, "answer"])
                        logger.debug("推送流式正文: %d 字符", len(stream_buffer))
                        stream_buffer = ""

                except json.JSONDecodeError:
                    logger.debug("跳过无法解析的流式 chunk: %s", data_str[:200])
                    continue

        if stream_buffer:
            result_queue.put([False, stream_buffer, "answer"])
            logger.debug("推送剩余流式正文: %d 字符", len(stream_buffer))

        final_answer = re.sub(
            r"<think>.*?</think>",
            "",
            full_content,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

        return final_answer if final_answer else full_content

    except requests.exceptions.Timeout:
        logger.error("流式请求超时（>%ss）: %s", timeout, api_url)
    except requests.exceptions.HTTPError:
        logger.error("HTTP 错误 %s: %s", response.status_code, response.text)
    except requests.exceptions.RequestException as e:
        logger.error("网络请求异常: %s", e)
    except Exception as e:
        logger.exception("流式处理异常: %s", e)
    logger.error("流式模型请求异常")
    return None


def safe_tokenize(tokenizer, text: str):
    """
    安全地分词，避免出现异常。

    Args:
        tokenizer: 分词器对象，用于文本的编码和解码。
        text (str): 需要进行分词的输入文本。

    Returns:
        list: 分词后的列表，每个元素是一个token字符串。

    Example:
        >>> tokenizer = SomeTokenizer()
        >>> text = "大模型开源榜单有哪些，详细的分析"
        >>> safe_tokenize(tokenizer, text)
        ['大', '模型', '开源', '榜单', '有哪些', '，', '详细的', '分析']
    """
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    # ['大', '模型', '开源', '榜单', '有哪些', '，', '详细的', '分析']
    return [tokenizer.decode([tid]) for tid in input_ids]


def parser_confidence(model_output: str, score: float = 0.85) -> Tuple[bool, float]:
    """
    从模型输出中解析总得分，并判断是否高于给定阈值。

    参数:
        model_output (str): 模型返回的字符串，应包含"总得分"字段。
        score (float): 判断阈值，默认为 0.85。

    返回:
        Tuple[bool, float]: 
            - bool: 如果解析出的得分高于阈值，返回 True；否则返回 False。
            - float: 解析出的实际得分值。

    示例:
        >>> result, score = parser_confidence("总得分: 0.92")
        >>> print(result)  # True
        >>> print(score)   # 0.92
    """
    # 提取"总得分"之后的内容
    score_confidence = model_output.split("总得分")[-1]

    # 使用正则表达式匹配浮点数
    match = re.search(r"\d+\.\d+", score_confidence)
    match_score: float = 0.0
    if match:
        match_score = float(match.group())
    if match_score >= score:
        return True, match_score
    else:
        return False, match_score


def parser_extracted_output(output: str):
    """
    解析抽取的输出结果，提取内容和相关性得分。

    Args:
        output (str): 待解析的输出字符串，包含抽取内容和相关性得分。

    Returns:
        tuple: 包含两个元素的元组：
            - str: 抽取到的内容，若未找到则为空字符串。
            - float: 相关性得分，若未找到则为0.0。

    Notes:
        - 使用正则表达式匹配"抽取内容"和"相关性得分"或"得分"。
        - 支持多种分隔符（冒号、中文冒号）和空格。
        - 使用re.DOTALL标志使.能匹配换行符，支持跨行内容。
        - 若匹配失败，返回默认值（空字符串和0.0）。
    """
    # 记录待解析的输出内容，用于调试和日志记录
    logger.info(f"解析抽取之前的结果：\n{output}")
    # content_match = re.search(r"抽取内容(.+?)(\n|$)", output)
    content_match = re.search(
        r"抽取内容\s*[:：]?\s*(.*?)(?=\n(?:相关性得分|得分)|$)",
        output,
        re.DOTALL,  # 使 . 能匹配换行符（如果你希望跨行）
    )
    # score_match = re.search(r"相关性得分(\d+\.\d+)", output)
    # score_match = re.search(r"(相关性得分|得分)(\d+\.\d+)", output)
    score_match = re.search(r"(?:相关性得分|得分)\s*[:：]?\s*(\d+\.\d+)", output)

    extracted_content = content_match.group(1).strip() if content_match else ""
    relevance_score = float(score_match.group(1)) if score_match else 0.0

    logger.info(f"抽取到的内容：{extracted_content} \n {relevance_score}")
    return extracted_content, relevance_score


def extract_sub_question(model_output: Union[str, None]) -> str:
    """
    从模型输出中抽取子问题。

    尝试去除常见前缀、多余换行和解释，仅保留核心子问题。

    参数:
        model_output (str): 模型生成的原始文本。

    返回:
        str: 抽取出的子问题；若未找到，返回空字符串。
    """
    if not model_output:
        return ""

    # 去除首尾空白
    text = model_output.strip()

    # 常见前缀正则（不区分大小写）
    prefixes = [
        r"^子问题[:：]?",
        r"^问题[:：]?",
        r"^请输出你的子问题[:：]?",
        r"^your sub-question[:：]?",
        r"^sub[-\s]?question[:：]?",
        r"^answer[:：]?",
        r"^[\"']",
    ]

    # 合并为一个正则（匹配开头的任意前缀）
    prefix_pattern = "|".join(f"({p})" for p in prefixes)
    text = re.sub(prefix_pattern, "", text, count=1, flags=re.IGNORECASE)

    # 移除开头可能残留的标点或引号
    text = re.sub(r"^[\"'\-–—:\s]+", "", text)

    # 只取第一行（避免模型输出多个问题或解释）
    first_line = text.splitlines()[0].strip() if text.splitlines() else ""

    # 如果第一行为空，尝试找第一个以问号结尾的行
    if not first_line or not first_line.endswith("？") and not first_line.endswith("?"):
        for line in text.splitlines():
            line = line.strip()
            if line and (line.endswith("?") or line.endswith("？")):
                first_line = line
                break

    return first_line


def extract_sub_question_test(llm_output: Union[str, None]) -> str:
    """
    从模型输出中抽取子问题文本。

    支持格式：
      - 子问题：xxx
      - xxx（直接以问题开头）

    返回纯子问题字符串，无前缀。
    """
    if llm_output is None or not isinstance(llm_output, str):
        return ""
    # 方法1：匹配“子问题：...”格式
    match = re.search(r"子问题[：:]\s*(.+)", llm_output)
    if match:
        return match.group(1).strip()

    # 方法2：若无前缀，假设整句是问题（以问号结尾或含疑问词）
    lines = [line.strip() for line in llm_output.split("\n") if line.strip()]
    if lines:
        first_line = lines[0]
        if (
            first_line.endswith("？")
            or first_line.endswith("?")
            or any(
                w in first_line for w in ["是否", "什么", "如何", "为什么", "有没有"]
            )
        ):
            return first_line

    # 默认返回清理后的首行
    return lines[0] if lines else ""


def normalize_citations(text: str) -> str:
    """
    规范化文本中的引用格式，将引用标记[n]移动到句末标点之后。
    
    Args:
        text (str): 待处理的文本字符串
        
    Returns:
        str: 处理后的文本字符串，其中引用标记已移动到句末标点之后
        
    Note:
        该函数执行以下转换：
        - 将形如 "...[1]。" 的文本转换为 "...。[1]"
        - 支持的引用标记格式为方括号内的数字、逗号和空格，如 "[1, 2, 3]"
        - 支持的句末标点符号包括：中文句号(。)、感叹号(！)、问号(？)
        
    Example:
        >>> normalize_citations("这是一个引用[1]。")
        "这是一个引用。[1]"
    """
    # 将 [n] 移到句末标点之后（简单版）
    # 匹配：...[1]。 → ...。[1]
    text = re.sub(r"(\[[\d,\s]+\])([。！？])", r"\2\1", text)
    return text


def check_llm_output(
    output: Any,
    *,
    function_name: str = "llm_inference",
    query: Optional[str] = None,
    raise_on_none: bool = False,
) -> bool:
    """
    检测大模型函数返回值是否为 None，并记录异常日志。

    Args:
        output: 大模型函数的返回值。
        function_name: 函数名称，用于日志标识（默认 "llm_inference"）。
        query: 可选的用户查询内容，便于排查问题。
        raise_on_none: 若为 True，当 output 为 None 时抛出 RuntimeError。

    Returns:
        bool: True 表示正常（非 None），False 表示返回值为 None。

    Example:
        result = call_llm("你好吗？")
        if not check_llm_output(result, function_name="call_qwen", query="你好吗？"):
            result = "抱歉，模型暂时无法回答。"
    """
    # if logger is None:
    #     logger = logging.getLogger(__name__)

    if output is None:
        msg = f"大模型函数 '{function_name}' 返回值为 None！"
        if query is not None:
            msg += f" 查询内容: {query!r}"
        logger.error(msg)

        if raise_on_none:
            raise RuntimeError(f"LLM function '{function_name}' returned None.")
        return False
    else:
        logger.debug(f"'{function_name}' 返回有效结果 (type: {type(output).__name__})")
        return True


def parser_session_output(output: Union[str, None]) -> float:

    if not output:
        return 0.0
    logger.info(f"解析抽取 session 之前的结果：\n{output}")
    pat = r"(?:(?:总得分|得分)\s*[:：]?\s*)?(\d+\.\d+)"
    score_match = re.search(pat, output)
    session_score = float(score_match.group(1)) if score_match else 0.0

    logger.info(f"抽取到的得分：{session_score} \n ")
    return session_score


def parser_sub_topic_output(text_llm_output: str) -> list[str]:
    """
    从大模型返回的文本中提取子问题列表。
    假设每行一个有效问题，过滤空行。
    """
    lines = text_llm_output.strip().split("\n")
    subquestions = [line.strip() for line in lines if line.strip()]
    return subquestions
    # return ["子问题1", "子问题2", "子问题3"]


def get_ref_deep_speed(query_deepresearch_try: HistoryMessage) -> str:
    """
    从深度研究查询历史中提取并格式化参考文献信息。

    Args:
        query_deepresearch_try (HistoryMessage): 包含查询历史和参考文献的输入消息

    Returns:
        str: 格式化后的参考文献字符串，如果没有参考文献则返回空字符串

    Notes:
        - 最多处理16条参考文献
        - 过滤掉标题长度小于等于3的参考文献
        - 参考文献按序号格式化输出
    """
    if len(query_deepresearch_try) == 0:
        return ""

    ref_source = query_deepresearch_try[-1]["ref"]
    if not ref_source or len(ref_source) == 0:
        return ""
    # ref="#### 参考文献：\n\n"
    ref = ""
    count = 0
    all_ref_text = set()
    all_ref = []
    for idx, ele_x in enumerate(ref_source):
        for idy, ele_y in enumerate(ele_x):            
            if count >= 16:
                break
            if len(ele_y["title"]) > 3:
                # ref += "[" + ele_y["title"] + "](" + "" + ele_y["url"] + ")" + "\n\n"
                if ele_y["title"] not in all_ref_text:
                    all_ref.append(ele_y)
                    all_ref_text.add(ele_y["title"])
                count += 1
    logger.info(f"speed deep research 提取到的 rag all_ref 数量: {len(all_ref)}")
    for idx, ele_y in enumerate(all_ref):
        # ref += "[" + str(idx + 1) + "]" + ele_y["title"] + "\n"
        ref += "[" + str(idx + 1) + "]" + ele_y["title"] + "\n"
    logger.info(f"speed deep research 最后的 ref: {ref}")

    return ref


# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    from deep_research.prompts import build_quality_judgment_prompt

    sample_answer = "根据《民法典》第1165条，行为人因过错侵害他人民事权益造成损害的，应当承担侵权责任。本案中，被告未尽到合理注意义务，构成过失，应赔偿原告损失。"

    model_name = "judge_model"
    url = "http://localhost:8008/v1/chat/completions"
    judgment = send_request_to_model(
        user_query=sample_answer,
        prompt_builder=build_quality_judgment_prompt,
        model_name=model_name,
        api_url=url,
    )

    if judgment:
        # 可进一步解析 #### 后的得分
        lines = judgment.strip().split("\n")
        score_line = next((line for line in lines if line.startswith("####")), None)
        if score_line:
            try:
                score = float(score_line.replace("####", "").strip())
                print(f"\n最终得分: {score:.3f}")
            except ValueError:
                print("无法解析得分")
    # 使用
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "/sharedata/mdl/Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True
    )
    tokens = safe_tokenize(tokenizer, "大模型开源榜单有哪些，详细的分析")
    print("分词结果：", tokens)

    import re

    strict_pat = r"^(?:\s*(?:总得分|得分)\s*[:：]?\s*)?(\d+\.\d+)\s*$"

    for s in ["0.80", "总得分：0.94", "  得分:0.77  ", "结果0.80"]:
        match = re.fullmatch(strict_pat, s)
        print(f"{s!r} → {match.group(1) if match else None}")

    pat = r"(?:(?:总得分|得分)\s*[:：]?\s*)?(\d+\.\d+)"

    test_cases = [
        "总得分：0.80",
        "得分: 0.95",
        "0.60",
        "结果是0.75",  # 注意：这个会匹配到 0.75
        "分数为 0.88 分",  # 也会匹配 0.88
    ]

    for s in test_cases:
        match = re.search(pat, s)
        print(f"{s!r} → {match.group(1) if match else None}")
