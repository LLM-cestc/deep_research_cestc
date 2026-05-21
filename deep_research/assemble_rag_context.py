"""
rag内容集成模块：集成各种并行生成的内容

Author: wjianxz
Date: 2025-11-13
"""
import datetime
from typing import Any, List, Union

from deep_research.extract_information_from_web import  clean_crawled_text
from deep_research.parser_config import AppConfig
from deep_research.prompts import build_extract_info_prompt
from deep_research.utils import send_request_to_model, parser_extracted_output
from deep_research.protocal import ReferenceItem, ReferenceList
import logging
logger = logging.getLogger(__name__)  # 自动继承 root logger 的 handlers

def extract_info_from_web_deep_research(
    query: str,
    ref_list: Union[ReferenceList, None],
    config: AppConfig,
) -> Union[ReferenceList, None]:
    """
    从参考文献列表中提取与查询相关的信息，并更新每项的 extracted_content 和 relevance_score。

    Args:
        query: 用户原始查询。
        ref_list: 参考文献列表，每个元素为包含 'content' 等字段的字典。
        config: 应用配置对象，包含模型和 API 信息。

    Returns:
        更新后的 ref_list，每项新增或覆盖 'extracted_content' 和 'relevance_score' 字段。
    """
    logger.info(f"由于html 信息杂乱，利用大模型，开始从参考文献中提取信息，查询: {query}")
    if ref_list is None:
        return ref_list
    
    for ele in ref_list:

        # 清洗爬取的文本内容
        logger.info(f"这是原网页内容：\n\n {ele['content']} \n\n")
        ele_clean = clean_crawled_text(ele["content"])
        logger.info(f"未经过大模型清洗之前的文本：\n\n {ele_clean} \n\n")
        if len(ele_clean) <200 or len(ele["title"])<3:
            logger.info("文本过短不需要就行内容判断，低质量文本直接跳过")
            ele["extracted_content"] = ""
            ele["relevance_score"] = 0
            continue

        # 调用大模型提取信息
        extract_info_rlt = send_request_to_model(
            user_query=query,
            history_answer=ele_clean,
            references=ele_clean,
            prompt_builder=build_extract_info_prompt,
            model_name=config.beautifulformat.name, # 这里可以改成抽取模型的配置
            api_url=config.beautifulformat.server,
        )
        logger.info(f"经过大模型清洗之后的文本：\n\n{extract_info_rlt}\n\n")
        # 解析模型返回结果
        extract_info_rlt = extract_info_rlt if isinstance(extract_info_rlt, str) else ""
        extracted_content, relevance_score = parser_extracted_output(extract_info_rlt)

        # 更新当前参考项
        ele["extracted_content"] = extracted_content
        ele["relevance_score"] = relevance_score

    return ref_list

def extract_info_from_kb(kb_reference_list: List[ReferenceItem]) -> Union[ReferenceList, None]:

    return kb_reference_list
        
def extract_info_from_web(
    query: str,
    ref_list: Union[ReferenceList, None],
    config: AppConfig,
) -> Union[ReferenceList, None]:
    """
    从参考文献列表中提取与查询相关的信息，并更新每项的 extracted_content 和 relevance_score。

    Args:
        query: 用户原始查询。
        ref_list: 参考文献列表，每个元素为包含 'content' 等字段的字典。
        config: 应用配置对象，包含模型和 API 信息。

    Returns:
        更新后的 ref_list，每项新增或覆盖 'extracted_content' 和 'relevance_score' 字段。
    """
    logger.info(f"由于html 信息杂乱，利用大模型，开始从参考文献中提取信息，查询: {query}")
    if ref_list is None:
        return ref_list
    
    for ele in ref_list:

        # 清洗爬取的文本内容
        ele_clean = clean_crawled_text(ele["content"])
        logger.info(f"未经过大模型清洗之前的文本：\n\n {ele_clean} \n\n")
        if len(ele_clean) <200 or len(ele["title"])<3:
            logger.info("文本过短不需要就行内容判断，低质量文本直接跳过")
            ele["extracted_content"] = ""
            ele["relevance_score"] = 0
            continue

        # 调用大模型提取信息
        extract_info_rlt = send_request_to_model(
            user_query=query,
            history_answer=ele_clean,
            references=ele_clean,
            prompt_builder=build_extract_info_prompt,
            model_name=config.beautifulformat.name,
            api_url=config.beautifulformat.server,
        )
        logger.info(f"经过大模型清洗之后的文本：\n\n{extract_info_rlt}\n\n")
        # 解析模型返回结果
        extract_info_rlt = extract_info_rlt if isinstance(extract_info_rlt, str) else ""
        extracted_content, relevance_score = parser_extracted_output(extract_info_rlt)

        # 更新当前参考项
        ele["extracted_content"] = extracted_content
        ele["relevance_score"] = relevance_score

    return ref_list


def segment_partial_content(extract_ref_list:  Union[ReferenceList, None], relativescore: float = 0.6) -> Union[ReferenceList, None]:
    """
    对参考内容列表按相关性分数降序排序，并截取总字符数（URL + 提取内容）不超过 30,000 的前缀。

    该函数用于在上下文长度受限（如大模型输入限制）时，优先保留最相关的参考片段，
    同时确保不会超出指定的字符总数阈值。

    参数:
        extract_ref_list (ReferenceList):
            包含参考项的列表，每个项为字典，必须包含以下键：
            - 'url' (str): 来源 URL。
            - 'extracted_content' (str): 从该 URL 提取的文本内容。
            - 'relevance_score' (float 或 int): 相关性评分，用于排序。

    返回:
        List[Dict[str, str]]:
            截断后的参考项列表，仅包含 'url' 和 'extracted_content' 两个字段，
            按 relevance_score 从高到低排序，且所有项的
            len(url) + len(extracted_content) 总和 ≤ 30,000。
            若第一个条目就超出限制，则返回空列表。

    示例:
        >>> refs = [
        ...     {'url': 'https://a.com', 'extracted_content': '内容A', 'relevance_score': 0.9},
        ...     {'url': 'https://b.com', 'extracted_content': '内容B', 'relevance_score': 0.7}
        ... ]
        >>> result = segment_partial_content(refs)
        >>> print(len(result))  # 取决于总长度是否超过 30000
    """

    logger.info("对参考内容按相关性分数降序排序")

    if extract_ref_list is None :
        return extract_ref_list
    
    extract_ref_list.sort(key=lambda x: x["relevance_score"], reverse=True)

    total_length = 0
    count = 0

    for ele in extract_ref_list:
        # 累加当前项的 URL 和内容长度
        total_length += len(ele["extracted_content"])
        logger.info("大模型抽取后的数据为：%r", ele["extracted_content"])
        logger.info("相关性得分的具体值：%f", ele["relevance_score"])
        # if total_length > 30000 or ele["relevance_score"] < relativescore:
        if total_length > 30000:
            break
        count += 1

    return extract_ref_list[:count]


def assemble_rag_context(
    kb_chunks: list[dict[str, Any]], web_results: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    将内部知识库片段与外部搜索结果融合为统一的 RAG 上下文，
    便于后续送入 LLM 进行生成（generation）。

    Args:
        kb_chunks: 来自知识库的检索结果
        web_results: 来自外部搜索的结果

    Returns:
        Dict: 结构化上下文 + 元信息
    """
    return {
        "status": "success",
        "retrieved_at": datetime.datetime.now().isoformat(),
        "context": {"knowledge_base": kb_chunks, "external_search": web_results},
        "metadata": {"kb_count": len(kb_chunks), "web_count": len(web_results)},
    }
