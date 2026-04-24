# -*- coding: utf-8 -*-  # noqa: UP009
"""
系统 data protocal 模块：定制化各种data protocal

Author: wjianxz
Date: 2025-11-13
"""

from __future__ import annotations

from typing import TypedDict


class ReferenceItem(TypedDict):
    """一个用于表示参考条目的 TypedDict 类，主要用于存储和管理文档或网页的引用信息。

    核心功能：
    - 存储参考条目的基本信息（标题、URL）
    - 保存原始内容和提取后的内容
    - 记录条目的相关性和专业深度评分

    代码示例：
    >>> ref = ReferenceItem(
    ...     title="示例文档",
    ...     url="https://example.com",
    ...     content="原始内容",
    ...     extracted_content="提取后的内容",
    ...     relevance_score=0.8,
    ...     prof_depth_score=0.7
    ... )

    注意事项：
    - 这是一个 TypedDict 类型，主要用于类型提示，实例化时需要提供所有必需的字段
    - 所有字段都是必需的，且类型必须严格匹配定义
    - relevance_score 和 prof_depth_score 应在 0-1 范围内
    """
    title: str
    url: str
    #content: Union[str, ReferenceItem]  # 假设 content 可嵌套
    content: str  # 假设 content 可嵌套
    extracted_content: str
    relevance_score: float
    prof_depth_score: float


class SubQueryItem(TypedDict):
    """
    表示子查询项的数据结构类，用于存储子问题文本及其关联的参考文献。
    
    核心功能：
    - 存储子问题的文本内容
    - 保存与子问题相关的参考文献列表
    
    示例用法：
    >>> sub_query = SubQueryItem(
    ...     sub_query="什么是机器学习？",
    ...     ref=[{"title": "机器学习基础", "author": ""}]
    ... )
    
    属性说明：
    - sub_query: str类型，表示子问题的文本内容
    - ref: 可选参数，list[ReferenceItem]类型或None，表示与子问题相关的参考文献列表
    
    使用限制：
    - sub_query字段为必填项
    - ref字段可以为None或空列表
    """
    sub_query: str  # 子问题文本
    ref: list[ReferenceItem] | None  # 关联的参考文献


class QueryResult(TypedDict):
    """查询结果的数据结构类，用于存储和传递查询处理后的完整信息。
    
    该类继承自TypedDict，定义了查询结果的标准数据格式，包含原始查询、改写后的查询、
    参考文献列表、评分、子查询和答案等关键信息。
    
    核心功能：
    - 存储查询处理过程中的所有相关数据
    - 支持多轮对话的参考文献和评分管理
    - 保存子查询和最终答案信息
    
    Attributes:
        query (str): 原始查询字符串
        rewrite (str): 改写后的查询字符串
        ref (list[list[ReferenceItem]]): 多轮对话的参考文献列表，每个子列表包含一轮对话的参考文献
        score (list[float]): 每轮对话的评分列表
        sub_query (list[SubQueryItem]): 子查询列表，包含查询分解后的子查询信息
        answer (list[str]): 答案列表，包含每轮对话的答案
    
    示例:
        result = QueryResult(
            query="原始查询",
            rewrite="改写后的查询",
            ref=[[ref1, ref2], [ref3]],
            score=[0.8, 0.7],
            sub_query=[sub_query1, sub_query2],
            answer=["答案1", "答案2"]
        )
    
    注意:
        - 该类作为TypedDict使用，创建实例时需要提供所有必需的字段
        - ref和score列表的长度应该保持一致，每个score对应ref中同一轮对话的评分
    """
    query: str
    rewrite: str
    ref: list[list[ReferenceItem]]  # 多轮参考文献列表
    score: list[float]  # 每轮 score
    sub_query: list[SubQueryItem]
    answer: list[str]


class DeepResearchResult(TypedDict):
    """
    一个用于存储深度研究结果的数据结构类，继承自TypedDict。
    该类用于组织和管理多轮深度研究过程中的各种信息，包括查询、报告、参考文献等。

    核心功能：
    - 存储研究的深度级别
    - 保存原始查询和子查询
    - 管理各级别的研究报告
    - 维护参考文献列表
    - 支持嵌套的深度研究结果

    使用示例：
    result = DeepResearchResult(
        depth=1,
        query="人工智能的发展",
        topicreport="AI发展报告",
        sub_query=[{"query": "机器学习"}, {"query": "深度学习"}],
        subtopicreport=["机器学习报告", "深度学习报告"],
        answerreport="综合分析报告",
        ref=[{"title": "AI论文", "url": "http://example.com"}],
        topicreportscore=0.85,
        subdeepresearch=[]
    )

    构造函数参数：
    - depth (int): 研究的深度级别
    - query (str): 原始查询内容
    - topicreport (str): 主题研究报告
    - sub_query (list[SubQueryItem]): 子查询列表
    - subtopicreport (list[str]): 子主题报告列表
    - answerreport (str): 最终答案报告
    - ref (list[ReferenceItem]): 参考文献列表
    - topicreportscore (float): 主题报告的评分
    - subdeepresearch (list["DeepResearchResult"]): 嵌套的深度研究结果列表

    注意事项：
    - 由于继承自TypedDict，所有字段都是可选的，但在实际使用中建议提供完整的数据
    - subdeepresearch字段允许递归嵌套，用于表示多层级的研究结构
    """
    depth: int
    query: str
    topicreport: str
    sub_query: list[SubQueryItem]
    subtopicreport: list[str]  # 子问题
    answerreport: str
    ref: list[ReferenceItem]  # 多轮参考文献列表
    topicreportscore: float  # 每轮 score
    subdeepresearch: list["DeepResearchResult"]


ReferenceList = list[ReferenceItem]
HistoryMessage = list[QueryResult]


