# -*- coding: utf-8 -*-  # noqa: UP009
"""
deepresearch agent模块

Author: wjianxz
Date: 2025-11-13
"""
from typing import List, Tuple
import copy

from deep_research.protocal import (
    HistoryMessage,
    QueryResult,
)
import queue
from deep_research.retrieve_knowledge import retrieve_from_knowledge_base
from deep_research.utils import send_request_to_model, send_request_to_model_dr
from deep_research.parser_config import AppConfig
from deep_research.prompts import (
    build_deep_research_report_topic_prompt,
    build_deep_research_report_subtopic_prompt,
    build_deep_research_report_prompt,
    build_deep_research_sub_report_prompt,
    build_sub_query_prompt,
    build_confidence_prompt,
    build_deep_research_prompt,
    build_deep_research_noref_prompt,
)
from deep_research.utils import parser_sub_topic_output
from deep_research.assemble_rag_context import (
    extract_info_from_kb,
    extract_info_from_web_deep_research,
    segment_partial_content,
)
# from deep_research.deep_web_search import perform_web_search  # 已禁用联网功能
from deep_research.protocal import DeepResearchResult, SubQueryItem, ReferenceItem
from deep_research.utils import (
    check_llm_output,
    parser_confidence,
    extract_sub_question,
)
from deep_research.local_logger import timing, Timer
from deep_research.format_answer import message_subquery, message_deepsearch_react

import logging

logger = logging.getLogger(__name__)  # 自动继承 root logger 的 handlers


@timing
def mutil_hop_search(
    query,
    query_result: QueryResult,
    result_queue: queue.Queue,
    depth,
    max_hops,
    config: AppConfig,
    epochs: int = 0,
) -> Tuple[bool, QueryResult]:
    logger.info("第：%d epochs + depth为：%d, 模型开始深度思考和 React ", epochs, depth)

    # 计算confidence
    last_answer = query_result["answer"][-1].replace(" ", "").split("</think>")[-1]
    with Timer(name="confidence计算", logger=logger, level=logging.INFO):
        logger.info("第：%d epochs + depth为：%d，进行confidence计算；", epochs, depth)
        confidence_req = send_request_to_model(
            user_query=query,
            history_answer=last_answer,
            references=None,
            prompt_builder=build_confidence_prompt,
            model_name=config.confidence.name,
            api_url=config.confidence.server,
        )
        if not check_llm_output(
            confidence_req,
            function_name="build_confidence_prompt" + f"_depth_{depth}",
            query=query,
        ):
            return False, query_result
    # send_request_to_model may return None; ensure we pass a str to parser_confidence
    if confidence_req is None:
        confidence_req = ""

    confidence_flag, match_score = parser_confidence(
        confidence_req, config.confidence.score
    )
    logger.info(
        "第：%d epochs + depth为：%d，计算得到的confidence得分: %f",
        epochs,
        depth,
        match_score,
    )
    query_result["score"][-1] = match_score
    if confidence_flag:
        return True, query_result

    # 假设输出结果
    if config.maxhop.upspeed:
        if depth >= 2 and match_score >= config.maxhop.secondconfscore:
            return True, query_result
        if depth >= 3 and match_score >= config.maxhop.secondconfscore:
            return True, query_result
    if depth > max_hops:
        logger.info(
            "第：%d epochs + depth为：%d，达到最大深度，终止多轮搜索；", epochs, depth
        )
        return False, query_result

    # subquery + 子问题重复用户问题/可能是answer不满足用户需求/需要重复提问
    if len(query_result["sub_query"]) > 0:
        temp_sub_query = joint_query_and_subquery(query_result)
    else:
        temp_sub_query = query

    # subquery + 子问题重复用户问题/可能是answer不满足用户需求/需要重复提问
    with Timer(name="大模型请求计算 subquery ", logger=logger, level=logging.INFO):
        sub_query_question = send_request_to_model(
            user_query=temp_sub_query,
            references=last_answer,
            prompt_builder=build_sub_query_prompt,
            model_name=config.subquestion.name,
            api_url=config.subquestion.server,
        )
    #
    sub_query: SubQueryItem = {
        "sub_query": "",
        "ref": [],
    }
    logger.info(
        "第：%d epochs + depth为：%d，子问题模型输出结果: %r",
        epochs,
        depth,
        sub_query_question,
    )
    #
    sub_query["sub_query"] = extract_sub_question(sub_query_question)
    logger.info(
        "第：%d epochs + depth为：%d，提取到的子问题: %r",
        epochs,
        depth,
        sub_query["sub_query"],
    )
    if len(sub_query["sub_query"].strip()) < 6:
        logger.info(
            "第：%d epochs + depth为：%d，子问题提取过短或者已经回答用户问题， 但是任然没有回答用户问题需要重新组装查询逻辑;",
            epochs,
            depth,
        )
        sub_query["sub_query"] = query
        # no_sub_query_flag= True
        # return False, query_result

    if sub_query_question is not None and depth < 2:
        result_queue.put([False, message_subquery() + "......<br>"])  # 流式输出测试
        result_queue.put(
            [False, "<br>" + sub_query["sub_query"] + "......<br>"]
        )  # 流式输出测试

    # 外部深度搜索
    deep_research_prompt = build_deep_research_prompt
    logger.info(
        "第：%d epochs + depth为：%d, 开始执行外部深度搜索，系统设置最大url结果数为：%d",
        epochs,
        depth,
        config.maxhop.urlmaxnum,
    )
    kb_reference_list: List[ReferenceItem] = []
    if not config.maxhop.norefturn:  # 走search
        with Timer(
            name=f"需要爬取网站信息, 当前的_epochs_{epochs}，当前的_depth_{depth}",
            logger=logger,
            level=logging.INFO,
        ):
            # 执行外部深度搜索
            # reference_list = perform_web_search(
            #     sub_query["sub_query"],
            #     max_results=config.maxhop.urlmaxnum,
            #     epochs=epochs,
            #     depth=depth,
            # )
            # 执行内部向量知识库检索
            kb_results = retrieve_from_knowledge_base(sub_query["sub_query"], config=config)
            if kb_results is not None and len(kb_results) > 0:
                logger.info(
                    "内部知识库检索共获取到 %d 条参考文献。",
                    len(kb_results),
                )
                # 将 kb_results 转换为 ReferenceItem 列表
                for item in kb_results:
                    kb_reference_list.append({"title": item.get("text", ""),"content": item.get("text", ""), "relevance_score": round(item.get("score", ""), 3), "url": "","extracted_content": item.get("text", ""), "prof_depth_score": 0.0})
        logger.info("外部深度搜索共获取到 %d 条参考文献。", len(kb_reference_list))
    else:
        deep_research_prompt = build_deep_research_noref_prompt  # 新调试对象
        reference_list = None
        logger.info("不走外部搜索，走模型自动推理")

    if not config.maxhop.norefturn and len(kb_reference_list) == 0:
        logger.error(
            "内部深度搜索未获取到任何参考文献，需要关注 Rag 抓取引擎"
            + f"当前搜索_depth_{depth}"
        )
    #     return False, query_result

    extract_ref_list = None
    if (
        not config.maxhop.norefturn
        and kb_reference_list is not None
        and len(kb_reference_list) > 0
    ):
        # extract_ref_list = extract_info_from_web(
        #     query, kb_reference_list, config
        # )  # 实际是走互联网获取
        # extract_ref_list = segment_partial_content(
        #     extract_ref_list, config.maxhop.relativescore
        # )
        extract_ref_list = extract_info_from_kb(kb_reference_list)
    #
    query_result["sub_query"].append(sub_query)
    if extract_ref_list is not None:
        sub_query["ref"] = extract_ref_list
        query_result["ref"].append(extract_ref_list)

    # ref 重新组装// 按照 score // answer 也可以重新组装优化
    # 获取结果 + 将 query + sub_query 拼接在一起;
    logger.info(
        "第：%d epochs + depth为：%d, 开始进行深度研究，回答生成问题", epochs, depth
    )
    
    # 这里面要不要将子问题也加入到 context 里面去？
    query_merge = joint_query_and_subquery(query_result)
    # query_merge = 
    history_answer = query_result["answer"]
    logger.info("将系统推理的问题和用户的问题进行合并；%r", query_merge)

    with Timer(name="大模型请求计算 deep research ", logger=logger, level=logging.INFO):
        current_answer = send_request_to_model(
            user_query=query_merge,
            history_answer=history_answer,
            references=extract_ref_list,
            prompt_builder=deep_research_prompt,
            model_name=config.deepresearch.name,
            api_url=config.deepresearch.server,
        )
    #
    if not check_llm_output(
        current_answer,
        function_name="build_deep_research_prompt" + f"_depth_{depth}",
        query=query,
    ):
        return False, query_result

    # 检测返回值，如果为 None 或非字符串，则赋值为空字符串
    current_answer = current_answer if isinstance(current_answer, str) else ""
    query_result["answer"].append(current_answer)
    query_result["score"].append(0)
    # 流式配置
    result_queue.put([False, message_deepsearch_react()])  # 流式输出测试
    result_queue.put([False, current_answer[:888] + "......<br>"])  # 流式输出测试

    # 递归调用
    logger.info(
        "第：%d epochs + depth为：%d，当前轮次大模型输出信息如下:\n %s",
        epochs,
        depth,
        current_answer,
    )
    flag, query_result_cp = mutil_hop_search(
        query, query_result, result_queue, depth + 1, max_hops, config, epochs=epochs
    )
    return flag, query_result_cp


def deep_research_agent(
    depth: int,
    wide: int,
    user_query: str,
    far_topic: str,
    deep_research_query_result: DeepResearchResult,
    result_queue: queue.Queue,
    history_message: list[HistoryMessage],
    config: AppConfig,
) -> DeepResearchResult:
    """
    user_query, query_result, result_queue, history_message, config
    deep research agent模块

    Args:
        query (str): 用户查询
        history (list[HistoryMessage]): 历史信息

    Returns:
        QueryResult: 查询结果

    depth 从1开始

    """
    # 实现deep research agent模块
    if depth >= config.maxhop.deepresearch_max_depth:
        # deep_research_query_result["answerreport"] = deep_research_query_result["subtopicreport"][0]
        return deep_research_query_result

    deep_research_query_result["depth"] = depth
    deep_research_query_result["query"] = user_query

    # 1. 解析用户查询，生成叶子节点查询列表
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

    # 1. 解析用户查询 Deep research 走不走session 模块;
    logger.info(
        "Deep research agent 模块开始处理深度为 %d, 广度为 %d 的查询", depth, wide
    )
    # logger.info("系统配置需要考虑步长 %d session 历史消息为: %s",config.maxhop.session, json.dumps(session_message, ensure_ascii=False))

    # 2. 生成主题报告
    # 生成主题报告topic框架
    topic_content = send_request_to_model_dr(
        user_query=user_query,
        history_answer=far_topic,
        references=None,
        prompt_builder=build_deep_research_report_topic_prompt,
        model_name=config.session.name,
        api_url=config.session.server,
    )
    if topic_content is None:
        result_queue.put([False, "<br>模型出现异常......<br>"])  # 流式输出测试
        logger.error("主题策略模型输出结果为空, 模型出现异常")
        return deep_research_query_result
    else:
        logger.info(
            "深度为 %d, 广度为 %d 的主题策略模型输出结果为: %r",
            depth,
            wide,
            topic_content,
        )
        deep_research_query_result["topicreport"] = topic_content

        result_queue.put(
            [False, "<br>为了回答用户的问题，现在将思路整理如下......\n<br>"]
        )  # 流式输出测试
        result_queue.put([False, topic_content])

    far_topic = topic_content

    # 生成子主题/问题列表
    sub_topic_content = send_request_to_model(
        user_query=user_query,
        history_answer=topic_content,
        references=None,
        prompt_builder=build_deep_research_report_subtopic_prompt,
        model_name=config.session.name,
        api_url=config.session.server,
    )

    if sub_topic_content is None:
        logger.error("子主题输出结果为空，出现异常")
        deep_research_query_result["answerreport"] = deep_research_query_result[
            "subtopicreport"
        ][0]
        return deep_research_query_result

    logger.info(
        "Deep research agent 模块开始处理深度为 %d, 广度为 %d 的查询, 子主题策略模型输出结果为: %r",
        depth,
        wide,
        sub_topic_content,
    )

    sub_topic_query = parser_sub_topic_output(sub_topic_content)
    logger.info("子主题策略模型 抽出的 子问题如下: %r", sub_topic_query)

    result_queue.put([False, "\n<br>根据上面的思路，下面深入思考下面的问题......<br>"])
    for idx, sub_query in enumerate(sub_topic_query):
        result_queue.put([False, "<br>" + sub_query + "<br>"])

    if len(sub_topic_query) == 0:
        logger.error("子主题策略模型 抽出的 子问题为空 返回")
        return deep_research_query_result

    for idx, sub_query in enumerate(sub_topic_query):
        # 已禁用联网功能，返回空列表
        # reference_list = perform_web_search(
        #     sub_query, max_results=config.maxhop.urlmaxnum, epochs=0, depth=0
        # )
        reference_list = []
        temp_query: SubQueryItem = {"sub_query": sub_query, "ref": None}
        if reference_list is not None and len(reference_list) > 0:
            logger.info(
                "deep research 处理深度为 %d, 广度为 %d 的查询, 首次 外部深度搜索共获取到 %d 条参考文献。",
                depth,
                wide,
                len(reference_list),
            )
            extract_ref_list = extract_info_from_web_deep_research(
                user_query, reference_list, config
            )
            extract_ref_list = segment_partial_content(extract_ref_list)
            temp_query["ref"] = extract_ref_list
            temp_dr_query_result["query"] = sub_query
            temp_dr_query_result["ref"] = reference_list
            temp_dr_query_result["sub_query"].append(copy.deepcopy(temp_query))
            deep_research_query_result["sub_query"].append(copy.deepcopy(temp_query))

            sub_topic_content = send_request_to_model_dr(
                user_query=sub_query,
                history_answer=None,
                references=extract_ref_list,
                prompt_builder=build_deep_research_sub_report_prompt,
                model_name=config.session.name,
                api_url=config.session.server,
            )
            #
            if sub_topic_content is None:
                logger.error("子主题策略模型输出结果为空")
                sub_topic_content = ""
            else:
                logger.info("子主题策略模型输出结果为: %r", sub_topic_content)

            result_queue.put(
                [
                    False,
                    "\n下面对子问题进行Deep Thinking......<br>" + sub_query + "<br>\n",
                ]
            )
            result_queue.put([False, sub_topic_content[:888] + "......<br>"])

            temp_dr_query_result["answerreport"] = sub_topic_content
            deep_research_query_result["subtopicreport"].append(sub_topic_content)
            deep_research_query_result["subdeepresearch"].append(
                copy.deepcopy(temp_dr_query_result)
            )

            # logger.info("deep research 处理深度为 %d, 广度为 %d 的查询,系统报错位置: %s", depth, wide,json.dumps(deep_research_query_result, ensure_ascii=False))
            # 子主题质量高/不需要进行细分/否则细分-- judge
            deep_research_agent(
                depth + 1,
                idx,
                sub_query,
                far_topic,
                deep_research_query_result["subdeepresearch"][idx],
                result_queue,
                history_message,
                config,
            )
        else:
            logger.info(
                "deep research 处理深度为 %d, 广度为 %d 的查询, 爬虫失败未获取到数据",
                depth,
                wide,
            )
            temp_dr_query_result["query"] = sub_query
            temp_dr_query_result["ref"] = []
            temp_dr_query_result["sub_query"].append(copy.deepcopy(temp_query))
            deep_research_query_result["sub_query"].append(copy.deepcopy(temp_query))

            sub_topic_content = send_request_to_model_dr(
                user_query=sub_query,
                history_answer=None,
                references=None,
                prompt_builder=build_deep_research_sub_report_prompt,
                model_name=config.session.name,
                api_url=config.session.server,
            )
            #
            if sub_topic_content is None:
                logger.error("子主题策略模型输出结果为空")
                sub_topic_content = ""
            else:
                logger.info("子主题策略模型输出结果为: %r", sub_topic_content)

            result_queue.put(
                [
                    False,
                    "\n下面对子问题进行Deep Thinking......<br>" + sub_query + "<br>\n",
                ]
            )
            result_queue.put([False, sub_topic_content[:888] + "......<br>"])

            temp_dr_query_result["answerreport"] = sub_topic_content
            deep_research_query_result["subtopicreport"].append(sub_topic_content)
            deep_research_query_result["subdeepresearch"].append(
                copy.deepcopy(temp_dr_query_result)
            )
            deep_research_agent(
                depth + 1,
                idx,
                sub_query,
                far_topic,
                deep_research_query_result["subdeepresearch"][idx],
                result_queue,
                history_message,
                config,
            )

    # 合并生成报告
    history_merge, _ = joint_subquery_report(
        depth, wide, deep_research_query_result["subdeepresearch"]
    )
    total_reference_list = get_total_reference_list(
        deep_research_query_result["subdeepresearch"]
    )

    gen_topic_report = send_request_to_model_dr(
        user_query=user_query,
        topic_report=deep_research_query_result["topicreport"],
        history_answer=history_merge,
        references=None,
        prompt_builder=build_deep_research_report_prompt,
        model_name=config.session.name,
        api_url=config.session.server,
    )

    if gen_topic_report is None:
        logger.error("主题报告模型输出结果为空, 不需要考虑session历史消息, 开启新会话")
        gen_topic_report = ""
    else:
        logger.info("主题报告模型输出结果为: %r", gen_topic_report)

    # 对最后的报告进行质量把控
    confidence_flag = False
    if depth == 0:
        confidence_req = send_request_to_model(
            user_query=user_query,
            history_answer=gen_topic_report,
            references=None,
            prompt_builder=build_confidence_prompt,
            model_name=config.confidence.name,
            api_url=config.confidence.server,
        )
        if not check_llm_output(
            confidence_req,
            function_name="build_confidence_prompt" + f"_depth_{depth}",
            query=user_query,
        ):
            confidence_flag = False
        # send_request_to_model may return None; ensure we pass a str to parser_confidence
        if confidence_req is None:
            confidence_req = ""
            confidence_flag = False

        confidence_flag, match_score = parser_confidence(
            confidence_req, config.maxhop.deepresearch_conf_score
        )

        while (
            match_score < config.maxhop.deepresearch_conf_score or not confidence_flag
        ):
            logger.info(
                "最终报告质量不达标，得分为 %f，需要重新生成最终报告", match_score
            )
            gen_topic_report = send_request_to_model_dr(
                user_query=user_query,
                topic_report=deep_research_query_result["topicreport"],
                history_answer=history_merge,
                references=None,
                prompt_builder=build_deep_research_report_prompt,
                model_name=config.session.name,
                api_url=config.session.server,
            )

            confidence_req = send_request_to_model(
                user_query=user_query,
                history_answer=gen_topic_report,
                references=None,
                prompt_builder=build_confidence_prompt,
                model_name=config.confidence.name,
                api_url=config.confidence.server,
            )
            if not check_llm_output(
                confidence_req,
                function_name="build_confidence_prompt" + f"_depth_{depth}",
                query=user_query,
            ):
                confidence_flag = False

            if confidence_req is None:
                confidence_req = ""
                confidence_flag = False
            confidence_flag, match_score = parser_confidence(
                confidence_req, config.maxhop.deepresearch_conf_score
            )
    if gen_topic_report is None:
        logger.error("主题报告模型输出结果为空, 不需要考虑session历史消息, 开启新会话")
        gen_topic_report = ""

    deep_research_query_result["answerreport"] = gen_topic_report
    deep_research_query_result["ref"] = total_reference_list
    deep_research_query_result["topicreportscore"] = match_score

    # result_queue.put([False, "<br>" + gen_topic_report + "<br>"])
    return deep_research_query_result


def get_total_reference_list(
    deep_research_query_result: list[DeepResearchResult],
) -> list[ReferenceItem]:
    """
    从深度研究查询结果中提取并合并所有参考文献列表。

    Args:
        deep_research_query_result (list[DeepResearchResult]): 深度研究查询结果列表，
            其中每个元素都包含参考文献信息。

    Returns:
        list[ReferenceItem]: 合并后的参考文献列表，包含所有输入结果中的参考文献。

    Example:
        >>> results = [{"ref": [ref1, ref2]}, {"ref": [ref3]}]
        >>> get_total_reference_list(results)
        [ref1, ref2, ref3]
    """
    total_reference_list: List[ReferenceItem] = []
    for ele in deep_research_query_result:
        total_reference_list.extend(ele["ref"])
    return total_reference_list


def joint_subquery_report(
    depth: int, wide: int, deep_research_query_result: list[DeepResearchResult]
) -> tuple[str, str]:
    """
    合并子问题报告并生成历史记录字符串。

    Args:
        depth (int): 深度研究的层级深度
        wide (int): 深度研究的广度
        deep_research_query_result (list[DeepResearchResult]): 深度研究查询结果列表，每个结果包含以下字段：
            - query (str): 查询内容
            - topicreport (str): 主题报告
            - sub_query (list[SubQueryItem]): 子问题列表
            - subtopicreport (list[str]): 子问题报告列表
            - answerreport (str): 回答报告
            - ref (list[list[ReferenceItem]]): 多轮参考文献列表
            - topicreportscore (float): 每轮得分

    Returns:
        tuple[str, str]: 返回一个元组，包含：
            - str: 合并后的历史记录字符串
            - str: 空字符串（预留字段）
    """
    # 合并子问题报告
    level_deep_research_query_result = deep_research_query_result
    history_merge = "系统推理认为需要考虑以下子问题：\n"
    for idx, ele in enumerate(level_deep_research_query_result):
        history_merge = (
            history_merge
            + f"【第{idx+1}问题】："
            + ele["query"]
            + f"\n【系统返回第{idx+1}回答】："
            + ele["answerreport"]
            + "\n"
        )
    logger.info(f"下面是deep research 抽取到的历史信息: \n{history_merge}\n")
    return history_merge, ""


def joint_query_and_subquery(query_result: QueryResult) -> str:
    """
    将用户的原始查询和系统生成的子查询合并成一个格式化的字符串。

    Args:
        query_result (QueryResult): 包含查询结果的字典，包含以下键：
            - "query": 用户的原始查询字符串
            - "sub_query": 子查询列表，每个子查询是包含"sub_query"键的字典

    Returns:
        str: 格式化后的字符串，包含所有子查询和原始查询，格式如下：
            "系统推理认为需要考虑以下子问题：
            第1子问题：{子查询1}
            第2子问题：{子查询2}
            ...
            用户原问题：{原始查询}"
    """
    query_merge = "系统推理认为需要考虑以下子问题：\n"
    query = query_result["query"]
    sub_querylist = query_result["sub_query"]
    for idx, ele in enumerate(sub_querylist):
        query_merge = query_merge + f"第{idx+1}子问题：" + ele["sub_query"] + "\n"
    query_merge = query_merge + f"用户原问题：{query}"
    return query_merge


def joint_history_message(
    partial_history_message: list[HistoryMessage],
) -> tuple[str, str, str]:
    """合并历史消息信息。

    将输入的历史消息列表进行合并，生成三种格式的字符串：
    1. 包含用户问题和系统回答的完整历史记录
    2. 仅包含所有用户问题的合并字符串
    3. 仅包含所有系统回答的合并字符串

    Args:
        partial_history_message (list[HistoryMessage]): 历史消息列表，每个元素包含用户查询、系统回答和评分信息

    Returns:
        tuple[str, str, str]: 返回三个字符串的元组：
            - history_merge: 包含格式化的用户问题和系统回答的完整历史记录
            - total_query: 所有用户问题的合并字符串
            - total_answer: 所有系统回答的合并字符串

    Note:
        - 对于每个历史消息，选择评分最高的回答
        - 使用logger记录合并后的历史信息
    """
    # 合并历史信息
    history_merge = "下面是用户提的问题和系统返回的答案：\n"
    total_query = ""
    total_answer = ""
    for idx, ele_history in enumerate(partial_history_message):
        max_index = ele_history[-1]["score"].index(max(ele_history[-1]["score"]))
        history_merge = (
            history_merge
            + f"【用户查询的第{idx+1}问题】："
            + ele_history[-1]["query"]
            + f"\n【系统返回的第{idx+1}回答】："
            + ele_history[-1]["answer"][max_index]
            + "\n"
        )
        total_query = total_query + ele_history[-1]["query"]
        total_answer = total_answer + ele_history[-1]["answer"][max_index]
        # total_query = total_query +f"【用户查询的第{idx+1}问题】："+ ele_history[-1]["query"]
    logger.info(f"下面是抽取到的历史信息: \n{history_merge}\n")
    logger.info(f"下面是抽取到的历史信息: \n{total_query}\n")
    
    return history_merge, total_query, total_answer
