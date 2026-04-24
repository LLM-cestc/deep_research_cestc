# -*- coding: utf-8 -*-  # noqa: UP009
"""
deepresearch agent 模块（通用写作版，无 RAG / 无联网）

Author: wjianxz
Date: 2025-11-13
"""
import json
import queue
import logging

from deep_research.local_logger import Timer, model_request_error, timing
from deep_research.parser_config import load_validated_config, AppConfig
from deep_research.utils import (
    send_request_to_model,
    send_request_to_model_streaming,
    check_llm_output,
)
from deep_research.prompts import (
    build_deep_research_session_chat_prompt,
    build_deep_research_beautiful_format_prompt,
    build_bid_beautiful_format_prompt,
)
from deep_research.protocal import (
    HistoryMessage,
    QueryResult,
    DeepResearchResult,
)
from deep_research.plan_pipline_search import deep_research_agent

logger = logging.getLogger(__name__)


@timing
def deep_search_rag(
    user_query: str,
    mode: str,
    result_queue: queue.Queue,
    history_message: list[HistoryMessage],
    config: AppConfig,
) -> str:
    """
    Deep Research 写作主流程（无 RAG、无联网，纯模型能力）。
    mode == "deep_research" 时进入多步写作管线；
    其他模式走单次流式对话。
    """
    logger.info("=== 进入 deep_search_rag 函数 ===")

    query_result: QueryResult = {
        "query": user_query,
        "rewrite": "",
        "ref": [],
        "score": [],
        "sub_query": [],
        "answer": [],
    }

    deep_research_query_result: DeepResearchResult = {
        "depth": 0,
        "query": user_query,
        "topicreport": "",
        "sub_query": [],
        "subtopicreport": [],
        "answerreport": "",
        "ref": [],
        "topicreportscore": -1.0,
        "subdeepresearch": [],
    }

    logger.info("用户原始查询问题: %r", user_query)

    # ── 模式整形 ──────────────────────────────────────────────
    if (
        config.pattern.select_pattern in ("deep_research", "bid_generation")
        or mode in ("deep_research", "bid_generation")
    ):
        config.maxhop.turn_on_deepresearch = True
        config.maxhop.norefturn = True
        if mode == "bid_generation":
            logger.info("当前模式为 AI评标模式（标书生成，无RAG、无联网）")
        else:
            logger.info("当前模式为 deep Research 写作模式（无RAG、无联网）")
    else:
        config.maxhop.turn_on_deepresearch = False
        config.maxhop.norefturn = True
        logger.info("当前模式为 chat 写作模式（无RAG、无联网）")

    config.pattern.select_pattern = mode

    # ── 分支 A: Deep Research 多步写作管线 ────────────────────
    if config.maxhop.turn_on_deepresearch:
        logger.info("========= 进入 Deep Research 写作模块 ========")

        deep_research_agent(
            0, 0, user_query, "",
            deep_research_query_result,
            result_queue, history_message, config,
        )

        logger.info(
            "Deep Research 写作结果: \n %s",
            json.dumps(deep_research_query_result, ensure_ascii=False),
        )

        # 先提示进入整合/润色阶段（该阶段可能耗时较长）
        result_queue.put(
            [
                False,
                "<br>\n📄 各章节撰写完成，正在整合润色全文……\n\n<br>",
            ]
        )

        if config.maxhop.skip_final_polish_llm:
            logger.info("skip_final_polish_llm=True，跳过文末润色模型")
            model_output = deep_research_query_result["answerreport"] or ""
        else:
            final_format_prompt_builder = (
                build_bid_beautiful_format_prompt
                if mode == "bid_generation"
                else build_deep_research_beautiful_format_prompt
            )
            # 润色阶段改为流式输出，避免卡住时用户无感知
            model_output = send_request_to_model_streaming(
                user_query=deep_research_query_result["answerreport"],
                references=deep_research_query_result["ref"],
                prompt_builder=final_format_prompt_builder,
                result_queue=result_queue,
                history_answer="",
                model_name=config.deepresearch.name,
                api_url=config.deepresearch.server,
                temperature=config.deepresearch.temperature,
                max_tokens=config.deepresearch.max_tokens,
                top_p=config.deepresearch.top_p,
                repetition_penalty=config.deepresearch.repetition_penalty,
            )

            if not model_output:
                logger.warning("润色模型返回空，直接使用原始报告输出")
                model_output = deep_research_query_result["answerreport"]

        result_queue.put([True, model_output if model_output else ""])

        logger.info("=== 退出 deep_search_rag 函数 ===")
        return deep_research_query_result["answerreport"]

    # ── 分支 B: 单次流式对话（chat 模式）─────────────────────
    with Timer(name="首次大模型请求", logger=logger, level=logging.INFO):
        answer = send_request_to_model_streaming(
            user_query=user_query,
            prompt_builder=build_deep_research_session_chat_prompt,
            result_queue=result_queue,
            history_answer="",
            references=None,
            model_name=config.speeddeepresearch.name,
            api_url=config.speeddeepresearch.server,
            temperature=config.speeddeepresearch.temperature,
            max_tokens=config.speeddeepresearch.max_tokens,
            top_p=config.speeddeepresearch.top_p,
            repetition_penalty=config.speeddeepresearch.repetition_penalty,
        )
        if not check_llm_output(
            answer, function_name="chat_prompt", query=user_query
        ):
            logger.error("模型请求失败，终止流程。")
            return model_request_error()
        answer = answer if isinstance(answer, str) else ""

    query_result["answer"].append(answer)
    query_result["score"].append(0)

    result_queue.put([True, answer])

    logger.info("=== 退出 deep_search_rag 函数 ===")
    return answer
