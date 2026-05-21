# // cspell:disable
import os
import requests
import logging
from typing import List, Dict, Any

logger = logging.getLogger("RequestModel")

# =============================
# 模型与角色配置
# =============================
# 以后你只需要改这里的地址和模型名称即可。
#
# 角色：
# 1. assistant：法律回答模型（普通单轮 chat 接口）
# 2. user_simulator：用户追问模拟模型
# 3. evaluator：质量评估模型
# 4. assistant_deep：deep_research 会话式回答（不是单纯 LLM，由 deep_research 包提供
#    检索 + 多轮历史 + 引用），通过 call_deep_chat 调用，不走 chat completions URL。

ASSISTANT_LLM_URL = os.environ.get(
    "ASSISTANT_LLM_URL",
    "http://127.0.0.1:8000/v1/chat/completions",
)
USER_SIMULATOR_LLM_URL = os.environ.get(
    "USER_SIMULATOR_LLM_URL",
    "http://127.0.0.1:8000/v1/chat/completions",
)
EVALUATOR_LLM_URL = os.environ.get(
    "EVALUATOR_LLM_URL",
    "http://127.0.0.1:8000/v1/chat/completions",
)

ASSISTANT_MODEL = os.environ.get("ASSISTANT_MODEL", "Qwen3.6-27B")
USER_SIMULATOR_MODEL = os.environ.get("USER_SIMULATOR_MODEL", "Qwen3.6-27B")
EVALUATOR_MODEL = os.environ.get("EVALUATOR_MODEL", "Qwen3.6-27B")

ROLE_MODEL_CONFIG = {
    "assistant": {
        "type": "chat_completions",
        "url": ASSISTANT_LLM_URL,
        "model": ASSISTANT_MODEL,
    },
    "user_simulator": {
        "type": "chat_completions",
        "url": USER_SIMULATOR_LLM_URL,
        "model": USER_SIMULATOR_MODEL,
    },
    "evaluator": {
        "type": "chat_completions",
        "url": EVALUATOR_LLM_URL,
        "model": EVALUATOR_MODEL,
    },
    "assistant_deep": {
        "type": "deep_research_session",
        "module": "deep_research.session_chat",
        "default_mode": os.environ.get("DEEP_CHAT_DEFAULT_MODE", "deep_thinking"),
    },
}


def get_role_model_config() -> Dict[str, Dict[str, Any]]:
    """
    返回当前角色配置（含 chat 通道与 deep_research 通道）。
    主要用于记录训练数据和启动日志。
    """
    return ROLE_MODEL_CONFIG


def send_chat_completion_request(
    url: str,
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout: int = 180,
) -> Dict[str, Any] | None:
    """
    向 OpenAI 兼容的 chat completions 接口发起请求，例如本地 vLLM。

    Args:
        url: 完整地址，例如 http://127.0.0.1:8000/v1/chat/completions。
        model_name: 与服务端 --served-model-name 一致。
        messages: role / content 消息列表。
        temperature: 采样温度。
        max_tokens: 最大生成 token 数。
        timeout: 请求超时时间，单位秒。

    Returns:
        模型响应的 JSON 字典。如果请求失败，返回 None。
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"请求模型失败: url={url}, model={model_name}, error={e}")
        return None


def extract_model_content(response: Dict[str, Any] | None) -> str:
    """
    从 OpenAI 兼容接口返回结果中提取 assistant 文本。

    兼容：纯 content、多段 content、以及部分推理模型把长文放在 reasoning_content /
    reasoning 中而 content 为空的情况（evaluator 等需要完整再解析的场景）。
    """
    if not response:
        return ""

    try:
        choices = response.get("choices") or []
        if not choices:
            logger.warning("API 响应缺少 choices，顶层键: %s", list(response.keys()))
            return ""

        choice = choices[0]
        msg = choice.get("message") or {}
        if not isinstance(msg, dict):
            return ""

        parts: list[str] = []

        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in ("text", "output_text"):
                    t = block.get("text") or block.get("content")
                    if isinstance(t, str) and t.strip():
                        parts.append(t.strip())

        for key in ("reasoning_content", "reasoning", "thought"):
            t = msg.get(key)
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())

        out = "\n".join(parts).strip()
        if not out:
            logger.warning(
                "API 返回空文本: finish_reason=%r message_keys=%r",
                choice.get("finish_reason"),
                list(msg.keys()),
            )
        return out
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        logger.warning("解析 choices[0].message 失败: %s", e)
        return ""


def call_role_model(
    role_name: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout: int = 180,
) -> str:
    """
    按角色调用模型。

    role_name 支持：
    - assistant
    - user_simulator
    - evaluator

    这样 LangGraph 主流程里不需要关心模型 URL 和模型名。
    """
    if role_name not in ROLE_MODEL_CONFIG:
        raise ValueError(
            f"未知角色: {role_name}，可选角色为: {list(ROLE_MODEL_CONFIG.keys())}"
        )

    config = ROLE_MODEL_CONFIG[role_name]

    if config.get("type") != "chat_completions":
        raise ValueError(
            f"角色 {role_name} 不是单轮 chat 通道（type={config.get('type')}），"
            f"请改用 call_deep_chat() 等专用入口。"
        )

    url = config["url"]
    model_name = config["model"]

    logger.info(f"调用角色模型: role={role_name}, model={model_name}, url={url}")

    response = send_chat_completion_request(
        url=url,
        model_name=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    return extract_model_content(response)


# =============================
# Deep Research 会话通道（assistant_deep）
# =============================
def call_deep_chat(
    session_id: str,
    query: str,
    *,
    reset: bool = False,
    mode: str | None = None,
    config_path: str | None = None,
) -> Dict[str, Any]:
    """
    调用 deep_research 的 session 入口（assistant_deep）。

    与 call_role_model 的区别：
    - 不传 messages，由 deep_research 内部根据 session_id 维护历史；
    - 返回结构化 dict（answer 已剥离引用块、references 单独返回），方便：
      1. evaluator 只读 answer，对 references 完全无感；
      2. 训练数据落盘时可保留 references 作为 grounding 元信息。

    Returns:
        {
          "session_id": str,
          "turn_index": int,
          "answer": str,
          "references": list,
          "mode": str,
          "history_size": int,
          "error": str | None,        # 失败时存在
        }
    """
    role = "assistant_deep"
    if role not in ROLE_MODEL_CONFIG:
        raise ValueError(f"未配置 {role} 角色，请检查 ROLE_MODEL_CONFIG。")

    role_cfg = ROLE_MODEL_CONFIG[role]
    if role_cfg.get("type") != "deep_research_session":
        raise ValueError(f"{role} 角色 type 异常: {role_cfg.get('type')}")

    try:
        from deep_research.session_chat import chat as _deep_chat
    except ImportError as e:
        logger.error(
            "导入 deep_research.session_chat 失败，请确认 PYTHONPATH 包含 "
            "deep_research_dev 目录：%s",
            e,
        )
        return {
            "session_id": session_id,
            "turn_index": -1,
            "answer": "",
            "references": [],
            "mode": mode or role_cfg.get("default_mode", "deep_thinking"),
            "history_size": 0,
            "error": f"ImportError: {e}",
        }

    actual_mode = mode or role_cfg.get("default_mode", "deep_thinking")
    logger.info(
        "调用 deep_research 会话: session=%s reset=%s mode=%s query=%r",
        session_id,
        reset,
        actual_mode,
        (query or "")[:80],
    )

    return _deep_chat(
        session_id,
        query,
        reset=reset,
        mode=actual_mode,
        config_path=config_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("当前角色模型配置：")
    for role, config in ROLE_MODEL_CONFIG.items():
        if config.get("type") == "chat_completions":
            print(f"{role}: model={config['model']}, url={config['url']}")
        else:
            print(f"{role}: type={config.get('type')}, module={config.get('module')}")

    test_messages = [
        {"role": "user", "content": "请用一句话解释什么是股东未实缴出资。"}
    ]

    result = call_role_model("assistant", test_messages)
    print("\n测试输出：")
    print(result)
