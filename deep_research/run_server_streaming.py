# -*- coding: utf-8 -*-
"""
# 服务主入口

Author: wjianxz
Date: 2025-11-13
"""

import os
import re
import gradio as gr
from deep_research.agent_deep_search import deep_search_rag
from deep_research.parser_config import load_validated_config
from functools import partial
import logging
from pathlib import Path
import threading
import queue
import time

from deep_research.protocal import HistoryMessage
from deep_research.parser_config import AppConfig
from deep_research.retrieve_knowledge import prebuild_indices
from deep_research.run_trace_log import init_run_log
from deep_research.utils import is_invalid_output

# 获取当前脚本所在目录
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("python_multipart").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

CURRENT_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
logger = logging.getLogger(__name__)


def update_mode(enable_deepresearch, current_state, config: AppConfig):
    new_state = {**current_state, "turn_on_deepresearch": enable_deepresearch}
    config.maxhop.turn_on_deepresearch = enable_deepresearch
    logger.info("turn_on_deepresearch: %s", new_state)
    return new_state


def new_chart_mode(enable_deepresearch, config: AppConfig):
    logger.info("turn_on_deepresearch: %s", enable_deepresearch)
    return enable_deepresearch


_THINK_TAG_RE = re.compile(r"</?think[^>]*>", re.IGNORECASE)


def _strip_think_tags(text: str) -> str:
    return _THINK_TAG_RE.sub("", text or "")


def _wrap_thinking(thinking_text: str, *, finished: bool) -> str:
    if not thinking_text.strip():
        return ""
    open_attr = "" if finished else " open"
    title = "思考过程（已完成，点击展开）" if finished else "思考过程（生成中…）"
    return (
        f"<details{open_attr}>\n"
        f"<summary>{title}</summary>\n\n"
        f"{thinking_text.strip()}\n\n"
        f"</details>\n\n"
    )


def stream_typewriter(
    user_input,
    history_message,
    mode,
    config: AppConfig,
    web_enabled_ui: bool | None = None,
):
    """单一 Markdown：上方折叠思考，下方正文，共用外层 ``.output-box`` 滚动条。"""
    web_cfg = getattr(config, "web", None)
    prev_web = None
    if web_enabled_ui is not None and web_cfg is not None:
        prev_web = web_cfg.enabled
        web_cfg.enabled = bool(web_enabled_ui)

    cancel_event = None
    try:
        # history_message: list[HistoryMessage] = [] # 测试历史session 这样写不行，因为每次调用都会重新初始化
        # 将用户输入加入历史
        logger.info(
            "----------mode: %r, select_pattern: %r", mode, config.pattern.select_pattern
        )
        config.pattern.select_pattern = mode
        # 1. 启动后台线程执行大模型任务
        result_typewriter_q: queue.Queue = queue.Queue()
        cancel_event = threading.Event()
        # 固定额外参数
        deep_search_rag_bound_fn = partial(
            deep_search_rag,
            mode=mode,
            result_queue=result_typewriter_q,
            history_message=history_message,
            config=config,
            cancel_event=cancel_event,
        )

        def worker_fn():
            try:
                deep_search_rag_bound_fn(user_input)
            except Exception as e:
                logger.exception("后台问答线程异常: %s", e)
                if not cancel_event.is_set():
                    result_typewriter_q.put([True, f"生成过程异常：{e}"])

        worker = threading.Thread(
            target=worker_fn, daemon=True
        )
        worker.start()
        thinking_text = ""
        answer_text = ""
        final_text = ""
        last_yield_ts = 0.0
        yield _wrap_thinking("正在准备检索与回答…", finished=False) + ""

        is_final = False
        while True:
            try:
                chunk = result_typewriter_q.get(timeout=0.1)
            except queue.Empty:
                if not worker.is_alive():
                    break
                now = time.time()
                if now - last_yield_ts > 10:
                    yield (
                        _wrap_thinking(
                            thinking_text or "正在等待模型返回流式内容…",
                            finished=False,
                        )
                        + answer_text
                    )
                    last_yield_ts = now
                continue

            if chunk is None:
                continue

            is_final = chunk[0]
            chunk_text = _strip_think_tags(chunk[1] or "")
            stream_kind = chunk[2] if len(chunk) > 2 else "thinking"

            if is_final:
                final_text = chunk_text
                break

            if stream_kind == "answer":
                answer_text += chunk_text
            else:
                thinking_text += chunk_text

            now = time.time()
            if stream_kind == "answer":
                if now - last_yield_ts > 0.12 or len(chunk_text) >= 64:
                    yield _wrap_thinking(thinking_text, finished=False) + answer_text
                    last_yield_ts = now
            elif now - last_yield_ts > 0.25 or len(chunk_text) >= 80:
                yield _wrap_thinking(thinking_text, finished=False) + answer_text
                last_yield_ts = now

        worker.join(timeout=5)

        if is_invalid_output(final_text):
            final_text = "输出异常，已拦截，请重试或换个问题。"
        yield _wrap_thinking(thinking_text, finished=True) + final_text
    finally:
        if cancel_event is not None:
            cancel_event.set()
        if prev_web is not None and web_cfg is not None:
            web_cfg.enabled = prev_web
    # return total, history_message


def reset_chat_state():
    return "", "", [], ""


def _normalize_mode(mode_value):
    if isinstance(mode_value, bool):
        return "deep_thinking" if mode_value else "chat"
    return mode_value or "chat"


def stream_with_last_query(
    user_input, history_message, mode, web_on, last_query, config: AppConfig
):
    if not user_input or not user_input.strip():
        return
    last_query = user_input.strip()
    for combined in stream_typewriter(
        last_query,
        history_message,
        _normalize_mode(mode),
        config,
        web_enabled_ui=web_on,
    ):
        yield combined, last_query


def refresh_with_last_query(
    last_query, history_message, mode, web_on, config: AppConfig
):
    if not last_query or not last_query.strip():
        yield "请先输入问题后再刷新。", last_query
        return
    for combined in stream_typewriter(
        last_query,
        history_message,
        _normalize_mode(mode),
        config,
        web_enabled_ui=web_on,
    ):
        yield combined, last_query


def main(config):
    init_run_log(CURRENT_DIR / "run.log")
    if os.environ.get("RAG_DISABLE_PREBUILD", "").strip().lower() in ("1", "true", "yes"):
        logger.info("已跳过知识库预热（环境变量 RAG_DISABLE_PREBUILD）")
    else:
        threading.Thread(
            target=prebuild_indices,
            args=(config,),
            daemon=True,
            name="kb-prebuild",
        ).start()
    logger.info("主函数入口 ！！！")

    css = """
    body { background: #f5f5f5; }
    
    .container {
        max-width: 960px;
        margin: 0 auto;
        padding: 24px 20px;
    }

    .header-area {
        position: relative;
        text-align: center;
        margin-bottom: 20px;
        border: none !important;
        box-shadow: none !important;
    }
    .header-title {
        text-align: center;
    }
    .header-toggles {
        /* 左对齐、整行占位；两开关同一基线、高度一致 */
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: flex-start !important;
        justify-content: flex-start !important;
        gap: 14px !important;
        width: 100% !important;
        max-width: 100% !important;
        margin: 10px 0 4px 0 !important;
        padding-left: 0 !important;
        min-height: 0 !important;
    }
    .header-toggles > div {
        flex: 0 0 auto !important;
        min-width: unset !important;
        width: auto !important;
        max-width: none !important;
        display: flex !important;
        align-items: center !important;
        align-self: flex-start !important;
        min-height: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    .header-toggles .gr-form,
    .header-toggles .form {
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
        min-height: unset !important;
        height: auto !important;
        overflow: visible !important;
        overflow-x: visible !important;
        box-shadow: none !important;
    }
    .header-toggles .block {
        overflow: visible !important;
        border: none !important;
        min-height: 0 !important;
    }
    .header-toggles .kb-toggle {
        margin: 0 !important;
        white-space: nowrap !important;
        align-self: flex-start !important;
    }
    /* 避免 Gradio 表单区出现横向滚动条灰色条带 */
    .header-toggles {
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
    }
    .header-toggles::-webkit-scrollbar {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
    }
    .header-mode {
        position: absolute;
        right: 10;
        top: 100%;
        transform: translateY(-50%);
        max-width: 220px;
    }
    .header-area h2 {
        font-size: 24px;
        font-weight: 600;
        color: #333;
        margin: 0 0 4px 0;
        border-bottom: none !important;
        box-shadow: none !important;
    }
    .header-area .gr-markdown,
    .header-area .prose {
        border: none !important;
        box-shadow: none !important;
        padding-bottom: 0 !important;
        margin-bottom: 0 !important;
    }
    .header-area hr {
        display: none !important;
        border: none !important;
        height: 0 !important;
    }
    .header-area * {
        border: none !important;
        box-shadow: none !important;
    }
    /* 知识库滑动开关 */
    .kb-toggle {
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;
        font-size: 14px !important;
        color: #555 !important;
        cursor: pointer !important;
    }
    .kb-toggle .wrap {
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }
    .kb-toggle input[type="checkbox"] {
        appearance: none !important;
        -webkit-appearance: none !important;
        width: 40px !important;
        height: 22px !important;
        background: #ccc !important;
        border-radius: 11px !important;
        position: relative !important;
        cursor: pointer !important;
        transition: background 0.2s !important;
        border: none !important;
        outline: none !important;
    }
    .kb-toggle input[type="checkbox"]::before {
        content: "" !important;
        position: absolute !important;
        top: 2px !important;
        left: 2px !important;
        width: 18px !important;
        height: 18px !important;
        background: #fff !important;
        border-radius: 50% !important;
        transition: left 0.2s !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.15) !important;
    }
    .kb-toggle input[type="checkbox"]:checked {
        background: #4f8cff !important;
    }
    .kb-toggle input[type="checkbox"]:checked::before {
        left: 20px !important;
    }
    .kb-toggle label span {
        font-weight: 500 !important;
        color: #333 !important;
    }
    /* 输入区整体卡片样式 - 压缩边距 */
    .input-card-row { 
        background: #f8f9fa; 
        border-radius: 12px; 
        padding: 4px !important;
        border: 1px solid #e5e7eb;
        position: relative;
    }
    /* 去掉 Textbox 原生边框，让输入区看起来像一个整体 */
    .input-card-row .gr-textbox,
    .input-card-row .gr-box,
    .input-card-row .gr-form,
    .input-card-row .gr-input,
    .no-border-input textarea {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }
    .no-border-input textarea {
        padding: 8px 12px !important;
        font-size: 16px !important;
        min-height: 52px !important;
        max-height: 120px !important;
    }
    .no-border-input textarea:focus {
        outline: none !important;
    }
    
    /* 按钮样式 */
    .input-actions button {
        font-size: 15px !important;
        padding: 8px 10px !important;
        min-width: 120px !important;
        width: 120px !important;
    }
    .send-btn {
        background: #4f8cff !important;
        border: 1px solid #4f8cff !important;
        color: #fff !important;
    }
    .send-btn:hover {
        background: #3f7ef2 !important;
        border-color: #3f7ef2 !important;
    }
    /* 刷新按钮不再绝对定位 */
    .refresh-btn {
        margin-top: 4px !important;
    }
    
    .input-row {
        display: flex !important;
        gap: 8px !important;
        align-items: stretch !important;
    }
    .icon-btn {
        width: 28px !important;
        height: 28px !important;
        min-width: 28px !important;
        padding: 0 !important;
        border-radius: 6px !important;
        background: #fff !important;
        border: 1px solid #e0e0e0 !important;
        box-shadow: none !important;
        color: #666 !important;
        font-size: 16px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
    }
    .icon-btn:hover {
        background: #f0f0f0 !important;
        color: #333 !important;
    }

    /* 回答区卡片样式（思考+正文合一滚动） */
    .output-box {
        margin-top: 12px;
        background: #fff;
        border-radius: 10px;
        border-left: 4px solid #4f8cff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        max-height: 60vh;
        overflow-y: auto;
    }
    .output-content {
        padding: 16px 20px;
        min-height: 100px;
        font-size: 15px;
        line-height: 1.8;
        color: #333;
    }

    /* 隐藏 label */
    .no-label label { display: none !important; }
    .no-label .gr-form { border: none !important; background: transparent !important; }

    /* 标题右侧版本信息：直接定位 Gradio 包装层 */
    .header-version-wrap {
        position: absolute !important;
        right: 12px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        padding: 0 !important;
        margin: 0 !important;
        min-height: 0 !important;
        text-align: right !important;
        font-size: 12px !important;
        color: #999 !important;
        line-height: 1.8 !important;
        white-space: nowrap !important;
        pointer-events: none !important;
        z-index: 10 !important;
    }
    """
    history_message: list[HistoryMessage] = []
    
    with gr.Blocks(title="法律智能问答") as demo:
        with gr.Column(elem_classes="container"):
            
            # === 标题 + 知识库 / 网页搜索开关（并排） ===
            with gr.Column(elem_classes="header-area"):
                gr.Markdown("## ⚖️ 法智助手", elem_classes="header-title")
                gr.HTML(
                    f"版本号: 3.1<br>模型: {config.deepresearch.name}",
                    elem_classes="header-version-wrap",
                )
                with gr.Row(elem_classes="header-toggles", equal_height=False):
                    mode_selector = gr.Checkbox(
                        value=True,
                        label="知识库",
                        interactive=True,
                        elem_classes="kb-toggle",
                        container=False,
                        min_width=0,
                    )
                    web_search_selector = gr.Checkbox(
                        value=bool(getattr(config, "web", None) and config.web.enabled),
                        label="网页搜索",
                        interactive=True,
                        elem_classes="kb-toggle",
                        container=False,
                        min_width=0,
                    )
            # === 状态 ===
            last_query_state = gr.State("")
            chat_history = gr.State(history_message)

            # === 输入区（紧凑卡片样式） ===
            with gr.Row(elem_classes="input-card-row"):
                name_input = gr.Textbox(
                    placeholder="请输入您的法律问题...（Shift+Enter 发送）",
                    lines=3,
                    max_lines=6,
                    show_label=False,
                    scale=5,
                    elem_classes="no-border-input",
                )
                # min_width 控制按钮列的最小宽度
                with gr.Column(scale=1, min_width=60, elem_classes="input-actions"):
                    greet_btn = gr.Button("发送", variant="primary", elem_id="send-btn")
                    clear_btn = gr.Button("新对话")
                    refresh_btn = gr.Button("🔄", size="sm")

            # === 输出区：折叠思考 + 正文，单一滚动条 ===
            with gr.Column(elem_classes="output-box", elem_id="output-box"):
                output_text = gr.Markdown(
                    elem_classes="output-content no-label",
                    elem_id="final-output-text",
                    show_label=False,
                )

            # === 事件 ===
            send_evt = greet_btn.click(
                fn=partial(stream_with_last_query, config=config),
                inputs=[
                    name_input,
                    chat_history,
                    mode_selector,
                    web_search_selector,
                    last_query_state,
                ],
                outputs=[output_text, last_query_state],
                show_progress="full",
            )
            submit_evt = name_input.submit(
                fn=partial(stream_with_last_query, config=config),
                inputs=[
                    name_input,
                    chat_history,
                    mode_selector,
                    web_search_selector,
                    last_query_state,
                ],
                outputs=[output_text, last_query_state],
                show_progress="full",
            )
            refresh_evt = refresh_btn.click(
                fn=partial(refresh_with_last_query, config=config),
                inputs=[
                    last_query_state,
                    chat_history,
                    mode_selector,
                    web_search_selector,
                ],
                outputs=[output_text, last_query_state],
                show_progress="minimal",
            )
            clear_btn.click(
                fn=reset_chat_state,
                inputs=None,
                outputs=[
                    name_input,
                    output_text,
                    chat_history,
                    last_query_state,
                ],
                show_progress="hidden",
                queue=False,
                # 仅取消「正在流式输出」的发送/提交；不要带上 refresh。
                # 否则会在 Gradio /cancel 里对已完成或从未启动的 event_id 重复取消，
                # 触发 routes.py 里 iterators 的竞态 KeyError（见 Gradio #cancel_event）。
                cancels=[send_evt, submit_evt],
            )
            
            # 自动滚动 JS
            gr.HTML("""
                <script>
                (function() {
                  if (window.__appBound) return;
                  window.__appBound = true;
                  
                  const outputBox = document.getElementById("output-box");
                  const outputTarget = document.getElementById("final-output-text");
                  
                  let lastOutputText = "";
                  
                  // 回答区自动滚动
                  if (outputTarget && outputBox) {
                    const outputObserver = new MutationObserver(function() {
                      outputBox.scrollTop = outputBox.scrollHeight;
                    });
                    outputObserver.observe(outputTarget, { childList: true, subtree: true, characterData: true });
                  }
                  
                  // 轮询检测回答内容，确保自动滚动生效
                  setInterval(function() {
                    if (!outputTarget) return;
                    const currentText = outputTarget.innerText || "";
                    lastOutputText = currentText;
                  }, 300);
                  
                  // Enter 直接发送，Shift+Enter 换行
                  document.addEventListener("keydown", function(e) {
                    if (e.target.tagName === "TEXTAREA" && e.target.closest(".input-card-row")) {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        lastOutputText = "";
                        const btn = document.querySelector("[id^='send-btn']") || document.querySelector(".input-card-row button.primary") || Array.from(document.querySelectorAll("button")).find(b => b.textContent.trim() === "发送");
                        if (btn) btn.click();
                      }
                    }
                  });
                })();
                </script>
            """)

    demo.queue(max_size=10)  # 显式启用队列，确保前端活跃轮询
    print("本地访问推荐: http://127.0.0.1:8090/ （若 localhost:8090 不可用请用此地址）")
    if os.environ.get("RAG_DISABLE_PREBUILD", "").strip().lower() not in ("1", "true", "yes"):
        print(
            "[kb-prebuild] 后台正在预热知识库索引；若开启稠密检索，终端可能出现「Loading weights」"
            "（BGE 加载），页面仍可正常打开。跳过预热：export RAG_DISABLE_PREBUILD=1",
            flush=True,
        )
    demo.launch(
        server_name="0.0.0.0",
        server_port=8090,
        share=False,
        css=css,
    )

# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # 仅对 localhost 不走代理，其他请求仍按环境变量走代理
    _no_proxy = os.environ.get("NO_PROXY", "") or os.environ.get("no_proxy", "")
    _add = "localhost,127.0.0.1"
    if _add not in _no_proxy:
        _no_proxy = (_no_proxy.rstrip(",") + "," + _add).lstrip(",") if _no_proxy else _add
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = _no_proxy
    # unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    # http://127.0.0.1:8086/
    # ./cpolar http 8087
    # https://www.cpolar.com/docs?_gl=1*rh18jt*_ga*MTA5MzI4NDcwLjE3NjIyNTQ4MzA.*_ga_WF16DPKZZ1*czE3NjM5NDg0NjYkbzQkZzEkdDE3NjM5NDkwMjYkajQwJGwwJGgw
    from deep_research.paths import default_config_path

    config = load_validated_config(default_config_path())
    # logger = setup_logger(__name__, log_file="deepresearch.log", level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    main(config)
