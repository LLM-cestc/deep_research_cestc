# -*- coding: utf-8 -*-
"""
# 服务主入口

Author: wjianxz
Date: 2025-11-13
"""

import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
import gradio as gr
from deep_research.agent_deep_search import deep_search_rag
from deep_research.parser_config import load_validated_config
from functools import partial
import logging
from pathlib import Path
import threading
import queue
import time

from deep_research.local_logger import setup_global_logger_root
from deep_research.protocal import HistoryMessage
from deep_research.parser_config import AppConfig
from deep_research.utils import is_invalid_output

# 获取当前脚本所在目录
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("python_multipart").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

CURRENT_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
setup_global_logger_root(
    log_file=str(CURRENT_DIR / "deepresearch.log"), level=logging.DEBUG
)


def update_mode(enable_deepresearch, current_state, config: AppConfig):
    new_state = {**current_state, "turn_on_deepresearch": enable_deepresearch}
    config.maxhop.turn_on_deepresearch = enable_deepresearch
    logger.info("turn_on_deepresearch: %s", new_state)
    return new_state


def new_chart_mode(enable_deepresearch, config: AppConfig):
    logger.info("turn_on_deepresearch: %s", enable_deepresearch)
    return enable_deepresearch


def stream_typewriter(user_input, history_message, mode, config: AppConfig):

    # history_message: list[HistoryMessage] = [] # 测试历史session 这样写不行，因为每次调用都会重新初始化
    # 将用户输入加入历史
    logger.info(
        "----------mode: %r, select_pattern: %r", mode, config.pattern.select_pattern
    )
    config.pattern.select_pattern = mode
    # 1. 启动后台线程执行大模型任务
    result_typewriter_q: queue.Queue = queue.Queue()
    # 固定额外参数
    deep_search_rag_bound_fn = partial(
        deep_search_rag,
        mode=mode,
        result_queue=result_typewriter_q,
        history_message=history_message,
        config=config,
    )

    worker = threading.Thread(
        target=deep_search_rag_bound_fn, args=(user_input,), daemon=True
    )
    worker.start()
    final_text = ""
    thinking_parts: list[str] = []
    start_ts = time.time()
    last_heartbeat_ts = 0.0

    while True:
        try:
            chunk = result_typewriter_q.get(timeout=0.1)
        except queue.Empty:
            if not worker.is_alive():
                break
            # 只要没完成全部任务（worker 还活着），就持续显示计时心跳
            now = time.time()
            if now - last_heartbeat_ts >= 1.0:
                elapsed = int(now - start_ts)
                thinking_display = "".join(thinking_parts)
                yield (
                    '<div class="thinking-live">'
                    '<div class="thinking-header">'
                    '💭 思考中…'
                    f'<span class="thinking-timer">已用时 {elapsed}s</span>'
                    "</div>"
                    f"{thinking_display}"
                    "</div>"
                )
                last_heartbeat_ts = now
            continue

        if chunk is None:
            continue

        chunk_text = chunk[1] or ""
        if not chunk[0]:
            thinking_parts.append(chunk_text)
            thinking_display = "".join(thinking_parts)
            elapsed = int(time.time() - start_ts)
            yield (
                '<div class="thinking-live">'
                '<div class="thinking-header">'
                '💭 思考中…'
                f'<span class="thinking-timer">已用时 {elapsed}s</span>'
                "</div>"
                f'{thinking_display}'
                '</div>'
            )
        else:
            final_text = chunk_text
            break

    worker.join(timeout=5)

    if is_invalid_output(final_text):
        final_text = "输出异常，已拦截，请重试或换个问题。"

    thinking_display = "".join(thinking_parts)
    if thinking_display.strip():
        output = (
            '<details class="thinking-block">\n'
            '<summary>💭 思考过程（点击展开）</summary>\n'
            f'<div class="thinking-content">{thinking_display}</div>\n'
            '</details>\n\n'
            f'<div class="answer-block">\n\n{final_text}\n\n</div>'
        )
    else:
        output = f'<div class="answer-block">\n\n{final_text}\n\n</div>'
    yield output
    # return total, history_message


def reset_chat_state():
    return "", "", [], ""


def _normalize_mode(mode_value):
    mode_map = {"文章生成": "deep_research", "AI评标": "bid_generation"}
    if isinstance(mode_value, str) and mode_value in mode_map:
        return mode_map[mode_value]
    if isinstance(mode_value, bool):
        return "deep_research" if mode_value else "chat"
    return mode_value or "deep_research"


def stream_with_last_query(user_input, history_message, mode, last_query, config: AppConfig):
    if not user_input or not user_input.strip():
        return
    last_query = user_input.strip()
    for final_text in stream_typewriter(
        last_query, history_message, _normalize_mode(mode), config
    ):
        yield final_text, last_query


def refresh_with_last_query(last_query, history_message, mode, config: AppConfig):
    if not last_query or not last_query.strip():
        yield "请先输入问题后再刷新。", last_query
        return
    for final_text in stream_typewriter(
        last_query, history_message, _normalize_mode(mode), config
    ):
        yield final_text, last_query


def main(config):
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
    .header-top {
        display: grid !important;
        grid-template-columns: 1fr auto 1fr !important;
        align-items: center !important;
        gap: 16px !important;
        margin-bottom: 10px !important;
    }
    .header-left {
        display: flex !important;
        justify-content: flex-start !important;
        align-items: center !important;
        min-width: 0 !important;
    }
    .header-center {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }
    .header-right {
        display: flex !important;
        justify-content: flex-end !important;
        align-items: center !important;
        min-width: 0 !important;
    }
    .header-title {
        text-align: center;
    }
    .header-title .gr-markdown,
    .header-title .prose {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    .header-area h2 {
        font-size: 24px;
        font-weight: 600;
        color: #333;
        margin: 0 !important;
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
    /* 模式选择器 */
    .mode-radio {
        display: flex !important;
        justify-content: flex-start !important;
        margin-top: 0 !important;
    }
    .mode-radio .wrap {
        display: flex !important;
        flex-direction: column !important;
        align-items: flex-start !important;
        gap: 8px !important;
    }
    .mode-radio label {
        font-size: 14px !important;
        padding: 5px 16px !important;
        border-radius: 20px !important;
        border: 1px solid #d0d5dd !important;
        background: #fff !important;
        color: #333 !important;
        cursor: pointer !important;
        transition: all 0.2s !important;
        width: 100% !important;
        justify-content: center !important;
    }
    .mode-radio label:has(input:checked) {
        background: #6c5ce7 !important;
        border-color: #6c5ce7 !important;
        color: #fff !important;
    }
    .mode-inline {
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        flex-wrap: nowrap !important;
        min-width: 0 !important;
    }

    /* 输入区整体卡片样式 */
    .input-card-row {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 4px !important;
        border: 1px solid #e5e7eb;
        position: relative;
    }
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

    /* 按钮列 */
    .input-actions {
        display: flex !important;
        flex-direction: column !important;
        gap: 8px !important;
    }
    .input-actions button {
        font-size: 15px !important;
        padding: 8px 10px !important;
        min-width: 120px !important;
        width: 120px !important;
        border-radius: 10px !important;
        white-space: nowrap !important;
    }
    .send-btn {
        background: #6c5ce7 !important;
        border: 1px solid #6c5ce7 !important;
        color: #fff !important;
    }
    .send-btn:hover {
        background: #5a4bd1 !important;
        border-color: #5a4bd1 !important;
    }
    .clear-btn {
        background: #f3f4f6 !important;
        border: 1px solid #e5e7eb !important;
        color: #555 !important;
    }
    .clear-btn:hover {
        background: #e5e7eb !important;
    }
    .refresh-btn {
        margin-top: 4px !important;
        background: #f3f4f6 !important;
        border: 1px solid #e5e7eb !important;
    }
    .refresh-btn:hover {
        background: #e5e7eb !important;
    }

    /* 思考过程 - 实时显示 */
    .thinking-live {
        background: #f0f0f5;
        border-left: 3px solid #a29bfe;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 0.88em;
        color: #636e7b;
        line-height: 1.7;
    }
    .thinking-header {
        font-weight: 600;
        color: #6c5ce7;
        font-style: normal;
        margin-bottom: 6px;
        font-size: 0.95em;
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 12px;
    }
    .thinking-timer {
        font-weight: 500;
        color: #7a7aa0;
        font-size: 0.92em;
        white-space: nowrap;
    }

    /* 思考过程 - 折叠区 */
    .thinking-block {
        background: #f0f0f5;
        border-left: 3px solid #a29bfe;
        border-radius: 6px;
        padding: 8px 14px;
        margin-bottom: 18px;
    }
    .thinking-block summary {
        cursor: pointer;
        font-weight: 600;
        color: #6c5ce7;
        padding: 4px 0;
        user-select: none;
        font-size: 0.92em;
    }
    .thinking-block[open] summary {
        margin-bottom: 8px;
        border-bottom: 1px dashed #c8c4f0;
        padding-bottom: 8px;
    }
    .thinking-content {
        font-size: 0.88em;
        color: #636e7b;
        line-height: 1.7;
    }

    /* 最终回答区 */
    .answer-block {
        font-size: 1em;
        color: #1a1a2e;
        line-height: 1.85;
    }

    /* 回答区卡片样式 */
    .output-box {
        margin-top: 12px;
        background: #fff;
        border-radius: 10px;
        border-left: 4px solid #6c5ce7;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        max-height: none;
        overflow-y: visible;
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

    /* 标题右侧版本信息 */
    .header-version-inline {
        padding: 0 !important;
        margin: 0 !important;
        font-size: 12px !important;
        color: #aaa !important;
        line-height: 1.4 !important;
        white-space: nowrap !important;
        text-align: right !important;
    }
    """
    history_message: list[HistoryMessage] = []

    with gr.Blocks(title="写作助手") as demo:
        with gr.Column(elem_classes="container"):

            # === 标题 ===
            with gr.Column(elem_classes="header-area"):
                with gr.Row(elem_classes="header-top"):
                    with gr.Row(elem_classes="header-left"):
                        with gr.Row(elem_classes="mode-inline"):
                            mode_selector = gr.Radio(
                                choices=["文章生成",  "AI评标"],
                                value="文章生成",
                                show_label=False,
                                interactive=True,
                                elem_classes="mode-radio",
                            )
                    with gr.Row(elem_classes="header-center"):
                        gr.Markdown("## 写作助手", elem_classes="header-title")
                    with gr.Row(elem_classes="header-right"):
                        gr.HTML(
                            "v1.0.0 &nbsp;|&nbsp; Qwen3-14B",
                            elem_classes="header-version-inline",
                        )
            # === 状态 ===
            last_query_state = gr.State("")
            chat_history = gr.State(history_message)

            # === 输入区 ===
            with gr.Row(elem_classes="input-card-row"):
                name_input = gr.Textbox(
                    placeholder="请输入写作主题，例如：人工智能在教育领域的应用趋势分析…",
                    lines=3,
                    max_lines=6,
                    show_label=False,
                    scale=5,
                    elem_classes="no-border-input",
                )
                with gr.Column(scale=1, min_width=60, elem_classes="input-actions"):
                    greet_btn = gr.Button(
                        "开始写作",
                        variant="primary",
                        elem_id="send-btn",
                        elem_classes="send-btn",
                    )
                    clear_btn = gr.Button("清空", elem_classes="clear-btn")
                    refresh_btn = gr.Button("🔄", size="sm", elem_classes="refresh-btn")

            # === 正式回答区（卡片样式，自动滚动） ===
            with gr.Column(elem_classes="output-box", elem_id="output-box"):
                output_text = gr.Markdown(
                    elem_classes="output-content no-label",
                    elem_id="final-output-text",
                    show_label=False,
                )

            # === 事件 ===
            send_evt = greet_btn.click(
                fn=partial(stream_with_last_query, config=config),
                inputs=[name_input, chat_history, mode_selector, last_query_state],
                outputs=[output_text, last_query_state],
                show_progress="full",
            )
            submit_evt = name_input.submit(
                fn=partial(stream_with_last_query, config=config),
                inputs=[name_input, chat_history, mode_selector, last_query_state],
                outputs=[output_text, last_query_state],
                show_progress="full",
            )
            refresh_evt = refresh_btn.click(
                fn=partial(refresh_with_last_query, config=config),
                inputs=[last_query_state, chat_history, mode_selector],
                outputs=[output_text, last_query_state],
                show_progress="minimal",
            )
            clear_btn.click(
                fn=reset_chat_state,
                inputs=None,
                outputs=[name_input, output_text, chat_history, last_query_state],
                show_progress="hidden",
                cancels=[send_evt, submit_evt, refresh_evt],
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
    config = load_validated_config("deep_research/config.yaml")
    # logger = setup_logger(__name__, log_file="deepresearch.log", level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    main(config)
