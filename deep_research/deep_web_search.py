# -*- coding: utf-8 -*-
"""
 外部深度搜索（补充性检索）加载模块

Author: wjianxz
Date: 2025-11-13
"""
import random
import time
import logging
import jsonlines
from playwright.sync_api import sync_playwright

from deep_research.protocal import ReferenceList
from deep_research.local_logger import timing
from deep_research.search_client import SearchClient

logger = logging.getLogger(__name__)  # 自动继承 root logger 的 handlers

# 模块级单例：Bing/Baidu HTML；Google 需 SerpAPI key（本模块未注入 key 时 google 无结果）
_SEARCH_CLIENT = SearchClient(
    engine_order=("bing", "baidu"),
    timeout=15.0,
    sleep_between=(0.3, 0.8),
)


@timing
def perform_web_search(query: str, max_results: int = 20, epochs: int = 0, depth: int = 0) -> ReferenceList:
    """
    执行外部深度搜索（Bing/Baidu HTML；Google 仅 SerpAPI，本模块默认未配置 key）。
    获取最新或开放域信息，作为 RAG 的外部上下文补充。

    Args:
        query (str): 搜索关键词
        max_results (int): 最大返回结果数
        epochs (int): 当前搜索轮次
        depth (int): 搜索深度
    Returns:
        ReferenceList: 每条 {"title", "url", "content"}
    """
    try:
        results = _SEARCH_CLIENT.search(query, max_results=max_results)
    except Exception as e:
        logger.error(f"搜索引擎调用失败: {e}")
        results = []

    if not results:
        logger.error("未提取到任何搜索结果（Bing/Baidu 均返回空，请检查网络或风控）")
    # else:
    #     for i, res in enumerate(results, 1):
    #         logger.info(f"{i}. 标题: {res['title']}")
    #         logger.info(f"   URL: {res['url']}\n")

    # url_list = [res["url"] for res in results[:max_results]]

    result_list = []
    for idx, ele in enumerate(results):
        url = ele["url"]
        result = extract_webpage_content(url)
        time.sleep(1)  # 避免请求过快  # noqa: F821
        if len(result['title']) > 3 and len(result['content']) > 10:
            logger.info("\n" + "=" * 50)
            logger.info(f"标题: {result['title']}")
            logger.info("=" * 50)
            result_list.append(result)
        if idx + 1 >= max_results:
            break
    logger.info(f"\n共抓取 {len(result_list)} 篇有效文章。")
    with jsonlines.open(f"deep_research/{epochs}_{depth}_temp_data.jsonl", mode="w") as writer:
        writer.write_all(result_list)
    
    logger.info(f"数据临时存储在 deep_research/{epochs}_{depth}_temp_data.jsonl")
    
    # 下面添加历史ref 相关信息// 充分利用更多的 ref信息(不仅利用当前的)
    with jsonlines.open(f"deep_research/{epochs}_{depth}_temp_data.jsonl") as reader:
        data = list(reader)
    logger.info(f"从文件中读取了 {len(data)} 条数据。")
    return data

@timing
def extract_webpage_content(page_url: str):
    """
    使用 Playwright 访问网页，提取页面标题和主要可见文本内容。

    Args:
        page_url (str): 要抓取的网页完整 URL

    Returns:
        dict: {"title": "页面标题", "content": "清洗后的正文文本"}
    """
    # args=["--headless=new", "--disable-blink-features=AutomationControlled"]
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False, args=[
                                  "--headless=new", 
                                  "--disable-blink-features=AutomationControlled"]
        )
        page = browser.new_page()
        # 隐藏自动化特征
        page.add_init_script("delete navigator.__proto__.webdriver;")
        # 设置真实 User-Agent
        page.set_extra_http_headers(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.142 Safari/537.36"
                
            }
        )

        try:
            logger.info(f"正在访问: {page_url}")
            # 在 goto 之后模拟滚动、鼠标移动
            page.goto(page_url, wait_until="domcontentloaded", timeout=5000)
            page.mouse.move(100, 200)
            page.mouse.move(300, 400)
            page.evaluate("window.scrollBy(0, 200)")
            time.sleep(random.uniform(0.5, 2))

            # 等待 body 加载（确保基本结构存在）
            page.wait_for_selector("body", timeout=2000)

            # 获取页面标题
            title = page.title() or ""

            # 执行 JavaScript 提取 <body> 中所有可见文本（排除 script/style）
            content = page.evaluate(
                """() => {
                // 移除不需要的标签（广告、脚本、样式等）
                const removeSelectors = [
                    'script', 'style', 'noscript', 'iframe', 'nav', 
                    'footer', 'header', 'aside', '.ad', '#ad', 
                    '[aria-hidden="true"]'
                ];
                removeSelectors.forEach(sel => {
                    document.querySelectorAll(sel).forEach(el => el.remove());
                });

                // 获取 body 所有文本并清理
                let text = document.body.innerText || '';
                // 合并多个空白符为单个空格，去除首尾空格
                text = text.replace(/\\s+/g, ' ').trim();
                return text;
            }"""
            )

            browser.close()
            return {"title": title.strip(), "url": page_url, "content": content}

        except Exception as e:
            logger.error(f"抓取失败: {e}")
            browser.close()
            return {"title": "", "url": page_url, "content": ""}

@timing
def scrape_google_search_results(search_url: str):

    # args=[ "--headless=new",

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            slow_mo=200,
            args=[
                
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        page = browser.new_page()

        # 隐藏 webdriver 特征
        page.add_init_script(
            """
            delete navigator.__proto__.webdriver;
            window.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
        """
        )

        # 设置真实 User-Agent
        page.set_extra_http_headers(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.142 Safari/537.36"
            }
        )

        # 在 goto 之后模拟滚动、鼠标移动
        page.goto(search_url, wait_until="domcontentloaded")
        page.mouse.move(100, 200)
        page.mouse.move(300, 400)
        page.evaluate("window.scrollBy(0, 200)")
        time.sleep(random.uniform(0, 2))

        logger.info("正在访问 Google...")
        page.goto(search_url, wait_until="domcontentloaded", timeout=5000)

        # 尝试等待任意一个 h3 出现（比 div.g 更可靠）
        try:
            page.wait_for_selector("h3", timeout=5000)
        except Exception as e:
            logger.warning(" 未找到任何 h3 标题，可能被拦截或页面结构变化")
            # 可选：保存页面快照用于调试
            page.screenshot(path="debug_google.png")
            with open("debug_google.html", "w") as f:
                f.write(page.content())
            browser.close()
            return []

        results = []
        # 查找所有包含 h3 的 a 标签（即搜索结果链接）
        links = page.query_selector_all("a:has(h3)")
        for link in links:
            title_elem = link.query_selector("h3")
            if title_elem:
                title = title_elem.inner_text()
                url = link.get_attribute("href")
                if title and url:
                    results.append({"title": title, "url": url})

        browser.close()
        return results


def scrape_google_search_results_test(search_url: str):
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            slow_mo=200,
            args=[
                "--headless=new", 
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
            ],
        )
        # 代理需要match domain，否则会报错
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
            locale="en-US",
            timezone_id="America/New_York"
        )

        # 注入 CONSENT cookie（关键！）
        context.add_cookies([{
            "name": "CONSENT",
            "value": "YES+cb.20240101-12-p0.en+FX+123456789",
            "domain": ".google.com",
            "path": "/"
        }])

        page = context.new_page()

        page.add_init_script("""
            delete navigator.__proto__.webdriver;
            window.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });
            Object.defineProperty(navigator, 'deviceMemory', { get: () => 8 });
            window.outerHeight = window.innerHeight + 100;
            window.outerWidth = window.innerWidth + 100;
        """)
        # 在 goto 之后模拟滚动、鼠标移动
        page.goto(search_url, wait_until="domcontentloaded")
        page.mouse.move(100, 200)
        page.mouse.move(300, 400)
        page.evaluate("window.scrollBy(0, 200)")
        time.sleep(random.uniform(1, 3))

        logger.info("正在访问 Google...")
        page.goto(search_url, wait_until="domcontentloaded", timeout=5000)

        # 等待可能的跳转或验证码
        time.sleep(random.uniform(2, 4))

        # 检查是否出现验证码或错误页
        if "unusual traffic" in page.content() or "recaptcha" in page.content().lower():
            logger.error("被 Google 识别为机器人！")
            page.screenshot(path="blocked.png")
            browser.close()
            return []

        try:
            page.wait_for_selector("h3", timeout=5000)
        except:
            logger.warning("未找到 h3，保存调试文件...")
            page.screenshot(path="debug.png")
            with open("debug.html", "w") as f:
                f.write(page.content())
            browser.close()
            return []

        results = []

        links = page.query_selector_all("a:has(h3)")
        for link in links:
            title_elem = link.query_selector("h3")
            if title_elem:
                title = title_elem.inner_text()
                url = link.get_attribute("href")
                if title and url:
                    if len(title) > 3 and len(url) > 3:
                        results.append({"title": title, "url": url})

        browser.close()
        return results


#---------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    query = "违章停车与违法停车是否有区别，解释"
    result = perform_web_search(query, max_results=2)
    print("最终抓取结果：", result)