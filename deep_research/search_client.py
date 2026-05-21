# -*- coding: utf-8 -*-
"""
统一搜索客户端：
- Bing / Baidu：解析公开 HTML 搜索结果（免 Key）。
- Google：仅支持 SerpAPI JSON（需 ``serpapi_api_key``），不再抓取 Google 网页。

设计目标：
1. 免费路径（Bing/Baidu）除引擎风控外无 API 费用。
2. 多引擎按顺序降级，首个返回非空结果的引擎即生效。
3. 仅返回 {title, url, snippet}，正文抓取由上层完成。

Author: 2026-05-18
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests
from bs4 import BeautifulSoup
from requests.exceptions import ConnectionError, ReadTimeout, Timeout

logger = logging.getLogger(__name__)

_BAIDU_HOME = "https://www.baidu.com/"
# 风控/验证码页特征（无 #content_left 自然结果）
_BAIDU_BLOCK_MARKERS = ("百度安全验证", "安全验证", "网络不给力", "验证码")

# 多 UA 轮换，降低被风控概率
_USER_AGENTS: tuple[str, ...] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    engine: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "engine": self.engine,
        }


@dataclass
class SearchClient:
    """无 Key 多引擎搜索客户端。

    Args:
        engine_order: 引擎降级顺序；默认 ("bing", "google")。Google 仅 SerpAPI。
        timeout: 单次 HTTP 请求超时（秒）。
        sleep_between: 每次外部请求间隔随机区间（秒），降低风控概率。
        proxy: 可选 HTTP/HTTPS 代理（写入 Session）。SerpAPI 在国外时如需可走代理；Bing/Baidu 在配置了 proxy 时会按需直连。
        merge_engines: 为 True 时合并所有引擎结果（去重），否则按降级顺序首个非空即返回。
    """

    engine_order: tuple[str, ...] = ("bing", "google")
    timeout: float = 8.0
    sleep_between: tuple[float, float] = (0.3, 0.9)
    proxy: str | None = None
    merge_engines: bool = False
    # SerpAPI：Google 唯一路径（JSON）；为空则 google 引擎始终返回空列表。
    serpapi_api_key: str | None = None
    serpapi_url: str = "https://serpapi.com/search.json"
    session: requests.Session = field(default_factory=requests.Session)
    _baidu_ready: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.session.headers.update(self._default_headers())
        if self.proxy:
            self.session.proxies.update({"http": self.proxy, "https": self.proxy})

    def reset_for_new_question(self) -> None:
        """新题开始前清空 Cookie/连接，避免上一题风控连累本题。"""
        try:
            self.session.close()
        except Exception:
            pass
        self.session = requests.Session()
        if self.proxy:
            self.session.proxies.update({"http": self.proxy, "https": self.proxy})
        self.session.headers.update(self._default_headers())
        self._baidu_ready = False

    # ---------- public ----------
    def search(
        self,
        query: str,
        max_results: int = 10,
        *,
        engine_order: tuple[str, ...] | None = None,
    ) -> list[dict]:
        query = (query or "").strip()
        if not query:
            return []

        order = engine_order if engine_order is not None else self.engine_order

        engines: dict[str, Callable[[str, int], list[SearchResult]]] = {
            "bing": self._search_bing,
            "baidu": self._search_baidu,
            "google": self._search_google,
        }

        if self.merge_engines:
            merged: list[SearchResult] = []
            seen: set[str] = set()
            for name in order:
                fn = engines.get(name)
                if not fn:
                    continue
                for item in self._safe_call(fn, name, query, max_results):
                    key = item.url.split("#", 1)[0].rstrip("/")
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append(item)
                    if len(merged) >= max_results:
                        return [x.to_dict() for x in merged]
            return [x.to_dict() for x in merged]

        for name in order:
            fn = engines.get(name)
            if not fn:
                logger.warning("[Search] 未知引擎: %s", name)
                continue
            items = self._safe_call(fn, name, query, max_results)
            if items:
                logger.info("[Search] engine=%s 返回 %d 条结果 (query=%s)", name, len(items), query)
                return [x.to_dict() for x in items[:max_results]]
            logger.warning("[Search] engine=%s 返回空，尝试下一个引擎", name)
        logger.error("[Search] 全部引擎均无结果 (query=%s order=%s)", query, order)
        return []

    def search_engine(self, engine: str, query: str, max_results: int = 10) -> list[dict]:
        """指定引擎单独搜索；连通性测试用。"""
        engines = {
            "bing": self._search_bing,
            "baidu": self._search_baidu,
            "google": self._search_google,
        }
        fn = engines.get(engine)
        if not fn:
            raise ValueError(f"未知引擎: {engine}")
        return [x.to_dict() for x in self._safe_call(fn, engine, query, max_results)]

    # ---------- internal ----------
    def _safe_call(
        self,
        fn: Callable[[str, int], list[SearchResult]],
        engine: str,
        query: str,
        max_results: int,
    ) -> list[SearchResult]:
        try:
            self._sleep()
            return fn(query, max_results)
        except Exception as e:
            logger.warning("[Search] engine=%s 抛出异常: %s", engine, e)
            return []

    def _sleep(self) -> None:
        lo, hi = self.sleep_between
        if hi > 0:
            time.sleep(random.uniform(max(0.0, lo), max(lo, hi)))

    def _default_headers(self) -> dict:
        return {
            "User-Agent": random.choice(_USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    def _timeout_tuple(self) -> tuple[float, float]:
        """(connect, read)：避免连接慢与读 body 慢混在一个数字里。"""
        read_s = max(3.0, float(self.timeout))
        connect_s = min(5.0, max(2.0, read_s * 0.25))
        return (connect_s, read_s)

    def _get(
        self,
        url: str,
        bypass_proxy: bool = False,
        *,
        rotate_ua: bool = True,
        max_retries: int = 1,
        **kwargs,
    ) -> requests.Response:
        headers = dict(kwargs.pop("headers", {}) or {})
        if rotate_ua:
            headers.setdefault("User-Agent", random.choice(_USER_AGENTS))
        else:
            headers.setdefault("User-Agent", self.session.headers.get("User-Agent") or random.choice(_USER_AGENTS))
        timeout = kwargs.pop("timeout", self._timeout_tuple())
        last_err: Exception | None = None
        for attempt in range(max(1, max_retries)):
            try:
                if bypass_proxy and self.proxy:
                    return requests.get(url, headers=headers, timeout=timeout, **kwargs)
                return self.session.get(url, headers=headers, timeout=timeout, **kwargs)
            except (ReadTimeout, Timeout, ConnectionError) as e:
                last_err = e
                if attempt + 1 >= max_retries:
                    raise
                wait = 0.4 * (attempt + 1)
                logger.warning(
                    "[Search] GET 超时/连接失败，%ds 后重试 (%d/%d): %s",
                    wait,
                    attempt + 1,
                    max_retries,
                    url[:80],
                )
                time.sleep(wait)
        if last_err:
            raise last_err
        raise RuntimeError("unreachable")

    def _is_baidu_blocked(self, html: str) -> bool:
        if not html or len(html) < 8000:
            if any(m in html for m in _BAIDU_BLOCK_MARKERS):
                return True
        if any(m in html for m in _BAIDU_BLOCK_MARKERS) and "#content_left" not in html:
            return True
        return False

    def _ensure_baidu_ready(self, *, force: bool = False) -> None:
        """访问百度首页拿 Cookie；未预热时极易返回安全验证页。"""
        if self._baidu_ready and not force:
            return
        ua = random.choice(_USER_AGENTS)
        self.session.headers["User-Agent"] = ua
        try:
            resp = self._get(
                _BAIDU_HOME,
                bypass_proxy=bool(self.proxy),
                rotate_ua=False,
                max_retries=2,
                headers={"Referer": _BAIDU_HOME},
            )
            resp.raise_for_status()
            self._baidu_ready = True
            logger.debug("[Search] baidu 预热完成 cookies=%s", list(self.session.cookies.keys()))
        except Exception as e:
            self._baidu_ready = False
            logger.warning("[Search] baidu 预热失败: %s", e)

    # ---------- Bing ----------
    def _search_bing(self, query: str, max_results: int) -> list[SearchResult]:
        # 用 cn.bing.com，中文结果更精准；mkt=zh-CN 防止跳转登录页
        url = f"https://cn.bing.com/search?q={quote_plus(query)}&mkt=zh-CN&FORM=BESBTB"
        resp = self._get(url, bypass_proxy=bool(self.proxy))
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        out: list[SearchResult] = []
        for li in soup.select("li.b_algo"):
            a = li.select_one("h2 a")
            if not a:
                continue
            href = (a.get("href") or "").strip()
            title = a.get_text(" ", strip=True)
            if not href.startswith("http") or not title:
                continue
            snippet_el = (
                li.select_one("div.b_caption p")
                or li.select_one("p.b_lineclamp4")
                or li.select_one("p.b_lineclamp3")
                or li.select_one("p.b_lineclamp2")
                or li.select_one("p")
            )
            snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
            out.append(SearchResult(title=title, url=href, snippet=snippet, engine="bing"))
            if len(out) >= max_results:
                break
        return out

    # ---------- Baidu ----------
    def _search_baidu(self, query: str, max_results: int) -> list[SearchResult]:
        self._ensure_baidu_ready()
        url = f"https://www.baidu.com/s?wd={quote_plus(query)}&rn={max(10, max_results)}"
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                resp = self._get(
                    url,
                    bypass_proxy=bool(self.proxy),
                    rotate_ua=False,
                    max_retries=2,
                    headers={"Referer": _BAIDU_HOME},
                )
                resp.raise_for_status()
            except Exception as e:
                logger.warning("[Search] baidu SERP 请求失败 attempt=%d: %s", attempt + 1, e)
                self._baidu_ready = False
                if attempt + 1 < max_attempts:
                    self._ensure_baidu_ready(force=True)
                    time.sleep(0.5 * (attempt + 1))
                    continue
                return []

            if resp.encoding and resp.encoding.lower() in ("iso-8859-1", "latin-1"):
                resp.encoding = "utf-8"

            if self._is_baidu_blocked(resp.text):
                logger.warning(
                    "[Search] baidu 返回风控/验证页 (attempt=%d, len=%d, query=%s)",
                    attempt + 1,
                    len(resp.text),
                    query,
                )
                self._baidu_ready = False
                if attempt + 1 < max_attempts:
                    self._ensure_baidu_ready(force=True)
                    time.sleep(0.6 * (attempt + 1))
                    continue
                return []

            out = self._parse_baidu_serp(resp.text, max_results)
            if out:
                return out
            logger.warning(
                "[Search] baidu 解析 0 条 (attempt=%d, len=%d, query=%s)",
                attempt + 1,
                len(resp.text),
                query,
            )
            if attempt + 1 < max_attempts:
                self._baidu_ready = False
                self._ensure_baidu_ready(force=True)
                time.sleep(0.4)
        return []

    def _parse_baidu_serp(self, html: str, max_results: int) -> list[SearchResult]:
        soup = BeautifulSoup(html, "lxml")
        out: list[SearchResult] = []
        for div in soup.select("div.result, div.c-container"):
            a = div.select_one("h3 a, h3.t a")
            if not a:
                continue
            jump_url = (a.get("href") or "").strip()
            title = a.get_text(" ", strip=True)
            if not jump_url.startswith("http") or not title:
                continue
            low = jump_url.lower()
            if "baidu.com/baidu.php" in low or "baidu.com/baidu?" in low or "baidu.com/sf?" in low:
                continue
            snippet_el = (
                div.select_one("span.content-right_2s-H4")
                or div.select_one("span.content-right_8Zs40")
                or div.select_one("div.c-abstract")
                or div.select_one("div.c-span-last")
                or div.select_one("span[class*='content-right']")
            )
            snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
            mu = (div.get("mu") or div.get("data-url") or "").strip()
            if mu.startswith("http"):
                real_url = mu
            else:
                # 搜索阶段不 HEAD 跟随跳转链，正文抓取时会 allow_redirects
                real_url = self._resolve_baidu_redirect_lazy(jump_url)
            out.append(SearchResult(title=title, url=real_url, snippet=snippet, engine="baidu"))
            if len(out) >= max_results:
                break
        return out

    def _resolve_baidu_redirect_lazy(self, jump_url: str) -> str:
        """仅解析 query 中的 url 参数；其余保留 link 链供上层 fetch 跟随。"""
        if "baidu.com/link?" not in jump_url:
            return jump_url
        try:
            qs = parse_qs(urlparse(jump_url).query)
            if "url" in qs and qs["url"]:
                cand = unquote(qs["url"][0])
                if cand.startswith("http"):
                    return cand
        except Exception as e:
            logger.debug("[Search] baidu redirect 懒解析失败 %s: %s", jump_url, e)
        return jump_url

    # ---------- Google（仅 SerpAPI JSON） ----------
    def _search_google(self, query: str, max_results: int) -> list[SearchResult]:
        if not self.serpapi_api_key:
            logger.warning(
                "[Search] google 仅支持 SerpAPI：未配置 serpapi_api_key 或未设置环境变量（见 web.serpapi_key_env），已跳过"
            )
            return []
        try:
            out = self._search_google_via_serpapi(query, max_results)
            if not out:
                logger.warning("[Search] google(serpapi) 返回空 organic_results")
            return out
        except Exception as e:
            logger.warning("[Search] google(serpapi) 失败: %s", e)
            return []

    def _search_google_via_serpapi(self, query: str, max_results: int) -> list[SearchResult]:
        """通过 SerpAPI 获取 Google organic_results。"""
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_api_key,
            "hl": "zh-cn",
            "num": str(max(10, max_results)),
        }
        resp = self.session.get(self.serpapi_url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json() or {}
        if data.get("error"):
            logger.warning("[Search] serpapi error: %s", data.get("error"))
            return []
        out: list[SearchResult] = []
        seen: set[str] = set()
        for item in (data.get("organic_results") or []):
            url = (item.get("link") or "").strip()
            title = (item.get("title") or "").strip()
            if not url.startswith("http") or not title:
                continue
            key = url.split("#", 1)[0]
            if key in seen:
                continue
            seen.add(key)
            snippet = ""
            sn = item.get("snippet")
            if isinstance(sn, str):
                snippet = sn.strip()
            if not snippet:
                hl = item.get("snippet_highlighted_words")
                if isinstance(hl, list):
                    snippet = " ".join(str(x) for x in hl).strip()
            out.append(SearchResult(title=title, url=url, snippet=snippet, engine="google"))
            if len(out) >= max_results:
                break
        return out


# ---------- 单例工厂 ----------
_GLOBAL_CLIENT: SearchClient | None = None


def get_default_client(
    engine_order: Iterable[str] | None = None,
    timeout: float = 8.0,
    proxy: str | None = None,
    merge_engines: bool = False,
) -> SearchClient:
    """模块级单例，复用 TCP 连接。"""
    global _GLOBAL_CLIENT
    if _GLOBAL_CLIENT is None:
        _GLOBAL_CLIENT = SearchClient(
            engine_order=tuple(engine_order) if engine_order else ("bing", "google"),
            timeout=timeout,
            proxy=proxy,
            merge_engines=merge_engines,
        )
    return _GLOBAL_CLIENT


# ---------- 自测入口 ----------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    client = SearchClient(
        engine_order=("bing", "baidu", "google"),
        timeout=8,
        sleep_between=(0.2, 0.5),
    )
    q = "违章停车 法律规定"
    print(f"\n========== query: {q} ==========")
    for eng in ("bing", "baidu", "google"):
        print(f"\n---------- engine: {eng} ----------")
        try:
            results = client.search_engine(eng, q, max_results=5)
            for i, r in enumerate(results, 1):
                print(f"{i}. [{r['engine']}] {r['title']}\n   url: {r['url']}\n   snippet: {r['snippet'][:120]}")
            if not results:
                print("(无结果：必应/百度可能被风控；google 需 SerpAPI key)")
        except Exception as e:
            print(f"engine={eng} 测试失败: {e}")
