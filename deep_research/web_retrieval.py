# -*- coding: utf-8 -*-
"""可信源联网检索与 embedding 证据抽取。"""
from __future__ import annotations

import html
import json
import logging
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests

from deep_research.parser_config import AppConfig
from deep_research.retrieve_knowledge import (
    _load_dense_model,
    _resolve_effective_dense_model,
    chinese_tokenizer,
)
from deep_research.search_client import SearchClient

logger = logging.getLogger(__name__)

_NEGATIVE_TITLE_TERMS = (
    "登录",
    "搜索",
    "检索",
    "下载",
    "招聘",
    "图片",
    "视频",
    "地图",
    "首页",
    "404",
)
_LEGAL_BOOST_TERMS = (
    "法律",
    "法规",
    "规定",
    "办法",
    "条例",
    "司法解释",
    "指导案例",
    "典型案例",
    "监管",
    "风险",
    "责任",
)
_HTML_DROP_RE = re.compile(
    r"(?is)<(script|style|noscript|iframe|nav|footer|header|aside|form|menu|svg|button|dialog)[^>]*>.*?</\1>"
)
_TAG_RE = re.compile(r"(?s)<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_WEB_NOISE_TERMS = (
    "免责声明",
    "不代表任何平台立场",
    "如若内容侵权",
    "投诉通道",
    "官方认证",
    "咨询我",
    "立即咨询",
    "免费咨询",
    "擅长",
    "浏览",
    "近期更新",
    "加入收藏",
    "设为首页",
    "无障碍",
    "相关推荐",
    "相关内容",
    "以上就是",
    "小编为大家整理",
    "希望能够对您有所帮助",
    "点击查看更多",
    "返回顶部",
    "网站地图",
    "扫码下载",
    "APP下载",
    "联系我们",
    "版权所有",
    "公安备案号",
    "ICP备",
    "关于我们",
    "分享至",
    "点赞",
    "当前位置",
    "您的位置",
    "所在位置",
    "首页",
    "频道",
    "正文",
    "来源：",
    "来源:",
    "作者：",
    "作者:",
    "编辑：",
    "编辑:",
    "记者",
    "时刻新闻",
    "分享—",
    "—分享—",
    "阅读全文",
    "点击查看",
    "下一篇",
    "上一篇",
    "相关阅读",
    "热门推荐",
    "猜你喜欢",
    "标签：",
    "发布时间",
    "发布于",
)
# 页内面包屑、元信息行（整段删除或句级过滤）
_NAV_INLINE_RE = re.compile(
    r"(?:当前位置[:：]\s*[^。！？]{0,160}?(?:正文|详情)|"
    r"(?:您的位置|所在位置)[:：]\s*[^。！？]{0,160}|"
    r"首页\s*[>＞/]\s*[^。！？]{0,120}|"
    r"(?:来源|作者|编辑|记者)[:：][^。！？]{0,80}(?:编辑|审核)[:：][^。！？]{0,40}|"
    r"时刻新闻\s*[—－-]\s*分享\s*[—－-]|"
    r"[-—]\s*分享\s*[-—])",
    re.IGNORECASE,
)
_BREADCRUMB_HEAVY_RE = re.compile(r"[>＞/]\s*[^>＞/。！？]{1,24}\s*[>＞/]")
_WEB_USEFUL_TERMS = (
    "可以",
    "不能",
    "不得",
    "应当",
    "有权",
    "支付",
    "结清",
    "工资",
    "赔偿",
    "维权",
    "投诉",
    "仲裁",
    "劳动",
    "合同",
    "试用期",
)

# 网页摘要模型判定与本案明显无关时，应只输出此标记（见 build_web_page_summary_prompt）
_PAGE_SUMMARY_IRRELEVANT_MARKER = "【无关】"


@dataclass
class WebSearchCandidate:
    title: str
    url: str
    snippet: str = ""
    source: str = ""
    published_at: str = ""
    search_query: str = ""
    title_score: float = 0.0


def _web_cfg(config: AppConfig) -> Any:
    return getattr(config, "web", None)


def web_enabled(config: AppConfig) -> bool:
    cfg = _web_cfg(config)
    return bool(cfg and getattr(cfg, "enabled", False))


def effective_user_question_for_web(user_query: str) -> str:
    """
    评测/批跑时常见问题被包在 JSON（如 {\"id\",\"input\"}）里传给联网层。
    若直接把整串 JSON 送去抽词与大模型生成 query，会出现「id 102 input …」一类噪音检索词。
    """
    s = (user_query or "").strip()
    if not s:
        return ""
    if s.startswith("{") and '"' in s:
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                for k in ("input", "question", "query", "text", "content", "problem"):
                    v = obj.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
        except json.JSONDecodeError:
            pass
    return s


def _hostname(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _is_trusted_url(url: str, trusted_domains: list[str]) -> bool:
    host = _hostname(url)
    if not host:
        return False
    for domain in trusted_domains:
        d = domain.lower().lstrip(".")
        if host == d or host.endswith("." + d):
            return True
    return False


def _site_clause(domains: list[str]) -> str:
    if not domains:
        return ""
    return " (" + " OR ".join(f"site:{domain}" for domain in domains) + ")"


def _keywords(text: str, max_terms: int = 12) -> list[str]:
    terms: list[str] = []
    for term in chinese_tokenizer(text or ""):
        term = term.strip()
        if len(term) > 1 and term not in terms:
            terms.append(term)
        if len(terms) >= max_terms:
            break
    return terms


def _page_summary_marked_irrelevant(summary: str) -> bool:
    """摘要模型判定网页与争点明显无关（见 prompts.build_web_page_summary_prompt）。"""
    s = (summary or "").strip()
    return s == _PAGE_SUMMARY_IRRELEVANT_MARKER or s.startswith(_PAGE_SUMMARY_IRRELEVANT_MARKER)


def _kb_field_str(item: dict[str, Any], key: str) -> str:
    """避免 item[key] 为 None 时被 str 成字面量 \"None\" 污染 query。"""
    if key not in item:
        return ""
    v = item.get(key)
    if v is None:
        return ""
    s = str(v).strip()
    if not s or s.lower() in ("none", "null", "<na>", "nan"):
        return ""
    return s


def _kb_terms(kb_results: list[dict[str, Any]] | None, max_terms: int = 10) -> list[str]:
    terms: list[str] = []
    for item in (kb_results or [])[:4]:
        for key in ("law_name", "category", "article", "number"):
            value = _kb_field_str(item, key)
            if not value:
                continue
            for part in _keywords(value, 4) or [value]:
                if part not in terms and part.lower() not in ("none", "null"):
                    terms.append(part)
            if len(terms) >= max_terms:
                return terms
    return terms


def _clean_search_query(q: str) -> str:
    q = re.sub(r"(?i)\bsite:[^\s]+", " ", q or "")
    q = re.sub(r"\b[Nn][Oo][Nn][Ee]\b", " ", q)
    q = q.replace("“", " ").replace("”", " ").replace('"', " ")
    q = q.replace("'", " ").replace("`", " ")
    q = _SPACE_RE.sub(" ", q).strip(" ，,。；;：:")
    return q[:80].strip()


def _strip_verbose_query_phrases(q: str) -> str:
    """去掉无检索价值的套话、口语框架与示例人物，便于短 query。"""
    if not q:
        return ""
    s = q
    s = re.sub(r"小[\u4e00-\u9fa5]{1,3}(?=向|到|在|因|把|将|被|与|和|的|，|,|。|\s)", " ", s)
    s = re.sub(r"(?:张三|李四|王五|赵六|甲某|乙某|丙某|丁某)", " ", s)
    for phrase in (
        "法律依据",
        "裁判观点",
        "权威解读",
        "官方政策",
        "相关规定",
        "法律适用",
        "司法解释",
        "构成要件",
        "效力",
        "裁判",
        "观点",
        "解读",
        "实务",
        "请问",
        "是否可以",
        "可不可以",
        "能否",
        "会不会",
        "为什么",
        "什么原因",
        "怎么办",
        "怎么处理",
        "先生",
        "某公司",
        "某个公司",
        "某人",
        "某甲",
        "某乙",
        "某丙",
        "某丁",
        "一名",
        "如果",
        "假如",
        "假设",
        "是否",
        "可否",
    ):
        s = s.replace(phrase, " ")
    return _SPACE_RE.sub(" ", s).strip()


# 联网检索词套话（整词剔除；仅无区分度的空泛表述）
_WEB_QUERY_STOP_TERMS = frozenset({
    "裁判观点", "效力", "法律依据", "权威解读", "相关规定", "法律适用",
    "司法解释", "构成要件", "解读", "实务",
})


def _shorten_search_query(q: str, max_chars: int = 36, max_terms: int = 3) -> str:
    """压成短串；每条检索词最多 max_terms 个词，剔除套话。"""
    q = _clean_search_query(q)
    q = _strip_verbose_query_phrases(q)
    q = re.sub(r"[？?！!。．…]", " ", q)
    q = _SPACE_RE.sub(" ", q).strip()
    if not q:
        return ""
    parts = [
        p for p in q.split()
        if p and p not in _WEB_QUERY_STOP_TERMS and len(p) > 1
    ][:max_terms]
    compact = " ".join(parts)
    while len(compact) > max_chars and len(parts) > 1:
        parts.pop()
        compact = " ".join(parts)
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].strip()


def _mild_search_query_trim(q: str, max_chars: int = 72) -> str:
    """仅清理与截断，不用 _strip_verbose_query_phrases，避免合法检索词被削成空。"""
    q = _clean_search_query(q)
    q = re.sub(r"[？?！!。．…]", " ", q)
    q = _SPACE_RE.sub(" ", q).strip()
    if not q:
        return ""
    return q[:max_chars].strip()


def build_search_queries(
    user_query: str,
    config: AppConfig,
    planned_queries: list[str],
) -> list[str]:
    """规范化 retrieval_plan 给出的联网检索词（由规划 LLM 生成，本层不做词表硬编码）。"""
    cfg = _web_cfg(config)
    if not cfg:
        return []
    max_queries = max(1, int(getattr(cfg, "max_queries", 3)))

    queries: list[str] = []
    for q in planned_queries:
        s = _shorten_search_query(str(q or ""), max_terms=3)
        if not s:
            s = _mild_search_query_trim(str(q or ""), max_chars=36)
            if s:
                s = _shorten_search_query(s, max_terms=3)
        if s and s not in queries:
            queries.append(s)
    if not queries:
        logger.warning("[Web] retrieval_plan 无有效 web_queries，使用问句兜底")
        q = _shorten_search_query(user_query, max_chars=50, max_terms=8)
        queries.append(q if q else user_query)

    if bool(getattr(cfg, "enforce_trusted_domains", False)):
        site_filter = _site_clause(list(getattr(cfg, "trusted_domains", []) or []))
        queries = [q if "site:" in q else f"{q}{site_filter}" for q in queries]
    logger.info("[Web] search queries: %s", queries[:max_queries])
    return queries[:max_queries]


_SEARCH_CLIENT: SearchClient | None = None
_SEARCH_CLIENT_FINGERPRINT: tuple[Any, ...] | None = None


_ALLOWED_SEARCH_ENGINES = frozenset({"bing", "baidu", "google"})


def _canonical_search_engine(name: str) -> str:
    return (name or "").strip().lower()


def _resolve_serpapi_key(cfg: Any) -> str | None:
    """优先读 config.web.serpapi_api_key；空则回退环境变量 serpapi_key_env(默认 SERPAPI_API_KEY)。"""
    key = (getattr(cfg, "serpapi_api_key", "") or "").strip()
    if key:
        return key
    env_name = (getattr(cfg, "serpapi_key_env", "") or "SERPAPI_API_KEY").strip()
    if env_name:
        env_val = (os.environ.get(env_name) or "").strip()
        if env_val:
            return env_val
    return None


def _search_client_fingerprint(cfg: Any) -> tuple[Any, ...]:
    """配置变更时重建 SearchClient，避免进程未重启仍用空代理等旧参数。"""
    engine_order = _effective_engine_order(cfg)
    if not engine_order:
        engine_order = ("bing", "google")
    timeout = float(
        getattr(cfg, "search_request_timeout", None)
        or getattr(cfg, "request_timeout", 10.0)
        or 10.0
    )
    proxy = (getattr(cfg, "search_proxy", "") or "").strip() or None
    merge_engines = bool(getattr(cfg, "search_merge_engines", False))
    sleep_lo = float(getattr(cfg, "search_sleep_min", 0.2) or 0.0)
    sleep_hi = float(getattr(cfg, "search_sleep_max", 0.6) or 0.0)
    serpapi_key = _resolve_serpapi_key(cfg)
    serpapi_url = (getattr(cfg, "serpapi_url", "") or "").strip() or "https://serpapi.com/search.json"
    return (engine_order, timeout, proxy, merge_engines, sleep_lo, sleep_hi, serpapi_key, serpapi_url)


def _effective_engine_order(cfg: Any) -> tuple[str, ...]:
    """优先使用 search_engine 单选；否则使用 engine_order 列表。"""
    se_raw = getattr(cfg, "search_engine", "") or ""
    if se_raw.strip():
        canon = _canonical_search_engine(se_raw)
        if canon in _ALLOWED_SEARCH_ENGINES:
            return (canon,)
        logger.warning(
            "[Web] web.search_engine=%s 无效，允许值为 bing / baidu / google（google 仅 SerpAPI）。改用 engine_order。",
            getattr(cfg, "search_engine", ""),
        )
    raw = getattr(cfg, "engine_order", None) or ("bing", "google")
    return tuple(_canonical_search_engine(str(x)) for x in raw if str(x).strip())


def _get_search_client(cfg: Any) -> SearchClient:
    """模块级 SearchClient；search_proxy / engine_order 等变化时自动重建。"""
    global _SEARCH_CLIENT, _SEARCH_CLIENT_FINGERPRINT
    fp = _search_client_fingerprint(cfg)
    if _SEARCH_CLIENT is not None and fp == _SEARCH_CLIENT_FINGERPRINT:
        return _SEARCH_CLIENT

    engine_order = fp[0]
    timeout = fp[1]
    proxy = fp[2]
    merge_engines = fp[3]
    sleep_lo, sleep_hi = fp[4], fp[5]
    serpapi_key = fp[6]
    serpapi_url = fp[7]

    _SEARCH_CLIENT = SearchClient(
        engine_order=engine_order,
        timeout=timeout,
        sleep_between=(sleep_lo, sleep_hi),
        proxy=proxy,
        merge_engines=merge_engines,
        serpapi_api_key=serpapi_key,
        serpapi_url=serpapi_url,
    )
    _SEARCH_CLIENT_FINGERPRINT = fp
    se_cfg = (getattr(cfg, "search_engine", "") or "").strip()
    logger.info(
        "[Web] SearchClient 初始化: search_engine=%s effective_order=%s timeout=%.1fs merge=%s proxy=%s serpapi=%s url=%s",
        se_cfg or "(use engine_order)", engine_order, timeout, merge_engines, proxy or "-",
        "set" if serpapi_key else "unset",
        serpapi_url,
    )
    return _SEARCH_CLIENT


def reset_search_client_for_question(cfg: Any) -> None:
    """每道新题开始前重置 SearchClient 会话（Cookie/百度预热状态）。"""
    client = _get_search_client(cfg)
    client.reset_for_new_question()
    logger.info("[Web] SearchClient 已为本题重置会话")


def _raw_search_to_candidates(
    raw: list[dict[str, Any]],
    query: str,
) -> list[WebSearchCandidate]:
    candidates: list[WebSearchCandidate] = []
    for item in raw:
        url = (item.get("url") or "").strip()
        title = (item.get("title") or "").strip()
        if not url or not title:
            continue
        candidates.append(
            WebSearchCandidate(
                title=title,
                url=url,
                snippet=(item.get("snippet") or "").strip(),
                source=(item.get("engine") or ""),
                published_at="",
                search_query=query,
            )
        )
    return candidates


def _dedupe_candidates(candidates: list[WebSearchCandidate]) -> list[WebSearchCandidate]:
    seen: set[str] = set()
    out: list[WebSearchCandidate] = []
    for c in candidates:
        key = c.url.split("#", 1)[0].rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _run_search_queries(
    queries: list[str],
    config: AppConfig,
    *,
    engine_order: tuple[str, ...],
) -> list[WebSearchCandidate]:
    """按指定引擎顺序执行一组 query（单引擎或完整降级链）。"""
    cfg = _web_cfg(config)
    if not cfg:
        return []
    client = _get_search_client(cfg)
    max_results = int(getattr(cfg, "max_search_results", 10) or 10)
    batch: list[WebSearchCandidate] = []
    for query in queries:
        try:
            raw = client.search(query, max_results=max_results, engine_order=engine_order)
        except Exception as e:
            logger.warning("[Web] 搜索失败 query=%s engines=%s err=%s", query, engine_order, e)
            continue
        batch.extend(_raw_search_to_candidates(raw, query))
    return batch


def collect_search_candidates(
    queries: list[str],
    config: AppConfig,
) -> tuple[list[WebSearchCandidate], dict[str, Any]]:
    """本题联网搜索：主引擎重试 → 仍空则按 engine_order 切换下一引擎。"""
    cfg = _web_cfg(config)
    if not cfg or not queries:
        return [], {"engines_tried": [], "rounds": 0}

    order = _effective_engine_order(cfg) or ("baidu", "bing")
    if not order:
        order = ("bing", "google")

    cooldown = float(getattr(cfg, "search_round_cooldown_s", 4.0) or 4.0)
    max_rounds = max(1, int(getattr(cfg, "search_round_retries", 2) or 2))

    reset_search_client_for_question(cfg)
    all_candidates: list[WebSearchCandidate] = []
    meta: dict[str, Any] = {
        "engine_order": list(order),
        "engines_tried": [],
        "rounds": 0,
        "fallback_engine": None,
    }

    primary = (order[0],)
    meta["engines_tried"].append(order[0])

    for round_idx in range(max_rounds):
        meta["rounds"] = round_idx + 1
        if round_idx > 0:
            logger.info(
                "[Web] 主引擎 %s 本轮无候选，冷却 %.1fs 后第 %d/%d 轮重试",
                order[0],
                cooldown,
                round_idx + 1,
                max_rounds,
            )
            time.sleep(cooldown)
            reset_search_client_for_question(cfg)

        batch = _run_search_queries(queries, config, engine_order=primary)
        all_candidates = _dedupe_candidates(batch)
        if all_candidates:
            meta["primary_round"] = round_idx + 1
            meta["candidate_count_after_primary"] = len(all_candidates)
            return all_candidates, meta

    for eng in order[1:]:
        logger.info(
            "[Web] 主引擎 %s 多轮仍无结果，切换降级引擎 %s（冷却 %.1fs）",
            order[0],
            eng,
            cooldown,
        )
        time.sleep(cooldown)
        reset_search_client_for_question(cfg)
        meta["engines_tried"].append(eng)
        meta["fallback_engine"] = eng
        batch = _run_search_queries(queries, config, engine_order=(eng,))
        all_candidates = _dedupe_candidates(batch)
        if all_candidates:
            meta["candidate_count_after_fallback"] = len(all_candidates)
            return all_candidates, meta

    logger.warning(
        "[Web] 全部引擎均无候选 queries=%s tried=%s",
        queries,
        meta["engines_tried"],
    )
    return [], meta


def search_web(
    query: str,
    config: AppConfig,
    *,
    engine_order: tuple[str, ...] | None = None,
) -> list[WebSearchCandidate]:
    """联网搜索：按 engine_order（如 baidu → bing → google）。"""
    cfg = _web_cfg(config)
    if not cfg:
        return []
    client = _get_search_client(cfg)
    max_results = int(getattr(cfg, "max_search_results", 10) or 10)
    order = engine_order if engine_order is not None else _effective_engine_order(cfg)
    try:
        raw = client.search(query, max_results=max_results, engine_order=order)
    except Exception as e:
        logger.warning("[Web] 搜索失败 query=%s err=%s", query, e)
        return []
    return _raw_search_to_candidates(raw, query)


# 向后兼容：旧代码仍可调用 search_serpapi，会自动转发到新实现。
def search_serpapi(query: str, config: AppConfig) -> list[WebSearchCandidate]:
    logger.debug("[Web] search_serpapi 已废弃，转发到 search_web")
    return search_web(query, config)


def _title_score(candidate: WebSearchCandidate, query_terms: list[str], search_terms: list[str] | None = None) -> float:
    text = f"{candidate.title} {candidate.snippet}".lower()
    score = 0.0
    for term in query_terms:
        if not term:
            continue
        t = term.lower()
        if t in candidate.title.lower():
            score += 1.5
        elif t in text:
            score += 0.7
    for term in _LEGAL_BOOST_TERMS:
        if term in text:
            score += 0.3
    for term in _NEGATIVE_TITLE_TERMS:
        if term in candidate.title:
            score -= 1.0
    for term in search_terms or []:
        t = term.lower()
        if len(t) > 1 and t in text:
            score += 0.5
    return score


def filter_candidates(
    candidates: list[WebSearchCandidate],
    user_query: str,
    kb_results: list[dict[str, Any]] | None,
    config: AppConfig,
) -> tuple[list[WebSearchCandidate], list[dict[str, str]]]:
    """
    第一阶段：仅依据搜索结果标题+摘要与问句的相关性筛 URL，通过者才进入抓取。
    """
    cfg = _web_cfg(config)
    if not cfg:
        return [], []
    trusted_domains = list(getattr(cfg, "trusted_domains", []) or [])
    enforce_domains = bool(getattr(cfg, "enforce_trusted_domains", False))
    threshold = float(getattr(cfg, "title_score_threshold", 1.0))
    query_terms = _keywords(user_query, 10) + _kb_terms(kb_results, 8)
    seen: set[str] = set()
    kept: list[WebSearchCandidate] = []
    rejects: list[dict[str, str]] = []
    for cand in candidates:
        url_key = cand.url.split("#", 1)[0].rstrip("/")
        if url_key in seen:
            rejects.append({"url": cand.url, "reason": "重复 URL"})
            continue
        seen.add(url_key)
        if enforce_domains and not _is_trusted_url(cand.url, trusted_domains):
            rejects.append({"url": cand.url, "reason": "非可信域名"})
            continue
        cand.title_score = _title_score(cand, query_terms, _keywords(cand.search_query, 8))
        if cand.title_score < threshold:
            rejects.append({"url": cand.url, "reason": f"标题摘要相关性低: {cand.title_score:.2f}"})
            continue
        kept.append(cand)
    kept.sort(key=lambda x: x.title_score, reverse=True)
    top_n = kept[: int(getattr(cfg, "max_fetch_pages", 3))]
    if top_n:
        logger.info(
            "[Web] 标题筛选保留 %d/%d 条，Top: %s",
            len(top_n),
            len(candidates),
            [(c.title[:40], round(c.title_score, 2)) for c in top_n[:3]],
        )
    return top_n, rejects


_UNRELIABLE_HTTP_ENCODINGS = frozenset(
    {"iso-8859-1", "latin-1", "latin1", "windows-1252", "cp1252"}
)


def _normalize_codec_name(name: str | None) -> str | None:
    if not name:
        return None
    n = name.strip().lower().replace("utf8", "utf-8")
    if n in ("gb2312", "gbk", "cp936"):
        return "gb18030"
    return n


def _html_decode_score(sample: str, hint_zh_site: bool) -> float:
    """偏好在 hint 为中文站点时命中汉字、惩罚 � 与典型「拉丁化乱码」字节形字符。"""
    if not sample:
        return 0.0
    t = sample[:12000]
    cjk = sum(1 for c in t if "\u4e00" <= c <= "\u9fff")
    rep = t.count("\ufffd")
    mojibake = sum(1 for c in t if "\u0080" <= c <= "\u00ff")
    base = min(len(t), 800) * 0.02
    if hint_zh_site:
        return float(cjk) * 4.0 + base - float(rep) * 90.0 - float(mojibake) * 0.35
    return float(cjk) * 3.0 + base - float(rep) * 90.0 - float(mojibake) * 0.2


def _try_repair_utf8_misread_as_latin1(text: str) -> str:
    """把 UTF-8 字节被误当成 latin-1 解码时产生的乱码尝试还原为 UTF-8。"""
    if not text or len(text) < 12:
        return text
    cjk0 = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    if cjk0 > len(text) * 0.06:
        return text
    try:
        raw_l1 = text.encode("latin-1")
        fixed = raw_l1.decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text
    cjk1 = sum(1 for c in fixed if "\u4e00" <= c <= "\u9fff")
    if cjk1 >= cjk0 + 8 and cjk1 >= max(12, len(fixed) * 0.04):
        return fixed
    return text


def _decode_http_response_html(resp: requests.Response, hint_url: str = "") -> str:
    """
    按响应头、<meta charset> 与 utf-8/gb18030 候选解码，避免 ISO-8859-1 默认误读中文官网站点。
    """
    raw: bytes = resp.content or b""
    if not raw:
        return ""
    hint_zh = any(
        x in (hint_url or "").lower()
        for x in (".cn", ".gov", "spp.gov", "court.gov", "npc.gov", "legalinfo")
    )
    candidates: list[str] = []

    def _push(enc: str | None) -> None:
        n = _normalize_codec_name(enc)
        if n and n not in candidates:
            candidates.append(n)

    hdr = requests.utils.get_encoding_from_headers(resp.headers)
    if hdr and hdr.strip().lower() not in _UNRELIABLE_HTTP_ENCODINGS:
        _push(hdr)

    prefix = raw[:65536]
    for m in re.finditer(rb'charset\s*=\s*["\']?([a-zA-Z0-9_. -]+)', prefix, re.I):
        try:
            _push(m.group(1).decode("ascii", errors="ignore"))
        except Exception:
            pass

    for fb in ("utf-8-sig", "utf-8", "gb18030"):
        _push(fb)

    best_text = ""
    best_sc = -1e18
    for enc in candidates:
        try:
            text = raw.decode(enc, errors="strict")
            sc = _html_decode_score(text, hint_zh)
        except (LookupError, UnicodeDecodeError):
            try:
                text = raw.decode(enc, errors="replace")
            except LookupError:
                continue
            sc = _html_decode_score(text, hint_zh) - float(text.count("\ufffd")) * 120.0
        if sc > best_sc:
            best_sc = sc
            best_text = text
    if not best_text:
        best_text = raw.decode("utf-8", errors="replace")
    return _try_repair_utf8_misread_as_latin1(best_text)


def _looks_like_mojibake(s: str) -> bool:
    if not s or len(s) < 6:
        return False
    cjk = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
    if cjk >= max(5, len(s) * 0.12):
        return False
    hi = sum(1 for c in s if "\u0080" <= c <= "\u00ff")
    return hi >= len(s) * 0.22


def _normalize_title_text(raw: str) -> str:
    t = html.unescape(_SPACE_RE.sub(" ", _TAG_RE.sub("", raw or "")).strip())
    # 常见站点后缀噪声
    for sep in (" - ", " | ", " _ ", "——", " -"):
        if sep in t:
            left = t.split(sep, 1)[0].strip()
            if len(left) >= 6:
                t = left
                break
    return t


def _extract_page_title(raw_html: str, api_title: str = "") -> str:
    """优先 og:title / h1，再 <title>，最后搜索引擎标题。"""
    html_src = raw_html or ""
    candidates: list[str] = []
    for pattern in (
        r'(?is)<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']',
        r'(?is)<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:title["\']',
        r'(?is)<meta[^>]+name=["\']twitter:title["\'][^>]+content=["\']([^"\']+)["\']',
        r"(?is)<h1[^>]*>(.*?)</h1>",
        r"(?is)<title[^>]*>(.*?)</title>",
    ):
        m = re.search(pattern, html_src)
        if not m:
            continue
        t = _normalize_title_text(m.group(1))
        if len(t) >= 4 and t not in candidates:
            candidates.append(t)
    if not candidates:
        return _normalize_title_text(api_title)
    # candidates 已按 og:title → h1 → <title> 优先级入列，取首个即可
    return candidates[0]


def _html_to_text(raw_html: str) -> tuple[str, str]:
    title = _extract_page_title(raw_html)
    body = _HTML_DROP_RE.sub(" ", raw_html or "")
    body = _TAG_RE.sub(" ", body)
    body = html.unescape(body)
    body = _SPACE_RE.sub(" ", body).strip()
    return title, body


def _strip_inline_nav(text: str) -> str:
    """去掉面包屑、来源/作者行等内联导航片段。"""
    s = _SPACE_RE.sub(" ", text or "").strip()
    if not s:
        return ""
    s = _NAV_INLINE_RE.sub(" ", s)
    return _SPACE_RE.sub(" ", s).strip()


def _is_noise_sentence(sentence: str) -> bool:
    s = (sentence or "").strip()
    if not s:
        return True
    if s.count("#") >= 2:
        return True
    if any(term in s for term in _WEB_NOISE_TERMS):
        return True
    # 面包屑：多级「> / ＞」且较短
    if len(s) < 120 and _BREADCRUMB_HEAVY_RE.search(s) and not any(
        term in s for term in _WEB_USEFUL_TERMS
    ):
        return True
    # 纯元信息行：来源/作者/编辑堆叠且无规范用语
    if re.match(
        r"^(?:来源|作者|编辑|记者|发布时间|发布于)[:：]",
        s,
    ) and not any(term in s for term in ("应当", "不得", "有权", "可以", "赔偿", "条")):
        if len(s) < 80:
            return True
    # 过短且不含实质判断词的碎片，多为导航或标签。
    if len(s) < 18 and not any(term in s for term in _WEB_USEFUL_TERMS):
        return True
    return False


def _trim_page_leading_noise(text: str) -> str:
    """去掉页头连续导航/元信息句，尽量从正文首句开始。"""
    text = _strip_inline_nav(text)
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return text
    start = 0
    for i, sent in enumerate(sentences):
        if _is_noise_sentence(sent):
            continue
        if len(sent) >= 30 or any(term in sent for term in _WEB_USEFUL_TERMS):
            start = i
            break
    trimmed = "".join(sentences[start:])
    return _SPACE_RE.sub(" ", trimmed).strip() or text


def _nav_noise_ratio(text: str) -> float:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text or "") if s.strip()]
    if not sentences:
        return 0.0
    noise = sum(1 for s in sentences if _is_noise_sentence(s))
    return noise / len(sentences)


def _clean_web_content(text: str) -> str:
    text = _strip_inline_nav(_SPACE_RE.sub(" ", text or "").strip())
    if not text:
        return ""
    # 律师/资讯站常见的一串 SEO 话题标签，对相关性打分干扰很大。
    text = re.sub(r"(?:#\s*[^#。！？；;]{1,48}\s*){2,}", " ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return _trim_page_leading_noise(text)
    cleaned: list[str] = []
    for sent in sentences:
        if _is_noise_sentence(sent):
            continue
        cleaned.append(sent)
    cleaned_text = _SPACE_RE.sub(" ", "".join(cleaned)).strip()
    if len(cleaned_text) >= 200:
        return _trim_page_leading_noise(cleaned_text)
    # 句级过滤过猛时，至少做页头裁剪
    leading_trimmed = _trim_page_leading_noise(text)
    if len(leading_trimmed) >= int(max(200, len(text) * 0.25)):
        return leading_trimmed
    return text


def _record_fetch_failure(
    failures: list[dict[str, str]] | None,
    candidate: WebSearchCandidate,
    reason: str,
    final_url: str = "",
) -> None:
    if failures is None:
        return
    failures.append(
        {
            "title": candidate.title,
            "url": candidate.url,
            "final_url": final_url or candidate.url,
            "reason": reason,
        }
    )


def _build_fetch_headers(target_url: str) -> dict[str, str]:
    """
    部分站点（如知乎）仅用极简 UA 会返回 403，补充浏览器常见头与同源 Referer 可降低被拒概率；
    若仍 403，多为 IP/账号风控，只能换代理或使用无头浏览器抓取。
    """
    ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )
    headers = {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Upgrade-Insecure-Requests": "1",
        "DNT": "1",
        "Connection": "keep-alive",
    }
    low = (target_url or "").lower()
    if "zhihu.com" in low:
        # 专栏与问题页同属 zhihu.com，首页作为 Referer 通常更易通过 WAF 校验
        headers["Referer"] = "https://www.zhihu.com/"
        headers["Sec-Fetch-Dest"] = "document"
        headers["Sec-Fetch-Mode"] = "navigate"
        headers["Sec-Fetch-Site"] = "same-site"
        headers["Sec-Fetch-User"] = "?1"
    elif "baike.baidu.com" in low:
        headers["Referer"] = "https://www.baidu.com/"
        headers["Sec-Fetch-Site"] = "same-site"
    elif "baijiahao.baidu.com" in low or "mbd.baidu.com" in low:
        headers["Referer"] = "https://www.baidu.com/"
        headers["Sec-Fetch-Site"] = "same-site"
    return headers


def fetch_webpage(
    candidate: WebSearchCandidate,
    config: AppConfig,
    failures: list[dict[str, str]] | None = None,
) -> dict[str, Any] | None:
    cfg = _web_cfg(config)
    if not cfg:
        return None
    trusted_domains = list(getattr(cfg, "trusted_domains", []) or [])
    enforce_domains = bool(getattr(cfg, "enforce_trusted_domains", False))
    headers = _build_fetch_headers(candidate.url)
    try:
        resp = requests.get(
            candidate.url,
            headers=headers,
            timeout=float(getattr(cfg, "request_timeout", 10.0)),
            allow_redirects=True,
        )
        resp.raise_for_status()
    except Exception as e:
        reason = f"请求失败: {type(e).__name__}: {e}"
        logger.info("[Web] 抓取失败 %s: %s", candidate.url, e)
        _record_fetch_failure(failures, candidate, reason)
        return None
    final_url = resp.url
    if enforce_domains and not _is_trusted_url(final_url, trusted_domains):
        logger.info("[Web] 跳转后非可信域名，丢弃: %s -> %s", candidate.url, final_url)
        _record_fetch_failure(failures, candidate, "跳转后非可信域名", final_url)
        return None
    content_type = (resp.headers.get("content-type") or "").lower()
    if "pdf" in content_type:
        logger.info("[Web] 暂不处理 PDF: %s", final_url)
        _record_fetch_failure(failures, candidate, "PDF 暂不处理", final_url)
        return None
    html_text = _decode_http_response_html(resp, hint_url=final_url)
    api_title = (candidate.title or "").strip()
    title, content = _html_to_text(html_text)
    if _looks_like_mojibake(title) and api_title:
        title = _normalize_title_text(api_title)
    elif not title.strip():
        title = _normalize_title_text(api_title)
    content = _clean_web_content(_try_repair_utf8_misread_as_latin1(content))
    if len(content) < int(getattr(cfg, "min_content_length", 300)):
        logger.info("[Web] 正文过短，丢弃: %s len=%d", final_url, len(content))
        _record_fetch_failure(failures, candidate, f"正文过短: len={len(content)}", final_url)
        return None
    return {
        "title": title,
        "url": final_url,
        "source": candidate.source or _hostname(final_url),
        "published_at": candidate.published_at,
        "content": content,
        "title_score": candidate.title_score,
        "search_query": candidate.search_query,
    }


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = _SPACE_RE.sub(" ", text or "").strip()
    if not text:
        return []
    chunk_size = max(200, chunk_size)
    overlap = max(0, min(overlap, chunk_size // 2))
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return [c for c in chunks if len(c) >= 80]


def chunk_text_spans(text: str, chunk_size: int, overlap: int) -> list[dict[str, Any]]:
    text = _SPACE_RE.sub(" ", text or "").strip()
    if not text:
        return []
    chunk_size = max(200, chunk_size)
    overlap = max(0, min(overlap, chunk_size // 2))
    spans: list[dict[str, Any]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if len(chunk) >= 80:
            spans.append({"start": start, "end": end, "text": chunk})
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return spans


def _unique_terms(*groups: list[str]) -> list[str]:
    terms: list[str] = []
    for group in groups:
        for term in group:
            t = (term or "").strip()
            if len(t) > 1 and t not in terms:
                terms.append(t)
    return terms


def _term_relevance_score(query_terms: list[str], text: str) -> float:
    """0-1 词面相关性：兼顾覆盖率、命中次数和关键长词。"""
    if not query_terms or not text:
        return 0.0
    lower = text.lower()
    weighted_total = 0.0
    weighted_hit = 0.0
    freq_score = 0.0
    for term in query_terms:
        t = term.lower().strip()
        if len(t) <= 1:
            continue
        weight = 1.0 + min(len(t), 8) / 8.0
        weighted_total += weight
        count = lower.count(t)
        if count:
            weighted_hit += weight
            freq_score += min(count, 3) * weight
    if weighted_total <= 0:
        return 0.0
    coverage = weighted_hit / weighted_total
    density = min(freq_score / math.sqrt(len(lower) + 1), 1.0)
    return min(1.0, 0.75 * coverage + 0.25 * density)


def _lexical_score(query_terms: list[str], text: str) -> float:
    if not query_terms:
        return 0.0
    return _term_relevance_score(query_terms, text)


def _normalize_embedding_score(score: float) -> float:
    if -1.0 <= score <= 1.0:
        return max(0.0, min(1.0, (score + 1.0) / 2.0))
    return max(0.0, min(1.0, score))


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；;.!?])\s*")


def _best_quote_window(text: str, query_terms: list[str], max_chars: int = 800) -> str:
    """从高相关 chunk 内再抽一个更聚焦的句子窗口，避免整块 quote 过长且主题发散。"""
    cleaned = _trim_page_leading_noise(_SPACE_RE.sub(" ", text or "").strip())
    if len(cleaned) <= max_chars:
        return cleaned
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(cleaned) if s.strip()]
    if not sentences:
        return cleaned[:max_chars].strip()
    windows: list[str] = []
    for i in range(len(sentences)):
        if _is_noise_sentence(sentences[i]):
            continue
        buf = ""
        for sent in sentences[i:]:
            if _is_noise_sentence(sent):
                continue
            if buf and len(buf) + len(sent) > max_chars:
                break
            buf += sent
            if len(buf) >= 200:
                windows.append(buf.strip())
        if len(windows) > 200:
            break
    if not windows:
        windows = [cleaned[:max_chars].strip()]
    return max(windows, key=lambda x: _term_relevance_score(query_terms, x) + _useful_sentence_bonus(x))


def _useful_sentence_bonus(text: str) -> float:
    if not text:
        return 0.0
    useful = sum(1 for term in _WEB_USEFUL_TERMS if term in text)
    noise = sum(1 for term in _WEB_NOISE_TERMS if term in text)
    return min(useful * 0.03, 0.18) - min(noise * 0.08, 0.4)


def _context_window(text: str, start: int, end: int, query_terms: list[str], max_chars: int = 1800) -> str:
    """给模型的长正文窗口：围绕高分 chunk 扩展上下文，但仍过滤页面噪声。"""
    cleaned = _trim_page_leading_noise(_SPACE_RE.sub(" ", text or "").strip())
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    center = max(0, (start + end) // 2)
    left = max(0, center - max_chars // 2)
    right = min(len(cleaned), left + max_chars)
    left = max(0, right - max_chars)

    # 尽量从句子边界开始/结束，避免截断重要句子。
    left_boundary = cleaned.rfind("。", 0, left)
    if left_boundary != -1 and left - left_boundary < 120:
        left = left_boundary + 1
    right_boundary = cleaned.find("。", right)
    if right_boundary != -1 and right_boundary - right < 120:
        right = right_boundary + 1
    window = cleaned[left:right].strip()
    # 如果扩展窗口关键词很少，则退回最相关短摘录，避免给模型太多噪声。
    if _term_relevance_score(query_terms, window) < 0.08:
        return _best_quote_window(cleaned[start:end], query_terms, max_chars=min(max_chars, 800))
    return window


def _hybrid_chunk_score(
    *,
    embedding_score: float | None,
    lexical_score: float,
    title_score: float,
    query_score: float,
) -> float:
    title_norm = max(0.0, min(1.0, title_score / 8.0))
    if embedding_score is None:
        return round(0.70 * lexical_score + 0.20 * title_norm + 0.10 * query_score, 4)
    emb_norm = _normalize_embedding_score(embedding_score)
    return round(0.55 * emb_norm + 0.25 * lexical_score + 0.12 * title_norm + 0.08 * query_score, 4)


def _embedding_scores(query: str, chunks: list[str], config: AppConfig) -> list[float] | None:
    if not chunks:
        return []
    try:
        import numpy as np
    except ImportError:
        return None
    model_name = _resolve_effective_dense_model(config)
    model = _load_dense_model(model_name, config)
    if model is None:
        return None
    try:
        vectors = model.encode([query, *chunks], batch_size=min(32, len(chunks) + 1), show_progress_bar=False)
        arr = np.asarray(vectors, dtype="float32")
        if arr.ndim != 2 or arr.shape[0] != len(chunks) + 1:
            return None
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1
        arr = arr / norms
        return [float(x) for x in (arr[1:] @ arr[0])]
    except Exception as e:
        logger.warning("[Web] embedding 重排失败，退化为关键词排序: %s", e)
        return None


def _normalize_evidence_url(url: str) -> str:
    return (url or "").strip().split("#", 1)[0].rstrip("/").lower()


def _dedupe_evidence_by_url(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """同一 URL 只保留 relevance_score 最高的一条，并重排 citation_id。"""
    best: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for e in evidence:
        u = _normalize_evidence_url(str(e.get("url", "")))
        if not u:
            continue
        sc = float(e.get("relevance_score", 0.0) or 0.0)
        if u not in best or sc > float(best[u].get("relevance_score", 0.0) or 0.0):
            if u not in best:
                order.append(u)
            best[u] = e
    out: list[dict[str, Any]] = []
    for u in order:
        item = dict(best[u])
        item["citation_id"] = f"W{len(out) + 1}"
        out.append(item)
    return out


def _page_text_for_scoring(page: dict[str, Any], max_body_chars: int = 5000) -> str:
    title = (page.get("title") or "").strip()
    body = (page.get("content") or "").strip()
    if len(body) > max_body_chars:
        body = body[:max_body_chars] + "…"
    return f"{title}\n{body}".strip()


def _score_page_relevance(
    page: dict[str, Any],
    rank_query: str,
    query_terms: list[str],
    config: AppConfig,
) -> tuple[float, dict[str, Any]]:
    text = _page_text_for_scoring(page)
    lexical = _lexical_score(query_terms, text)
    emb_list = _embedding_scores(rank_query, [text], config)
    emb = emb_list[0] if emb_list else None
    title_score = float(page.get("title_score", 0.0) or 0.0)
    page_terms = _keywords(
        " ".join([page.get("title", ""), page.get("search_query", "")]), 8
    )
    query_score = _term_relevance_score(page_terms, text)
    score = _hybrid_chunk_score(
        embedding_score=emb,
        lexical_score=lexical,
        title_score=title_score,
        query_score=query_score,
    )
    nav_penalty = min(0.35, _nav_noise_ratio(page.get("content", "") or "") * 0.5)
    final = round(score * (1.0 - nav_penalty), 4)
    detail = {
        "embedding": round(float(emb), 4) if emb is not None else None,
        "lexical": round(lexical, 4),
        "title": round(title_score, 4),
        "query": round(query_score, 4),
        "nav_penalty": round(nav_penalty, 4),
    }
    return final, detail


def _evidence_from_full_page(
    page: dict[str, Any],
    score: float,
    score_detail: dict[str, Any],
    *,
    delivery: str = "full_page",
) -> dict[str, Any]:
    body = (page.get("content") or "").strip()
    return {
        "title": page.get("title", ""),
        "url": page.get("url", ""),
        "source": page.get("source", ""),
        "published_at": page.get("published_at", ""),
        "quote": body[: min(2000, len(body))],
        "content": body,
        "extracted_content": body,
        "content_chars": len(body),
        "relevance_score": score,
        "score_detail": score_detail,
        "source_type": "web",
        "delivery": delivery,
    }


def _build_title_selected_evidence(
    pages: list[dict[str, Any]],
    config: AppConfig,
) -> list[dict[str, Any]]:
    """
    第二阶段：对标题筛选后且抓取成功的页面，按标题分排序，整页正文（可截断）作为一条证据。
    不再用切块+embedding 在页内抢片段，避免标题已相关但正文被切块丢掉。
    """
    cfg = _web_cfg(config)
    if not pages:
        return []
    full_max = int(getattr(cfg, "full_page_max_chars", 12000) or 12000)
    max_items = int(getattr(cfg, "max_evidence_items", 5) or 5)
    max_items = min(max_items, int(getattr(cfg, "max_full_page_items", max_items) or max_items))

    ordered = sorted(
        pages,
        key=lambda p: float(p.get("title_score", 0.0) or 0.0),
        reverse=True,
    )
    evidence: list[dict[str, Any]] = []
    for page in ordered:
        if len(evidence) >= max_items:
            break
        body = (page.get("content") or "").strip()
        if not body:
            continue
        truncated = False
        if len(body) > full_max:
            body = body[:full_max].rstrip() + "…（正文过长已截断）"
            truncated = True
        title_score = float(page.get("title_score", 0.0) or 0.0)
        rel = round(min(1.0, title_score / 8.0), 4)
        detail = {
            "title": round(title_score, 4),
            "selection": "title_snippet_filter",
            "truncated": truncated,
        }
        item = _evidence_from_full_page(
            {**page, "content": body},
            rel,
            detail,
            delivery="title_selected_full",
        )
        evidence.append(item)
        logger.info(
            "[Web] 标题入选页全文注入 url=%s title_score=%.2f chars=%d truncated=%s",
            page.get("url", ""),
            title_score,
            item["content_chars"],
            truncated,
        )
    return evidence


def _build_chunk_evidence(
    pages: list[dict[str, Any]],
    page_indices: list[int],
    user_query: str,
    kb_results: list[dict[str, Any]] | None,
    config: AppConfig,
    search_queries: list[str] | None,
    query_terms: list[str],
    rank_query: str,
    max_items: int,
) -> list[dict[str, Any]]:
    cfg = _web_cfg(config)
    if not cfg or not page_indices:
        return []
    subset = [pages[i] for i in page_indices]
    all_chunks: list[dict[str, Any]] = []
    for local_idx, page in enumerate(subset):
        orig_idx = page_indices[local_idx]
        chunks = chunk_text_spans(
            page.get("content", ""),
            int(getattr(cfg, "chunk_size", 800)),
            int(getattr(cfg, "chunk_overlap", 120)),
        )
        for chunk_idx, span in enumerate(chunks):
            chunk = span["text"]
            page_terms = _keywords(
                " ".join([page.get("title", ""), page.get("search_query", "")]), 8
            )
            all_chunks.append(
                {
                    "page_idx": local_idx,
                    "orig_page_idx": orig_idx,
                    "chunk_idx": chunk_idx,
                    "text": chunk,
                    "start": span["start"],
                    "end": span["end"],
                    "query_score": _term_relevance_score(page_terms, chunk),
                }
            )
    if not all_chunks:
        return []
    chunk_texts = [x["text"] for x in all_chunks]
    embedding_scores = _embedding_scores(rank_query, chunk_texts, config)
    for idx, item in enumerate(all_chunks):
        page = subset[int(item["page_idx"])]
        lexical = _lexical_score(query_terms, item["text"])
        emb = embedding_scores[idx] if embedding_scores is not None else None
        item["embedding_score"] = emb
        item["lexical_score"] = lexical
        base_score = _hybrid_chunk_score(
            embedding_score=emb,
            lexical_score=lexical,
            title_score=float(page.get("title_score", 0.0) or 0.0),
            query_score=float(item.get("query_score", 0.0) or 0.0),
        )
        nav_penalty = min(0.45, _nav_noise_ratio(item["text"]) * 0.6)
        item["score"] = round(base_score * (1.0 - nav_penalty), 4)

    all_chunks.sort(key=lambda x: x["score"], reverse=True)
    per_page_count: dict[int, int] = {}
    evidence: list[dict[str, Any]] = []
    max_per_page = int(getattr(cfg, "max_chunks_per_page", 2))
    for item in all_chunks:
        local_idx = int(item["page_idx"])
        if per_page_count.get(local_idx, 0) >= max_per_page:
            continue
        page = subset[local_idx]
        quote = _best_quote_window(item["text"], query_terms, max_chars=800)
        evidence_content = _context_window(
            page.get("content", ""),
            int(item.get("start", 0) or 0),
            int(item.get("end", 0) or 0),
            query_terms,
            max_chars=2400,
        )
        if not quote:
            continue
        per_page_count[local_idx] = per_page_count.get(local_idx, 0) + 1
        evidence.append(
            {
                "title": page.get("title", ""),
                "url": page.get("url", ""),
                "source": page.get("source", ""),
                "published_at": page.get("published_at", ""),
                "quote": quote,
                "content": evidence_content or quote,
                "extracted_content": evidence_content or quote,
                "content_chars": len(evidence_content or quote),
                "relevance_score": round(float(item["score"]), 4),
                "score_detail": {
                    "embedding": (
                        round(float(item["embedding_score"]), 4)
                        if item.get("embedding_score") is not None
                        else None
                    ),
                    "lexical": round(float(item.get("lexical_score", 0.0) or 0.0), 4),
                    "title": round(float(page.get("title_score", 0.0) or 0.0), 4),
                    "query": round(float(item.get("query_score", 0.0) or 0.0), 4),
                },
                "source_type": "web",
                "delivery": "chunk",
            }
        )
        if len(evidence) >= max_items:
            break
    return evidence


def build_evidence_from_pages(
    user_query: str,
    pages: list[dict[str, Any]],
    kb_results: list[dict[str, Any]] | None,
    config: AppConfig,
    search_queries: list[str] | None = None,
) -> list[dict[str, Any]]:
    cfg = _web_cfg(config)
    if not cfg or not pages:
        return []
    query_terms = _unique_terms(
        _keywords(user_query, 12),
        _kb_terms(kb_results, 8),
        _keywords(" ".join(search_queries or []), 12),
    )
    rank_query = " ".join([user_query, *(search_queries or []), *query_terms])
    mode = (getattr(cfg, "evidence_mode", "title_then_full") or "title_then_full").strip().lower()

    if mode == "chunk":
        evidence = _build_chunk_evidence(
            pages,
            list(range(len(pages))),
            user_query,
            kb_results,
            config,
            search_queries,
            query_terms,
            rank_query,
            int(getattr(cfg, "max_evidence_items", 5)),
        )
    else:
        # title_then_full / auto / full_page：信任标题筛选结果，整页正文送模型
        evidence = _build_title_selected_evidence(pages, config)

    for i, item in enumerate(evidence):
        item["citation_id"] = f"W{i + 1}"
    return _dedupe_evidence_by_url(evidence)


def _strip_model_thinking(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    if "</think>" in t:
        t = t.split("</think>")[-1].strip()
    if "<think>" in t:
        t = re.sub(r"<think>[\s\S]*?</think>", "", t, flags=re.IGNORECASE).strip()
    return t


def _summarize_one_web_page(
    user_query: str,
    item: dict[str, Any],
    config: AppConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """单页由 extract 模型筛选重要原文；无关标【无关】，失败时保留原正文。"""
    from deep_research.prompts import build_web_page_summary_prompt
    from deep_research.utils import send_request_to_model

    cfg = _web_cfg(config)
    title = str(item.get("title") or "")
    url = str(item.get("url") or "")
    raw = (item.get("raw_content") or item.get("content") or "").strip()
    max_in = int(getattr(cfg, "page_summary_max_input_chars", 16000) or 16000)
    max_out = int(getattr(cfg, "page_summary_max_output_chars", 4500) or 4500)
    timeout = int(getattr(cfg, "page_summary_timeout", 120) or 120)
    body_in = raw[:max_in] if len(raw) > max_in else raw
    input_truncated = len(raw) > max_in

    meta: dict[str, Any] = {
        "url": url,
        "title": title,
        "input_chars": len(body_in),
        "input_truncated": input_truncated,
        "ok": False,
    }
    if not body_in or len(body_in) < 80:
        meta["skip_reason"] = "body_too_short"
        return item, meta

    model_cfg = getattr(config, "extractinfo", None) or config.deepresearch

    def _prompt_builder(uq: str, _hist: Any, _refs: Any) -> str:
        return build_web_page_summary_prompt(uq, title, url, body_in)

    t0 = time.perf_counter()
    summary = send_request_to_model(
        user_query=user_query,
        prompt_builder=_prompt_builder,
        model_name=model_cfg.name,
        api_url=model_cfg.server,
        timeout=timeout,
        temperature=float(getattr(model_cfg, "temperature", 0.1) or 0.1),
        max_tokens=min(max(int(getattr(model_cfg, "max_tokens", 4096) or 4096), max_out * 2), 8192),
        top_p=float(getattr(model_cfg, "top_p", 0.9) or 0.9),
        repetition_penalty=float(getattr(model_cfg, "repetition_penalty", 1.1) or 1.1),
    )
    meta["duration_s"] = round(time.perf_counter() - t0, 3)

    if not summary or not str(summary).strip():
        meta["skip_reason"] = "llm_empty"
        return item, meta

    summary = _strip_model_thinking(str(summary)).strip()
    if len(summary) > max_out:
        summary = summary[:max_out].rstrip() + "…（摘要过长已截断）"

    if _page_summary_marked_irrelevant(summary):
        meta["skip_reason"] = "irrelevant_page"
        meta["ok"] = False
        excluded = dict(item)
        excluded["_web_exclude"] = True
        return excluded, meta

    out = dict(item)
    out["raw_content"] = raw
    out["content"] = summary
    out["extracted_content"] = summary
    out["quote"] = summary[: min(2000, len(summary))]
    out["content_chars"] = len(summary)
    prev_delivery = str(out.get("delivery") or "title_selected_full")
    out["delivery"] = (
        "title_selected_summary"
        if "title_selected" in prev_delivery or prev_delivery in ("full_page", "chunk")
        else "page_summary"
    )
    detail = dict(out.get("score_detail") or {})
    detail["page_summary"] = True
    detail["extract_mode"] = "important_original_text"
    detail["summary_input_chars"] = len(body_in)
    detail["summary_output_chars"] = len(summary)
    detail["input_truncated"] = input_truncated
    out["score_detail"] = detail
    meta["ok"] = True
    meta["output_chars"] = len(summary)
    return out, meta


def summarize_web_evidence(
    user_query: str,
    evidence: list[dict[str, Any]],
    config: AppConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """对入选联网证据逐页调用 extract 模型：判重要、摘录原文、去无关并控篇幅。"""
    cfg = _web_cfg(config)
    if not evidence or not bool(getattr(cfg, "page_summary_enabled", False)):
        return evidence, {"enabled": False, "count": 0}

    parallel = max(1, int(getattr(cfg, "page_summary_parallel", 3) or 3))
    results: list[dict[str, Any] | None] = [None] * len(evidence)
    page_meta: list[dict[str, Any]] = []

    def _task(idx_item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any], dict[str, Any]]:
        idx, it = idx_item
        if "raw_content" not in it and (it.get("content") or ""):
            it = {**it, "raw_content": it.get("content")}
        updated, meta = _summarize_one_web_page(user_query, it, config)
        meta["id"] = it.get("citation_id") or f"W{idx + 1}"
        return idx, updated, meta

    with ThreadPoolExecutor(max_workers=min(parallel, len(evidence))) as pool:
        futures = [pool.submit(_task, (i, dict(ev))) for i, ev in enumerate(evidence)]
        for fut in as_completed(futures):
            try:
                idx, updated, meta = fut.result()
                results[idx] = updated
                page_meta.append(meta)
            except Exception as e:
                logger.exception("[Web] 网页摘要任务失败: %s", e)
                page_meta.append({"ok": False, "error": str(e)})

    summarized: list[dict[str, Any]] = []
    dropped_weak = 0
    for i, r in enumerate(results):
        item = r if r is not None else evidence[i]
        meta_i = page_meta[i] if i < len(page_meta) else {}
        if item.get("_web_exclude"):
            dropped_weak += 1
            logger.info(
                "[Web] 模型判定网页无关，丢弃 url=%s title=%s",
                item.get("url", ""),
                (item.get("title") or "")[:50],
            )
            continue
        summarized.append(item)
    if dropped_weak:
        for idx, item in enumerate(summarized, 1):
            item["citation_id"] = f"W{idx}"
    ok_count = sum(1 for m in page_meta if m.get("ok"))
    trace = {
        "enabled": True,
        "parallel": parallel,
        "page_count": len(evidence),
        "success_count": ok_count,
        "pages": sorted(page_meta, key=lambda x: str(x.get("id", ""))),
    }
    logger.info(
        "[Web] 网页摘要完成 success=%d/%d parallel=%d",
        ok_count,
        len(evidence),
        parallel,
    )
    return summarized, trace


def retrieve_web_evidence(
    user_query: str,
    kb_results: list[dict[str, Any]] | None,
    config: AppConfig,
    *,
    planned_search_queries: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not web_enabled(config):
        return [], {"enabled": False}
    uq = effective_user_question_for_web(user_query)
    queries = build_search_queries(uq, config, planned_search_queries)
    all_candidates, search_meta = collect_search_candidates(queries, config)
    filtered, rejects = filter_candidates(all_candidates, uq, kb_results, config)
    pages: list[dict[str, Any]] = []
    fetch_failures: list[dict[str, str]] = []
    for candidate in filtered:
        page = fetch_webpage(candidate, config, fetch_failures)
        if page:
            pages.append(page)
    evidence = build_evidence_from_pages(uq, pages, kb_results, config, queries)
    summary_trace: dict[str, Any] = {"enabled": False}
    if evidence and bool(getattr(_web_cfg(config), "page_summary_enabled", False)):
        evidence, summary_trace = summarize_web_evidence(uq, evidence, config)
    trace = {
        "enabled": True,
        "queries": queries,
        "query_source": "retrieval_plan",
        "planned_queries_in": list(planned_search_queries),
        "enforce_trusted_domains": bool(getattr(_web_cfg(config), "enforce_trusted_domains", False)),
        "page_selection": "title_snippet_score",
        "evidence_mode": (getattr(_web_cfg(config), "evidence_mode", "title_then_full") or "title_then_full"),
        "page_summary": summary_trace,
        "relevance_scoring": (
            "title_filter_then_page_summary"
            if summary_trace.get("enabled")
            else "title_filter_then_full_page_body"
        ),
        "search_meta": search_meta,
        "candidate_count": len(all_candidates),
        "filtered_count": len(filtered),
        "fetched_count": len(pages),
        "fetch_failures": fetch_failures[:20],
        "evidence_count": len(evidence),
        "filtered_urls": [
            {
                "title": c.title,
                "url": c.url,
                "score": round(c.title_score, 3),
                "query": c.search_query,
            }
            for c in filtered
        ],
        "rejects": rejects[:20],
        "evidence_preview": [
            {
                "id": e.get("citation_id"),
                "title": e.get("title"),
                "url": e.get("url"),
                "score": e.get("relevance_score"),
                "score_detail": e.get("score_detail"),
                "content_chars": e.get("content_chars"),
                "delivery": e.get("delivery"),
                "quote": (e.get("quote") or "")[:800],
                "summary": bool((e.get("score_detail") or {}).get("page_summary")),
            }
            for e in evidence
        ],
    }
    return evidence, trace
