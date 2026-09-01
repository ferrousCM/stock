"""코인 뉴스 — 네이버 뉴스 키워드 검색 스크래핑.

news.py(주식)는 finance.naver.com의 종목코드 기반 뉴스 목록 페이지를 쓰는데, 네이버에는
코인 종목코드 기반 뉴스가 없다(m.stock.naver.com/crypto는 API 호출이 JS 번들 안에 숨어있는
SPA라 안정적으로 스크래핑하기 어렵다). 대신 news.naver.com 키워드 검색으로 코인 한글명을
검색해 목록을 만든다.

기사 원문 조회·요약·감성 판정·감성 히스토리 기록은 news.py 함수를 그대로 재사용한다
(fetch_article_body/summarize/sentiment.score/log_sentiment_from_news/sentiment_history) —
원문이 결국 전부 n.news.naver.com이라 이 뒷단은 코인이든 주식이든 완전히 동일하다. 감성
히스토리는 종목코드 대신 업비트 마켓코드('KRW-BTC' 등)를 키로 써서
data/raw/news_sentiment/KRW-BTC.parquet처럼 기록된다 — 코드 형식이 달라(6자리 숫자 vs
'KRW-' 접두사) 기존 주식 로그와 절대 겹치지 않는다.

**알려진 제약** (실측, 2026-08-14): 종목코드 기반이 아니라 키워드 검색이라 2음절처럼 짧고
모호한 코인명은 검색 결과가 0건일 수 있다 — '리플'은 0건, '리플 코인'은 5건이었다. 호출부가
빈 결과를 정상적으로 처리해야 한다(news.py의 finance.naver.com 방식과 달리 종목코드 자체가
틀렸을 때와 "검색 결과가 우연히 없을 때"를 구분할 수 없다).

또한 이 검색 결과 페이지는 finance.naver.com보다 훨씬 자주 마크업이 바뀌는 최신 컴포넌트
기반 구조(클래스명이 빌드마다 해시로 바뀌는 것으로 보임)라, 클래스명이 아니라 "네이버뉴스
배지 링크를 기준으로 같은 카드 안에서 외부 언론사로 연결되는 첫 링크를 제목으로 본다"는
구조적 휴리스틱을 쓴다 — news.py보다 셀렉터가 깨질 가능성이 더 높다.
"""

from __future__ import annotations

import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

from . import news
from .cache_utils import cache_path, is_fresh
from .config import NEWS_CACHE_TTL_SEC

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
_SEARCH_URL = "https://search.naver.com/search.naver"
_ARTICLE_URL = "https://n.news.naver.com/mnews/article/{office_id}/{article_id}"
_ARTICLE_HREF_RE = re.compile(r"article/(\d+)/(\d+)")

__all__ = ["fetch_news_list", "fetch_news_with_sentiment"]


def _extract_meta(badge) -> tuple[str | None, str, str]:
    """'네이버뉴스' 배지 링크의 조상을 거슬러 올라가며 헤드라인·언론사·상대시각을 찾는다.

    배지 링크 자신은 "네이버뉴스" 라벨 텍스트만 가지고 있을 뿐 실제 헤드라인이 아니다 —
    헤드라인은 같은 카드 안의 원문 언론사 링크(외부 도메인)에 달려 있다.
    """
    title = None
    press = ""
    date = ""
    node = badge
    for _ in range(6):
        node = node.find_parent()
        if node is None:
            break
        if title is None:
            for a in node.select("a[href]"):
                href = a.get("href", "")
                text = a.get_text(strip=True)
                if href.startswith("http") and "naver.com" not in href and text:
                    title = text.replace("새 창 열림", "").strip()
                    break
        if not press:
            p = node.select_one('a[href*="media.naver.com/press/"]')
            if p:
                t = p.get_text(strip=True).replace("새 창 열림", "").strip()
                if t:
                    press = t
        if not date:
            classes = node.get("class") or []
            if any("subtexts" in c for c in classes):
                text = node.get_text(" ", strip=True)
                date = text.replace("네이버뉴스", "").replace("새 창 열림", "").strip()
        if title and press and date:
            break
    return title, press, date


def fetch_news_list(keyword: str, n: int = 10, use_cache: bool = True) -> pd.DataFrame:
    """코인명(또는 임의 키워드)으로 네이버 뉴스를 검색해 목록을 만든다.

    news.fetch_news_list()(종목코드 기반)의 코인용 대응 함수 — 반환 컬럼(article_id,
    office_id, title, press, date, url)을 동일하게 맞춰서 news.fetch_article_body() 등
    뒷단 파이프라인을 그대로 재사용할 수 있게 했다.
    """
    path = cache_path("crypto_news_list", f"{keyword}|{n}")
    if use_cache and is_fresh(path, ttl_sec=NEWS_CACHE_TTL_SEC):
        return pd.read_parquet(path)

    r = requests.get(_SEARCH_URL, params={"where": "news", "query": keyword}, headers=_HEADERS, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")

    rows = []
    seen: set[str] = set()
    for badge in soup.select('a[href*="n.news.naver.com/mnews/article"]'):
        m = _ARTICLE_HREF_RE.search(badge.get("href", ""))
        if not m:
            continue
        office_id, article_id = m.group(1), m.group(2)
        if article_id in seen:
            continue
        title, press, date = _extract_meta(badge)
        if not title:
            continue
        seen.add(article_id)
        rows.append(
            {
                "article_id": article_id,
                "office_id": office_id,
                "title": title,
                "press": press,
                "date": date,
                "url": _ARTICLE_URL.format(office_id=office_id, article_id=article_id),
            }
        )
        if len(rows) >= n:
            break

    df = pd.DataFrame(rows)
    if use_cache:
        df.to_parquet(path)
    return df


def fetch_news_with_sentiment(keyword: str, n: int = 10, use_cache: bool = True) -> pd.DataFrame:
    """뉴스 목록 + 본문 요약 + 상승지표까지 채운 DataFrame.

    news.fetch_news_with_sentiment()의 코인용 대응 함수 — 원문 조회·요약·감성 판정은
    news.py의 함수를 그대로 쓴다(원문이 결국 n.news.naver.com이라 완전히 동일하게 동작).
    """
    listing = fetch_news_list(keyword, n=n, use_cache=use_cache)
    return news.enrich_with_sentiment(listing, use_cache=use_cache)
