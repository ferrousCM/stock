"""네이버 종목 뉴스 조회.

종목별 뉴스 목록을 네이버 증권 모바일 JSON API에서 가져오고, 각 기사 원문
(n.news.naver.com)을 받아 요약 + 감성 점수를 매긴다.

**2026-10-10 소스 교체**: 원래는 `finance.naver.com/item/news_news.naver` HTML을
스크래핑했는데, 네이버 금융이 "Npay 증권"으로 개편되면서 이 엔드포인트가 **HTTP 410
Gone**으로 폐기됐다(응답은 빈 `table.type5` 껍데기라 파싱 오류 없이 0건이 나왔다 —
그래서 5주 넘게 조용히 깨진 채 방치됐고, 그 기간 국내 종목 감성 히스토리가 비었다).
대체 소스는 모바일 웹이 실제로 쓰는 공개 JSON API다:

    https://m.stock.naver.com/api/news/stock/{종목코드}?pageSize=20&page=1

응답은 `[{total, items:[{officeId, articleId, officeName, datetime, title,
titleFull, body, mobileNewsUrl, ...}]}, ...]` 꼴의 **클러스터 배열**이다(묶음 기사의
대표 1건만 items에 담겨 오고, total은 그 묶음 크기). 기존 HTML 스크래핑보다 오히려
안정적이다 — euc-kr 인코딩 처리가 필요 없고(UTF-8 JSON), Referer 헤더 없이도 되고,
officeId/articleId가 그대로 있어 기사 원문 URL을 지금 방식대로 조립할 수 있다.

**실패하면 조용히 빈 결과를 주지 말고 예외를 낸다**(`raise_for_status`). 이번 고장이
오래 안 보였던 원인이 "스크래핑 실패 = 빈 DataFrame"이었기 때문이다. 대신 매매
파이프라인(`scripts/run_daily_trading.py`)은 뉴스 조회 실패를 감성 중립(0)으로 흡수해
매매 자체는 계속한다 — 화면에서는 오류가 보이고, 봇은 멈추지 않는 쪽이 맞다.

주의: 비공식 API다. 네이버가 응답 구조를 바꾸면 깨질 수 있다.
"""

from __future__ import annotations

import html as html_lib
import math
import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from . import sentiment
from .cache_utils import cache_path, is_fresh
from .config import NEWS_CACHE_TTL_SEC, RAW_DIR

_SENTIMENT_LOG_DIR = RAW_DIR / "news_sentiment"

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
_LIST_URL = "https://m.stock.naver.com/api/news/stock/{code}"
_ARTICLE_URL = "https://n.news.naver.com/mnews/article/{office_id}/{article_id}"
_ITEMS_PER_LIST_PAGE = 20  # JSON API의 pageSize — 페이지당 기사 수(실측: 중복 없이 20건씩)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?다요음]\s)|(?<=[.!?다요음]$)")


def _list_referer(code: str) -> str:
    """API를 실제로 호출하는 모바일 뉴스 페이지. 없어도 응답하지만 맞춰 보낸다."""
    return f"https://m.stock.naver.com/domestic/stock/{code}/news"


def _format_datetime(raw: str) -> str:
    """API의 'YYYYMMDDHHMM' → 'YYYY.MM.DD HH:MM'(기존 HTML 목록과 같은 표기)."""
    s = str(raw)
    if len(s) < 12 or not s.isdigit():
        return s
    return f"{s[0:4]}.{s[4:6]}.{s[6:8]} {s[8:10]}:{s[10:12]}"


def fetch_news_list(code: str, pages: int = 1, use_cache: bool = True) -> pd.DataFrame:
    """종목 뉴스 목록 (제목/언론사/일시/기사ID). 중복 게재 기사는 제거한다.

    pages는 요청할 페이지 수(페이지당 `_ITEMS_PER_LIST_PAGE`건) — 요청 수가 pages에
    비례하도록 고정돼 있고 기사 본문은 받지 않는다.
    """
    path = cache_path("news_list", f"{code}|{pages}")
    if use_cache and is_fresh(path, ttl_sec=NEWS_CACHE_TTL_SEC):
        return pd.read_parquet(path)

    headers = {**_HEADERS, "Referer": _list_referer(code)}
    rows = []
    for page in range(1, pages + 1):
        r = requests.get(
            _LIST_URL.format(code=code),
            headers=headers,
            params={"pageSize": _ITEMS_PER_LIST_PAGE, "page": page},
            timeout=10,
        )
        r.raise_for_status()  # 엔드포인트가 또 폐기되면 빈 결과가 아니라 예외로 드러나게
        clusters = r.json()
        if not isinstance(clusters, list):
            raise RuntimeError(f"뉴스 API 응답 형식이 바뀌었습니다 (list 아님: {type(clusters).__name__})")

        for cluster in clusters:
            for item in cluster.get("items", []):
                office_id, article_id = str(item.get("officeId", "")), str(item.get("articleId", ""))
                if not office_id or not article_id:
                    continue
                title = item.get("titleFull") or item.get("title") or ""
                rows.append(
                    {
                        "article_id": article_id,
                        "office_id": office_id,
                        # API가 &quot; 같은 HTML 엔티티를 그대로 주는 경우가 있어 풀어준다
                        "title": html_lib.unescape(title).strip(),
                        "press": html_lib.unescape(str(item.get("officeName", ""))).strip(),
                        "date": _format_datetime(item.get("datetime", "")),
                        "url": _ARTICLE_URL.format(office_id=office_id, article_id=article_id),
                    }
                )

    # article_id는 언론사별 연번이라 단독으로는 서로 다른 기사끼리 충돌할 수 있다 —
    # (office_id, article_id) 조합으로 중복을 제거한다.
    df = (
        pd.DataFrame(rows, columns=["article_id", "office_id", "title", "press", "date", "url"])
        .drop_duplicates(subset=["office_id", "article_id"])
        .reset_index(drop=True)
    )
    # 빈 결과는 캐시하지 않는다. 소스가 고장 나 0건이 돌아온 시점의 캐시가 남아 있으면
    # 고친 뒤에도 TTL이 끝날 때까지 계속 빈 값이 나온다 — 2026-10-10 엔드포인트 교체
    # 직후 실제로 겪었다. 뉴스가 정말 없는 종목은 드물어 재조회 비용도 무시할 만하다.
    if use_cache and not df.empty:
        df.to_parquet(path)
    return df


def fetch_article(office_id: str, article_id: str, use_cache: bool = True) -> dict:
    """네이버 뉴스 원문의 제목·본문. 요청 1회로 둘 다 받아 같은 캐시에 담는다.

    제목까지 받는 이유: 목록 API(`fetch_news_list`)가 긴 제목을 45자 근처에서
    `...`로 잘라서 준다. 본문은 어차피 받아야 하므로(요약·감성) 같은 응답에서 전체
    제목을 함께 꺼내면 추가 네트워크 비용 없이 잘린 제목을 복원할 수 있다.
    """
    path = cache_path("news_body", f"{office_id}|{article_id}")
    if use_cache and is_fresh(path, ttl_sec=24 * 3600):  # 원문은 안 바뀌므로 하루 캐시
        cached = pd.read_parquet(path)
        # title 컬럼은 나중에 추가된 것 — 그 전에 만들어진 캐시에는 없다
        title = str(cached["title"].iloc[0]) if "title" in cached.columns else ""
        return {"title": title, "body": str(cached["body"].iloc[0])}

    url = _ARTICLE_URL.format(office_id=office_id, article_id=article_id)
    r = requests.get(url, headers=_HEADERS, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    body_el = soup.select_one("#dic_area")
    body = body_el.get_text(" ", strip=True) if body_el else ""
    title_el = soup.select_one("#title_area")
    title = title_el.get_text(" ", strip=True) if title_el else ""

    if use_cache:
        pd.DataFrame({"body": [body], "title": [title]}).to_parquet(path)
    return {"title": title, "body": body}


def fetch_article_body(office_id: str, article_id: str, use_cache: bool = True) -> str:
    """네이버 뉴스 원문 본문 텍스트."""
    return fetch_article(office_id, article_id, use_cache=use_cache)["body"]


def summarize(body: str, max_sentences: int = 2, max_chars: int = 200) -> str:
    """본문 앞부분 몇 문장을 요약처럼 보여준다 (LLM 요약 아님, 리드 문단 발췌)."""
    if not body:
        return ""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(body) if s.strip()]
    summary = " ".join(sentences[:max_sentences])
    if len(summary) > max_chars:
        summary = summary[:max_chars].rstrip() + "…"
    return summary


def fetch_news_list_n(code: str, n: int = 100, use_cache: bool = True) -> pd.DataFrame:
    """최근 뉴스 목록을 최대 n개까지. 네이버 목록은 페이지당 ~20개라 필요한 만큼 페이지를 넘긴다.

    요청 수는 (n / 20)개로 고정 — 기사 본문은 받지 않으므로 n이 커도 가볍다
    (본문·감성은 화면에 실제로 보여줄 페이지에 대해서만 enrich_with_sentiment로 채운다).
    """
    pages = max(1, math.ceil(n / _ITEMS_PER_LIST_PAGE))
    return fetch_news_list(code, pages=pages, use_cache=use_cache).head(n).reset_index(drop=True)


def enrich_with_sentiment(listing: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    """뉴스 목록 DataFrame에 summary/sentiment_label/sentiment_score 컬럼을 채운다.

    주식(news.py)·코인(crypto_news.py) 공용 — 원문이 결국 전부 n.news.naver.com이라
    본문 조회·요약·감성 판정이 동일하다. 본문을 행 수만큼 개별 요청하므로 넘기는 목록이
    크면 느리다(기사 본문은 24시간 캐시라 한 번 채우면 이후는 빠르다).
    빈/None DataFrame이면 그대로 돌려준다.
    """
    if listing is None or listing.empty:
        return listing

    listing = listing.copy()
    titles, summaries, labels, scores = [], [], [], []
    for _, row in listing.iterrows():
        article = fetch_article(row["office_id"], row["article_id"], use_cache=use_cache)
        body = article["body"]
        # 목록 API가 잘라서 준 제목(…/...로 끝남)은 원문 제목으로 되살린다
        title = article["title"] or row["title"]
        titles.append(title)
        summaries.append(summarize(body))
        result = sentiment.score(f"{title} {body}")
        labels.append(result["label"])
        scores.append(result["score"])

    listing["title"] = titles
    listing["summary"] = summaries
    listing["sentiment_label"] = labels
    listing["sentiment_score"] = scores
    return listing


def fetch_news_with_sentiment(code: str, n: int = 10, use_cache: bool = True) -> pd.DataFrame:
    """뉴스 목록 + 본문 요약 + 상승지표(긍정/중립/부정)까지 채운 DataFrame.

    본문을 기사 수만큼 개별 요청하므로 캐시가 없으면 n개에 비례해 느려진다.
    """
    listing = fetch_news_list(code, use_cache=use_cache).head(n).copy()
    return enrich_with_sentiment(listing, use_cache=use_cache)


def _sentiment_log_path(code: str) -> Path:
    return _SENTIMENT_LOG_DIR / f"{code}.parquet"


def _values_match(stored, incoming) -> bool:
    """기록된 값과 새 값이 같은지. 양쪽 다 결측이면 같은 것으로 본다."""
    stored_na, incoming_na = pd.isna(stored), pd.isna(incoming)
    if stored_na or incoming_na:
        return bool(stored_na and incoming_na)
    return math.isclose(float(stored), float(incoming), rel_tol=1e-12, abs_tol=1e-12)


def _already_logged(
    existing: pd.DataFrame,
    date: pd.Timestamp,
    avg_score: float,
    positive_count: int | None,
    negative_count: int | None,
    article_count: int | None,
) -> bool:
    """해당 날짜에 완전히 같은 값이 이미 기록돼 있으면 True (재작성 생략용)."""
    match = existing[existing["date"] == date]
    if len(match) != 1:
        return False
    stored = match.iloc[0]
    return all(
        _values_match(stored.get(col), value)
        for col, value in (
            ("avg_sentiment", avg_score),
            ("positive_count", positive_count),
            ("negative_count", negative_count),
            ("article_count", article_count),
        )
    )


def log_daily_sentiment(
    code: str,
    avg_score: float,
    positive_count: int | None = None,
    negative_count: int | None = None,
    article_count: int | None = None,
    date: pd.Timestamp | None = None,
) -> None:
    """오늘자 감성 통계를 종목별 히스토리에 누적 기록한다 (같은 날짜는 최신 값으로 덮어씀).

    predictor.py가 뉴스 감성을 학습 피처로 쓰려면 '그날의 감성 점수'가 날짜별로
    쌓여 있어야 한다. 하지만 뉴스 소스(finance.naver.com)는 최신 기사 목록만
    보여줄 뿐 과거 특정 날짜의 감성 히스토리를 조회하는 API가 아니다. 그래서
    앱을 실행할 때마다(뉴스를 조회할 때마다) 오늘 날짜의 통계를 여기 기록해
    시간이 지날수록 실제 히스토리가 쌓이도록 한다. 기록을 시작하기 전 과거
    구간은 예측 모델에서 중립(0)으로 채워진다.

    positive_count/negative_count/article_count는 predictor.py의 심층 모델
    (train_and_predict_advanced)이 쓰는 추가 피처다 — 기본 모델은 avg_sentiment만 쓴다.

    이미 같은 날짜에 같은 값이 기록돼 있으면 파일을 아예 건드리지 않는다. parquet은
    같은 내용을 다시 써도 바이트가 달라져서, 그냥 덮어쓰면 앱을 켤 때마다 이 파일이
    변경된 것으로 잡힌다 (이 디렉토리는 git으로 추적하므로 무의미한 diff가 쌓인다).
    """
    date = (date or pd.Timestamp.today()).normalize()
    path = _sentiment_log_path(code)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = pd.DataFrame(
        {
            "date": [date],
            "avg_sentiment": [float(avg_score)],
            "positive_count": [positive_count],
            "negative_count": [negative_count],
            "article_count": [article_count],
        }
    )
    if path.exists():
        existing = pd.read_parquet(path)
        if _already_logged(existing, date, avg_score, positive_count, negative_count, article_count):
            return
        existing = existing[existing["date"] != date]
        combined = pd.concat([existing, row], ignore_index=True).sort_values("date")
    else:
        combined = row
    combined.to_parquet(path)


def sentiment_history(code: str) -> pd.Series:
    """날짜 인덱스의 일별 평균 감성 점수 시계열. 기록이 없으면 빈 Series. (기본 예측 모델용)"""
    path = _sentiment_log_path(code)
    if not path.exists():
        return pd.Series(dtype=float, name="avg_sentiment")
    df = pd.read_parquet(path)
    return df.set_index("date")["avg_sentiment"].rename("avg_sentiment")


def sentiment_history_full(code: str) -> pd.DataFrame:
    """날짜 인덱스의 감성 통계 전체(avg_sentiment/positive_count/negative_count/article_count).

    기록이 없으면 빈 DataFrame. ("정확한 예측" 심층 모델용 — 기본 모델은 sentiment_history() 사용)
    이 컬럼들이 추가되기 전에 기록된 옛 로그 파일에는 positive_count 등이 없을 수 있어
    없으면 NaN으로 채운다 (predictor.py에서 다시 0으로 채워짐).

    카운트가 전부 None인 옛 파일은 parquet에서 object dtype으로 돌아와 그대로 두면
    학습 피처가 object로 흘러간다. 여기서 숫자로 강제 변환해 항상 float으로 넘긴다.
    """
    path = _sentiment_log_path(code)
    cols = ["avg_sentiment", "positive_count", "negative_count", "article_count"]
    if not path.exists():
        return pd.DataFrame(columns=cols, dtype=float)
    df = pd.read_parquet(path).set_index("date")
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else float("nan")
    return df[cols]


def log_sentiment_from_news(code: str, news_df: pd.DataFrame, date: pd.Timestamp | None = None) -> None:
    """뉴스 DataFrame에서 그날의 감성 통계를 집계해 기록한다.

    fetch_news_with_sentiment()의 결과를 그대로 넘기면 된다. 평균 점수뿐 아니라
    긍정/부정 기사 수와 총 기사 수까지 채워야 심층 예측 모델의 뉴스 피처가 실제로
    값을 갖는다 (평균만 기록하면 그 피처들이 항상 0이라 무의미해진다).
    빈 DataFrame이면 아무것도 하지 않는다.
    """
    if news_df is None or news_df.empty:
        return
    labels = news_df.get("sentiment_label")
    log_daily_sentiment(
        code,
        float(news_df["sentiment_score"].mean()),
        positive_count=int((labels == "긍정").sum()) if labels is not None else None,
        negative_count=int((labels == "부정").sum()) if labels is not None else None,
        article_count=int(len(news_df)),
        date=date,
    )
