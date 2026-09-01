"""프로젝트 전역 설정 — 경로, 기본값, 자주 쓰는 티커."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# src/config.py 기준 한 단계 위가 프로젝트 루트
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")  # 있으면 로드, 없으면 조용히 무시

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "output"
PORTFOLIO_DIR = DATA_DIR / "portfolio"  # 모의투자 원장 (git으로 추적 — data/cache·raw와 다름)

for _d in (RAW_DIR, CACHE_DIR, OUTPUT_DIR, PORTFOLIO_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def portfolio_dir_for(bot_id: str) -> Path:
    """봇 버전별 원장 디렉터리. "default"는 기존 경로(data/portfolio/)를 그대로 쓴다 —
    이미 매일 자동 실행되며 커밋되는 라이브 원장이라 마이그레이션하지 않는다. 그 외 봇은
    data/portfolio/{bot_id}/ 아래 완전히 새 원장에서 시작한다."""
    return PORTFOLIO_DIR if bot_id == "default" else PORTFOLIO_DIR / bot_id


def portfolio_dir_for_v2(bot_id: str) -> Path:
    """v2 봇(PRD.md 11장) 원장 디렉터리. v1의 `portfolio_dir_for()`와 달리 "default"도
    특별 취급하지 않는다 — v2의 세 봇 전부 data/portfolio/v2/{bot_id}/ 아래 완전히 새
    원장(1억원)으로 시작한다. v2 "기본형"이 v1의 라이브 원장(data/portfolio/)을 실수로
    재사용하는 사고를 막으려고 함수 자체를 분리했다(PRD 11.2)."""
    return PORTFOLIO_DIR / "v2" / bot_id


# 데이터 조회 기본 시작일 (지정 안 하면 여기서부터)
DEFAULT_START = "2015-01-01"

# 캐시 만료 시간(초). 장중 반복 조회를 막되 당일 데이터는 갱신되도록 6시간.
CACHE_TTL_SEC = 6 * 60 * 60

# 뉴스/공시는 가격보다 자주 바뀌므로 캐시를 짧게 둔다 (30분).
NEWS_CACHE_TTL_SEC = 30 * 60

# DART(전자공시시스템) OpenAPI 키. https://opendart.fss.or.kr 에서 무료 발급.
# 로컬: 프로젝트 루트에 .env 파일을 만들어 DART_API_KEY=... 로 설정 (.env는 gitignore됨).
# Streamlit Cloud: 앱 설정 Secrets에 DART_API_KEY = "..." 를 섹션 없이 최상위에 추가.
# 아래 상수는 import 시점 스냅샷일 뿐 — 런타임 조회는 반드시 get_dart_api_key()를 쓴다.
DART_API_KEY = os.environ.get("DART_API_KEY", "")


def get_dart_api_key() -> str:
    """DART OpenAPI 키를 런타임에 조회한다.

    우선순위: (1) 환경변수 DART_API_KEY(로컬 .env 포함), (2) Streamlit secrets.

    모듈 상수(DART_API_KEY)로 고정하지 않는 이유: Streamlit Cloud에 Secrets로 넣은
    키가 os.environ에 안 실리는 경우가 있다 — 섹션([xxx]) 안에 넣었거나, secrets 지연
    로딩이 config import보다 늦은 경우. import 시점에 값을 굳히면 이후 secrets가 로드돼도
    빈 문자열로 남는다. 그래서 매 호출마다 os.environ과 st.secrets를 다시 확인한다.
    """
    key = os.environ.get("DART_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st  # 대시보드에서만 존재(노트북/GitHub Actions엔 없음)

        # secrets 파일이 아예 없으면 접근 시 예외가 나므로 통째로 감싼다.
        return str(st.secrets.get("DART_API_KEY", "") or "").strip()
    except Exception:
        return ""


# 자주 쓰는 지수 심볼 (FinanceDataReader 표기)
INDICES = {
    "KOSPI": "KS11",
    "KOSDAQ": "KQ11",
    "KOSPI200": "KS200",
    "NASDAQ": "IXIC",
    "S&P500": "US500",
    "DOW": "DJI",
    "NIKKEI225": "N225",
    "VIX": "VIX",
}

# 자주 쓰는 환율/원자재
MACRO = {
    "USD/KRW": "USD/KRW",
    "US10YT": "US10YT=X",
    "WTI": "CL=F",
    "GOLD": "GC=F",
}

# 연간 거래일 수 — 연율화(annualization)에 사용
TRADING_DAYS = 252
