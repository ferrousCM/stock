"""다크 모드(바이낸스 스타일) 공통 색상·CSS — 모니터링/모의투자 페이지가 공유한다.

**색상 관행 변경**: 기존에는 국내 관행(상승=빨강/하락=파랑)을 썼지만, 다크 모드 전환과 함께
바이낸스 등 글로벌 거래소 다크 UI에서 쓰는 상승=초록/하락=빨강 배색으로 바꿨다. 대시보드
전체(차트·표·텍스트)에서 이 파일의 색상만 참조하도록 통일한다.

`app.py`는 별도로 공유 중인 배포라 이 모듈을 쓰지 않고 기존 색상을 그대로 유지한다
(CLAUDE.md "app.py는 더 이상 손대지 않는다" 원칙).
"""

from __future__ import annotations

import streamlit as st

# 실측 시세용 (원색) — 바이낸스 다크모드 배색: 상승=초록, 하락=빨강, 보합=회색
UP_COLOR, DOWN_COLOR, FLAT_COLOR = "#0ecb81", "#f6465d", "#848e9c"
# 예측값(사실이 아니라 추정치)용 — 위 원색과 색상(H)은 같고 채도만 낮춘 파스텔 톤
UP_SOFT, DOWN_SOFT, FLAT_SOFT = "#5f9c82", "#c9727c", "#9199a1"

# 모의투자 봇 비교색 — 서로 튀지 않도록 파랑→인디고→보라 한 계열(유사 톤)로 통일
BOT_COLORS = {"default": "#60a5fa", "aggressive": "#818cf8", "momentum": "#c084fc"}
KOSPI_COLOR = "#f59e0b"  # 벤치마크(코스피) 라인 — 봇 색과 구분되도록 의도적으로 다른 톤

_SIDEBAR_WIDTH_PX = 235  # 기존 기본 사이드바 너비(약 336px)의 0.7배


def inject_base_css() -> None:
    """컴팩트 레이아웃·사이드바 너비·가독성용 전역 CSS. 여러 페이지에서 반복 호출해도
    안전하다(같은 규칙이 중복 삽입될 뿐 부작용 없음) — 페이지 진입 순서와 무관하게
    항상 같은 모양을 보장하기 위해 각 페이지 진입점에서 매번 호출한다."""
    st.markdown(
        f"""
<style>
.block-container {{padding-top: 1rem; padding-bottom: 1rem;}}
h1, h2, h3, h4, h5 {{margin-top: 0.1rem; margin-bottom: 0.3rem;}}
div[data-testid="stVerticalBlock"] {{gap: 0.45rem;}}
[data-testid="stMetricValue"] {{font-size: 1.25rem;}}
[data-testid="stMetricLabel"] {{font-size: 0.88rem;}}
[data-testid="stMetricDelta"] {{font-size: 0.88rem;}}
.stTabs [data-baseweb="tab-list"] {{gap: 4px;}}
.stTabs [data-baseweb="tab"] {{padding: 4px 10px; font-size: 0.92rem;}}
div[data-testid="stWidgetLabel"] p {{font-size: 0.88rem; margin-bottom: 0.1rem;}}
.stButton button {{padding: 0.25rem 0.7rem; font-size: 0.88rem;}}
hr {{margin: 0.5rem 0;}}

/* 가독성: 한글은 줄간격이 좁으면 위아래 글자가 붙어 보이므로 최소 줄간격을 보장한다 */
p, .stCaption, .stMarkdown, label, span, div, li {{font-size: 0.92rem; line-height: 1.55;}}
[data-testid="stMarkdownContainer"] p {{line-height: 1.55;}}
[data-testid="stExpander"] summary p {{font-size: 0.95rem; line-height: 1.5;}}
[data-testid="stAlert"] p {{line-height: 1.6;}}

/* 사이드 내비게이션 너비 — 기존 기본값의 0.7배 */
section[data-testid="stSidebar"] {{width: {_SIDEBAR_WIDTH_PX}px !important; min-width: {_SIDEBAR_WIDTH_PX}px !important;}}
section[data-testid="stSidebar"] > div {{width: {_SIDEBAR_WIDTH_PX}px !important;}}
</style>
""",
        unsafe_allow_html=True,
    )
