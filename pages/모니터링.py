"""모니터링 — 국내증시/코인 스크리닝·차트·예측·뉴스. app.py의 확장판.

app.py는 지금 지인에게 공유 중인 배포라 한 글자도 수정하지 않는다(CLAUDE.md 원칙). 이
페이지는 demo_app.py 전용 새 페이지로, 좌상단 시장 선택(국내증시/코인)에 따라 화면 내용을
다르게 그린다. 국내증시 모드는 app.py와 동일한 기능이고, 코인 모드는 src/crypto_*.py
(업비트 공개 API + 네이버 뉴스 검색)를 쓴다.

indicators.py/charts.py/predictor.py는 OHLCV 컬럼 계약만 맞으면 시장과 무관하게 그대로
동작하는 걸 이미 실측 확인했다 — 그래서 이 세 모듈은 분기가 없다. 분기가 필요한 곳은
유니버스 조회(screener vs crypto_screener vs us_screener), 종목 검색(data_loader vs
crypto_loader — 해외증시는 data_loader.find_symbol(market="NASDAQ")로 그대로 재사용),
뉴스(news vs crypto_news — 코인·해외증시 둘 다 종목코드 기반 뉴스가 없어 종목명/코인명
키워드 검색으로 crypto_news를 공유 재사용한다), 공시(DART는 코인·해외증시 둘 다 없어 탭
자체를 숨긴다), 그리고 코인·해외증시는 주간 등락률 데이터가 없어 해당 탭/옵션을 뺀다
(해외증시는 나스닥 스크리너 API가 시가총액은 주므로 그 탭은 코인과 달리 유지한다).
"""

from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import (
    charts,
    config,
    crypto_loader,
    crypto_news,
    crypto_screener,
    dart,
    news,
    predictor,
    resizable_chart,
    screener,
    theme,
    us_screener,
)
from src import data_loader as dl
from src import indicators as ind

DATE_RANGES = {"1개월": 30, "3개월": 90, "6개월": 180, "1년": 365, "2년": 730, "3년": 1095, "전체": None}
SMA_CHOICES = (5, 20, 60, 120)

# 바이낸스 다크모드 배색(상승=초록/하락=빨강) — src/theme.py에서 공유. app.py는 국내 관행
# (상승=빨강/하락=파랑)을 그대로 유지하므로 이 파일에서만 이 색상을 쓴다.
UP_COLOR, DOWN_COLOR, FLAT_COLOR = theme.UP_COLOR, theme.DOWN_COLOR, theme.FLAT_COLOR
UP_SOFT, DOWN_SOFT, FLAT_SOFT = theme.UP_SOFT, theme.DOWN_SOFT, theme.FLAT_SOFT
_SENT_COLOR = {"긍정": UP_COLOR, "중립": FLAT_COLOR, "부정": DOWN_COLOR}
_MODEL_LABELS = {"Ridge": "안정형", "RandomForest": "다각형", "GradientBoosting": "정밀형"}

_CRYPTO_BENCHMARKS = {"비트코인": "KRW-BTC", "이더리움": "KRW-ETH", "리플": "KRW-XRP"}
_CRYPTO_DEFAULT = ("KRW-BTC", "비트코인")
_BAR_PERIODS = {"일봉": None, "주봉": "W", "월봉": "ME"}  # 차트 전용 — 시세 요약·예측은 항상 일봉 기준
_CHART_TYPES = ("캔들", "라인", "하이킨아시")
# 분봉/시간봉은 업비트 API 자체(코인)에만 있다 — 주식(FinanceDataReader/GitHub 스냅샷)은
# 일봉 이하 데이터가 아예 없어서 국내증시 모드에서는 이 옵션 자체를 보여주지 않는다.
_CRYPTO_MINUTE_PERIODS = {
    "1분": 1,
    "3분": 3,
    "5분": 5,
    "10분": 10,
    "15분": 15,
    "30분": 30,
    "1시간": 60,
    "4시간": 240,
}
_MINUTE_CANDLE_COUNT = 300  # 최근 N개만 (업비트 요청 상한 200개/회 대비 최대 2회 페이지네이션)
_STOCK_DEFAULT = ("005930", "삼성전자")
_US_DEFAULT = ("AAPL", "Apple Inc")
# config.INDICES 키 중 미국 비교에 적합한 항목만 (KOSPI/KOSDAQ류 제외)
_US_BENCHMARKS = ("NASDAQ", "S&P500", "DOW", "NIKKEI225", "VIX")


def _change_html(
    diff: float,
    pct: float,
    *,
    soft: bool = False,
    unit: str = "원",
    decimals: int = 0,
    usd_rate: float | None = None,
) -> str:
    up, down, flat = (UP_SOFT, DOWN_SOFT, FLAT_SOFT) if soft else (UP_COLOR, DOWN_COLOR, FLAT_COLOR)
    color = up if diff > 0 else down if diff < 0 else flat
    arrow = "▲" if diff > 0 else "▼" if diff < 0 else "―"
    if unit == "$":
        diff_str = f"${diff:+,.{decimals}f}"
    else:
        diff_str = f"{diff * usd_rate if usd_rate else diff:+,.{decimals}f}{unit}"
    return (
        f"<span style='color:{color};font-weight:600;font-size:0.95rem'>"
        f"{arrow} {diff_str} ({pct:+.2f}%)</span>"
    )


def _money(v: float, *, unit: str = "원", usd_rate: float | None = None) -> str:
    """가격 한 값을 통화 단위에 맞게 포맷한다. unit="$"면 소수 2자리 접두사 표기, unit="원"이면
    정수 접미사 표기 — 이때 usd_rate가 주어지면(해외증시 원화 토글 켜짐) v(달러)를 원화로
    환산한 뒤 포맷한다. usd_rate가 없으면(국내/코인처럼 원래부터 원화 값) 그대로 포맷."""
    if unit == "$":
        return f"${v:,.2f}"
    if usd_rate:
        v = v * usd_rate
    return f"{v:,.0f}원"


def _big_money(v: float, *, unit: str = "원", usd_rate: float | None = None) -> str:
    """시가총액/거래대금처럼 큰 금액을 스케일링해서 포맷한다. unit="$"면 10억달러(B) 단위,
    unit="원"이면 억원 단위 — usd_rate가 주어지면 환산 후 억원으로 포맷(위 _money와 동일 원리)."""
    if unit == "$":
        return f"${v / 1e9:,.2f}B"
    if usd_rate:
        v = v * usd_rate
    return f"{v / 1e8:,.0f}억원"


def _section_title(text: str) -> None:
    """카드 제목을 스크롤 가능한 높이 고정 박스(st.container(height=N)) *바깥*, 바로 위에
    그린다 — 이 프로젝트가 쓰는 Streamlit 버전에서는 컨테이너 내부에 position:sticky를 써도
    실측 결과 그대로 스크롤을 따라 밀려 올라가 사라진다(래퍼 구조 문제로 추정, CSS만으로는
    못 고침). 그래서 제목을 아예 스크롤 영역 밖에 둬 내부 스크롤과 무관하게 항상 보이게
    한다 — 호출부가 `with st.container(height=N):` *진입 전에* 이 함수를 호출해야 한다.
    """
    st.markdown(
        "<div style='background:rgba(127,127,127,0.55);border-radius:6px;"
        "padding:0.4rem 0.7rem;margin-bottom:0.4rem;font-weight:700;"
        f"font-size:1.05rem;line-height:1.3;'>{text}</div>",
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=config.CACHE_TTL_SEC, show_spinner="전종목 시세 불러오는 중...")
def _screen(market: str, days_back: int, min_marcap: float, min_volume: float) -> pd.DataFrame:
    return screener.screen(market=market, days_back=days_back, min_marcap=min_marcap, min_volume=min_volume)


@st.cache_data(ttl=config.CACHE_TTL_SEC, show_spinner="코인 시세 불러오는 중...")
def _crypto_screen() -> pd.DataFrame:
    return crypto_screener.screen()


@st.cache_data(ttl=config.CACHE_TTL_SEC, show_spinner="해외증시 시세 불러오는 중...")
def _us_screen() -> pd.DataFrame:
    return us_screener.screen()


@st.cache_data(ttl=config.CACHE_TTL_SEC, show_spinner=False)
def _usdkrw_rate() -> float:
    """해외증시 원화 환산 토글용 캐시 래퍼. 실제 조회는 data_loader.get_usdkrw_rate()를
    그대로 쓴다(scripts/run_daily_trading.py의 해외증시 매매 환산도 같은 함수를 쓴다 —
    fdr 심볼 문자열을 두 곳에 따로 두지 않는다). 실패하면 NaN을 돌려주고,
    호출부(_us_currency_state)가 이걸 감지해서 원화 표기를 강제로 끄고 달러로 폴백한다."""
    return dl.get_usdkrw_rate()


def _us_currency_state() -> tuple[str, float, int]:
    """해외증시 상단 '원화(₩)로 표기' 체크박스(session_state 키 "us_show_krw") 상태를 읽어
    (통화단위, 환율, 소수자리)를 돌려준다. 국내증시/코인은 원래부터 원화 값이라 항상
    ("원", 1.0(환산 안 함), 0)을 돌려주고(usd_rate=1.0으로 배수 곱해도 그대로라 호출부가
    is_us 분기를 따로 안 해도 된다), 해외증시는 토글에 따라 ("$", 1.0, 2) 또는
    ("원", 환율, 0)이 된다."""
    if not is_us:
        return "원", 1.0, 0
    show_krw = st.session_state.get("us_show_krw", False)
    if not show_krw:
        return "$", 1.0, 2
    rate = _usdkrw_rate()
    if not rate or pd.isna(rate):
        return "$", 1.0, 2  # 환율 조회 실패 — 원화 강제 해제하고 달러로 폴백
    return "원", rate, 0


@st.cache_data(ttl=config.CACHE_TTL_SEC, show_spinner="가격 데이터 불러오는 중...")
def _price(symbol: str, start: str) -> pd.DataFrame:
    return dl.get_price(symbol, start=start)


@st.cache_data(ttl=config.CACHE_TTL_SEC, show_spinner="코인 가격 데이터 불러오는 중...")
def _crypto_price(market: str, start: str) -> pd.DataFrame:
    return crypto_loader.get_price(market, start=start)


@st.cache_data(ttl=300, show_spinner="분봉/시간봉 불러오는 중...")  # 일봉보다 훨씬 짧은 캐시 — 자주 바뀐다
def _crypto_minute_price(market: str, unit: int, count: int) -> pd.DataFrame:
    return crypto_loader.get_minute_price(market, unit=unit, count=count)


@st.cache_data(ttl=config.NEWS_CACHE_TTL_SEC, show_spinner="뉴스 불러오는 중...")
def _news(code: str, n: int) -> pd.DataFrame:
    return news.fetch_news_with_sentiment(code, n=n)


@st.cache_data(ttl=config.NEWS_CACHE_TTL_SEC, show_spinner="뉴스 불러오는 중...")
def _crypto_news(keyword: str, n: int) -> pd.DataFrame:
    return crypto_news.fetch_news_with_sentiment(keyword, n=n)


@st.cache_data(ttl=config.NEWS_CACHE_TTL_SEC, show_spinner="공시 불러오는 중...")
def _dart(code: str) -> tuple[pd.DataFrame, str]:
    """(공시 DataFrame, 상태). 상태: "ok" | "unavailable" | "no_key".

    예외를 그대로 올리면 st.cache_data가 결과를 캐시하지 않아 매 렌더마다 DART를 다시
    때리고(해외에서 매번 수 초씩 매달림), 안 잡히면 대시보드 전체가 죽는다. 실패도
    값으로 돌려 캐시에 태운다 — TTL(30분) 동안은 재시도하지 않는다."""
    try:
        return dart.fetch_disclosures(code), "ok"
    except dart.DartUnavailable:
        return pd.DataFrame(), "unavailable"
    except dart.DartKeyMissing:
        return pd.DataFrame(), "no_key"


def _safe_predict_advanced(
    price_df: pd.DataFrame, horizon: int, sentiment_hist_full: pd.DataFrame | None
) -> dict:
    try:
        return predictor.train_and_predict_advanced(
            price_df, horizon=horizon, sentiment_hist_full=sentiment_hist_full
        )
    except Exception as e:
        return {"error": f"예측 중 오류가 발생했습니다: {e}"}


# ------------------------------------------------------------------ 컴팩트 레이아웃·다크모드·사이드바 너비 공통 CSS
theme.inject_base_css()
st.markdown("##### 🖥️ 모니터링")

# ==================================================================== 맨 위: 시장 선택
top_left, _top_rest = st.columns([2, 8])
with top_left:
    market_type = st.selectbox("시장 구분", ["국내증시", "해외증시", "코인"], key="market_type")
is_crypto = market_type == "코인"
is_us = market_type == "해외증시"
if is_crypto:
    st.caption(
        "⚠️ 코인 시세는 업비트 공개 API, 뉴스는 네이버 뉴스 키워드 검색 기반입니다. "
        "주간 등락률·시가총액은 데이터 소스 한계로 제공하지 않습니다."
    )
elif is_us:
    st.caption(
        "⚠️ 해외증시(나스닥) 시세는 나스닥 공개 스크리너 API 기준 당일 스냅샷이고, 뉴스는 "
        "네이버 뉴스 키워드 검색 기반입니다. 주간 등락률은 데이터 소스 한계로 제공하지 않고, "
        "거래대금은 거래량×종가 근사치입니다."
    )

left_col, right_col = st.columns([3, 7], gap="medium")

_TOP_ROW_HEIGHT = 800
_BOTTOM_ROW_HEIGHT = 760  # 가격예측·뉴스&공시 박스 세로 (기존 380의 2배)
# 뉴스 탭 스크롤 박스 / 공시 탭 표의 세로 높이 — _BOTTOM_ROW_HEIGHT가 2배로 커진 만큼
# 안쪽 내용 박스도 최대한 키운다. 박스 상단의 탭바·표시개수 선택줄·여백을 뺀 값(대략치라
# 브라우저로 보며 미세조정 필요할 수 있음).
_NEWS_INNER_HEIGHT = 620
# 스크리닝 테이블 높이 상한 — _TOP_ROW_HEIGHT(800)는 우측 "종목 상세" 박스와 시작줄을
# 맞추려고 고정한 값이라, 좌측 "주가 요약"에는 필터·탭 등을 빼면 여유 공간이 남는다.
# 예전엔 260으로 낮게 고정해뒀더니 기본 표시개수(30개)에서도 테이블 아래로 빈 여백이
# 크게 남았다 — 그 위의 필터 행 개수가 시장별로 달라(국내증시가 코인보다 한 줄 더 많음)
# 상한도 다르게 잡는다. 브라우저로 직접 확인하며 미세조정한 값은 아니라, 실제로 보면서
# 더 조정이 필요할 수 있다.
_STOCK_TABLE_HEIGHT = 480
_CRYPTO_TABLE_HEIGHT = 520

# ==================================================================== 좌측 상단: 스크리닝
with left_col:
    _section_title("🌎 해외증시 요약" if is_us else "🪙 코인 요약" if is_crypto else "📊 주가 요약")
    with st.container(key="summary", border=True, height=_TOP_ROW_HEIGHT):

        if is_crypto:
            top_n = st.selectbox("표시개수", [10, 20, 30, 50, 100], index=2)
            if st.button("🔄", help="새로고침 (캐시 초기화)"):
                st.cache_data.clear()
                st.rerun()

            try:
                universe = _crypto_screen()
            except Exception as e:
                st.error(f"코인 시세를 불러오지 못했습니다: {e}")
                st.stop()

            market_label = "업비트 KRW"

            def _render_table(ranked, key, value_col, value_label, fmt, scale=1.0):
                display = ranked.rename(columns={"Name": "종목명", "Close": "종가"}).copy()
                display[value_label] = ranked[value_col] / scale
                col_config = {
                    "종가": st.column_config.NumberColumn(format="%,.0f원"),
                    value_label: st.column_config.NumberColumn(format=fmt),
                }
                return st.dataframe(
                    display[["종목명", "종가", value_label]],
                    width="stretch",
                    hide_index=True,
                    height=min(30 * (len(display) + 1), _CRYPTO_TABLE_HEIGHT),
                    column_config=col_config,
                    on_select="rerun",
                    selection_mode="single-row",
                    key=key,
                )

            rise_tab, amount_tab = st.tabs(["📈 등락률", "💰 거래대금"])
            picks = []

            with rise_tab:
                direction = st.selectbox("방향", ["상승", "하락"])
                ranked_rise = screener.top_movers(
                    universe, by="DailyChangeRatio", n=top_n, ascending=(direction == "하락")
                )
                st.caption(
                    f"{market_label} {len(universe):,}종목 중 24시간 {direction}률 상위 {len(ranked_rise)}개"
                )
                picks.append(
                    (
                        "tbl_rise",
                        ranked_rise,
                        _render_table(ranked_rise, "tbl_rise", "DailyChangeRatio", "등락%", "%.2f%%"),
                    )
                )

            with amount_tab:
                ranked_amount = screener.top_movers(universe, by="Amount", n=top_n, ascending=False)
                st.caption(f"{market_label} {len(universe):,}종목 중 24시간 거래대금 상위 {len(ranked_amount)}개")
                picks.append(
                    (
                        "tbl_amount",
                        ranked_amount,
                        _render_table(ranked_amount, "tbl_amount", "Amount", "거래대금", "%,.0f억원", scale=1e8),
                    )
                )
        elif is_us:
            us_top1, _us_spacer, us_top2, us_top3 = st.columns(
                [2, 3, 2.4, 0.9], vertical_alignment="bottom"
            )
            with us_top1:
                top_n = st.selectbox("표시개수", [10, 20, 30, 50, 100], index=2)
            with us_top2:
                st.checkbox("원화(₩)로 표기", key="us_show_krw")
            with us_top3:
                if st.button("🔄", help="새로고침 (캐시 초기화)"):
                    st.cache_data.clear()
                    st.rerun()

            _unit, _usd_rate, _ = _us_currency_state()
            if st.session_state.get("us_show_krw") and _unit == "$":
                st.caption("⚠️ 환율 정보를 가져오지 못해 달러로 표시합니다.")
            _px_fmt = "%,.0f원" if _unit == "원" else "$%,.2f"
            _big_fmt = "%,.0f억원" if _unit == "원" else "$%,.2fB"
            _big_scale = 1e8 if _unit == "원" else 1e9

            try:
                universe = _us_screen()
            except Exception as e:
                st.error(f"해외증시 시세를 불러오지 못했습니다: {e}")
                st.stop()

            market_label = "NASDAQ"

            def _render_table(ranked, key, value_col, value_label, *, is_pct=False):
                display = ranked.rename(columns={"Name": "종목명", "Close": "종가"}).copy()
                display["종가"] = display["종가"] * _usd_rate
                if is_pct:
                    display[value_label] = ranked[value_col]
                    value_fmt = "%.2f%%"
                else:
                    display[value_label] = ranked[value_col] * _usd_rate / _big_scale
                    value_fmt = _big_fmt
                col_config = {
                    "종가": st.column_config.NumberColumn(format=_px_fmt),
                    value_label: st.column_config.NumberColumn(format=value_fmt),
                }
                return st.dataframe(
                    display[["종목명", "종가", value_label]],
                    width="stretch",
                    hide_index=True,
                    height=min(30 * (len(display) + 1), _STOCK_TABLE_HEIGHT),
                    column_config=col_config,
                    on_select="rerun",
                    selection_mode="single-row",
                    key=key,
                )

            rise_tab, amount_tab, marcap_tab = st.tabs(["📈 등락률", "💰 거래대금", "🏢 시가총액"])
            picks = []

            with rise_tab:
                direction = st.selectbox("방향", ["상승", "하락"])
                ranked_rise = screener.top_movers(
                    universe, by="DailyChangeRatio", n=top_n, ascending=(direction == "하락")
                )
                st.caption(
                    f"{market_label} {len(universe):,}종목 중 당일 {direction}률 상위 {len(ranked_rise)}개"
                )
                picks.append(
                    (
                        "tbl_rise",
                        ranked_rise,
                        _render_table(ranked_rise, "tbl_rise", "DailyChangeRatio", "등락%", is_pct=True),
                    )
                )

            with amount_tab:
                ranked_amount = screener.top_movers(universe, by="Amount", n=top_n, ascending=False)
                st.caption(f"{market_label} {len(universe):,}종목 중 거래대금(근사) 상위 {len(ranked_amount)}개")
                picks.append(
                    (
                        "tbl_amount",
                        ranked_amount,
                        _render_table(ranked_amount, "tbl_amount", "Amount", "거래대금"),
                    )
                )

            with marcap_tab:
                ranked_marcap = screener.top_movers(universe, by="Marcap", n=top_n, ascending=False)
                st.caption(f"{market_label} {len(universe):,}종목 중 시가총액 상위 {len(ranked_marcap)}개")
                picks.append(
                    (
                        "tbl_marcap",
                        ranked_marcap,
                        _render_table(ranked_marcap, "tbl_marcap", "Marcap", "시가총액"),
                    )
                )
        else:
            fc1, fc2 = st.columns(2)
            with fc1:
                market = st.selectbox(
                    "시장",
                    ["ALL", "KOSPI", "KOSDAQ"],
                    format_func=lambda m: {"ALL": "전체", "KOSPI": "KOSPI", "KOSDAQ": "KOSDAQ"}[m],
                )
            with fc2:
                top_n = st.selectbox("표시개수", [10, 20, 30, 50, 100], index=2)

            fc3, fc4, fc5 = st.columns([1.3, 1.3, 0.6], vertical_alignment="bottom")
            with fc3:
                min_marcap_eok = st.selectbox(
                    "최소 시총",
                    [0, 100, 300, 500, 1000, 3000],
                    index=2,
                    format_func=lambda v: f"{v}억+" if v else "시총 전체",
                )
            with fc4:
                min_volume = st.selectbox(
                    "최소 거래량",
                    [0, 1000, 5000, 10000, 50000],
                    index=1,
                    format_func=lambda v: f"{v:,}주+" if v else "거래량 전체",
                )
            with fc5:
                if st.button("🔄", help="새로고침 (캐시 초기화)"):
                    st.cache_data.clear()
                    dl.clear_cache()
                    st.rerun()

            try:
                universe = _screen(market, 7, min_marcap_eok * 1e8, float(min_volume))
            except Exception as e:
                st.error(f"스크리닝 데이터를 불러오지 못했습니다: {e}")
                st.stop()

            market_label = "KOSPI+KOSDAQ" if market == "ALL" else market

            def _render_table(ranked, key, value_col, value_label, fmt, scale=1.0):
                display = ranked.rename(columns={"Name": "종목명", "Close": "종가"}).copy()
                display[value_label] = ranked[value_col] / scale
                col_config = {
                    "종가": st.column_config.NumberColumn(format="%,d원"),
                    value_label: st.column_config.NumberColumn(format=fmt),
                }
                return st.dataframe(
                    display[["종목명", "종가", value_label]],
                    width="stretch",
                    hide_index=True,
                    height=min(30 * (len(display) + 1), _STOCK_TABLE_HEIGHT),
                    column_config=col_config,
                    on_select="rerun",
                    selection_mode="single-row",
                    key=key,
                )

            rise_tab, amount_tab, marcap_tab = st.tabs(["📈 상승률", "💰 거래대금", "🏢 시가총액"])
            picks = []

            with rise_tab:
                c1, c2 = st.columns(2)
                with c1:
                    basis = st.selectbox("기준", ["일간", "주간", "월간"])
                with c2:
                    direction = st.selectbox("방향", ["상승", "하락"])
                basis_col = {
                    "일간": "DailyChangeRatio",
                    "주간": "WeeklyChangeRatio",
                    "월간": "MonthlyChangeRatio",
                }[basis]
                ranked_rise = screener.top_movers(
                    universe, by=basis_col, n=top_n, ascending=(direction == "하락")
                )
                st.caption(
                    f"{market_label} {len(universe):,}종목 중 {basis} {direction}률 상위 {len(ranked_rise)}개"
                )
                picks.append(
                    (
                        "tbl_rise",
                        ranked_rise,
                        _render_table(ranked_rise, "tbl_rise", basis_col, f"{basis}%", "%.2f%%"),
                    )
                )

            with amount_tab:
                ranked_amount = screener.top_movers(universe, by="Amount", n=top_n, ascending=False)
                st.caption(f"{market_label} {len(universe):,}종목 중 거래대금 상위 {len(ranked_amount)}개")
                picks.append(
                    (
                        "tbl_amount",
                        ranked_amount,
                        _render_table(ranked_amount, "tbl_amount", "Amount", "거래대금", "%,.0f억원", scale=1e8),
                    )
                )

            with marcap_tab:
                ranked_marcap = screener.top_movers(universe, by="Marcap", n=top_n, ascending=False)
                st.caption(f"{market_label} {len(universe):,}종목 중 시가총액 상위 {len(ranked_marcap)}개")
                picks.append(
                    (
                        "tbl_marcap",
                        ranked_marcap,
                        _render_table(ranked_marcap, "tbl_marcap", "Marcap", "시가총액", "%,.0f억원", scale=1e8),
                    )
                )

        # 시장 전환 시 이전 선택이 다른 시장 코드로 남아있으면 안 되므로, 선택 상태를 시장별로 분리해 둔다.
        _sel_code_key = (
            "selected_code_crypto" if is_crypto else "selected_code_us" if is_us else "selected_code_stock"
        )
        _sel_name_key = (
            "selected_name_crypto" if is_crypto else "selected_name_us" if is_us else "selected_name_stock"
        )

        for tbl_key, ranked_df, event in picks:
            if not event.selection.rows:
                continue
            picked = ranked_df.iloc[event.selection.rows[0]]
            if st.session_state.get(f"_last_{tbl_key}_{market_type}") != picked["Code"]:
                st.session_state[f"_last_{tbl_key}_{market_type}"] = picked["Code"]
                st.session_state[_sel_code_key] = picked["Code"]
                st.session_state[_sel_name_key] = picked["Name"]

# ==================================================================== 우측 상단: 종목 상세
with right_col:
    selected_code = st.session_state.get(_sel_code_key)
    selected_name = st.session_state.get(_sel_name_key, "")

    _detail_title = "🌎 해외증시 상세" if is_us else "🪙 코인 상세" if is_crypto else "📈 종목 상세"
    if selected_code:
        _detail_title += f" · {selected_name}"
    _section_title(_detail_title)
    with st.container(key="detail", border=True, height=_TOP_ROW_HEIGHT):

        search_col, period_col, bar_col, idx_col = st.columns([2, 1, 1, 2])
        with search_col:
            if is_crypto:
                placeholder = "예: KRW-BTC, 비트코인"
            elif is_us:
                placeholder = "예: AAPL, 애플"
            else:
                placeholder = "예: 005930, 삼성전자"
            manual = st.text_input("코드/이름 검색", value="", placeholder=placeholder)
        with period_col:
            period_label = st.selectbox("조회 기간", list(DATE_RANGES.keys()), index=3)
        with bar_col:
            bar_options = (
                list(_CRYPTO_MINUTE_PERIODS.keys()) + list(_BAR_PERIODS.keys())
                if is_crypto
                else list(_BAR_PERIODS.keys())
            )
            bar_label = st.selectbox("봉 주기", bar_options, index=len(bar_options) - 3 if is_crypto else 0)
        with idx_col:
            if is_crypto:
                idx_sel = st.multiselect("코인 비교", list(_CRYPTO_BENCHMARKS.keys()), default=[])
            elif is_us:
                idx_sel = st.multiselect("지수 비교", list(_US_BENCHMARKS), default=["NASDAQ", "S&P500"])
            else:
                idx_sel = st.multiselect("지수 비교", list(config.INDICES.keys()), default=["KOSPI"])

        # 자동완성 최소 글자수 — 숫자(종목코드)는 3자, 문자(종목명)는 2자부터 목록을 보여준다.
        # 너무 짧은 입력(1글자 등)에서 바로 검색하면 결과가 너무 많아 오히려 고르기 어렵다.
        query = manual.strip()
        _autocomplete_min_len = 3 if query.isdigit() else 2
        if query and len(query) < _autocomplete_min_len:
            st.caption(f"{_autocomplete_min_len}글자 이상 입력하면 자동완성 목록이 나타납니다.")
        elif query:
            if is_crypto:
                hits = crypto_loader.find_symbol(query)
            elif is_us:
                hits = dl.find_symbol(query, market="NASDAQ")
            else:
                hits = dl.find_symbol(query)
            code_col = "Code" if "Code" in hits.columns else "Symbol"
            if not hits.empty:
                options = {
                    f"{row[code_col]} · {row['Name']}": row[code_col] for _, row in hits.head(20).iterrows()
                }
                pick = st.selectbox(f"자동완성 ({len(options)}건)", list(options.keys()))
                selected_code = options[pick]
                selected_name = pick.split(" · ", 1)[1]
            else:
                st.warning("검색 결과가 없습니다.")

        if not selected_code:
            if is_crypto:
                selected_code, selected_name = _CRYPTO_DEFAULT
            elif is_us:
                selected_code, selected_name = _US_DEFAULT
            else:
                selected_code, selected_name = _STOCK_DEFAULT

        # 예전엔 이 이름(코드)를 차트 자체의 Plotly title로 그렸는데, 매물대/레인지셀렉터가
        # 추가되면서 위쪽 공간이 빡빡해져 차트 상단(캔들/레인지셀렉터 버튼)과 겹쳐 보였다 —
        # 검색창 바로 아래에 별도 Streamlit 텍스트로 빼서 차트 영역과 아예 분리했다.
        st.markdown(f"##### {selected_name} ({selected_code})")

        days_back = DATE_RANGES[period_label]
        start_date = (
            config.DEFAULT_START
            if days_back is None
            else (pd.Timestamp.today() - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d")
        )

        price_df = (
            _crypto_price(selected_code, start_date) if is_crypto else _price(selected_code, start_date)
        )
        if price_df.empty:
            st.error(f"'{selected_code}' 가격 데이터를 찾을 수 없습니다.")
            st.stop()

        # ------------------------------------------------------ 시세 요약 (원 단위)
        _last = price_df.iloc[-1]
        _last_date = price_df.index[-1]
        _close = float(_last["Close"])
        _prev_close = float(price_df["Close"].iloc[-2]) if len(price_df) >= 2 else None

        _urow = universe[universe["Code"] == selected_code]
        _amount_approx = _urow.empty
        _amount_stale = False
        _marcap = None
        if not _urow.empty:
            _amount = float(_urow.iloc[0]["Amount"])
            _marcap_raw = _urow.iloc[0]["Marcap"]
            _marcap = float(_marcap_raw) if pd.notna(_marcap_raw) else None
            if not is_crypto:
                _snap_vol = float(_urow.iloc[0]["Volume"])
                _live_vol = float(_last["Volume"])
                _amount_stale = _live_vol > 0 and abs(_snap_vol - _live_vol) / _live_vol > 0.01
        else:
            _amount = _close * float(_last["Volume"])

        _vol_unit = selected_code.replace("KRW-", "") if is_crypto else "주"
        _vol_fmt = f"{float(_last['Volume']):,.4f}" if is_crypto else f"{float(_last['Volume']):,.0f}"

        _unit, _usd_rate, _decimals = _us_currency_state()

        # 시세 요약 8개 항목(현재가~거래대금)은 컬럼 하나가 85px 안팎이라, 거래량·거래대금처럼
        # 자릿수가 큰 값(예: "12,765,756주")은 전역 stMetricValue 폰트 크기(1.25rem)로는
        # 글자가 넘쳐 "..."로 잘렸다(실측 확인: 필요 폭 109px > 가용 85px). 이 행에만
        # 적용되는 스코프 CSS로 폰트를 줄이고 컬럼 사이 여백도 좁혀 실제 가용 폭을 넓힌다.
        st.markdown(
            """
<style>
.st-key-price_summary [data-testid="stMetricValue"] p {
    font-size: 0.86rem !important;
    line-height: 1.3 !important;
    overflow: visible !important;
    text-overflow: clip !important;
    white-space: nowrap !important;
}
.st-key-price_summary [data-testid="stMetricLabel"] p {font-size: 0.74rem !important;}
.st-key-price_summary div[data-testid="stHorizontalBlock"] {gap: 0.3rem !important;}
</style>
""",
            unsafe_allow_html=True,
        )
        with st.container(key="price_summary"):
            _r1 = st.columns(8)
            _r1[0].metric("현재가", _money(_close, unit=_unit, usd_rate=_usd_rate))
            if _prev_close:
                _diff = _close - _prev_close
                _r1[0].markdown(
                    _change_html(
                        _diff, _diff / _prev_close * 100, unit=_unit, decimals=_decimals, usd_rate=_usd_rate
                    ),
                    unsafe_allow_html=True,
                )
            _r1[1].metric("시가", _money(float(_last["Open"]), unit=_unit, usd_rate=_usd_rate))
            _r1[2].metric("고가", _money(float(_last["High"]), unit=_unit, usd_rate=_usd_rate))
            _r1[3].metric("저가", _money(float(_last["Low"]), unit=_unit, usd_rate=_usd_rate))
            _r1[4].metric("전일종가", _money(_prev_close, unit=_unit, usd_rate=_usd_rate) if _prev_close else "—")
            _r1[5].metric(
                "변동폭", _money(float(_last["High"]) - float(_last["Low"]), unit=_unit, usd_rate=_usd_rate)
            )
            _r1[6].metric("거래량", f"{_vol_fmt}{_vol_unit}")
            _r1[7].metric(
                "거래대금" if not is_crypto else "24H 거래대금",
                _big_money(_amount, unit=_unit, usd_rate=_usd_rate),
            )

        _notes = [f"{_last_date:%Y-%m-%d (%a)} 기준"]
        if _marcap:
            _notes.append(f"시총 {_big_money(_marcap, unit=_unit, usd_rate=_usd_rate)}")
        if is_crypto:
            _notes.append("거래대금은 최근 24시간 누적(업비트 기준)")
        elif is_us:
            _notes.append("거래대금은 거래량×종가 근사치(나스닥 스크리너 API 기준)")
        elif _amount_approx:
            _notes.append("거래대금 추정치")
        elif _amount_stale:
            _notes.append("거래대금·시총은 KRX 스냅샷 기준")
        _notes.append("상승=초록 / 하락=빨강")
        st.caption(" · ".join(_notes))

        # 봉 주기: 시세 요약·예측은 항상 price_df(일봉)를 쓰고, 차트만 바꾼다 —
        # 업비트도 상단 현재가 티커는 실시간 그대로 두고 캔들 굵기만 바꾼다.
        if bar_label in _CRYPTO_MINUTE_PERIODS:
            # 분봉/시간봉은 일봉을 리샘플링해서 만들 수 없다(더 잘게 쪼개는 건 원본에
            # 없던 정보를 만드는 것) — 업비트 분봉 API를 따로 호출한다. 코인 모드에서만
            # 이 옵션이 보이므로 is_crypto 분기가 필요 없다.
            chart_price_df = _crypto_minute_price(
                selected_code, _CRYPTO_MINUTE_PERIODS[bar_label], _MINUTE_CANDLE_COUNT
            )
            st.caption(f"분봉/시간봉은 최근 {_MINUTE_CANDLE_COUNT}개까지만 제공됩니다(업비트 API 한계).")
        else:
            bar_rule = _BAR_PERIODS[bar_label]
            chart_price_df = ind.resample_ohlcv(price_df, bar_rule) if bar_rule else price_df
        if chart_price_df.empty:  # 데이터가 비면(너무 짧은 상장 이력 등) 안전하게 일봉으로 폴백
            chart_price_df = price_df

        enriched = ind.add_all(chart_price_df)

        # 컬럼을 잘게 쪼갤수록(예전엔 체크박스 10개를 한 줄에) 좁은 화면에서 Streamlit이
        # 모바일 스택 레이아웃으로 전환해버려서 PC 화면 비율이 세로로 길게 깨졌다 —
        # 개별 체크박스 대신 멀티셀렉트로 묶어서 한 줄당 컬럼 수를 4개로 줄였다.
        ma_col, ind_col, ctype_col, log_col = st.columns(
            [1.5, 2.1, 1.3, 0.9], vertical_alignment="bottom"
        )
        with ma_col:
            ma_sel = st.multiselect("이동평균", SMA_CHOICES, default=[20, 60])
        with ind_col:
            ind_sel = st.multiselect(
                "보조지표",
                ["볼린저밴드", "거래량", "RSI", "MACD", "스토캐스틱", "일목균형표", "매물대"],
                default=["거래량"],
            )
        with ctype_col:
            chart_type_label = st.selectbox("차트 유형", _CHART_TYPES, index=0)
        with log_col:
            log_y = st.checkbox("로그축", value=False)

        show_bb = "볼린저밴드" in ind_sel
        show_vol = "거래량" in ind_sel
        show_rsi = "RSI" in ind_sel
        show_macd = "MACD" in ind_sel
        show_stoch = "스토캐스틱" in ind_sel
        show_ichimoku = "일목균형표" in ind_sel
        show_vp = "매물대" in ind_sel

        if show_bb:
            bbcol1, bbcol2 = st.columns(2)
            bb_window = bbcol1.number_input("볼밴드 기간", min_value=5, max_value=120, value=20, step=1)
            bb_std = bbcol2.number_input("볼밴드 표준편차", min_value=0.5, max_value=4.0, value=2.0, step=0.1)
            enriched = enriched.drop(columns=["upper", "lower", "pct_b"], errors="ignore").join(
                ind.bollinger(enriched["Close"], window=int(bb_window), num_std=bb_std)[
                    ["upper", "lower", "pct_b"]
                ]
            )

        sma_windows = list(ma_sel)
        for w in sma_windows:
            col = f"sma{w}"
            if col not in enriched.columns:
                enriched[col] = ind.sma(enriched["Close"], w)

        if show_stoch:
            enriched = enriched.join(ind.stochastic(enriched))
        if show_ichimoku:
            enriched = enriched.join(ind.ichimoku(enriched))

        chart_source = ind.heikin_ashi(enriched) if chart_type_label == "하이킨아시" else enriched
        chart_type = "line" if chart_type_label == "라인" else "candle"
        vp_df = ind.volume_profile(chart_source) if show_vp else None

        overlays = {}
        for label in idx_sel:
            ov_symbol = _CRYPTO_BENCHMARKS[label] if is_crypto else label
            idx_df = _crypto_price(ov_symbol, start_date) if is_crypto else _price(ov_symbol, start_date)
            if not idx_df.empty:
                overlays[label] = idx_df["Close"]

        _extra_indicator_rows = int(show_rsi) + int(show_stoch) + int(show_macd)

        if show_vol:
            # 가격·거래량은 각각 독립된 Plotly 컴포넌트로 그려서 그 사이 경계를 마우스
            # 드래그로 위아래 리사이즈할 수 있게 한다(resizable_chart.py 참고) — Plotly
            # make_subplots 자체는 서브플롯 행 높이를 드래그로 바꾸는 기능이 없어서, 하나로
            # 합쳐 그리던 기존 방식(가격+거래량+RSI...을 한 Figure에 고정 비율로 배치)으로는
            # 이 기능을 만들 수 없었다.
            price_fig = charts.build_chart(
                chart_source,
                title="",
                sma_windows=tuple(sma_windows),
                show_bollinger=show_bb,
                show_volume=False,
                chart_type=chart_type,
                show_rangeselector=True,
                log_y=log_y,
                show_ichimoku=show_ichimoku,
                index_overlays=overlays or None,
                base_height=320,
                panel_height=0,
                volume_profile=vp_df,
                crosshair=True,
                drag_pan=True,
                drawing_tools=True,
                up_color=UP_COLOR,
                down_color=DOWN_COLOR,
            )
            volume_fig = charts.build_volume_chart(
                chart_source, up_color=UP_COLOR, down_color=DOWN_COLOR, crosshair=True, height=110
            )
            extra_fig = None
            if _extra_indicator_rows:
                extra_fig = charts.build_chart(
                    chart_source,
                    title="",
                    include_price=False,
                    show_volume=False,
                    show_rsi=show_rsi,
                    show_stochastic=show_stoch,
                    show_macd=show_macd,
                    crosshair=True,
                    base_height=85,
                    panel_height=85,
                    up_color=UP_COLOR,
                    down_color=DOWN_COLOR,
                )
            resizable_chart.render(
                price_fig,
                volume_fig,
                extra_fig,
                price_height=320,
                volume_height=110,
                extra_height=85 * _extra_indicator_rows,
            )
        else:
            fig = charts.build_chart(
                chart_source,
                title="",
                sma_windows=tuple(sma_windows),
                show_bollinger=show_bb,
                show_volume=False,
                show_rsi=show_rsi,
                show_macd=show_macd,
                index_overlays=overlays or None,
                base_height=320,
                panel_height=85,
                chart_type=chart_type,
                show_rangeselector=True,
                log_y=log_y,
                show_stochastic=show_stoch,
                show_ichimoku=show_ichimoku,
                volume_profile=vp_df,
                crosshair=True,
                drag_pan=True,
                drawing_tools=True,
                up_color=UP_COLOR,
                down_color=DOWN_COLOR,
            )
            # scrollZoom은 Figure 속성이 아니라 렌더링 옵션이라 여기서 켠다 — 업비트처럼
            # 마우스 휠로 확대/축소, 드래그로 이동(위 drag_pan=True)하는 조작감을 맞춘다.
            st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})

        summary = ind.summary(price_df["Close"])
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("수익률", f"{summary['total_return'] * 100:.1f}%")
        m2.metric("CAGR", f"{summary['cagr'] * 100:.1f}%")
        m3.metric("변동성", f"{summary['volatility'] * 100:.1f}%")
        m4.metric("샤프", f"{summary['sharpe']:.2f}")
        m5.metric("MDD", f"{summary['max_drawdown'] * 100:.1f}%")

_HORIZON_PRESETS = (1, 5, 15, 30, 60, 90)


def _apply_horizon_preset() -> None:
    """프리셋 드롭다운에서 고른 값을 "거래일 수" 입력값에 그대로 반영한다."""
    st.session_state["custom_horizon"] = st.session_state["horizon_preset"]


# ==================================================================== 좌측 하단: 가격 예측
with left_col:
    _section_title("💹 가격 예측")
    with st.container(key="predict", border=True, height=_BOTTOM_ROW_HEIGHT):
        # 이 박스 안 세로 요소 간격(행간)만 전역 0.45rem → 약 1.3배로 넓힌다.
        st.markdown(
            "<style>.st-key-predict div[data-testid='stVerticalBlock']{gap:0.6rem;}</style>",
            unsafe_allow_html=True,
        )
        st.caption("⚠️ 참고용 추정치이며 투자 조언이 아닙니다. 과거 시세 흐름을 바탕으로 자동 계산한 값입니다.")

        try:
            if is_crypto or is_us:
                # 코인·해외증시 둘 다 종목코드 기반 뉴스 소스가 없어 종목명 키워드 검색(crypto_news)을 공유 재사용한다.
                news.log_sentiment_from_news(selected_code, _crypto_news(selected_name, 10))
            else:
                news.log_sentiment_from_news(selected_code, _news(selected_code, 10))
        except Exception:
            pass

        st.session_state.setdefault("custom_horizon", 5)
        st.selectbox(
            "몇 거래일 후를 예측할까요?",
            _HORIZON_PRESETS,
            index=_HORIZON_PRESETS.index(5),
            format_func=lambda d: f"{d}일",
            key="horizon_preset",
            on_change=_apply_horizon_preset,
        )

        hcol1, hcol2 = st.columns([2, 1], vertical_alignment="bottom")
        with hcol1:
            st.number_input("거래일 수", min_value=1, max_value=90, step=1, key="custom_horizon")
        with hcol2:
            query_clicked = st.button("조회", key="custom_horizon_query")

        if query_clicked:
            horizon = int(st.session_state["custom_horizon"])
            with st.spinner("여러 예측 방식을 비교하며 계산하는 중..."):
                full_price_df = (
                    _crypto_price(selected_code, config.DEFAULT_START)
                    if is_crypto
                    else _price(selected_code, config.DEFAULT_START)
                )
                sentiment_hist_full = news.sentiment_history_full(selected_code)
                st.session_state["adv_pred"] = {
                    "code": selected_code,
                    "horizon": horizon,
                    "result": _safe_predict_advanced(
                        full_price_df, horizon=horizon, sentiment_hist_full=sentiment_hist_full
                    ),
                }

        adv_pred = st.session_state.get("adv_pred")
        if adv_pred and adv_pred["code"] == selected_code:
            horizon = adv_pred["horizon"]
            pred = adv_pred["result"]
            if "error" in pred:
                st.info(pred["error"])
            else:
                _pred_unit, _pred_rate, _pred_decimals = _us_currency_state()
                # 첫 줄: "N거래일 후 예측" 바로 오른쪽에 예측 금액을 나란히, 그 아래 줄 오른쪽에
                # 등락(+금액/+%)을 이어 붙인다 — 박스 전체 너비를 쓰므로 컬럼으로 반씩 나누지
                # 않는다(반으로 나누면 아래 세부 정보 줄이 줄바꿈된다).
                st.markdown(
                    "<div style='display:flex;justify-content:space-between;align-items:baseline;'>"
                    f"<span style='font-weight:700;'>{horizon}거래일 후 예측"
                    f" ({pred['target_date'].strftime('%m-%d(%a)')})</span>"
                    "<span style='font-size:1.35rem;font-weight:700;'>"
                    f"{_money(pred['predicted_price'], unit=_pred_unit, usd_rate=_pred_rate)}</span>"
                    "</div>"
                    "<div style='text-align:right;'>"
                    + _change_html(
                        pred["predicted_price"] - pred["last_close"],
                        pred["predicted_return"] * 100,
                        soft=True,
                        unit=_pred_unit,
                        decimals=_pred_decimals,
                        usd_rate=_pred_rate,
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"{_MODEL_LABELS.get(pred['best_model'], pred['best_model'])} 방식 · "
                    f"최근 {pred['n_holdout']}일로 검증 · "
                    f"평균 오차 {_money(pred['mae'], unit=_pred_unit, usd_rate=_pred_rate)}"
                    f"({pred['mape'] * 100:.1f}%) · 방향적중 {pred['directional_accuracy'] * 100:.0f}%"
                )

                # ---------------------------------------------- 실제 추이(실선) + 예측 추이(대시선)
                lookback = max(30, horizon * 5)  # 예측 기간에 비례해 과거 구간도 넉넉히 보여준다
                hist = price_df.tail(lookback)
                pred_color = (
                    UP_SOFT
                    if pred["predicted_price"] > pred["last_close"]
                    else DOWN_SOFT
                    if pred["predicted_price"] < pred["last_close"]
                    else FLAT_SOFT
                )
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=hist.index, y=hist["Close"], name="실제", line=dict(width=1.8, color="#0ea5e9"))
                )
                fig.add_trace(
                    go.Scatter(
                        x=[hist.index[-1], pred["target_date"]],
                        y=[hist["Close"].iloc[-1], pred["predicted_price"]],
                        name=f"{horizon}거래일 후 예측",
                        line=dict(width=2, color=pred_color, dash="dash"),
                    )
                )
                fig.update_layout(
                    height=200,
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    "예측선은 마지막 실제 종가와 예측값을 직선으로 이은 것으로, 그 사이의 실제 경로를 뜻하지 않습니다."
                )

                with st.expander("📋 예측 상세 보기"):
                    st.markdown(
                        f"선택된 방식: **{_MODEL_LABELS.get(pred['best_model'], pred['best_model'])}** · "
                        f"학습에 쓴 데이터 {pred['n_train']}일 · 검증 데이터 {pred['n_holdout']}일 · "
                        f"참고한 뉴스 기록 {pred['news_days']}일"
                    )
                    cv_df = pd.DataFrame(
                        {
                            "예측 방식": [_MODEL_LABELS.get(m, m) for m in pred["cv_scores"]],
                            "오차(작을수록 정확)": list(pred["cv_scores"].values()),
                        }
                    )
                    st.dataframe(cv_df, hide_index=True, width="stretch")
                    st.dataframe(
                        pred["feature_importance"].rename(columns={"label": "설명", "coef": "영향도"})[
                            ["설명", "영향도"]
                        ],
                        hide_index=True,
                        width="stretch",
                    )
                    st.caption(
                        "위 정확도는 예측 방식을 고를 때 쓰지 않은 별도 기간으로 확인한 값입니다. "
                        "아래 표는 예측에 영향을 많이 준 항목일수록 위쪽에 있습니다."
                    )
        else:
            st.caption("거래일 수를 정하고 조회를 누르면 예측 결과가 여기 표시됩니다.")

# ==================================================================== 우측 하단: 뉴스 & 공시
with right_col:
    _section_title("📰 뉴스" if (is_crypto or is_us) else "📰 뉴스 & 공시")
    with st.container(key="news", border=True, height=_BOTTOM_ROW_HEIGHT):

        # st.tabs()를 컬럼 안에 넣으면 그 컬럼 너비만큼만 차지해 내용(뉴스 카드)이 좁아진다
        # (예전엔 표시개수 셀렉트박스를 옆에 두려고 [4,1] 컬럼 안에 tabs를 넣었더니 뉴스
        # 제목·요약이 실제 배정된 탭 너비보다 훨씬 좁게 잘렸다). 탭은 "뉴스" 박스 전체
        # 너비로 그리고, 표시개수 선택은 뉴스 탭 내부의 작은 우측 정렬 줄로 옮긴다.
        tab_labels = ["📰 뉴스"] if (is_crypto or is_us) else ["📰 뉴스", "📋 공시 (DART)"]
        tabs = st.tabs(tab_labels)
        news_tab = tabs[0]
        dart_tab = tabs[1] if len(tabs) > 1 else None

        with news_tab:
            _hdr_l, hdr_r = st.columns([5, 1])
            with hdr_r:
                news_n = st.selectbox(
                    "표시개수",
                    [5, 10, 15, 20],
                    index=1,
                    key="news_n",
                    label_visibility="collapsed",
                    help="뉴스 표시 개수",
                )
            news_df = (
                _crypto_news(selected_name, news_n) if (is_crypto or is_us) else _news(selected_code, news_n)
            )

            if news_df.empty:
                st.info("최근 뉴스를 찾지 못했습니다.")
            else:
                with st.container(height=_NEWS_INNER_HEIGHT):
                    for _, row in news_df.iterrows():
                        with st.container(border=True):
                            left, right = st.columns([6, 1])
                            with left:
                                title_safe = html.escape(row["title"])
                                url_safe = html.escape(row["url"], quote=True)
                                meta_safe = html.escape(f"{row['press']} · {row['date']}")
                                summary_safe = html.escape(row["summary"])
                                st.markdown(
                                    "<div style='display:flex;justify-content:space-between;"
                                    "align-items:baseline;gap:0.6em;'>"
                                    f"<a href='{url_safe}' target='_blank'>{title_safe}</a>"
                                    f"<span style='font-size:0.8em;opacity:0.65;white-space:nowrap;'>{meta_safe}</span>"
                                    "</div>"
                                    "<div style='white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
                                    f"font-size:0.88em;opacity:0.85;' title='{summary_safe}'>{summary_safe}</div>",
                                    unsafe_allow_html=True,
                                )
                            with right:
                                color = _SENT_COLOR[row["sentiment_label"]]
                                st.markdown(
                                    "<div style='text-align:center;padding-top:0.3em'>"
                                    f"<span style='color:{color};font-weight:700;font-size:0.95em'>{row['sentiment_label']}</span><br>"
                                    f"<span style='color:{color};font-size:0.78em'>{row['sentiment_score']:+d}</span>"
                                    "</div>",
                                    unsafe_allow_html=True,
                                )

        if dart_tab is not None:
            with dart_tab:
                dart_df, dart_status = _dart(selected_code)
                if dart_status == "no_key":
                    st.info(
                        "DART_API_KEY가 설정되지 않아 공시를 불러올 수 없습니다. "
                        "앱 설정 Secrets에 DART_API_KEY를 (섹션 없이 최상위에) 추가하세요."
                    )
                elif dart_status == "unavailable":
                    st.info(
                        "DART 서버에 연결하지 못했습니다. 한국 외 지역(예: Streamlit Cloud) "
                        "배포에서는 opendart.fss.or.kr 접속이 제한될 수 있습니다. "
                        "잠시 후 다시 시도해 주세요."
                    )
                elif dart_df.empty:
                    st.info("최근 90일 내 공시가 없습니다.")
                else:
                    st.dataframe(
                        dart_df.rename(
                            columns={"rcept_dt": "접수일", "report_nm": "보고서명", "flr_nm": "제출인"}
                        ),
                        column_config={"url": st.column_config.LinkColumn("링크", display_text="열기")},
                        hide_index=True,
                        height=_NEWS_INNER_HEIGHT,
                        width="stretch",
                    )
