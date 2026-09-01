"""KOSPI/KOSDAQ 상승률 스크리닝 대시보드.

실행: .venv\\Scripts\\streamlit.exe run app.py
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from src import charts, config, dart, news, predictor, screener
from src import data_loader as dl
from src import indicators as ind

st.set_page_config(page_title="주가 스크리닝 대시보드", layout="wide")

DATE_RANGES = {"1개월": 30, "3개월": 90, "6개월": 180, "1년": 365, "2년": 730, "3년": 1095, "전체": None}
SMA_CHOICES = (5, 20, 60, 120)

# 색상 관행: 상승=빨강 / 하락=파랑 (미국식과 반대). charts.py의 캔들 색(_UP/_DOWN)과 같은 계열.
# 확정된 시세는 캔들과 동일한 원색을 쓰고, '예측값'은 사실이 아니라 추정치이므로
# 같은 색상 계열을 유지하되 채도를 낮춘 파스텔 톤으로 그려 시각적으로 구분한다.
# 파스텔 톤은 원색과 색상(H)·명도(L)를 그대로 두고 채도(S)만 84~91% -> 50%로 낮춘 값이라
# 같은 색 계열로 보이면서 톤만 부드러워진다. 라이트/다크 배경 모두에서 대비 3:1 이상을
# 유지하도록 명도는 건드리지 않았다 (더 밝히면 흰 배경에서 읽기 어려워진다).
UP_COLOR, DOWN_COLOR, FLAT_COLOR = "#ef4444", "#3b82f6", "#6b7280"  # 실측 시세용 (원색)
UP_SOFT, DOWN_SOFT, FLAT_SOFT = "#cc6666", "#668dcc", "#9ca3af"  # 예측값용 (파스텔)


def _change_html(diff: float, pct: float, *, soft: bool = False) -> str:
    """등락 표시 마크업. soft=True면 예측값용 파스텔 톤을 쓴다.

    st.metric의 delta는 초록/빨강 조합만 지원해 국내 관행(상승=빨강/하락=파랑)과
    어긋나므로, delta를 쓰지 않고 색을 직접 지정해 그린다.
    """
    up, down, flat = (UP_SOFT, DOWN_SOFT, FLAT_SOFT) if soft else (UP_COLOR, DOWN_COLOR, FLAT_COLOR)
    color = up if diff > 0 else down if diff < 0 else flat
    arrow = "▲" if diff > 0 else "▼" if diff < 0 else "―"
    return (
        f"<span style='color:{color};font-weight:600;font-size:0.95rem'>"
        f"{arrow} {diff:+,.0f}원 ({pct:+.2f}%)</span>"
    )


def _section_title(text: str) -> None:
    """카드(박스) 제목을 본문과 구분되는 강조 바로 그린다 — 4개 영역 제목에 공통으로 쓴다."""
    st.markdown(
        "<div style='background:rgba(127,127,127,0.16);border-radius:6px;"
        "padding:0.4rem 0.7rem;margin-bottom:0.5rem;font-weight:700;"
        f"font-size:1.05rem;line-height:1.3;'>{text}</div>",
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=config.CACHE_TTL_SEC, show_spinner="전종목 시세 불러오는 중...")
def _screen(market: str, days_back: int, min_marcap: float, min_volume: float) -> pd.DataFrame:
    return screener.screen(market=market, days_back=days_back, min_marcap=min_marcap, min_volume=min_volume)


@st.cache_data(ttl=config.CACHE_TTL_SEC, show_spinner="가격 데이터 불러오는 중...")
def _price(symbol: str, start: str) -> pd.DataFrame:
    return dl.get_price(symbol, start=start)


@st.cache_data(ttl=config.NEWS_CACHE_TTL_SEC, show_spinner="뉴스 불러오는 중...")
def _news(code: str, n: int) -> pd.DataFrame:
    return news.fetch_news_with_sentiment(code, n=n)


@st.cache_data(ttl=config.NEWS_CACHE_TTL_SEC, show_spinner="공시 불러오는 중...")
def _dart(code: str) -> pd.DataFrame:
    return dart.fetch_disclosures(code)


def _safe_predict(price_df: pd.DataFrame, horizon: int, sentiment_hist: pd.Series | None) -> dict:
    """train_and_predict()가 예기치 못한 예외를 던져도 페이지 전체가 죽지 않도록 감싼다."""
    try:
        return predictor.train_and_predict(price_df, horizon=horizon, sentiment_hist=sentiment_hist)
    except Exception as e:
        return {"error": f"예측 중 오류가 발생했습니다: {e}"}


def _safe_predict_advanced(
    price_df: pd.DataFrame, horizon: int, sentiment_hist_full: pd.DataFrame | None
) -> dict:
    """train_and_predict_advanced()용 _safe_predict() 짝 — 심층 학습 버튼도 동일하게 보호한다."""
    try:
        return predictor.train_and_predict_advanced(
            price_df, horizon=horizon, sentiment_hist_full=sentiment_hist_full
        )
    except Exception as e:
        return {"error": f"예측 중 오류가 발생했습니다: {e}"}


# ------------------------------------------------------------------ 컴팩트 레이아웃용 전역 CSS
# 네 영역(주가요약·종목상세·가격예측·뉴스&공시) 모두 같은 폰트 크기를 쓴다 — 영역별로 다른
# 크기를 주지 않고 아래 규칙 하나로 전부 통일한다.
st.markdown(
    """
<style>
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
h1, h2, h3, h4, h5 {margin-top: 0.1rem; margin-bottom: 0.3rem;}
div[data-testid="stVerticalBlock"] {gap: 0.45rem;}
[data-testid="stMetricValue"] {font-size: 1.25rem;}
[data-testid="stMetricLabel"] {font-size: 0.88rem;}
[data-testid="stMetricDelta"] {font-size: 0.88rem;}
.stTabs [data-baseweb="tab-list"] {gap: 4px;}
.stTabs [data-baseweb="tab"] {padding: 4px 10px; font-size: 0.92rem;}
div[data-testid="stWidgetLabel"] p {font-size: 0.88rem; margin-bottom: 0.1rem;}
.stButton button {padding: 0.25rem 0.7rem; font-size: 0.88rem;}
hr {margin: 0.5rem 0;}
p, .stCaption, .stMarkdown, label, span {font-size: 0.92rem;}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown("##### 📊 주가 스크리닝 대시보드")

_SENT_COLOR = {"긍정": UP_COLOR, "중립": FLAT_COLOR, "부정": DOWN_COLOR}  # 상승=빨강, 하락=파랑

left_col, right_col = st.columns([3, 7], gap="medium")

# ==================================================================== 좌측 (30%): 주가 요약 + 가격 예측
# 위·아래 두 쌍(주가요약↔종목상세, 가격예측↔뉴스&공시)의 시작 줄을 픽셀 단위로 정확히
# 맞추려면, 한쪽 내용 높이를 추정해 다른 쪽에 맞추는 방식(근사치)으론 미세한 오차가 남는다.
# 대신 각 쌍의 두 박스에 "동일한" 고정 높이를 줘서 애초에 오차가 생길 수 없게 한다.
_TOP_ROW_HEIGHT = 800  # 종목상세(우측 상단) 내용이 다 들어가고도 여유가 있도록 넉넉히 잡은 값
_BOTTOM_ROW_HEIGHT = 380  # 가격예측 기준 — 뉴스&공시도 동일하게 맞춘다

with left_col, st.container(key="summary", border=True, height=_TOP_ROW_HEIGHT):
    _section_title("📊 주가 요약")

    # ---------------------------------------------------------- 공통 필터 (구 사이드바를 컴팩트한 행으로)
    fc1, fc2 = st.columns(2)
    with fc1:
        market = st.selectbox(
            "시장",
            ["ALL", "KOSPI", "KOSDAQ"],
            format_func=lambda m: {"ALL": "전체", "KOSPI": "KOSPI", "KOSDAQ": "KOSDAQ"}[m],
        )
    with fc2:
        top_n = st.selectbox("표시개수", [10, 20, 30, 50, 100], index=2)

    fc3, fc4, fc5 = st.columns([1.3, 1.3, 0.6])
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
        st.markdown("<div style='height:1.55em'></div>", unsafe_allow_html=True)
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

    def _render_table(ranked: pd.DataFrame, key: str, value_col: str, value_label: str, fmt: str, scale: float = 1.0):
        """스크리닝 결과를 컴팩트 3열(종목명·종가·값)로 그리고 선택 이벤트를 돌려준다."""
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
            height=min(30 * (len(display) + 1), 260),
            column_config=col_config,
            on_select="rerun",
            selection_mode="single-row",
            key=key,
        )

    rise_tab, amount_tab, marcap_tab = st.tabs(["📈 상승률", "💰 거래대금", "🏢 시가총액"])
    picks = []  # (탭 key, 정렬된 df, 선택 이벤트)

    with rise_tab:
        # 등락률은 '어느 기간을 볼지 / 오름·내림 중 무엇을 볼지'가 의미가 있어서 이 탭에만 컨트롤을 둔다.
        c1, c2 = st.columns(2)
        with c1:
            basis = st.radio("기준", ["일간", "주간"], horizontal=True)
        with c2:
            direction = st.radio("방향", ["상승", "하락"], horizontal=True)
        basis_col = "DailyChangeRatio" if basis == "일간" else "WeeklyChangeRatio"
        ranked_rise = screener.top_movers(universe, by=basis_col, n=top_n, ascending=(direction == "하락"))
        st.caption(f"{market_label} {len(universe):,}종목 중 {basis} {direction}률 상위 {len(ranked_rise)}개")
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

    # 세 테이블의 선택 상태는 탭을 옮겨도 각자 남아 있다. 매 rerun마다 전부 "선택됨"으로 보이므로
    # 그대로 반영하면 마지막 탭이 항상 이겨버린다 — 직전에 처리한 값과 달라진 탭만 반영한다.
    for tbl_key, ranked_df, event in picks:
        if not event.selection.rows:
            continue
        picked = ranked_df.iloc[event.selection.rows[0]]
        if st.session_state.get(f"_last_{tbl_key}") != picked["Code"]:
            st.session_state[f"_last_{tbl_key}"] = picked["Code"]
            st.session_state["selected_code"] = picked["Code"]
            st.session_state["selected_name"] = picked["Name"]

# ==================================================================== 우측 (70%): 종목 상세
with right_col:
    with st.container(key="detail", border=True, height=_TOP_ROW_HEIGHT):
        _detail_title = "📈 종목 상세"
        if st.session_state.get("selected_code"):
            _detail_title += f" · {st.session_state.get('selected_name', '')}"
        _section_title(_detail_title)

        search_col, period_col, idx_col = st.columns([2, 1, 2])
        with search_col:
            manual = st.text_input("종목코드/이름 검색", value="", placeholder="예: 005930, 삼성전자")
        with period_col:
            period_label = st.selectbox("조회 기간", list(DATE_RANGES.keys()), index=3)
        with idx_col:
            idx_sel = st.multiselect("지수 비교", list(config.INDICES.keys()), default=["KOSPI"])

        selected_code = st.session_state.get("selected_code")
        selected_name = st.session_state.get("selected_name", "")

        if manual.strip():
            hits = dl.find_symbol(manual.strip())
            code_col = "Code" if "Code" in hits.columns else "Symbol"
            if not hits.empty:
                options = {f"{row[code_col]} · {row['Name']}": row[code_col] for _, row in hits.head(20).iterrows()}
                pick = st.selectbox("검색 결과", list(options.keys()))
                selected_code = options[pick]
                selected_name = pick.split(" · ", 1)[1]
            else:
                st.warning("검색 결과가 없습니다.")

        if not selected_code:
            selected_code, selected_name = "005930", "삼성전자"

        days_back = DATE_RANGES[period_label]
        start_date = (
            config.DEFAULT_START
            if days_back is None
            else (pd.Timestamp.today() - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d")
        )

        price_df = _price(selected_code, start_date)
        if price_df.empty:
            st.error(f"'{selected_code}' 가격 데이터를 찾을 수 없습니다.")
            st.stop()

        # ------------------------------------------------------ 시세 요약 (원 단위)
        _last = price_df.iloc[-1]
        _last_date = price_df.index[-1]
        _close = float(_last["Close"])
        _prev_close = float(price_df["Close"].iloc[-2]) if len(price_df) >= 2 else None

        # 거래대금·시가총액은 KRX 스냅샷(universe)에 정확한 값이 있다 (거래대금은 체결가마다 다르므로
        # 종가×거래량으로는 정확히 못 구한다). 필터에 걸려 빠졌거나 해외 종목이면 없으므로 그때만 근사한다.
        _urow = universe[universe["Code"] == selected_code]
        _amount_approx = _urow.empty
        _amount_stale = False
        if not _urow.empty:
            _amount, _marcap = float(_urow.iloc[0]["Amount"]), float(_urow.iloc[0]["Marcap"])
            _snap_vol = float(_urow.iloc[0]["Volume"])
            _live_vol = float(_last["Volume"])
            _amount_stale = _live_vol > 0 and abs(_snap_vol - _live_vol) / _live_vol > 0.01
        else:
            _amount, _marcap = _close * float(_last["Volume"]), None

        # 시세 요약 8개 항목을 한 줄로 (기존 2줄 → 1줄로 압축, 아낀 세로 공간은 아래 차트를 키우는 데 쓴다)
        _r1 = st.columns(8)
        _r1[0].metric("현재가", f"{_close:,.0f}원")
        if _prev_close:
            _diff = _close - _prev_close
            # 실제 체결된 시세이므로 캔들 차트와 같은 원색을 쓴다 (예측값만 파스텔).
            _r1[0].markdown(_change_html(_diff, _diff / _prev_close * 100), unsafe_allow_html=True)
        _r1[1].metric("시가", f"{float(_last['Open']):,.0f}원")
        _r1[2].metric("고가", f"{float(_last['High']):,.0f}원")
        _r1[3].metric("저가", f"{float(_last['Low']):,.0f}원")
        _r1[4].metric("전일종가", f"{_prev_close:,.0f}원" if _prev_close else "—")
        _r1[5].metric("변동폭", f"{float(_last['High']) - float(_last['Low']):,.0f}원")
        _r1[6].metric("거래량", f"{float(_last['Volume']):,.0f}주")
        _r1[7].metric("거래대금", f"{_amount / 1e8:,.0f}억원")

        _notes = [f"{_last_date:%Y-%m-%d (%a)} 기준"]
        if _marcap:
            _notes.append(f"시총 {_marcap / 1e8:,.0f}억원")
        if _amount_approx:
            _notes.append("거래대금 추정치")
        elif _amount_stale:
            _notes.append("거래대금·시총은 KRX 스냅샷 기준")
        _notes.append("상승=빨강 / 하락=파랑")
        st.caption(" · ".join(_notes))

        enriched = ind.add_all(price_df)

        # 지표 체크박스: 세로 사이드가 아니라 차트 위 가로 한 줄로 (공간 절약)
        ic = st.columns(8)
        show_sma5 = ic[0].checkbox("SMA5", value=False)
        show_sma20 = ic[1].checkbox("SMA20", value=True)
        show_sma60 = ic[2].checkbox("SMA60", value=True)
        show_sma120 = ic[3].checkbox("SMA120", value=False)
        show_bb = ic[4].checkbox("볼밴드", value=False)
        show_vol = ic[5].checkbox("거래량", value=True)
        show_rsi = ic[6].checkbox("RSI", value=False)
        show_macd = ic[7].checkbox("MACD", value=False)

        sma_windows = [w for w, on in [(5, show_sma5), (20, show_sma20), (60, show_sma60), (120, show_sma120)] if on]
        for w in sma_windows:
            col = f"sma{w}"
            if col not in enriched.columns:
                enriched[col] = ind.sma(enriched["Close"], w)

        overlays = {}
        for label in idx_sel:
            idx_df = _price(label, start_date)
            if not idx_df.empty:
                overlays[label] = idx_df["Close"]

        fig = charts.build_chart(
            enriched,
            title=f"{selected_name} ({selected_code})",
            sma_windows=tuple(sma_windows),
            show_bollinger=show_bb,
            show_volume=show_vol,
            show_rsi=show_rsi,
            show_macd=show_macd,
            index_overlays=overlays or None,
            base_height=320,
            panel_height=85,
        )
        st.plotly_chart(fig, width="stretch")

        # ------------------------------------------------------ 성과 요약
        summary = ind.summary(price_df["Close"])
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("수익률", f"{summary['total_return'] * 100:.1f}%")
        m2.metric("CAGR", f"{summary['cagr'] * 100:.1f}%")
        m3.metric("변동성", f"{summary['volatility'] * 100:.1f}%")
        m4.metric("샤프", f"{summary['sharpe']:.2f}")
        m5.metric("MDD", f"{summary['max_drawdown'] * 100:.1f}%")

# ==================================================================== 좌측 하단: 가격 예측
with left_col, st.container(key="predict", border=True, height=_BOTTOM_ROW_HEIGHT):
    _section_title("💹 가격 예측")
    st.caption("⚠️ 참고용 추정치이며 투자 조언이 아닙니다.")

    # 예측 피처로 쓸 뉴스 감성 히스토리를 먼저 갱신한다 (조회 실패해도 페이지 전체가 죽지 않도록 방어).
    try:
        news.log_sentiment_from_news(selected_code, _news(selected_code, 10))
    except Exception:
        pass
    sentiment_hist = news.sentiment_history(selected_code)

    pred_1d = _safe_predict(price_df, horizon=1, sentiment_hist=sentiment_hist)
    pred_5d = _safe_predict(price_df, horizon=5, sentiment_hist=sentiment_hist)

    pcol1, pcol2 = st.columns(2)
    for col, pred, label in [
        (pcol1, pred_1d, "다음 거래일"),
        (pcol2, pred_5d, "5거래일 후"),
    ]:
        with col:
            st.markdown(f"**{label}**")
            if "error" in pred:
                st.info(pred["error"])
                continue
            delta = pred["predicted_price"] - pred["last_close"]
            st.metric(pred["target_date"].strftime("%m-%d(%a)"), f"{pred['predicted_price']:,.0f}원")
            st.markdown(_change_html(delta, pred["predicted_return"] * 100, soft=True), unsafe_allow_html=True)
            st.caption(
                f"검증{pred['n_test']}일 · MAE {pred['mae']:,.0f}원 · "
                f"MAPE {pred['mape'] * 100:.1f}% · 방향적중 {pred['directional_accuracy'] * 100:.0f}%"
            )

    if st.button(
        "🎯 정확한 예측 (심층 학습)",
        help=(
            "Ridge/RandomForest/GradientBoosting을 시계열 교차검증으로 비교해 가장 좋은 모델로 다시 "
            "예측합니다. 조회 기간과 무관하게 보유한 전체 기간 데이터를 쓰며, 기본 예측보다 시간이 더 걸립니다."
        ),
    ):
        with st.spinner("여러 모델을 교차검증하며 심층 학습하는 중..."):
            full_price_df = _price(selected_code, config.DEFAULT_START)
            sentiment_hist_full = news.sentiment_history_full(selected_code)
            st.session_state["adv_pred"] = {
                "code": selected_code,
                "1d": _safe_predict_advanced(
                    full_price_df, horizon=1, sentiment_hist_full=sentiment_hist_full
                ),
                "5d": _safe_predict_advanced(
                    full_price_df, horizon=5, sentiment_hist_full=sentiment_hist_full
                ),
            }

    adv_pred = st.session_state.get("adv_pred")
    if adv_pred and adv_pred["code"] == selected_code:
        st.markdown("**🎯 심층 학습 결과**")
        acol1, acol2 = st.columns(2)
        for col, pred, label in [
            (acol1, adv_pred["1d"], "다음 거래일"),
            (acol2, adv_pred["5d"], "5거래일 후"),
        ]:
            with col:
                st.markdown(f"**{label}**")
                if "error" in pred:
                    st.info(pred["error"])
                    continue
                delta = pred["predicted_price"] - pred["last_close"]
                st.metric(pred["target_date"].strftime("%m-%d(%a)"), f"{pred['predicted_price']:,.0f}원")
                st.markdown(_change_html(delta, pred["predicted_return"] * 100, soft=True), unsafe_allow_html=True)
                st.caption(
                    f"{pred['best_model']} · 검증{pred['n_holdout']}일 · MAE {pred['mae']:,.0f}원 · "
                    f"MAPE {pred['mape'] * 100:.1f}% · 방향적중 {pred['directional_accuracy'] * 100:.0f}%"
                )

        with st.expander("심층 모델 상세 (교차검증 비교 · 피처 영향도)"):
            for pred, label in [(adv_pred["1d"], "다음 거래일 모델"), (adv_pred["5d"], "5거래일 후 모델")]:
                if "error" in pred:
                    continue
                st.markdown(
                    f"**{label}** — 선정: {pred['best_model']} · 학습 {pred['n_train']}행 / "
                    f"홀드아웃 {pred['n_holdout']}행 · 뉴스 감성 히스토리 {pred['news_days']}일 누적"
                )
                cv_df = pd.DataFrame(
                    {"모델": list(pred["cv_scores"].keys()), "교차검증 MAE(수익률)": list(pred["cv_scores"].values())}
                )
                st.dataframe(cv_df, hide_index=True, width="stretch")
                st.dataframe(
                    pred["feature_importance"].rename(columns={"label": "설명", "coef": "중요도"})[["설명", "중요도"]],
                    hide_index=True,
                    width="stretch",
                )
            st.caption(
                "교차검증 점수는 모델을 고르는 데만 쓰였고, 위 정확도는 모델 선정에 관여하지 않은 "
                "마지막 홀드아웃 구간 기준입니다. 중요도 값은 모델별로 계산 방식이 다릅니다"
                "(회귀계수 / 트리 중요도 / 순열 중요도)."
            )
    else:
        st.caption("클릭 시 여러 모델을 교차검증으로 비교해 보유한 전체 기간 데이터로 다시 예측합니다.")

    with st.expander("모델 상세 (피처 영향도)"):
        for pred, label in [(pred_1d, "다음 거래일 모델"), (pred_5d, "5거래일 후 모델")]:
            if "error" in pred:
                continue
            st.markdown(
                f"**{label}** — 학습 {pred['n_train']}행 / 검증 {pred['n_test']}행 · "
                f"뉴스 감성 히스토리 {pred['news_days']}일 누적"
            )
            st.dataframe(
                pred["feature_importance"].rename(columns={"label": "설명", "coef": "회귀계수"})[["설명", "회귀계수"]],
                hide_index=True,
                width="stretch",
            )
        st.caption("회귀계수가 클수록(절대값 기준) 예측에 미치는 영향이 큽니다. 뉴스 감성은 히스토리가 쌓일수록 값이 유의미해집니다.")

# ==================================================================== 우측 하단: 뉴스 & 공시
with right_col:
    with st.container(key="news", border=True, height=_BOTTOM_ROW_HEIGHT):
        _section_title("📰 뉴스 & 공시")

        # 탭 선택과 "표시개수"를 같은 줄에 — 예전엔 뉴스 탭 안에 별도 줄로 있어서 세로를 더 먹었다.
        tabs_col, count_col = st.columns([4, 1])
        with tabs_col:
            news_tab, dart_tab = st.tabs(["📰 뉴스", "📋 공시 (DART)"])
        with count_col:
            news_n = st.selectbox(
                "표시개수",
                [5, 10, 15, 20],
                index=1,
                key="news_n",
                label_visibility="collapsed",
                help="뉴스 표시 개수",
            )

        with news_tab:
            news_df = _news(selected_code, news_n)

            if news_df.empty:
                st.info("최근 뉴스를 찾지 못했습니다.")
            else:
                with st.container(height=260):
                    for _, row in news_df.iterrows():
                        with st.container(border=True):
                            left, right = st.columns([6, 1])
                            with left:
                                title_safe = html.escape(row["title"])
                                url_safe = html.escape(row["url"], quote=True)
                                meta_safe = html.escape(f"{row['press']} · {row['date']}")
                                summary_safe = html.escape(row["summary"])
                                # 제목·출처/날짜를 한 줄에 두고 출처/날짜는 우측 정렬, 요약은 한 줄로 말줄임.
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

        with dart_tab:
            try:
                dart_df = _dart(selected_code)
            except (dart.DartKeyMissing, dart.DartUnavailable) as e:
                st.info(str(e))
            else:
                if dart_df.empty:
                    st.info("최근 90일 내 공시가 없습니다.")
                else:
                    st.dataframe(
                        dart_df.rename(columns={"rcept_dt": "접수일", "report_nm": "보고서명", "flr_nm": "제출인"}),
                        column_config={"url": st.column_config.LinkColumn("링크", display_text="열기")},
                        hide_index=True,
                        height=260,
                        width="stretch",
                    )
