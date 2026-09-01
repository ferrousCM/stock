"""모의투자 성과 리포팅 — `pages/모의투자.py`(기본 모델/전략 모델 펼치기 UI)가 공유하는
순수 렌더 함수들. PRD.md 5.6·10장 8단계·11장 3단계 참고.

**읽기 전용.** 매매를 유발하는 코드를 단 한 줄도 포함하지 않는다 — `trading_agent`/
`trading_agent_v2`나 `portfolio`의 쓰기 함수(`apply_trade`/`save_daily_result`)는
import조차 하지 않는다. 실행(매매)은 `scripts/run_daily_trading.py` + GitHub Actions가
전담하고, 여기는 그 결과(원장)만 봇 레지스트리·원장 디렉터리 함수를 받아 보여준다.

**기본 모델/전략 모델 통합**: 원래 v1(`BOT_STRATEGIES`)과 v2(`BOT_STRATEGIES_V2`)는 각각
별도 페이지(`pages/모의투자.py`/`pages/모의투자_v2.py`)였다. 사이드바를 단순하게 유지하려고
한 페이지("모의투자")로 합치고, `st.expander`로 "기본 모델"/"전략 모델"을 펼치기/숨기기
하도록 바꿨다 — 그래서 머리말(`render_header`)과 모델별 탭 묶음(`render_body`)을 분리했다.
`render_body`는 `(bot_id, label, portfolio_dir)` 등 인자만 받는 순수 렌더 함수이고,
`key_prefix`로 두 모델이 같은 bot_id("default" 등)를 쓰더라도 위젯 key가 겹치지 않게 한다.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, portfolio, predictor, screener, theme
from src import data_loader as dl
from src import indicators as ind

PORTFOLIO_COLOR = theme.BOT_COLORS["default"]  # 봇 자체 대시보드의 자산 라인 기본색(기본형 색 재사용)
KOSPI_COLOR = theme.KOSPI_COLOR  # 코스피 비교 라인
UP_COLOR, DOWN_COLOR, FLAT_COLOR = theme.UP_COLOR, theme.DOWN_COLOR, theme.FLAT_COLOR
UP_SOFT, DOWN_SOFT, FLAT_SOFT = theme.UP_SOFT, theme.DOWN_SOFT, theme.FLAT_SOFT

_BOT_COMPARE_COLOR = theme.BOT_COLORS


def _change_html(diff: float, pct: float) -> str:
    color = UP_COLOR if diff > 0 else DOWN_COLOR if diff < 0 else FLAT_COLOR
    arrow = "▲" if diff > 0 else "▼" if diff < 0 else "―"
    return f"<span style='color:{color};font-weight:600'>{arrow} {diff:+,.0f}원 ({pct:+.2f}%)</span>"


@st.cache_data(ttl=config.CACHE_TTL_SEC, show_spinner="보유종목 현재가 불러오는 중...")
def _current_prices() -> dict[str, float]:
    snapshot = screener.screen(market="ALL")
    return dict(zip(snapshot["Code"], snapshot["Close"], strict=True))


@st.cache_data(ttl=config.CACHE_TTL_SEC, show_spinner="시세 불러오는 중...")
def _price(code: str) -> pd.DataFrame:
    return dl.get_price(code)


def _trades_show_df(df: pd.DataFrame) -> pd.DataFrame:
    """trades.csv 행들을 화면 표시용 컬럼(한국어 라벨 + 거래총액)으로 변환한다."""
    out = df.copy()
    out["action"] = out["action"].map({"buy": "매수", "sell": "매도"})
    return out[["date", "name", "code", "action", "quantity", "price", "amount", "reason"]].rename(
        columns={
            "date": "날짜",
            "name": "종목명",
            "code": "종목코드",
            "action": "구분",
            "quantity": "수량",
            "price": "체결가",
            "amount": "거래총액",
            "reason": "판단 근거",
        }
    )


def _color_action(val: str) -> str:
    if val == "매수":
        return f"color:{UP_SOFT};font-weight:600"
    if val == "매도":
        return f"color:{DOWN_SOFT};font-weight:600"
    return ""


def _bold(_val: object) -> str:
    return "font-weight:700"


_TRADES_COLUMN_CONFIG = {
    "체결가": st.column_config.NumberColumn(format="%,.0f원"),
    "거래총액": st.column_config.NumberColumn(format="%,.0f원"),
}


@st.dialog("종목 상세", width="large")
def _stock_detail_dialog(code: str, name: str, all_trades: pd.DataFrame) -> None:
    st.markdown(f"### {name} ({code})")
    st.caption("⚠️ 참고용 통계 모델이며 투자 조언이 아닙니다.")

    price_df = _price(code)
    if price_df.empty:
        st.info("가격 데이터를 불러오지 못했습니다.")
    else:
        close = float(price_df["Close"].iloc[-1])
        prev_close = float(price_df["Close"].iloc[-2]) if len(price_df) >= 2 else None

        pcol, chartcol = st.columns([1, 2])
        with pcol:
            st.metric("현재가", f"{close:,.0f}원")
            if prev_close:
                diff = close - prev_close
                st.markdown(_change_html(diff, diff / prev_close * 100), unsafe_allow_html=True)
        with chartcol:
            recent = price_df.tail(120)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=recent.index, y=recent["Close"], line=dict(width=1.8, color=PORTFOLIO_COLOR))
            )
            fig.update_layout(height=160, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig, width="stretch")

        st.markdown("**예측 종가**")
        p1, p2 = st.columns(2)
        for col, horizon, label in [(p1, 1, "다음 거래일"), (p2, 5, "5거래일 후")]:
            with col:
                try:
                    pred = predictor.train_and_predict(price_df, horizon=horizon, sentiment_hist=None)
                except Exception as e:
                    pred = {"error": f"예측 중 오류: {e}"}
                if "error" in pred:
                    st.caption(pred["error"])
                    continue
                pdiff = pred["predicted_price"] - pred["last_close"]
                st.metric(label, f"{pred['predicted_price']:,.0f}원")
                pcolor = UP_SOFT if pdiff > 0 else DOWN_SOFT if pdiff < 0 else FLAT_SOFT
                parrow = "▲" if pdiff > 0 else "▼" if pdiff < 0 else "―"
                st.markdown(
                    f"<span style='color:{pcolor};font-weight:600'>{parrow} {pdiff:+,.0f}원 "
                    f"({pred['predicted_return'] * 100:+.2f}%)</span>",
                    unsafe_allow_html=True,
                )
                st.caption(f"방향적중 {pred['directional_accuracy'] * 100:.0f}%")

    st.divider()
    st.markdown(f"**{name} 거래 내역**")
    stock_trades = all_trades[all_trades["code"] == code].sort_values("date", ascending=False)
    if stock_trades.empty:
        st.caption("이 종목의 거래 이력이 없습니다.")
    else:
        st.dataframe(
            _trades_show_df(stock_trades)
            .style.map(_bold, subset=["종목명"])
            .map(_color_action, subset=["구분"]),
            column_config=_TRADES_COLUMN_CONFIG,
            hide_index=True,
            width="stretch",
            height=200,
        )


def _render_bot_dashboard(bot_id: str, label: str, portfolio_dir: Path, *, key_prefix: str = "") -> dict:
    """봇 하나의 전체 리포트(요약 지표·자산 추이·보유종목·거래내역·거래 요약)를 그리고,
    "성과 비교" 탭에서 쓸 요약 dict를 반환한다. 원장을 읽기만 한다.

    key_prefix: 위젯 key 접두사. v1/v2 두 봇 레지스트리가 같은 bot_id("default" 등)를
    재사용하므로, 한 페이지에 v1/v2를 같이 그릴 때(모의투자 탭의 기본/전략 모델 펼치기)
    key 충돌을 막으려고 호출부가 "v1_"/"v2_" 같은 접두사를 넘긴다."""
    state = portfolio.get_state(portfolio_dir)
    holdings = portfolio.get_holdings(portfolio_dir)
    trades = portfolio.get_trades(portfolio_dir)
    equity_hist = portfolio.get_equity_history(portfolio_dir)

    if holdings.empty:
        holdings_mtm, holdings_value = (
            holdings.assign(
                current_price=pd.Series(dtype=float),
                market_value=pd.Series(dtype=float),
                unrealized_pnl=pd.Series(dtype=float),
                price_is_stale=pd.Series(dtype=bool),
            ),
            0.0,
        )
    else:
        try:
            prices = _current_prices()
        except Exception:
            prices = {}
        holdings_mtm, holdings_value = portfolio.mark_to_market(holdings, prices)

    total_equity = state["cash"] + holdings_value
    last_run = state["last_run_date"]

    # ================================================================ 상단 요약
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("총자산", f"{total_equity:,.0f}원")
    m1.caption(f"현금 {state['cash']:,.0f}원 + 주식평가금액 {holdings_value:,.0f}원")

    cum_return_pct = (total_equity / portfolio.INITIAL_CASH - 1) * 100
    m2.metric("누적 수익률 (원금 대비)", f"{cum_return_pct:+.2f}%")

    excess_pct = None
    if len(equity_hist) >= 2 and equity_hist["kospi_close"].notna().sum() >= 2:
        kospi_valid = equity_hist.dropna(subset=["kospi_close"])
        port_return = equity_hist["total_equity"].iloc[-1] / equity_hist["total_equity"].iloc[0] - 1
        kospi_return = kospi_valid["kospi_close"].iloc[-1] / kospi_valid["kospi_close"].iloc[0] - 1
        excess_pct = (port_return - kospi_return) * 100
        m3.metric("코스피 대비 초과수익률", f"{excess_pct:+.2f}%")
    else:
        m3.metric("코스피 대비 초과수익률", "―")

    m4.metric("마지막 매매 기준일", last_run if last_run else "아직 없음")

    if last_run is None:
        st.info(
            f"{label} 봇은 아직 첫 매매가 실행되지 않았습니다. GitHub Actions가 처음 실행되면 "
            "이 화면에 자산 추이·보유종목·거래내역이 표시됩니다."
        )

    st.divider()

    # ================================================================ 자산 추이 차트
    st.subheader("📈 자산 추이 (vs 코스피)")
    if len(equity_hist) < 2:
        st.caption("아직 비교할 만큼 자산 추이 기록이 쌓이지 않았습니다 (최소 2거래일 필요).")
    else:
        eh = equity_hist.copy()
        eh["date"] = pd.to_datetime(eh["date"])

        port_base = eh["total_equity"].iloc[0]
        port_idx = eh["total_equity"] / port_base * 100

        kospi_filled = eh["kospi_close"].ffill().bfill()
        kospi_base = kospi_filled.iloc[0]
        kospi_idx = kospi_filled / kospi_base * 100 if kospi_base and pd.notna(kospi_base) else None

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=eh["date"],
                y=port_idx,
                name=label,
                line=dict(width=2.4, color=_BOT_COMPARE_COLOR.get(bot_id, PORTFOLIO_COLOR)),
            )
        )
        if kospi_idx is not None:
            fig.add_trace(
                go.Scatter(
                    x=eh["date"],
                    y=kospi_idx,
                    name="코스피",
                    line=dict(width=1.6, color=KOSPI_COLOR, dash="dot"),
                )
            )
        fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis_title="시작일=100 기준 지수",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption("두 선 모두 최초 기록일 값을 100으로 맞춘 상대 비교(리베이스)입니다.")

    st.divider()

    # ================================================================ 보유종목
    st.subheader("📦 보유종목")
    if holdings_mtm.empty:
        st.caption("보유 중인 종목이 없습니다.")
    else:
        st.caption("행을 클릭하면 해당 종목의 시세·예측·거래 이력이 팝업으로 열립니다.")
        display = holdings_mtm.copy()
        display["평가손익률"] = (display["current_price"] / display["avg_price"] - 1) * 100
        display["비중"] = display["market_value"] / total_equity * 100 if total_equity else 0.0
        if display["price_is_stale"].any():
            st.caption("⚠️ 표시에 ※가 붙은 종목은 오늘 종가 조회에 실패해 평단가로 대체 평가한 값입니다.")
        display["종목명"] = display["name"] + display["price_is_stale"].map({True: " ※", False: ""})

        show_holdings = display[
            ["종목명", "code", "quantity", "avg_price", "current_price", "market_value", "평가손익률", "비중"]
        ].rename(
            columns={
                "code": "종목코드",
                "quantity": "수량",
                "avg_price": "평단가",
                "current_price": "현재가",
                "market_value": "보유총액",
            }
        )
        holdings_event = st.dataframe(
            show_holdings.style.map(_bold, subset=["종목명"]),
            column_config={
                "평단가": st.column_config.NumberColumn(format="%,.0f원"),
                "현재가": st.column_config.NumberColumn(format="%,.0f원"),
                "보유총액": st.column_config.NumberColumn(format="%,.0f원"),
                "평가손익률": st.column_config.NumberColumn(format="%+.2f%%"),
                "비중": st.column_config.NumberColumn(format="%.1f%%"),
            },
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            key=f"holdings_table_{key_prefix}{bot_id}",
        )
        holdings_selected = holdings_event.selection.rows if hasattr(holdings_event, "selection") else []
        if holdings_selected:
            sel_holding = display.iloc[holdings_selected[0]]
            _stock_detail_dialog(sel_holding["code"], sel_holding["name"], trades)

    st.divider()

    # ================================================================ 거래 내역
    st.subheader("📜 거래 내역")
    st.caption(
        "체결가는 그날(장 마감 후 확정된) 종가 기준으로 가정한 시뮬레이션 값입니다. "
        "행을 클릭하면 해당 종목의 시세·예측·거래 이력이 팝업으로 열립니다."
    )
    if trades.empty:
        st.caption("거래 이력이 없습니다.")
    else:
        trades_all = trades.copy()
        trades_all["date"] = pd.to_datetime(trades_all["date"])

        fcol1, fcol2 = st.columns([1, 1])
        with fcol1:
            min_d, max_d = trades_all["date"].min().date(), trades_all["date"].max().date()
            date_range = st.date_input(
                "기간", value=(min_d, max_d), min_value=min_d, max_value=max_d, key=f"date_range_{key_prefix}{bot_id}"
            )
        with fcol2:
            query = st.text_input(
                "종목명 또는 종목코드 검색", placeholder="예: 삼성전자 또는 005930", key=f"query_{key_prefix}{bot_id}"
            )

        filtered = trades_all
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = date_range
            filtered = filtered[(filtered["date"].dt.date >= start) & (filtered["date"].dt.date <= end)]
        elif isinstance(date_range, tuple) and len(date_range) == 1:
            filtered = filtered[filtered["date"].dt.date == date_range[0]]
        if query.strip():
            q = query.strip()
            filtered = filtered[
                filtered["name"].str.contains(q, case=False, na=False)
                | filtered["code"].str.contains(q, case=False, na=False)
            ]

        filtered = filtered.sort_values("date", ascending=False).reset_index(drop=True)
        filtered["date"] = filtered["date"].dt.strftime("%Y-%m-%d")

        if filtered.empty:
            st.caption("조건에 맞는 거래 내역이 없습니다.")
        else:
            event = st.dataframe(
                _trades_show_df(filtered)
                .style.map(_bold, subset=["종목명"])
                .map(_color_action, subset=["구분"]),
                column_config=_TRADES_COLUMN_CONFIG,
                hide_index=True,
                width="stretch",
                height=360,  # 계속 쌓일 이력 대비 고정 높이 — 넘치면 표 안에서 스크롤
                on_select="rerun",
                selection_mode="single-row",
                key=f"trades_table_{key_prefix}{bot_id}",
            )
            selected_rows = event.selection.rows if hasattr(event, "selection") else []
            if selected_rows:
                sel = filtered.iloc[selected_rows[0]]
                _stock_detail_dialog(sel["code"], sel["name"], trades)
        st.caption("판단 근거는 정해진 규칙(임계값)에 따른 자동 판단 서술일 뿐, 종목 추천이 아닙니다.")

    st.divider()

    # ================================================================ 거래 요약
    st.subheader("🧾 거래 요약")
    if trades.empty or equity_hist.empty:
        st.caption("아직 거래 이력이 없습니다.")
    else:
        daily_counts = (
            trades.groupby("date")["action"]
            .value_counts()
            .unstack(fill_value=0)
            .reindex(columns=["buy", "sell"], fill_value=0)
            .rename(columns={"buy": "매수", "sell": "매도"})
            .reset_index()
        )

        eq = equity_hist[["date", "cash", "holdings_value", "total_equity"]].copy()
        # 첫 거래일은 diff()/pct_change()가 NaN을 주는데, NumberColumn은 NaN도 빈 칸이 아니라
        # 문자열 "None"으로 그대로 보여준다(_pct_or_dash 참고) — 여기서 직접 문자열로 포맷한다.
        eq["전일대비"] = eq["total_equity"].diff().map(lambda v: "—" if pd.isna(v) else f"{v:+,.0f}원")
        eq["전일대비율"] = (eq["total_equity"].pct_change() * 100).map(_pct_or_dash)
        eq["시작대비"] = (eq["total_equity"] / portfolio.INITIAL_CASH - 1) * 100

        daily_summary = eq.merge(daily_counts, on="date", how="left")
        daily_summary["매수"] = daily_summary["매수"].fillna(0).astype(int)
        daily_summary["매도"] = daily_summary["매도"].fillna(0).astype(int)
        daily_summary = daily_summary.sort_values("date", ascending=False).rename(
            columns={
                "date": "날짜",
                "cash": "현금잔고",
                "holdings_value": "주식평가금액",
                "total_equity": "총자산",
            }
        )

        def _color_diff(val: float | str) -> str:
            """전일대비/전일대비율(문자열로 미리 포맷됨)과 시작대비(숫자) 둘 다 처리한다."""
            if isinstance(val, str):
                if val in ("—", ""):
                    return ""
                positive, negative = val.startswith("+"), val.startswith("-")
            else:
                if pd.isna(val) or val == 0:
                    return ""
                positive, negative = val > 0, val < 0
            if positive:
                return f"color:{UP_COLOR};font-weight:600"
            if negative:
                return f"color:{DOWN_COLOR};font-weight:600"
            return ""

        show_summary = daily_summary[
            [
                "날짜",
                "총자산",
                "매수",
                "매도",
                "전일대비",
                "전일대비율",
                "시작대비",
                "현금잔고",
                "주식평가금액",
            ]
        ]
        st.dataframe(
            show_summary.style.map(_color_diff, subset=["전일대비", "전일대비율", "시작대비"]),
            column_config={
                "총자산": st.column_config.NumberColumn(format="%,.0f원"),
                "시작대비": st.column_config.NumberColumn(format="%+.2f%%"),
                "현금잔고": st.column_config.NumberColumn(format="%,.0f원"),
                "주식평가금액": st.column_config.NumberColumn(format="%,.0f원"),
            },
            hide_index=True,
            width="stretch",
            height=280,
        )
        st.caption(
            "전일대비는 직전 거래일 대비, 시작대비는 시작 원금(1억원) 대비 총자산 변동입니다. "
            "매매가 0건인 날에도 보유종목 평가금액이 바뀌면 총자산이 변동될 수 있습니다."
        )

    return {
        "bot_id": bot_id,
        "label": label,
        "total_equity": total_equity,
        "cum_return_pct": cum_return_pct,
        "excess_pct": excess_pct,
        "last_run": last_run,
        "n_holdings": len(holdings_mtm),
        "n_trades": len(trades),
        "equity_hist": equity_hist,
    }


_MIN_RISK_STAT_DAYS = 20  # 연환산(CAGR/샤프)은 기간이 너무 짧으면 값이 비정상적으로 부풀려진다


def _pct_or_dash(v: float | None, *, signed: bool = True) -> str:
    """st.column_config.NumberColumn은 값이 None/NaN이면 빈 칸이 아니라 문자열 "None"을
    그대로 보여준다(이 Streamlit 버전의 동작) — 표에 "None"이 찍히지 않도록 여기서 직접
    포맷 문자열로 바꾸고, 없는 값은 "—"로 통일한다."""
    if v is None or pd.isna(v):
        return "—"
    return f"{v:+.2f}%" if signed else f"{v:.2f}%"


def _num_or_dash(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:.2f}"


def _risk_stats(equity_hist: pd.DataFrame) -> dict:
    """자산 추이만으로 계산 가능한 위험/수익 지표(CAGR·변동성·샤프지수·MDD, 문자열로 이미
    포맷됨) — 연율화 지표라 최소 _MIN_RISK_STAT_DAYS(약 한 달)치는 쌓여야 의미가 있다(예:
    3주치로 연율화하면 +-90% 같은 값이 나와 오히려 오해를 부른다). indicators.summary()가
    이미 종목 성과 요약에 쓰는 계산을 그대로 재사용한다."""
    if len(equity_hist) < 2:
        return {"cagr": "—", "volatility": "—", "sharpe": "—", "max_drawdown": "—"}
    eh = equity_hist.copy()
    eh["date"] = pd.to_datetime(eh["date"])
    stat = ind.summary(eh.set_index("date")["total_equity"])
    # MDD(관측된 낙폭 그 자체)는 기간과 무관하게 바로 의미가 있어 항상 보여준다 — CAGR/변동성/
    # 샤프지수만 연율화(annualize)된 값이라 기간이 짧으면 값이 부풀려져 최소 기간을 요구한다.
    enough_history = len(equity_hist) >= _MIN_RISK_STAT_DAYS
    return {
        "cagr": _pct_or_dash(stat["cagr"] * 100) if enough_history else "—",
        "volatility": _pct_or_dash(stat["volatility"] * 100, signed=False) if enough_history else "—",
        "sharpe": _num_or_dash(stat["sharpe"]) if enough_history else "—",
        "max_drawdown": _pct_or_dash(stat["max_drawdown"] * 100, signed=False),
    }


def _render_comparison(summaries: list[dict]) -> None:
    """세 봇의 요약 표 + 자산 추이 오버레이 차트. 각 봇 탭을 다 그린 뒤 호출되지만,
    st.tabs()가 반환하는 컨테이너는 코드 실행 순서와 무관하게 그 탭 위치에 렌더링되므로
    "성과 비교" 탭이 맨 왼쪽에 있어도 문제없다."""
    st.subheader("📊 봇별 성과 비교")
    st.caption(
        "세 봇은 같은 날 같은 시세·신호를 보고 서로 다른 판단 로직(기본형/공격적/모멘텀)으로 "
        "독립적으로 매매합니다. 봇 간 자금 이동은 없고, 각자 1억원으로 시작합니다."
    )

    table = pd.DataFrame(
        [
            {
                "봇": s["label"],
                "총자산": s["total_equity"],
                "누적수익률": s["cum_return_pct"],
                "코스피 대비 초과수익률": _pct_or_dash(s["excess_pct"]),
                **_risk_stats(s["equity_hist"]),
                "보유종목수": s["n_holdings"],
                "누적거래횟수": s["n_trades"],
                "마지막 매매 기준일": s["last_run"] if s["last_run"] else "아직 없음",
            }
            for s in summaries
        ]
    ).rename(
        columns={
            "cagr": "연환산 수익률",
            "volatility": "변동성",
            "sharpe": "샤프지수",
            "max_drawdown": "최대낙폭(MDD)",
        }
    )
    st.dataframe(
        table,
        column_config={
            "총자산": st.column_config.NumberColumn(format="%,.0f원"),
            "누적수익률": st.column_config.NumberColumn(format="%+.2f%%"),
        },
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"최대낙폭(MDD)은 자산 추이 기록이 2거래일만 쌓여도 계산되지만, 연환산 수익률(CAGR)·"
        f"변동성·샤프지수는 최소 {_MIN_RISK_STAT_DAYS}거래일치가 쌓여야 표시됩니다(기간이 "
        "너무 짧으면 연 단위로 환산했을 때 값이 실제보다 크게 부풀려집니다). "
        "샤프지수는 위험 대비 수익 — 높을수록, MDD는 고점 대비 최대 하락폭 — 0에 가까울수록 유리합니다."
    )

    plot_data = [s for s in summaries if len(s["equity_hist"]) >= 2]
    if not plot_data:
        st.caption("아직 비교할 만큼 자산 추이 기록이 쌓인 봇이 없습니다 (최소 2거래일 필요).")
        return

    fig = go.Figure()

    # 코스피 비교선은 kospi_close 기록이 가장 길게 쌓인 봇(보통 기본형)에서 가져온다 —
    # 같은 날짜라면 봇마다 같은 값이 기록되므로 어느 봇에서 가져와도 값은 같다.
    kospi_source = max(plot_data, key=lambda s: s["equity_hist"]["kospi_close"].notna().sum())
    kospi_eh = kospi_source["equity_hist"].copy()
    kospi_eh["date"] = pd.to_datetime(kospi_eh["date"])
    kospi_filled = kospi_eh["kospi_close"].ffill().bfill()
    if kospi_filled.notna().any() and pd.notna(kospi_filled.iloc[0]) and kospi_filled.iloc[0]:
        kospi_idx = kospi_filled / kospi_filled.iloc[0] * 100
        fig.add_trace(
            go.Scatter(
                x=kospi_eh["date"],
                y=kospi_idx,
                name="코스피",
                line=dict(width=1.6, color=KOSPI_COLOR, dash="dot"),
            )
        )

    for s in plot_data:
        eh = s["equity_hist"].copy()
        eh["date"] = pd.to_datetime(eh["date"])
        base = eh["total_equity"].iloc[0]
        idx = eh["total_equity"] / base * 100
        fig.add_trace(
            go.Scatter(
                x=eh["date"],
                y=idx,
                name=s["label"],
                line=dict(width=2.2, color=_BOT_COMPARE_COLOR.get(s["bot_id"])),
            )
        )

    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="각 봇 시작일=100 기준 지수",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "각 선은 그 봇이 기록을 시작한 날을 100으로 맞춘 상대 비교(리베이스)입니다 — "
        "봇마다 실행 시작일이 다를 수 있습니다."
    )


def render_header(title: str = "💰 모의투자", *, extra_caption: str | None = None) -> None:
    """모의투자 페이지 공통 머리말(제목·면책 경고·안내 캡션)을 한 번만 그린다.

    `pages/모의투자.py`가 기본/전략 두 모델을 펼치기(expander)로 함께 보여주므로, 모델별로
    반복되는 render_body()와 분리해 이 머리말은 페이지당 한 번만 호출한다.
    """
    theme.inject_base_css()
    st.title(title)

    st.warning(
        "⚠️ **실제 돈이 아닌 모의(가상) 투자입니다. 투자 조언이 아닙니다.** "
        "가상 현금 1억원을 규칙 기반 엔진이 자동으로 매매한 결과이며, "
        "특정 종목을 사거나 팔라고 권하는 것이 아닙니다.",
        icon="⚠️",
    )
    st.caption(
        "장중 실시간이 아니라 **하루 1회, 장 마감 후** 자동 매매된 결과입니다 "
        "(평일 저녁 자동 실행). 방문 시점과 매매 실행 시점은 무관합니다."
    )
    if extra_caption:
        st.caption(extra_caption)


def render_body(
    bot_strategies: dict,
    portfolio_dir_fn: Callable[[str], Path],
    *,
    key_prefix: str = "",
) -> None:
    """모델 하나(봇 레지스트리 하나)의 탭 묶음(성과 비교 + 봇별 탭)을 그린다.

    `bot_strategies`는 `{bot_id: {label, decide_trades, rules}}` 형태(`decide_trades`/
    `rules`는 이 함수가 안 씀 — 화면은 원장만 읽으므로 무시된다), `portfolio_dir_fn`은
    `config.portfolio_dir_for`류(봇 id → 원장 경로) 함수다. `key_prefix`는 위젯 key 접두사
    — 기본 모델/전략 모델이 한 페이지에 같이 그려질 때 bot_id가 겹쳐도 key 충돌이
    나지 않게 한다(_render_bot_dashboard 참고).
    """
    st.caption(
        "**기본형/공격적/모멘텀** 세 봇이 같은 날 같은 시세·신호를 보고 서로 다른 판단 "
        "로직으로 독립적으로 매매합니다 — 성과 비교 목적이며 봇 간 자금 이동은 없습니다."
    )
    tab_labels = ["📊 성과 비교"] + [meta["label"] for meta in bot_strategies.values()]
    tabs = st.tabs(tab_labels)

    summaries = []
    for (bot_id, meta), tab in zip(bot_strategies.items(), tabs[1:], strict=True):
        with tab:
            summaries.append(
                _render_bot_dashboard(bot_id, meta["label"], portfolio_dir_fn(bot_id), key_prefix=key_prefix)
            )

    with tabs[0]:
        _render_comparison(summaries)
