"""모의투자 성과 리포팅 — 원장을 읽기만 하는 화면. PRD.md 5.6·10장 8단계 참고.

**읽기 전용.** 매매를 유발하는 코드를 단 한 줄도 포함하지 않는다 — src.trading_agent나
portfolio의 쓰기 함수(apply_trade/save_daily_result)는 import조차 하지 않는다. 실행(매매)은
scripts/run_daily_trading.py + GitHub Actions가 매일 1회 전담하고, 여기는 그 결과만 보여준다
(4장 실행/조회 분리 원칙).

**봇 다중화**: "기본형"(기존 그대로) + "공격적" + "모멘텀" 세 봇을 봇 탭으로 나눠 각자의
전체 리포트를 보여주고, "성과 비교" 탭에서 세 봇을 나란히 비교한다. 어떤 봇이 있는지는
`trading_agent.BOT_STRATEGIES`(단일 레지스트리) 하나만 보고 순회한다 — 봇이 늘어나도 이
파일을 구조적으로 다시 손댈 필요가 없다.

demo_app.py의 st.navigation을 통해서만 로드되므로 st.set_page_config는 호출하지 않는다
(app.py가 이미 호출하고, 같은 세션에서 두 번 호출하면 에러).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, portfolio, predictor, screener
from src import data_loader as dl
from src.trading_agent import BOT_STRATEGIES

PORTFOLIO_COLOR = "#0ea5e9"  # 포트폴리오 자산 라인 (하늘색) — 기본형 봇 비교 색으로도 재사용
AGGRESSIVE_COLOR = "#f97316"  # 공격적 봇 비교 색 (주황)
MOMENTUM_COLOR = "#8b5cf6"  # 모멘텀 봇 비교 색 (보라)
KOSPI_COLOR = "#f59e0b"  # 코스피 비교 라인 — charts.py의 지수 오버레이 색(amber)과 통일
UP_COLOR, DOWN_COLOR, FLAT_COLOR = "#ef4444", "#3b82f6", "#6b7280"  # app.py와 동일한 국내 관행
UP_SOFT, DOWN_SOFT, FLAT_SOFT = "#cc6666", "#668dcc", "#9ca3af"  # 파스텔 — 매수/매도 표시·예측값용

_BOT_COMPARE_COLOR = {"default": PORTFOLIO_COLOR, "aggressive": AGGRESSIVE_COLOR, "momentum": MOMENTUM_COLOR}

st.title("💰 모의투자")

st.warning(
    "⚠️ **실제 금전 거래가 아닌 시뮬레이션입니다. 투자 조언이 아닙니다.** "
    "가상 현금 1억원으로 시작해 규칙 기반 엔진이 자동으로 매매한 결과를 보여줄 뿐, "
    "특정 종목의 매수·매도를 권유하는 것이 아닙니다.",
    icon="⚠️",
)
st.caption(
    "장중 실시간이 아니라 **하루 1회, 장 마감 후** 자동 매매된 결과입니다 "
    "(GitHub Actions가 평일 19:30 KST 전후 실행). 방문 시점과 매매 실행 시점은 무관합니다. "
    "**기본형/공격적/모멘텀** 세 봇이 같은 날 같은 시세·신호를 보고 서로 다른 판단 로직으로 "
    "독립적으로 매매합니다 — 성과 비교 목적이며 봇 간 자금 이동은 없습니다."
)


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


def _render_bot_dashboard(bot_id: str, label: str, portfolio_dir: Path) -> dict:
    """봇 하나의 전체 리포트(요약 지표·자산 추이·보유종목·거래내역·거래 요약)를 그리고,
    "성과 비교" 탭에서 쓸 요약 dict를 반환한다. 원장을 읽기만 한다."""
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
            go.Scatter(x=eh["date"], y=port_idx, name=label, line=dict(width=2.4, color=PORTFOLIO_COLOR))
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
            key=f"holdings_table_{bot_id}",
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
                "기간", value=(min_d, max_d), min_value=min_d, max_value=max_d, key=f"date_range_{bot_id}"
            )
        with fcol2:
            query = st.text_input(
                "종목명 또는 종목코드 검색", placeholder="예: 삼성전자 또는 005930", key=f"query_{bot_id}"
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
                key=f"trades_table_{bot_id}",
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
        eq["전일대비"] = eq["total_equity"].diff()
        eq["전일대비율"] = eq["total_equity"].pct_change() * 100
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

        def _color_diff(val: float) -> str:
            if pd.isna(val) or val == 0:
                return ""
            return f"color:{UP_COLOR};font-weight:600" if val > 0 else f"color:{DOWN_COLOR};font-weight:600"

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
                "전일대비": st.column_config.NumberColumn(format="%+,.0f원"),
                "전일대비율": st.column_config.NumberColumn(format="%+.2f%%"),
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
                "코스피 대비 초과수익률": s["excess_pct"],
                "보유종목수": s["n_holdings"],
                "누적거래횟수": s["n_trades"],
                "마지막 매매 기준일": s["last_run"] if s["last_run"] else "아직 없음",
            }
            for s in summaries
        ]
    )
    st.dataframe(
        table,
        column_config={
            "총자산": st.column_config.NumberColumn(format="%,.0f원"),
            "누적수익률": st.column_config.NumberColumn(format="%+.2f%%"),
            "코스피 대비 초과수익률": st.column_config.NumberColumn(format="%+.2f%%"),
        },
        hide_index=True,
        width="stretch",
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


_tab_labels = ["📊 성과 비교"] + [meta["label"] for meta in BOT_STRATEGIES.values()]
_tabs = st.tabs(_tab_labels)

_summaries = []
for (bot_id, meta), tab in zip(BOT_STRATEGIES.items(), _tabs[1:], strict=True):
    with tab:
        _summaries.append(_render_bot_dashboard(bot_id, meta["label"], config.portfolio_dir_for(bot_id)))

with _tabs[0]:
    _render_comparison(_summaries)
