"""인터랙티브 Plotly 차트 — 캔들스틱 + 이동평균 + 지수 오버레이 + 거래량/RSI/MACD.

Streamlit 대시보드에서 쓰지만, 노트북에서 fig.show() 로도 그대로 쓸 수 있게
Streamlit에 의존하지 않는다.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_INDEX_COLORS = ["#f59e0b", "#10b981", "#8b5cf6", "#06b6d4", "#ec4899"]
# 상승/하락 기본색 — app.py(고정 배포, 국내 관행 상승=빨강/하락=파랑)와의 호환을 위해
# 기본값은 그대로 두고, 다른 색을 쓰고 싶은 호출부(예: 바이낸스 배색을 쓰는
# pages/모니터링.py)는 build_chart(up_color=..., down_color=...)로 덮어쓴다.
_UP, _DOWN = "#ef4444", "#3b82f6"


def _rebase(series: pd.Series, to_index: pd.DatetimeIndex, base_value: float) -> pd.Series:
    """series를 to_index에 맞춰 정렬(ffill)하고, 첫 값이 base_value가 되도록 비례 스케일한다.

    가격 스케일이 전혀 다른 지수(예: 코스피 3,200 vs 개별주 5만원)를
    같은 y축에 겹쳐 그리기 위한 리베이스. 절대값이 아니라 '같은 출발점에서
    상대적으로 얼마나 움직였는가'를 비교하는 용도.

    reindex에 method="ffill"을 반드시 줘야 한다 — 그냥 reindex 후 .ffill()을 하면
    to_index에 정확히 일치하는 날짜가 없는 값은 전부 NaN으로 채워진 뒤 그 NaN들끼리
    ffill되어 아무것도 못 채운다(예: 주봉/월봉처럼 to_index가 원본 series의 날짜와
    거의 안 겹치는 경우 전체가 NaN이 되는 버그가 있었다). method="ffill"은 원본
    series의 인덱스에서 to_index 각 시점 "직전의 가장 가까운" 값을 찾아 채운다.
    """
    aligned = series.reindex(to_index, method="ffill").bfill()
    first = aligned.iloc[0] if len(aligned) else None
    if not aligned.empty and first not in (0, None) and pd.notna(first):
        aligned = aligned / first * base_value
    return aligned


def build_chart(
    df: pd.DataFrame,
    title: str,
    sma_windows: tuple[int, ...] = (20, 60),
    show_bollinger: bool = False,
    show_volume: bool = True,
    show_rsi: bool = False,
    show_macd: bool = False,
    index_overlays: dict[str, pd.Series] | None = None,
    base_height: int = 520,
    panel_height: int = 140,
    chart_type: str = "candle",
    show_rangeselector: bool = False,
    log_y: bool = False,
    show_stochastic: bool = False,
    show_ichimoku: bool = False,
    crosshair: bool = False,
    drag_pan: bool = False,
    volume_profile: pd.DataFrame | None = None,
    drawing_tools: bool = False,
    up_color: str = _UP,
    down_color: str = _DOWN,
    include_price: bool = True,
) -> go.Figure:
    """df: indicators.add_all()을 거친 OHLCV+지표 DataFrame (컬럼: Open/High/Low/Close/Volume/sma*/rsi14/macd/signal).

    index_overlays: {"KOSPI": 가격시계열, ...} — df의 시작 종가에 맞춰 리베이스해 가격 패널에 겹쳐 그린다.

    chart_type: "candle"(기본) | "line". 하이킨아시는 이 함수가 모른다 — 호출부가
    indicators.heikin_ashi()로 미리 변환한 df를 chart_type="candle"로 넘기면 된다
    (캔들 시각화 자체는 동일하고 데이터만 다르다).

    show_stochastic/show_ichimoku: df에 각각 stoch_k/stoch_d, ichimoku_tenkan 등의
    컬럼이 미리 계산돼 있어야 한다(indicators.stochastic()/indicators.ichimoku() 참고) —
    이 함수는 계산하지 않고 시각화만 한다(기존 SMA/RSI/MACD와 같은 원칙). 컬럼이 없으면
    조용히 건너뛴다.

    crosshair: 모든 서브플롯을 가로지르는 세로 점선 + OHLC 스파이크(업비트 스타일 크로스헤어).
    drag_pan: 기본 Plotly 드래그 동작(사각형 확대)을 팬(이동)으로 바꾼다 — 업비트는 기본이
    팬이고 확대는 마우스 휠로 한다. 휠 확대(scrollZoom)는 Figure 속성이 아니라
    st.plotly_chart(fig, config={"scrollZoom": True})처럼 렌더링 시점에 호출부가 켜야 한다.

    volume_profile: indicators.volume_profile(df)의 결과(price_low/price_high/price_mid/
    volume 컬럼)를 그대로 넘기면 가격 패널 오른쪽에 좁은 열을 하나 더 만들어 가격대별
    누적 거래량을 수평 막대로 그린다(이 함수는 계산하지 않고 시각화만 한다 — 다른
    지표들과 같은 원칙). None이거나 비어 있으면 기존과 똑같이 1열 레이아웃을 쓴다 —
    app.py는 이 파라미터를 아예 안 넘기므로 항상 이 경로를 타서 화면이 그대로다.

    drawing_tools: 추세선/사각형/원/자유선 + 지우개 버튼을 모드바에 추가한다. Plotly가
    자체 내장한 도형 그리기 기능(dragmode="drawline" 등)을 쓴 것으로, 신규 의존성이
    전혀 없다 — 별도 차트 라이브러리(streamlit-lightweight-charts 등)를 검토했으나
    TradingView 무료 Lightweight Charts를 감싼 래퍼라 그리기 도구 자체가 없어서
    (그리기 도구는 유료 Advanced Charts 제품 전용) 채택하지 않았다. 단, 피보나치
    되돌림처럼 특정 비율을 자동 계산해 그려주는 도구는 Plotly에 없어 지원하지 않는다
    (직선/열린 경로/사각형/원만 가능). 그린 도형은 브라우저 세션 안에서만 유지되고
    Streamlit이 다른 위젯 조작으로 스크립트를 재실행하면 사라진다 — relayout 이벤트를
    Python으로 되돌려받지 않는 한(추가 의존성 필요) 이건 이 방식의 근본적 한계다.
    """
    panels = ["가격"] if include_price else []
    if show_volume:
        panels.append("거래량")
    if show_rsi:
        panels.append("RSI(14)")
    if show_stochastic:
        panels.append("스토캐스틱")
    if show_macd:
        panels.append("MACD")

    rows = len(panels)
    raw_heights = [0.55] + [0.45 / (rows - 1)] * (rows - 1) if rows > 1 else [1.0]

    has_vp = include_price and volume_profile is not None and not volume_profile.empty
    if has_vp:
        # 매물대는 시간축이 아니라 거래량 크기를 가로축으로 쓰는 완전히 다른 축이 필요해서,
        # 가격 패널 옆에 좁은 열을 하나 더 만든다(같은 행이라 y축은 shared_yaxes로 가격
        # 패널과 맞춘다 — 안 맞추면 매물대 막대 높이가 실제 가격과 안 맞아 보인다).
        # 다른 행(거래량/RSI/...)은 colspan=2로 이 열까지 통째로 차지해서 그대로 전체 폭을 쓴다.
        specs = [[{}, {}]] + [[{"colspan": 2}, None] for _ in range(rows - 1)]
        subplot_titles = [panels[0], "매물대"]
        for p in panels[1:]:
            subplot_titles += [p, ""]
        fig = make_subplots(
            rows=rows,
            cols=2,
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.04,
            horizontal_spacing=0.015,
            row_heights=raw_heights,
            column_widths=[0.84, 0.16],
            specs=specs,
            subplot_titles=subplot_titles,
        )
        # shared_xaxes=True는 colspan을 쓴 specs와 같이 쓰면 실제로 축을 안 이어준다(실측
        # 확인 — matches가 전부 None으로 남는다). 가격/거래량/RSI 등 시간축을 쓰는 1열
        # 축들을 수동으로 이어줘야 한 패널에서 확대·이동했을 때 나머지도 같이 움직인다.
        # shared_yaxes(가격↔매물대, 같은 행이라 문제없이 연결됨)는 이 버그의 영향이 없다.
        for r in range(2, rows + 1):
            fig.update_xaxes(matches="x", row=r, col=1)
    else:
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=raw_heights,
            subplot_titles=panels,
        )

    if include_price:
        if chart_type == "line":
            fig.add_trace(
                go.Scatter(x=df.index, y=df["Close"], name="종가", line=dict(width=1.6, color=up_color)),
                row=1,
                col=1,
            )
        else:
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="가격",
                    increasing_line_color=up_color,
                    decreasing_line_color=down_color,
                ),
                row=1,
                col=1,
            )

        for w in sma_windows:
            col = f"sma{w}"
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[col], name=f"SMA{w}", line=dict(width=1.3)),
                    row=1,
                    col=1,
                )

        if show_bollinger and {"upper", "lower"} <= set(df.columns):
            band_color = "rgba(148,163,184,0.9)"
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["upper"], name="BB 상단", line=dict(width=1, color=band_color, dash="dot")
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["lower"],
                    name="BB 하단",
                    line=dict(width=1, color=band_color, dash="dot"),
                    fill="tonexty",
                    fillcolor="rgba(148,163,184,0.12)",
                ),
                row=1,
                col=1,
            )

        if show_ichimoku and {
            "ichimoku_tenkan",
            "ichimoku_kijun",
            "ichimoku_senkou_a",
            "ichimoku_senkou_b",
        } <= set(df.columns):
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["ichimoku_tenkan"], name="전환선", line=dict(width=1, color="#ef4444")
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["ichimoku_kijun"], name="기준선", line=dict(width=1, color="#3b82f6")
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["ichimoku_senkou_a"],
                    name="선행스팬A",
                    line=dict(width=0.6, color="rgba(16,185,129,0.6)"),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["ichimoku_senkou_b"],
                    name="선행스팬B",
                    line=dict(width=0.6, color="rgba(239,68,68,0.6)"),
                    fill="tonexty",
                    fillcolor="rgba(148,163,184,0.15)",
                ),
                row=1,
                col=1,
            )

        if has_vp:
            fig.add_trace(
                go.Bar(
                    x=volume_profile["volume"],
                    y=volume_profile["price_mid"],
                    orientation="h",
                    marker_color="rgba(100,116,139,0.55)",
                    showlegend=False,
                    name="매물대",
                    width=(volume_profile["price_high"] - volume_profile["price_low"]) * 0.9,
                ),
                row=1,
                col=2,
            )
            fig.update_xaxes(showticklabels=False, row=1, col=2)
            fig.update_yaxes(showticklabels=False, row=1, col=2)

        if index_overlays:
            base = float(df["Close"].iloc[0])
            for i, (label, series) in enumerate(index_overlays.items()):
                rebased = _rebase(series, df.index, base)
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=rebased,
                        name=f"{label} (비교)",
                        line=dict(width=1.2, dash="dot", color=_INDEX_COLORS[i % len(_INDEX_COLORS)]),
                    ),
                    row=1,
                    col=1,
                )

    row = 1 if include_price else 0
    if show_volume:
        row += 1
        colors = [up_color if c >= o else down_color for o, c in zip(df["Open"], df["Close"], strict=True)]
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="거래량", marker_color=colors, showlegend=False),
            row=row,
            col=1,
        )

    if show_rsi:
        row += 1
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["rsi14"],
                name="RSI14",
                line=dict(color="#7c3aed", width=1.2),
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="gray", line_width=0.8, row=row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="gray", line_width=0.8, row=row, col=1)

    if show_stochastic:
        row += 1
        fig.add_trace(
            go.Scatter(x=df.index, y=df["stoch_k"], name="%K", line=dict(color="#7c3aed", width=1.2)),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["stoch_d"], name="%D", line=dict(color="#f59e0b", width=1.2)),
            row=row,
            col=1,
        )
        fig.add_hline(y=80, line_dash="dash", line_color="gray", line_width=0.8, row=row, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="gray", line_width=0.8, row=row, col=1)

    if show_macd:
        row += 1
        fig.add_trace(
            go.Scatter(x=df.index, y=df["macd"], name="MACD", line=dict(color="#0ea5e9", width=1.2)),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["signal"], name="Signal", line=dict(color="#f97316", width=1.2)),
            row=row,
            col=1,
        )
        hist_colors = [up_color if v >= 0 else down_color for v in df["hist"]]
        fig.add_trace(
            go.Bar(x=df.index, y=df["hist"], name="Hist", marker_color=hist_colors, showlegend=False),
            row=row,
            col=1,
        )

    fig.update_layout(
        title=title,
        height=base_height + panel_height * (rows - 1),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=70, b=30),
        hovermode="x unified",
    )

    if show_rangeselector and include_price:
        fig.update_xaxes(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1개월", step="month", stepmode="backward"),
                    dict(count=3, label="3개월", step="month", stepmode="backward"),
                    dict(count=6, label="6개월", step="month", stepmode="backward"),
                    dict(count=1, label="1년", step="year", stepmode="backward"),
                    dict(step="all", label="전체"),
                ],
                y=1.18,
                yanchor="top",
            ),
            row=1,
            col=1,
        )
    if log_y and include_price:
        fig.update_yaxes(type="log", row=1, col=1)

    if crosshair:
        # row/col을 지정하지 않으면 모든 서브플롯(가격+거래량+RSI+...)에 다 적용된다 —
        # 업비트처럼 세로선이 패널 전체를 가로지르는 크로스헤어를 만든다.
        fig.update_xaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="rgba(148,163,184,0.7)",
            spikethickness=1,
            spikedash="dot",
        )
        fig.update_yaxes(
            showspikes=True,
            spikesnap="cursor",
            spikecolor="rgba(148,163,184,0.7)",
            spikethickness=1,
            spikedash="dot",
        )
    if drag_pan:
        fig.update_layout(dragmode="pan")

    if drawing_tools:
        # dragmode는 안 건드린다 — drag_pan이 켠 "pan"을 기본값으로 유지하고, 사용자가
        # 모드바에서 그리기 버튼을 클릭하면 Plotly가 알아서 dragmode를 그 도구로 바꾼다.
        fig.update_layout(
            newshape={"line": {"color": up_color, "width": 1.5}},
            modebar_add=["drawline", "drawopenpath", "drawrect", "drawcircle", "eraseshape"],
        )

    return fig


def build_volume_chart(
    df: pd.DataFrame,
    up_color: str = _UP,
    down_color: str = _DOWN,
    crosshair: bool = False,
    height: int | None = None,
) -> go.Figure:
    """거래량만 그리는 단독(단일 서브플롯) 차트 — build_chart()의 거래량 패널과 색상·로직이
    동일하다. 가격 차트와 별도 Plotly 컴포넌트로 그려서 둘 사이 경계를 드래그로 리사이즈할
    수 있게 할 때 쓴다(pages/모니터링.py의 드래그 리사이즈 뷰 참고) — 그 경우 x축 동기화는
    두 컴포넌트가 공유하는 JS가 처리하므로 이 함수 자체는 다른 차트와의 연동을 모른다.
    """
    colors = [up_color if c >= o else down_color for o, c in zip(df["Open"], df["Close"], strict=True)]
    fig = go.Figure(
        go.Bar(x=df.index, y=df["Volume"], name="거래량", marker_color=colors, showlegend=False)
    )
    fig.update_layout(
        margin=dict(l=40, r=20, t=10, b=30),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        **({"height": height} if height else {}),
    )
    if crosshair:
        fig.update_xaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="rgba(148,163,184,0.7)",
            spikethickness=1,
            spikedash="dot",
        )
        fig.update_yaxes(
            showspikes=True, spikesnap="cursor", spikecolor="rgba(148,163,184,0.7)", spikethickness=1, spikedash="dot"
        )
    return fig
