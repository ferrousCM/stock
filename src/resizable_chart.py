"""가격/거래량 차트를 마우스로 드래그해 위아래 크기를 조절할 수 있게 보여주는 커스텀 뷰.

Plotly의 make_subplots는 서브플롯 행 높이를 사용자가 드래그로 바꾸는 기능 자체가 없다
(줌·이동만 가능하고, 행이 차지하는 비율은 그려질 때 고정된다) — 그래서 가격/거래량을 각각
독립된 Plotly.js 인스턴스(순수 HTML/JS, `streamlit.components.v1.html`)로 따로 그리고, 그
사이에 드래그 가능한 손잡이를 직접 구현해 붙인다. RSI/MACD/스토캐스틱처럼 추가로 켤 수 있는
지표 패널은 크기 조절 없이 고정 높이로 그 아래 이어 붙이되, 같은 컴포넌트(=같은 iframe) 안에
두어 x축(줌·이동) 동기화는 그대로 유지한다 — 서로 다른 iframe이면 JS로 이어줄 방법이 없다.

Streamlit의 `st.plotly_chart(theme="streamlit")`가 자동으로 입혀주는 다크 테마를 이 컴포넌트는
받지 못하므로(components.v1.html은 별도 iframe), `_apply_dark_theme()`로 배경·격자·글자색을
직접 맞춘다.
"""

from __future__ import annotations

import json

import plotly.graph_objects as go
import streamlit.components.v1 as components

_PLOTLY_JS = "https://cdn.plot.ly/plotly-2.32.0.min.js"

# .streamlit/config.toml의 다크 테마 배경과 맞춘다 (src/theme.py 참고).
_BG = "#0b0e11"
_GRID = "rgba(234,236,239,0.08)"
_TEXT = "#eaecef"
_HANDLE_TRACK = "#1e2329"
_HANDLE_BAR = "rgba(234,236,239,0.35)"


def _apply_dark_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(paper_bgcolor=_BG, plot_bgcolor=_BG, font=dict(color=_TEXT))
    fig.update_xaxes(gridcolor=_GRID, zerolinecolor=_GRID, linecolor=_GRID)
    fig.update_yaxes(gridcolor=_GRID, zerolinecolor=_GRID, linecolor=_GRID)
    # st.plotly_chart(theme="streamlit")를 쓰는 일반 경로는 스트림릿이 rangeselector 버튼
    # 배경/글자색까지 자동으로 다크 테마에 맞춰주지만, 이 컴포넌트는 그 처리를 받지 못해
    # 버튼이 기본값(흰 배경)으로 남는다 — 직접 지정한다. update_xaxes(rangeselector=...)는
    # 기존 buttons 목록은 그대로 둔 채 나머지 속성만 병합한다.
    fig.update_xaxes(
        rangeselector=dict(bgcolor=_HANDLE_TRACK, activecolor="#f0b90b", font=dict(color=_TEXT))
    )
    return fig


def render(
    price_fig: go.Figure,
    volume_fig: go.Figure,
    extra_fig: go.Figure | None = None,
    *,
    price_height: int = 320,
    volume_height: int = 110,
    extra_height: int = 0,
) -> None:
    """가격(위, 드래그로 리사이즈)·거래량(아래, 드래그로 리사이즈) + 선택적 추가 지표 패널
    (RSI/MACD/스토캐스틱, 고정 높이)을 그린다. extra_fig가 None이면 그 패널은 생략한다.
    """
    figs = [_apply_dark_theme(price_fig), _apply_dark_theme(volume_fig)]
    chart_ids = ["price", "volume"]
    if extra_fig is not None:
        figs.append(_apply_dark_theme(extra_fig))
        chart_ids.append("extra")

    plot_calls = "\n".join(
        f'var spec_{cid} = {fig.to_json()};\n'
        f'Plotly.newPlot("{cid}", spec_{cid}.data, spec_{cid}.layout, '
        f'{{scrollZoom: true, responsive: true}});'
        for cid, fig in zip(chart_ids, figs, strict=True)
    )
    extra_div = f'<div id="extra" style="height:{extra_height}px;"></div>' if extra_fig is not None else ""
    total_height = price_height + volume_height + extra_height + 16

    html = f"""
<style>
  html, body {{ margin: 0; padding: 0; background: {_BG}; }}
</style>
<div id="pv-container" style="display:flex;flex-direction:column;width:100%;background:{_BG};">
  <div id="price" style="height:{price_height}px;"></div>
  <div id="pv-resizer" title="드래그해서 가격/거래량 크기 조절"
       style="height:10px;cursor:row-resize;display:flex;align-items:center;
              justify-content:center;background:{_HANDLE_TRACK};">
    <div style="width:36px;height:4px;border-radius:2px;background:{_HANDLE_BAR};"></div>
  </div>
  <div id="volume" style="height:{volume_height}px;"></div>
  {extra_div}
</div>
<script src="{_PLOTLY_JS}"></script>
<script>
{plot_calls}

(function() {{
  var priceDiv = document.getElementById("price");
  var volumeDiv = document.getElementById("volume");
  var resizer = document.getElementById("pv-resizer");
  var dragging = false;

  resizer.addEventListener("mousedown", function(e) {{ dragging = true; e.preventDefault(); }});
  document.addEventListener("mouseup", function() {{ dragging = false; }});
  document.addEventListener("mousemove", function(e) {{
    if (!dragging) return;
    var priceRect = priceDiv.getBoundingClientRect();
    var totalPV = priceRect.height + volumeDiv.getBoundingClientRect().height;
    var minH = 60;
    var newPriceH = e.clientY - priceRect.top;
    newPriceH = Math.max(minH, Math.min(totalPV - minH, newPriceH));
    priceDiv.style.height = newPriceH + "px";
    volumeDiv.style.height = (totalPV - newPriceH) + "px";
    Plotly.Plots.resize(priceDiv);
    Plotly.Plots.resize(volumeDiv);
  }});

  // x축 동기화 — 한 차트를 줌/이동하면 나머지 차트에도 같은 범위를 적용한다.
  //
  // 버그였던 예전 방식: 전역 boolean 플래그 하나를 "relayout 시작 시 true, 루프 끝나면
  // false"로 썼는데, Plotly.relayout()이 비동기로 완료되면(내부적으로 requestAnimationFrame
  // 등을 씀) 다른 차트의 plotly_relayout 이벤트가 우리가 플래그를 false로 되돌린 "뒤"에
  // 도착한다 — 그러면 그 이벤트가 "새 사용자 조작"으로 오인되어 다시 전체에 relayout을
  // 걸고, 그게 또 이벤트를 발생시키고... 하는 핑퐁이 반복되며 CPU를 100%로 고정해 몇 번
  // 조작하면 탭이 멈추는 원인이 됐다(실측 확인). 고쳐서 차트마다 자기 자신의 "_syncing"
  // 플래그를 두고, 그 플래그는 프로그램적으로 건 relayout이 실제로 이 차트에 도착했을
  // 때(타이밍과 무관하게) 자기 핸들러 안에서만 소비하도록 바꿨다 — 비동기로 아무리 늦게
  // 와도 정확히 한 번만 무시하고 끝난다(핑퐁이 구조적으로 불가능).
  var allIds = {json.dumps(chart_ids)};
  allIds.forEach(function(id) {{
    var div = document.getElementById(id);
    div._syncing = false;
    div.on("plotly_relayout", function(ev) {{
      if (div._syncing) {{ div._syncing = false; return; }}
      var upd = null;
      if (ev["xaxis.range[0]"] !== undefined && ev["xaxis.range[1]"] !== undefined) {{
        upd = {{"xaxis.range[0]": ev["xaxis.range[0]"], "xaxis.range[1]": ev["xaxis.range[1]"]}};
      }} else if (ev["xaxis.range"] !== undefined) {{
        upd = {{"xaxis.range": ev["xaxis.range"]}};
      }} else if (ev["xaxis.autorange"]) {{
        upd = {{"xaxis.autorange": true}};
      }}
      if (!upd) return;
      allIds.forEach(function(otherId) {{
        if (otherId === id) return;
        var otherDiv = document.getElementById(otherId);
        otherDiv._syncing = true;
        Plotly.relayout(otherDiv, upd);
      }});
    }});
  }});
}})();
</script>
"""
    components.html(html, height=total_height, scrolling=False)
