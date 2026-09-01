"""모의투자 성과 리포팅 — 원장을 읽기만 하는 화면. PRD.md 5.6·10장 8단계·11장 3단계 참고.

**읽기 전용.** 매매를 유발하는 코드를 단 한 줄도 포함하지 않는다 — src.trading_agent나
portfolio의 쓰기 함수(apply_trade/save_daily_result)는 import조차 하지 않는다. 실행(매매)은
scripts/run_daily_trading.py + GitHub Actions가 매일 1회 전담하고, 여기는 그 결과만 보여준다
(4장 실행/조회 분리 원칙).

**기본 모델(v1) / 전략 모델(v2)를 펼치기(expander) 하나의 탭에서 함께 보여준다.** 원래는
"모의투자"/"모의투자(ver2)" 두 페이지였지만, 사이드바 항목을 줄이려고 한 페이지로 합쳤다.
"기본 모델"은 규칙 기반 최초 매매 로직(`trading_agent.BOT_STRATEGIES`)이고, "전략 모델"은
tradermonty Claude 스킬 방법론을 결정론적으로 포팅한 두 번째 실험용 로직
(`trading_agent_v2.BOT_STRATEGIES_V2`)이다 — 원장은 완전히 분리돼 있다
(`data/portfolio/` vs `data/portfolio/v2/`). 실제 렌더링은 `src/portfolio_ui.py`의
`render_header()`(공통 머리말, 한 번만)와 `render_body()`(모델별 탭 묶음, 두 번)로 나눠
호출한다 — 두 모델이 같은 bot_id("default" 등)를 쓰므로 위젯 key 충돌을 막으려고
`key_prefix`를 다르게 넘긴다.

demo_app.py의 st.navigation을 통해서만 로드되므로 st.set_page_config는 호출하지 않는다
(app.py가 이미 호출하고, 같은 세션에서 두 번 호출하면 에러).
"""

from __future__ import annotations

import streamlit as st

from src import config
from src.portfolio_ui import render_body, render_header
from src.trading_agent import BOT_STRATEGIES
from src.trading_agent_v2 import BOT_STRATEGIES_V2

render_header("💰 모의투자")

with st.expander("🧩 기본 모델", expanded=True):
    st.caption("규칙 기반(임계값) 매매 로직의 최초 버전입니다. 실전 검증 이력이 가장 깁니다.")
    render_body(BOT_STRATEGIES, config.portfolio_dir_for, key_prefix="v1_")

with st.expander("🧪 전략 모델", expanded=False):
    st.caption(
        "기술적 분석·모멘텀 스크리닝 방법론을 반영한 두 번째 실험용 매매 로직입니다. "
        "기본 모델과는 완전히 다른 별도 계좌(1억원)로 운용되며, 두 모델 간 자금 이동은 없습니다. "
        "아직 실전 검증 이력이 짧아 기본 모델보다 더 실험적입니다."
    )
    render_body(BOT_STRATEGIES_V2, config.portfolio_dir_for_v2, key_prefix="v2_")
