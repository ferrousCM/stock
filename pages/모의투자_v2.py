"""모의투자 v2 성과 리포팅 — 원장을 읽기만 하는 화면. PRD.md 11장(3·4단계) 참고.

**읽기 전용.** v1(`pages/모의투자.py`)과 완전히 동일한 원칙 — `trading_agent_v2`나
`portfolio`의 쓰기 함수는 import조차 하지 않는다. 실행(매매)은
`scripts/run_daily_trading.py` + GitHub Actions가 전담하고, 여기는 그 결과만 보여준다.

**v2는 tradermonty Claude 스킬(position-sizer·technical-analyst·vcp-screener·
stockbee-momentum-burst-screener) 방법론을 결정론적으로 포팅한 두 번째 실험용 매매
로직**이다(PRD 11.0·11.1) — v1과 원장이 완전히 분리돼 있고(`data/portfolio/v2/`, 각자
1억원으로 시작, 자금 이동 없음), 아직 실전 검증 이력이 없어 v1보다 더 실험적이다(PRD
11.9). "기본형"/"공격적"/"모멘텀" 세 봇 구조와 화면 프레임은 v1과 동일하게 유지한다 —
`src/portfolio_ui.py`의 `render_page()`를 v1과 똑같이 호출하되 봇 레지스트리
(`BOT_STRATEGIES_V2`)와 원장 디렉터리 함수(`config.portfolio_dir_for_v2`)만 다르다
(PRD 11장 3단계 리팩터 덕분에 코드 중복 없이 재사용).

demo_app.py의 st.navigation을 통해서만 로드되므로 st.set_page_config는 호출하지 않는다.
"""

from __future__ import annotations

from src import config
from src.portfolio_ui import render_page
from src.trading_agent_v2 import BOT_STRATEGIES_V2

render_page(
    BOT_STRATEGIES_V2,
    config.portfolio_dir_for_v2,
    title="💰 모의투자 (ver2)",
    extra_caption=(
        "🧪 **이 화면은 v2 — tradermonty Claude 스킬(포지션 사이징·기술적 분석·모멘텀 스크리닝) "
        "방법론을 결정론적으로 포팅한 두 번째 실험용 매매 로직**입니다. "
        "[모의투자] 탭(v1)과는 완전히 분리된 별도 계좌(1억원)이며 봇 간·버전 간 자금 이동은 없습니다. "
        "아직 실전 검증 이력이 없어 v1보다 더 실험적입니다."
    ),
)
