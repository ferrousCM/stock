"""모의투자 성과 리포팅 — 원장을 읽기만 하는 화면. PRD.md 5.6·10장 8단계·11장 3단계 참고.

**읽기 전용.** 매매를 유발하는 코드를 단 한 줄도 포함하지 않는다 — src.trading_agent나
portfolio의 쓰기 함수(apply_trade/save_daily_result)는 import조차 하지 않는다. 실행(매매)은
scripts/run_daily_trading.py + GitHub Actions가 매일 1회 전담하고, 여기는 그 결과만 보여준다
(4장 실행/조회 분리 원칙).

**봇 다중화**: "기본형"(기존 그대로) + "공격적" + "모멘텀" 세 봇을 봇 탭으로 나눠 각자의
전체 리포트를 보여주고, "성과 비교" 탭에서 세 봇을 나란히 비교한다. 어떤 봇이 있는지는
`trading_agent.BOT_STRATEGIES`(단일 레지스트리) 하나만 보고 순회한다 — 봇이 늘어나도 이
파일을 구조적으로 다시 손댈 필요가 없다.

**실제 렌더링 로직은 `src/portfolio_ui.py`로 옮겼다**(PRD 11장 3단계 리팩터, 순수 이동 —
로직 변경 없음). 이 파일과 `pages/모의투자_v2.py`가 그 모듈의 `render_page()`를 각자 다른
봇 레지스트리·원장 디렉터리 함수로 호출해 "기본 프레임 동일 유지" 요구사항을 코드 중복
없이 만족한다.

demo_app.py의 st.navigation을 통해서만 로드되므로 st.set_page_config는 호출하지 않는다
(app.py가 이미 호출하고, 같은 세션에서 두 번 호출하면 에러).
"""

from __future__ import annotations

from src import config
from src.portfolio_ui import render_page
from src.trading_agent import BOT_STRATEGIES

render_page(BOT_STRATEGIES, config.portfolio_dir_for, title="💰 모의투자")
