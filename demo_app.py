"""데모 대시보드 진입점 — 모니터링(국내증시/코인) + 모의투자(기본 모델/전략 모델) 화면을
하나로 묶는다.

PRD.md 5.6·11장 4단계 참고. app.py는 지금 지인에게 공유 중인 배포이므로 한 글자도
수정하지 않는다. "주가 스크리닝"은 app.py를 그대로 재사용하는 대신, 국내증시/코인을
함께 다루는 pages/모니터링.py(app.py의 상위 확장판)로 대체했다 — app.py 자체는 별도
배포로 여전히 독립 운영된다(이 파일이 참조하지 않을 뿐 삭제되거나 바뀌지 않았다).
Streamlit Cloud에는 이 파일을 main file로 하는 별도 배포를 새로 추가한다(기존 app.py
배포는 그대로 유지).

**사이드바를 이 두 페이지로만 유지한다** — `streamlit run app.py`로 직접 실행하면
Streamlit이 `pages/` 폴더를 자동 인식해 "app" 탭이 하나 더 생기므로, 항상 이 파일
(`demo_app.py`)로 실행해야 한다.

"모의투자" 페이지는 tradermonty Claude 스킬 방법론을 결정론적으로 포팅한 두 번째 실험용
매매 로직(PRD 11장, "전략 모델")을 펼치기(expander)로 함께 보여준다 — 원장이 기본 모델과
완전히 분리돼 있어(`data/portfolio/v2/`) 같은 페이지에 있어도 서로 영향이 없다.

실행: .venv\\Scripts\\streamlit.exe run demo_app.py
"""

import streamlit as st

st.set_page_config(page_title="주가 대시보드", layout="wide")

pg = st.navigation(
    [
        st.Page("pages/모니터링.py", title="모니터링", icon="🖥️"),
        st.Page("pages/모의투자.py", title="모의투자", icon="💰"),
    ]
)
pg.run()
