# CLAUDE.md

이 파일은 이 저장소에서 작업할 때 따라야 할 공통 규칙이다. 새 기능(스크리닝, 백테스트, 리포트 등)을 추가할 때마다 이 문서를 기준으로 삼고, 새로운 제약이나 결정이 생기면 이 문서를 업데이트한다.

## 프로젝트 개요

KOSPI/KOSDAQ/NASDAQ 등 주가 데이터를 수집·분석하는 작업환경. 데이터 계층(`src/`)과 그 위의 Streamlit 대시보드(`app.py`), 탐색용 노트북(`notebooks/`)으로 구성된다.

## 환경

- Python 3.14.7, Windows, `.venv` 기준. 패키지 관리는 **pip + requirements.txt** (uv/poetry 아님).
- pandas 3.0 — Copy-on-Write가 기본이고 `df.append()`, `fillna(method=...)` 같은 구 API가 제거됐다. 오래된 예제 코드를 그대로 쓰지 말 것.
- **새 의존성을 추가하기 전에 실제로 이 환경에서 동작하는지 스모크 테스트로 먼저 확인한다.** `pip install`이 성공했다고 끝난 게 아니다 — 실제 호출까지 해봐야 한다. (`pykrx`가 설치는 됐지만 실호출에서 로그인 요구로 막혔던 사례 참고 — 아래 "알려진 제약" 참고.)
- 새 라이브러리가 기존 pandas/numpy 버전을 강제로 바꾸면(`pip install`이 다운그레이드하면) 반드시 인지하고, 필요 없어지면 제거 후 버전을 원복한다.

## 아키텍처

```
src/
  config.py       경로, 기본값, 심볼 별칭(INDICES, MACRO) — 다른 모든 모듈이 여기서 상수를 가져온다
  cache_utils.py  parquet 캐시 공용 유틸 (cache_path, is_fresh)
  data_loader.py  개별 종목/지수 조회 (FDR 래퍼 + 캐시)
  screener.py     전종목 대량 스크리닝 (개별 조회가 아니라 벌크 소스 사용)
  indicators.py   기술적 지표 + 성과 지표 (순수 함수, pandas Series/DataFrame in-out)
  charts.py       Plotly 인터랙티브 차트 빌더 (Streamlit 비의존)
  news.py         네이버 금융 뉴스 스크래핑 + 요약(리드 문단 발췌, LLM 아님)
  sentiment.py    키워드 기반 감성 판정 (POSITIVE_WORDS/NEGATIVE_WORDS 매칭)
  dart.py         DART 전자공시 OpenAPI 연동 (API 키 필요, 없으면 DartKeyMissing)
  predictor.py    가격 예측 (릿지 회귀, 종목별 즉석 학습 + 홀드아웃 검증) — 투자 조언 아님
  crypto_loader.py    data_loader.py의 코인 버전 (업비트 공개 API, OHLCV 컬럼 계약 동일)
  crypto_screener.py  screener.py의 코인 버전 (업비트 KRW 마켓 벌크 스크리닝)
  crypto_news.py       코인 뉴스(네이버 뉴스 키워드 검색) — 본문 조회 이후는 news.py 재사용
  us_screener.py       screener.py의 해외증시(나스닥) 버전 — 나스닥 공개 스크리너 API 벌크 스크리닝.
                        개별 종목 히스토리/검색은 새 모듈 없이 data_loader.py를 그대로 재사용한다
                        (FDR가 이미 미국 티커를 지원 — market="NASDAQ"만 넘기면 됨)
  plotting.py     matplotlib 한글 폰트 설정 (노트북 전용)
app.py            기존 스크리닝 대시보드 단독 배포 진입점 — 더는 손대지 않는다(아래 참고)
pages/모니터링.py  app.py의 확장판 — 좌상단에서 국내증시/해외증시/코인 전환 (demo_app.py 전용)
demo_app.py       pages/모니터링.py + pages/모의투자.py를 묶은 데모 진입점 (st.navigation)
notebooks/        탐색용. src를 import해서 재사용 (%autoreload 사용)
tests/            pytest
```

**규칙**: UI 코드에서 `FinanceDataReader`나 외부 URL을 직접 호출하지 않는다. 항상 `src/data_loader.py`/`src/screener.py`(주식) 또는 `src/crypto_loader.py`/`src/crypto_screener.py`(코인)를 거친다. 데이터 로직과 화면 로직을 분리해야 나중에 노트북·다른 대시보드에서도 재사용 가능하다.

**`app.py`는 더 이상 손대지 않는다.** 지인에게 공유 중인 배포라 계속 그대로 둔다 — 스크리닝 기능이 발전할 곳은 `pages/모니터링.py`(demo_app.py 전용)다. `indicators.py`/`charts.py`/`predictor.py`는 OHLCV 컬럼 계약(Open/High/Low/Close/Volume, DatetimeIndex)만 맞으면 소스가 주식이든 코인이든 그대로 동작하도록 설계돼 있다 — 새 데이터 소스를 추가할 때도 이 세 모듈은 건드릴 필요가 없어야 한다(실제로 코인 추가 때 그랬다).

## 데이터 접근 규칙

- **캐시는 항상 쓴다.** 새 조회 함수를 추가할 때 `cache_utils.cache_path(kind, key)` + `is_fresh()` 패턴을 따르고, `use_cache: bool = True` 파라미터로 강제 갱신 여지를 남긴다. TTL은 `config.CACHE_TTL_SEC`.
- **전종목/대량 데이터가 필요하면 종목별 반복 호출을 절대 하지 않는다.** 수천 종목을 하나씩 조회하면 느리고 상대 서버에도 부담이다. 먼저 벌크로 가져올 방법을 찾는다 (예: `screener.py`가 KRX 일자별 스냅샷을 통째로 받아 조인하는 방식 — 요청 수가 종목 수와 무관하게 고정됨).
- 심볼 표기는 `config.INDICES` / `config.MACRO` 별칭을 따른다. 새 지수·매크로 지표를 자주 쓰게 되면 거기 추가한다.
- 네트워크 호출을 테스트/조사할 때도 반복문 규모를 작게 유지한다 (예: 과거 데이터 존재 여부 확인 시 400일치를 다 찔러보지 않고 40~50일 정도로 충분히 검증).

## 알려진 제약 (재조사 방지용)

- **pykrx는 못 쓴다.** 최신 버전의 벌크 조회 함수(`get_market_ohlcv_by_ticker` 등)가 `KRX_ID`/`KRX_PW` 환경변수 기반 로그인을 요구한다. 로그인 정보가 없으므로 배제.
- **`data.krx.co.kr`에 직접 POST로 스크래핑하는 것도 막혀 있다** (세션을 워밍업해도 `LOGOUT` 응답). KRX가 자체 방어를 강화한 상태.
- 대신 FinanceDataReader가 실제로 쓰는 **GitHub 미러** `FinanceData/fdr_krx_data_cache`의 일자별 스냅샷 CSV(`data/listing/krx/{YYYY-MM-DD}.csv`)를 직접 지정한 날짜로 내려받는 방식이 로그인 없이 동작한다. `screener.py`가 이 방식을 쓴다. 과거 스냅샷은 최소 45일 이상 존재하는 것을 확인했다(그보다 오래된 비교가 필요하면 존재 여부를 먼저 확인할 것).
- `pandas 3.0` + `streamlit`을 같이 쓰면 `streamlit`이 `pyarrow` 상한 버전을 요구해 최신 `pyarrow`보다 낮은 버전이 깔릴 수 있다 (현재 24.0.0). 문제는 없지만 버전 충돌 경고가 뜨면 이게 원인일 가능성이 높다.
- **네이버 금융 뉴스 스크래핑**(`news.py`)은 `Referer` 헤더(`https://finance.naver.com/item/news.naver?code={종목코드}`)가 없으면 빈 결과("검색된 뉴스가 없습니다")를 반환한다. 또한 응답 인코딩이 `euc-kr`이라 `r.encoding = "euc-kr"`을 명시해야 한글이 깨지지 않는다. 기사 원문은 목록 페이지가 아니라 `n.news.naver.com/mnews/article/{office_id}/{article_id}`에 있다 (목록의 `news_read.naver` 링크는 JS로 이 URL에 리다이렉트만 함 — office_id/article_id를 href에서 파싱해 바로 이 URL을 구성하면 리다이렉트 왕복을 생략할 수 있다).
- **DART API**는 무료지만 각자 https://opendart.fss.or.kr 에서 이메일 인증 후 키를 발급받아야 한다 — Claude가 대신 발급받을 수 없다. 키는 `.env`(gitignore됨)의 `DART_API_KEY`로 관리하고 `config.py`가 `python-dotenv`로 로드한다. 종목코드(6자리)와 DART의 corp_code(8자리)는 다른 체계라 `corpCode.xml`(zip) 전체를 내려받아 매핑해야 한다 — 이것도 상장사 전체를 한 번에 주므로 종목별 반복 호출이 아니다.
- **API 키는 절대 코드에 하드코딩하거나 커밋되는 파일에 넣지 않는다.** 항상 `.env` + `config.py`의 `os.environ.get(...)` 패턴을 따른다. 키가 없을 때는 조용히 실패하지 말고 (`DartKeyMissing`처럼) 무엇을 어디서 발급받아야 하는지 알려주는 예외/메시지를 낸다.
- **뉴스 감성의 과거 히스토리는 조회할 방법이 없다** (뉴스 소스가 최신 기사 목록만 제공). `predictor.py`가 뉴스 피처를 쓰려면 매 실행마다 그날의 감성을 `data/raw/news_sentiment/{종목코드}.parquet`에 누적 기록하는 수밖에 없다 — 기록 시작 전 과거는 중립(0)으로 채운다. 기록은 **`news.log_sentiment_from_news(code, news_df)`** 를 쓴다(저수준 `log_daily_sentiment()`를 직접 부르지 말 것). 평균 점수만 넘기면 심층 모델이 쓰는 긍정/부정/기사 수 피처가 영원히 비어 항상 0이 된다 — 실제로 그렇게 방치됐던 적이 있다.
  - 이 로그는 git으로 추적하는데 **parquet은 같은 내용을 다시 써도 바이트가 달라진다.** 그래서 `log_daily_sentiment()`는 같은 날짜에 같은 값이면 파일을 아예 건드리지 않는다(`_already_logged()`). 이 생략 로직을 지우면 앱을 켤 때마다 워킹 트리가 dirty해진다 — 무의미한 diff 커밋이 실제로 쌓였던 적이 있다. 이 로그는 `data/raw/`에 있어 `.gitignore`가 "재생성 가능한 캐시"로 취급하지만, 실제로는 한 번 지우면 복구 불가능한 유일한 데이터다(다른 data/raw·data/cache 파일과 성격이 다름).
- **업비트 공개 API(`crypto_loader.py`/`crypto_screener.py`)는 인증 없이 되지만, 벌크로 못 주는 데이터가 있다.** "N일 전 스냅샷"이 없어 `WeeklyChangeRatio`는 항상 NaN(코인 283개 전부의 주간 등락률을 구하려면 종목별 개별 호출이 필요해 "벌크 조회" 원칙에 어긋난다). 시가총액(`Marcap`)도 업비트가 아예 제공하지 않는다(코인마다 발행량 개념도 달라 애매하기도 하다). 재조사해도 데이터 소스 자체의 한계라 안 바뀐다 — `pages/모니터링.py`는 코인 모드에서 시가총액 탭·주간 옵션을 아예 숨긴다.
- **코인 뉴스는 종목코드 기반 소스가 없다.** 네이버 금융처럼 코인 코드로 뉴스 목록을 주는 곳이 없어서(`m.stock.naver.com/crypto`는 API가 JS 번들에 숨은 SPA라 스크래핑 불가), `crypto_news.py`는 `search.naver.com` **키워드 검색**(코인 한글명)으로 우회한다. 검색 결과 카드에서 "네이버뉴스" 배지 링크를 기준으로 같은 카드 안의 원문 언론사 링크를 헤드라인으로 추출하는 구조적 휴리스틱을 쓴다(클래스명이 아니라 구조에 의존 — 이 페이지의 컴포넌트 클래스명은 `finance.naver.com`보다 훨씬 자주 바뀌는 것으로 보인다). 짧고 모호한 코인명은 검색 결과가 0건일 수 있다(실측: "리플" 0건, "리플 코인" 5건) — 호출부가 빈 결과를 정상 처리해야 한다. 기사 본문 조회·요약·감성 판정·감성 히스토리 기록은 원문이 결국 `n.news.naver.com`이라 `news.py` 함수를 그대로 재사용한다(중복 구현 없음).
- **업비트 캔들 API의 페이지네이션 커서(`to` 파라미터)는 반드시 `candle_date_time_utc`를 써야 한다 — `candle_date_time_kst`를 넘기면 업비트가 그걸 UTC로 해석해 9시간 어긋난, 사실상 "최신 데이터 근처"만 반복 조회하게 된다.** 일봉은 KST 라벨이 항상 정확히 `09:00:00`(=같은 날 `00:00 UTC`)이라 날짜 경계를 안 넘어서 이 버그가 겉으로 드러나지 않았다 — 분봉/시간봉(`get_minute_price()`)을 붙이면서 실측으로 발견했다. `crypto_loader.py`의 `get_price()`/`get_minute_price()` 둘 다 UTC 커서를 쓰도록 고정돼 있고, 회귀 테스트(`test_crypto_loader.py`)가 KST와 다른 UTC 값을 픽스처에 넣어 틀린 필드를 쓰면 바로 실패하게 만들어져 있다.
- **Plotly `make_subplots`에서 `specs`로 `colspan`을 쓰면 `shared_xaxes=True`가 조용히 무력화된다** — 각 행 x축의 `matches` 속성이 전부 `None`으로 남아 크로스헤어/줌이 행마다 따로 논다(직접 `fig.layout.xaxisN.matches`를 찍어봐야 드러나는 문제라 겉보기엔 정상처럼 보인다). `charts.py`의 매물대(Volume Profile) 2열 레이아웃에서 실측으로 발견했다 — `colspan`을 쓰는 모든 행에 대해 `fig.update_xaxes(matches="x", row=r, col=1)`을 수동으로 걸어야 한다. 같은 행 다른 열(가격 패널↔매물대)의 y축 연동은 `shared_yaxes=True`만으로 정상 동작했다(이건 버그 아님).
- **해외증시(나스닥) 스크리닝은 나스닥 자체 공개 API**(`api.nasdaq.com/api/screener/stocks?tableonly=true&exchange=NASDAQ&download=true&limit=10000`)**를 쓴다.** 로그인 불필요, `User-Agent` 헤더만 있으면 되고, `exchange=NASDAQ` 한 번의 요청으로 전종목(실측 4,118개)을 심볼/현재가/등락률/거래량/시가총액/섹터/업종까지 통째로 준다 — KRX 스냅샷·업비트 ticker와 동일하게 "요청 수가 종목 수와 무관하게 고정"된다. `download=true`가 없으면 응답 구조가 `data.table.rows`로 바뀌고 `volume`/`sector`/`industry` 필드가 아예 빠지니 반드시 붙여야 한다. **오늘자 스냅샷만 주고 N일 전 히스토리가 없어** `WeeklyChangeRatio`는 코인과 동일한 이유로 항상 NaN이다. 거래대금(Amount) 필드가 없어 `Volume × Close`로 근사한다. `marketCap`이 `"0.00"`으로 오는 종목(실측 588/4,118건, 대부분 스팩·워런트·라이츠)은 NaN 처리한다 — 진짜 시가총액 0인 종목은 없으므로 0이 "데이터 없음" 신호다. 반면 **개별 종목의 히스토리 조회(차트·예측용)는 이미 설치된 FinanceDataReader가 그대로 지원**한다 — `fdr.DataReader("AAPL")`이 1999년부터 나오는 걸 실측 확인했고, `data_loader.get_price()`/`find_symbol(market="NASDAQ")`는 이미 시장 파라미터를 받는 범용 함수라 새 코드 없이 그대로 재사용된다. 지금은 NASDAQ만 지원 — NYSE/유럽/중국은 `exchange` 파라미터를 바꾸면 되는 구조지만 아직 미구현.
- **차트에 그리기 도구(추세선 등)를 붙일 때 `streamlit-lightweight-charts` 같은 별도 컴포넌트를 새로 들이지 않는다.** 실제로 스모크 테스트해본 결과 이 패키지는 TradingView의 **무료** Lightweight Charts JS를 감싼 래퍼라 캔들/라인 등 시리즈 렌더링 API만 있고 그리기 도구 자체가 없다(그리기 도구는 TradingView의 유료 Advanced Charts 제품 전용 기능). 대신 Plotly(이미 프로젝트에 있음, 신규 의존성 없음)가 `dragmode="drawline"/"drawrect"/"drawcircle"/"drawopenpath"` + `modebar_add=["eraseshape", ...]`로 추세선/사각형/원/자유선 그리기와 지우개를 네이티브로 지원한다(`charts.build_chart(..., drawing_tools=True)`). 단, 피보나치 되돌림처럼 비율을 자동 계산해주는 도구는 Plotly에 없다(직선/도형만 가능). 그린 도형은 브라우저 세션에만 남고 Streamlit이 다른 위젯으로 스크립트를 재실행하면 사라진다(relayout 이벤트를 Python으로 되돌려 받으려면 추가 의존성이 필요 — 지금은 안 함).

## 가격 예측(ML) 기능 작성 규칙

- **투자 조언처럼 보이지 않게 한다.** "매수/매도 추천" 같은 문구를 쓰지 않는다. UI에 "참고용 통계 모델, 투자 조언 아님" 경고를 항상 표시한다.
- **정확도는 반드시 홀드아웃(학습에 안 쓴 구간)으로 검증한 값만 보여준다.** 학습 데이터 자체에 대한 적합도(in-sample)를 정확도인 것처럼 보여주지 않는다 — 실제보다 좋아 보이게 부풀리는 것이기 때문.
- 개별 종목의 일봉 데이터는 많아야 수백~수천 행이라 복잡한 모델(딥러닝 등)은 과적합 위험이 크다. 릿지 회귀처럼 단순한 모델을 우선한다.
- 종목마다 그때그때 새로 학습하고 모델을 저장해두지 않는다 — 데이터가 자주 바뀌고 종목별 모델을 다 저장/관리하는 비용이 이득보다 크다.

## UI/차트 컨벤션

- **언어: 코드의 docstring·주석·UI 라벨은 모두 한국어**로 작성한다 (변수/함수명은 영어 유지).
- **색상 관행**: 상승 = 빨강(`#ef4444`), 하락 = 파랑(`#3b82f6`) — 미국식(초록/빨강)과 반대이므로 새 차트를 만들 때 헷갈리지 않도록 주의.
  - **실측값은 원색, 예측·추정값은 파스텔**(`#cc6666`/`#668dcc`)로 구분한다. 파스텔은 원색과 색상(H)·명도(L)가 같고 채도(S)만 84~91%→50%로 낮춘 값이라 같은 계열로 읽힌다. 명도는 올리지 말 것 — 흰 배경에서 대비가 3:1 아래로 떨어진다. `app.py`의 `UP_COLOR`/`UP_SOFT` 등 상수와 `_change_html()` 헬퍼를 쓰고 hex를 새로 하드코딩하지 않는다.
  - **`st.metric`의 `delta`는 등락 표시에 쓰지 않는다.** 초록/빨강 조합만 지원해 국내 관행과 정반대가 된다. 값만 `st.metric`으로 그리고 등락은 `_change_html()`로 따로 그린다.
- 대시보드(웹)는 **Plotly**로 인터랙티브하게, 노트북은 **matplotlib + `plotting.setup()`**(한글 폰트 적용)으로 그린다.
- 스케일이 다른 여러 시계열(개별주 vs 지수)을 한 축에 겹쳐 그릴 때는 절대값이 아니라 **시작값 기준 리베이스**(`charts._rebase` 참고)해서 비교한다.

## 테스트 규칙

- `pytest` 사용. 외부 네트워크가 필요한 테스트는 `@pytest.mark.network`로 표시한다 (`pytest -m "not network"`로 오프라인 실행 가능).
- 오프라인 테스트는 합성 데이터로 검증한다 (`tests/test_dashboard.py`의 `_fake_ohlcv` 같은 헬퍼 패턴 참고). 실제 API 응답 형태를 가정하지 말고 함수의 계산 로직 자체를 검증한다.
- **Streamlit 앱을 수정하면**: 먼저 `streamlit.testing.v1.AppTest`로 헤드리스 검증(위젯 조작 시 예외 없는지)을 하고, 그다음 실제로 `streamlit run app.py`를 띄워 브라우저에서 확인한다. 스크린샷이 안 되는 상황이면 `get_page_text` / 콘솔 로그로 대체 확인하되, 반드시 실제 실행 결과를 눈으로(혹은 텍스트로) 확인하고 나서 완료로 보고한다 — 코드가 "그럴듯해 보인다"는 이유로 완료 처리하지 않는다.
- **검증용 서버를 띄우기 전에 포트 8501이 이미 쓰이고 있는지 확인한다.** 사용자가 직접 `streamlit run app.py`를 실행해 둔 세션일 수 있다 — 그 프로세스를 죽이지 말고, 검증은 다른 포트(예: 8502, 8503)로 별도 실행한다. 검증이 끝나면 자신이 띄운 포트의 프로세스만 정리한다.
- 새 기능을 추가하면 관련 테스트도 같이 추가한다. 커밋 전 `pytest` 전체 통과를 확인한다.
- **린트/포맷은 ruff를 쓴다** (`ruff.toml` 설정, `notebooks/`는 제외 — `%autoreload` 매직이 import보다 먼저 오는 게 정상이라). 커밋 전 `ruff check .` / `ruff format .` 확인.

## 모의투자(페이퍼 트레이딩) 기능

가상 현금 1억원으로 시작해 코스피/코스닥 + 해외증시(나스닥) + 코인(업비트) 종목을 자동 매매하고 실적을 공유하는 기능(2026-08-14, 국내 전용에서 3개 시장으로 확장 → 2026-08-22, 기본형 하나에서 기본형/공격적/모멘텀 3개 봇으로 확장). **구현 진행 중 — 설계 문서는 [PRD.md](./PRD.md)에 전체가 있다.** `src/portfolio.py`(원장), `src/trading_agent.py`(규칙 엔진+가드레일), `scripts/run_daily_trading.py`(①~④를 묶는 단일 진입점), `pages/모의투자.py`+`demo_app.py`(리포팅 UI), `.github/workflows/daily_trading.yml`+`requirements-trading.txt`(스케줄러)까지 완료. `workflow_dispatch` 수동 실행으로 실전 매매 1회 성공을 확인한 뒤(2026-08-14), **`schedule`(평일 19:30 KST 자동 cron)도 활성화 완료** — 이제 매 평일 자동으로 돈다(PRD.md 10장 진행 상태 참고, 다음은 며칠 관찰하는 6단계). **원장은 단일 원화(KRW)로 통합**돼 있다 — 해외증시는 매매 시점 환율로 원화 환산하고(`data_loader.get_usdkrw_rate()`), 코인은 업비트가 이미 원화라 그대로 쓴다. 원장 스키마에 market 컬럼을 두지 않고 `trading_agent.infer_market(code)`가 코드 패턴(6자리 숫자=KR, `"KRW-"` 접두사=COIN, 그 외=US)만으로 시장을 판별한다 — **코인은 1주 단위가 아니라 소수 수량**(고가 코인이 정수 수량으로는 예산 안에서 1개도 못 사는 문제 때문)이라 `TradeAction.quantity`는 `int | float`다. 리스크 규칙(`TRADING_RULES`)·보유 한도는 세 시장에 공통 적용(시장별 서브 한도 없음, PRD 10장 7단계 참고). GitHub Actions 관련 두 가지 함정을 겪었다: (1) `src/__init__.py`가 서브모듈을 eager import하면 `requirements-trading.txt`처럼 일부러 뺀 패키지(matplotlib)까지 끌려 들어와 죽는다 — 서브모듈은 각자 필요한 것만 명시적으로 import하고 `__init__.py`에서 미리 로드하지 않는다. (2) GitHub UI의 "Re-run jobs"는 새로 푸시한 커밋을 반영하지 않는다(그 run이 트리거된 시점의 커밋을 그대로 재사용) — 수정 후 재검증은 반드시 "Run workflow"로 새 run을 만들어야 한다. 여기는 앞으로 이 기능을 만들거나 건드릴 때 CLAUDE.md 규칙에 편입되는 요약만 적는다. 자세한 근거·수치·프롬프트 스키마는 PRD.md를 본다.

**용어**: "에이전트"가 두 가지 뜻으로 쓰인다 — (1) 이 기능을 만드는 Claude Code 서브에이전트(메인/코딩/검증, `클로드 에이전트/` 폴더), (2) 대시보드 안에서 실제로 도는 파이썬 모듈인 가격예측 에이전트/투자전략 에이전트. 이 섹션은 (2)를 다룬다.

- **단일 공유 포트폴리오** — 로그인 없음. 링크를 여는 모두가 같은 포트폴리오를 본다. 멀티유저는 범위 밖(PRD 9장).
- **봇 다중화 — 기본형/공격적/모멘텀 (2026-08-22, PRD 10장 8단계).** 같은 날 같은 시세·예측신호·뉴스감성을 세 봇이 공유하고(계산 비용이 커서 하루 한 번만 계산), "무엇을 살지" 판단하는 함수만 봇마다 분리돼 있다 — `trading_agent.py`의 `decide_trades`(기본형)/`decide_trades_aggressive`(진입 임계값 완화 + 보유 종목 추가매수(피라미딩) 허용)/`decide_trades_momentum`(예측 수익률이 아니라 SMA5·SMA20 추세 정렬이 1차 진입 조건, 청산에 추세 이탈 조건 추가)로 나뉜다. 세 봇은 `trading_agent.BOT_STRATEGIES`(`{bot_id: {label, decide_trades, rules}}`) 하나에 등록돼 있고, `run_daily_trading.py`/`pages/모의투자.py`가 이 레지스트리만 순회한다 — 봇을 추가/제거할 땐 이 등록만 바꾸면 된다. 가드레일(`apply_risk_guardrail`)은 전략이 아니라 한도 검증이라 세 봇이 공유한다. **원장은 봇마다 완전히 분리된 디렉터리**(`config.portfolio_dir_for(bot_id)` — "default"는 기존 `data/portfolio/` 그대로 유지해 라이브 원장을 마이그레이션하지 않고, 그 외는 `data/portfolio/{bot_id}/`에 1억원으로 새로 시작)이며, `portfolio.py`의 모든 함수가 `portfolio_dir: Path | None = None`을 받아 이를 지원한다. **`.gitignore`의 `!data/portfolio/*.csv`는 한 단계 깊이만 un-ignore해서 봇 하위 디렉터리의 CSV를 계속 무시하는 버그였다** — `!data/portfolio/**/*.csv`로 재귀 매치하도록 고쳤다(새 원장 경로를 추가할 땐 `git add --dry-run <path>`로 실제 추적 대상이 되는지 먼저 확인할 것).
- **실행 환경과 조회 환경을 완전히 분리한다.** 매매는 유저의 대시보드 방문과 무관하게 **GitHub Actions**(`.github/workflows/daily_trading.yml`)가 매일 1회(평일 19:30 KST 전후 — GitHub 커밋 이력으로 실측한 값, 아래 참고) 독립적으로 실행한다 — Streamlit 앱 안에 "방문 시 매매 실행" 같은 트리거를 절대 넣지 않는다. Streamlit 쪽(`pages/모의투자.py`, `demo_app.py`에서만 노출 — 기존 `app.py`는 수정하지 않는다)은 GitHub Actions가 커밋해 둔 원장을 **읽기만** 한다. 장중(09:00~17:00) 내내 반복 실행하는 것은 검토했지만 채택하지 않았다 — 이 프로젝트 데이터가 일봉 기준이라 장중에 다시 돌려도 새 정보가 없고, 진짜 장중 실시간은 지금 없는 실시간 시세 소스가 새로 필요해서 스코프가 커진다 (PRD 2장·9장 "검토했지만 채택하지 않은 것"/"향후 확장" 참고).
- **신규 파일 계획**: `src/portfolio.py`(원장 읽기/쓰기), `src/trading_agent.py`(매매 규칙 엔진 + 리스크 가드레일), `scripts/run_daily_trading.py`(①~④를 묶는 단일 진입점, GitHub Actions와 로컬 수동 실행 양쪽에서 호출), `pages/모의투자.py`(성과 리포팅 UI), `demo_app.py`(신규 진입점 — `st.navigation`으로 기존 `app.py`와 `pages/모의투자.py`를 묶는다. **기존 `app.py`는 수정하지 않는다** — 지금 공유 중인 배포가 그대로 유지되고, 데모 대시보드는 별도 배포로 얹는다), `.github/workflows/daily_trading.yml`(스케줄러).
- **매매 판단은 (V1 한정) LLM이 아니라 규칙 기반(결정론적) 엔진이다.** 원래 설계는 Claude API 기반 판단형 에이전트였으나, 유료 서비스라 지금은 키 발급 없이 진행하기로 했다 — 예측치·RSI·뉴스감성에 임계값을 적용해 매매를 결정한다. **외부 API 호출이 전혀 없고, 신규 의존성도 없다**(표준 라이브러리 `dataclasses`만 사용) — `anthropic`/`pydantic`은 설치는 확인해뒀지만 지금은 안 쓴다. LLM 원안은 PRD 9.1에 보존돼 있고, 나중에 되살릴 때는 `trading_agent.py`의 `decide_trades()` 내부 구현만 바꾸면 된다(그때 가서 Claude API 코드를 쓸 때는 `claude-api` 스킬을 먼저 로드해서 모델 ID·파라미터를 검증할 것 — 기억에 의존하지 않는다, 자주 바뀐다).
- **매매 판단 로직의 출력은 제안일 뿐, 최종 승인은 항상 별도의 결정론적 코드(`trading_agent.py`의 리스크 가드레일)가 한다.** 종목당 최대 비중·동시보유 한도·현금 한도를 판단 로직이 지켰다고 가정하지 않고 다시 검증한다 — 둘 다 코드지만 "판단"과 "한도 강제"를 분리해두면 각각 독립적으로 테스트할 수 있다. 매매 규칙과 가드레일은 반드시 같은 상수 딕셔너리(`TRADING_RULES`) 하나를 참조한다 — 두 군데 따로 하드코딩하면 반드시 어긋난다. 그날 매매는 그날 종가로 체결된 것으로 가정한다(실시간 시세가 없으므로).
- **영속성 — 이 기능에서 가장 중요한 제약**: Streamlit Community Cloud는 GitHub에서 재배포될 때마다 컨테이너를 새로 만들어, git에 커밋되지 않은 로컬 파일(포트폴리오 상태 포함)은 다음 배포 때 사라진다. 그래서 `data/portfolio/*`는 뉴스 감성 히스토리(`data/raw/news_sentiment/`)와 **동일하게** `.gitignore` 예외로 넣는다(`*.parquet`·`*.csv` 줄보다 **뒤에** 와야 적용된다). 커밋 주체는 **GitHub Actions 워크플로 자신**이다 — 워크플로 YAML에 `permissions: contents: write`를 주면 Actions가 기본 제공하는 `GITHUB_TOKEN`만으로 push까지 되므로, 별도 PAT를 발급받아 시크릿에 저장할 필요가 없다. Streamlit 앱 프로세스에는 쓰기 권한이 있는 토큰을 아예 두지 않는다.
- **원장 파일만은 parquet이 아니라 CSV로 쓴다.** 다른 데이터는 parquet이지만 원장은 예외이며, 이건 취향이 아니라 이 저장소가 이미 겪은 문제 때문이다 — 위 "알려진 제약"에 적힌 대로 **parquet은 같은 내용을 다시 써도 바이트가 달라져서**, git 추적 대상인 뉴스 감성 로그는 `_already_logged()`라는 우회 로직을 따로 만들어야 했다(없으면 켤 때마다 워킹 트리가 dirty해지고 무의미한 diff 커밋이 쌓인다). 원장을 parquet으로 하면 같은 함정을 반복하게 되고, 매일 커밋하므로 통짜 바이너리 blob이 매일 쌓인다. CSV는 내용이 같으면 바이트도 같고, git이 줄 단위로 저장하며, GitHub 웹에서 거래 이력을 그대로 읽을 수 있다. 원장은 1년 돌려도 수백 행이라 parquet의 이점이 의미 없다.
- **시가평가는 매매와 별개로 매 거래일 기록한다.** 매매 0건인 날에도 보유종목 종가가 바뀌면 총자산이 변하므로, `equity_history.csv`는 거래 유무와 무관하게 매 거래일 한 행씩 쌓인다 — "매매한 날만" 기록하면 자산 추이 차트에 구멍이 생긴다. 시세 조회가 실패한 날도 매매만 건너뛰고 시가평가는 기록한다.
- **실패 시 원장이 반쯤 갱신된 상태로 남으면 안 된다.** 원장 파일들을 메모리에서 모두 갱신한 뒤 한 번에 쓰고, 그 다음에 `state.json`의 `last_run_date`를 쓴다. 중간 실패 시엔 아무것도 안 쓴 상태가 되어야 한다 — 체결만 반영되고 `last_run_date`를 못 쓰면 다음 실행이 같은 날 매매를 두 번 한다.
- **비밀값**: V1은 규칙 기반이라 GitHub Actions Secrets에 새로 등록할 키가 **없다** (`ANTHROPIC_API_KEY`는 9.1 LLM 전환 시에나 필요 — 그때 위치는 GitHub Actions Secrets, Streamlit Cloud Secrets에는 두지 않는다). Streamlit Cloud Secrets에는 여전히 조회용 `DART_API_KEY`만 있으면 된다.
- **면책 조항 강화**: 기존 "가격 예측 기능은 투자 조언처럼 보이지 않게 한다" 규칙이 이 기능에서는 더 중요하다 — 실제로 "매수/매도"를 자동 **실행**하기 때문. "실제 금전 거래가 아닌 시뮬레이션 게임"과 "장중 실시간이 아니라 하루 1회 장마감 후 자동 매매" 두 문구를 상시 노출한다.
- **테스트**: `trading_agent.py`는 외부 API가 없는 순수 함수라 모킹이 필요 없다 — 합성 신호 데이터로 매수/매도/보유 각 조건의 **경계값**(임계값 바로 위·아래)을 촘촘히 단위테스트한다. `scripts/run_daily_trading.py`는 로컬에서 수동 실행해서 원장 갱신 로직을 GitHub Actions에 올리기 전에 먼저 확인한다.

## 새 기능 추가 체크리스트

1. 필요한 데이터가 `data_loader.py`/`screener.py`에 이미 있는지 확인. 없으면 캐시 규칙을 지키며 추가.
2. 대량 조회가 필요하면 종목별 반복이 아니라 벌크 방법을 먼저 찾는다.
3. 계산/데이터 로직은 `src/` 모듈에, `app.py`는 그걸 얇게 호출만 한다.
4. 새 외부 라이브러리가 필요하면 추가 전에 스모크 테스트로 실동작을 확인한다.
5. 오프라인/네트워크 테스트를 분리해서 작성한다.
6. Streamlit 화면이면 AppTest → 실제 서버 실행 → 브라우저(또는 텍스트) 확인 순서로 검증한다.
7. `requirements.txt`와 `README.md`를 새 의존성/사용법 기준으로 갱신한다.
