# 주가 데이터 수집 · 분석 작업환경

KOSPI/KOSDAQ, NASDAQ 등의 주가 데이터를 [FinanceDataReader](https://github.com/FinanceData/FinanceDataReader)로 가져와 분석하기 위한 환경.

## 환경

| 항목 | 버전 |
|---|---|
| Python | 3.14.7 |
| pandas / numpy | 3.0.5 / 2.5.2 |
| FinanceDataReader | 0.9.202 |
| streamlit | 1.61.1 |
| plotly / matplotlib | 6.9.0 / 3.11.1 |

> pandas 3.0 은 Copy-on-Write 가 기본이고 `df.append()`, `fillna(method=...)` 등 구 API가 제거됐습니다. 인터넷의 오래된 예제를 붙여넣을 때 주의하세요.

## 시작하기

```powershell
# 패키지 설치 (이미 설치되어 있음 — 새 환경에서만)
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 테스트
.venv\Scripts\python.exe -m pytest              # 전체
.venv\Scripts\python.exe -m pytest -m "not network"   # 오프라인

# 스크리닝 대시보드
.venv\Scripts\streamlit.exe run app.py

# 노트북
.venv\Scripts\jupyter-lab.exe
```

VS Code에서는 인터프리터가 `.venv` 로, 노트북 커널이 **Python (vsc_workspace)** 로 잡혀 있습니다.

## 구조

```
├── app.py                스크리닝 대시보드 (Streamlit, 단독 배포 — 아래 pages/모니터링.py와 별개로 그대로 유지)
├── src/
│   ├── config.py         경로·기본값·자주 쓰는 티커(INDICES, MACRO)
│   ├── cache_utils.py    parquet 캐시 공용 유틸
│   ├── data_loader.py    FDR 래퍼 + parquet 캐시
│   ├── screener.py       전종목 일간·주간 등락률 스크리닝
│   ├── indicators.py     기술적 지표(SMA/EMA/RSI/MACD/볼린저) + 성과지표(CAGR/샤프/MDD)
│   ├── charts.py         Plotly 인터랙티브 차트(캔들+이평선+볼린저+지수오버레이+RSI/MACD)
│   ├── news.py           네이버 금융 종목 뉴스 스크래핑 + 요약(리드 문단 발췌) + 감성 히스토리 누적 기록
│   ├── sentiment.py      키워드 기반 감성(상승지표) 판정
│   ├── dart.py           DART 전자공시 OpenAPI 연동 (API 키 필요)
│   ├── predictor.py      가격 예측 (기본: 릿지 회귀 / "정확한 예측": Ridge·RandomForest·GradientBoosting 교차검증 비교, 종목별 그때그때 학습 + 홀드아웃 검증)
│   ├── crypto_loader.py  업비트 공개 API 래퍼 — data_loader.py의 코인 버전 (OHLCV 컬럼 계약 동일)
│   ├── crypto_screener.py 업비트 KRW 마켓 전종목 시세 벌크 스크리닝 — screener.py의 코인 버전
│   ├── crypto_news.py    코인 뉴스 (네이버 뉴스 키워드 검색) — 이후 파이프라인은 news.py 재사용
│   ├── portfolio.py      모의투자 원장(현금·보유종목·거래이력·자산추이) 읽기/쓰기 — CSV
│   ├── trading_agent.py  모의투자 매매 규칙 엔진 + 리스크 가드레일 (외부 API 없는 결정론적 로직)
│   └── plotting.py       matplotlib 한글 폰트 및 기본 스타일 (노트북용)
├── pages/
│   ├── 모니터링.py        국내증시/코인 전환형 스크리닝·차트·예측·뉴스 (app.py의 확장판, demo_app.py 전용)
│   └── 모의투자.py        모의투자 성과 리포팅 화면 (원장 읽기 전용, demo_app.py 전용)
├── scripts/
│   └── run_daily_trading.py  모의투자 일일 매매 실행 진입점 (GitHub Actions가 매일 호출)
├── demo_app.py            pages/모니터링.py + pages/모의투자.py를 묶은 별도 데모 진입점
├── .github/workflows/
│   └── daily_trading.yml  모의투자 일일 매매 스케줄러
├── notebooks/
│   └── 01_quickstart.ipynb
├── data/
│   ├── raw/              원본 저장용
│   ├── cache/             자동 parquet 캐시 (gitignore)
│   └── portfolio/         모의투자 원장 (CSV — git으로 추적, gitignore 예외)
├── output/                차트·리포트 산출물 (gitignore)
└── tests/
```

## 주가 요약 대시보드

```powershell
.venv\Scripts\streamlit.exe run app.py

# 또는 국내증시/코인을 좌상단에서 전환할 수 있는 확장판 (아래 "코인 모드" 참고)
.venv\Scripts\streamlit.exe run demo_app.py
```

아래 설명은 `app.py` 기준이며, `demo_app.py`의 "모니터링" 페이지에서 **국내증시**를 선택했을 때도 완전히 동일합니다(`app.py`는 수정 없이 별도 배포로 계속 유지되고, `demo_app.py`는 그 기능을 코인까지 확장한 새 페이지를 얹은 것뿐입니다).

브라우저(기본 http://localhost:8501)에서:

- **좌측 사이드바**: 시장(전체/KOSPI/KOSDAQ), 표시 개수(기본 30), 최소 시가총액·거래량 필터. 이 조건은 아래 세 탭에 공통으로 적용됩니다.
- **상단 테이블 (조건별 3개 탭)**: 같은 종목 집합을 세 가지 기준으로 각각 상위 N개(기본 30) 보여줍니다. 행을 클릭하면 아래 차트가 해당 종목으로 바뀝니다.
  - 📈 **상승률** — 일간/주간 등락률. 이 탭에만 정렬 기준(일간·주간)과 방향(상승·하락) 선택이 있습니다(나머지 두 탭은 "상위"가 곧 내림차순이라 불필요).
  - 💰 **거래대금** — 그날 실제로 돈이 가장 많이 오간 종목.
  - 🏢 **시가총액** — 덩치가 가장 큰 종목.
- **종목 상세**: 먼저 **시세 요약**(현재가·시가·고가·저가·전일 종가·일중 변동폭·거래량·거래대금)을 원 단위로 표시하고, 전일대비는 상승=빨강/하락=파랑으로 표기합니다. 이어서 캔들스틱 + 조회 기간(1개월~전체) + 지수 비교선(코스피·코스닥·나스닥 등, 시작가 기준 리베이스). 차트 우측에 지표 체크박스(SMA5/20/60/120, 볼린저밴드, 거래량, RSI, MACD) — 체크하면 즉시 차트에 표시됩니다. 기간 성과지표(수익률·CAGR·변동성·샤프·MDD)
- **가격 예측**: 성과지표 아래, 뉴스/공시 위에 위치. 다음 거래일 종가와 5거래일(약 1주일) 후 종가를 릿지 회귀 모델로 추정하고, 홀드아웃 검증 기준 MAE·MAPE·방향 적중률을 함께 보여줍니다. 바로 아래 **"🎯 정확한 예측 (심층 학습)"** 버튼을 누르면 같은 두 시점(다음 거래일 / 5거래일 후)을 여러 모델을 비교해 더 무겁게 다시 추정합니다(자세한 방법론은 아래 "가격 예측 방식" 참고).
- **뉴스 & 공시**: 차트 아래에서 확인. 📰 뉴스 탭은 네이버 금융에서 해당 종목 기사를 가져와 제목·요약과 함께 우측에 **상승지표**(긍정/중립/부정 + 점수)를 표시합니다. 📋 공시 탭은 DART 공식 공시 목록(API 키 필요 — 아래 설정 참고)
- 테이블에 없는 종목은 "종목코드 또는 이름으로 검색"에서 직접 조회 가능

### 전종목 스크리닝이 빠른 이유

KOSPI+KOSDAQ 약 2,900종목을 개별 조회하면 수 분이 걸립니다. 대신 FinanceDataReader가 실제로 쓰는 KRX 일자별 스냅샷 미러(GitHub: `FinanceData/fdr_krx_data_cache`)를 이용해, 최근 거래일과 비교 기준일(기본 1주 전) 스냅샷 **딱 2번만** 내려받아 종목코드로 조인합니다. 요청 수가 스크리닝 대상 종목 수와 무관하게 항상 2회로 고정됩니다.

> `pykrx`도 검토했지만 최신 버전은 벌크 조회에 KRX 로그인(KRX_ID/KRX_PW)을 요구해 제외했습니다.

### 뉴스 & 공시 설정

- **뉴스**는 별도 설정 없이 바로 동작합니다 (네이버 금융 비공식 스크래핑).
- **상승지표**는 기사 제목+본문에 등장하는 금융 긍정/부정 키워드 빈도로 계산합니다(`src/sentiment.py`의 `POSITIVE_WORDS`/`NEGATIVE_WORDS`). **문맥이나 반어법은 이해하지 못하는 참고용 보조지표**입니다 — 매매 판단의 근거로 쓰지 마세요.
- **공시(DART)**를 쓰려면 API 키가 필요합니다:
  1. https://opendart.fss.or.kr 에서 무료로 키 발급 (본인 이메일 인증 필요)
  2. 프로젝트 루트에 `.env` 파일을 만들고 `DART_API_KEY=발급받은키` 추가 (`.env`는 gitignore되어 커밋되지 않음)
  3. 앱을 재시작하면 자동으로 로드됩니다. 키가 없으면 공시 탭에 안내 메시지만 표시되고 나머지 기능은 정상 동작합니다.

### 가격 예측 방식

**⚠️ 참고용 통계 모델이며 투자 조언이 아닙니다.**

- 종목별로 저장된 모델 없이 조회할 때마다 그 종목의 과거 데이터로 새로 학습합니다(`predictor.py`). 개별 종목 일봉은 많아야 수백~수천 행이라 복잡한 모델은 과적합 위험이 커서 **릿지 회귀**(선형 모델)를 씁니다.
- 피처: 1/5/20일 수익률, SMA5/20/60 대비 괴리율, RSI14, 20일 변동성, 거래량 변화율, MACD 히스토그램, 볼린저 %B, 뉴스 감성.
- **정확도는 학습에 안 쓴 최근 구간(홀드아웃)으로 검증**해서 MAE(평균절대오차)·MAPE(평균절대비율오차)·방향 적중률(상승/하락 방향을 맞춘 비율)을 그대로 보여줍니다 — 학습 데이터로 자체 채점하는 식으로 부풀리지 않습니다. 방향 적중률이 50%에 가깝게 나오는 게 오히려 정상입니다(단순 선형 모델로 며칠 뒤 방향을 맞추는 건 원래 어렵습니다).
- 조회 기간이 짧으면(기본 최소 120거래일 필요) 학습을 못 하고 안내 메시지만 표시됩니다 — 기간을 늘려주세요.
- **예측값은 파스텔 톤으로 표시됩니다.** 등락 색은 캔들 차트와 같은 계열(상승=빨강/하락=파랑)이지만, 채도를 50%로 낮춘 흐린 색(`#cc6666`/`#668dcc`)을 씁니다. 실제 체결된 시세는 원색(`#ef4444`/`#3b82f6`)이라, **진한 색 = 확정된 사실 / 흐린 색 = 추정치**로 한눈에 구분됩니다.
- **"🎯 정확한 예측 (심층 학습)" 버튼**: 기본 릿지 모델은 그대로 둔 채, 버튼을 눌렀을 때만 도는 더 무거운 버전입니다(`predictor.train_and_predict_advanced`). 차이점:
  - 차트에 선택한 조회 기간과 무관하게 **항상 보유한 전체 기간**으로 다시 불러와 학습합니다(최소 200거래일 필요).
  - 피처를 늘립니다(EMA12/26 괴리율, 볼린저 밴드폭, 일중 변동폭, 종가-시가 갭, 거래량 20일 평균 대비, RSI 기울기, 요일 주기성분, 뉴스 긍정/부정 기사 수 등).
  - Ridge / RandomForest / GradientBoosting 세 모델을 `TimeSeriesSplit` 교차검증으로 비교해 가장 좋은 것을 고릅니다.
  - 화면에 보이는 정확도(MAE/MAPE/방향 적중률)는 **모델 선정에 전혀 관여하지 않은 마지막 홀드아웃 구간**으로만 잰 값입니다 — 교차검증 점수는 모델을 고르는 데만 쓰고 정확도로 보여주지 않습니다.
  - 여러 모델을 학습하고 교차검증까지 돌기 때문에 기본 예측보다 느립니다.
- **뉴스 감성 피처의 한계**: 뉴스 소스는 과거 특정 날짜의 감성을 조회할 방법이 없고 최신 기사만 볼 수 있습니다. 그래서 앱을 실행할 때마다(뉴스를 조회할 때마다) 그날의 감성 통계(평균 점수 + 긍정·부정 기사 수 + 총 기사 수)를 종목별로 `data/raw/news_sentiment/{종목코드}.parquet`에 누적 기록하고, 기록 이전 과거 구간은 중립(0)으로 채웁니다. 즉 **오늘부터 실제 히스토리가 쌓이기 시작**하는 구조라, 초반에는 이 피처의 영향력이 거의 0입니다(모델 상세의 회귀계수로 확인 가능) — 매일 대시보드를 쓸수록 점점 유의미해집니다.
  - ⚠️ 이 로그 파일은 `data/raw/`에 있어 현재 `.gitignore`(다른 캐시처럼 "재생성 가능"으로 분류)에 걸려 있습니다. 하지만 이건 한 번 지우면 과거로 되돌아가 다시 쌓을 수 없는 유일한 데이터입니다 — 나중에 git을 초기화하게 되면 `data/raw/news_sentiment/`만 예외로 추적할지 고려하세요.

### 코인 모드 (모니터링 페이지 전용)

`demo_app.py`의 "모니터링" 페이지 좌상단에서 **국내증시 / 코인**을 전환할 수 있습니다. 코인
모드는 위 기능과 최대한 동일하게 맞췄지만, 데이터 소스가 완전히 달라 아래 차이가 있습니다.

- **시세**: [업비트](https://upbit.com) 공개 API(`src/crypto_loader.py`). 로그인·키 불필요. KRW
  마켓 전종목 시세를 요청 1번으로 벌크 조회하고(`src/crypto_screener.py`), 개별 코인의 일봉은
  `data_loader.get_price()`와 똑같은 OHLCV 형태로 반환되어 지표·차트·예측 로직을 그대로 씁니다.
- **시가총액 탭·주간 등락률이 없습니다.** 업비트 API가 이 둘을 벌크로 제공하지 않아서입니다
  (코인마다 발행량 개념도 달라 시가총액 자체가 애매하기도 합니다) — 코인은 24시간 등락률과
  거래대금 탭만 있습니다.
- **공시(DART) 탭이 없습니다.** 코인에는 해당 개념이 없어 탭 자체가 안 뜹니다.
- **뉴스**는 종목코드 기반이 아니라 **네이버 뉴스 키워드 검색**(코인 한글명)입니다
  (`src/crypto_news.py`). 짧고 모호한 코인명은 검색 결과가 0건일 수 있습니다(예: "리플"). 검색
  결과가 나오면 그 이후(본문 조회·요약·상승지표 판정·감성 히스토리 기록)는 주식과 완전히
  동일한 파이프라인(`news.py`)을 재사용합니다 — 감성 로그는 `data/raw/news_sentiment/KRW-BTC.parquet`처럼
  업비트 마켓코드로 저장되어 종목코드(6자리 숫자) 로그와 겹치지 않습니다.
- **거래량 단위**가 "주"가 아니라 코인 자체 단위입니다(예: `62.4944BTC`).
- 화면 전환 시 이전에 보던 종목이 다른 시장으로 새지 않도록, 선택 상태는 국내증시/코인 각각
  따로 기억됩니다.

## 모의투자 (페이퍼 트레이딩)

**⚠️ 실제 금전 거래가 아닌 시뮬레이션입니다. 투자 조언이 아닙니다.** 가상 현금 1억원으로 시작해 KOSPI/KOSDAQ 종목을 규칙 기반(결정론적) 엔진이 자동으로 매매합니다. 상세 설계는 [PRD.md](./PRD.md) 참고.

```powershell
# 기존 app.py는 그대로 두고, 모의투자 화면을 더한 별도 데모 진입점으로 확인
.venv\Scripts\streamlit.exe run demo_app.py

# 로컬에서 매매 로직만 수동 실행 (원장을 건드리지 않는 dry-run)
.venv\Scripts\python.exe scripts\run_daily_trading.py --dry-run
```

- **실행과 조회가 분리되어 있습니다.** 실제 매매는 GitHub Actions(`.github/workflows/daily_trading.yml`)가 매 평일 1회(장 마감 후, 19:30 KST 전후) 자동으로 실행하고 원장(`data/portfolio/`)을 직접 커밋합니다 — Streamlit 앱을 누가 방문하는지와 무관합니다. `demo_app.py`의 "모의투자" 페이지는 그 원장을 **읽기만** 합니다.
- **"기본형"/"공격적"/"모멘텀" 세 봇이 같은 날 나란히 매매합니다** — 판단 로직만 다르고(`src/trading_agent.py`의 `decide_trades`/`decide_trades_aggressive`/`decide_trades_momentum`), 원장은 봇마다 완전히 분리(`data/portfolio/`, `data/portfolio/aggressive/`, `data/portfolio/momentum/`)돼 각자 1억원으로 시작합니다. "모의투자" 페이지는 봇별 탭 + 세 봇을 비교하는 "성과 비교" 탭으로 구성됩니다.
- 매매 판단은 예측 수익률·RSI·뉴스 감성에 임계값을 적용하는 규칙 엔진(`src/trading_agent.py`)이고, 제안된 매매는 별도 리스크 가드레일(종목당 비중·동시보유·하루 거래횟수·현금 한도)이 한 번 더 검증해 위반 시 거부합니다.
- 워크플로는 처음엔 `workflow_dispatch`(수동 버튼)로만 동작하도록 커밋돼 있습니다 — GitHub 저장소에서 수동 실행으로 커밋·푸시가 정상인지 확인한 뒤 `daily_trading.yml`의 `schedule` 주석을 해제해야 매일 자동으로 돕니다.
- GitHub Actions는 `requirements.txt` 전체가 아니라 `requirements-trading.txt`(매매 스크립트에 실제로 필요한 패키지만)를 설치합니다.

## 사용 예

```python
from src import data_loader as dl, indicators as ind, plotting

plotting.setup()  # 한글 폰트 적용

kospi = dl.get_price("KOSPI", "2020-01-01")  # 별칭 사용
samsung = dl.get_price("005930", "2020-01-01")  # 6자리 종목코드
aapl = dl.get_price("AAPL", "2020-01-01")  # 해외 티커

closes = dl.get_closes(["KOSPI", "NASDAQ", "AAPL"], "2023-01-01").dropna()
ind.summary(kospi["Close"])  # 수익률·변동성·샤프·MDD
ind.add_all(samsung)  # 지표 열 일괄 추가

dl.find_symbol("삼성")  # 종목명으로 코드 검색
```

### 심볼 표기

- **한국 종목**: 6자리 코드 — `005930`(삼성전자), `000660`(SK하이닉스)
- **지수**: `KS11`(코스피), `KQ11`(코스닥), `IXIC`(나스닥), `US500`(S&P500) — 또는 `config.INDICES` 의 별칭 `"KOSPI"`, `"NASDAQ"` 등
- **해외 종목**: 티커 그대로 — `AAPL`, `NVDA`
- **환율**: `USD/KRW`

## 캐시

`get_price` / `get_listing` 결과는 `data/cache/` 에 parquet으로 저장되고 6시간(`config.CACHE_TTL_SEC`) 동안 재사용됩니다. 뉴스/공시는 더 자주 바뀌므로 30분(`config.NEWS_CACHE_TTL_SEC`)만 캐시합니다. 강제 갱신은 `use_cache=False`, 전체 삭제는 `dl.clear_cache()` (사이드바의 "새로고침" 버튼도 이걸 호출합니다).

## 아직 없는 것

- 백테스트 엔진
- 재무제표·수급 데이터 (PER/PBR, 투자자별 매매동향 등)
- 나스닥/미국 종목 스크리닝 (현재 스크리너는 KRX 전용)
