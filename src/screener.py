"""KOSPI+KOSDAQ 전종목 일간·주간 등락률 스크리닝.

종목별로 개별 조회하면(~2,900회 호출) 너무 느리고 서버에도 부담이 크다.
대신 FinanceDataReader가 실제로 사용하는 KRX 일자별 스냅샷 미러
(GitHub: FinanceData/fdr_krx_data_cache)를 날짜를 지정해 두 번
(최근 거래일 / N영업일 전) 내려받아 Code 기준으로 조인하는 방식을 쓴다.
요청 수가 전종목 스크리닝 범위와 무관하게 항상 2회로 고정된다.
"""

from __future__ import annotations

import pandas as pd

from .cache_utils import cache_path, is_fresh

_GH_LISTING_URL = (
    "https://raw.githubusercontent.com/FinanceData/fdr_krx_data_cache"
    "/refs/heads/master/data/listing/krx/{date}.csv"
)

_RESULT_COLS = [
    "Code",
    "Name",
    "Market",
    "Close",
    "DailyChangeRatio",
    "WeeklyChangeRatio",
    "MonthlyChangeRatio",
    "Volume",
    "Amount",
    "Marcap",
]

_MONTHLY_DAYS_BACK = 30  # "월간" 비교 기준 — days_back(주간용, 기본 7)과 별개로 고정

# 미러가 장 마감 전/휴장일에도 그날짜 CSV를 미리 만들어 두는데, 그 파일은 가격 컬럼이
# 전부 "-"(문자열)이다. dtype이 숫자가 아니라 문자열로 읽히므로 아래 컬럼들을 항상
# 숫자로 강제 변환한다 (거래정지 종목 몇 개만 "-"인 정상 스냅샷도 같은 처리로 흡수된다).
_NUMERIC_COLS = ("Close", "Open", "High", "Low", "Changes", "ChagesRatio", "Volume", "Amount", "Marcap")


def _fetch_csv(url: str) -> pd.DataFrame:
    """스냅샷 CSV 한 장을 그대로 읽는다.

    네트워크 경계를 이 함수 하나로 좁혀 둬서, 테스트가 이것만 대체하면
    오프라인으로 스냅샷 파싱 로직을 검증할 수 있다.
    """
    return pd.read_csv(url, index_col=0, dtype={"Code": str})


def _snapshot_on(date: pd.Timestamp) -> pd.DataFrame | None:
    """해당 날짜의 KRX 전종목 스냅샷. 휴장일 등으로 없거나 아직 안 채워졌으면 None.

    **종가가 하나도 없는 "더미 스냅샷"도 None으로 취급한다.** 미러는 장 마감 전이나
    휴장일에도 그날짜 파일을 미리 만들어 두는데, 그 파일은 행 수만 정상(~2,900개)이고
    `Close`가 전부 `"-"`다(2026-09-25·10-09 실측). 예전에는 "비어 있지 않으면 유효"로
    판단해 이 파일을 최신 스냅샷으로 집어, 등락률 계산에서 문자열끼리 빼다가
    `TypeError`로 죽었다 — 자동매매 워크플로가 이 때문에 반복 실패했다.
    None을 주면 `_nearest_snapshot()`이 하루 더 과거로 내려가 직전 거래일을 쓴다.
    """
    url = _GH_LISTING_URL.format(date=date.strftime("%Y-%m-%d"))
    try:
        df = _fetch_csv(url)
    except Exception:
        return None
    df = df.reset_index(drop=True)
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Close" not in df.columns or not df["Close"].notna().any():
        return None
    return df


def _nearest_snapshot(around: pd.Timestamp, max_back_days: int = 10):
    """around 기준 가장 가까운 과거(포함) 거래일의 (날짜, 스냅샷)을 찾는다."""
    for i in range(max_back_days):
        d = around - pd.Timedelta(days=i)
        snap = _snapshot_on(d)
        if snap is not None and not snap.empty:
            return d, snap
    return None, None


_KRX_MARKETS = ("KOSPI", "KOSDAQ")  # KONEX(초소형 시장)는 기본 제외


def screen(
    market: str = "ALL",
    days_back: int = 7,
    min_marcap: float = 0.0,
    min_volume: float = 0.0,
    use_cache: bool = True,
) -> pd.DataFrame:
    """일간·주간·월간 등락률이 포함된 전종목 스크리닝 테이블을 반환한다.

    market: 'ALL'(KOSPI+KOSDAQ) | 'KOSPI' | 'KOSDAQ'
    days_back: "주간" 비교 기준 며칠 전(달력 기준)인지. 기본 7일 = 1주일 전.
    min_marcap: 이 시가총액(원) 미만인 종목은 제외. 초소형주 노이즈 제거용.
    min_volume: 이 거래량 미만(거래정지 등 0거래량 포함)인 종목은 제외.

    "월간" 비교는 days_back과 별개로 항상 30일 전 스냅샷을 쓴다(파라미터화하지 않음 —
    호출부가 굳이 바꿀 이유가 없다). 30일 전 스냅샷이 없으면(신규 상장 등) 해당 종목만
    MonthlyChangeRatio가 NaN이 되고 전체 조회가 실패하지는 않는다.
    """
    key = f"{market}|{days_back}|{min_marcap}|{min_volume}"
    path = cache_path("screen2", key)  # v2: MonthlyChangeRatio 추가로 스키마가 바뀌어 kind를 분리
    if use_cache and is_fresh(path):
        return pd.read_parquet(path)

    latest_date, latest = _nearest_snapshot(pd.Timestamp.today())
    if latest is None:
        raise RuntimeError("최근 거래일 스냅샷을 찾지 못했습니다 (네트워크 확인 필요)")

    _, past = _nearest_snapshot(latest_date - pd.Timedelta(days=days_back))
    if past is None:
        raise RuntimeError(f"{days_back}일 전 근처 스냅샷을 찾지 못했습니다")

    _, past_month = _nearest_snapshot(latest_date - pd.Timedelta(days=_MONTHLY_DAYS_BACK))

    merged = latest.merge(
        past[["Code", "Close"]].rename(columns={"Close": "ClosePrev"}),
        on="Code",
        how="left",
    )
    merged["WeeklyChangeRatio"] = (merged["Close"] - merged["ClosePrev"]) / merged["ClosePrev"] * 100

    if past_month is not None:
        merged = merged.merge(
            past_month[["Code", "Close"]].rename(columns={"Close": "ClosePrevMonth"}),
            on="Code",
            how="left",
        )
        merged["MonthlyChangeRatio"] = (
            (merged["Close"] - merged["ClosePrevMonth"]) / merged["ClosePrevMonth"] * 100
        )
    else:
        merged["MonthlyChangeRatio"] = float("nan")

    merged = merged.rename(columns={"ChagesRatio": "DailyChangeRatio"})

    markets = _KRX_MARKETS if market == "ALL" else (market,)
    merged = merged[merged["Market"].isin(markets)]
    # 거래정지 등으로 종가가 없는 행은 제외한다 — 이 테이블을 쓰는 모든 호출부(차트·
    # 워치리스트·모의투자 시가평가)가 가격이 있다고 전제하므로 NaN을 흘려보내면 안 된다.
    merged = merged[merged["Close"].notna()]
    if min_marcap:
        merged = merged[merged["Marcap"] >= min_marcap]
    if min_volume:
        merged = merged[merged["Volume"] >= min_volume]

    result = merged[_RESULT_COLS].reset_index(drop=True)

    if use_cache:
        result.to_parquet(path)
    return result


def top_movers(
    df: pd.DataFrame,
    by: str = "DailyChangeRatio",
    n: int = 30,
    ascending: bool = False,
) -> pd.DataFrame:
    """등락률 기준 상위 n개. ascending=True면 하락률 상위(급락주)."""
    return df.dropna(subset=[by]).sort_values(by, ascending=ascending).head(n).reset_index(drop=True)
