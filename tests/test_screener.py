"""KRX 전종목 스냅샷 스크리닝(src/screener.py) 오프라인 테스트.

네트워크 경계가 `_fetch_csv()` 하나이므로 그것만 합성 스냅샷으로 대체해 검증한다
(실제 응답 형태를 가정하기보다 파싱·계산 로직 자체를 검증 — tests 규칙 참고).

가장 중요한 건 **더미 스냅샷 회귀 테스트**다: 미러가 장 마감 전/휴장일에 미리 만들어
두는 "행 수는 정상인데 가격이 전부 `-`"인 파일을 최신 스냅샷으로 집으면 안 된다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import screener

_TODAY = pd.Timestamp.today().normalize()


def _snapshot(closes: dict[str, object]) -> pd.DataFrame:
    """{종목코드: 종가} → 미러 CSV를 읽은 직후 모양의 DataFrame.

    종가가 문자열이면 미러가 그렇게 주는 경우(거래정지 "-", 더미 스냅샷)를 재현하기
    위해 컬럼 전체가 문자열 dtype이 된다 — 실제 `pd.read_csv` 동작과 같다.
    """
    names = {"005930": "삼성전자", "247540": "에코프로비엠", "900100": "뉴프라이드"}
    markets = {"005930": "KOSPI", "247540": "KOSDAQ", "900100": "KOSPI"}
    rows = []
    for code, close in closes.items():
        rows.append(
            {
                "Code": code,
                "Name": names[code],
                "Market": markets[code],
                "Close": close,
                "ChagesRatio": 1.5,
                "Volume": 1000,
                "Amount": 2000,
                "Marcap": 3000,
            }
        )
    return pd.DataFrame(rows)


def _patch_fetch(monkeypatch, snapshots: dict[str, pd.DataFrame]) -> None:
    """날짜별 스냅샷 매핑으로 `_fetch_csv`를 대체한다. 없는 날짜는 404처럼 예외."""

    def _fetch(url: str) -> pd.DataFrame:
        for date_str, df in snapshots.items():
            if date_str in url:
                return df.copy()
        raise FileNotFoundError(url)  # 휴장일 등으로 파일이 없는 경우

    monkeypatch.setattr(screener, "_fetch_csv", _fetch)


def _d(days_ago: int) -> str:
    return (_TODAY - pd.Timedelta(days=days_ago)).strftime("%Y-%m-%d")


# --- 더미 스냅샷(가격 전부 "-") 처리 -------------------------------------------------


def test_snapshot_on_returns_none_for_placeholder(monkeypatch):
    """행 수는 정상이지만 종가가 전부 "-"인 더미 스냅샷은 None이어야 한다."""
    _patch_fetch(monkeypatch, {_d(0): _snapshot({"005930": "-", "247540": "-"})})

    assert screener._snapshot_on(_TODAY) is None


def test_snapshot_on_coerces_numeric_strings(monkeypatch):
    """일부 종목만 "-"인 정상 스냅샷은 유효하며, 숫자 컬럼이 숫자 dtype이 된다."""
    _patch_fetch(monkeypatch, {_d(0): _snapshot({"005930": "70000", "900100": "-"})})

    snap = screener._snapshot_on(_TODAY)

    assert snap is not None
    assert snap.loc[snap["Code"] == "005930", "Close"].iloc[0] == pytest.approx(70000.0)
    assert pd.isna(snap.loc[snap["Code"] == "900100", "Close"].iloc[0])


def test_nearest_snapshot_falls_back_past_placeholder(monkeypatch):
    """오늘 파일이 더미면 하루 더 과거로 내려가 직전 거래일을 써야 한다."""
    _patch_fetch(
        monkeypatch,
        {
            _d(0): _snapshot({"005930": "-"}),  # 더미
            _d(1): _snapshot({"005930": 70000}),  # 직전 거래일(정상)
        },
    )

    date, snap = screener._nearest_snapshot(_TODAY)

    assert date.strftime("%Y-%m-%d") == _d(1)
    assert snap["Close"].iloc[0] == pytest.approx(70000.0)


def test_screen_survives_placeholder_latest_snapshot(monkeypatch):
    """회귀 테스트 — 더미 스냅샷을 집어 문자열끼리 빼다가 TypeError로 죽던 버그.

    자동매매 워크플로(.github/workflows/daily_trading.yml)가 2026-09-24·09-25·10-01·
    10-02·10-05·10-09에 이 때문에 실패했다.
    """
    _patch_fetch(
        monkeypatch,
        {
            _d(0): _snapshot({"005930": "-", "247540": "-"}),  # 오늘자 더미
            _d(1): _snapshot({"005930": 70000, "247540": 110000}),
            _d(8): _snapshot({"005930": 50000, "247540": 100000}),
        },
    )

    df = screen_offline()

    assert list(df.columns) == screener._RESULT_COLS
    assert set(df["Code"]) == {"005930", "247540"}
    samsung = df.loc[df["Code"] == "005930"].iloc[0]
    assert samsung["Close"] == pytest.approx(70000.0)  # 더미가 아니라 직전 거래일 종가
    assert samsung["WeeklyChangeRatio"] == pytest.approx(40.0)  # 50000 -> 70000


# --- 기본 계산 ----------------------------------------------------------------------


def screen_offline(**kwargs):
    """캐시를 거치지 않는 screen() 호출 — 테스트가 실제 캐시 파일을 건드리지 않게."""
    return screener.screen(use_cache=False, **kwargs)


def test_screen_computes_weekly_and_monthly_ratios(monkeypatch):
    _patch_fetch(
        monkeypatch,
        {
            _d(0): _snapshot({"005930": 70000}),
            _d(7): _snapshot({"005930": 35000}),
            _d(30): _snapshot({"005930": 70000}),
        },
    )

    row = screen_offline().iloc[0]

    assert row["WeeklyChangeRatio"] == pytest.approx(100.0)  # 35000 -> 70000
    assert row["MonthlyChangeRatio"] == pytest.approx(0.0)  # 70000 -> 70000
    assert row["DailyChangeRatio"] == pytest.approx(1.5)  # ChagesRatio 그대로 리네임


def test_screen_monthly_nan_when_old_snapshot_missing(monkeypatch):
    """30일 전 스냅샷이 없어도 전체 조회가 실패하지 않고 월간만 NaN이어야 한다."""
    _patch_fetch(
        monkeypatch,
        {_d(0): _snapshot({"005930": 70000}), _d(7): _snapshot({"005930": 35000})},
    )

    df = screen_offline()

    assert df["MonthlyChangeRatio"].isna().all()
    assert df["WeeklyChangeRatio"].notna().all()


def test_screen_drops_rows_without_close(monkeypatch):
    """거래정지 등으로 종가가 없는 행은 결과에서 제외한다(호출부가 가격을 전제)."""
    _patch_fetch(
        monkeypatch,
        {
            _d(0): _snapshot({"005930": "70000", "900100": "-"}),
            _d(7): _snapshot({"005930": "35000", "900100": "-"}),
        },
    )

    df = screen_offline()

    assert set(df["Code"]) == {"005930"}


def test_screen_market_filter(monkeypatch):
    _patch_fetch(
        monkeypatch,
        {
            _d(0): _snapshot({"005930": 70000, "247540": 110000}),
            _d(7): _snapshot({"005930": 35000, "247540": 100000}),
        },
    )

    assert set(screen_offline(market="KOSDAQ")["Code"]) == {"247540"}
    assert set(screen_offline(market="KOSPI")["Code"]) == {"005930"}
    assert set(screen_offline(market="ALL")["Code"]) == {"005930", "247540"}


def test_screen_raises_when_no_snapshot_at_all(monkeypatch):
    _patch_fetch(monkeypatch, {})

    with pytest.raises(RuntimeError):
        screen_offline()


def test_screen_raises_when_only_placeholders(monkeypatch):
    """최근 10일이 전부 더미면 '스냅샷 없음'과 같이 RuntimeError여야 한다 —
    run_daily_trading.py가 이 예외만 잡아 매매 없이 조용히 종료한다."""
    _patch_fetch(monkeypatch, {_d(i): _snapshot({"005930": "-"}) for i in range(12)})

    with pytest.raises(RuntimeError):
        screen_offline()


# --- top_movers ---------------------------------------------------------------------


def test_top_movers_sorts_and_drops_nan():
    df = pd.DataFrame(
        {
            "Code": ["A", "B", "C"],
            "DailyChangeRatio": [1.0, float("nan"), 5.0],
        }
    )

    top = screener.top_movers(df, n=2)

    assert list(top["Code"]) == ["C", "A"]  # NaN 행은 제외
    assert list(screener.top_movers(df, n=1, ascending=True)["Code"]) == ["A"]


@pytest.mark.network
def test_screen_live():
    """실제 미러 조회 — 더미 스냅샷이 올라와 있는 날에도 숫자 종가가 나와야 한다."""
    df = screener.screen(use_cache=False)

    assert len(df) > 1000
    assert list(df.columns) == screener._RESULT_COLS
    assert pd.api.types.is_numeric_dtype(df["Close"])
    assert df["Close"].notna().all()
