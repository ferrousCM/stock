"""src/indicators.py 단위테스트. 순수 함수라 오프라인에서 바로 검증 가능."""

from __future__ import annotations

import pandas as pd
import pytest

from src import indicators as ind


def test_atr_matches_hand_calculated_values():
    """직접 계산한 True Range·Wilder 평활 값과 대조 — PRD.md 11장(v2)의 손절폭·포지션
    사이징 기준이 되므로 공식이 정확한지 회귀로 고정해둔다."""
    df = pd.DataFrame(
        {
            "Open": [100.0, 105.0, 112.0],
            "High": [110.0, 115.0, 108.0],
            "Low": [100.0, 104.0, 101.0],
            "Close": [105.0, 112.0, 103.0],
        }
    )

    result = ind.atr(df, window=3)

    # TR1 = High-Low = 10 (전일 종가 없음)
    # TR2 = max(115-104, |115-105|, |104-105|) = max(11, 10, 1) = 11
    # TR3 = max(108-101, |108-112|, |101-112|) = max(7, 4, 11) = 11
    # ATR(alpha=1/3, adjust=False): ATR1=10, ATR2=(1/3)*11+(2/3)*10, ATR3=(1/3)*11+(2/3)*ATR2
    assert result.iloc[0] == pytest.approx(10.0)
    assert result.iloc[1] == pytest.approx(10.333333, abs=1e-5)
    assert result.iloc[2] == pytest.approx(10.555556, abs=1e-5)


def test_atr_is_never_negative():
    rng_df = pd.DataFrame(
        {
            "Open": [10, 12, 9, 15, 11],
            "High": [11, 13, 10, 16, 12],
            "Low": [9, 11, 8, 14, 10],
            "Close": [10, 12, 9, 15, 11],
        },
        dtype=float,
    )
    result = ind.atr(rng_df, window=3)
    assert (result.dropna() >= 0).all()


def test_atr_flat_price_is_zero():
    """고가=저가=종가로 변동이 전혀 없으면 True Range도 0, ATR도 0이어야 한다."""
    flat = pd.DataFrame({"Open": [100.0] * 5, "High": [100.0] * 5, "Low": [100.0] * 5, "Close": [100.0] * 5})
    result = ind.atr(flat, window=3)
    assert (result == 0).all()
