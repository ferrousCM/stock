"""trading_agent_v2.py 테스트 — v2 매매 규칙 엔진(기본형/공격적/모멘텀).

v1(test_trading_agent.py)과 동일한 원칙: 외부 API가 없는 순수 함수라 모킹이 필요 없다.
정상 케이스보다 각 조건의 경계값(임계값 바로 위/아래)을 더 촘촘히 본다. 전부 오프라인.
"""

import pandas as pd

from src.trading_agent_v2 import (
    BOT_STRATEGIES_V2,
    TRADING_RULES_V2_AGGRESSIVE,
    TRADING_RULES_V2_DEFAULT,
    TRADING_RULES_V2_MOMENTUM,
    _calc_quantity_v2,
    decide_trades_v2_aggressive,
    decide_trades_v2_default,
    decide_trades_v2_momentum,
)


def _holdings(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["code", "name", "quantity", "avg_price"])


def _signal_v2(
    code: str,
    name: str = "테스트종목",
    predicted_return_5d: float = 0.05,
    rsi14: float = 50,
    news_sentiment: float = 0.0,
    directional_accuracy: float = 0.6,
    atr14: float | None = 1000.0,
    sma20: float | None = None,
    sma60: float | None = None,
    atr_contraction_ratio: float | None = None,
    dist_from_60d_high: float | None = None,
    day_change: float | None = None,
    volume_ratio_1d: float | None = None,
    day_range: float | None = None,
    prev_range_max3: float | None = None,
    close_location: float | None = None,
    low5: float | None = None,
) -> dict:
    return {
        "code": code,
        "name": name,
        "predicted_return_5d": predicted_return_5d,
        "rsi14": rsi14,
        "news_sentiment": news_sentiment,
        "directional_accuracy": directional_accuracy,
        "atr14": atr14,
        "sma20": sma20,
        "sma60": sma60,
        "atr_contraction_ratio": atr_contraction_ratio,
        "dist_from_60d_high": dist_from_60d_high,
        "day_change": day_change,
        "volume_ratio_1d": volume_ratio_1d,
        "day_range": day_range,
        "prev_range_max3": prev_range_max3,
        "close_location": close_location,
        "low5": low5,
    }


EMPTY_HOLDINGS = _holdings([])

# ==================================================================== _calc_quantity_v2


def test_calc_quantity_v2_kr_floors_to_int():
    qty = _calc_quantity_v2(
        "005930", price=70_000, atr14=1000.0, total_equity=100_000_000, rules=TRADING_RULES_V2_DEFAULT
    )
    # risk_dollars = 100_000_000*0.01 = 1_000_000, stop_distance = 1000*2.0 = 2000 -> 500주
    assert qty == 500
    assert isinstance(qty, int)


def test_calc_quantity_v2_coin_rounds_to_8_decimals():
    qty = _calc_quantity_v2(
        "KRW-BTC",
        price=150_000_000,
        atr14=3_000_000.0,
        total_equity=100_000_000,
        rules=TRADING_RULES_V2_DEFAULT,
    )
    # risk_dollars = 1_000_000, stop_distance = 3_000_000*2.0 = 6_000_000 -> 0.16666667
    assert isinstance(qty, float)
    assert round(qty, 8) == qty


def test_calc_quantity_v2_zero_atr_returns_zero():
    assert (
        _calc_quantity_v2(
            "005930", price=70_000, atr14=0.0, total_equity=100_000_000, rules=TRADING_RULES_V2_DEFAULT
        )
        == 0
    )


# ==================================================================== 기본형(v2) — 매수


def test_default_v2_buy_when_ma_aligned():
    signals = [_signal_v2("005930", sma20=65_000, sma60=60_000)]
    actions = decide_trades_v2_default(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert len(actions) == 1
    assert actions[0].action == "buy"
    assert actions[0].quantity > 0


def test_default_v2_no_buy_when_price_below_sma20():
    # 종가(70,000) <= SMA20(70,000) -> 정배열 아님(엄격 부등호)
    signals = [_signal_v2("005930", sma20=70_000, sma60=60_000)]
    actions = decide_trades_v2_default(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert actions == []


def test_default_v2_no_buy_when_sma20_below_sma60():
    # 종가>SMA20이지만 SMA20<=SMA60 -> 정배열 아님
    signals = [_signal_v2("005930", sma20=65_000, sma60=66_000)]
    actions = decide_trades_v2_default(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert actions == []


def test_default_v2_no_buy_when_ma_fields_missing():
    signals = [_signal_v2("005930", sma20=None, sma60=None)]
    actions = decide_trades_v2_default(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert actions == []


def test_default_v2_rejected_directional_accuracy_below_threshold():
    just_below = TRADING_RULES_V2_DEFAULT["min_directional_accuracy"] - 0.001
    signals = [_signal_v2("005930", sma20=65_000, sma60=60_000, directional_accuracy=just_below)]
    actions = decide_trades_v2_default(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert actions == []


def test_default_v2_accepted_directional_accuracy_at_exact_threshold():
    at_threshold = TRADING_RULES_V2_DEFAULT["min_directional_accuracy"]
    signals = [_signal_v2("005930", sma20=65_000, sma60=60_000, directional_accuracy=at_threshold)]
    actions = decide_trades_v2_default(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert len(actions) == 1


def test_default_v2_rejected_rsi_above_max_entry():
    just_above = TRADING_RULES_V2_DEFAULT["max_rsi_entry"] + 1
    signals = [_signal_v2("005930", sma20=65_000, sma60=60_000, rsi14=just_above)]
    actions = decide_trades_v2_default(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert actions == []


def test_default_v2_rejected_news_sentiment_below_threshold():
    just_below = TRADING_RULES_V2_DEFAULT["min_news_sentiment"] - 0.001
    signals = [_signal_v2("005930", sma20=65_000, sma60=60_000, news_sentiment=just_below)]
    actions = decide_trades_v2_default(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert actions == []


def test_default_v2_held_code_excluded_from_buy_candidates():
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 60_000}])
    signals = [_signal_v2("005930", sma20=65_000, sma60=60_000, predicted_return_5d=0.0, rsi14=50)]
    actions = decide_trades_v2_default(signals, holdings, {"005930": 70_000}, cash=100_000_000)
    assert all(a.action != "buy" for a in actions)


# ==================================================================== 기본형(v2) — 매도(ATR 기반)


def test_default_v2_sell_at_atr_stop_price():
    rules = TRADING_RULES_V2_DEFAULT
    avg_price = 70_000
    atr14 = 1000.0
    stop_price = avg_price - atr14 * rules["atr_multiplier"]  # 68,000
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": avg_price}])
    signals = [_signal_v2("005930", atr14=atr14, predicted_return_5d=0.05, rsi14=50)]
    actions = decide_trades_v2_default(signals, holdings, {"005930": stop_price}, cash=10_000_000)
    assert len(actions) == 1
    assert actions[0].action == "sell"
    assert "손절" in actions[0].reason


def test_default_v2_no_sell_just_above_atr_stop_price():
    rules = TRADING_RULES_V2_DEFAULT
    avg_price = 70_000
    atr14 = 1000.0
    stop_price = avg_price - atr14 * rules["atr_multiplier"]
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": avg_price}])
    signals = [_signal_v2("005930", atr14=atr14, predicted_return_5d=0.05, rsi14=50)]
    actions = decide_trades_v2_default(signals, holdings, {"005930": stop_price + 1}, cash=10_000_000)
    assert actions == []


def test_default_v2_sell_at_atr_target_price():
    rules = TRADING_RULES_V2_DEFAULT
    avg_price = 70_000
    atr14 = 1000.0
    target_price = avg_price + atr14 * rules["take_profit_atr_multiplier"]  # 73,000
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": avg_price}])
    signals = [_signal_v2("005930", atr14=atr14, predicted_return_5d=0.05, rsi14=50)]
    actions = decide_trades_v2_default(signals, holdings, {"005930": target_price}, cash=10_000_000)
    assert len(actions) == 1
    assert "익절" in actions[0].reason


def test_default_v2_sell_on_negative_signal():
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 70_000}])
    signals = [_signal_v2("005930", atr14=1000.0, predicted_return_5d=-0.01, rsi14=50)]
    actions = decide_trades_v2_default(signals, holdings, {"005930": 70_500}, cash=10_000_000)
    assert len(actions) == 1
    assert "신호 소멸" in actions[0].reason


def test_default_v2_sell_on_rsi_overbought_exit():
    rules = TRADING_RULES_V2_DEFAULT
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 70_000}])
    signals = [_signal_v2("005930", atr14=1000.0, predicted_return_5d=0.05, rsi14=rules["max_rsi_exit"])]
    actions = decide_trades_v2_default(signals, holdings, {"005930": 70_500}, cash=10_000_000)
    assert len(actions) == 1
    assert "과매수" in actions[0].reason


def test_default_v2_holding_without_signal_is_skipped():
    """ATR을 모르면(신호 생성 실패) 이번 실행에서는 이 보유 종목을 판단하지 않는다."""
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 70_000}])
    actions = decide_trades_v2_default([], holdings, {"005930": 1}, cash=10_000_000)  # 폭락해도
    assert actions == []


# ==================================================================== 공격적(v2) — 변동성 수축 + 피라미딩


def test_aggressive_v2_buy_when_contracted_and_near_high():
    rules = TRADING_RULES_V2_AGGRESSIVE
    signals = [
        _signal_v2(
            "005930",
            atr_contraction_ratio=rules["contraction_ratio_max"] - 0.01,
            dist_from_60d_high=-0.01,
            rsi14=60,
        )
    ]
    actions = decide_trades_v2_aggressive(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert len(actions) == 1
    assert actions[0].action == "buy"


def test_aggressive_v2_rejected_contraction_above_max():
    rules = TRADING_RULES_V2_AGGRESSIVE
    just_above = rules["contraction_ratio_max"] + 0.001
    signals = [_signal_v2("005930", atr_contraction_ratio=just_above, dist_from_60d_high=-0.01)]
    actions = decide_trades_v2_aggressive(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert actions == []


def test_aggressive_v2_accepted_contraction_at_exact_max():
    rules = TRADING_RULES_V2_AGGRESSIVE
    signals = [
        _signal_v2("005930", atr_contraction_ratio=rules["contraction_ratio_max"], dist_from_60d_high=-0.01)
    ]
    actions = decide_trades_v2_aggressive(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert len(actions) == 1


def test_aggressive_v2_rejected_too_far_from_pivot():
    rules = TRADING_RULES_V2_AGGRESSIVE
    just_below = -rules["pivot_proximity_pct"] - 0.001
    signals = [_signal_v2("005930", atr_contraction_ratio=0.5, dist_from_60d_high=just_below)]
    actions = decide_trades_v2_aggressive(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert actions == []


def test_aggressive_v2_pyramiding_allows_topup_on_held_code():
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 65_000}])
    signals = [_signal_v2("005930", atr_contraction_ratio=0.5, dist_from_60d_high=-0.01, rsi14=60)]
    actions = decide_trades_v2_aggressive(signals, holdings, {"005930": 70_000}, cash=100_000_000)
    buys = [a for a in actions if a.action == "buy"]
    assert len(buys) == 1
    assert "피라미딩" in buys[0].reason


def test_aggressive_v2_no_take_profit_target_only_stop():
    """공격적 봇은 take_profit_atr_multiplier=None — 가격이 아무리 올라도 익절로 안 판다."""
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 70_000}])
    signals = [_signal_v2("005930", atr14=1000.0, predicted_return_5d=0.05, rsi14=50)]
    actions = decide_trades_v2_aggressive(signals, holdings, {"005930": 1_000_000}, cash=10_000_000)
    sells = [a for a in actions if a.action == "sell"]
    assert sells == []  # 신호소멸도 아니고 RSI과매수도 아니므로 매도 없음


# ==================================================================== 모멘텀(v2) — 4% 버스트


def _burst_signal(code: str = "005930", **overrides) -> dict:
    rules = TRADING_RULES_V2_MOMENTUM
    base = dict(
        day_change=rules["burst_threshold"],
        volume_ratio_1d=rules["volume_ratio_min"],
        day_range=110.0,
        prev_range_max3=100.0,
        close_location=rules["close_location_min"],
        rsi14=60,
    )
    base.update(overrides)
    return _signal_v2(code, **base)


def test_momentum_v2_buy_when_burst_conditions_met():
    actions = decide_trades_v2_momentum(
        [_burst_signal()], EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000
    )
    assert len(actions) == 1
    assert actions[0].action == "buy"
    assert "브레이크아웃" in actions[0].reason


def test_momentum_v2_rejected_day_change_below_threshold():
    just_below = TRADING_RULES_V2_MOMENTUM["burst_threshold"] - 0.001
    actions = decide_trades_v2_momentum(
        [_burst_signal(day_change=just_below)], EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000
    )
    assert actions == []


def test_momentum_v2_rejected_range_not_expanded():
    # day_range == prev_range_max3 -> 확장 아님(엄격 부등호)
    actions = decide_trades_v2_momentum(
        [_burst_signal(day_range=100.0, prev_range_max3=100.0)],
        EMPTY_HOLDINGS,
        {"005930": 70_000},
        cash=100_000_000,
    )
    assert actions == []


def test_momentum_v2_rejected_close_location_below_min():
    just_below = TRADING_RULES_V2_MOMENTUM["close_location_min"] - 0.001
    actions = decide_trades_v2_momentum(
        [_burst_signal(close_location=just_below)], EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000
    )
    assert actions == []


def test_momentum_v2_rejected_volume_not_expanded():
    just_below = TRADING_RULES_V2_MOMENTUM["volume_ratio_min"] - 0.001
    actions = decide_trades_v2_momentum(
        [_burst_signal(volume_ratio_1d=just_below)], EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000
    )
    assert actions == []


def test_momentum_v2_sell_on_trend_break_below_5day_low():
    # avg_price 70,000·atr14 1,000·모멘텀 atr_multiplier 1.5 -> ATR 손절가는 68,500원.
    # low5 조건만 단독으로 확인하려면 가격이 그 위(69,000)이면서 low5(69,000) 이하여야 한다.
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 70_000}])
    signals = [_signal_v2("005930", atr14=1000.0, predicted_return_5d=0.05, rsi14=50, low5=69_000)]
    actions = decide_trades_v2_momentum(signals, holdings, {"005930": 69_000}, cash=10_000_000)
    assert len(actions) == 1
    assert "추세 이탈" in actions[0].reason


def test_momentum_v2_no_sell_just_above_5day_low():
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 70_000}])
    signals = [_signal_v2("005930", atr14=1000.0, predicted_return_5d=0.05, rsi14=50, low5=69_000)]
    actions = decide_trades_v2_momentum(signals, holdings, {"005930": 69_001}, cash=10_000_000)
    assert actions == []


# ==================================================================== BOT_STRATEGIES_V2 레지스트리


def test_bot_strategies_v2_registry_shape():
    assert set(BOT_STRATEGIES_V2.keys()) == {"default", "aggressive", "momentum"}
    for meta in BOT_STRATEGIES_V2.values():
        assert "label" in meta and "decide_trades" in meta and "rules" in meta
        assert callable(meta["decide_trades"])

    assert BOT_STRATEGIES_V2["default"]["decide_trades"] is decide_trades_v2_default
    assert BOT_STRATEGIES_V2["aggressive"]["decide_trades"] is decide_trades_v2_aggressive
    assert BOT_STRATEGIES_V2["momentum"]["decide_trades"] is decide_trades_v2_momentum
    assert BOT_STRATEGIES_V2["default"]["rules"] is TRADING_RULES_V2_DEFAULT
    assert BOT_STRATEGIES_V2["aggressive"]["rules"] is TRADING_RULES_V2_AGGRESSIVE
    assert BOT_STRATEGIES_V2["momentum"]["rules"] is TRADING_RULES_V2_MOMENTUM
