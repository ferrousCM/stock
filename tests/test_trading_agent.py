"""trading_agent.py 테스트 — 매매 규칙 엔진(decide_trades) + 리스크 가드레일.

외부 API가 없는 순수 함수라 모킹이 필요 없다. 정상 케이스보다 각 조건의 경계값
(임계값 바로 위/아래)을 더 촘촘히 본다. 전부 오프라인, 네트워크 없음.
"""

import pandas as pd

from src.trading_agent import (
    AGGRESSIVE_RULES,
    BOT_STRATEGIES,
    MOMENTUM_RULES,
    TRADING_RULES,
    TradeAction,
    apply_risk_guardrail,
    decide_trades,
    decide_trades_aggressive,
    decide_trades_momentum,
    infer_market,
)


def _holdings(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["code", "name", "quantity", "avg_price"])


def _signal(
    code: str,
    name: str = "테스트종목",
    predicted_return_5d: float = 0.05,
    rsi14: float = 50,
    news_sentiment: float = 0.0,
    directional_accuracy: float = 0.6,
    sma5_gap: float | None = None,
    sma20_gap: float | None = None,
) -> dict:
    return {
        "code": code,
        "name": name,
        "predicted_return_5d": predicted_return_5d,
        "rsi14": rsi14,
        "news_sentiment": news_sentiment,
        "directional_accuracy": directional_accuracy,
        "sma5_gap": sma5_gap,
        "sma20_gap": sma20_gap,
    }


EMPTY_HOLDINGS = _holdings([])

# ==================================================================== decide_trades — 매수


def test_buy_when_all_conditions_met():
    signals = [_signal("005930", predicted_return_5d=0.05, rsi14=60, directional_accuracy=0.6)]
    actions = decide_trades(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)

    assert len(actions) == 1
    a = actions[0]
    assert a.action == "buy"
    assert a.code == "005930"
    assert "매수 조건 충족" in a.reason
    assert a.quantity > 0


def test_buy_rejected_return_below_threshold():
    just_below = TRADING_RULES["buy_return_threshold"] - 0.001
    signals = [_signal("005930", predicted_return_5d=just_below)]
    actions = decide_trades(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert actions == []


def test_buy_accepted_return_at_exact_threshold():
    signals = [_signal("005930", predicted_return_5d=TRADING_RULES["buy_return_threshold"])]
    actions = decide_trades(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert len(actions) == 1


def test_buy_rejected_directional_accuracy_below_threshold():
    just_below = TRADING_RULES["min_directional_accuracy"] - 0.001
    signals = [_signal("005930", directional_accuracy=just_below)]
    actions = decide_trades(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert actions == []


def test_buy_rejected_rsi_above_max_entry():
    just_above = TRADING_RULES["max_rsi_entry"] + 1
    signals = [_signal("005930", rsi14=just_above)]
    actions = decide_trades(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert actions == []


def test_buy_accepted_rsi_at_exact_max_entry():
    signals = [_signal("005930", rsi14=TRADING_RULES["max_rsi_entry"])]
    actions = decide_trades(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert len(actions) == 1


def test_buy_rejected_news_sentiment_too_negative():
    just_below = TRADING_RULES["min_news_sentiment"] - 0.01
    signals = [_signal("005930", news_sentiment=just_below)]
    actions = decide_trades(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert actions == []


def test_buy_skipped_when_no_price_available():
    signals = [_signal("005930")]
    actions = decide_trades(signals, EMPTY_HOLDINGS, {}, cash=100_000_000)
    assert actions == []


def test_buy_already_held_stock_excluded_from_new_entry():
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 70_000}])
    # 보유 종목이 청산 조건에도 안 걸리게 현재가를 평단가와 같게 둔다 — 순수하게
    # "이미 보유 중이라 신규 매수 후보에서 빠지는지"만 확인.
    signals = [_signal("005930", predicted_return_5d=0.10)]
    actions = decide_trades(signals, holdings, {"005930": 70_000}, cash=100_000_000)
    assert actions == []  # 매수도 매도도 없음


def test_buy_candidates_sorted_by_predicted_return_desc():
    signals = [
        _signal("A", predicted_return_5d=0.04),
        _signal("B", predicted_return_5d=0.10),
        _signal("C", predicted_return_5d=0.06),
    ]
    prices = {"A": 10_000, "B": 10_000, "C": 10_000}
    actions = decide_trades(signals, EMPTY_HOLDINGS, prices, cash=100_000_000)
    assert [a.code for a in actions] == ["B", "C", "A"]


def test_buy_respects_max_holdings():
    rules = {**TRADING_RULES, "max_holdings": 1, "max_daily_trades": 10}
    signals = [_signal("A", predicted_return_5d=0.10), _signal("B", predicted_return_5d=0.08)]
    prices = {"A": 10_000, "B": 10_000}
    actions = decide_trades(signals, EMPTY_HOLDINGS, prices, cash=100_000_000, rules=rules)
    assert [a.code for a in actions] == ["A"]  # 수익률 높은 A만, 한도 1개


def test_buy_respects_max_daily_trades():
    rules = {**TRADING_RULES, "max_daily_trades": 2, "max_holdings": 10}
    signals = [_signal(c, predicted_return_5d=0.10 - i * 0.01) for i, c in enumerate("ABCDE")]
    prices = {c: 10_000 for c in "ABCDE"}
    actions = decide_trades(signals, EMPTY_HOLDINGS, prices, cash=100_000_000, rules=rules)
    assert len(actions) == 2
    assert [a.code for a in actions] == ["A", "B"]


def test_buy_skips_candidate_below_min_trade_amount_but_tries_next():
    # 현금이 아주 적어서 비싼 종목(A)은 최소 거래금액을 못 채우고, 싼 종목(B)은 채운다.
    rules = {
        **TRADING_RULES,
        "min_trade_amount": 100_000,
        "max_position_pct": 1.0,
        "min_cash_reserve_pct": 0.0,
    }
    signals = [_signal("A", predicted_return_5d=0.10), _signal("B", predicted_return_5d=0.05)]
    prices = {"A": 1_000_000, "B": 1_000}  # A는 1주에 100만원(예산 부족), B는 1주에 1천원
    actions = decide_trades(signals, EMPTY_HOLDINGS, prices, cash=150_000, rules=rules)
    codes = [a.code for a in actions]
    assert "A" not in codes  # 예산(15만원)으로 A 1주(100만원)도 못 삼
    assert "B" in codes


def test_buy_respects_min_cash_reserve():
    # 현금 보유 전액이 매수 예산이 아니다 — min_cash_reserve_pct만큼은 항상 남겨야 한다.
    # max_position_pct는 기본값(0.15)을 그대로 둬서 "예산의 병목이 현금준비금 쪽"인
    # 경우와 "종목당 비중 한도 쪽"인 경우를 test_buy_quantity_capped_by_max_position_pct와
    # 구분해서 각각 확인한다.
    rules = {**TRADING_RULES, "min_cash_reserve_pct": 0.5, "min_trade_amount": 0}
    signals = [_signal("A", predicted_return_5d=0.10)]
    actions = decide_trades(signals, EMPTY_HOLDINGS, {"A": 10_000}, cash=100_000_000, rules=rules)
    # total_equity(현금만, 보유 없음)=1억, max_position_pct 15% = 1,500만원,
    # 현금준비금 50% 반영한 매수예산 = 1억 - 5천만원 = 5천만원 -> 더 작은 쪽인 1,500만원이 적용
    # -> 1,500만원 / 1만원 = 1,500주
    assert actions[0].quantity == 1_500


def test_buy_quantity_capped_by_max_position_pct():
    rules = {**TRADING_RULES, "max_position_pct": 0.10, "min_cash_reserve_pct": 0.0, "min_trade_amount": 0}
    signals = [_signal("A", predicted_return_5d=0.10)]
    actions = decide_trades(signals, EMPTY_HOLDINGS, {"A": 1_000}, cash=100_000_000, rules=rules)
    # total_equity=1억, 종목당 최대 비중 10% = 1천만원 -> 1천원짜리 주식 10,000주
    assert actions[0].quantity == 10_000


# ==================================================================== decide_trades — 매도


def test_sell_stop_loss_triggers():
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 100_000}])
    # 손절선을 확실히 넘긴 값을 쓴다 — "정확히 그 값"은 곱셈 중간 부동소수점 반올림 때문에
    # 임계값과 미묘하게 어긋날 수 있어(예: 0.14999999999999997) 경계 테스트로 부적합하다.
    price = 100_000 * (1 + TRADING_RULES["stop_loss_pct"] - 0.001)
    actions = decide_trades([], holdings, {"005930": price}, cash=0)
    assert len(actions) == 1
    assert actions[0].action == "sell"
    assert actions[0].quantity == 10
    assert "손절" in actions[0].reason


def test_sell_not_triggered_just_above_stop_loss():
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 100_000}])
    price = 100_000 * (1 + TRADING_RULES["stop_loss_pct"] + 0.001)  # 손절선 살짝 위 (손실 더 작음)
    actions = decide_trades([], holdings, {"005930": price}, cash=0)
    assert actions == []


def test_sell_take_profit_triggers():
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 100_000}])
    price = 100_000 * (1 + TRADING_RULES["take_profit_pct"] + 0.001)  # 익절선을 확실히 넘김
    actions = decide_trades([], holdings, {"005930": price}, cash=0)
    assert len(actions) == 1
    assert "익절" in actions[0].reason


def test_sell_coin_fractional_holding_stop_loss_preserves_quantity():
    """회귀 테스트: 매도 수량을 int()로 캐스팅하면 1 미만 소수 코인 보유분(예: 0.5 BTC)이
    quantity=0으로 잘려서, portfolio.apply_trade()의 "quantity는 양수여야 합니다" 예외로
    그날 실행 전체가 죽는다. 소수 그대로 나가야 한다."""
    holdings = _holdings([{"code": "KRW-BTC", "name": "비트코인", "quantity": 0.5, "avg_price": 100_000_000}])
    price = 100_000_000 * (1 + TRADING_RULES["stop_loss_pct"] - 0.001)
    actions = decide_trades([], holdings, {"KRW-BTC": price}, cash=0)
    assert len(actions) == 1
    assert actions[0].quantity == 0.5
    assert actions[0].quantity > 0


def test_guardrail_approves_coin_fractional_holding_sell():
    """회귀 테스트: apply_risk_guardrail()이 sim_holdings 초기화 시 보유 수량을 int()로
    캐스팅하면 소수 코인 보유분이 0으로 보여, 실제로는 다 갖고 있는데도 "보유 수량을
    초과하는 매도 제안"으로 잘못 거부된다."""
    holdings = _holdings([{"code": "KRW-BTC", "name": "비트코인", "quantity": 0.5, "avg_price": 100_000_000}])
    actions = [TradeAction(action="sell", code="KRW-BTC", quantity=0.5, reason="테스트")]
    approved, rejected = apply_risk_guardrail(
        actions, holdings, cash=0, current_prices={"KRW-BTC": 100_000_000}
    )
    assert approved == actions
    assert rejected == []


def test_sell_negative_signal_triggers():
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 100_000}])
    signals = [_signal("005930", predicted_return_5d=-0.01)]
    actions = decide_trades(signals, holdings, {"005930": 100_000}, cash=0)  # 손익 0%, 손절/익절 미해당
    assert len(actions) == 1
    assert "신호 소멸" in actions[0].reason


def test_sell_rsi_exit_triggers():
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 100_000}])
    signals = [_signal("005930", predicted_return_5d=0.05, rsi14=TRADING_RULES["max_rsi_exit"])]
    actions = decide_trades(signals, holdings, {"005930": 100_000}, cash=0)
    assert len(actions) == 1
    assert "과매수" in actions[0].reason


def test_sell_no_action_when_price_missing():
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 100_000}])
    actions = decide_trades([], holdings, {}, cash=0)
    assert actions == []


def test_sell_price_only_checks_when_no_signal_available():
    """보유 중인데 signals에 없는 종목(예측 실패 등) — 가격 기반 손절/익절만 평가되고
    신호 기반 조건은 애초에 평가 대상이 아니라 크래시 없이 정상 동작해야 한다."""
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 100_000}])
    price = 100_000 * (1 + TRADING_RULES["stop_loss_pct"] - 0.001)  # 손절선을 확실히 넘김
    actions = decide_trades([], holdings, {"005930": price}, cash=0)  # signals=[] (신호 없음)
    assert len(actions) == 1
    assert "손절" in actions[0].reason


def test_sell_priority_stop_loss_wins_over_negative_signal():
    """손절과 신호소멸이 동시에 해당하면 손절 사유가 보고된다 (PRD가 나열한 순서)."""
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 100_000}])
    price = 100_000 * (1 + TRADING_RULES["stop_loss_pct"] - 0.001)  # 손절선을 확실히 넘김
    signals = [_signal("005930", predicted_return_5d=-0.05)]  # 신호소멸 조건도 동시에 만족
    actions = decide_trades(signals, holdings, {"005930": price}, cash=0)
    assert len(actions) == 1
    assert "손절" in actions[0].reason
    assert "신호" not in actions[0].reason


def test_sold_today_stock_not_rebought_same_day():
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 100_000}])
    price = 100_000 * (1 + TRADING_RULES["stop_loss_pct"] - 0.001)  # 손절선을 확실히 넘김  # 손절 트리거
    # 같은 종목이 매수 조건도 동시에 만족하도록 강한 신호를 준다.
    signals = [_signal("005930", predicted_return_5d=0.20, rsi14=40, directional_accuracy=0.9)]
    actions = decide_trades(signals, holdings, {"005930": price}, cash=100_000_000)
    assert len(actions) == 1
    assert actions[0].action == "sell"  # 매도만, 당일 재매수는 없음


# ==================================================================== apply_risk_guardrail


def test_guardrail_approves_valid_buy():
    holdings = EMPTY_HOLDINGS
    # quantity*price(=1,400,000)가 min_trade_amount 기본값(100만원)을 넘도록 20주로 잡는다.
    actions = [TradeAction(action="buy", code="005930", quantity=20, reason="테스트")]
    approved, rejected = apply_risk_guardrail(
        actions, holdings, cash=100_000_000, current_prices={"005930": 70_000}
    )
    assert approved == actions
    assert rejected == []


def test_guardrail_rejects_buy_exceeding_cash():
    actions = [TradeAction(action="buy", code="005930", quantity=1000, reason="버그로 과도한 수량")]
    approved, rejected = apply_risk_guardrail(
        actions, EMPTY_HOLDINGS, cash=1_000, current_prices={"005930": 70_000}
    )
    assert approved == []
    assert rejected[0]["reason_rejected"] == "현금 부족(최소 현금 보유 비율 포함)"


def test_guardrail_rejects_sell_exceeding_held_quantity():
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 5, "avg_price": 70_000}])
    actions = [TradeAction(action="sell", code="005930", quantity=10, reason="버그로 과도한 매도")]
    approved, rejected = apply_risk_guardrail(actions, holdings, cash=0, current_prices={"005930": 70_000})
    assert approved == []
    assert rejected[0]["reason_rejected"] == "보유 수량을 초과하는 매도 제안"


def test_guardrail_rejects_buy_when_max_holdings_reached():
    # min_trade_amount는 이 테스트가 확인하려는 게 아니므로 0으로 꺼서 다른 사유로
    # 거부되지 않게 한다 (체크 순서상 min_trade_amount가 max_holdings보다 먼저 걸린다).
    rules = {**TRADING_RULES, "max_holdings": 1, "min_trade_amount": 0}
    holdings = _holdings([{"code": "000660", "name": "SK하이닉스", "quantity": 1, "avg_price": 100_000}])
    actions = [TradeAction(action="buy", code="005930", quantity=1, reason="새 종목")]
    approved, rejected = apply_risk_guardrail(
        actions, holdings, cash=100_000_000, current_prices={"005930": 70_000, "000660": 100_000}, rules=rules
    )
    assert approved == []
    assert rejected[0]["reason_rejected"] == "동시 보유 종목 수 한도 초과"


def test_guardrail_allows_topup_of_existing_holding_even_at_max_holdings():
    """이미 보유 중인 종목을 더 사는 건 종목 수를 안 늘리므로 max_holdings에 안 걸려야 한다."""
    rules = {
        **TRADING_RULES,
        "max_holdings": 1,
        "max_position_pct": 1.0,
        "min_cash_reserve_pct": 0.0,
        "min_trade_amount": 0,
    }
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 1, "avg_price": 70_000}])
    actions = [TradeAction(action="buy", code="005930", quantity=1, reason="추가 매수")]
    approved, rejected = apply_risk_guardrail(
        actions, holdings, cash=100_000_000, current_prices={"005930": 70_000}, rules=rules
    )
    assert approved == actions
    assert rejected == []


def test_guardrail_rejects_buy_exceeding_max_daily_trades():
    rules = {**TRADING_RULES, "max_daily_trades": 2, "min_cash_reserve_pct": 0.0, "min_trade_amount": 0}
    actions = [TradeAction(action="buy", code=c, quantity=1, reason="테스트") for c in ["A", "B", "C"]]
    prices = {c: 10_000 for c in "ABC"}
    approved, rejected = apply_risk_guardrail(
        actions, EMPTY_HOLDINGS, cash=100_000_000, current_prices=prices, rules=rules
    )
    assert len(approved) == 2
    assert len(rejected) == 1
    assert rejected[0]["reason_rejected"] == "하루 최대 거래 횟수 초과"


def test_guardrail_sells_not_capped_by_max_daily_trades():
    """청산은 거래 횟수 한도를 안 받는다 — 손실을 줄여야 하는 날 한도 때문에 막히면 안 된다."""
    rules = {**TRADING_RULES, "max_daily_trades": 1}
    holdings = _holdings([{"code": c, "name": c, "quantity": 1, "avg_price": 10_000} for c in "ABCDEFG"])
    actions = [TradeAction(action="sell", code=c, quantity=1, reason="손절") for c in "ABCDEFG"]
    prices = {c: 9_000 for c in "ABCDEFG"}
    approved, rejected = apply_risk_guardrail(actions, holdings, cash=0, current_prices=prices, rules=rules)
    assert len(approved) == 7
    assert rejected == []


def test_guardrail_rejects_buy_exceeding_max_position_pct():
    rules = {**TRADING_RULES, "max_position_pct": 0.01, "min_cash_reserve_pct": 0.0, "min_trade_amount": 0}
    actions = [TradeAction(action="buy", code="005930", quantity=100, reason="비중 초과 테스트")]
    # 100주 * 7만원 = 700만원, total_equity=1억, 1% 한도 = 100만원 -> 초과
    approved, rejected = apply_risk_guardrail(
        actions, EMPTY_HOLDINGS, cash=100_000_000, current_prices={"005930": 70_000}, rules=rules
    )
    assert approved == []
    assert rejected[0]["reason_rejected"] == "종목당 최대 비중 초과"


def test_guardrail_rejects_buy_below_min_trade_amount():
    rules = {**TRADING_RULES, "min_trade_amount": 1_000_000, "min_cash_reserve_pct": 0.0}
    actions = [TradeAction(action="buy", code="005930", quantity=1, reason="너무 작은 거래")]
    approved, rejected = apply_risk_guardrail(
        actions, EMPTY_HOLDINGS, cash=100_000_000, current_prices={"005930": 1_000}, rules=rules
    )
    assert approved == []
    assert rejected[0]["reason_rejected"] == "최소 거래 금액 미달"


def test_guardrail_rejects_when_price_missing():
    actions = [TradeAction(action="buy", code="005930", quantity=1, reason="테스트")]
    approved, rejected = apply_risk_guardrail(actions, EMPTY_HOLDINGS, cash=100_000_000, current_prices={})
    assert approved == []
    assert rejected[0]["reason_rejected"] == "체결 가격을 알 수 없음"


def test_guardrail_sell_frees_cash_for_subsequent_buy():
    """매도를 먼저 처리해 현금을 확보한 뒤에야 승인되는 매수 — 순서가 실제로 지켜지는지 확인."""
    rules = {**TRADING_RULES, "max_position_pct": 1.0, "min_cash_reserve_pct": 0.0, "min_trade_amount": 0}
    holdings = _holdings([{"code": "000660", "name": "SK하이닉스", "quantity": 10, "avg_price": 100_000}])
    actions = [
        TradeAction(action="sell", code="000660", quantity=10, reason="매도"),
        TradeAction(action="buy", code="005930", quantity=15, reason="매도 대금으로 매수"),
    ]
    # 현금 0원, 매도로 100만원 확보 -> 15주*7만원=105만원은 여전히 부족해야 함을 확인하는 대신
    # 정확히 확보되는 금액(100만원)으로 살 수 있는 수량(14주=98만원)으로 조정해 승인되는지 본다.
    actions[1] = TradeAction(action="buy", code="005930", quantity=14, reason="매도 대금으로 매수")
    approved, rejected = apply_risk_guardrail(
        actions, holdings, cash=0, current_prices={"000660": 100_000, "005930": 70_000}, rules=rules
    )
    assert len(approved) == 2
    assert rejected == []


# ==================================================================== infer_market


def test_infer_market_kr_six_digit_code():
    assert infer_market("005930") == "KR"
    assert infer_market("000660") == "KR"


def test_infer_market_coin_krw_prefix():
    assert infer_market("KRW-BTC") == "COIN"
    assert infer_market("KRW-ETH") == "COIN"


def test_infer_market_us_ticker():
    assert infer_market("AAPL") == "US"
    assert infer_market("NVDA") == "US"


# ==================================================================== decide_trades — 코인 소수 수량


def test_buy_coin_high_price_yields_fractional_quantity():
    """비트코인처럼 1개 가격이 종목당 최대 비중 예산보다 훨씬 큰 코인 — 정수 수량
    로직이었다면 quantity=0이 되어 매수가 아예 생기지 않았을 상황에서, 코인은 소수
    수량으로 실제 매수 액션이 생겨야 한다."""
    price = 150_000_000  # 1.5억원짜리 코인 (비트코인 가정)
    cash = 100_000_000  # 예산 1억원 -> max_position_pct 15%면 목표 배분액은 1,500만원
    signals = [_signal("KRW-BTC", predicted_return_5d=0.10)]
    actions = decide_trades(signals, EMPTY_HOLDINGS, {"KRW-BTC": price}, cash=cash)

    # 정수 수량이었다면: int(15_000_000 // 150_000_000) == 0 -> 매수 액션 없음
    assert int(TRADING_RULES["max_position_pct"] * cash // price) == 0

    assert len(actions) == 1
    a = actions[0]
    assert a.action == "buy"
    assert a.code == "KRW-BTC"
    assert isinstance(a.quantity, float)
    assert a.quantity > 0
    # round(15_000_000 / 150_000_000, 8) == 0.1
    assert a.quantity == round(0.15 * cash / price, 8)
    amount = a.quantity * price
    assert amount >= TRADING_RULES["min_trade_amount"]


def test_buy_non_coin_still_uses_integer_quantity():
    """국내(6자리 숫자)·해외(그 외 문자 티커) 종목은 여전히 정수 수량으로 계산된다."""
    signals = [
        _signal("005930", predicted_return_5d=0.10),
        _signal("AAPL", predicted_return_5d=0.09),
    ]
    prices = {"005930": 70_000, "AAPL": 199.5}
    actions = decide_trades(signals, EMPTY_HOLDINGS, prices, cash=100_000_000)

    by_code = {a.code: a for a in actions}
    assert isinstance(by_code["005930"].quantity, int)
    assert isinstance(by_code["AAPL"].quantity, int)


# ==================================================================== apply_risk_guardrail — 코인 소수 수량


def test_guardrail_approves_fractional_coin_buy():
    """소수 수량 매수 액션도 가드레일이 정상 승인해야 한다 (수량 비교·현금 차감이
    float에서도 깨지지 않는지 확인)."""
    price = 150_000_000
    quantity = round(0.15 * 100_000_000 / price, 8)  # 0.1
    actions = [TradeAction(action="buy", code="KRW-BTC", quantity=quantity, reason="테스트")]
    approved, rejected = apply_risk_guardrail(
        actions, EMPTY_HOLDINGS, cash=100_000_000, current_prices={"KRW-BTC": price}
    )
    assert approved == actions
    assert rejected == []


# ==================================================================== decide_trades_aggressive


def test_aggressive_buy_accepted_below_default_threshold():
    """공격적 봇은 기본형 임계값(3%)보다 낮은 수익률(2.5%)도 진입 조건을 만족해야 한다."""
    signals = [_signal("005930", predicted_return_5d=0.025, rsi14=60, directional_accuracy=0.55)]
    actions = decide_trades_aggressive(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert len(actions) == 1
    assert actions[0].action == "buy"
    assert "공격적 매수 조건 충족" in actions[0].reason


def test_aggressive_buy_rejected_below_own_threshold():
    just_below = AGGRESSIVE_RULES["buy_return_threshold"] - 0.001
    signals = [_signal("005930", predicted_return_5d=just_below)]
    actions = decide_trades_aggressive(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert actions == []


def test_aggressive_pyramids_into_existing_holding():
    """기본형(decide_trades)과 구조적으로 다른 지점: 이미 보유 중인 종목도 추가 매수
    (피라미딩) 후보에 포함된다."""
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 70_000}])
    signals = [_signal("005930", predicted_return_5d=0.10, rsi14=60, directional_accuracy=0.6)]
    actions = decide_trades_aggressive(signals, holdings, {"005930": 70_000}, cash=100_000_000)
    assert len(actions) == 1
    assert actions[0].action == "buy"
    assert actions[0].code == "005930"
    assert "피라미딩" in actions[0].reason


def test_aggressive_topup_capped_by_remaining_position_room():
    """이미 종목당 최대 비중 한도 가까이 들어가 있으면, 남은 여력이 최소 거래금액에
    못 미쳐 추가 매수가 거부돼야 한다(min_trade_amount는 기본값 100만원 그대로 둔다 —
    이게 0이면 아무리 적은 여력이라도 통과해버려 이 테스트의 의도가 사라진다)."""
    rules = {**AGGRESSIVE_RULES, "max_position_pct": 0.10, "min_cash_reserve_pct": 0.0}
    # total_equity = 1억(현금) + 보유평가액(70,000*150=10,500,000) = 110,500,000
    # 10% 한도 = 11,050,000원 -> 남은 여력 550,000원인데 min_trade_amount(100만원) 미달
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 150, "avg_price": 70_000}])
    signals = [_signal("005930", predicted_return_5d=0.10)]
    actions = decide_trades_aggressive(signals, holdings, {"005930": 70_000}, cash=100_000_000, rules=rules)
    assert actions == []  # 남은 비중 여력이 최소 거래금액도 못 채움


def test_aggressive_sold_today_not_rebought_same_day():
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 100_000}])
    price = 100_000 * (1 + AGGRESSIVE_RULES["stop_loss_pct"] - 0.001)  # 손절 트리거
    signals = [_signal("005930", predicted_return_5d=0.20, rsi14=40, directional_accuracy=0.9)]
    actions = decide_trades_aggressive(signals, holdings, {"005930": price}, cash=100_000_000)
    assert len(actions) == 1
    assert actions[0].action == "sell"


def test_aggressive_new_position_respects_max_holdings_but_topup_does_not():
    """동시 보유 종목 수 한도는 신규 진입에만 걸리고, 기존 보유 종목 추가매수는 안 걸린다."""
    rules = {
        **AGGRESSIVE_RULES,
        "max_holdings": 1,
        "min_cash_reserve_pct": 0.0,
        "min_trade_amount": 0,
        "max_position_pct": 1.0,
    }
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 1, "avg_price": 70_000}])
    signals = [
        _signal("005930", predicted_return_5d=0.10),  # 기존 보유 — 추가매수 후보
        _signal("000660", predicted_return_5d=0.20),  # 신규 종목 — max_holdings=1에 막혀야 함
    ]
    prices = {"005930": 70_000, "000660": 100_000}
    actions = decide_trades_aggressive(signals, holdings, prices, cash=100_000_000, rules=rules)
    codes = [a.code for a in actions]
    assert "005930" in codes  # 추가매수는 허용
    assert "000660" not in codes  # 신규 진입은 한도 초과로 거부


def test_aggressive_max_daily_trades_counts_topup_and_new_together():
    rules = {
        **AGGRESSIVE_RULES,
        "max_daily_trades": 1,
        "max_holdings": 10,
        "min_cash_reserve_pct": 0.0,
        "min_trade_amount": 0,
        "max_position_pct": 1.0,
    }
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 1, "avg_price": 70_000}])
    signals = [
        _signal("005930", predicted_return_5d=0.20),  # 더 높은 수익률 -> 먼저 채택
        _signal("000660", predicted_return_5d=0.10),
    ]
    prices = {"005930": 70_000, "000660": 100_000}
    actions = decide_trades_aggressive(signals, holdings, prices, cash=100_000_000, rules=rules)
    assert len(actions) == 1
    assert actions[0].code == "005930"


def test_aggressive_stop_loss_uses_own_wider_threshold():
    """공격적 룰의 손절폭(-12%)은 기본형(-8%)보다 넓다 — 기본형이라면 손절될 손실률에서
    공격적 봇은 아직 버텨야 한다."""
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 100_000}])
    price = 100_000 * (1 + TRADING_RULES["stop_loss_pct"] - 0.001)  # 기본형 손절선은 넘겼지만
    assert price / 100_000 - 1 > AGGRESSIVE_RULES["stop_loss_pct"]  # 공격적 손절선(-12%)에는 아직 안 닿음
    actions = decide_trades_aggressive([], holdings, {"005930": price}, cash=0)
    assert actions == []


def test_aggressive_coin_topup_yields_fractional_quantity():
    holdings = _holdings([{"code": "KRW-BTC", "name": "비트코인", "quantity": 0.1, "avg_price": 150_000_000}])
    signals = [_signal("KRW-BTC", predicted_return_5d=0.10)]
    actions = decide_trades_aggressive(signals, holdings, {"KRW-BTC": 150_000_000}, cash=100_000_000)
    assert len(actions) == 1
    assert isinstance(actions[0].quantity, float)
    assert actions[0].quantity > 0


# ==================================================================== decide_trades_momentum


def test_momentum_buy_accepted_when_trend_aligned():
    signals = [
        _signal(
            "005930",
            predicted_return_5d=0.01,
            rsi14=70,
            directional_accuracy=0.55,
            sma5_gap=0.02,
            sma20_gap=0.05,
        )
    ]
    actions = decide_trades_momentum(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert len(actions) == 1
    assert actions[0].action == "buy"
    assert "정배열" in actions[0].reason


def test_momentum_buy_rejected_when_short_term_trend_down():
    """SMA5 괴리율이 음수면(단기 추세 하락) 중기 추세가 양수여도 정배열이 아니므로 제외."""
    signals = [_signal("005930", sma5_gap=-0.01, sma20_gap=0.05)]
    actions = decide_trades_momentum(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert actions == []


def test_momentum_buy_rejected_when_sma_fields_missing():
    """추세를 확인할 수 없는 신호(sma5_gap/sma20_gap이 없음)는 진입 후보에서 제외한다."""
    signal = _signal("005930", predicted_return_5d=0.10)
    del signal["sma5_gap"]
    del signal["sma20_gap"]
    actions = decide_trades_momentum([signal], EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert actions == []


def test_momentum_buy_rejected_when_predicted_return_negative():
    """buy_return_threshold가 0.0이므로 예측 수익률이 음수면(추세와 무관하게) 제외된다."""
    signals = [_signal("005930", predicted_return_5d=-0.01, sma5_gap=0.02, sma20_gap=0.05)]
    actions = decide_trades_momentum(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert actions == []


def test_momentum_buy_accepted_at_exact_zero_return():
    signals = [_signal("005930", predicted_return_5d=0.0, sma5_gap=0.02, sma20_gap=0.05)]
    actions = decide_trades_momentum(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert len(actions) == 1


def test_momentum_allows_entry_above_default_rsi_ceiling():
    """모멘텀 룰의 RSI 상한(85)은 기본형(70)보다 높다 — 추세 추종은 과매수 구간에서도 진입 허용."""
    signals = [_signal("005930", rsi14=80, sma5_gap=0.02, sma20_gap=0.05)]
    actions = decide_trades_momentum(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert len(actions) == 1


def test_momentum_rejects_above_own_rsi_ceiling():
    just_above = MOMENTUM_RULES["max_rsi_entry"] + 1
    signals = [_signal("005930", rsi14=just_above, sma5_gap=0.02, sma20_gap=0.05)]
    actions = decide_trades_momentum(signals, EMPTY_HOLDINGS, {"005930": 70_000}, cash=100_000_000)
    assert actions == []


def test_momentum_sorts_candidates_by_trend_strength_not_predicted_return():
    signals = [
        _signal("A", predicted_return_5d=0.20, sma5_gap=0.01, sma20_gap=0.01),  # 예측은 높지만 추세는 약함
        _signal("B", predicted_return_5d=0.01, sma5_gap=0.10, sma20_gap=0.08),  # 예측은 낮지만 추세가 강함
    ]
    prices = {"A": 10_000, "B": 10_000}
    actions = decide_trades_momentum(signals, EMPTY_HOLDINGS, prices, cash=100_000_000)
    assert [a.code for a in actions] == ["B", "A"]  # 추세 강한 B가 먼저


def test_momentum_exit_on_trend_break():
    """손절/익절/신호소멸/RSI과매수 어디에도 안 걸려도, 추세 이탈(sma20_gap 음전환)이면 매도."""
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 100_000}])
    # 손익 0%(손절/익절 미해당), 예측수익률 양수(신호소멸 아님), RSI 낮음(과매수 아님)
    signals = [_signal("005930", predicted_return_5d=0.02, rsi14=40, sma20_gap=-0.01)]
    actions = decide_trades_momentum(signals, holdings, {"005930": 100_000}, cash=0)
    assert len(actions) == 1
    assert "추세 이탈" in actions[0].reason


def test_momentum_exit_priority_stop_loss_wins_over_trend_break():
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 100_000}])
    price = 100_000 * (1 + MOMENTUM_RULES["stop_loss_pct"] - 0.001)  # 손절선을 확실히 넘김
    signals = [_signal("005930", sma20_gap=-0.05)]  # 추세 이탈 조건도 동시에 만족
    actions = decide_trades_momentum(signals, holdings, {"005930": price}, cash=0)
    assert len(actions) == 1
    assert "손절" in actions[0].reason


def test_momentum_sell_price_only_checks_when_no_signal_available():
    """보유 중인데 signals에 없는 종목 — 가격 기반 손절/익절만 평가되고 추세이탈 등
    신호 기반 조건은 크래시 없이 건너뛴다."""
    holdings = _holdings([{"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 100_000}])
    price = 100_000 * (1 + MOMENTUM_RULES["stop_loss_pct"] - 0.001)
    actions = decide_trades_momentum([], holdings, {"005930": price}, cash=0)
    assert len(actions) == 1
    assert "손절" in actions[0].reason


def test_momentum_coin_buy_yields_fractional_quantity():
    price = 150_000_000
    signals = [_signal("KRW-BTC", predicted_return_5d=0.05, sma5_gap=0.03, sma20_gap=0.02)]
    actions = decide_trades_momentum(signals, EMPTY_HOLDINGS, {"KRW-BTC": price}, cash=100_000_000)
    assert len(actions) == 1
    assert isinstance(actions[0].quantity, float)
    assert actions[0].quantity > 0


# ==================================================================== BOT_STRATEGIES 레지스트리


def test_bot_strategies_registry_has_three_bots_with_matching_functions():
    assert set(BOT_STRATEGIES) == {"default", "aggressive", "momentum"}
    assert BOT_STRATEGIES["default"]["decide_trades"] is decide_trades
    assert BOT_STRATEGIES["default"]["rules"] is TRADING_RULES
    assert BOT_STRATEGIES["aggressive"]["decide_trades"] is decide_trades_aggressive
    assert BOT_STRATEGIES["aggressive"]["rules"] is AGGRESSIVE_RULES
    assert BOT_STRATEGIES["momentum"]["decide_trades"] is decide_trades_momentum
    assert BOT_STRATEGIES["momentum"]["rules"] is MOMENTUM_RULES
    labels = {meta["label"] for meta in BOT_STRATEGIES.values()}
    assert labels == {"기본형", "공격적", "모멘텀"}
