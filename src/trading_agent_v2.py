"""모의투자 v2 매매 규칙 엔진 — tradermonty Claude 스킬 방법론을 결정론적으로 포팅.

PRD.md 11장 참고. v1(`trading_agent.py`)은 한 글자도 건드리지 않는다 — 라이브 원장을
매일 갱신 중인 코드라 리스크를 최소화하기 위해서다. `infer_market()`·`TradeAction`·
`apply_risk_guardrail()`은 전략이 아니라 공통 유틸이므로 v1에서 그대로 import해 재사용한다.

**v1과의 핵심 차이 — "고정 비중" → "리스크 기반 사이징"**: v1은 포지션 크기를 총자산의
고정 %(`max_position_pct`)로 정했지만, v2는 `position-sizer` 스킬의 방식(Fixed
Fractional/ATR 기반)을 그대로 포팅해 "계좌 자산의 `risk_pct_per_trade`%만 잃을 각오로,
손절폭(`atr_multiplier` × ATR14)에 맞춰 수량을 역산"한다 — `_calc_quantity_v2()` 참고.
`max_position_pct` 등 v1과 동일한 리스크 한도는 여전히 `apply_risk_guardrail()`이 재검증한다
(변경 없음, PRD 11.3).

**진입 조건이 세 봇마다 구조적으로 다르다** (PRD 11.1 — 실제 tradermonty SKILL.md 원문에서
포팅):
- 기본형(`decide_trades_v2_default`): `technical-analyst`의 MA 정렬 — 종가 > SMA20 > SMA60
- 공격적(`decide_trades_v2_aggressive`): `vcp-screener`(Minervini VCP)의 변동성 수축 + 고점
  근접(피벗). v1 공격적 봇과 동일하게 **피라미딩(기존 보유 종목 추가매수)을 허용**한다.
- 모멘텀(`decide_trades_v2_momentum`): `stockbee-momentum-burst-screener`의 4% 브레이크아웃
  + 거래량/레인지 확장 + 종가 위치(고가 근처 마감)

세 봇 모두 `predicted_return_5d`/`directional_accuracy`/`news_sentiment`(predictor.py 기반,
v1이 이미 매일 운영 중인 신호)를 진입 필터로 그대로 재사용한다 — PRD 11.0에서 정한 대로
"tradermonty 스킬을 이용한다"는 것은 이 신호들을 대체하는 게 아니라, "무엇을 살지"
판단하는 진입/청산 메커니즘과 포지션 크기 계산 방식만 스킬 방법론으로 바꾸는 것이다.

**신호 스키마 — v1과 다른 점**: 아래 함수들은 `signals`의 각 dict에 v1 필드(code, name,
predicted_return_5d, rsi14, news_sentiment, directional_accuracy)에 더해 v2 전용 필드가
있어야 정상 동작한다 (없으면 그 종목은 해당 조건에서 조용히 제외된다 — v1의 "신호를 못
만들면 관망" 원칙과 동일):
- 공통: `atr14`
- 기본형: `sma20`, `sma60` (SMA5/20/60_gap이 아니라 **원래 가격 단위** — 종가와 직접
  비교해 정배열을 판정하므로)
- 공격적: `atr_contraction_ratio`(최근 ATR / `contraction_lookback`일 전 ATR),
  `dist_from_60d_high`(종가/60일 고가 - 1)
- 모멘텀: `day_change`(전일 대비 등락률), `volume_ratio_1d`(전일 거래량 대비),
  `day_range`(당일 고가-저가), `prev_range_max3`(직전 3일 레인지 최댓값),
  `close_location`((종가-저가)/(고가-저가)), `low5`(최근 5거래일 저가 중 최솟값 —
  모멘텀 봇 전용 "추세 이탈" 청산 조건에 쓴다)

**ATR 기반 손절/익절의 한계**: v1의 고정% 손절은 가격만 있으면 계산되지만, v2의 ATR 기반
손절/익절은 그 종목의 최신 신호(`atr14`)가 있어야 계산할 수 있다. 보유 종목인데 그날
신호 생성이 실패하면(개별 종목 예외 등) 이번 실행에서는 그 종목의 손절/익절/신호소멸/RSI
과매수 판단을 전부 건너뛴다(가격만으로 판단 가능한 v1과 다른 지점 — PRD 11.4). 호출부
(`scripts/run_daily_trading.py`)가 보유 종목을 항상 신호 생성 대상에 포함시키므로(v1과
동일한 원칙) 실무에서는 드물게만 발생할 것으로 예상한다.

**진입 시점 ATR을 저장하지 않는다**: `holdings.csv` 스키마에 컬럼을 추가하면 v1과 같은
마이그레이션 리스크가 생긴다(PRD 11.4). 대신 매도 판단 시점마다 그 종목의 "현재" ATR14로
손절/익절가를 다시 계산한다 — 손절폭이 시간에 따라 조금씩 움직이는 근사이지만, v1의 CSV
스키마를 그대로 재사용할 수 있는 실용적 절충이다.
"""

from __future__ import annotations

import pandas as pd

from .trading_agent import TradeAction, apply_risk_guardrail, infer_market

__all__ = [
    "TRADING_RULES_V2_DEFAULT",
    "TRADING_RULES_V2_AGGRESSIVE",
    "TRADING_RULES_V2_MOMENTUM",
    "decide_trades_v2_default",
    "decide_trades_v2_aggressive",
    "decide_trades_v2_momentum",
    "apply_risk_guardrail",  # v1과 그대로 공유 — pages/모의투자_v2.py 등 호출부 편의상 재노출
    "BOT_STRATEGIES_V2",
]

# PRD.md 11.3 — 11.7 백테스트(scripts/backtest_v2_strategies.py, 2026-08-23 실측)로 검증된
# 최종 수치. 모멘텀의 risk_pct_per_trade는 초안 1.5%에서 1.0%로 낮췄다(MDD -40.5% → -29.1%).
TRADING_RULES_V2_DEFAULT = {
    "risk_pct_per_trade": 0.01,  # 계좌 자산의 1% — position-sizer "1% rule"
    "atr_multiplier": 2.0,  # 손절 = 평단가 - ATR14 × 2.0
    "take_profit_atr_multiplier": 3.0,  # 익절 = 평단가 + ATR14 × 3.0
    "min_directional_accuracy": 0.55,  # v1과 동일 — 예측 신뢰도 필터
    "min_news_sentiment": -0.3,
    "max_rsi_entry": 70,
    "max_rsi_exit": 80,
    "exit_on_negative_signal": True,
    # 아래 4개는 v1 TRADING_RULES와 동일한 구조 — apply_risk_guardrail()을 그대로
    # 재사용하려면 반드시 같은 키 이름을 써야 한다
    "max_position_pct": 0.15,
    "max_holdings": 10,
    "min_trade_amount": 1_000_000,
    "max_daily_trades": 5,
    "min_cash_reserve_pct": 0.05,
}

TRADING_RULES_V2_AGGRESSIVE = {
    **TRADING_RULES_V2_DEFAULT,
    "risk_pct_per_trade": 0.02,  # position-sizer "예외적 사유 없인 2% 초과 금지" 상한을 그대로 채용
    "atr_multiplier": 2.5,
    "take_profit_atr_multiplier": None,  # 피라미딩 전략이라 개별 익절 대신 종목당 비중 상한이 사실상의 상한
    "contraction_lookback": 10,  # ATR 수축 비교 구간(최근 vs contraction_lookback일 전)
    "contraction_ratio_max": 0.85,  # 최근 ATR ≤ 이전 ATR × 0.85 → 수축으로 판정
    "pivot_proximity_pct": 0.05,  # 최근 60일 고점 대비 -5% 이내
    "max_position_pct": 0.25,
    "max_holdings": 15,
    "max_daily_trades": 8,
    "min_cash_reserve_pct": 0.03,
}

TRADING_RULES_V2_MOMENTUM = {
    **TRADING_RULES_V2_DEFAULT,
    "risk_pct_per_trade": 0.01,  # 초안 1.5% → 11.7 백테스트 실측(MDD -40.5%)으로 1.0% 하향
    "atr_multiplier": 1.5,  # 버스트 추종은 손절을 타이트하게(stockbee 스타일 — 빠른 손절)
    "take_profit_atr_multiplier": 2.0,  # 짧게 먹고 나간다
    "max_rsi_entry": 85,  # 추세 추종은 과매수 구간에서도 진입 허용(v1 모멘텀과 동일한 완화)
    "max_rsi_exit": 75,
    "burst_threshold": 0.04,  # stockbee "4% breakout" 원안 그대로
    "volume_ratio_min": 1.0,  # 거래량이 전일 이상
    "close_location_min": 0.6,  # (종가-저가)/(고가-저가) ≥ 0.6 — 고가 근처 마감
}


def _calc_quantity_v2(code: str, price: float, atr14: float, total_equity: float, rules: dict) -> int | float:
    """리스크 기반 포지션 사이징(position-sizer Fixed Fractional/ATR 모드 포팅, PRD 11.3).

    risk_dollars(계좌 자산의 risk_pct_per_trade%) / stop_distance(ATR14 × atr_multiplier,
    원 단위 손절폭) = 수량. v1의 `_calc_quantity()`와 동일하게 코인은 소수 8자리 반올림,
    그 외(국내/해외증시)는 정수 주 단위로 내림한다. `max_position_pct` 등 상한은 여기서
    걸지 않는다 — `apply_risk_guardrail()`이 그대로 재검증한다(변경 없음, PRD 11.3).
    """
    stop_distance = atr14 * rules["atr_multiplier"]
    if stop_distance <= 0:
        return 0
    risk_dollars = total_equity * rules["risk_pct_per_trade"]
    raw_quantity = risk_dollars / stop_distance
    if infer_market(code) == "COIN":
        return round(raw_quantity, 8)
    return int(raw_quantity)


def _atr_exit_reason(row_avg_price: float, price: float, atr14: float | None, rules: dict) -> str | None:
    """ATR 기반 손절/익절 판단 — 세 봇의 청산 루프가 공통으로 쓴다(PRD 11.4).

    atr14가 없거나(신호 생성 실패) 0 이하이면 이 조건 자체를 건너뛴다(None 반환) — 위
    모듈 docstring "ATR 기반 손절/익절의 한계" 참고.
    """
    if atr14 is None or pd.isna(atr14) or atr14 <= 0:
        return None
    stop_price = row_avg_price - atr14 * rules["atr_multiplier"]
    if price <= stop_price:
        return (
            f"ATR14({atr14:,.1f}) × {rules['atr_multiplier']}배 손절가 {stop_price:,.0f}원 이하로 하락 — 손절"
        )
    take_profit_mult = rules.get("take_profit_atr_multiplier")
    if take_profit_mult is not None:
        target_price = row_avg_price + atr14 * take_profit_mult
        if price >= target_price:
            return f"ATR14({atr14:,.1f}) × {take_profit_mult}배 익절가 {target_price:,.0f}원 이상 도달 — 익절"
    return None


# ==================================================================== 기본형(v2) — MA 정렬 + 리스크 사이징


def decide_trades_v2_default(
    signals: list[dict],
    holdings: pd.DataFrame,
    current_prices: dict[str, float],
    cash: float,
    rules: dict = TRADING_RULES_V2_DEFAULT,
) -> list[TradeAction]:
    """`technical-analyst` 스킬의 MA 정렬(종가 > SMA20 > SMA60)을 1차 진입 조건으로 쓰고,
    `position-sizer`식 ATR 리스크 사이징으로 수량을 정한다(모듈 docstring 참고).

    v1 `decide_trades()`와 달리 predicted_return_5d 임계값(buy_return_threshold)은 진입
    게이트가 아니다 — MA 정렬이 게이트이고, predicted_return_5d는 후보가 여럿일 때
    우선순위 정렬 기준으로만 쓴다(추세 방향이 맞는 종목 중 예측 수익률이 더 높은 쪽을
    먼저 채택).
    """
    signal_by_code = {s["code"]: s for s in signals}
    actions: list[TradeAction] = []

    for _, row in holdings.iterrows():
        code = row["code"]
        price = current_prices.get(code)
        if price is None:
            continue
        signal = signal_by_code.get(code)
        if signal is None:
            continue  # ATR을 모르면 이번 실행에서 이 종목은 판단 보류(모듈 docstring 참고)

        reason = _atr_exit_reason(row["avg_price"], price, signal.get("atr14"), rules)
        if reason is None and rules["exit_on_negative_signal"] and signal["predicted_return_5d"] < 0:
            reason = f"예측 수익률이 {signal['predicted_return_5d']:+.1%}로 음전환 — 신호 소멸로 매도"
        if reason is None and signal["rsi14"] >= rules["max_rsi_exit"]:
            reason = f"RSI {signal['rsi14']:.0f} — 과매수 구간 진입으로 매도"

        if reason is not None:
            actions.append(TradeAction(action="sell", code=code, quantity=row["quantity"], reason=reason))

    held_codes = set(holdings["code"])
    holdings_after_sells = len(holdings) - sum(1 for a in actions if a.action == "sell")

    buy_candidates = []
    for s in signals:
        code = s["code"]
        if code in held_codes:
            continue
        price = current_prices.get(code)
        if price is None:
            continue
        atr14 = s.get("atr14")
        sma20, sma60 = s.get("sma20"), s.get("sma60")
        if atr14 is None or pd.isna(atr14) or atr14 <= 0:
            continue
        if sma20 is None or sma60 is None or pd.isna(sma20) or pd.isna(sma60):
            continue
        if not (price > sma20 > sma60):
            continue  # MA 정렬(정배열) 아님 — 진입 게이트 불충족
        if s["directional_accuracy"] < rules["min_directional_accuracy"]:
            continue
        if s["rsi14"] > rules["max_rsi_entry"]:
            continue
        if s["news_sentiment"] < rules["min_news_sentiment"]:
            continue
        buy_candidates.append((s, price, atr14))

    buy_candidates.sort(key=lambda t: t[0]["predicted_return_5d"], reverse=True)

    holdings_value = sum(
        row["quantity"] * current_prices.get(row["code"], row["avg_price"]) for _, row in holdings.iterrows()
    )
    total_equity = cash + holdings_value
    buy_budget = max(cash - total_equity * rules["min_cash_reserve_pct"], 0.0)
    room_for_new = max(rules["max_holdings"] - holdings_after_sells, 0)

    n_buys = 0
    for s, price, atr14 in buy_candidates:
        if n_buys >= rules["max_daily_trades"] or n_buys >= room_for_new:
            break
        quantity = _calc_quantity_v2(s["code"], price, atr14, total_equity, rules)
        amount = quantity * price
        if quantity <= 0 or amount < rules["min_trade_amount"] or amount > buy_budget:
            continue  # 예산이 부족한 후보만 건너뛰고 다음 후보를 계속 시도한다
        reason = (
            f"MA 정렬(종가>SMA20>SMA60), 예측 5일 수익률 {s['predicted_return_5d']:+.1%}, "
            f"방향적중률 {s['directional_accuracy']:.0%} — ATR14 리스크 {rules['risk_pct_per_trade']:.0%} 사이징 매수"
        )
        actions.append(TradeAction(action="buy", code=s["code"], quantity=quantity, reason=reason))
        buy_budget -= amount
        n_buys += 1

    return actions


# ==================================================================== 공격적(v2) — 변동성 수축(VCP) + 피라미딩


def decide_trades_v2_aggressive(
    signals: list[dict],
    holdings: pd.DataFrame,
    current_prices: dict[str, float],
    cash: float,
    rules: dict = TRADING_RULES_V2_AGGRESSIVE,
) -> list[TradeAction]:
    """`vcp-screener`(Minervini VCP)의 변동성 수축 + 고점 근접(피벗)을 진입 조건으로 쓴다.

    v1 `decide_trades_aggressive()`와 동일하게 **이미 보유한 종목도 추가 매수(피라미딩)
    후보에 포함**한다(당일 매도한 종목만 제외). 종목당 최대 비중 한도는 여기서
    선제적으로 계산해 자르지 않고 `apply_risk_guardrail()`이 통째로 거부하게 둔다 — v1의
    선제 클램핑보다 단순하고, "한도 초과는 거부"라는 가드레일의 기존 철학과 일관된다.

    청산은 ATR 기반 손절만 있다(`take_profit_atr_multiplier=None` — 피라미딩 전략은
    개별 익절 대신 종목당 비중 상한이 사실상의 익절 역할을 한다, PRD 11.3/11.8).
    """
    signal_by_code = {s["code"]: s for s in signals}
    actions: list[TradeAction] = []

    for _, row in holdings.iterrows():
        code = row["code"]
        price = current_prices.get(code)
        if price is None:
            continue
        signal = signal_by_code.get(code)
        if signal is None:
            continue

        reason = _atr_exit_reason(row["avg_price"], price, signal.get("atr14"), rules)
        if reason is None and rules["exit_on_negative_signal"] and signal["predicted_return_5d"] < 0:
            reason = f"예측 수익률이 {signal['predicted_return_5d']:+.1%}로 음전환 — 신호 소멸로 매도"
        if reason is None and signal["rsi14"] >= rules["max_rsi_exit"]:
            reason = f"RSI {signal['rsi14']:.0f} — 과매수 구간 진입으로 매도"

        if reason is not None:
            actions.append(TradeAction(action="sell", code=code, quantity=row["quantity"], reason=reason))

    sold_today = {a.code for a in actions}
    holdings_after_sells = len(holdings) - len(actions)
    held_value_by_code = {
        row["code"]: row["quantity"] * current_prices.get(row["code"], row["avg_price"])
        for _, row in holdings.iterrows()
        if row["code"] not in sold_today
    }

    buy_candidates = []
    for s in signals:
        code = s["code"]
        if code in sold_today:
            continue  # 당일 매도한 종목만 재매수 후보에서 제외 — 그 외 보유종목은 피라미딩 후보 유지
        price = current_prices.get(code)
        if price is None:
            continue
        atr14 = s.get("atr14")
        contraction, dist_high = s.get("atr_contraction_ratio"), s.get("dist_from_60d_high")
        if atr14 is None or pd.isna(atr14) or atr14 <= 0:
            continue
        if contraction is None or dist_high is None or pd.isna(contraction) or pd.isna(dist_high):
            continue
        if contraction > rules["contraction_ratio_max"]:
            continue  # 충분히 수축되지 않음
        if dist_high < -rules["pivot_proximity_pct"]:
            continue  # 피벗(최근 고점)에서 너무 멀리 떨어짐
        if s["directional_accuracy"] < rules["min_directional_accuracy"]:
            continue
        if s["rsi14"] > rules["max_rsi_entry"]:
            continue
        if s["news_sentiment"] < rules["min_news_sentiment"]:
            continue
        buy_candidates.append((s, price, atr14, contraction))

    buy_candidates.sort(key=lambda t: t[3])  # 변동성 수축비 오름차순 — 더 강하게 수축된(타이트한) 종목 우선

    holdings_value = sum(held_value_by_code.values())
    total_equity = cash + holdings_value
    buy_budget = max(cash - total_equity * rules["min_cash_reserve_pct"], 0.0)
    room_for_new = max(rules["max_holdings"] - holdings_after_sells, 0)

    n_buys = 0
    n_new_positions = 0
    for s, price, atr14, contraction in buy_candidates:
        code = s["code"]
        is_topup = code in held_value_by_code
        if n_buys >= rules["max_daily_trades"]:
            break
        if not is_topup and n_new_positions >= room_for_new:
            continue  # 신규 종목 한도 — 기존 보유 종목 추가매수는 이 한도에 안 걸린다

        quantity = _calc_quantity_v2(code, price, atr14, total_equity, rules)
        amount = quantity * price
        if quantity <= 0 or amount < rules["min_trade_amount"] or amount > buy_budget:
            continue

        if is_topup:
            reason = (
                f"이미 보유 중 — 변동성 수축비 {contraction:.2f}, 고점대비 {s['dist_from_60d_high']:+.1%} "
                f"— ATR14 리스크 {rules['risk_pct_per_trade']:.0%} 사이징 추가 매수(피라미딩)"
            )
        else:
            reason = (
                f"변동성 수축비 {contraction:.2f}(≤{rules['contraction_ratio_max']}), "
                f"고점대비 {s['dist_from_60d_high']:+.1%} — ATR14 리스크 {rules['risk_pct_per_trade']:.0%} "
                f"사이징 공격적 매수"
            )
        actions.append(TradeAction(action="buy", code=code, quantity=quantity, reason=reason))
        buy_budget -= amount
        held_value_by_code[code] = held_value_by_code.get(code, 0.0) + amount
        n_buys += 1
        if not is_topup:
            n_new_positions += 1

    return actions


# ==================================================================== 모멘텀(v2) — 4% 버스트


def decide_trades_v2_momentum(
    signals: list[dict],
    holdings: pd.DataFrame,
    current_prices: dict[str, float],
    cash: float,
    rules: dict = TRADING_RULES_V2_MOMENTUM,
) -> list[TradeAction]:
    """`stockbee-momentum-burst-screener`의 4% 브레이크아웃 + 거래량/레인지 확장 + 종가
    위치(고가 근처 마감)를 진입 조건으로 쓴다.

    청산은 ATR 기반 손절/익절(기본형과 동일한 `_atr_exit_reason`)에 **추세 이탈**(종가가
    최근 5거래일 저가 하회, 신호의 `low5` 필드)이 하나 더 붙는다 — v1 모멘텀 봇의 "SMA20
    이탈"보다 촘촘한 윈도우를 쓴다(버스트 전략은 짧은 보유기간을 전제, PRD 11.4).
    """
    signal_by_code = {s["code"]: s for s in signals}
    actions: list[TradeAction] = []

    for _, row in holdings.iterrows():
        code = row["code"]
        price = current_prices.get(code)
        if price is None:
            continue
        signal = signal_by_code.get(code)
        if signal is None:
            continue

        reason = _atr_exit_reason(row["avg_price"], price, signal.get("atr14"), rules)
        low5 = signal.get("low5")
        if reason is None and low5 is not None and not pd.isna(low5) and price <= low5:
            reason = f"최근 5거래일 저점({low5:,.0f}원) 이하로 하락 — 추세 이탈로 매도"
        if reason is None and rules["exit_on_negative_signal"] and signal["predicted_return_5d"] < 0:
            reason = f"예측 수익률이 {signal['predicted_return_5d']:+.1%}로 음전환 — 신호 소멸로 매도"
        if reason is None and signal["rsi14"] >= rules["max_rsi_exit"]:
            reason = f"RSI {signal['rsi14']:.0f} — 과매수 구간 진입으로 매도"

        if reason is not None:
            actions.append(TradeAction(action="sell", code=code, quantity=row["quantity"], reason=reason))

    held_codes = set(holdings["code"])
    holdings_after_sells = len(holdings) - len(actions)

    buy_candidates = []
    for s in signals:
        code = s["code"]
        if code in held_codes:
            continue
        price = current_prices.get(code)
        if price is None:
            continue
        atr14 = s.get("atr14")
        fields = (
            s.get("day_change"),
            s.get("volume_ratio_1d"),
            s.get("day_range"),
            s.get("prev_range_max3"),
            s.get("close_location"),
        )
        if atr14 is None or pd.isna(atr14) or atr14 <= 0:
            continue
        if any(v is None or pd.isna(v) for v in fields):
            continue
        day_change, vol_ratio, day_range, prev_range_max3, close_loc = fields
        if day_change < rules["burst_threshold"]:
            continue
        if vol_ratio < rules["volume_ratio_min"]:
            continue
        if day_range <= prev_range_max3:
            continue
        if close_loc < rules["close_location_min"]:
            continue
        if s["directional_accuracy"] < rules["min_directional_accuracy"]:
            continue
        if s["rsi14"] > rules["max_rsi_entry"]:
            continue
        if s["news_sentiment"] < rules["min_news_sentiment"]:
            continue
        buy_candidates.append((s, price, atr14))

    buy_candidates.sort(key=lambda t: t[0]["day_change"], reverse=True)  # 버스트 강도(당일 상승률) 내림차순

    holdings_value = sum(
        row["quantity"] * current_prices.get(row["code"], row["avg_price"]) for _, row in holdings.iterrows()
    )
    total_equity = cash + holdings_value
    buy_budget = max(cash - total_equity * rules["min_cash_reserve_pct"], 0.0)
    room_for_new = max(rules["max_holdings"] - holdings_after_sells, 0)

    n_buys = 0
    for s, price, atr14 in buy_candidates:
        if n_buys >= rules["max_daily_trades"] or n_buys >= room_for_new:
            break
        quantity = _calc_quantity_v2(s["code"], price, atr14, total_equity, rules)
        amount = quantity * price
        if quantity <= 0 or amount < rules["min_trade_amount"] or amount > buy_budget:
            continue
        reason = (
            f"당일 {s['day_change']:+.1%} 급등(4% 브레이크아웃), 거래량 {s['volume_ratio_1d']:.1f}배, "
            f"종가위치 {s['close_location']:.0%} — 모멘텀 버스트 매수 조건 충족"
        )
        actions.append(TradeAction(action="buy", code=s["code"], quantity=quantity, reason=reason))
        buy_budget -= amount
        n_buys += 1

    return actions


# ==================================================================== 봇 버전 레지스트리

# v1 BOT_STRATEGIES와 같은 형태({bot_id: {label, decide_trades, rules}}) — 별도 딕셔너리로
# 완전히 분리한다(PRD 11.2). scripts/run_daily_trading.py와 pages/모의투자_v2.py가 각자
# 하드코딩하지 않고 이 레지스트리 하나만 순회한다.
BOT_STRATEGIES_V2 = {
    "default": {
        "label": "기본형",
        "decide_trades": decide_trades_v2_default,
        "rules": TRADING_RULES_V2_DEFAULT,
    },
    "aggressive": {
        "label": "공격적",
        "decide_trades": decide_trades_v2_aggressive,
        "rules": TRADING_RULES_V2_AGGRESSIVE,
    },
    "momentum": {
        "label": "모멘텀",
        "decide_trades": decide_trades_v2_momentum,
        "rules": TRADING_RULES_V2_MOMENTUM,
    },
}
