"""매매 규칙 엔진 + 리스크 가드레일 — 모의투자 매매 판단.

PRD.md 5.2·5.3·6장 참고. LLM을 쓰지 않는다(원안은 PRD 9.1에 보존, 유료 서비스라 보류) —
예측 신호(predictor.py 산출물 기반)에 임계값 규칙을 적용해 결정론적으로 매매를 판단한다.
외부 API 호출이 없으므로 전부 순수 함수이고, 오프라인에서 바로 테스트할 수 있다.

국내(KOSPI/KOSDAQ)·해외증시(나스닥)·코인(업비트) 세 시장을 모두 다룬다. 원장 스키마에
market 컬럼을 두지 않고, `infer_market()`이 종목코드 문자열의 형식(6자리 숫자/"KRW-"
접두사/그 외)만으로 시장을 판별한다 — 리스크 규칙(TRADING_RULES)은 세 시장 공통이다.

decide_trades()가 "무엇을 얼마나 살지" 제안하고, apply_risk_guardrail()이 그 제안을
현금·보유수량·리스크 한도로 다시 검증한다. 5.2가 이미 같은 TRADING_RULES를 참조해
판단하므로 원칙적으로 위반이 없어야 맞지만, 로직 버그(부호 반전, 여러 매수가 겹쳐
현금이 실제로는 모자란 경우 등)에 대비한 두 번째 독립 방어선으로 가드레일을 분리해뒀다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# 매수 조건, 청산 조건, 리스크 한도를 한 곳에 모은다 — decide_trades()와
# apply_risk_guardrail()이 반드시 같은 딕셔너리를 참조해야 "판단 쪽과 검증 쪽 숫자가
# 어긋나는" 문제가 안 생긴다. PRD.md 6장 표와 정확히 같은 값이다.
TRADING_RULES = {
    # 매수 조건
    "buy_return_threshold": 0.03,  # 예측 5거래일 수익률 +3% 이상
    "min_directional_accuracy": 0.55,  # 홀드아웃 방향 적중률 55% 이상 (예측 신뢰도 필터)
    "max_rsi_entry": 70,  # RSI 70 이하만 진입 — 과매수 구간 추격매수 방지
    "min_news_sentiment": -0.3,  # 뉴스 감성이 심하게 부정적이지 않을 것
    # 청산(매도) 조건 — 보유 종목에 적용
    "stop_loss_pct": -0.08,  # 평단가 대비 -8% 손절
    "take_profit_pct": 0.15,  # 평단가 대비 +15% 익절
    "max_rsi_exit": 80,  # RSI 80 이상이면 과매수 청산
    "exit_on_negative_signal": True,  # 예측 5일 수익률이 음전환되면 청산
    # 리스크 한도 (PRD 6장과 동일)
    "max_position_pct": 0.15,
    "max_holdings": 10,
    "min_trade_amount": 1_000_000,
    "max_daily_trades": 5,
    "min_cash_reserve_pct": 0.05,
}

# 공격적 봇 — 진입 문턱을 낮추고 포지션을 더 크게/많이 가져가며, 손절·익절 폭도 넓혀
# 변동성을 더 많이 감내한다. decide_trades_aggressive()가 참조한다.
AGGRESSIVE_RULES = {
    **TRADING_RULES,
    "buy_return_threshold": 0.02,
    "min_directional_accuracy": 0.52,
    "max_rsi_entry": 80,
    "min_news_sentiment": -0.5,
    "stop_loss_pct": -0.12,
    "take_profit_pct": 0.25,
    "max_rsi_exit": 85,
    "max_position_pct": 0.25,
    "max_holdings": 15,
    "max_daily_trades": 8,
    "min_cash_reserve_pct": 0.03,
}

# 모멘텀 봇 — 진입의 1차 조건이 예측 수익률 크기가 아니라 추세 정렬(SMA5·SMA20 위)이라
# buy_return_threshold를 0으로 낮춘다("예측이 최소 양전환"만 확인). 추세는 꺾이면 빨리
# 정리하는 게 전략의 핵심이라 손절·익절 폭은 기본형보다 좁힌다. decide_trades_momentum()이
# 참조한다.
MOMENTUM_RULES = {
    **TRADING_RULES,
    "buy_return_threshold": 0.0,
    "min_directional_accuracy": 0.50,
    "max_rsi_entry": 85,
    "min_news_sentiment": -0.3,
    "stop_loss_pct": -0.06,
    "take_profit_pct": 0.12,
    "max_rsi_exit": 75,
}


def infer_market(code: str) -> str:
    """종목코드 문자열 패턴만으로 시장을 판별한다 — 원장 스키마에 market 컬럼을 추가하는
    마이그레이션을 피하려고 코드 자체의 형식적 특징에 의존한다. 국내(KR)는 항상 정확히
    6자리 숫자, 코인(COIN)은 항상 "KRW-" 접두사, 그 외(미국 티커 등, 숫자로만 이루어지지
    않는다)는 해외증시(US)로 본다. 세 패턴은 서로 겹치지 않는다."""
    if code.startswith("KRW-"):
        return "COIN"
    if code.isdigit() and len(code) == 6:
        return "KR"
    return "US"


@dataclass(frozen=True)
class TradeAction:
    """decide_trades()가 제안하고 apply_risk_guardrail()이 승인/거부하는 매매 액션 1건.

    price·name은 일부러 안 담는다 — 체결가는 그날 종가 하나로 캐노니컬하게 정해지므로
    (PRD 5.3 "체결가 가정") 호출부가 current_prices에서 조회해 쓰고, name도 호출부가
    holdings/candidates에서 이미 갖고 있는 값을 그대로 쓰면 된다. 여기서 복사해 들고
    다니면 두 값이 어긋날 여지만 생긴다.

    quantity는 국내/해외증시는 정수(주 단위)지만, 코인은 소수일 수 있다.
    """

    action: str  # "buy" | "sell"
    code: str
    quantity: int | float
    reason: str


def _action_dict(a: TradeAction) -> dict:
    return {"action": a.action, "code": a.code, "quantity": a.quantity, "reason": a.reason}


def _calc_quantity(code: str, target_amount: float, price: float) -> int | float:
    """목표 배분 금액을 그날 가격으로 나눠 매수 수량을 정한다 — 코인은 1주 단위 개념이
    없고 고가 코인(비트코인 등)은 정수 단위로는 예산 안에서 1개도 못 사는 경우가 흔해
    소수 8자리로 반올림하고, 그 외(국내/해외증시)는 정수 주 단위로 내림한다. 세 판단
    함수(decide_trades/_aggressive/_momentum)가 전략은 다르지만 이 수량 산출 로직만은
    공유한다(전략 차이가 아니라 반복 계산이라 DRY 대상)."""
    if infer_market(code) == "COIN":
        return round(target_amount / price, 8)
    return int(target_amount // price)


# ==================================================================== ② 매매 규칙 엔진


def decide_trades(
    signals: list[dict],
    holdings: pd.DataFrame,
    current_prices: dict[str, float],
    cash: float,
    rules: dict = TRADING_RULES,
) -> list[TradeAction]:
    """워치리스트+보유종목 신호와 현재 포지션을 보고 오늘의 매매를 제안한다.

    국내·해외증시·코인 신호가 섞여 들어올 수 있다 — `infer_market(code)`가 코드 패턴으로
    시장을 판별해, 코인은 매수 수량을 정수 주가 아니라 소수(8자리 반올림)로 계산한다.

    signals: [{code, name, predicted_return_5d, rsi14, news_sentiment,
    directional_accuracy}, ...] — predictor.py 산출물 기반. **보유 종목도 반드시
    포함해야 한다** — 청산 조건(신호 소멸, RSI 과매수)이 보유 종목의 최신 신호를
    필요로 하므로, 호출부는 워치리스트뿐 아니라 현재 보유 종목까지 합쳐 예측을 돌린
    뒤 넘겨야 한다. 보유 종목인데 signals에 없으면(예측 실패 등) 가격 기반 청산
    조건(손절/익절)만 평가하고 신호 기반 조건(신호소멸/RSI과매수)은 건너뛴다.

    current_prices: {code: 그날 종가}. 값이 없는 종목은 이 함수가 그 종목에 대해
    아무 매매도 결정하지 않는다 — 거래정지 등으로 가격이 없으면 관망.

    판단 순서: 청산(매도) 전부 판단 → 신규 매수 후보를 예측 5일 수익률 내림차순
    정렬 → 한도 내에서 채택.

    **max_daily_trades는 신규 매수에만 적용한다.** 손절/익절 등 청산은 자본을
    보호하는 행동이라 거래 횟수 한도로 막으면 정작 손실을 줄여야 하는 날에 못 파는
    모순이 생기기 때문이다 — PRD 6장 표에 명시된 해석은 아니고, 이 함수가 내린
    판단이다. apply_risk_guardrail()도 반드시 같은 해석을 따른다.

    **이미 보유 중인 종목은(오늘 매도됐어도) 신규 매수 후보에서 제외한다** — 당일
    매도 후 재매수(휩쏘) 방지 및 무제한 물타기 방지. PRD는 이 부분까지 명시하지
    않았고, "신규 진입"이라는 표현에 맞춰 보수적으로 해석한 것이다.
    """
    signal_by_code = {s["code"]: s for s in signals}
    actions: list[TradeAction] = []

    # --- 청산(매도): 보유 종목마다 조건을 순서대로 확인, 하나라도 맞으면 즉시 전량 매도 ---
    for _, row in holdings.iterrows():
        code = row["code"]
        price = current_prices.get(code)
        if price is None:
            continue  # 가격을 모르면 팔 수도 없다 — 오늘은 그냥 보유 유지

        pnl_pct = (price - row["avg_price"]) / row["avg_price"]
        signal = signal_by_code.get(code)
        reason = None

        if pnl_pct <= rules["stop_loss_pct"]:
            reason = f"평단가 대비 {pnl_pct:+.1%} — 손절 기준({rules['stop_loss_pct']:.0%}) 도달로 매도"
        elif pnl_pct >= rules["take_profit_pct"]:
            reason = f"평단가 대비 {pnl_pct:+.1%} — 익절 기준({rules['take_profit_pct']:.0%}) 도달로 매도"
        elif signal is not None and rules["exit_on_negative_signal"] and signal["predicted_return_5d"] < 0:
            reason = f"예측 수익률이 {signal['predicted_return_5d']:+.1%}로 음전환 — 신호 소멸로 매도"
        elif signal is not None and signal["rsi14"] >= rules["max_rsi_exit"]:
            reason = f"RSI {signal['rsi14']:.0f} — 과매수 구간 진입으로 매도"

        if reason is not None:
            # int()로 캐스팅하면 코인처럼 1 미만 소수 수량을 보유 중인 경우 quantity가 0이
            # 되어, portfolio.apply_trade()의 "quantity는 양수여야 합니다" 예외로 그날 실행
            # 전체가 죽는다 — 보유 수량을 그대로(코인이면 float) 써서 전량 매도한다.
            actions.append(TradeAction(action="sell", code=code, quantity=row["quantity"], reason=reason))

    # --- 신규 매수: 조건을 모두 만족하는 후보를 예측 5일 수익률 내림차순으로 채택 ---
    held_codes = set(holdings["code"])
    num_sells = sum(1 for a in actions if a.action == "sell")
    holdings_after_sells = len(holdings) - num_sells

    buy_candidates = []
    for s in signals:
        code = s["code"]
        if code in held_codes:
            continue
        price = current_prices.get(code)
        if price is None:
            continue
        if s["predicted_return_5d"] < rules["buy_return_threshold"]:
            continue
        if s["directional_accuracy"] < rules["min_directional_accuracy"]:
            continue
        if s["rsi14"] > rules["max_rsi_entry"]:
            continue
        if s["news_sentiment"] < rules["min_news_sentiment"]:
            continue
        buy_candidates.append((s, price))

    buy_candidates.sort(key=lambda sp: sp[0]["predicted_return_5d"], reverse=True)

    holdings_value = sum(
        row["quantity"] * current_prices.get(row["code"], row["avg_price"]) for _, row in holdings.iterrows()
    )
    total_equity = cash + holdings_value
    buy_budget = max(cash - total_equity * rules["min_cash_reserve_pct"], 0.0)
    room_for_new = max(rules["max_holdings"] - holdings_after_sells, 0)

    n_buys = 0
    for s, price in buy_candidates:
        if n_buys >= rules["max_daily_trades"] or n_buys >= room_for_new:
            break
        target_amount = min(total_equity * rules["max_position_pct"], buy_budget)
        quantity = _calc_quantity(s["code"], target_amount, price)
        amount = quantity * price
        if quantity <= 0 or amount < rules["min_trade_amount"]:
            continue  # 예산이 부족한 후보만 건너뛰고, 더 저렴한 다음 후보는 계속 시도한다
        reason = (
            f"예측 5일 수익률 {s['predicted_return_5d']:+.1%}, "
            f"방향적중률 {s['directional_accuracy']:.0%}, RSI {s['rsi14']:.0f} — 매수 조건 충족"
        )
        actions.append(TradeAction(action="buy", code=s["code"], quantity=quantity, reason=reason))
        buy_budget -= amount
        n_buys += 1

    return actions


def decide_trades_aggressive(
    signals: list[dict],
    holdings: pd.DataFrame,
    current_prices: dict[str, float],
    cash: float,
    rules: dict = AGGRESSIVE_RULES,
) -> list[TradeAction]:
    """공격적 봇의 매매 판단.

    청산(매도) 조건은 decide_trades()와 완전히 같은 4조건 구조를 그대로 쓴다 — rules만
    AGGRESSIVE_RULES로 다르게 참조할 뿐, "언제 파는가"의 판단 로직 자체는 다르지 않다.

    신규 매수 후보 산출 방식이 decide_trades()와 구조적으로 다른 지점: decide_trades()는
    이미 보유 중인 종목을 무조건 신규 매수 후보에서 제외하지만, 이 함수는 **이미 보유한
    종목도 추가 매수(피라미딩) 후보에 포함**한다 — 공격적 전략은 확신이 큰 포지션을
    키우는 것도 전략의 일부이기 때문이다. 다만 그 종목의 기존 평가액을 감안해 종목당
    최대 비중(max_position_pct)을 넘지 않는 만큼만 더 사고, 오늘 이미 매도한 종목만
    당일 재매수 후보에서 제외한다(휩쏘 방지, decide_trades()와 같은 취지).

    동시 보유 종목 수 한도(max_holdings)는 **신규 진입 종목 수**에만 적용된다 — 기존
    보유 종목을 추가 매수하는 건 종목 수를 늘리지 않으므로 이 한도에 안 걸린다
    (apply_risk_guardrail()의 기존 해석과 동일하게 맞춘 것).
    """
    signal_by_code = {s["code"]: s for s in signals}
    actions: list[TradeAction] = []

    # --- 청산: decide_trades()와 완전히 같은 4조건, rules만 AGGRESSIVE_RULES ---
    for _, row in holdings.iterrows():
        code = row["code"]
        price = current_prices.get(code)
        if price is None:
            continue

        pnl_pct = (price - row["avg_price"]) / row["avg_price"]
        signal = signal_by_code.get(code)
        reason = None

        if pnl_pct <= rules["stop_loss_pct"]:
            reason = f"평단가 대비 {pnl_pct:+.1%} — 손절 기준({rules['stop_loss_pct']:.0%}) 도달로 매도"
        elif pnl_pct >= rules["take_profit_pct"]:
            reason = f"평단가 대비 {pnl_pct:+.1%} — 익절 기준({rules['take_profit_pct']:.0%}) 도달로 매도"
        elif signal is not None and rules["exit_on_negative_signal"] and signal["predicted_return_5d"] < 0:
            reason = f"예측 수익률이 {signal['predicted_return_5d']:+.1%}로 음전환 — 신호 소멸로 매도"
        elif signal is not None and signal["rsi14"] >= rules["max_rsi_exit"]:
            reason = f"RSI {signal['rsi14']:.0f} — 과매수 구간 진입으로 매도"

        if reason is not None:
            actions.append(TradeAction(action="sell", code=code, quantity=row["quantity"], reason=reason))

    sold_today = {a.code for a in actions}
    holdings_after_sells = len(holdings) - len(actions)

    # 오늘 매도한 종목을 뺀 보유 평가액 — 피라미딩 시 "이 종목에 이미 얼마나 들어가
    # 있는지"를 알아야 종목당 최대 비중을 넘지 않게 추가 매수분을 계산할 수 있다.
    held_value_by_code = {
        row["code"]: row["quantity"] * current_prices.get(row["code"], row["avg_price"])
        for _, row in holdings.iterrows()
        if row["code"] not in sold_today
    }

    buy_candidates = []
    for s in signals:
        code = s["code"]
        if code in sold_today:
            continue  # 당일 매도한 종목만 재매수 후보에서 제외 — 그 외 보유종목은 후보 유지(피라미딩 허용)
        price = current_prices.get(code)
        if price is None:
            continue
        if s["predicted_return_5d"] < rules["buy_return_threshold"]:
            continue
        if s["directional_accuracy"] < rules["min_directional_accuracy"]:
            continue
        if s["rsi14"] > rules["max_rsi_entry"]:
            continue
        if s["news_sentiment"] < rules["min_news_sentiment"]:
            continue
        buy_candidates.append((s, price))

    buy_candidates.sort(key=lambda sp: sp[0]["predicted_return_5d"], reverse=True)

    holdings_value = sum(held_value_by_code.values())
    total_equity = cash + holdings_value
    buy_budget = max(cash - total_equity * rules["min_cash_reserve_pct"], 0.0)
    room_for_new = max(rules["max_holdings"] - holdings_after_sells, 0)

    n_buys = 0
    n_new_positions = 0
    for s, price in buy_candidates:
        code = s["code"]
        is_topup = code in held_value_by_code
        if n_buys >= rules["max_daily_trades"]:
            break
        if not is_topup and n_new_positions >= room_for_new:
            continue  # 신규 종목 한도 — 기존 보유 종목 추가매수는 이 한도에 안 걸린다

        existing_value = held_value_by_code.get(code, 0.0)
        position_cap = total_equity * rules["max_position_pct"]
        target_amount = min(max(position_cap - existing_value, 0.0), buy_budget)
        quantity = _calc_quantity(code, target_amount, price)
        amount = quantity * price
        if quantity <= 0 or amount < rules["min_trade_amount"]:
            continue  # 예산 부족한 후보만 건너뛰고 다음 후보를 계속 시도한다

        if is_topup:
            reason = (
                f"이미 보유 중 — 예측 5일 수익률 {s['predicted_return_5d']:+.1%}, "
                f"방향적중률 {s['directional_accuracy']:.0%}, RSI {s['rsi14']:.0f} — 추가 매수(피라미딩)"
            )
        else:
            reason = (
                f"예측 5일 수익률 {s['predicted_return_5d']:+.1%}, "
                f"방향적중률 {s['directional_accuracy']:.0%}, RSI {s['rsi14']:.0f} — 공격적 매수 조건 충족"
            )
        actions.append(TradeAction(action="buy", code=code, quantity=quantity, reason=reason))
        buy_budget -= amount
        held_value_by_code[code] = existing_value + amount
        n_buys += 1
        if not is_topup:
            n_new_positions += 1

    return actions


def decide_trades_momentum(
    signals: list[dict],
    holdings: pd.DataFrame,
    current_prices: dict[str, float],
    cash: float,
    rules: dict = MOMENTUM_RULES,
) -> list[TradeAction]:
    """모멘텀 봇의 매매 판단 — 진입의 1차 조건이 예측 수익률 "크기"가 아니라 **추세
    정렬**이다. 신호의 sma5_gap/sma20_gap(종가가 SMA5·SMA20 대비 몇 % 위/아래인지 —
    run_daily_trading.py의 _build_signal()이 indicators.sma()로 계산해 채운다)이 **둘
    다 양수**여야("정배열", 단기·중기 추세가 함께 상승 중) 매수 후보가 된다. 이 필드가
    신호에 없으면(구버전 신호 등) 추세를 확인할 수 없으므로 후보에서 제외한다.

    청산은 decide_trades()의 4조건(손절/익절/신호소멸/RSI과매수)에 **추세 이탈**
    (sma20_gap이 음전환 — 종가가 SMA20 아래로 내려감) 조건이 하나 더 붙는다. 추세를
    좇는 전략이라 추세가 꺾이면 예측 신호·RSI 값과 무관하게 바로 정리한다 — 그래서
    이 조건을 손절/익절 다음, 신호소멸/RSI과매수보다 먼저 확인한다.
    """
    signal_by_code = {s["code"]: s for s in signals}
    actions: list[TradeAction] = []

    for _, row in holdings.iterrows():
        code = row["code"]
        price = current_prices.get(code)
        if price is None:
            continue

        pnl_pct = (price - row["avg_price"]) / row["avg_price"]
        signal = signal_by_code.get(code)
        sma20_gap = signal.get("sma20_gap") if signal is not None else None
        reason = None

        if pnl_pct <= rules["stop_loss_pct"]:
            reason = f"평단가 대비 {pnl_pct:+.1%} — 손절 기준({rules['stop_loss_pct']:.0%}) 도달로 매도"
        elif pnl_pct >= rules["take_profit_pct"]:
            reason = f"평단가 대비 {pnl_pct:+.1%} — 익절 기준({rules['take_profit_pct']:.0%}) 도달로 매도"
        elif sma20_gap is not None and sma20_gap < 0:
            reason = f"종가가 SMA20 대비 {sma20_gap:+.1%} — 추세 이탈로 매도"
        elif signal is not None and rules["exit_on_negative_signal"] and signal["predicted_return_5d"] < 0:
            reason = f"예측 수익률이 {signal['predicted_return_5d']:+.1%}로 음전환 — 신호 소멸로 매도"
        elif signal is not None and signal["rsi14"] >= rules["max_rsi_exit"]:
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
        sma5_gap = s.get("sma5_gap")
        sma20_gap = s.get("sma20_gap")
        if sma5_gap is None or sma20_gap is None:
            continue  # 추세를 확인할 수 없으면 진입하지 않는다
        if not (sma5_gap > 0 and sma20_gap > 0):
            continue  # 정배열이 아니면 제외 — 예측 수익률이 아무리 높아도 추세가 우선
        if s["predicted_return_5d"] < rules["buy_return_threshold"]:
            continue
        if s["directional_accuracy"] < rules["min_directional_accuracy"]:
            continue
        if s["rsi14"] > rules["max_rsi_entry"]:
            continue
        if s["news_sentiment"] < rules["min_news_sentiment"]:
            continue
        buy_candidates.append((s, price))

    # 추세 강도(SMA5 괴리율) 내림차순 — 예측 수익률이 아니라 추세가 가장 강한 종목부터 채택
    buy_candidates.sort(key=lambda sp: sp[0]["sma5_gap"], reverse=True)

    holdings_value = sum(
        row["quantity"] * current_prices.get(row["code"], row["avg_price"]) for _, row in holdings.iterrows()
    )
    total_equity = cash + holdings_value
    buy_budget = max(cash - total_equity * rules["min_cash_reserve_pct"], 0.0)
    room_for_new = max(rules["max_holdings"] - holdings_after_sells, 0)

    n_buys = 0
    for s, price in buy_candidates:
        if n_buys >= rules["max_daily_trades"] or n_buys >= room_for_new:
            break
        target_amount = min(total_equity * rules["max_position_pct"], buy_budget)
        quantity = _calc_quantity(s["code"], target_amount, price)
        amount = quantity * price
        if quantity <= 0 or amount < rules["min_trade_amount"]:
            continue
        reason = (
            f"SMA5 {s['sma5_gap']:+.1%}·SMA20 {s['sma20_gap']:+.1%} 위 정배열, "
            f"예측 5일 수익률 {s['predicted_return_5d']:+.1%} — 추세 추종 매수 조건 충족"
        )
        actions.append(TradeAction(action="buy", code=s["code"], quantity=quantity, reason=reason))
        buy_budget -= amount
        n_buys += 1

    return actions


# ==================================================================== ③ 리스크 가드레일


def apply_risk_guardrail(
    actions: list[TradeAction],
    holdings: pd.DataFrame,
    cash: float,
    current_prices: dict[str, float],
    rules: dict = TRADING_RULES,
) -> tuple[list[TradeAction], list[dict]]:
    """decide_trades()가 제안한 액션을 현금·보유수량·리스크 한도로 다시 검증한다.

    한도를 넘는 액션은 수량을 줄이지 않고 **통째로 거부**한다. 이 가드레일은 "정상
    흐름에서 규칙을 못 지킨 제안이 나온다"는 걸 전제로 만든 게 아니라 decide_trades()의
    버그를 잡기 위한 두 번째 방어선이므로, 조용히 수량만 보정하면 버그가 있다는 신호
    자체가 묻힌다.

    매도부터 순서대로 검증해 현금·보유수량을 시뮬레이션한다(매도가 먼저 현금을
    풀어줘야 뒤따르는 매수 검증이 의미가 있다) — decide_trades()의 내부 순서와 같다.
    max_daily_trades는 신규 매수에만 적용한다 — decide_trades()와 반드시 같은 해석을
    써야 하므로 그쪽 docstring의 근거를 그대로 따른다.

    반환: (승인된 액션 목록, 거부된 액션 로그). 거부 로그는
    {"action", "code", "quantity", "reason", "reason_rejected"} 딕셔너리 리스트 — UI에서
    "오늘 거부된 제안"으로 보여줄 수 있게 원래 reason도 그대로 남긴다.
    """
    # int()로 캐스팅하면 코인처럼 소수 수량인 보유분이 0으로 잘려서(예: 0.5 BTC → 0),
    # 실제로 보유 중인데도 "보유 수량 초과" 매도 거부가 나는 버그가 된다 — 그대로 둔다.
    sim_holdings: dict[str, tuple[int | float, float]] = {
        row["code"]: (row["quantity"], float(row["avg_price"])) for _, row in holdings.iterrows()
    }
    sim_cash = cash
    approved: list[TradeAction] = []
    rejected: list[dict] = []

    sells = [a for a in actions if a.action == "sell"]
    buys = [a for a in actions if a.action == "buy"]

    for a in sells:
        held = sim_holdings.get(a.code)
        price = current_prices.get(a.code)
        if held is None or a.quantity > held[0]:
            rejected.append({**_action_dict(a), "reason_rejected": "보유 수량을 초과하는 매도 제안"})
            continue
        if price is None:
            rejected.append({**_action_dict(a), "reason_rejected": "체결 가격을 알 수 없음"})
            continue

        sim_cash += a.quantity * price
        remaining = held[0] - a.quantity
        if remaining == 0:
            del sim_holdings[a.code]
        else:
            sim_holdings[a.code] = (remaining, held[1])
        approved.append(a)

    n_buys_approved = 0
    for a in buys:
        price = current_prices.get(a.code)
        if price is None:
            rejected.append({**_action_dict(a), "reason_rejected": "체결 가격을 알 수 없음"})
            continue

        amount = a.quantity * price
        total_equity = sim_cash + sum(
            qty * current_prices.get(code, avg) for code, (qty, avg) in sim_holdings.items()
        )
        min_cash_reserve = total_equity * rules["min_cash_reserve_pct"]

        if n_buys_approved >= rules["max_daily_trades"]:
            rejected.append({**_action_dict(a), "reason_rejected": "하루 최대 거래 횟수 초과"})
            continue
        if amount < rules["min_trade_amount"]:
            rejected.append({**_action_dict(a), "reason_rejected": "최소 거래 금액 미달"})
            continue
        if a.code not in sim_holdings and len(sim_holdings) >= rules["max_holdings"]:
            rejected.append({**_action_dict(a), "reason_rejected": "동시 보유 종목 수 한도 초과"})
            continue
        if amount > sim_cash - min_cash_reserve:
            rejected.append({**_action_dict(a), "reason_rejected": "현금 부족(최소 현금 보유 비율 포함)"})
            continue

        existing_qty = sim_holdings.get(a.code, (0, 0.0))[0]
        position_value = amount + existing_qty * price
        if position_value > total_equity * rules["max_position_pct"]:
            rejected.append({**_action_dict(a), "reason_rejected": "종목당 최대 비중 초과"})
            continue

        sim_cash -= amount
        old_qty, old_avg = sim_holdings.get(a.code, (0, price))
        new_qty = old_qty + a.quantity
        sim_holdings[a.code] = (new_qty, (old_qty * old_avg + amount) / new_qty)
        n_buys_approved += 1
        approved.append(a)

    return approved, rejected


# ==================================================================== 봇 버전 레지스트리

# 모의투자에서 나란히 돌리는 봇들의 단일 소스 — run_daily_trading.py(실행)와
# pages/모의투자.py(조회)가 각자 하드코딩하지 않고 이 딕셔너리 하나만 순회한다. 봇을
# 추가/제거할 때 이 등록만 바꾸면 되게 하려는 목적(가드레일은 세 봇이 공유 —
# apply_risk_guardrail() 자체는 전략이 아니라 한도 검증이라 봇마다 다를 이유가 없다).
BOT_STRATEGIES = {
    "default": {"label": "기본형", "decide_trades": decide_trades, "rules": TRADING_RULES},
    "aggressive": {"label": "공격적", "decide_trades": decide_trades_aggressive, "rules": AGGRESSIVE_RULES},
    "momentum": {"label": "모멘텀", "decide_trades": decide_trades_momentum, "rules": MOMENTUM_RULES},
}
