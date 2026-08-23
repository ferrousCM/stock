"""모의투자 v2 매매 규칙 검증 — backtest-expert 방법론 적용 (PRD.md 11.7, 0단계).

**로컬 전용, 1회성 검증 도구.** `data/portfolio/`를 전혀 건드리지 않고, GitHub Actions에도
등록하지 않는다. v2 판단 함수(`src/trading_agent_v2.py`)를 짜기 *전에* 11.3에 제안된 초안
규칙(ATR 배수·리스크%·변동성 수축 임계값·모멘텀 버스트 임계값)이 과거 데이터에서 말이
되는 조합인지 먼저 확인하는 것이 목적이다 — v1이 "0단계는 코딩이 아니라 실측"을 지킨 것과
같은 이유.

**스코프**: v2에서 새로 추가되는 진입/청산 메커니즘(MA 정렬, ATR 손절·익절, 변동성 수축,
4% 모멘텀 버스트)만 검증한다. `predicted_return_5d`/`directional_accuracy`(predictor.py)와
`news_sentiment` 필터는 v1이 이미 매일 운영 중인 신호를 그대로 재사용하는 것이라(PRD 11.3)
이번 백테스트 대상이 아니다 — 특히 뉴스 감성은 과거 시점 히스토리를 조회할 방법 자체가
없어(CLAUDE.md "알려진 제약") 재현이 불가능하다. 이 필터들을 뺀 채로 검증하므로, 실제
운영 시 승률·기대값은 여기 수치보다 보수적으로(필터가 추가되니 진입 빈도는 줄고 품질은
올라갈 가능성) 나올 수 있다는 점을 감안해서 읽는다.

**룩어헤드 방지**: 진입 조건은 그날 t 종가까지의 데이터로만 판단하고, 그날 종가로 즉시
체결됐다고 가정한다(PRD 5.3 "체결가 가정"과 동일한 단순화 — 실제로는 하루 지연이 있지만
v1 전체가 이미 이 단순화를 표준으로 쓰고 있다). 이후 손절/익절은 진입일 다음 날부터
확인한다.

실행:
    .venv\\Scripts\\python.exe scripts\\backtest_v2_strategies.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import crypto_loader, crypto_screener, screener, us_screener
from src import data_loader as dl
from src import indicators as ind

# ----------------------------------------------------------------- 유니버스/기간

KR_UNIVERSE_SIZE = 20
US_UNIVERSE_SIZE = 15
COIN_UNIVERSE_SIZE = 15
LOOKBACK_YEARS = 2  # 종목당 히스토리 길이 — 짧을수록 빠르지만 표본이 줄어든다
MAX_HOLD_DAYS = 20  # 손절·익절 모두 안 맞으면 이 날짜 뒤엔 시간 청산
SLIPPAGE_PCT = 0.002  # backtest-expert "add friction" — 모든 체결가에 비관적으로 반영

# ----------------------------------------------------------------- 봇별 파라미터 (PRD 11.3 초안)


@dataclass(frozen=True)
class BotParams:
    label: str
    atr_multiplier: float
    take_profit_atr_multiplier: float | None  # None이면 시간 청산에만 의존(공격적 봇)
    max_rsi_entry: float
    risk_pct_per_trade: float  # PRD 11.3 TRADING_RULES_V2_* — 자산 대비 이 트레이드가 지는 리스크
    extra: dict = field(default_factory=dict)


DEFAULT_PARAMS = BotParams(
    label="기본형",
    atr_multiplier=2.0,
    take_profit_atr_multiplier=3.0,
    max_rsi_entry=70,
    risk_pct_per_trade=0.01,
)
AGGRESSIVE_PARAMS = BotParams(
    label="공격적",
    atr_multiplier=2.5,
    take_profit_atr_multiplier=None,
    max_rsi_entry=80,
    risk_pct_per_trade=0.02,
    extra={"contraction_lookback": 10, "contraction_ratio_max": 0.85, "pivot_proximity_pct": 0.05},
)
MOMENTUM_PARAMS = BotParams(
    label="모멘텀",
    atr_multiplier=1.5,
    take_profit_atr_multiplier=2.0,
    max_rsi_entry=85,
    risk_pct_per_trade=0.015,
    extra={"burst_threshold": 0.04, "volume_ratio_min": 1.0, "close_location_min": 0.6},
)


@dataclass
class Trade:
    bot: str
    code: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    exit_reason: str
    return_pct: float  # 가격 기준 등락률 — 셋업 품질 참고용(포지션 크기와 무관)
    equity_impact_pct: float  # 계좌 자산 대비 실제 손익 — position-sizer식 리스크 기반
    # 사이징(11.3)을 반영: 손절에 정확히 맞으면 -risk_pct_per_trade가 되도록 정규화한다.
    # MDD·기대값은 이 값으로 평가해야 "거래당 계좌의 몇 %가 걸렸는지"를 반영한 의미 있는
    # 수치가 된다 — return_pct(가격 등락률)를 그대로 트레이드마다 100% 복리하면 실제로는
    # 1~2%만 거는 전략인데도 MDD가 -90%대로 나오는 왜곡이 생긴다(최초 실행에서 실측).
    days_held: int


# ----------------------------------------------------------------- 유니버스 수집


def _build_universe() -> list[str]:
    """국내/해외/코인에서 거래대금 상위 종목코드를 모은다 — 워치리스트 규모와 같은
    수준(run_daily_trading.py의 WATCHLIST_SIZE류)으로 네트워크 호출을 작게 유지한다."""
    codes: list[str] = []

    try:
        kr = screener.screen(market="ALL", days_back=7)
        codes += list(screener.top_movers(kr, by="Amount", n=KR_UNIVERSE_SIZE)["Code"])
    except Exception as e:
        print(f"[경고] 국내 유니버스 조회 실패: {e}")

    try:
        us = us_screener.screen()
        codes += list(screener.top_movers(us, by="Amount", n=US_UNIVERSE_SIZE)["Code"])
    except Exception as e:
        print(f"[경고] 해외증시 유니버스 조회 실패: {e}")

    try:
        coin = crypto_screener.screen()
        codes += list(screener.top_movers(coin, by="Amount", n=COIN_UNIVERSE_SIZE)["Code"])
    except Exception as e:
        print(f"[경고] 코인 유니버스 조회 실패: {e}")

    return codes


def _infer_market(code: str) -> str:
    if code.startswith("KRW-"):
        return "COIN"
    if code.isdigit() and len(code) == 6:
        return "KR"
    return "US"


def _fetch_history(code: str) -> pd.DataFrame:
    start = (pd.Timestamp.today() - pd.DateOffset(years=LOOKBACK_YEARS)).strftime("%Y-%m-%d")
    if _infer_market(code) == "COIN":
        return crypto_loader.get_price(code, start=start)
    return dl.get_price(code, start=start)


# ----------------------------------------------------------------- 지표 준비


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """진입 조건 판단에 필요한 컬럼을 전부 붙인다. 전부 그날 t까지의 데이터로만 계산되는
    rolling/shift 연산이라 룩어헤드가 없다."""
    out = df.copy()
    close, high, low, volume = out["Close"], out["High"], out["Low"], out["Volume"]

    out["atr14"] = ind.atr(out, 14)
    out["sma20"] = ind.sma(close, 20)
    out["sma60"] = ind.sma(close, 60)
    out["rsi14"] = ind.rsi(close, 14)

    # 모멘텀 버스트 필드
    out["day_change"] = close.pct_change(1)
    out["day_range"] = high - low
    out["prev_range_max3"] = out["day_range"].shift(1).rolling(3).max()
    out["volume_ratio_1d"] = volume / volume.shift(1)
    out["close_location"] = (close - low) / (high - low)

    # 공격적(변동성 수축) 필드
    out["atr_contraction_ratio"] = out["atr14"] / out["atr14"].shift(
        AGGRESSIVE_PARAMS.extra["contraction_lookback"]
    )
    out["dist_from_60d_high"] = close / high.rolling(60).max() - 1

    return out


# ----------------------------------------------------------------- 진입 조건


def _entry_default(row) -> bool:
    if pd.isna(row["sma20"]) or pd.isna(row["sma60"]) or pd.isna(row["rsi14"]):
        return False
    ma_aligned = row["Close"] > row["sma20"] > row["sma60"]
    return bool(ma_aligned and row["rsi14"] <= DEFAULT_PARAMS.max_rsi_entry)


def _entry_aggressive(row) -> bool:
    if pd.isna(row["atr_contraction_ratio"]) or pd.isna(row["dist_from_60d_high"]) or pd.isna(row["rsi14"]):
        return False
    contracted = row["atr_contraction_ratio"] <= AGGRESSIVE_PARAMS.extra["contraction_ratio_max"]
    near_high = row["dist_from_60d_high"] >= -AGGRESSIVE_PARAMS.extra["pivot_proximity_pct"]
    return bool(contracted and near_high and row["rsi14"] <= AGGRESSIVE_PARAMS.max_rsi_entry)


def _entry_momentum(row, burst_threshold: float) -> bool:
    if pd.isna(row["day_change"]) or pd.isna(row["volume_ratio_1d"]) or pd.isna(row["prev_range_max3"]):
        return False
    burst = row["day_change"] >= burst_threshold
    vol_ok = row["volume_ratio_1d"] >= MOMENTUM_PARAMS.extra["volume_ratio_min"]
    range_ok = row["day_range"] > row["prev_range_max3"]
    loc_ok = row["close_location"] >= MOMENTUM_PARAMS.extra["close_location_min"]
    return bool(burst and vol_ok and range_ok and loc_ok)


# ----------------------------------------------------------------- 트레이드 시뮬레이션


def _simulate_trade(df: pd.DataFrame, entry_idx: int, code: str, bot: str, params: BotParams) -> Trade | None:
    entry_row = df.iloc[entry_idx]
    entry_price = float(entry_row["Close"])
    atr_at_entry = float(entry_row["atr14"])
    if pd.isna(atr_at_entry) or atr_at_entry <= 0:
        return None
    if entry_idx >= len(df) - 1:
        # 이 종목 히스토리의 마지막 날 진입은 다음 날 데이터가 없어 시뮬레이션할 구간이
        # 없다 — 이걸 그대로 진행하면 last_idx == entry_idx가 되어 days_held=0인 트레이드가
        # 나오고, 호출부(_run_bot_backtest)의 인덱스가 전혀 전진하지 못해 같은 지점에서
        # 동일한 트레이드를 무한히 반복 생성한다(실측: 메모리 2GB대까지 폭주하며 응답 없음).
        return None

    stop_price = entry_price - atr_at_entry * params.atr_multiplier
    target_price = (
        entry_price + atr_at_entry * params.take_profit_atr_multiplier
        if params.take_profit_atr_multiplier
        else None
    )
    # 손절폭(진입가 대비 %)으로 정규화 — position-sizer식 리스크 기반 사이징에서는
    # "손절에 맞으면 정확히 risk_pct_per_trade만큼 잃는다"는 것이 사이징 공식의 정의 그
    # 자체이므로(PRD 11.3), 가격 등락률을 이 폭으로 나눠 risk_pct_per_trade 단위로 환산하면
    # "실제 계좌에 몇 % 영향을 줬는지"가 된다.
    stop_distance_pct = (atr_at_entry * params.atr_multiplier) / entry_price

    last_idx = min(entry_idx + MAX_HOLD_DAYS, len(df) - 1)
    for i in range(entry_idx + 1, last_idx + 1):
        day = df.iloc[i]
        if day["Low"] <= stop_price:
            exit_price = stop_price * (1 - SLIPPAGE_PCT)  # 비관적 체결(worst-case fill)
            return _make_trade(
                bot,
                code,
                df.index[entry_idx],
                df.index[i],
                entry_price,
                exit_price,
                "stop",
                i - entry_idx,
                stop_distance_pct,
                params,
            )
        if target_price is not None and day["High"] >= target_price:
            exit_price = target_price * (1 - SLIPPAGE_PCT)
            return _make_trade(
                bot,
                code,
                df.index[entry_idx],
                df.index[i],
                entry_price,
                exit_price,
                "target",
                i - entry_idx,
                stop_distance_pct,
                params,
            )

    # 시간 청산 — 손절/익절 둘 다 안 맞고 보유기간이 다 참
    exit_price = float(df.iloc[last_idx]["Close"]) * (1 - SLIPPAGE_PCT)
    return _make_trade(
        bot,
        code,
        df.index[entry_idx],
        df.index[last_idx],
        entry_price,
        exit_price,
        "time",
        last_idx - entry_idx,
        stop_distance_pct,
        params,
    )


def _make_trade(
    bot,
    code,
    entry_date,
    exit_date,
    entry_price,
    exit_price,
    reason,
    days_held,
    stop_distance_pct,
    params: BotParams,
) -> Trade:
    return_pct = (exit_price - entry_price) / entry_price
    equity_impact_pct = params.risk_pct_per_trade * (return_pct / stop_distance_pct)
    return Trade(
        bot=bot,
        code=code,
        entry_date=entry_date,
        exit_date=exit_date,
        entry_price=entry_price,
        exit_price=exit_price,
        exit_reason=reason,
        return_pct=return_pct,
        equity_impact_pct=equity_impact_pct,
        days_held=days_held,
    )


def _run_bot_backtest(
    histories: dict[str, pd.DataFrame], bot_id: str, params: BotParams, burst_threshold: float | None = None
) -> list[Trade]:
    trades: list[Trade] = []
    for code, df in histories.items():
        i = 60  # SMA60/60일 고점 워밍업 이후부터 시작
        in_position = False
        while i < len(df):
            if not in_position:
                row = df.iloc[i]
                triggered = (
                    _entry_default(row)
                    if bot_id == "default"
                    else _entry_aggressive(row)
                    if bot_id == "aggressive"
                    else _entry_momentum(row, burst_threshold or MOMENTUM_PARAMS.extra["burst_threshold"])
                )
                if triggered:
                    trade = _simulate_trade(df, i, code, params.label, params)
                    if trade is not None:
                        trades.append(trade)
                        # max(..., 1)로 항상 최소 1은 전진시킨다 — days_held가 0인 트레이드가
                        # 생기면(이론상 _simulate_trade가 막아주지만, 이중 방어선) i가 멈춰
                        # 같은 지점에서 무한히 트레이드를 찍어내는 사고(실측 재현됨)를 방지한다.
                        i += max(trade.days_held, 1)  # 같은 종목 중복(겹치는) 트레이드 방지
                        in_position = False
                        continue
            i += 1
    return trades


# ----------------------------------------------------------------- 평가 (backtest-expert 5축)


def _evaluate(trades: list[Trade], label: str) -> None:
    n = len(trades)
    print(f"\n{'=' * 60}\n[{label}] 트레이드 {n}건")
    if n == 0:
        print("  트레이드가 0건 — 진입 조건이 이 유니버스/기간에서 전혀 안 걸렸습니다.")
        print("  판정: ❌ Abandon (표본 없음) — 임계값을 완화하거나 유니버스를 넓혀야 합니다.")
        return

    # 가격 등락률(setup 자체의 품질) — 포지션 크기와 무관하게 참고용으로만 본다
    price_returns = np.array([t.return_pct for t in trades])
    price_win_rate = (price_returns > 0).mean()

    # 계좌 영향(equity_impact_pct) — 이게 실제 판정 기준이다. position-sizer식 리스크
    # 사이징을 반영해 손절에 맞으면 정확히 -risk_pct_per_trade가 되도록 정규화한 값이므로,
    # 이 값으로 MDD·기대값을 계산해야 "이 규칙으로 실제 거래했다면 계좌가 얼마나
    # 흔들렸을지"를 뜻한다(가격 등락률을 그대로 100% 복리하면 실제로는 1~2%만 거는
    # 전략인데도 MDD가 -90%대로 왜곡되는 문제를 최초 실행에서 실측했다).
    impacts = np.array([t.equity_impact_pct for t in trades])
    wins = impacts[impacts > 0]
    losses = impacts[impacts <= 0]
    win_rate = len(wins) / n
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    # 진입일 순으로 정렬해 "한 번에 한 포지션만 보유"를 가정한 복리 곡선으로 MDD를 근사한다
    # — 실제로는 최대 10~15종목을 동시에 들 수 있어(가드레일) 리스크가 그만큼 분산되므로,
    # 이 근사는 대체로 보수적(실제보다 나쁘게) 방향이다. 다만 여러 포지션이 동시에 손절나는
    # 상관 리스크(예: 시장 급락)까지는 반영하지 못한다는 한계는 남는다.
    ordered = sorted(trades, key=lambda t: t.entry_date)
    equity = np.cumprod([1 + t.equity_impact_pct for t in ordered])
    mdd = float((equity / np.maximum.accumulate(equity) - 1).min())

    exit_reason_counts = pd.Series([t.exit_reason for t in trades]).value_counts().to_dict()

    print(f"  표본 수         : {n}건 ({'충분(≥30)' if n >= 30 else '부족(<30, 참고용)'})")
    print(f"  승률(가격 기준)  : {price_win_rate:.1%}  ※ 셋업 자체 품질 참고용")
    print(f"  승률(계좌 기준)  : {win_rate:.1%}")
    print(f"  평균 이익(승)    : {avg_win:+.2%}  (계좌 대비)")
    print(f"  평균 손실(패)    : {avg_loss:+.2%}  (계좌 대비)")
    print(f"  기대값(건당)     : {expectancy:+.2%}  (계좌 대비)")
    print(f"  누적 MDD(근사)   : {mdd:+.2%}  (한 번에 한 포지션 가정 — 실제 분산 시 더 완만할 가능성)")
    print(f"  청산 사유 분포   : {exit_reason_counts}")

    sample_ok = n >= 30
    expectancy_ok = expectancy > 0
    if sample_ok and expectancy_ok and mdd > -0.30:
        verdict = "✅ Deploy — 표본 충분, 기대값 양수, MDD 허용 범위"
    elif expectancy_ok:
        verdict = "🔄 Refine — 방향은 맞지만 표본 부족 또는 MDD 과도, 파라미터/유니버스 조정 검토"
    else:
        verdict = "❌ Abandon — 기대값이 음수, 이 형태의 규칙으로는 승산 없음"
    print(f"  판정            : {verdict}")


def _sensitivity_sweep(
    histories: dict[str, pd.DataFrame], bot_id: str, params: BotParams, sweep_name: str, values: list[float]
) -> None:
    print(f"\n[{params.label}] 파라미터 민감도 — {sweep_name}")
    for v in values:
        if bot_id == "momentum":
            trades = _run_bot_backtest(histories, bot_id, params, burst_threshold=v)
        else:
            swept = BotParams(
                label=params.label,
                atr_multiplier=v,
                take_profit_atr_multiplier=params.take_profit_atr_multiplier,
                max_rsi_entry=params.max_rsi_entry,
                risk_pct_per_trade=params.risk_pct_per_trade,
                extra=params.extra,
            )
            trades = _run_bot_backtest(histories, bot_id, swept)
        n = len(trades)
        if n == 0:
            print(f"  {sweep_name}={v}: 트레이드 0건")
            continue
        impacts = np.array([t.equity_impact_pct for t in trades])
        win_rate = (impacts > 0).mean()
        expectancy = impacts.mean()
        print(f"  {sweep_name}={v}: n={n}, 승률(계좌기준)={win_rate:.1%}, 기대값(계좌기준)={expectancy:+.2%}")


def main() -> None:
    print("모의투자 v2 백테스트 검증 — PRD.md 11.7 (backtest-expert 방법론)\n")
    print("유니버스 수집 중 (국내+해외+코인 거래대금 상위)...")
    codes = _build_universe()
    print(f"유니버스 {len(codes)}종목: {codes}\n")

    print(f"종목별 과거 {LOOKBACK_YEARS}년 히스토리 조회 및 지표 계산 중...")
    histories: dict[str, pd.DataFrame] = {}
    for code in codes:
        try:
            raw = _fetch_history(code)
            if len(raw) < 90:  # SMA60/60일 고점 워밍업에 못 미치면 스킵
                continue
            histories[code] = _prepare(raw)
        except Exception as e:
            print(f"  [경고] {code} 히스토리 조회 실패: {e}")
    print(f"유효 히스토리 {len(histories)}종목 확보\n")

    default_trades = _run_bot_backtest(histories, "default", DEFAULT_PARAMS)
    aggressive_trades = _run_bot_backtest(histories, "aggressive", AGGRESSIVE_PARAMS)
    momentum_trades = _run_bot_backtest(histories, "momentum", MOMENTUM_PARAMS)

    _evaluate(default_trades, "기본형")
    _evaluate(aggressive_trades, "공격적")
    _evaluate(momentum_trades, "모멘텀")

    _sensitivity_sweep(histories, "default", DEFAULT_PARAMS, "atr_multiplier", [1.5, 2.0, 2.5, 3.0])
    _sensitivity_sweep(histories, "aggressive", AGGRESSIVE_PARAMS, "atr_multiplier", [1.5, 2.0, 2.5, 3.0])
    _sensitivity_sweep(histories, "momentum", MOMENTUM_PARAMS, "burst_threshold", [0.03, 0.04, 0.05])

    print(f"\n{'=' * 60}")
    print("검증 종료 — 위 판정을 참고해 PRD.md 11.3의 TRADING_RULES_V2_* 초안 수치를")
    print("확정하거나 조정한 뒤 1단계(src/trading_agent_v2.py 구현)로 진행한다.")


if __name__ == "__main__":
    main()
