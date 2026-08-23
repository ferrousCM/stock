"""모의투자 일일 매매 — GitHub Actions가 매 평일 19:30 KST에 실행하는 단일 진입점.

PRD.md 5.5·10장 3·8단계 참고. ①(가격예측 신호 생성) → ②(매매 규칙 엔진) → ③(리스크
가드레일) → ④(포트폴리오 원장 반영)를 순서대로 묶는다. git 커밋은 이 스크립트가 아니라
`.github/workflows/daily_trading.yml`이 한다 — 여기는 `data/portfolio/` 로컬 파일까지만
책임진다.

**봇 다중화(8단계)**: "기본형"(기존 그대로) 외에 "공격적"·"모멘텀" 두 봇을 같은 날부터
나란히 돌려 성과를 비교한다. 세 봇은 `trading_agent.BOT_STRATEGIES`(단일 레지스트리)에
등록돼 있고, 봇마다 원장 디렉터리가 분리된다(`config.portfolio_dir_for(bot_id)` —
"기본형"은 기존 `data/portfolio/` 그대로, 나머지는 `data/portfolio/{bot_id}/`). 시세
스냅샷·예측 신호·뉴스 감성은 계산 비용이 커서(종목별 학습+뉴스 조회) 세 봇이 하루 한 번
계산한 결과를 공유하고, 그 신호를 "무엇을 살지" 판단하는 함수(`decide_trades`/
`_aggressive`/`_momentum`)만 봇마다 다르다. 가드레일(`apply_risk_guardrail`)은 전략이
아니라 한도 검증이라 세 봇이 공유한다.

국내(KOSPI/KOSDAQ)·해외증시(나스닥)·코인(업비트) 세 시장을 함께 매매한다. 원장은 단일
원화(KRW) 기준이라, 해외증시 시세만 그날 환율로 원화 환산해서 이후 파이프라인(현금·
포지션 한도 등)에 넣는다 — 코인은 업비트 자체가 원화 시세라 환산이 필요 없다. 어느
종목이 어느 시장인지는 원장 스키마에 컬럼을 두지 않고 `trading_agent.infer_market()`이
종목코드 패턴만으로 판별한다(모의투자·모니터링 두 화면이 같은 방식을 쓴다).

로컬 실행: .venv\\Scripts\\python.exe scripts\\run_daily_trading.py [--dry-run]
--dry-run은 원장 파일을 건드리지 않고 판단 결과만 출력한다(디버깅용, PRD 5.5
"로컬 수동 실행과의 충돌" 참고 — 원장을 실제로 갱신하는 실행은 GitHub Actions로 일원화한다).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import (
    config,
    crypto_loader,
    crypto_news,
    crypto_screener,
    news,
    portfolio,
    predictor,
    screener,
    us_screener,
)
from src import data_loader as dl
from src import indicators as ind
from src.trading_agent import BOT_STRATEGIES, apply_risk_guardrail, infer_market
from src.trading_agent_v2 import BOT_STRATEGIES_V2, TRADING_RULES_V2_AGGRESSIVE

# v2 공격적 봇의 변동성 수축 판정(atr_contraction_ratio)이 참조하는 비교 구간 — 이 상수
# 하나만 trading_agent_v2.py가 정의하고 여기서 그대로 읽는다(같은 값을 두 곳에 따로
# 하드코딩하면 반드시 어긋난다).
_V2_CONTRACTION_LOOKBACK = TRADING_RULES_V2_AGGRESSIVE["contraction_lookback"]

WATCHLIST_SIZE = 40  # PRD 5.1 — 종목별 개별 학습이 필요해 전종목이 아니라 워치리스트로 제한
# 해외증시·코인은 이번에 새로 추가된 시장이라 국내보다 작은 워치리스트로 시작한다 —
# 종목당 예측 학습 + 뉴스 조회 비용이 그대로 곱해지므로, 안정성을 지켜본 뒤 필요하면
# 나중에 키운다(값 하나만 바꾸면 됨, 구조 변경 불필요).
US_WATCHLIST_SIZE = 15
COIN_WATCHLIST_SIZE = 15
PREDICT_HORIZON = 5  # 매매 규칙(TRADING_RULES)이 5거래일 예측 수익률 기준이므로 고정


def _build_signal(code: str, name: str) -> dict | None:
    """종목 하나의 매매 신호(예측 5일 수익률·RSI·뉴스감성·방향적중률·추세정렬)를 만든다.

    `infer_market(code)`로 시장을 판별해 가격 이력·뉴스 소스를 그 시장에 맞는 걸로
    가져온다 — 예측 수익률(%)·RSI·방향적중률은 전부 비율/등급이라 통화와 무관하므로
    여기서는 환율 환산이 필요 없다(환산은 run()이 current_prices를 만들 때 한 번만 한다).

    sma5_gap/sma20_gap(종가가 SMA5·SMA20 대비 몇 % 위/아래인지)은 모멘텀 봇
    (`trading_agent.decide_trades_momentum`) 전용 필드다 — 이미 받은 price_df에서
    `indicators.sma()`로 바로 계산하므로 추가 네트워크 호출이 없다. 기본형·공격적 봇은
    이 필드를 보지 않는다.

    **v2 전용 필드**(PRD.md 11.2·`trading_agent_v2.py` 모듈 docstring 참고) — 이 신호
    dict 하나를 v1·v2 봇이 그대로 공유하므로(예측·뉴스 조회가 비싸 재계산하지 않는다),
    같은 price_df에서 v2가 쓸 필드까지 한 번에 계산해 얹는다(추가 네트워크 호출 없음):
    atr14, sma20/sma60(gap 아닌 원래 가격 단위 — v2 기본형의 MA 정렬 판정용),
    atr_contraction_ratio·dist_from_60d_high(v2 공격적), day_change·volume_ratio_1d·
    day_range·prev_range_max3·close_location·low5(v2 모멘텀). v1은 이 필드들을 보지 않는다.

    가격 이력 부족·예측 실패 등으로 신호를 못 만들면 None — 호출부가 그 종목을
    이번 판단에서 스킵한다(워치리스트 종목이면 매수 후보에서 빠지고, 보유 종목이면
    가격 기반 손절/익절만 평가된다 — decide_trades() docstring 참고).
    """
    market = infer_market(code)
    price_df = crypto_loader.get_price(code) if market == "COIN" else dl.get_price(code)
    if price_df.empty:
        return None

    # app.py와 동일한 순서: 오늘자 뉴스를 먼저 히스토리에 기록한 뒤, 그 히스토리를
    # 학습 피처로 읽는다 (CLAUDE.md 규칙 — log_daily_sentiment 직접 호출 금지, 반드시
    # log_sentiment_from_news를 거쳐 긍정/부정/기사 수 피처까지 채운다).
    if market in ("COIN", "US"):
        # 코인·해외증시는 종목코드 기반 뉴스 소스가 없어(pages/모니터링.py와 동일한 사정)
        # 종목명 키워드 검색(crypto_news)을 공유 재사용한다.
        news_df = crypto_news.fetch_news_with_sentiment(name, n=10)
    else:
        news_df = news.fetch_news_with_sentiment(code, n=10)
    news.log_sentiment_from_news(code, news_df)
    sentiment_hist = news.sentiment_history(code)

    result = predictor.train_and_predict(price_df, horizon=PREDICT_HORIZON, sentiment_hist=sentiment_hist)
    if "error" in result:
        return None

    rsi14 = ind.add_all(price_df)["rsi14"].iloc[-1]
    if pd.isna(rsi14):
        return None

    news_sentiment = float(news_df["sentiment_score"].mean()) if not news_df.empty else 0.0

    close, high, low, volume = price_df["Close"], price_df["High"], price_df["Low"], price_df["Volume"]
    last_close = float(close.iloc[-1])
    sma5 = ind.sma(close, 5).iloc[-1]
    sma20 = ind.sma(close, 20).iloc[-1]
    sma60 = ind.sma(close, 60).iloc[-1]
    sma5_gap = float(last_close / sma5 - 1) if pd.notna(sma5) else None
    sma20_gap = float(last_close / sma20 - 1) if pd.notna(sma20) else None

    # ---- v2 전용 필드 (PRD 11.2) — 이미 들고 있는 price_df에서 계산, 신규 네트워크 호출 없음
    atr_series = ind.atr(price_df, 14)
    atr14 = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else None

    atr_contraction_ratio = None
    if atr14 is not None and len(atr_series) > _V2_CONTRACTION_LOOKBACK:
        atr_past = atr_series.iloc[-1 - _V2_CONTRACTION_LOOKBACK]
        if pd.notna(atr_past) and atr_past > 0:
            atr_contraction_ratio = float(atr14 / atr_past)

    dist_from_60d_high = None
    if len(high) >= 60:
        high60 = high.rolling(60).max().iloc[-1]
        if pd.notna(high60) and high60 > 0:
            dist_from_60d_high = float(last_close / high60 - 1)

    day_change = float(close.pct_change(1).iloc[-1]) if len(close) >= 2 and pd.notna(close.iloc[-2]) else None

    volume_ratio_1d = None
    if len(volume) >= 2 and pd.notna(volume.iloc[-2]) and volume.iloc[-2] > 0:
        volume_ratio_1d = float(volume.iloc[-1] / volume.iloc[-2])

    day_range_series = high - low
    day_range = float(day_range_series.iloc[-1])
    prev_range_max3 = None
    prev_max3 = day_range_series.shift(1).rolling(3).max().iloc[-1]
    if pd.notna(prev_max3):
        prev_range_max3 = float(prev_max3)

    close_location = None
    day_high, day_low = float(high.iloc[-1]), float(low.iloc[-1])
    if day_high > day_low:
        close_location = float((last_close - day_low) / (day_high - day_low))

    low5 = float(low.tail(5).min())

    return {
        "code": code,
        "name": name,
        "predicted_return_5d": result["predicted_return"],
        "rsi14": float(rsi14),
        "news_sentiment": news_sentiment,
        "directional_accuracy": result["directional_accuracy"],
        "sma5_gap": sma5_gap,
        "sma20_gap": sma20_gap,
        "atr14": atr14,
        "sma20": float(sma20) if pd.notna(sma20) else None,
        "sma60": float(sma60) if pd.notna(sma60) else None,
        "atr_contraction_ratio": atr_contraction_ratio,
        "dist_from_60d_high": dist_from_60d_high,
        "day_change": day_change,
        "volume_ratio_1d": volume_ratio_1d,
        "day_range": day_range,
        "prev_range_max3": prev_range_max3,
        "close_location": close_location,
        "low5": low5,
    }


def _run_bot(
    bot_id: str,
    today_str: str,
    holdings: pd.DataFrame,
    cash: float,
    signals: list[dict],
    current_prices: dict[str, float],
    name_by_code: dict[str, str],
    held_names: dict[str, str],
    kospi_close: float | None,
    dry_run: bool,
) -> None:
    """봇 하나(BOT_STRATEGIES 항목 하나)의 판단→가드레일→체결→시가평가→원장 반영을
    끝까지 수행한다. 원장이 봇마다 분리돼 있어 이 호출 하나가 완결된 단위다 — run()이
    봇별로 이 함수를 try/except로 감싸서 한 봇의 실패가 다른 봇에 번지지 않게 한다."""
    meta = BOT_STRATEGIES[bot_id]
    label = meta["label"]
    rules = meta["rules"]
    portfolio_dir = config.portfolio_dir_for(bot_id)

    actions = meta["decide_trades"](signals, holdings, current_prices, cash, rules=rules)
    approved, rejected = apply_risk_guardrail(actions, holdings, cash, current_prices, rules=rules)

    for r in rejected:
        print(f"  [{label}] 거부: {r['action']} {r['code']} x{r['quantity']} — {r['reason_rejected']}")

    new_trades = []
    for a in approved:
        price = current_prices[a.code]
        name = name_by_code.get(a.code, held_names.get(a.code, a.code))
        holdings, cash, trade = portfolio.apply_trade(
            holdings,
            cash,
            date=today_str,
            code=a.code,
            name=name,
            action=a.action,
            quantity=a.quantity,
            price=price,
            reason=a.reason,
        )
        new_trades.append(trade)
        unit_label = "개" if infer_market(a.code) == "COIN" else "주"
        print(
            f"  [{label}] 체결: {a.action} {a.code}({name}) {a.quantity}{unit_label} @ {price:,.0f}원 — {a.reason}"
        )

    holdings_mtm, holdings_value = portfolio.mark_to_market(holdings, current_prices)
    stale = holdings_mtm[holdings_mtm["price_is_stale"]] if not holdings_mtm.empty else holdings_mtm
    for _, row in stale.iterrows():
        print(f"  [{label}] 주의: {row['code']}({row['name']}) 오늘 종가를 못 가져와 평단가로 대체 평가")

    total_equity = cash + holdings_value
    equity_row = {
        "date": today_str,
        "cash": cash,
        "holdings_value": holdings_value,
        "total_equity": total_equity,
        "kospi_close": kospi_close,
    }

    n_buy = sum(1 for a in approved if a.action == "buy")
    n_sell = sum(1 for a in approved if a.action == "sell")
    print(
        f"[{today_str}] [{label}] 매수 {n_buy}건, 매도 {n_sell}건. "
        f"총자산 {total_equity:,.0f}원 (현금 {cash:,.0f} + 평가금액 {holdings_value:,.0f})"
    )

    if dry_run:
        print(f"[dry-run] [{label}] 원장을 갱신하지 않았습니다.")
        return

    portfolio.save_daily_result(
        today_str, cash, holdings, new_trades, equity_row, portfolio_dir=portfolio_dir
    )
    print(f"[{today_str}] [{label}] 원장 갱신 완료.")


def _run_bot_v2(
    bot_id: str,
    today_str: str,
    holdings: pd.DataFrame,
    cash: float,
    signals: list[dict],
    current_prices: dict[str, float],
    name_by_code: dict[str, str],
    held_names: dict[str, str],
    kospi_close: float | None,
    dry_run: bool,
) -> None:
    """v2 봇(BOT_STRATEGIES_V2 항목 하나)의 판단→가드레일→체결→시가평가→원장 반영.

    `_run_bot()`과 구조가 거의 같다(신호·가드레일·원장 계층을 v1과 그대로 공유하므로) —
    `config.portfolio_dir_for_v2()`를 쓴다는 점만 다르다. v1 라이브 코드 경로를 절대
    건드리지 않는다는 원칙(PRD 11.5)에 따라 `_run_bot()`을 일반화하지 않고 별도 함수로
    둔다. 로그에는 v1과 같은 봇 라벨("기본형" 등)이 v1 쪽에도 찍히므로 "v2:" 접두어로
    구분한다(원장 자체는 완전히 분리돼 있어 실제 데이터가 섞일 위험은 없다).
    """
    meta = BOT_STRATEGIES_V2[bot_id]
    label = f"v2:{meta['label']}"
    rules = meta["rules"]
    portfolio_dir = config.portfolio_dir_for_v2(bot_id)

    actions = meta["decide_trades"](signals, holdings, current_prices, cash, rules=rules)
    approved, rejected = apply_risk_guardrail(actions, holdings, cash, current_prices, rules=rules)

    for r in rejected:
        print(f"  [{label}] 거부: {r['action']} {r['code']} x{r['quantity']} — {r['reason_rejected']}")

    new_trades = []
    for a in approved:
        price = current_prices[a.code]
        name = name_by_code.get(a.code, held_names.get(a.code, a.code))
        holdings, cash, trade = portfolio.apply_trade(
            holdings,
            cash,
            date=today_str,
            code=a.code,
            name=name,
            action=a.action,
            quantity=a.quantity,
            price=price,
            reason=a.reason,
        )
        new_trades.append(trade)
        unit_label = "개" if infer_market(a.code) == "COIN" else "주"
        print(
            f"  [{label}] 체결: {a.action} {a.code}({name}) {a.quantity}{unit_label} @ {price:,.0f}원 — {a.reason}"
        )

    holdings_mtm, holdings_value = portfolio.mark_to_market(holdings, current_prices)
    stale = holdings_mtm[holdings_mtm["price_is_stale"]] if not holdings_mtm.empty else holdings_mtm
    for _, row in stale.iterrows():
        print(f"  [{label}] 주의: {row['code']}({row['name']}) 오늘 종가를 못 가져와 평단가로 대체 평가")

    total_equity = cash + holdings_value
    equity_row = {
        "date": today_str,
        "cash": cash,
        "holdings_value": holdings_value,
        "total_equity": total_equity,
        "kospi_close": kospi_close,
    }

    n_buy = sum(1 for a in approved if a.action == "buy")
    n_sell = sum(1 for a in approved if a.action == "sell")
    print(
        f"[{today_str}] [{label}] 매수 {n_buy}건, 매도 {n_sell}건. "
        f"총자산 {total_equity:,.0f}원 (현금 {cash:,.0f} + 평가금액 {holdings_value:,.0f})"
    )

    if dry_run:
        print(f"[dry-run] [{label}] 원장을 갱신하지 않았습니다.")
        return

    portfolio.save_daily_result(
        today_str, cash, holdings, new_trades, equity_row, portfolio_dir=portfolio_dir
    )
    print(f"[{today_str}] [{label}] 원장 갱신 완료.")


def run(dry_run: bool = False) -> None:
    today = pd.Timestamp.now(tz="Asia/Seoul").normalize().tz_localize(None)
    today_str = today.strftime("%Y-%m-%d")

    # 봇마다 원장이 분리돼 있어(config.portfolio_dir_for) last_run_date도 봇별로 다를 수
    # 있다 — 전부 오늘 이미 돌았으면 시세/신호 계산(비용이 큼) 자체를 생략한다. v2 봇
    # (PRD 11장)도 같은 원칙으로 별도 원장·별도 pending 목록을 갖되, 신호 계산은 v1과
    # 공유하므로 "v1+v2 전부 오늘 이미 돌았을 때만" 조기 종료한다.
    bot_states = {bot_id: portfolio.get_state(config.portfolio_dir_for(bot_id)) for bot_id in BOT_STRATEGIES}
    pending_bots = [bot_id for bot_id, s in bot_states.items() if s["last_run_date"] != today_str]
    bot_states_v2 = {
        bot_id: portfolio.get_state(config.portfolio_dir_for_v2(bot_id)) for bot_id in BOT_STRATEGIES_V2
    }
    pending_bots_v2 = [bot_id for bot_id, s in bot_states_v2.items() if s["last_run_date"] != today_str]

    if not pending_bots and not pending_bots_v2:
        print(f"[{today_str}] 모든 봇(v1+v2)이 오늘 이미 실행했습니다 — 종료.")
        return
    if pending_bots and len(pending_bots) < len(BOT_STRATEGIES):
        already = [BOT_STRATEGIES[b]["label"] for b in BOT_STRATEGIES if b not in pending_bots]
        print(f"[{today_str}] {', '.join(already)} 봇은 오늘 이미 실행했습니다 — 나머지만 진행.")
    if pending_bots_v2 and len(pending_bots_v2) < len(BOT_STRATEGIES_V2):
        already_v2 = [BOT_STRATEGIES_V2[b]["label"] for b in BOT_STRATEGIES_V2 if b not in pending_bots_v2]
        print(f"[{today_str}] v2:{', v2:'.join(already_v2)} 봇은 오늘 이미 실행했습니다 — 나머지만 진행.")

    try:
        kr_snapshot = screener.screen(market="ALL", days_back=7)
    except RuntimeError as e:
        print(f"[{today_str}] 오늘 KRX 스냅샷을 아직 가져올 수 없습니다 ({e}) — 매매 없이 종료.")
        return

    current_prices = dict(zip(kr_snapshot["Code"], kr_snapshot["Close"], strict=True))
    name_by_code = dict(zip(kr_snapshot["Code"], kr_snapshot["Name"], strict=True))
    watchlist_codes = set(screener.top_movers(kr_snapshot, by="Amount", n=WATCHLIST_SIZE)["Code"])

    # 해외증시·코인은 이번에 새로 추가된 시장이라, 국내(KRX)와 달리 시세를 못 가져와도
    # 전체 실행을 중단하지 않는다 — 그 시장 후보만 빼고 국내 매매는 그대로 진행한다.
    try:
        us_snapshot = us_screener.screen()
        usdkrw_rate = dl.get_usdkrw_rate()
        if not usdkrw_rate or pd.isna(usdkrw_rate):
            raise RuntimeError("USD/KRW 환율을 가져오지 못함")
        name_by_code.update(dict(zip(us_snapshot["Code"], us_snapshot["Name"], strict=True)))
        # 이후 파이프라인(현금 확인·포지션 한도·최소거래금액 등)이 전부 원화 기준이라
        # 여기서 딱 한 번 환산해 넣는다 — _build_signal()의 예측 수익률(%)·RSI는 비율이라
        # 통화와 무관하므로 거기서는 환산하지 않는다.
        current_prices.update(
            {
                code: close * usdkrw_rate
                for code, close in zip(us_snapshot["Code"], us_snapshot["Close"], strict=True)
            }
        )
        watchlist_codes |= set(screener.top_movers(us_snapshot, by="Amount", n=US_WATCHLIST_SIZE)["Code"])
    except Exception as e:
        print(f"[{today_str}] 해외증시 시세를 가져오지 못했습니다 ({e}) — 해외증시 후보 없이 진행.")

    try:
        coin_snapshot = crypto_screener.screen()
        name_by_code.update(dict(zip(coin_snapshot["Code"], coin_snapshot["Name"], strict=True)))
        current_prices.update(dict(zip(coin_snapshot["Code"], coin_snapshot["Close"], strict=True)))
        watchlist_codes |= set(screener.top_movers(coin_snapshot, by="Amount", n=COIN_WATCHLIST_SIZE)["Code"])
    except Exception as e:
        print(f"[{today_str}] 코인 시세를 가져오지 못했습니다 ({e}) — 코인 후보 없이 진행.")

    # 봇마다 원장이 분리돼 있으므로 holdings/cash도 봇별로 따로 갖고 있어야 한다 —
    # 신호 계산(비용이 큼)만 v1+v2 전체 봇이 공유한다.
    bot_holdings = {
        bot_id: portfolio.get_holdings(config.portfolio_dir_for(bot_id)) for bot_id in pending_bots
    }
    bot_holdings_v2 = {
        bot_id: portfolio.get_holdings(config.portfolio_dir_for_v2(bot_id)) for bot_id in pending_bots_v2
    }
    held_names: dict[str, str] = {}
    held_codes: set[str] = set()
    for holdings in list(bot_holdings.values()) + list(bot_holdings_v2.values()):
        held_names.update(dict(zip(holdings["code"], holdings["name"], strict=True)))
        held_codes |= set(holdings["code"])

    candidate_codes = sorted(watchlist_codes | held_codes)

    signals = []
    skipped = []
    for code in candidate_codes:
        name = name_by_code.get(code, held_names.get(code, code))
        try:
            sig = _build_signal(code, name)
        except Exception as e:
            print(f"  {code}({name}) 신호 생성 중 오류: {e}")
            sig = None
        if sig is None:
            skipped.append(code)
        else:
            signals.append(sig)

    print(
        f"[{today_str}] 워치리스트 {len(watchlist_codes)}종목(국내+해외+코인) + 보유(전 봇 합집합) "
        f"{len(held_codes)}종목 중 신호 {len(signals)}개 생성, {len(skipped)}개 스킵"
    )

    kospi_df = dl.get_price("KOSPI")
    kospi_close = float(kospi_df["Close"].iloc[-1]) if not kospi_df.empty else None

    for bot_id in pending_bots:
        label = BOT_STRATEGIES[bot_id]["label"]
        try:
            _run_bot(
                bot_id=bot_id,
                today_str=today_str,
                holdings=bot_holdings[bot_id],
                cash=bot_states[bot_id]["cash"],
                signals=signals,
                current_prices=current_prices,
                name_by_code=name_by_code,
                held_names=held_names,
                kospi_close=kospi_close,
                dry_run=dry_run,
            )
        except Exception as e:
            print(f"[{today_str}] [{label}] 실행 중 오류로 이 봇은 이번 실행을 건너뜁니다: {e}")

    # v2 봇(PRD 11장) — 위 v1 신호(signals)를 그대로 재사용해 판단→가드레일→체결→원장
    # 반영까지 독립적으로 완결한다. 원장이 v1과 완전히 분리돼 있어(config.
    # portfolio_dir_for_v2) v2 봇 하나의 예외가 v1 봇이나 다른 v2 봇을 막지 않는다.
    for bot_id in pending_bots_v2:
        label = f"v2:{BOT_STRATEGIES_V2[bot_id]['label']}"
        try:
            _run_bot_v2(
                bot_id=bot_id,
                today_str=today_str,
                holdings=bot_holdings_v2[bot_id],
                cash=bot_states_v2[bot_id]["cash"],
                signals=signals,
                current_prices=current_prices,
                name_by_code=name_by_code,
                held_names=held_names,
                kospi_close=kospi_close,
                dry_run=dry_run,
            )
        except Exception as e:
            print(f"[{today_str}] [{label}] 실행 중 오류로 이 봇은 이번 실행을 건너뜁니다: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="모의투자 일일 매매 실행")
    parser.add_argument(
        "--dry-run", action="store_true", help="원장 파일을 건드리지 않고 판단 결과만 출력한다"
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
