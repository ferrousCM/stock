"""모의투자 포트폴리오 원장 — 현금·보유종목·거래이력·자산추이를 관리하는 데이터 계층.

PRD.md 5.4 참고. 이 모듈은 파일 I/O와 잔고 계산만 담당한다. 어떤 종목을 얼마나 살지
판단하는 로직(규칙 엔진)은 trading_agent.py의 몫이고, git 커밋은 GitHub Actions
워크플로의 몫이다 — 여기선 순수하게 원장 상태만 다룬다.

**parquet이 아니라 CSV를 쓴다.** 다른 데이터 파일(캐시·뉴스 감성 로그)은 parquet인데
원장만 CSV인 이유: parquet은 같은 내용을 다시 써도 바이트가 달라진다(CLAUDE.md "알려진
제약" 참고) — 이 원장은 매일 git에 커밋되므로, parquet을 썼다면 news.py의
`log_daily_sentiment()`가 겪은 것과 같은 문제(무의미한 diff 커밋)를 반복하게 된다. CSV는
내용이 같으면 바이트도 같아서, 그 우회 로직(`_already_logged()`) 자체가 필요 없다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config

INITIAL_CASH = 100_000_000.0  # 모의투자 시작 자금 1억원

_STATE_PATH = config.PORTFOLIO_DIR / "state.json"
_HOLDINGS_PATH = config.PORTFOLIO_DIR / "holdings.csv"
_TRADES_PATH = config.PORTFOLIO_DIR / "trades.csv"
_EQUITY_PATH = config.PORTFOLIO_DIR / "equity_history.csv"

_HOLDINGS_COLS = ["code", "name", "quantity", "avg_price"]
_TRADES_COLS = ["date", "code", "name", "action", "quantity", "price", "amount", "reason"]
_EQUITY_COLS = ["date", "cash", "holdings_value", "total_equity", "kospi_close"]


def _paths(portfolio_dir: Path | None) -> dict[str, Path]:
    """portfolio_dir가 None이면 모듈 상수(_STATE_PATH 등)를 그대로 참조한다 — 이름으로
    참조하므로 tests/test_portfolio.py의 monkeypatch.setattr(portfolio, "_STATE_PATH", ...)
    패턴이 그대로 동작한다(기본 봇 하위호환). portfolio_dir가 주어지면 그 디렉터리 기준으로
    새 경로 4개를 계산한다 — 봇 버전별(공격적/모멘텀) 원장 분리에 쓴다."""
    if portfolio_dir is None:
        return {
            "state": _STATE_PATH,
            "holdings": _HOLDINGS_PATH,
            "trades": _TRADES_PATH,
            "equity": _EQUITY_PATH,
        }
    return {
        "state": portfolio_dir / "state.json",
        "holdings": portfolio_dir / "holdings.csv",
        "trades": portfolio_dir / "trades.csv",
        "equity": portfolio_dir / "equity_history.csv",
    }


def _ensure_initialized(portfolio_dir: Path | None = None) -> None:
    """원장 파일이 없으면 초기 상태(현금 1억원, 무보유)로 만든다. 매 읽기/쓰기 앞에서 호출."""
    if portfolio_dir is not None:
        portfolio_dir.mkdir(parents=True, exist_ok=True)
    paths = _paths(portfolio_dir)
    if not paths["state"].exists():
        paths["state"].write_text(
            json.dumps({"cash": INITIAL_CASH, "last_run_date": None}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if not paths["holdings"].exists():
        pd.DataFrame(columns=_HOLDINGS_COLS).to_csv(paths["holdings"], index=False)
    if not paths["trades"].exists():
        pd.DataFrame(columns=_TRADES_COLS).to_csv(paths["trades"], index=False)
    if not paths["equity"].exists():
        pd.DataFrame(columns=_EQUITY_COLS).to_csv(paths["equity"], index=False)


def get_state(portfolio_dir: Path | None = None) -> dict:
    """{"cash": float, "last_run_date": str | None} — last_run_date는 YYYY-MM-DD 또는 아직 한 번도 안 돌았으면 None."""
    _ensure_initialized(portfolio_dir)
    return json.loads(_paths(portfolio_dir)["state"].read_text(encoding="utf-8"))


def get_holdings(portfolio_dir: Path | None = None) -> pd.DataFrame:
    """columns: code, name, quantity, avg_price."""
    _ensure_initialized(portfolio_dir)
    return pd.read_csv(_paths(portfolio_dir)["holdings"], dtype={"code": str})


def get_trades(portfolio_dir: Path | None = None) -> pd.DataFrame:
    """columns: date, code, name, action, quantity, price, amount, reason (append-only 이력)."""
    _ensure_initialized(portfolio_dir)
    return pd.read_csv(_paths(portfolio_dir)["trades"], dtype={"code": str})


def get_equity_history(portfolio_dir: Path | None = None) -> pd.DataFrame:
    """columns: date, cash, holdings_value, total_equity, kospi_close (일별 자산 스냅샷)."""
    _ensure_initialized(portfolio_dir)
    return pd.read_csv(_paths(portfolio_dir)["equity"])


def apply_trade(
    holdings: pd.DataFrame,
    cash: float,
    *,
    date: str,
    code: str,
    name: str,
    action: str,
    quantity: int | float,
    price: float,
    reason: str,
) -> tuple[pd.DataFrame, float, dict]:
    """매수/매도 1건을 보유종목·현금에 반영한다. 파일은 건드리지 않는 순수 함수 —
    호출부(매매 규칙 엔진)가 하루치 거래를 전부 이 함수로 누적 반영한 뒤,
    save_daily_result()로 한 번에 기록한다.

    매수는 수량가중평균으로 평단가를 갱신하고, 매도는 평단가를 바꾸지 않는다(실현손익은
    평단가 계산에 영향을 주지 않는 것이 표준 회계 방식). 반환값: (갱신된 holdings,
    갱신된 cash, trades.csv에 append할 한 행짜리 dict).
    """
    if quantity <= 0:
        raise ValueError(f"quantity는 양수여야 합니다: {quantity}")
    if price <= 0:
        raise ValueError(f"price는 양수여야 합니다: {price}")

    holdings = holdings.copy()
    amount = quantity * price
    existing = holdings[holdings["code"] == code]

    if action == "buy":
        if amount > cash:
            raise ValueError(f"현금 부족: 필요 {amount:,.0f}원, 보유 {cash:,.0f}원")
        cash -= amount
        if existing.empty:
            new_row = pd.DataFrame([{"code": code, "name": name, "quantity": quantity, "avg_price": price}])
            holdings = pd.concat([holdings, new_row], ignore_index=True)
        else:
            idx = existing.index[0]
            old_qty = holdings.at[idx, "quantity"]
            old_avg = holdings.at[idx, "avg_price"]
            new_qty = old_qty + quantity
            holdings.at[idx, "quantity"] = new_qty
            holdings.at[idx, "avg_price"] = (old_qty * old_avg + quantity * price) / new_qty
    elif action == "sell":
        if existing.empty:
            raise ValueError(f"보유하지 않은 종목은 매도할 수 없습니다: {code}")
        idx = existing.index[0]
        old_qty = holdings.at[idx, "quantity"]
        if quantity > old_qty:
            raise ValueError(f"보유 수량({old_qty})보다 많이 매도할 수 없습니다: {quantity}")
        cash += amount
        remaining = old_qty - quantity
        if remaining == 0:
            holdings = holdings.drop(index=idx).reset_index(drop=True)
        else:
            holdings.at[idx, "quantity"] = remaining
    else:
        raise ValueError(f"action은 'buy' 또는 'sell'이어야 합니다: {action!r}")

    trade_record = {
        "date": date,
        "code": code,
        "name": name,
        "action": action,
        "quantity": quantity,
        "price": price,
        "amount": amount,
        "reason": reason,
    }
    return holdings, cash, trade_record


def mark_to_market(holdings: pd.DataFrame, close_prices: dict[str, float]) -> tuple[pd.DataFrame, float]:
    """보유종목을 그날 종가로 평가한다. holdings에 current_price/market_value/unrealized_pnl/
    price_is_stale 컬럼을 추가해 반환하고, 전체 평가금액 합계도 같이 돌려준다.

    close_prices에 없는 종목(거래정지·상장폐지 등으로 그날 종가를 못 가져온 경우)은
    호출부가 이미 최선의 대체값(예: 마지막으로 알려진 시세)을 골라 넣어주는 것이 원칙이다 —
    이 함수는 그래도 값이 없는 경우에 대비한 최후의 안전장치로만 평단가를 대신 쓰고,
    price_is_stale=True로 표시해 UI가 "추정치"임을 보여줄 수 있게 한다.
    """
    holdings = holdings.copy()
    if holdings.empty:
        holdings["current_price"] = pd.Series(dtype=float)
        holdings["market_value"] = pd.Series(dtype=float)
        holdings["unrealized_pnl"] = pd.Series(dtype=float)
        holdings["price_is_stale"] = pd.Series(dtype=bool)
        return holdings, 0.0

    current_prices = []
    is_stale = []
    for _, row in holdings.iterrows():
        price = close_prices.get(row["code"])
        if price is None or pd.isna(price):
            current_prices.append(row["avg_price"])
            is_stale.append(True)
        else:
            current_prices.append(price)
            is_stale.append(False)

    holdings["current_price"] = current_prices
    holdings["price_is_stale"] = is_stale
    holdings["market_value"] = holdings["current_price"] * holdings["quantity"]
    holdings["unrealized_pnl"] = (holdings["current_price"] - holdings["avg_price"]) * holdings["quantity"]

    return holdings, float(holdings["market_value"].sum())


def save_daily_result(
    date: str,
    cash: float,
    holdings: pd.DataFrame,
    new_trades: list[dict],
    equity_row: dict,
    portfolio_dir: Path | None = None,
) -> None:
    """그날의 매매·시가평가 결과를 원장 4개 파일에 반영한다.

    쓰기 순서가 중요하다 — holdings/trades/equity_history를 먼저 쓰고, state.json(last_run_date
    포함)을 마지막에 쓴다. 실행이 중간에 실패하면 last_run_date가 갱신되지 않은 채 남아서
    다음 실행이 "오늘이 아직 안 끝났다"고 판단해 재시도할 수 있다 — 순서가 반대면 체결은
    이미 반영됐는데 재시도가 그걸 모르고 같은 매매를 한 번 더 하게 된다.

    equity_row는 매매 건수와 무관하게 매 거래일 반드시 하나씩 쌓인다(호출부 책임 — 관망한
    날에도 보유종목 종가가 바뀌므로 총자산은 변한다).

    portfolio_dir를 넘기면 그 디렉터리의 원장에 쓴다(봇 버전별 분리) — 생략하면 기본형 봇의
    기존 경로를 그대로 쓴다.
    """
    _ensure_initialized(portfolio_dir)
    paths = _paths(portfolio_dir)

    # 컬럼 순서를 고정해서 매번 같은 형태로 직렬화되게 한다 — CSV가 안정적인 이유는
    # "내용이 같으면 바이트도 같다"는 성질 덕분인데, 컬럼 순서가 흔들리면 그 성질이 깨진다.
    holdings_out = holdings[_HOLDINGS_COLS].reset_index(drop=True)

    trades_out = get_trades(portfolio_dir)
    if new_trades:
        trades_out = pd.concat([trades_out, pd.DataFrame(new_trades)[_TRADES_COLS]], ignore_index=True)

    equity_out = pd.concat(
        [get_equity_history(portfolio_dir), pd.DataFrame([equity_row])[_EQUITY_COLS]], ignore_index=True
    )

    # 여기까지는 전부 메모리 위에서만 계산했다 — 디스크에는 아직 아무것도 안 썼다.
    holdings_out.to_csv(paths["holdings"], index=False)
    trades_out.to_csv(paths["trades"], index=False)
    equity_out.to_csv(paths["equity"], index=False)

    # state.json은 반드시 마지막 — "오늘 실행이 끝났다"는 유일한 신호이기 때문이다.
    paths["state"].write_text(
        json.dumps({"cash": cash, "last_run_date": date}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
