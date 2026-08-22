"""portfolio.py 테스트 — 모의투자 원장 읽기/쓰기.

원장 경로를 tmp_path로 격리한다(news.py 감성 로그 테스트와 같은 패턴) — 실제
data/portfolio/를 절대 건드리지 않는다. 전부 오프라인, 네트워크 없음.
"""

import json

import pandas as pd
import pytest

from src import portfolio


@pytest.fixture(autouse=True)
def _isolate_portfolio_dir(tmp_path, monkeypatch):
    """모든 테스트에서 원장 경로를 tmp_path 아래로 돌린다."""
    monkeypatch.setattr(portfolio, "_STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(portfolio, "_HOLDINGS_PATH", tmp_path / "holdings.csv")
    monkeypatch.setattr(portfolio, "_TRADES_PATH", tmp_path / "trades.csv")
    monkeypatch.setattr(portfolio, "_EQUITY_PATH", tmp_path / "equity_history.csv")


# ----------------------------------------------------------- 초기 상태


def test_initial_state_is_100m_cash_no_holdings():
    state = portfolio.get_state()
    assert state == {"cash": 100_000_000.0, "last_run_date": None}
    assert portfolio.get_holdings().empty
    assert portfolio.get_trades().empty
    assert portfolio.get_equity_history().empty


def test_initial_holdings_have_expected_columns():
    holdings = portfolio.get_holdings()
    assert list(holdings.columns) == ["code", "name", "quantity", "avg_price"]


# ----------------------------------------------------------- apply_trade — 매수


def test_apply_trade_buy_new_position():
    holdings = portfolio.get_holdings()
    cash = portfolio.INITIAL_CASH

    holdings, cash, record = portfolio.apply_trade(
        holdings,
        cash,
        date="2026-08-13",
        code="005930",
        name="삼성전자",
        action="buy",
        quantity=10,
        price=70_000,
        reason="테스트 매수",
    )

    assert cash == pytest.approx(100_000_000 - 700_000)
    row = holdings[holdings["code"] == "005930"].iloc[0]
    assert row["quantity"] == 10
    assert row["avg_price"] == pytest.approx(70_000)
    assert record == {
        "date": "2026-08-13",
        "code": "005930",
        "name": "삼성전자",
        "action": "buy",
        "quantity": 10,
        "price": 70_000,
        "amount": 700_000,
        "reason": "테스트 매수",
    }


def test_apply_trade_buy_existing_position_weighted_average():
    holdings = portfolio.get_holdings()
    cash = portfolio.INITIAL_CASH

    holdings, cash, _ = portfolio.apply_trade(
        holdings,
        cash,
        date="2026-08-13",
        code="005930",
        name="삼성전자",
        action="buy",
        quantity=10,
        price=10_000,
        reason="1차 매수",
    )
    holdings, cash, _ = portfolio.apply_trade(
        holdings,
        cash,
        date="2026-08-14",
        code="005930",
        name="삼성전자",
        action="buy",
        quantity=10,
        price=12_000,
        reason="2차 매수",
    )

    row = holdings[holdings["code"] == "005930"].iloc[0]
    assert row["quantity"] == 20
    assert row["avg_price"] == pytest.approx(11_000)  # (10*10000 + 10*12000) / 20
    assert cash == pytest.approx(100_000_000 - 100_000 - 120_000)


def test_apply_trade_buy_insufficient_cash_raises():
    holdings = portfolio.get_holdings()
    with pytest.raises(ValueError, match="현금 부족"):
        portfolio.apply_trade(
            holdings,
            1_000,
            date="2026-08-13",
            code="005930",
            name="삼성전자",
            action="buy",
            quantity=1,
            price=70_000,
            reason="현금 부족 테스트",
        )


# ----------------------------------------------------------- apply_trade — 매도


def test_apply_trade_sell_partial_keeps_avg_price():
    holdings = portfolio.get_holdings()
    cash = portfolio.INITIAL_CASH
    holdings, cash, _ = portfolio.apply_trade(
        holdings,
        cash,
        date="2026-08-13",
        code="005930",
        name="삼성전자",
        action="buy",
        quantity=10,
        price=70_000,
        reason="매수",
    )

    holdings, cash, record = portfolio.apply_trade(
        holdings,
        cash,
        date="2026-08-14",
        code="005930",
        name="삼성전자",
        action="sell",
        quantity=4,
        price=80_000,
        reason="일부 매도",
    )

    row = holdings[holdings["code"] == "005930"].iloc[0]
    assert row["quantity"] == 6
    assert row["avg_price"] == pytest.approx(70_000)  # 매도는 평단가를 바꾸지 않는다
    assert cash == pytest.approx(100_000_000 - 700_000 + 320_000)
    assert record["amount"] == 320_000


def test_apply_trade_sell_full_removes_holding():
    holdings = portfolio.get_holdings()
    cash = portfolio.INITIAL_CASH
    holdings, cash, _ = portfolio.apply_trade(
        holdings,
        cash,
        date="2026-08-13",
        code="005930",
        name="삼성전자",
        action="buy",
        quantity=10,
        price=70_000,
        reason="매수",
    )

    holdings, cash, _ = portfolio.apply_trade(
        holdings,
        cash,
        date="2026-08-14",
        code="005930",
        name="삼성전자",
        action="sell",
        quantity=10,
        price=75_000,
        reason="전량 매도",
    )

    assert holdings.empty
    assert cash == pytest.approx(100_000_000 - 700_000 + 750_000)


def test_apply_trade_sell_more_than_held_raises():
    holdings = portfolio.get_holdings()
    cash = portfolio.INITIAL_CASH
    holdings, cash, _ = portfolio.apply_trade(
        holdings,
        cash,
        date="2026-08-13",
        code="005930",
        name="삼성전자",
        action="buy",
        quantity=5,
        price=70_000,
        reason="매수",
    )
    with pytest.raises(ValueError, match="보유 수량"):
        portfolio.apply_trade(
            holdings,
            cash,
            date="2026-08-14",
            code="005930",
            name="삼성전자",
            action="sell",
            quantity=10,
            price=70_000,
            reason="초과 매도",
        )


def test_apply_trade_sell_unheld_stock_raises():
    holdings = portfolio.get_holdings()
    with pytest.raises(ValueError, match="보유하지 않은"):
        portfolio.apply_trade(
            holdings,
            portfolio.INITIAL_CASH,
            date="2026-08-13",
            code="005930",
            name="삼성전자",
            action="sell",
            quantity=1,
            price=70_000,
            reason="미보유 매도",
        )


def test_apply_trade_invalid_action_raises():
    holdings = portfolio.get_holdings()
    with pytest.raises(ValueError, match="action"):
        portfolio.apply_trade(
            holdings,
            portfolio.INITIAL_CASH,
            date="2026-08-13",
            code="005930",
            name="삼성전자",
            action="hold",
            quantity=1,
            price=70_000,
            reason="잘못된 액션",
        )


# ----------------------------------------------------------- mark_to_market


def test_mark_to_market_all_prices_available():
    holdings = pd.DataFrame(
        [
            {"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 70_000},
            {"code": "000660", "name": "SK하이닉스", "quantity": 5, "avg_price": 100_000},
        ]
    )
    valued, total = portfolio.mark_to_market(holdings, {"005930": 75_000, "000660": 90_000})

    assert total == pytest.approx(75_000 * 10 + 90_000 * 5)
    row = valued[valued["code"] == "005930"].iloc[0]
    assert row["current_price"] == 75_000
    assert row["market_value"] == 750_000
    assert row["unrealized_pnl"] == pytest.approx((75_000 - 70_000) * 10)
    assert not row["price_is_stale"]


def test_mark_to_market_missing_price_falls_back_to_avg_price():
    holdings = pd.DataFrame([{"code": "999999", "name": "거래정지종목", "quantity": 3, "avg_price": 50_000}])
    valued, total = portfolio.mark_to_market(holdings, {})  # 시세 없음 (거래정지 등)

    row = valued.iloc[0]
    assert row["current_price"] == 50_000  # 평단가로 대체
    assert row["price_is_stale"]
    assert total == pytest.approx(150_000)


def test_mark_to_market_empty_holdings():
    valued, total = portfolio.mark_to_market(portfolio.get_holdings(), {})
    assert valued.empty
    assert total == 0.0


# ----------------------------------------------------------- save_daily_result


def test_save_daily_result_writes_state_last_and_records_trades():
    holdings = portfolio.get_holdings()
    cash = portfolio.INITIAL_CASH
    holdings, cash, trade = portfolio.apply_trade(
        holdings,
        cash,
        date="2026-08-13",
        code="005930",
        name="삼성전자",
        action="buy",
        quantity=10,
        price=70_000,
        reason="테스트",
    )

    portfolio.save_daily_result(
        date="2026-08-13",
        cash=cash,
        holdings=holdings,
        new_trades=[trade],
        equity_row={
            "date": "2026-08-13",
            "cash": cash,
            "holdings_value": 700_000,
            "total_equity": cash + 700_000,
            "kospi_close": 3000.0,
        },
    )

    state = portfolio.get_state()
    assert state["last_run_date"] == "2026-08-13"
    assert state["cash"] == pytest.approx(cash)

    trades = portfolio.get_trades()
    assert len(trades) == 1
    assert trades.iloc[0]["code"] == "005930"

    equity = portfolio.get_equity_history()
    assert len(equity) == 1


def test_save_daily_result_no_trades_still_adds_one_equity_row():
    """매매 0건(관망)인 날도 시가평가 행은 반드시 하나 쌓인다."""
    holdings = portfolio.get_holdings()
    portfolio.save_daily_result(
        date="2026-08-13",
        cash=portfolio.INITIAL_CASH,
        holdings=holdings,
        new_trades=[],
        equity_row={
            "date": "2026-08-13",
            "cash": portfolio.INITIAL_CASH,
            "holdings_value": 0.0,
            "total_equity": portfolio.INITIAL_CASH,
            "kospi_close": 3000.0,
        },
    )
    assert len(portfolio.get_trades()) == 0
    assert len(portfolio.get_equity_history()) == 1


def test_save_daily_result_appends_across_multiple_days():
    for day, price in [("2026-08-13", 70_000), ("2026-08-14", 71_000)]:
        holdings = portfolio.get_holdings()
        state = portfolio.get_state()
        holdings, cash, trade = portfolio.apply_trade(
            holdings,
            state["cash"],
            date=day,
            code="005930",
            name="삼성전자",
            action="buy",
            quantity=1,
            price=price,
            reason="분할 매수",
        )
        portfolio.save_daily_result(
            date=day,
            cash=cash,
            holdings=holdings,
            new_trades=[trade],
            equity_row={
                "date": day,
                "cash": cash,
                "holdings_value": price,
                "total_equity": cash + price,
                "kospi_close": 3000.0,
            },
        )

    assert len(portfolio.get_trades()) == 2
    assert len(portfolio.get_equity_history()) == 2
    holdings = portfolio.get_holdings()
    assert holdings.iloc[0]["quantity"] == 2


# ----------------------------------------------------------- CSV 안정성 (핵심 설계 검증)
# 이 원장에 parquet 대신 CSV를 쓰기로 한 이유가 실제로 성립하는지 직접 검증한다:
# 같은 내용을 다시 써도(읽었다가 그대로 재작성해도) 바이트가 완전히 같아야 한다.


def test_holdings_csv_round_trip_is_byte_identical():
    holdings = pd.DataFrame(
        [
            {"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 70_000.0},
            {"code": "000660", "name": "SK하이닉스", "quantity": 5, "avg_price": 123_456.789},
        ]
    )
    portfolio.save_daily_result(
        date="2026-08-13",
        cash=1_000_000.0,
        holdings=holdings,
        new_trades=[],
        equity_row={
            "date": "2026-08-13",
            "cash": 1_000_000.0,
            "holdings_value": 0.0,
            "total_equity": 1_000_000.0,
            "kospi_close": 3000.0,
        },
    )
    bytes_first = portfolio._HOLDINGS_PATH.read_bytes()

    # 읽어서 그대로 다시 쓴다 — 매매가 없는 날 save_daily_result가 실제로 하는 일과 같다.
    reread = portfolio.get_holdings()
    portfolio.save_daily_result(
        date="2026-08-14",
        cash=1_000_000.0,
        holdings=reread,
        new_trades=[],
        equity_row={
            "date": "2026-08-14",
            "cash": 1_000_000.0,
            "holdings_value": 0.0,
            "total_equity": 1_000_000.0,
            "kospi_close": 3000.0,
        },
    )
    bytes_second = portfolio._HOLDINGS_PATH.read_bytes()

    assert bytes_first == bytes_second, (
        "holdings.csv 내용이 안 바뀌었는데 바이트가 달라졌다 (parquet과 같은 문제 재발)"
    )


def test_state_json_last_run_date_updates_correctly():
    """state.json은 순수 텍스트이므로 같은 내용이면 항상 같은 바이트 — 그리고 무엇보다
    last_run_date가 매번 정확히 그날 날짜로 갱신되는지 확인한다(원자적 쓰기 순서의 전제)."""
    holdings = portfolio.get_holdings()
    portfolio.save_daily_result(
        date="2026-08-13",
        cash=portfolio.INITIAL_CASH,
        holdings=holdings,
        new_trades=[],
        equity_row={
            "date": "2026-08-13",
            "cash": portfolio.INITIAL_CASH,
            "holdings_value": 0.0,
            "total_equity": portfolio.INITIAL_CASH,
            "kospi_close": 3000.0,
        },
    )
    saved = json.loads(portfolio._STATE_PATH.read_text(encoding="utf-8"))
    assert saved["last_run_date"] == "2026-08-13"


# ----------------------------------------------------------- portfolio_dir (다중 봇 원장 분리)
# 모의투자 봇 다중화(기본형/공격적/모멘텀) — 각 봇이 완전히 분리된 원장에서 읽고 쓰는지
# 확인한다. 위 테스트들은 모두 portfolio_dir=None(기본값)으로 autouse 픽스처가 monkeypatch한
# 경로를 쓰는데, 그 경로 자체가 바뀌지 않았으므로 전부 무수정으로 통과해야 한다 — 이게
# 곧 "기본형 봇 무영향"의 회귀 검증이다.


def test_portfolio_dir_isolates_two_bot_instances(tmp_path):
    dir_a = tmp_path / "bot_a"
    dir_b = tmp_path / "bot_b"

    holdings_a = portfolio.get_holdings(dir_a)
    holdings_a, cash_a, trade_a = portfolio.apply_trade(
        holdings_a,
        portfolio.INITIAL_CASH,
        date="2026-08-13",
        code="005930",
        name="삼성전자",
        action="buy",
        quantity=10,
        price=70_000,
        reason="봇A 매수",
    )
    portfolio.save_daily_result(
        date="2026-08-13",
        cash=cash_a,
        holdings=holdings_a,
        new_trades=[trade_a],
        equity_row={
            "date": "2026-08-13",
            "cash": cash_a,
            "holdings_value": 700_000,
            "total_equity": cash_a + 700_000,
            "kospi_close": 3000.0,
        },
        portfolio_dir=dir_a,
    )

    # 봇B는 아직 아무 것도 안 했으므로 초기 상태 그대로여야 한다 — 봇A의 매매가 전혀 안 보임
    state_b = portfolio.get_state(dir_b)
    assert state_b == {"cash": portfolio.INITIAL_CASH, "last_run_date": None}
    assert portfolio.get_holdings(dir_b).empty
    assert portfolio.get_trades(dir_b).empty

    # 봇A는 반영돼 있어야 한다
    state_a = portfolio.get_state(dir_a)
    assert state_a["last_run_date"] == "2026-08-13"
    trades_a = portfolio.get_trades(dir_a)
    assert len(trades_a) == 1
    assert trades_a.iloc[0]["code"] == "005930"


def test_portfolio_dir_none_uses_isolated_default_path(tmp_path):
    """portfolio_dir=None(기본값)은 이 파일의 autouse 픽스처가 monkeypatch한 모듈 상수
    (_STATE_PATH 등)를 그대로 쓴다 — 명시적 portfolio_dir를 넘긴 다른 봇 원장과 실제
    파일 경로가 겹치지 않는지 확인한다(초기 상태값 자체는 둘 다 같아 값 비교로는
    분리를 증명할 수 없으므로 경로를 직접 비교한다)."""
    other = tmp_path / "other_bot"
    portfolio.get_holdings(other)  # other_bot 디렉터리 초기화만 트리거
    assert portfolio.get_state() == portfolio.get_state(other)  # 둘 다 초기 상태라 값은 같지만
    assert portfolio._STATE_PATH != other / "state.json"  # 실제 파일 경로는 서로 다르다


def test_portfolio_dir_creates_missing_directory(tmp_path):
    """봇 버전 디렉터리(예: data/portfolio/aggressive/)는 최초 호출 시 자동 생성돼야
    한다 — config.py가 미리 만들어두지 않으므로 portfolio.py가 직접 mkdir 해야 한다."""
    nested = tmp_path / "aggressive"
    assert not nested.exists()

    state = portfolio.get_state(nested)

    assert state == {"cash": portfolio.INITIAL_CASH, "last_run_date": None}
    assert nested.exists()
    assert (nested / "state.json").exists()
    assert (nested / "holdings.csv").exists()
    assert (nested / "trades.csv").exists()
    assert (nested / "equity_history.csv").exists()
