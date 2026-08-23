"""기술적 지표 및 성과 지표.

모든 함수는 pandas Series/DataFrame을 받아 같은 인덱스의 결과를 돌려준다.
입력을 변형하지 않는다(부작용 없음).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS

# ---------------------------------------------------------------- 추세 지표


def sma(s: pd.Series, window: int = 20) -> pd.Series:
    """단순이동평균."""
    return s.rolling(window).mean()


def ema(s: pd.Series, span: int = 20) -> pd.Series:
    """지수이동평균."""
    return s.ewm(span=span, adjust=False).mean()


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD 선, 시그널선, 히스토그램."""
    line = ema(s, fast) - ema(s, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


# ---------------------------------------------------------------- 모멘텀/변동성


def rsi(s: pd.Series, window: int = 14) -> pd.Series:
    """Wilder RSI (0~100). 통상 70 이상 과매수, 30 이하 과매도로 읽는다."""
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder 평활 = alpha 1/window 인 EMA
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range — 가격 단위 변동성. df는 High/Low/Close 컬럼이 필요하다.

    True Range = max(고가-저가, |고가-전일종가|, |저가-전일종가|). RSI와 동일한 Wilder
    평활(alpha=1/window EMA)을 쓴다 — 표준 ATR 정의. 모의투자 v2(PRD.md 11장)의 손절폭·
    포지션 사이징 기준으로 쓴다.
    """
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def bollinger(s: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """볼린저 밴드. %B는 밴드 내 위치(0=하단, 1=상단)."""
    mid = sma(s, window)
    std = s.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame(
        {
            "mid": mid,
            "upper": upper,
            "lower": lower,
            "bandwidth": (upper - lower) / mid,
            "pct_b": (s - lower) / (upper - lower),
        }
    )


def stochastic(df: pd.DataFrame, k_window: int = 14, d_window: int = 3, smooth_k: int = 3) -> pd.DataFrame:
    """스토캐스틱(Slow) %K/%D. df는 High/Low/Close 컬럼이 필요하다.

    통상 80 이상 과매수, 20 이하 과매도로 읽는다. raw %K를 smooth_k로 한 번 평활한 값을
    "%K"로, 그걸 다시 d_window로 평활한 값을 "%D"로 삼는 슬로우 스토캐스틱 방식이다
    (업비트를 포함해 대부분의 차팅 툴 기본값).
    """
    low_min = df["Low"].rolling(k_window).min()
    high_max = df["High"].rolling(k_window).max()
    raw_k = (df["Close"] - low_min) / (high_max - low_min) * 100
    k = raw_k.rolling(smooth_k).mean()
    d = k.rolling(d_window).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


def ichimoku(
    df: pd.DataFrame,
    tenkan_window: int = 9,
    kijun_window: int = 26,
    senkou_b_window: int = 52,
) -> pd.DataFrame:
    """일목균형표(Ichimoku) — 전환선/기준선/선행스팬A·B/후행스팬. df는 High/Low/Close 필요.

    선행스팬(구름)은 관례대로 kijun_window(기본 26)만큼 미래로 밀어서(shift) 그린다 —
    지금 보이는 구름은 kijun_window일 전 데이터로 계산된 것이라는 뜻. 후행스팬은
    반대로 과거로 밀어서(shift 음수) 그린다.
    """
    tenkan = (df["High"].rolling(tenkan_window).max() + df["Low"].rolling(tenkan_window).min()) / 2
    kijun = (df["High"].rolling(kijun_window).max() + df["Low"].rolling(kijun_window).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(kijun_window)
    senkou_b = (
        (df["High"].rolling(senkou_b_window).max() + df["Low"].rolling(senkou_b_window).min()) / 2
    ).shift(kijun_window)
    chikou = df["Close"].shift(-kijun_window)
    return pd.DataFrame(
        {
            "ichimoku_tenkan": tenkan,
            "ichimoku_kijun": kijun,
            "ichimoku_senkou_a": senkou_a,
            "ichimoku_senkou_b": senkou_b,
            "ichimoku_chikou": chikou,
        }
    )


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """하이킨아시 캔들로 변환한 OHLC(Volume은 그대로) DataFrame.

    시가/고가/저가/종가를 전 봉과 평활해서 다시 계산하므로 노이즈가 줄고 추세가
    더 매끈하게 보인다 — 대신 실제 체결가와는 다르다(그리기 전용 변환).
    """
    ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    close_vals = ha_close.to_numpy()
    open_vals = np.empty(len(df))
    open_vals[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2
    for i in range(1, len(df)):
        open_vals[i] = (open_vals[i - 1] + close_vals[i - 1]) / 2
    ha_open = pd.Series(open_vals, index=df.index)
    ha_high = pd.concat([df["High"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df["Low"], ha_open, ha_close], axis=1).min(axis=1)

    out = df.copy()
    out["Open"], out["High"], out["Low"], out["Close"] = ha_open, ha_high, ha_low, ha_close
    return out


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """일봉 OHLCV를 더 긴 봉(주봉/월봉 등)으로 리샘플링한다.

    rule: pandas resample 규칙 문자열 — 주봉 "W", 월봉 "ME". 시가=첫값, 고가=최댓값,
    저가=최솟값, 종가=마지막값, 거래량=합계라는 표준 OHLCV 집계 규칙을 쓴다. 원본에
    없는(휴장 등) 구간은 만들지 않는다 — 시가가 없는(원본 데이터가 하나도 없는) 봉은
    버린다.
    """
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df.resample(rule).agg(agg).dropna(subset=["Open"])


def volume_profile(df: pd.DataFrame, bins: int = 24) -> pd.DataFrame:
    """가격대별 매물대(누적 거래량). df는 High/Low/Volume 컬럼이 필요하다.

    조회 구간 전체의 최저가~최고가를 bins개 가격 구간으로 나누고, 각 봉의 거래량을
    "그 봉이 저가~고가 사이 어느 구간에서 거래됐는지"에 비례해 나눠 담는다(종가 하나에만
    몰아주지 않는다 — 그러면 봉 하나가 실제로 거래된 가격 범위를 무시하게 된다). 도지처럼
    저가=고가인 봉은 그 가격이 속한 구간 하나에 전량 배정한다.

    반환: price_low/price_high/price_mid/volume 컬럼, bins행, 가격 낮은 구간부터 순서대로.
    """
    cols = ["price_low", "price_high", "price_mid", "volume"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    lo, hi = float(df["Low"].min()), float(df["High"].max())
    if lo >= hi:
        return pd.DataFrame(columns=cols)

    edges = np.linspace(lo, hi, bins + 1)
    vol = np.zeros(bins)
    for row_low, row_high, row_vol in zip(df["Low"], df["High"], df["Volume"], strict=True):
        if row_vol <= 0:
            continue
        if row_high <= row_low:
            idx = min(max(np.searchsorted(edges, row_low, side="right") - 1, 0), bins - 1)
            vol[idx] += row_vol
            continue
        start_idx = max(np.searchsorted(edges, row_low, side="right") - 1, 0)
        end_idx = min(np.searchsorted(edges, row_high, side="left"), bins - 1)
        span = row_high - row_low
        for i in range(start_idx, end_idx + 1):
            overlap = min(row_high, edges[i + 1]) - max(row_low, edges[i])
            if overlap > 0:
                vol[i] += row_vol * (overlap / span)

    return pd.DataFrame(
        {
            "price_low": edges[:-1],
            "price_high": edges[1:],
            "price_mid": (edges[:-1] + edges[1:]) / 2,
            "volume": vol,
        }
    )


# ---------------------------------------------------------------- 수익률/성과


def returns(s: pd.Series, log: bool = False) -> pd.Series:
    """일간 수익률. log=True면 로그수익률."""
    return np.log(s / s.shift(1)) if log else s.pct_change()


def cumulative_returns(s: pd.Series) -> pd.Series:
    """첫날 대비 누적 수익률 (0.0 = 원금)."""
    return s / s.iloc[0] - 1


def volatility(s: pd.Series, window: int | None = None) -> pd.Series | float:
    """연율화 변동성. window를 주면 rolling, 없으면 전체 기간 스칼라."""
    r = returns(s)
    if window:
        return r.rolling(window).std() * np.sqrt(TRADING_DAYS)
    return float(r.std() * np.sqrt(TRADING_DAYS))


def cagr(s: pd.Series) -> float:
    """연평균 성장률."""
    years = len(s) / TRADING_DAYS
    if years <= 0 or s.iloc[0] <= 0:
        return float("nan")
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1)


def drawdown(s: pd.Series) -> pd.Series:
    """전고점 대비 낙폭 (음수)."""
    return s / s.cummax() - 1


def max_drawdown(s: pd.Series) -> float:
    """최대낙폭(MDD)."""
    return float(drawdown(s).min())


def sharpe(s: pd.Series, risk_free: float = 0.03) -> float:
    """연율화 샤프지수. risk_free는 연 무위험수익률(예: 0.03 = 3%)."""
    r = returns(s).dropna()
    if r.empty or r.std() == 0:
        return float("nan")
    excess = r.mean() * TRADING_DAYS - risk_free
    return float(excess / (r.std() * np.sqrt(TRADING_DAYS)))


def summary(s: pd.Series) -> pd.Series:
    """한 종목의 성과 요약 지표 묶음."""
    return pd.Series(
        {
            "start": s.index[0].date(),
            "end": s.index[-1].date(),
            "total_return": float(s.iloc[-1] / s.iloc[0] - 1),
            "cagr": cagr(s),
            "volatility": volatility(s),
            "sharpe": sharpe(s),
            "max_drawdown": max_drawdown(s),
        }
    )


def add_all(df: pd.DataFrame, col: str = "Close") -> pd.DataFrame:
    """OHLCV DataFrame에 주요 지표 열을 붙여 새 DataFrame으로 반환한다."""
    out = df.copy()
    s = out[col]
    out["sma20"] = sma(s, 20)
    out["sma60"] = sma(s, 60)
    out["sma120"] = sma(s, 120)
    out["rsi14"] = rsi(s, 14)
    out = out.join(macd(s))
    out = out.join(bollinger(s)[["upper", "lower", "pct_b"]])
    out["ret"] = returns(s)
    out["drawdown"] = drawdown(s)
    return out
