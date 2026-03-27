from __future__ import annotations

from typing import List, Dict, Any, Tuple


def compute_ma(closes: List[float], window: int) -> List[float | None]:
    result: List[float | None] = []
    s = 0.0
    for i, v in enumerate(closes):
        s += v
        if i >= window:
            s -= closes[i - window]
        if i >= window - 1:
            result.append(s / window)
        else:
            result.append(None)
    return result


def compute_ema(closes: List[float], span: int) -> List[float]:
    ema: List[float] = []
    k = 2 / (span + 1)
    prev = None
    for v in closes:
        prev = v if prev is None else (v - prev) * k + prev
        ema.append(prev)
    return ema


def compute_macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
    ema_fast = compute_ema(closes, fast)
    ema_slow = compute_ema(closes, slow)
    dif = [f - s for f, s in zip(ema_fast, ema_slow)]
    dea = compute_ema(dif, signal)
    macd = [(d - e) * 2 for d, e in zip(dif, dea)]
    return dif, dea, macd


def compute_rsi(closes: List[float], period: int = 14) -> List[float | None]:
    rsis: List[float | None] = []
    gains: List[float] = [0.0]
    losses: List[float] = [0.0]
    for i in range(1, len(closes)):
        chg = closes[i] - closes[i - 1]
        gains.append(max(chg, 0.0))
        losses.append(max(-chg, 0.0))
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(len(closes)):
        if i < period:
            rsis.append(None)
            if i > 0:
                avg_gain += gains[i]
                avg_loss += losses[i]
            if i == period - 1:
                avg_gain /= period
                avg_loss /= period
            continue
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsis.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsis.append(100 - (100 / (1 + rs)))
    return rsis


def build_indicators(klines: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    closes = [float(k["close"]) for k in klines]
    ma5 = compute_ma(closes, 5)
    ma10 = compute_ma(closes, 10)
    dif, dea, macd = compute_macd(closes)
    rsi = compute_rsi(closes)
    return {
        "ma5": ma5,
        "ma10": ma10,
        "dif": dif,
        "dea": dea,
        "macd": macd,
        "rsi": rsi,
    }


