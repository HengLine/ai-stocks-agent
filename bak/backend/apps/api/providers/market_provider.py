from __future__ import annotations

from typing import Any, Dict, List
import yfinance as yf


def normalize_ticker(ticker: str) -> str:
    t = (ticker or '').upper().replace(' ', '')
    if ':' in t:
        exch, code = t.split(':', 1)
        if exch in ('SZ', 'SH') and len(code) == 6:
            suffix = '.SZ' if exch == 'SZ' else '.SS'
            return f"{code}{suffix}"
    # 没有交易所前缀时，默认深交所
    if len(t) == 6 and t.isdigit():
        # 以首位 6 常见为上交所，这里做简单推断
        if t.startswith('6'):
            return f"{t}.SS"
        return f"{t}.SZ"
    return t


def window_to_period(window: str) -> str:
    w = (window or '3M').upper()
    if w.endswith('M'):
        n = w[:-1]
        return f"{n}mo"
    if w.endswith('Y'):
        n = w[:-1]
        return f"{n}y"
    if w.endswith('W'):
        # yfinance 不直接支持周，近似为月
        return '1mo'
    if w.endswith('D'):
        return '1mo'
    return '3mo'


def fetch_kline(ticker: str, window: str = '3M') -> List[Dict[str, Any]]:
    symbol = normalize_ticker(ticker)
    period = window_to_period(window)
    data = yf.download(symbol, period=period, interval='1d', progress=False, auto_adjust=False)
    if data is None or data.empty:
        return []
    data = data.reset_index()
    results: List[Dict[str, Any]] = []
    for _, row in data.iterrows():
        # yfinance 返回的列名大小写固定
        results.append({
            "date": str(row['Date'].date()) if hasattr(row['Date'], 'date') else str(row['Date']),
            "open": float(row['Open']),
            "high": float(row['High']),
            "low": float(row['Low']),
            "close": float(row['Close']),
            "volume": float(row['Volume']) if 'Volume' in data.columns else None,
        })
    return results


