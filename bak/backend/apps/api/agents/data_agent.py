from __future__ import annotations

from typing import Any, Dict
from .base import BaseAgent, AgentResult
from ..providers.market_provider import fetch_kline


class DataAgent(BaseAgent):
    name = "DataAgent"

    def run(self, params: Dict[str, Any], context: Dict[str, Any]) -> AgentResult:
        ticker = params.get("ticker") or context.get("ticker")
        time_window = params.get("time_window") or context.get("time_window", "3M")
        klines = fetch_kline(ticker=ticker, window=time_window) if ticker else []
        data = {"ticker": ticker, "time_window": time_window, "prices": klines}
        return AgentResult(name=self.name, success=True, data=data)


