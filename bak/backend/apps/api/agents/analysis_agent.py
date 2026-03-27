from __future__ import annotations

from typing import Any, Dict, List
from datetime import datetime
from .base import BaseAgent, AgentResult
from ..indicators import build_indicators


def _moving_average(values: List[float], window: int) -> List[float]:
    if window <= 0 or window > len(values):
        return []
    result: List[float] = []
    for i in range(window - 1, len(values)):
        result.append(sum(values[i - window + 1 : i + 1]) / window)
    return result


class AnalysisAgent(BaseAgent):
    name = "AnalysisAgent"

    def run(self, params: Dict[str, Any], context: Dict[str, Any]) -> AgentResult:
        ticker = params.get("ticker") or context.get("ticker", "UNKNOWN")
        prices = params.get("prices") or context.get("prices")
        if not prices:
            data = context.get("DataAgent", {}).get("data", {})
            prices = [p["close"] for p in data.get("prices", [])]
        
        closes = [p if isinstance(p, (int, float)) else p.get("close", 0) for p in prices]
        inds = build_indicators([
            {"close": float(c)} for c in closes
        ]) if closes else {}
        
        # 构建结构化分析结果
        analysis_result = self._build_structured_analysis(ticker, closes, inds)
        
        return AgentResult(name=self.name, success=True, data=analysis_result)
    
    def _build_structured_analysis(self, ticker: str, closes: List[float], indicators: Dict[str, Any]) -> Dict[str, Any]:
        """构建结构化的分析结果"""
        current_price = closes[-1] if closes else 0
        ma5 = (indicators.get("ma5") or [None])[-1] if indicators else None
        ma20 = (indicators.get("ma20") or [None])[-1] if indicators else None
        rsi = (indicators.get("rsi") or [None])[-1] if indicators else None
        macd = indicators.get("macd", {}) if indicators else {}
        
        # 技术面分析
        technical_analysis = self._analyze_technical_indicators(current_price, ma5, ma20, rsi, macd)
        
        # 趋势分析
        trend_analysis = self._analyze_trend(closes, ma5, ma20)
        
        # 风险评级
        risk_level = self._calculate_risk_level(rsi, macd, trend_analysis)
        
        return {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "current_price": current_price,
            "technical_indicators": {
                "ma5": ma5,
                "ma20": ma20,
                "rsi": rsi,
                "macd": macd
            },
            "technical_analysis": technical_analysis,
            "trend_analysis": trend_analysis,
            "risk_level": risk_level,
            "investment_suggestion": self._generate_investment_suggestion(technical_analysis, trend_analysis, risk_level),
            "key_points": self._extract_key_points(technical_analysis, trend_analysis)
        }
    
    def _analyze_technical_indicators(self, price: float, ma5: float, ma20: float, rsi: float, macd: Dict) -> Dict[str, Any]:
        """分析技术指标"""
        analysis = {
            "ma_signal": "neutral",
            "rsi_signal": "neutral", 
            "macd_signal": "neutral",
            "overall_signal": "neutral"
        }
        
        # MA分析
        if ma5 and ma20:
            if ma5 > ma20 and price > ma5:
                analysis["ma_signal"] = "bullish"
            elif ma5 < ma20 and price < ma5:
                analysis["ma_signal"] = "bearish"
        
        # RSI分析
        if rsi:
            if rsi > 70:
                analysis["rsi_signal"] = "overbought"
            elif rsi < 30:
                analysis["rsi_signal"] = "oversold"
            elif 40 <= rsi <= 60:
                analysis["rsi_signal"] = "neutral"
        
        # MACD分析
        if macd and "macd" in macd and "signal" in macd:
            macd_line = macd["macd"][-1] if macd["macd"] else 0
            signal_line = macd["signal"][-1] if macd["signal"] else 0
            if macd_line > signal_line:
                analysis["macd_signal"] = "bullish"
            elif macd_line < signal_line:
                analysis["macd_signal"] = "bearish"
        
        # 综合信号
        bullish_count = sum(1 for signal in [analysis["ma_signal"], analysis["rsi_signal"], analysis["macd_signal"]] 
                          if signal in ["bullish", "oversold"])
        bearish_count = sum(1 for signal in [analysis["ma_signal"], analysis["rsi_signal"], analysis["macd_signal"]] 
                          if signal in ["bearish", "overbought"])
        
        if bullish_count > bearish_count:
            analysis["overall_signal"] = "bullish"
        elif bearish_count > bullish_count:
            analysis["overall_signal"] = "bearish"
        
        return analysis
    
    def _analyze_trend(self, closes: List[float], ma5: float, ma20: float) -> Dict[str, Any]:
        """分析趋势"""
        if len(closes) < 2:
            return {"trend": "unknown", "strength": "weak"}
        
        # 短期趋势（基于最近5个交易日）
        recent_closes = closes[-5:] if len(closes) >= 5 else closes
        short_trend = "up" if recent_closes[-1] > recent_closes[0] else "down"
        
        # 长期趋势（基于MA）
        long_trend = "up" if ma5 and ma20 and ma5 > ma20 else "down" if ma5 and ma20 and ma5 < ma20 else "sideways"
        
        # 趋势强度
        price_change = (closes[-1] - closes[0]) / closes[0] * 100 if closes[0] != 0 else 0
        strength = "strong" if abs(price_change) > 5 else "moderate" if abs(price_change) > 2 else "weak"
        
        return {
            "short_term": short_trend,
            "long_term": long_trend,
            "strength": strength,
            "price_change_percent": round(price_change, 2)
        }
    
    def _calculate_risk_level(self, rsi: float, macd: Dict, trend_analysis: Dict) -> str:
        """计算风险等级"""
        risk_score = 0
        
        # RSI风险
        if rsi:
            if rsi > 80 or rsi < 20:
                risk_score += 2
            elif rsi > 70 or rsi < 30:
                risk_score += 1
        
        # 趋势风险
        if trend_analysis["strength"] == "strong":
            risk_score += 1
        
        # MACD风险
        if macd and "macd" in macd and "signal" in macd:
            macd_line = macd["macd"][-1] if macd["macd"] else 0
            signal_line = macd["signal"][-1] if macd["signal"] else 0
            if abs(macd_line - signal_line) > 0.1:  # 大幅背离
                risk_score += 1
        
        if risk_score >= 3:
            return "high"
        elif risk_score >= 1:
            return "medium"
        else:
            return "low"
    
    def _generate_investment_suggestion(self, technical_analysis: Dict, trend_analysis: Dict, risk_level: str) -> Dict[str, Any]:
        """生成投资建议"""
        signal = technical_analysis["overall_signal"]
        trend = trend_analysis["long_term"]
        
        if signal == "bullish" and trend == "up" and risk_level == "low":
            action = "buy"
            confidence = "high"
        elif signal == "bearish" and trend == "down" and risk_level == "low":
            action = "sell"
            confidence = "high"
        elif signal == "bullish" and risk_level in ["low", "medium"]:
            action = "buy"
            confidence = "medium"
        elif signal == "bearish" and risk_level in ["low", "medium"]:
            action = "sell"
            confidence = "medium"
        else:
            action = "hold"
            confidence = "low"
        
        return {
            "action": action,
            "confidence": confidence,
            "reasoning": f"基于{signal}技术信号、{trend}趋势和{risk_level}风险等级的综合判断"
        }
    
    def _extract_key_points(self, technical_analysis: Dict, trend_analysis: Dict) -> List[str]:
        """提取关键要点"""
        points = []
        
        if technical_analysis["ma_signal"] == "bullish":
            points.append("MA5上穿MA20，形成金叉信号")
        elif technical_analysis["ma_signal"] == "bearish":
            points.append("MA5下穿MA20，形成死叉信号")
        
        if technical_analysis["rsi_signal"] == "overbought":
            points.append("RSI指标显示超买，注意回调风险")
        elif technical_analysis["rsi_signal"] == "oversold":
            points.append("RSI指标显示超卖，可能存在反弹机会")
        
        if technical_analysis["macd_signal"] == "bullish":
            points.append("MACD金叉，上涨动能增强")
        elif technical_analysis["macd_signal"] == "bearish":
            points.append("MACD死叉，下跌动能增强")
        
        if trend_analysis["strength"] == "strong":
            points.append(f"趋势强度较高，价格变化{trend_analysis['price_change_percent']}%")
        
        return points


