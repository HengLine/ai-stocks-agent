#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@FileName: technical_agent.py
@Description: 技术分析智能体，负责分析股票的价格走势、成交量和技术指标
@Author: HengLine
@Time: 2025/11/10
"""

import json
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd
from ta import trend, momentum, volume, volatility

from hengline.agents.base_agent import BaseAgent, AgentConfig, AgentResult
from hengline.logger import debug, error, warning
# 从stock_manage统一获取数据
from hengline.stock.stock_manage import StockDataManager


class TechnicalAgent(BaseAgent):
    """技术面分析智能体"""

    def __init__(self, config: AgentConfig = None):
        """
        初始化技术面分析智能体
        
        Args:
            config: 智能体配置
        """
        super().__init__(config)

        # 初始化股票数据管理器
        self.stock_manager = StockDataManager()

        # 技术分析关键维度
        self.analysis_dimensions = [
            "trend_analysis",  # 趋势分析
            "momentum_analysis",  # 动量分析
            "volume_analysis",  # 量价分析
            "volatility_analysis",  # 波动率分析
            "pattern_recognition",  # 形态识别
            "support_resistance"  # 支撑阻力
        ]

    def analyze(self, stock_code: str, time_range: str = "1y", **kwargs) -> AgentResult:
        """
        执行技术面分析
        
        Args:
            stock_code: 股票代码
            time_range: 时间范围
            **kwargs: 其他参数
            
        Returns:
            AgentResult: 分析结果
        """
        try:
            debug(f"开始对股票 {stock_code} 进行技术面分析")

            # 获取历史价格数据
            price_data = self._get_price_data(stock_code, time_range)

            # 计算技术指标
            technical_indicators = self._calculate_technical_indicators(price_data)

            # 检索相关知识库信息
            knowledge = self._retrieve_knowledge(f"股票技术分析 MACD RSI {stock_code}")

            # 生成分析提示词
            prompt = self._generate_analysis_prompt(stock_code, price_data, technical_indicators)

            # 使用LLM生成分析
            llm_analysis = self._generate_analysis(prompt, knowledge)

            # 结构化分析结果
            result = self._structure_result(stock_code, llm_analysis, technical_indicators, price_data)

            debug(f"股票 {stock_code} 技术面分析完成")

            return AgentResult(
                agent_name=self.agent_name,
                success=True,
                result=result,
                confidence_score=result.get("confidence_score", 0.85)
            )

        except Exception as e:
            error(f"技术面分析失败: {str(e)}")
            return AgentResult(
                agent_name=self.agent_name,
                success=False,
                result=self.get_result_template(),
                error_message=str(e),
                confidence_score=0.0
            )

    def _get_price_data(self, stock_code: str, time_range: str) -> pd.DataFrame:
        """
        获取股票历史价格数据
        
        Args:
            stock_code: 股票代码
            time_range: 时间范围
            
        Returns:
            pd.DataFrame: 价格数据
        """
        try:
            # 使用统一的数据管理器获取价格数据
            price_data = self.stock_manager.get_stock_price_data(stock_code, period=time_range, interval="1d")

            # 确保数据不为空
            if price_data.empty:
                warning(f"无法获取 {stock_code} 的价格数据或数据为空")
                raise ValueError(f"无法获取 {stock_code} 的价格数据")

            return price_data

        except Exception as e:
            error(f"获取价格数据失败: {str(e)}")
            raise

    def _calculate_technical_indicators(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        计算技术指标
        
        Args:
            price_data: 价格数据
            
        Returns:
            Dict[str, Any]: 技术指标
        """
        indicators = {}

        try:
            # 移动平均线
            indicators["moving_averages"] = {
                "ma_50": float(price_data['Close'].rolling(window=50).mean().iloc[-1]),
                "ma_200": float(price_data['Close'].rolling(window=200).mean().iloc[-1]),
                "ma_50_trend": "rising" if self._is_rising(price_data['Close'].rolling(window=50).mean(), 10) else "falling",
                "ma_200_trend": "rising" if self._is_rising(price_data['Close'].rolling(window=200).mean(), 20) else "falling"
            }

            # MACD
            macd = trend.MACD(price_data['Close'])
            indicators["macd"] = {
                "macd": float(macd.macd().iloc[-1]),
                "signal": float(macd.macd_signal().iloc[-1]),
                "histogram": float(macd.macd_diff().iloc[-1]),
                "signal_type": self._get_macd_signal(macd)
            }

            # RSI
            rsi = momentum.RSIIndicator(price_data['Close'])
            indicators["rsi"] = {
                "rsi_14": float(rsi.rsi().iloc[-1]),
                "oversold": float(rsi.rsi().iloc[-1]) < 30,
                "overbought": float(rsi.rsi().iloc[-1]) > 70
            }

            # 布林带
            bb = volatility.BollingerBands(price_data['Close'])
            indicators["bollinger_bands"] = {
                "upper": float(bb.bollinger_hband().iloc[-1]),
                "middle": float(bb.bollinger_mavg().iloc[-1]),
                "lower": float(bb.bollinger_lband().iloc[-1]),
                "width": float(bb.bollinger_wband().iloc[-1])
            }

            # 成交量分析
            indicators["volume"] = {
                "current_volume": float(price_data['Volume'].iloc[-1]),
                "avg_volume_20": float(price_data['Volume'].rolling(window=20).mean().iloc[-1]),
                "volume_ratio": float(price_data['Volume'].iloc[-1] / price_data['Volume'].rolling(window=20).mean().iloc[-1])
            }

            # OBV
            obv = volume.OnBalanceVolumeIndicator(price_data['Close'], price_data['Volume'])
            indicators["obv"] = {
                "obv": float(obv.on_balance_volume().iloc[-1]),
                "obv_trend": "rising" if self._is_rising(obv.on_balance_volume(), 20) else "falling"
            }

            # 价格趋势
            indicators["price_trend"] = {
                "current_price": float(price_data['Close'].iloc[-1]),
                "price_change_1m": float(self._calculate_percentage_change(price_data['Close'], 30)),
                "price_change_3m": float(self._calculate_percentage_change(price_data['Close'], 90)),
                "price_change_6m": float(self._calculate_percentage_change(price_data['Close'], 180)),
                "price_change_1y": float(self._calculate_percentage_change(price_data['Close'], 365))
            }

            # 支撑阻力位（简单实现）
            recent_highs = price_data['High'].rolling(window=50).max()
            recent_lows = price_data['Low'].rolling(window=50).min()
            indicators["support_resistance"] = {
                "resistance": float(recent_highs.iloc[-1]),
                "support": float(recent_lows.iloc[-1]),
                "distance_to_resistance": float((recent_highs.iloc[-1] / price_data['Close'].iloc[-1] - 1) * 100),
                "distance_to_support": float((price_data['Close'].iloc[-1] / recent_lows.iloc[-1] - 1) * 100)
            }

        except Exception as e:
            error(f"计算技术指标失败: {str(e)}")

        return indicators

    def _is_rising(self, series: pd.Series, window: int = 20) -> bool:
        """
        判断序列是否上升趋势
        
        Args:
            series: 数据序列
            window: 窗口大小
            
        Returns:
            bool: 是否上升趋势
        """
        try:
            # 计算线性回归斜率
            if len(series) < window:
                return False

            recent = series.iloc[-window:]
            x = np.arange(len(recent))
            slope = np.polyfit(x, recent.values, 1)[0]
            return slope > 0
        except:
            return False

    def _get_macd_signal(self, macd) -> str:
        """
        获取MACD信号
        
        Args:
            macd: MACD对象
            
        Returns:
            str: 信号类型
        """
        try:
            current_macd = macd.macd().iloc[-1]
            current_signal = macd.macd_signal().iloc[-1]
            previous_macd = macd.macd().iloc[-2]
            previous_signal = macd.macd_signal().iloc[-2]

            # 金叉
            if previous_macd < previous_signal and current_macd > current_signal:
                return "golden_cross"
            # 死叉
            elif previous_macd > previous_signal and current_macd < current_signal:
                return "death_cross"
            # 多头
            elif current_macd > current_signal and current_macd > 0:
                return "bullish"
            # 空头
            elif current_macd < current_signal and current_macd < 0:
                return "bearish"
            else:
                return "neutral"
        except:
            return "neutral"

    def _calculate_percentage_change(self, series: pd.Series, days: int) -> float:
        """
        计算百分比变化
        
        Args:
            series: 价格序列
            days: 天数
            
        Returns:
            float: 百分比变化
        """
        try:
            if len(series) < days:
                days = len(series) - 1
                if days <= 0:
                    return 0

            return ((series.iloc[-1] - series.iloc[-days]) / series.iloc[-days]) * 100
        except:
            return 0

    def _generate_analysis_prompt(self, stock_code: str, price_data: pd.DataFrame, technical_indicators: Dict[str, Any]) -> str:
        """
        生成分析提示词
        
        Args:
            stock_code: 股票代码
            price_data: 价格数据
            technical_indicators: 技术指标
            
        Returns:
            str: 分析提示词
        """
        # 获取最近的价格数据
        latest_price = float(price_data['Close'].iloc[-1])

        # 导入langchain相关组件和提示词管理器
        from langchain_core.prompts import ChatPromptTemplate
        from hengline.prompts.prompt_manage import get_prompt

        # 从提示词管理器获取提示模板
        template = get_prompt('technical_agent', 'analysis')

        # 如果获取失败，使用备用模板
        if not template:
            template = """
请对股票代码 {{stock_code}} 进行全面的技术面分析。

当前价格: {{latest_price}}

技术指标数据：
{{technical_indicators}}

请提供详细分析。
"""

        # 创建提示模板实例
        prompt_template = ChatPromptTemplate.from_template(template)

        # 格式化并返回提示内容
        prompt = prompt_template.format(
            stock_code=stock_code,
            latest_price=latest_price,
            technical_indicators=json.dumps(technical_indicators, indent=2)
        )

        return prompt

    def _structure_result(self, stock_code: str, llm_analysis: Dict[str, Any], technical_indicators: Dict[str, Any], price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        结构化分析结果
        
        Args:
            stock_code: 股票代码
            llm_analysis: LLM生成的分析
            technical_indicators: 技术指标
            price_data: 价格数据
            
        Returns:
            Dict[str, Any]: 结构化的结果
        """
        result = self.get_result_template()
        result.update({
            "stock_code": stock_code,
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_price": float(price_data['Close'].iloc[-1]),
            "key_findings": llm_analysis.get("key_findings", []),
            "detailed_analysis": llm_analysis.get("detailed_analysis", {}),
            "short_term_outlook": llm_analysis.get("short_term_outlook", ""),
            "medium_term_outlook": llm_analysis.get("medium_term_outlook", ""),
            "signal_strength": llm_analysis.get("signal_strength", "neutral"),
            "key_price_levels": llm_analysis.get("key_price_levels", {}),
            "confidence_score": llm_analysis.get("confidence_score", 0.85),
            "technical_summary": {
                "trend": technical_indicators.get("price_trend", {}),
                "momentum": {
                    "rsi": technical_indicators.get("rsi", {}).get("rsi_14", 0),
                    "macd_signal": technical_indicators.get("macd", {}).get("signal_type", "")
                },
                "volume": technical_indicators.get("volume", {})
            }
        })

        # 验证结果
        if not self.validate_result(result):
            warning("技术面分析结果验证失败，使用默认结构")
            result = self.get_result_template()

        return result
