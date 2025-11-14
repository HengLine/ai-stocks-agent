#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@FileName: fund_flow_agent.py
@Description: 资金流向分析智能体，负责分析股票的资金流入流出情况
@Author: HengLine
@Time: 2025/11/10
"""

import json
from datetime import datetime
from typing import Dict, Any
import yfinance as yf
from hengline.agents.base_agent import BaseAgent, AgentConfig, AgentResult
from hengline.logger import debug, error, warning
# 从stock_manage统一获取数据
from hengline.stock.stock_manage import StockDataManager


class FundFlowAgent(BaseAgent):
    """股东与资金流分析智能体"""

    def __init__(self, config: AgentConfig = None):
        """
        初始化股东与资金流分析智能体
        
        Args:
            config: 智能体配置
        """
        super().__init__(config)

        # 初始化股票数据管理器
        self.stock_manager = StockDataManager()

        # 资金流分析关键维度
        self.analysis_dimensions = [
            "institutional_holdings",  # 机构持仓分析
            "major_shareholder_changes",  # 大股东变动
            "money_flow",  # 资金流向
            "volume_analysis",  # 成交量分析
            "insider_transactions",  # 内部人交易
            "foreign_investment_flow"  # 外资流向
        ]

        # 资金流向分类阈值
        self.money_flow_thresholds = {
            "strong_inflow": 500000000,  # 强势流入 (>5亿)
            "moderate_inflow": 100000000,  # 中度流入 (1-5亿)
            "weak_inflow": 10000000,  # 弱势流入 (1000万-1亿)
            "neutral": 0,  # 平衡
            "weak_outflow": -10000000,  # 弱势流出 (-1亿到-1000万)
            "moderate_outflow": -100000000,  # 中度流出 (-5亿到-1亿)
            "strong_outflow": -500000000  # 强势流出 (<-5亿)
        }

    def analyze(self, stock_code: str, time_range: str = "3m", **kwargs) -> AgentResult:
        """
        执行股东与资金流分析
        
        Args:
            stock_code: 股票代码
            time_range: 时间范围
            **kwargs: 其他参数
            
        Returns:
            AgentResult: 分析结果
        """
        try:
            debug(f"开始对股票 {stock_code} 进行股东与资金流分析")

            # 获取股票基本信息
            stock_info = self._get_stock_info(stock_code)

            # 获取价格和交易量数据
            price_data = self._get_price_data(stock_code, time_range)

            # 获取机构持仓数据
            institutional_data = self._get_institutional_data(stock_code)

            # 获取内部人交易数据
            insider_data = self._get_insider_transactions(stock_code)

            # 计算资金流向指标
            money_flow_indicators = self._calculate_money_flow(price_data)

            # 检索相关知识库信息
            knowledge = self._retrieve_knowledge(f"机构持仓分析 资金流向 成交量分析")

            # 生成分析提示词
            prompt = self._generate_analysis_prompt(
                stock_code, stock_info, price_data, institutional_data,
                insider_data, money_flow_indicators
            )

            # 使用LLM生成分析
            llm_analysis = self._generate_analysis(prompt, knowledge)

            # 结构化分析结果
            result = self._structure_result(
                stock_code, llm_analysis, stock_info, money_flow_indicators,
                institutional_data
            )

            debug(f"股票 {stock_code} 股东与资金流分析完成")

            return AgentResult(
                agent_name=self.agent_name,
                success=True,
                result=result,
                confidence_score=result.get("confidence_score", 0.85)
            )

        except Exception as e:
            error(f"股东与资金流分析失败: {str(e)}")
            return AgentResult(
                agent_name=self.agent_name,
                success=False,
                result=self.get_result_template(),
                error_message=str(e),
                confidence_score=0.0
            )

    def _get_stock_info(self, stock_code: str) -> Dict[str, Any]:
        """
        获取股票基本信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Dict[str, Any]: 股票基本信息
        """
        try:
            # 从stock_manage获取股票信息
            stock_info = self.stock_manager.get_stock_info(stock_code)
            return {
                "name": stock_info.get("name", ""),
                "sector": stock_info.get("sector", ""),
                "industry": stock_info.get("industry", ""),
                "market_cap": stock_info.get("market_cap", 0),
                "shares_outstanding": stock_info.get("shares_outstanding", 0)
            }
        except Exception as e:
            error(f"获取股票信息失败: {str(e)}")
            return {}

    def _get_price_data(self, stock_code: str, time_range: str = "3m") -> Dict[str, Any]:
        """
        获取股票价格和交易量数据
        
        Args:
            stock_code: 股票代码
            time_range: 时间范围
            
        Returns:
            Dict[str, Any]: 价格和交易量数据
        """
        try:
            # 从stock_manage获取价格数据
            price_data = self.stock_manager.get_stock_price_data(stock_code, time_range)
            return price_data
        except Exception as e:
            error(f"获取价格数据失败: {str(e)}")
            return {"market_data": [], "volume_stats": {}}

    def _get_institutional_data(self, stock_code: str) -> Dict[str, Any]:
        """
        获取机构持仓数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Dict[str, Any]: 机构持仓数据
        """
        institutional_data = {
            "major_holders": [],
            "institutional_holders": [],
            "top_holders_summary": {
                "total_institutional_holders": 0,
                "total_shares_held": 0,
                "percent_outstanding": 0
            }
        }

        try:
            stock = yf.Ticker(stock_code)

            # 获取主要持有者
            major_holders = stock.major_holders
            if not major_holders.empty:
                major_holders_list = []
                for idx, row in major_holders.iterrows():
                    major_holders_list.append({
                        "holder_type": row[0],
                        "percentage": float(row[1].strip('%')) / 100 if isinstance(row[1], str) else row[1]
                    })
                institutional_data["major_holders"] = major_holders_list

            # 获取机构持有者
            institutional_holders = stock.institutional_holders
            if not institutional_holders.empty:
                institutional_holders_list = []
                total_shares = 0

                for idx, row in institutional_holders.iterrows():
                    shares = int(row["Shares"])
                    total_shares += shares
                    institutional_holders_list.append({
                        "holder": row["Holder"],
                        "shares": shares,
                        "date_reported": row["Date Reported"].strftime("%Y-%m-%d"),
                        "percentage": float(row["% Out"])
                    })

                institutional_data["institutional_holders"] = institutional_holders_list[:10]  # 取前10个
                institutional_data["top_holders_summary"] = {
                    "total_institutional_holders": len(institutional_holders_list),
                    "total_shares_held": total_shares,
                    "percent_outstanding": sum([h["percentage"] for h in institutional_holders_list])
                }

        except Exception as e:
            error(f"获取机构持仓数据失败: {str(e)}")

        return institutional_data

    def _get_insider_transactions(self, stock_code: str) -> Dict[str, Any]:
        """
        获取内部人交易数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Dict[str, Any]: 内部人交易数据
        """
        insider_data = {
            "transactions": [],
            "summary": {
                "total_buys": 0,
                "total_sells": 0,
                "net_volume": 0,
                "recent_trend": ""
            }
        }

        try:
            stock = yf.Ticker(stock_code)
            insider_transactions = stock.insider_transactions

            if not insider_transactions.empty:
                transactions = []
                total_buys = 0
                total_sells = 0

                for idx, row in insider_transactions.iterrows():
                    transaction_type = row["Transaction"]
                    shares = int(row["Shares"])

                    if "Buy" in transaction_type:
                        total_buys += shares
                    elif "Sell" in transaction_type:
                        total_sells += shares

                    transactions.append({
                        "filer": row["Filer"],
                        "relationship": row["Relationship"],
                        "transaction_type": transaction_type,
                        "cost": float(row["Cost"]),
                        "shares": shares,
                        "value": float(row["Value"]),
                        "acquisition_date": row["Acquistion Date"].strftime("%Y-%m-%d")
                    })

                insider_data["transactions"] = transactions[:10]  # 取最近10条
                insider_data["summary"] = {
                    "total_buys": total_buys,
                    "total_sells": total_sells,
                    "net_volume": total_buys - total_sells,
                    "recent_trend": "buying" if total_buys > total_sells else "selling" if total_sells > total_buys else "neutral"
                }
        except Exception as e:
            error(f"获取内部人交易数据失败: {str(e)}")

        return insider_data

    def _calculate_money_flow(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算资金流向指标
        
        Args:
            price_data: 价格和交易量数据
            
        Returns:
            Dict[str, Any]: 资金流向指标
        """
        money_flow = {
            "positive_volume_index": [],
            "negative_volume_index": [],
            "on_balance_volume": [],
            "money_flow_index": 0,
            "flow_classification": "neutral",
            "flow_trend": "stable"
        }

        try:
            market_data = price_data.get("market_data", [])
            if len(market_data) < 14:  # MFI需要至少14天数据
                return money_flow

            # 计算OBV (On-Balance Volume)
            obv = [0]
            for i in range(1, len(market_data)):
                if market_data[i]["close"] > market_data[i - 1]["close"]:
                    obv.append(obv[-1] + market_data[i]["volume"])
                elif market_data[i]["close"] < market_data[i - 1]["close"]:
                    obv.append(obv[-1] - market_data[i]["volume"])
                else:
                    obv.append(obv[-1])
            money_flow["on_balance_volume"] = obv

            # 计算MFI (Money Flow Index)
            typical_prices = []
            raw_money_flow = []

            for data in market_data:
                typical_price = (data["high"] + data["low"] + data["close"]) / 3
                typical_prices.append(typical_price)
                raw_money_flow.append(typical_price * data["volume"])

            # 计算正、负资金流
            positive_money_flow = []
            negative_money_flow = []

            for i in range(1, len(typical_prices)):
                if typical_prices[i] > typical_prices[i - 1]:
                    positive_money_flow.append(raw_money_flow[i])
                    negative_money_flow.append(0)
                elif typical_prices[i] < typical_prices[i - 1]:
                    positive_money_flow.append(0)
                    negative_money_flow.append(raw_money_flow[i])
                else:
                    positive_money_flow.append(0)
                    negative_money_flow.append(0)

            # 计算14天MFI
            period = 14
            mfi_values = []

            for i in range(period - 1, len(positive_money_flow)):
                pos_flow_sum = sum(positive_money_flow[i - period + 1:i + 1])
                neg_flow_sum = sum(negative_money_flow[i - period + 1:i + 1])

                if neg_flow_sum == 0:
                    mfi = 100
                else:
                    money_ratio = pos_flow_sum / neg_flow_sum
                    mfi = 100 - (100 / (1 + money_ratio))

                mfi_values.append(mfi)

            if mfi_values:
                money_flow["money_flow_index"] = round(mfi_values[-1], 2)

            # 确定资金流向分类
            if len(money_flow["on_balance_volume"]) >= 2:
                obv_change = money_flow["on_balance_volume"][-1] - money_flow["on_balance_volume"][-2]

                # 基于OBV变化量和MFI值确定流向分类
                if obv_change > self.money_flow_thresholds["strong_inflow"] and money_flow["money_flow_index"] > 80:
                    money_flow["flow_classification"] = "strong_inflow"
                elif obv_change > self.money_flow_thresholds["moderate_inflow"] and money_flow["money_flow_index"] > 70:
                    money_flow["flow_classification"] = "moderate_inflow"
                elif obv_change > self.money_flow_thresholds["weak_inflow"]:
                    money_flow["flow_classification"] = "weak_inflow"
                elif obv_change < self.money_flow_thresholds["strong_outflow"] and money_flow["money_flow_index"] < 20:
                    money_flow["flow_classification"] = "strong_outflow"
                elif obv_change < self.money_flow_thresholds["moderate_outflow"] and money_flow["money_flow_index"] < 30:
                    money_flow["flow_classification"] = "moderate_outflow"
                elif obv_change < self.money_flow_thresholds["weak_outflow"]:
                    money_flow["flow_classification"] = "weak_outflow"

                # 确定趋势
                if len(money_flow["on_balance_volume"]) >= 10:
                    recent_obv = money_flow["on_balance_volume"][-10:]
                    if recent_obv[-1] > recent_obv[0] * 1.1:
                        money_flow["flow_trend"] = "increasing_rapidly"
                    elif recent_obv[-1] > recent_obv[0] * 1.05:
                        money_flow["flow_trend"] = "increasing"
                    elif recent_obv[-1] < recent_obv[0] * 0.9:
                        money_flow["flow_trend"] = "decreasing_rapidly"
                    elif recent_obv[-1] < recent_obv[0] * 0.95:
                        money_flow["flow_trend"] = "decreasing"

        except Exception as e:
            error(f"计算资金流向指标失败: {str(e)}")

        return money_flow

    def _generate_analysis_prompt(self, stock_code: str, stock_info: Dict[str, Any],
                                  price_data: Dict[str, Any], institutional_data: Dict[str, Any],
                                  insider_data: Dict[str, Any], money_flow_indicators: Dict[str, Any]) -> str:
        """
        生成分析提示词
        
        Args:
            stock_code: 股票代码
            stock_info: 股票基本信息
            price_data: 价格和交易量数据
            institutional_data: 机构持仓数据
            insider_data: 内部人交易数据
            money_flow_indicators: 资金流向指标
            
        Returns:
            str: 分析提示词
        """
        # 导入langchain相关组件和提示词管理器
        from langchain_core.prompts import ChatPromptTemplate
        from hengline.prompts.prompt_manage import get_prompt

        # 从提示词管理器获取提示模板
        template = get_prompt('fund_flow_agent', 'analysis')

        # 如果获取失败，使用备用模板
        if not template:
            template = """
请对股票代码 {{stock_code}} 的股东结构与资金流向进行全面分析。

股票基本信息：
{{stock_info}}

价格和交易量数据：
{{volume_stats}}

请提供资金流向分析和潜在影响评估。
"""

        # 创建提示模板实例
        prompt_template = ChatPromptTemplate.from_template(template)

        # 格式化并返回提示内容
        prompt = prompt_template.format(
            stock_code=stock_code,
            stock_info=json.dumps(stock_info, indent=2, ensure_ascii=False),
            volume_stats=json.dumps(price_data.get("volume_stats", {}), indent=2, ensure_ascii=False),
            institutional_data=json.dumps(institutional_data, indent=2, ensure_ascii=False),
            insider_data=json.dumps(insider_data, indent=2, ensure_ascii=False),
            money_flow_indicators=json.dumps(money_flow_indicators, indent=2, ensure_ascii=False)
        )

        return prompt

    def _structure_result(self, stock_code: str, llm_analysis: Dict[str, Any],
                          stock_info: Dict[str, Any], money_flow_indicators: Dict[str, Any],
                          institutional_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        结构化分析结果
        
        Args:
            stock_code: 股票代码
            llm_analysis: LLM生成的分析
            stock_info: 股票基本信息
            money_flow_indicators: 资金流向指标
            institutional_data: 机构持仓数据
            
        Returns:
            Dict[str, Any]: 结构化的结果
        """
        result = self.get_result_template()
        result.update({
            "stock_code": stock_code,
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "company_name": stock_info.get("name", ""),
            "key_findings": llm_analysis.get("key_findings", []),
            "detailed_analysis": llm_analysis.get("detailed_analysis", {}),
            "fund_flow_summary": llm_analysis.get("fund_flow_summary", {}),
            "potential_impact": llm_analysis.get("potential_impact", ""),
            "alert_signals": llm_analysis.get("alert_signals", []),
            "institutional_behavior": llm_analysis.get("institutional_behavior", {}),
            "confidence_score": llm_analysis.get("confidence_score", 0.85),
            "key_metrics": {
                "money_flow_index": money_flow_indicators.get("money_flow_index", 0),
                "flow_classification": money_flow_indicators.get("flow_classification", "neutral"),
                "institutional_ownership": institutional_data.get("top_holders_summary", {}).get("percent_outstanding", 0)
            }
        })

        # 验证结果
        if not self.validate_result(result):
            warning("股东与资金流分析结果验证失败，使用默认结构")
            result = self.get_result_template()

        return result
