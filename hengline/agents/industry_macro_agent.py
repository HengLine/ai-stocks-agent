#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@FileName: industry_macro_agent.py
@Description: 行业宏观分析智能体，负责分析行业发展趋势和宏观经济环境对股票的影响
@Author: HengLine
@Time: 2025/11/10
"""

import json
from datetime import datetime
from typing import Dict, Any
import yfinance as yf
# 从stock_manage统一获取数据
from hengline.stock.stock_manage import StockDataManager

from hengline.agents.base_agent import BaseAgent, AgentConfig, AgentResult
from hengline.logger import debug, error, warning


class IndustryMacroAgent(BaseAgent):
    """行业与宏观环境分析智能体"""

    def __init__(self, config: AgentConfig = None):
        """
        初始化行业与宏观环境分析智能体
        
        Args:
            config: 智能体配置
        """
        if config is None:
            config = AgentConfig(
                agent_name="IndustryMacroAgent",
                description="专业的行业与宏观经济分析专家，擅长分析行业趋势、宏观经济因素、政策环境和市场周期对投资的影响",
                model_name="gpt-4",
                temperature=0.2,
                max_tokens=2000
            )
        super().__init__(config)

        # 初始化股票数据管理器
        self.stock_manager = StockDataManager()

        # 行业与宏观分析关键维度
        self.analysis_dimensions = [
            "industry_outlook",  # 行业前景
            "macro_economy",  # 宏观经济
            "policy_environment",  # 政策环境
            "industry_competition",  # 行业竞争格局
            "industry_cycle",  # 行业周期
            "market_trends"  # 市场趋势
        ]

        # 主要行业ETF代码，用于行业表现比较
        self.industry_etfs = {
            "Technology": "XLK",
            "Financial": "XLF",
            "Healthcare": "XLV",
            "Consumer Discretionary": "XLY",
            "Consumer Staples": "XLP",
            "Energy": "XLE",
            "Utilities": "XLU",
            "Materials": "XLB",
            "Industrials": "XLI",
            "Communication Services": "XLC",
            "Real Estate": "XLRE"
        }

    def analyze(self, stock_code: str, time_range: str = "1y", **kwargs) -> AgentResult:
        """
        执行行业与宏观环境分析
        
        Args:
            stock_code: 股票代码
            time_range: 时间范围
            **kwargs: 其他参数
            
        Returns:
            AgentResult: 分析结果
        """
        try:
            debug(f"开始对股票 {stock_code} 进行行业与宏观环境分析")

            # 获取股票基本信息
            stock_info = self._get_stock_info(stock_code)
            sector = stock_info.get("sector", "")
            industry = stock_info.get("industry", "")

            # 获取行业数据
            industry_data = self._get_industry_data(sector, industry, time_range)

            # 获取宏观经济指标
            macro_data = self._get_macro_indicators()

            # 检索相关知识库信息
            knowledge = self._retrieve_knowledge(f"行业分析 {sector} {industry} 宏观经济")

            # 生成分析提示词
            prompt = self._generate_analysis_prompt(
                stock_code, stock_info, industry_data, macro_data
            )

            # 使用LLM生成分析
            llm_analysis = self._generate_analysis(prompt, knowledge)

            # 结构化分析结果
            result = self._structure_result(
                stock_code, llm_analysis, stock_info, industry_data, macro_data
            )

            debug(f"股票 {stock_code} 行业与宏观环境分析完成")

            return AgentResult(
                agent_name=self.agent_name,
                success=True,
                result=result,
                confidence_score=result.get("confidence_score", 0.85)
            )

        except Exception as e:
            error(f"行业与宏观环境分析失败: {str(e)}")
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
                "description": stock_info.get("description", ""),
                "country": stock_info.get("country", ""),
                "market_cap": stock_info.get("market_cap", 0),
                "peers": stock_info.get("peers", "")
            }
        except Exception as e:
            error(f"获取股票信息失败: {str(e)}")
            return {}

    def _get_industry_data(self, sector: str, industry: str, time_range: str) -> Dict[str, Any]:
        """
        获取行业相关数据
        
        Args:
            sector: 行业板块
            industry: 具体行业
            time_range: 时间范围
            
        Returns:
            Dict[str, Any]: 行业数据
        """
        industry_data = {
            "sector": sector,
            "industry": industry,
            "sector_performance": {},
            "industry_outlook": "",
            "growth_drivers": [],
            "challenges": []
        }

        try:
            # 获取行业ETF表现（如果有对应的ETF）
            if sector in self.industry_etfs:
                etf_code = self.industry_etfs[sector]
                etf = yf.Ticker(etf_code)

                # 根据时间范围确定period参数
                period_mapping = {
                    "1d": "1d",
                    "1w": "1wk",
                    "1m": "1mo",
                    "3m": "3mo",
                    "6m": "6mo",
                    "1y": "1y",
                    "2y": "2y",
                    "5y": "5y",
                    "10y": "10y"
                }
                period = period_mapping.get(time_range, "1y")

                hist = etf.history(period=period)
                if not hist.empty:
                    latest_price = hist['Close'].iloc[-1]
                    earliest_price = hist['Close'].iloc[0]
                    performance = ((latest_price - earliest_price) / earliest_price) * 100

                    industry_data["sector_performance"] = {
                        "etf_code": etf_code,
                        "current_price": float(latest_price),
                        "period_performance": float(performance),
                        "volatility": float(hist['Close'].pct_change().std() * 100)
                    }

        except Exception as e:
            error(f"获取行业数据失败: {str(e)}")

        return industry_data

    def _get_macro_indicators(self) -> Dict[str, Any]:
        """
        获取宏观经济指标
        
        Returns:
            Dict[str, Any]: 宏观经济指标
        """
        macro_data = {
            "interest_rates": {},
            "inflation": {},
            "economic_growth": {},
            "market_sentiment": {},
            "key_macro_trends": []
        }

        try:
            # 获取主要市场指数表现作为情绪指标
            indices = {
                "SP500": "^GSPC",
                "NASDAQ": "^IXIC",
                "DOW": "^DJI"
            }

            index_performances = {}
            for name, code in indices.items():
                try:
                    index = yf.Ticker(code)
                    hist = index.history(period="1m")
                    if not hist.empty:
                        latest_price = hist['Close'].iloc[-1]
                        earliest_price = hist['Close'].iloc[0]
                        performance = ((latest_price - earliest_price) / earliest_price) * 100
                        index_performances[name] = {
                            "price": float(latest_price),
                            "1m_performance": float(performance)
                        }
                except:
                    pass

            macro_data["market_sentiment"]["major_indices"] = index_performances

            # 这里可以扩展获取更多宏观数据
            # 例如通过API获取利率、通胀率等

        except Exception as e:
            error(f"获取宏观经济数据失败: {str(e)}")

        return macro_data

    def _generate_analysis_prompt(self, stock_code: str, stock_info: Dict[str, Any],
                                  industry_data: Dict[str, Any], macro_data: Dict[str, Any]) -> str:
        """
        生成分析提示词
        
        Args:
            stock_code: 股票代码
            stock_info: 股票基本信息
            industry_data: 行业数据
            macro_data: 宏观经济数据
            
        Returns:
            str: 分析提示词
        """
        # 导入langchain相关组件和提示词管理器
        from langchain_core.prompts import ChatPromptTemplate
        from hengline.prompts.prompt_manage import get_prompt

        # 从提示词管理器获取提示模板
        template = get_prompt('industry_macro_agent', 'analysis')

        # 如果获取失败，使用备用模板
        if not template:
            template = """
请对股票代码 {{stock_code}} 的行业与宏观环境进行全面分析。

股票基本信息：
{{stock_info}}

行业数据：
{{industry_data}}

宏观经济数据：
{{macro_data}}

请提供详细分析。
"""

        # 创建提示模板实例
        prompt_template = ChatPromptTemplate.from_template(template)

        # 格式化并返回提示内容
        prompt = prompt_template.format(
            stock_code=stock_code,
            stock_info=json.dumps(stock_info, indent=2, ensure_ascii=False),
            industry_data=json.dumps(industry_data, indent=2, ensure_ascii=False),
            macro_data=json.dumps(macro_data, indent=2, ensure_ascii=False)
        )

        return prompt

    def _structure_result(self, stock_code: str, llm_analysis: Dict[str, Any],
                          stock_info: Dict[str, Any], industry_data: Dict[str, Any],
                          macro_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        结构化分析结果
        
        Args:
            stock_code: 股票代码
            llm_analysis: LLM生成的分析
            stock_info: 股票基本信息
            industry_data: 行业数据
            macro_data: 宏观经济数据
            
        Returns:
            Dict[str, Any]: 结构化的结果
        """
        result = self.get_result_template()
        result.update({
            "stock_code": stock_code,
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "company_name": stock_info.get("name", ""),
            "sector": stock_info.get("sector", ""),
            "industry": stock_info.get("industry", ""),
            "key_findings": llm_analysis.get("key_findings", []),
            "detailed_analysis": llm_analysis.get("detailed_analysis", {}),
            "industry_outlook": llm_analysis.get("industry_outlook", ""),
            "macro_impact": llm_analysis.get("macro_impact", {}),
            "opportunities": llm_analysis.get("opportunities", []),
            "threats": llm_analysis.get("threats", []),
            "confidence_score": llm_analysis.get("confidence_score", 0.85),
            "industry_summary": {
                "sector_performance": industry_data.get("sector_performance", {}),
                "market_sentiment": macro_data.get("market_sentiment", {})
            }
        })

        # 验证结果
        if not self.validate_result(result):
            warning("行业与宏观环境分析结果验证失败，使用默认结构")
            result = self.get_result_template()

        return result
