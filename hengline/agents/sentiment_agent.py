#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@FileName: sentiment_agent.py
@Description: 市场情绪分析智能体，负责分析股票相关的市场情绪和舆情数据
@Author: HengLine
@Time: 2025/11/10
"""

import json
from datetime import datetime
from typing import Dict, Any, List

# 从stock_manage统一获取数据
from hengline.stock.stock_manage import StockDataManager

from hengline.agents.base_agent import BaseAgent, AgentConfig, AgentResult
from hengline.logger import debug, error, warning


class SentimentAgent(BaseAgent):
    """舆情与情绪分析智能体"""

    def __init__(self, config: AgentConfig = None):
        """
        初始化舆情与情绪分析智能体
        
        Args:
            config: 智能体配置
        """
        super().__init__(config)

        # 初始化股票数据管理器
        self.stock_manager = StockDataManager()

        # 舆情与情绪分析关键维度
        self.analysis_dimensions = [
            "news_sentiment",  # 新闻情感
            "social_media_sentiment",  # 社交媒体情绪
            "investor_sentiment",  # 投资者情绪
            "market_sentiment",  # 市场整体情绪
            "sentiment_trend",  # 情绪趋势
            "event_impact"  # 事件影响
        ]

        # 情绪分类
        self.sentiment_categories = {
            "positive": ["积极", "乐观", "利好", "上涨", "增长", "创新", "突破", "强劲"],
            "negative": ["消极", "悲观", "利空", "下跌", "衰退", "风险", "挑战", "担忧"],
            "neutral": ["中性", "平稳", "正常", "一般", "常规", "预期", "维持", "持平"]
        }

    def analyze(self, stock_code: str, time_range: str = "1y", **kwargs) -> AgentResult:
        """
        执行舆情与情绪分析
        
        Args:
            stock_code: 股票代码
            time_range: 时间范围
            **kwargs: 其他参数
            
        Returns:
            AgentResult: 分析结果
        """
        try:
            debug(f"开始对股票 {stock_code} 进行舆情与情绪分析")

            # 获取股票基本信息
            stock_info = self._get_stock_info(stock_code)

            # 获取新闻数据
            news_data = self._get_news_data(stock_code)

            # 获取社交媒体数据（模拟）
            social_media_data = self._get_social_media_data(stock_code)

            # 获取市场情绪指标
            market_sentiment = self._get_market_sentiment()

            # 检索相关知识库信息
            knowledge = self._retrieve_knowledge(f"投资者情绪分析 市场心理 行为金融")

            # 生成分析提示词
            prompt = self._generate_analysis_prompt(
                stock_code, stock_info, news_data, social_media_data, market_sentiment
            )

            # 使用LLM生成分析
            llm_analysis = self._generate_analysis(prompt, knowledge)

            # 结构化分析结果
            result = self._structure_result(
                stock_code, llm_analysis, stock_info, news_data, market_sentiment
            )

            debug(f"股票 {stock_code} 舆情与情绪分析完成")

            return AgentResult(
                agent_name=self.agent_name,
                success=True,
                result=result,
                confidence_score=result.get("confidence_score", 0.85)
            )

        except Exception as e:
            error(f"舆情与情绪分析失败: {str(e)}")
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
                "description": stock_info.get("description", "")
            }
        except Exception as e:
            error(f"获取股票信息失败: {str(e)}")
            return {}

    def _get_news_data(self, stock_code: str) -> Dict[str, Any]:
        """
        获取新闻数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Dict[str, Any]: 新闻数据
        """
        news_data = {
            "recent_news": [],
            "sentiment_overview": {
                "positive": 0,
                "negative": 0,
                "neutral": 0
            },
            "key_topics": []
        }

        try:
            # 从stock_manage获取新闻数据
            news = self.stock_manager.get_stock_news(stock_code)

            if news:
                recent_news = []
                positive_count = 0
                negative_count = 0
                neutral_count = 0

                for item in news[:10]:  # 获取最近10条新闻
                    title = item.get("title", "")
                    publisher = item.get("publisher", "")
                    link = item.get("link", "")
                    published_at = item.get("published_at", datetime.now().isoformat())

                    # 转换时间
                    try:
                        publish_time = datetime.fromisoformat(published_at).strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        publish_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # 简单的情感分析
                    sentiment = self._analyze_text_sentiment(title)
                    if sentiment == "positive":
                        positive_count += 1
                    elif sentiment == "negative":
                        negative_count += 1
                    else:
                        neutral_count += 1

                    recent_news.append({
                        "title": title,
                        "publisher": publisher,
                        "link": link,
                        "publish_time": publish_time,
                        "sentiment": sentiment
                    })

                news_data["recent_news"] = recent_news
                news_data["sentiment_overview"] = {
                    "positive": positive_count,
                    "negative": negative_count,
                    "neutral": neutral_count
                }

                # 提取关键主题
                key_topics = self._extract_key_topics([news_item["title"] for news_item in recent_news])
                news_data["key_topics"] = key_topics[:5]  # 取前5个主题

        except Exception as e:
            error(f"获取新闻数据失败: {str(e)}")

        return news_data

    def _get_social_media_data(self, stock_code: str) -> Dict[str, Any]:
        """
        获取社交媒体数据（模拟）
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Dict[str, Any]: 社交媒体数据
        """
        # 这里是模拟数据，实际项目中应该接入真实的社交媒体API
        social_media_data = {
            "platforms": [
                {
                    "name": "Twitter",
                    "mention_count": self._get_random_int(100, 1000),
                    "sentiment_score": self._get_random_float(-0.5, 0.5),
                    "trending_topics": ["新产品发布", "财报预期", "行业动态"]
                },
                {
                    "name": "Reddit",
                    "mention_count": self._get_random_int(50, 500),
                    "sentiment_score": self._get_random_float(-0.5, 0.5),
                    "trending_topics": ["技术分析", "机构持仓", "风险讨论"]
                },
                {
                    "name": "StockTwits",
                    "mention_count": self._get_random_int(200, 1500),
                    "sentiment_score": self._get_random_float(-0.5, 0.5),
                    "trending_topics": ["短期交易", "期权活动", "分析师评级"]
                }
            ],
            "overall_sentiment": self._get_random_float(-1.0, 1.0),
            "engagement_trend": "increasing" if self._get_random_float(0, 1) > 0.5 else "decreasing",
            "key_influencers": ["金融分析师A", "投资策略师B", "行业专家C"]
        }

        return social_media_data

    def _get_market_sentiment(self) -> Dict[str, Any]:
        """
        获取市场整体情绪指标
        
        Returns:
            Dict[str, Any]: 市场情绪指标
        """
        market_sentiment = {
            "fear_greed_index": {
                "current": self._get_random_int(20, 80),
                "trend": "improving" if self._get_random_float(0, 1) > 0.5 else "worsening"
            },
            "vix_equivalent": self._get_random_float(15, 30),
            "put_call_ratio": self._get_random_float(0.8, 1.5),
            "market_breadth": self._get_random_float(0.4, 0.6),
            "retail_investor_sentiment": "neutral" if self._get_random_float(0, 1) > 0.7 else "bullish" if self._get_random_float(0, 1) > 0.5 else "bearish"
        }

        return market_sentiment

    def _analyze_text_sentiment(self, text: str) -> str:
        """
        简单的文本情感分析
        
        Args:
            text: 要分析的文本
            
        Returns:
            str: 情感分类（positive, negative, neutral）
        """
        text_lower = text.lower()

        # 检查正面词汇
        for word in self.sentiment_categories["positive"]:
            if word in text_lower:
                return "positive"

        # 检查负面词汇
        for word in self.sentiment_categories["negative"]:
            if word in text_lower:
                return "negative"

        # 检查中性词汇
        for word in self.sentiment_categories["neutral"]:
            if word in text_lower:
                return "neutral"

        # 默认中性
        return "neutral"

    def _extract_key_topics(self, texts: List[str]) -> List[str]:
        """
        从文本列表中提取关键主题
        
        Args:
            texts: 文本列表
            
        Returns:
            List[str]: 关键主题列表
        """
        # 这里使用简单的关键词提取，实际项目中应该使用更复杂的NLP技术
        common_topics = [
            "财报", "盈利", "营收", "增长", "业绩", "收购", "合作", "产品",
            "技术", "创新", "市场", "竞争", "监管", "政策", "风险", "挑战"
        ]

        # 计算每个主题的出现频率
        topic_counts = {}
        for topic in common_topics:
            count = 0
            for text in texts:
                if topic in text:
                    count += 1
            if count > 0:
                topic_counts[topic] = count

        # 按频率排序并返回
        sorted_topics = sorted(topic_counts.keys(), key=lambda x: topic_counts[x], reverse=True)
        return sorted_topics

    def _get_random_int(self, min_val: int, max_val: int) -> int:
        """
        生成随机整数
        
        Args:
            min_val: 最小值
            max_val: 最大值
            
        Returns:
            int: 随机整数
        """
        import random
        return random.randint(min_val, max_val)

    def _get_random_float(self, min_val: float, max_val: float) -> float:
        """
        生成随机浮点数
        
        Args:
            min_val: 最小值
            max_val: 最大值
            
        Returns:
            float: 随机浮点数
        """
        import random
        return round(random.uniform(min_val, max_val), 2)

    def _generate_analysis_prompt(self, stock_code: str, stock_info: Dict[str, Any],
                                  news_data: Dict[str, Any], social_media_data: Dict[str, Any],
                                  market_sentiment: Dict[str, Any]) -> str:
        """
        生成分析提示词
        
        Args:
            stock_code: 股票代码
            stock_info: 股票基本信息
            news_data: 新闻数据
            social_media_data: 社交媒体数据
            market_sentiment: 市场情绪数据
            
        Returns:
            str: 分析提示词
        """
        # 导入langchain相关组件和提示词管理器
        from langchain_core.prompts import ChatPromptTemplate
        from hengline.prompts.prompt_manage import get_prompt

        # 从提示词管理器获取提示模板
        template = get_prompt('sentiment_agent', 'analysis')

        # 如果获取失败，使用备用模板
        if not template:
            template = """
请对股票代码 {{stock_code}} 的舆情与市场情绪进行全面分析。

股票基本信息：
{{stock_info}}

新闻数据：
{{news_data}}

社交媒体数据：
{{social_media_data}}

市场情绪指标：
{{market_sentiment}}

请提供详细分析。
"""

        # 创建提示模板实例
        prompt_template = ChatPromptTemplate.from_template(template)

        # 格式化并返回提示内容
        prompt = prompt_template.format(
            stock_code=stock_code,
            stock_info=json.dumps(stock_info, indent=2, ensure_ascii=False),
            news_data=json.dumps(news_data, indent=2, ensure_ascii=False),
            social_media_data=json.dumps(social_media_data, indent=2, ensure_ascii=False),
            market_sentiment=json.dumps(market_sentiment, indent=2, ensure_ascii=False)
        )

        return prompt

    def _structure_result(self, stock_code: str, llm_analysis: Dict[str, Any],
                          stock_info: Dict[str, Any], news_data: Dict[str, Any],
                          market_sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """
        结构化分析结果
        
        Args:
            stock_code: 股票代码
            llm_analysis: LLM生成的分析
            stock_info: 股票基本信息
            news_data: 新闻数据
            market_sentiment: 市场情绪数据
            
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
            "sentiment_summary": llm_analysis.get("sentiment_summary", {}),
            "potential_impact": llm_analysis.get("potential_impact", ""),
            "risk_signals": llm_analysis.get("risk_signals", []),
            "sentiment_drivers": llm_analysis.get("sentiment_drivers", []),
            "confidence_score": llm_analysis.get("confidence_score", 0.85),
            "sentiment_metrics": {
                "news_sentiment": news_data.get("sentiment_overview", {}),
                "market_sentiment": market_sentiment.get("fear_greed_index", {})
            }
        })

        # 验证结果
        if not self.validate_result(result):
            warning("舆情与情绪分析结果验证失败，使用默认结构")
            result = self.get_result_template()

        return result
