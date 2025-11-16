#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Yahoo Finance 数据源实现
使用yfinance库获取股票数据
"""
from typing import Dict, List, Any

import pandas as pd
import yfinance as yf

# 配置日志
from hengline.logger import debug, info, error, warning
# 导入缓存工具
from hengline.tools.cache_tool import get_price_cache, set_price_cache, get_stock_info_cache, set_stock_info_cache, get_news_cache, set_news_cache, get_financial_cache, set_financial_cache


class YFinanceSource:
    """
    Yahoo Finance 数据源类
    """

    def __init__(self, api_keys: Dict[str, Any]):
        """
        初始化Yahoo Finance数据源
        """
        info("初始化Yahoo Finance数据源")

        # 设置yfinance的缓存路径
        # self.cache_dir = "data/cache/yfinance"
        # os.makedirs(self.cache_dir, exist_ok=True)
        # yf.set_tz_cache_location(self.cache_dir)

        # 设置请求超时
        self.timeout = 30

        # 从配置中获取API密钥
        # self.api_key = api_keys.get("yfinance")
        # if not self.api_key:
        #     warning("未设置 YFinance API 密钥，请在环境变量中设置 YFINANCE_API_KEY")

    def get_stock_price_data(self, stock_code: str, period: str = "1d", interval: str = "1m") -> pd.DataFrame:
        """
        获取股票价格数据
        
        Args:
            stock_code: 股票代码
            period: 时间周期
            interval: 时间间隔
        
        Returns:
            价格数据DataFrame
        """
        try:
            # 尝试从缓存获取
            cached_data = get_price_cache(stock_code, period, interval)
            if cached_data is not None:
                debug(f"YFinance: 从缓存获取价格数据: {stock_code}，周期: {period}")
                return cached_data
            
            # 周期映射
            period_map = {
                "1d": "1d",
                "1w": "1wk",
                "1m": "1mo",
                "3m": "3mo",
                "6m": "6mo",
                "1y": "1y",
                "2y": "2y",
                "5y": "5y",
                "10y": "10y",
                "ytd": "ytd",
                "max": "max"
            }

            # 获取数据
            yf_period = period_map.get(period, "1d")
            debug(f"YFinance: 获取股票 {stock_code} 的价格数据，周期: {yf_period}，间隔: {interval}")

            # 使用yfinance获取数据
            data = yf.download(
                            stock_code,
                            period=yf_period,
                            interval=interval,
                            threads=False,  # 禁用多线程，减少并发
                            timeout=10,
                            proxy=None
                        )

            # 记录数据量
            debug(f"YFinance: 获取到 {len(data)} 行价格数据")
            
            # 缓存数据
            if not data.empty:
                set_price_cache(stock_code, period, interval, data)
                debug(f"YFinance: 缓存价格数据: {stock_code}，周期: {period}")

            return data
        except Exception as e:
            error(f"YFinance: 获取价格数据失败: {str(e)}")
            raise

    def get_stock_info(self, stock_code: str) -> Dict[str, Any]:
        """
        获取股票基本信息
        
        Args:
            stock_code: 股票代码
        
        Returns:
            股票信息字典
        """
        try:
            # 尝试从缓存获取
            cached_info = get_stock_info_cache(stock_code)
            if cached_info:
                debug(f"YFinance: 从缓存获取股票信息: {stock_code}")
                return cached_info
            
            debug(f"YFinance: 获取股票 {stock_code} 的基本信息")

            # 创建Ticker对象
            ticker = yf.Ticker(stock_code)

            # 获取信息
            info = ticker.info

            # 过滤必要信息
            filtered_info = {
                "symbol": info.get("symbol"),
                "long_name": info.get("longName"),
                "industry": info.get("industry"),
                "sector": info.get("sector"),
                "market_cap": info.get("marketCap"),
                "previous_close": info.get("regularMarketPreviousClose"),
                "current_price": info.get("regularMarketPrice"),
                "day_high": info.get("regularMarketDayHigh"),
                "day_low": info.get("regularMarketDayLow"),
                "volume": info.get("regularMarketVolume"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "eps": info.get("trailingEps"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow")
            }

            # 移除None值
            filtered_info = {k: v for k, v in filtered_info.items() if v is not None}

            # 缓存数据
            if filtered_info:
                set_stock_info_cache(stock_code, filtered_info)
                debug(f"YFinance: 缓存股票信息: {stock_code}")

            debug(f"YFinance: 成功获取股票信息: {filtered_info.get('long_name', stock_code)}")

            return filtered_info
        except Exception as e:
            error(f"YFinance: 获取股票信息失败: {str(e)}")
            raise

    def get_stock_news(self, stock_code: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        获取股票相关新闻
        
        Args:
            stock_code: 股票代码
            limit: 返回新闻数量限制
        
        Returns:
            新闻列表
        """
        try:
            # 尝试从缓存获取
            cached_news = get_news_cache(stock_code)
            if cached_news:
                debug(f"YFinance: 从缓存获取新闻数据: {stock_code}")
                return cached_news
            
            debug(f"YFinance: 获取股票 {stock_code} 的新闻，限制: {limit}")

            # 创建Ticker对象
            ticker = yf.Ticker(stock_code)

            # 获取新闻
            news = ticker.news

            # 限制返回数量
            news = news[:limit] if news else []

            # 格式化新闻
            formatted_news = []
            for item in news:
                formatted_news.append({
                    "title": item.get("title"),
                    "publisher": item.get("publisher"),
                    "link": item.get("link"),
                    "providerPublishTime": item.get("providerPublishTime"),
                    "summary": item.get("summary")
                })

            # 缓存数据
            if formatted_news:
                set_news_cache(stock_code, formatted_news)
                debug(f"YFinance: 缓存新闻数据: {stock_code}")

            debug(f"YFinance: 成功获取 {len(formatted_news)} 条新闻")

            return formatted_news
        except Exception as e:
            error(f"YFinance: 获取新闻数据失败: {str(e)}")
            raise

    def get_financial_data(self, stock_code: str) -> Dict[str, pd.DataFrame]:
        """
        获取财务数据
        
        Args:
            stock_code: 股票代码
        
        Returns:
            财务数据字典
        """
        try:
            # 尝试从缓存获取
            cached_data = get_financial_cache(stock_code, 'all')
            if cached_data:
                debug(f"YFinance: 从缓存获取财务数据: {stock_code}")
                return cached_data
            
            debug(f"YFinance: 获取股票 {stock_code} 的财务数据")

            # 创建Ticker对象
            ticker = yf.Ticker(stock_code)

            # 获取财务数据
            financial_data = {
                "income_statement": ticker.income_stmt,
                "balance_sheet": ticker.balance_sheet,
                "cash_flow": ticker.cashflow
            }

            # 缓存数据
            if financial_data:
                set_financial_cache(stock_code, financial_data, 'all')
                debug(f"YFinance: 缓存财务数据: {stock_code}")

            debug("YFinance: 财务数据获取完成")

            return financial_data
        except Exception as e:
            error(f"YFinance: 获取财务数据失败: {str(e)}")
            raise