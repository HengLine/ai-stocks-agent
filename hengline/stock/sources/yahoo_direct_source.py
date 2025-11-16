#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接Yahoo Finance API 数据源实现
使用requests直接访问Yahoo Finance API获取数据
"""

from typing import Dict, List, Any

import pandas as pd
import requests

# 配置日志
from hengline.logger import debug, info, error, warning
from hengline.tools.cache_tool import get_price_cache, set_price_cache, get_stock_info_cache, set_stock_info_cache, get_financial_cache, set_financial_cache


class YahooDirectSource:
    """
    直接Yahoo Finance API 数据源类
    """

    def __init__(self, api_keys: Dict[str, Any]):
        """
        初始化直接Yahoo Finance API数据源
        """
        self.base_url = "https://query1.finance.yahoo.com"
        # 从配置中获取API密钥，使用yfinance作为键名以保持一致性
        # self.api_key = api_keys.get("yfinance")
        # if not self.api_key:
        #     warning("未设置 Yahoo API 密钥，请在环境变量中设置 YFINANCE_API_KEY 或 YAHOO_API_KEY")

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        info("初始化直接Yahoo Finance API数据源")

    def _make_request(self, url: str, params: Dict = None) -> Dict:
        """
        发送HTTP请求
        
        Args:
            url: 请求URL
            params: 查询参数
        
        Returns:
            JSON响应数据
        """
        if params is None:
            params = {}

        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()  # 检查HTTP错误
            return response.json()
        except requests.exceptions.RequestException as e:
            error(f"Yahoo Direct: HTTP请求失败: {str(e)}")
            raise

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
                debug(f"Yahoo Direct: 从缓存获取价格数据: {stock_code}")
                return cached_data

            debug(f"Yahoo Direct: 获取股票 {stock_code} 的价格数据")

            # 构建URL和参数
            url = f"{self.base_url}/v8/finance/chart/{stock_code}"

            # 周期和间隔映射
            period_map = {
                "1d": "1d",
                "1w": "5d",  # Yahoo直接API的周期表示不同
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

            # 间隔映射
            interval_map = {
                "1m": "1m",
                "2m": "2m",
                "5m": "5m",
                "15m": "15m",
                "30m": "30m",
                "60m": "60m",
                "90m": "90m",
                "1h": "1h",
                "1d": "1d",
                "5d": "5d",
                "1wk": "1wk",
                "1mo": "1mo",
                "3mo": "3mo"
            }

            # 使用当前时间戳
            import time
            params = {
                'period1': int(time.time() - 365 * 24 * 3600),  # 默认获取最近一年数据
                'period2': int(time.time()),
                'interval': interval_map.get(interval, "1m"),
                'includePrePost': 'false',
                'events': 'div|split'
            }

            # 发送请求
            data = self._make_request(url, params)

            # 解析数据
            chart_data = data.get('chart', {}).get('result', [])
            if not chart_data:
                warning(f"Yahoo Direct: 未获取到 {stock_code} 的价格数据")
                return pd.DataFrame()

            result = chart_data[0]
            timestamp = result.get('timestamp', [])
            quote_data = result.get('indicators', {}).get('quote', [])

            if not timestamp or not quote_data:
                warning(f"Yahoo Direct: 价格数据不完整")
                return pd.DataFrame()

            # 创建DataFrame
            df = pd.DataFrame({
                'Open': quote_data[0].get('open', []),
                'High': quote_data[0].get('high', []),
                'Low': quote_data[0].get('low', []),
                'Close': quote_data[0].get('close', []),
                'Volume': quote_data[0].get('volume', [])
            }, index=pd.to_datetime(timestamp, unit='s'))

            # 重命名列以符合yfinance格式
            df.index.name = 'Date'

            # 保存到缓存
            set_price_cache(stock_code, period, interval, df)
            debug(f"Yahoo Direct: 成功获取并缓存 {len(df)} 行价格数据")

            return df
        except Exception as e:
            error(f"Yahoo Direct: 获取价格数据失败: {str(e)}")
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
            if cached_info is not None:
                debug(f"Yahoo Direct: 从缓存获取股票信息: {stock_code}")
                return cached_info

            debug(f"Yahoo Direct: 获取股票 {stock_code} 的基本信息")

            # 构建URL和参数
            url = f"{self.base_url}/v11/finance/quoteSummary/{stock_code}"
            params = {
                'modules': 'summaryProfile,financialData,price'
            }

            # 发送请求
            data = self._make_request(url, params)

            # 解析数据
            result = data.get('quoteSummary', {}).get('result', [])
            if not result:
                warning(f"Yahoo Direct: 未获取到 {stock_code} 的基本信息")
                return {}

            summary = result[0]
            info = {}

            # 提取基本信息
            if 'summaryProfile' in summary:
                profile = summary['summaryProfile']
                info['long_name'] = profile.get('longName')
                info['industry'] = profile.get('industry')
                info['sector'] = profile.get('sector')

            # 提取财务数据
            if 'financialData' in summary:
                financial = summary['financialData']
                info['market_cap'] = financial.get('marketCap', {}).get('raw')
                info['pe_ratio'] = financial.get('trailingPE', {}).get('raw')
                info['forward_pe'] = financial.get('forwardPE', {}).get('raw')
                info['peg_ratio'] = financial.get('pegRatio', {}).get('raw')
                info['eps'] = financial.get('trailingEps', {}).get('raw')
                info['dividend_yield'] = financial.get('dividendYield', {}).get('raw')

            # 提取价格数据
            if 'price' in summary:
                price = summary['price']
                info['symbol'] = price.get('symbol')
                info['previous_close'] = price.get('regularMarketPreviousClose', {}).get('raw')
                info['current_price'] = price.get('regularMarketPrice', {}).get('raw')
                info['day_high'] = price.get('regularMarketDayHigh', {}).get('raw')
                info['day_low'] = price.get('regularMarketDayLow', {}).get('raw')
                info['volume'] = price.get('regularMarketVolume', {}).get('raw')
                info['beta'] = price.get('beta', {}).get('raw')
                info['52_week_high'] = price.get('fiftyTwoWeekHigh', {}).get('raw')
                info['52_week_low'] = price.get('fiftyTwoWeekLow', {}).get('raw')

            # 移除None值
            info = {k: v for k, v in info.items() if v is not None}

            # 保存到缓存，设置30分钟过期
            set_stock_info_cache(stock_code, info)
            debug(f"Yahoo Direct: 成功获取并缓存股票信息")

            return info
        except Exception as e:
            error(f"Yahoo Direct: 获取股票信息失败: {str(e)}")
            raise

    def get_stock_news(self, stock_code: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        获取股票相关新闻（使用Yahoo Finance的新闻API）
        
        Args:
            stock_code: 股票代码
            limit: 返回新闻数量限制
        
        Returns:
            新闻列表
        """
        try:
            # 注意：Yahoo Finance的新闻API可能需要特定权限或有不同的端点
            # 这里使用一个通用的新闻搜索API作为示例
            debug(f"Yahoo Direct: 获取股票 {stock_code} 的新闻")

            # 由于Yahoo的新闻API可能不太容易直接访问，这里返回空列表
            # 在实际应用中，可以替换为可用的新闻API
            warning("Yahoo Direct: 新闻API暂不可用")
            return []
        except Exception as e:
            error(f"Yahoo Direct: 获取新闻数据失败: {str(e)}")
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
            if cached_data is not None:
                debug(f"Yahoo Direct: 从缓存获取财务数据: {stock_code}")
                return cached_data

            debug(f"Yahoo Direct: 获取股票 {stock_code} 的财务数据")

            # 构建URL和参数
            url = f"{self.base_url}/v11/finance/quoteSummary/{stock_code}"
            params = {
                'modules': 'incomeStatementHistory,balanceSheetHistory,cashflowStatementHistory'
            }

            # 发送请求
            data = self._make_request(url, params)

            # 解析数据
            result = data.get('quoteSummary', {}).get('result', [])
            if not result:
                warning(f"Yahoo Direct: 未获取到 {stock_code} 的财务数据")
                return {}

            summary = result[0]
            financial_data = {}

            # 提取收入报表
            if 'incomeStatementHistory' in summary:
                income_data = summary['incomeStatementHistory'].get('incomeStatementHistory', [])
                income_statements = []

                for stmt in income_data:
                    stmt_dict = {
                        'Total Revenue': stmt.get('totalRevenue', {}).get('raw'),
                        'Cost Of Revenue': stmt.get('costOfRevenue', {}).get('raw'),
                        'Gross Profit': stmt.get('grossProfit', {}).get('raw'),
                        'Operating Income': stmt.get('operatingIncome', {}).get('raw'),
                        'Net Income': stmt.get('netIncome', {}).get('raw')
                    }
                    date = pd.to_datetime(stmt.get('endDate', {}).get('fmt'))
                    income_statements.append((date, stmt_dict))

                # 创建DataFrame
                if income_statements:
                    df = pd.DataFrame([s[1] for s in income_statements],
                                      index=[s[0] for s in income_statements])
                    financial_data['income_statement'] = df

            # 保存到缓存
            set_financial_cache(stock_code, financial_data, 'all')
            debug(f"Yahoo Direct: 成功获取并缓存财务数据")

            # 类似地提取资产负债表和现金流量表...
            # 这里简化处理，实际应用中需要更详细的数据提取

            debug("Yahoo Direct: 财务数据获取完成")

            return financial_data
        except Exception as e:
            error(f"Yahoo Direct: 获取财务数据失败: {str(e)}")
            raise
