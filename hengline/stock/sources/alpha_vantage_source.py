#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Alpha Vantage 数据源实现
使用Alpha Vantage API获取股票数据
"""

from typing import Dict, List, Any

import pandas as pd
import requests

# 配置日志
from hengline.logger import debug, info, error, warning
# 导入缓存工具
from hengline.tools.cache_tool import get_api_cache, set_api_cache


class AlphaVantageSource:
    """
    Alpha Vantage 数据源类
    """

    def __init__(self, api_key):
        """
        初始化Alpha Vantage数据源
        """
        self.base_url = "https://www.alphavantage.co/query"
        # 尝试从环境变量获取API密钥，如果没有则使用默认的免费密钥
        self.api_key = api_key

        info("初始化Alpha Vantage数据源")

    def _generate_cache_key(self, params: Dict) -> str:
        """
        生成缓存键
        
        Args:
            params: 请求参数
        
        Returns:
            缓存键字符串
        """
        # 创建一个不包含api_key的参数副本用于生成缓存键
        params_without_key = params.copy()
        if 'apikey' in params_without_key:
            del params_without_key['apikey']
        if 'datatype' in params_without_key:
            del params_without_key['datatype']

        # 排序参数键以确保相同参数生成相同的键
        sorted_params = sorted(params_without_key.items())
        # 生成缓存键
        cache_key = "_".join([f"{k}={v}" for k, v in sorted_params])
        return cache_key

    def _make_request(self, params: Dict) -> Dict:
        """
        发送API请求（带缓存）
        
        Args:
            params: 请求参数
        
        Returns:
            JSON响应数据
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(params)

        # 尝试从缓存获取
        cached_data = get_api_cache(cache_key)
        if cached_data:
            debug(f"Alpha Vantage: 从缓存获取数据: {cache_key}")
            return cached_data

        try:
            # 添加API密钥
            params['apikey'] = self.api_key
            params['datatype'] = 'json'  # 使用JSON格式

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # 保存到缓存（如果没有错误）
            if 'Error Message' not in data and 'Note' not in data:
                # API数据缓存24小时
                set_api_cache(cache_key, data, ttl=24 * 3600)
                debug(f"Alpha Vantage: 保存数据到缓存: {cache_key}")

            return data
        except requests.exceptions.RequestException as e:
            error(f"Alpha Vantage: HTTP请求失败: {str(e)}")
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
            debug(f"Alpha Vantage: 获取股票 {stock_code} 的价格数据")

            # 根据周期选择不同的API
            if period == "1d":
                # 日内数据
                params = {
                    'function': 'TIME_SERIES_INTRADAY',
                    'symbol': stock_code,
                    'interval': '1min'  # Alpha Vantage支持的间隔
                }
            else:
                # 日线及以上数据
                function_map = {
                    "1w": "TIME_SERIES_DAILY",  # Alpha Vantage不直接支持周线，可以从日线计算
                    "1m": "TIME_SERIES_MONTHLY",
                    "3m": "TIME_SERIES_MONTHLY",
                    "6m": "TIME_SERIES_MONTHLY",
                    "1y": "TIME_SERIES_MONTHLY",
                    "2y": "TIME_SERIES_MONTHLY",
                    "5y": "TIME_SERIES_MONTHLY",
                    "10y": "TIME_SERIES_MONTHLY",
                    "ytd": "TIME_SERIES_DAILY",
                    "max": "TIME_SERIES_MONTHLY_ADJUSTED"
                }

                function = function_map.get(period, "TIME_SERIES_DAILY")

                params = {
                    'function': function,
                    'symbol': stock_code
                }

                # 对于日线数据，请求完整输出
                if function == "TIME_SERIES_DAILY":
                    params['outputsize'] = 'full'

            # 发送请求
            data = self._make_request(params)

            # 检查是否有错误
            if 'Error Message' in data:
                raise Exception(f"Alpha Vantage API错误: {data['Error Message']}")

            # 检查API调用限制
            if 'Note' in data and 'rate limit' in data['Note'].lower():
                warning(f"Alpha Vantage: API调用达到限制: {data['Note']}")

            # 确定时间序列键
            if period == "1d":
                time_series_key = 'Time Series (1min)'
            elif period == "1w" or period == "ytd":
                time_series_key = 'Time Series (Daily)'
            else:
                time_series_key = 'Monthly Time Series' if 'MONTHLY' in params['function'] else 'Time Series (Daily)'

            # 检查时间序列数据是否存在
            if time_series_key not in data:
                warning(f"Alpha Vantage: 未找到时间序列数据: {time_series_key}")
                return pd.DataFrame()

            # 解析数据
            time_series = data[time_series_key]

            # 创建DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # 重命名列
            columns_map = {
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            }
            df = df.rename(columns=columns_map)

            # 转换数据类型
            df = df.astype({'Open': float, 'High': float, 'Low': float, 'Close': float, 'Volume': int})

            # 根据周期过滤数据
            if period == "1w":
                # 从日线数据计算周线
                df = df.resample('W').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                })
            elif period == "ytd":
                # 今年至今的数据
                current_year = pd.Timestamp.now().year
                df = df[df.index.year == current_year]

            debug(f"Alpha Vantage: 成功获取 {len(df)} 行价格数据")

            return df
        except Exception as e:
            error(f"Alpha Vantage: 获取价格数据失败: {str(e)}")
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
            debug(f"Alpha Vantage: 获取股票 {stock_code} 的基本信息")

            # 使用GLOBAL_QUOTE API获取基本信息
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': stock_code
            }

            # 发送请求
            data = self._make_request(params)

            # 检查是否有错误
            if 'Error Message' in data:
                raise Exception(f"Alpha Vantage API错误: {data['Error Message']}")

            # 检查全局报价数据
            if 'Global Quote' not in data or not data['Global Quote']:
                warning(f"Alpha Vantage: 未找到 {stock_code} 的报价数据")
                # 尝试使用BATCH_STOCK_QUOTES作为备选
                return self._get_stock_info_alternative(stock_code)

            # 解析数据
            quote = data['Global Quote']
            info = {
                'symbol': quote.get('01. symbol'),
                'previous_close': float(quote.get('02. open', 0)),
                'current_price': float(quote.get('05. price', 0)),
                'day_high': float(quote.get('03. high', 0)),
                'day_low': float(quote.get('04. low', 0)),
                'volume': int(quote.get('06. volume', 0)),
                'pe_ratio': float(quote.get('08. peRatio', 0)) if quote.get('08. peRatio') else None
            }

            # 尝试获取更多信息
            info.update(self._get_company_overview(stock_code))

            # 移除None值
            info = {k: v for k, v in info.items() if v is not None}

            debug(f"Alpha Vantage: 成功获取股票信息")

            return info
        except Exception as e:
            error(f"Alpha Vantage: 获取股票信息失败: {str(e)}")
            raise

    def _get_stock_info_alternative(self, stock_code: str) -> Dict[str, Any]:
        """
        使用替代方法获取股票基本信息
        
        Args:
            stock_code: 股票代码
        
        Returns:
            股票信息字典
        """
        try:
            params = {
                'function': 'BATCH_STOCK_QUOTES',
                'symbols': stock_code
            }

            data = self._make_request(params)

            if 'Stock Quotes' not in data or not data['Stock Quotes']:
                return {}

            quote = data['Stock Quotes'][0]
            return {
                'symbol': quote.get('1. symbol'),
                'current_price': float(quote.get('2. price', 0)),
                'volume': int(quote.get('3. volume', 0))
            }
        except Exception as e:
            warning(f"Alpha Vantage: 替代方法获取股票信息失败: {str(e)}")
            return {}

    def _get_company_overview(self, stock_code: str) -> Dict[str, Any]:
        """
        获取公司概览信息
        
        Args:
            stock_code: 股票代码
        
        Returns:
            公司信息字典
        """
        try:
            params = {
                'function': 'OVERVIEW',
                'symbol': stock_code
            }

            data = self._make_request(params)

            # 检查是否有有效的响应
            if not data or 'Symbol' not in data:
                return {}

            return {
                'long_name': data.get('Name'),
                'industry': data.get('Industry'),
                'sector': data.get('Sector'),
                'market_cap': float(data.get('MarketCapitalization', 0)) if data.get('MarketCapitalization') else None,
                'forward_pe': float(data.get('ForwardPE', 0)) if data.get('ForwardPE') else None,
                'eps': float(data.get('EPS', 0)) if data.get('EPS') else None,
                'dividend_yield': float(data.get('DividendYield', 0).replace('%', '')) / 100 if data.get('DividendYield') else None,
                'beta': float(data.get('Beta', 0)) if data.get('Beta') else None
            }
        except Exception as e:
            warning(f"Alpha Vantage: 获取公司概览失败: {str(e)}")
            return {}

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
            debug(f"Alpha Vantage: 获取股票 {stock_code} 的新闻")

            # Alpha Vantage的NEWS_SENTIMENT API需要高级订阅
            # 这里使用一个简单的实现，实际应用中可能需要其他新闻源

            # 注意：这个API需要付费订阅
            try:
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'tickers': stock_code,
                    'limit': limit
                }

                data = self._make_request(params)

                if 'feed' not in data:
                    warning("Alpha Vantage: 未找到新闻数据（可能需要高级订阅）")
                    return []

                # 解析新闻数据
                news_list = []
                for item in data['feed'][:limit]:
                    news_list.append({
                        'title': item.get('title'),
                        'publisher': item.get('source'),
                        'link': item.get('url'),
                        'providerPublishTime': item.get('time_published'),
                        'summary': item.get('summary')
                    })

                return news_list
            except Exception as e:
                warning(f"Alpha Vantage: 新闻API调用失败: {str(e)}")
                return []
        except Exception as e:
            error(f"Alpha Vantage: 获取新闻数据失败: {str(e)}")
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
            debug(f"Alpha Vantage: 获取股票 {stock_code} 的财务数据")

            financial_data = {}

            # 获取收入报表
            try:
                params = {
                    'function': 'INCOME_STATEMENT',
                    'symbol': stock_code
                }

                data = self._make_request(params)

                if 'annualReports' in data:
                    reports = data['annualReports']
                    df = pd.DataFrame(reports)
                    df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
                    df.set_index('fiscalDateEnding', inplace=True)

                    # 转换数值列
                    numeric_columns = ['totalRevenue', 'costOfRevenue', 'grossProfit',
                                       'operatingIncome', 'netIncome']
                    for col in numeric_columns:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                    financial_data['income_statement'] = df
            except Exception as e:
                warning(f"Alpha Vantage: 获取收入报表失败: {str(e)}")

            # 获取资产负债表
            try:
                params = {
                    'function': 'BALANCE_SHEET',
                    'symbol': stock_code
                }

                data = self._make_request(params)

                if 'annualReports' in data:
                    reports = data['annualReports']
                    df = pd.DataFrame(reports)
                    df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
                    df.set_index('fiscalDateEnding', inplace=True)

                    financial_data['balance_sheet'] = df
            except Exception as e:
                warning(f"Alpha Vantage: 获取资产负债表失败: {str(e)}")

            # 获取现金流量表
            try:
                params = {
                    'function': 'CASH_FLOW',
                    'symbol': stock_code
                }

                data = self._make_request(params)

                if 'annualReports' in data:
                    reports = data['annualReports']
                    df = pd.DataFrame(reports)
                    df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
                    df.set_index('fiscalDateEnding', inplace=True)

                    financial_data['cash_flow'] = df
            except Exception as e:
                warning(f"Alpha Vantage: 获取现金流量表失败: {str(e)}")

            debug("Alpha Vantage: 财务数据获取完成")

            return financial_data
        except Exception as e:
            error(f"Alpha Vantage: 获取财务数据失败: {str(e)}")
            raise
