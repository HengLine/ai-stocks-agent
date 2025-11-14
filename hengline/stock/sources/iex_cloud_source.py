#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IEX Cloud 数据源实现
使用IEX Cloud API获取股票数据
"""
import os
from typing import Dict, List, Any

import pandas as pd
import requests

# 配置日志
from hengline.logger import debug, info, error, warning
from hengline.tools.cache_tool import get_price_cache, set_price_cache, get_stock_info_cache, set_stock_info_cache, get_financial_cache, set_financial_cache


class IEXCloudSource:
    """
    IEX Cloud 数据源类
    """

    def __init__(self, api_keys: Dict[str, Any]):
        """
        初始化IEX Cloud数据源
        
        Args:
            api_keys: IEX Cloud API密钥，如果不提供则尝试从环境变量获取
        """
        self.api_key = api_keys.get("iex_cloud")
        if not self.api_key:
            warning("IEX Cloud API密钥未设置，某些功能可能不可用")

        # 确定使用的环境（生产或沙盒）
        self.use_sandbox = os.environ.get('IEX_CLOUD_USE_SANDBOX', 'False').lower() == 'true'

        # 设置基础URL
        if self.use_sandbox:
            self.base_url = "https://sandbox.iexapis.com/stable"
            info("初始化IEX Cloud沙盒环境数据源")
        else:
            self.base_url = "https://cloud.iexapis.com/stable"
            info("初始化IEX Cloud生产环境数据源")

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        发送API请求
        
        Args:
            endpoint: API端点
            params: 请求参数
        
        Returns:
            JSON响应数据
        """
        try:
            url = f"{self.base_url}/{endpoint}"

            # 添加API密钥
            if not params:
                params = {}

            params['token'] = self.api_key

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            # 检查响应内容类型
            if response.headers.get('Content-Type') == 'application/json':
                return response.json()
            else:
                # 如果不是JSON，尝试解析为文本
                return {'text': response.text}
        except requests.exceptions.RequestException as e:
            error(f"IEX Cloud: HTTP请求失败: {str(e)}")
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
                debug(f"IEX Cloud: 从缓存获取价格数据: {stock_code}")
                return cached_data

            debug(f"IEX Cloud: 获取股票 {stock_code} 的价格数据")

            # 根据周期选择不同的端点
            if period == "1d":
                # 日内数据
                endpoint = f"stock/{stock_code}/intraday-prices"
                params = {
                    'chartInterval': 1  # 1分钟间隔
                }
            else:
                # 日线及以上数据
                chart_range_map = {
                    "1w": 5,
                    "1m": "1m",
                    "3m": "3m",
                    "6m": "6m",
                    "1y": "1y",
                    "2y": "2y",
                    "5y": "5y",
                    "10y": "10y",
                    "ytd": 'ytd',
                    "max": 'max'
                }

                # 处理时间周期
                chart_range = chart_range_map.get(period, '1m')

                endpoint = f"stock/{stock_code}/chart/{chart_range}"
                params = {
                    'chartCloseOnly': 'true'  # 只获取收盘价以减少数据量
                }

            # 发送请求
            data = self._make_request(endpoint, params)

            # 检查是否是列表格式的数据
            if not isinstance(data, list):
                warning(f"IEX Cloud: 价格数据格式不正确")
                return pd.DataFrame()

            # 创建DataFrame
            df = pd.DataFrame(data)

            if df.empty:
                warning(f"IEX Cloud: 未获取到 {stock_code} 的价格数据")
                return pd.DataFrame()

            # 处理日期索引
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            elif 'minute' in df.columns:
                df['minute'] = pd.to_datetime(df['minute'])
                df.set_index('minute', inplace=True)

            # 重命名列以符合标准格式
            columns_map = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            df = df.rename(columns=columns_map)

            # 确保必要的列存在
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    warning(f"IEX Cloud: 价格数据缺少 {col} 列")

            # 转换数据类型
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # 保存到缓存
            set_price_cache(stock_code, period, interval, df)
            debug(f"IEX Cloud: 成功获取并缓存 {len(df)} 行价格数据")

            return df
        except Exception as e:
            error(f"IEX Cloud: 获取价格数据失败: {str(e)}")
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
                debug(f"IEX Cloud: 从缓存获取股票信息: {stock_code}")
                return cached_info

            debug(f"IEX Cloud: 获取股票 {stock_code} 的基本信息")

            # 获取公司基本信息
            company_data = self._make_request(f"stock/{stock_code}/company")

            # 获取价格信息
            price_data = self._make_request(f"stock/{stock_code}/quote")

            # 合并信息
            info = {
                'symbol': stock_code,
                'long_name': company_data.get('companyName'),
                'industry': company_data.get('industry'),
                'sector': company_data.get('sector'),
                'market_cap': price_data.get('marketCap'),
                'previous_close': price_data.get('previousClose'),
                'current_price': price_data.get('latestPrice'),
                'day_high': price_data.get('high'),
                'day_low': price_data.get('low'),
                'volume': price_data.get('latestVolume'),
                'pe_ratio': price_data.get('peRatio'),
                'beta': price_data.get('beta'),
                '52_week_high': price_data.get('week52High'),
                '52_week_low': price_data.get('week52Low')
            }

            # 尝试获取额外的财务指标
            try:
                stats_data = self._make_request(f"stock/{stock_code}/stats")
                info.update({
                    'forward_pe': stats_data.get('forwardPE'),
                    'eps': stats_data.get('ttmEPS'),
                    'dividend_yield': stats_data.get('dividendYield')
                })
            except Exception as e:
                warning(f"IEX Cloud: 获取额外财务指标失败: {str(e)}")

            # 移除None值
            info = {k: v for k, v in info.items() if v is not None}

            # 保存到缓存
            set_stock_info_cache(stock_code, info)
            debug(f"IEX Cloud: 成功获取并缓存股票信息")

            return info
        except Exception as e:
            error(f"IEX Cloud: 获取股票信息失败: {str(e)}")
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
            debug(f"IEX Cloud: 获取股票 {stock_code} 的新闻")

            # 获取新闻数据
            endpoint = f"stock/{stock_code}/news/last/{limit}"
            data = self._make_request(endpoint)

            # 检查数据格式
            if not isinstance(data, list):
                warning(f"IEX Cloud: 新闻数据格式不正确")
                return []

            # 格式化新闻
            news_list = []
            for item in data:
                news_list.append({
                    'title': item.get('headline'),
                    'publisher': item.get('source'),
                    'link': item.get('url'),
                    'providerPublishTime': item.get('datetime'),
                    'summary': item.get('summary')
                })

            debug(f"IEX Cloud: 成功获取 {len(news_list)} 条新闻")

            return news_list
        except Exception as e:
            error(f"IEX Cloud: 获取新闻数据失败: {str(e)}")
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
            cached_financials = get_financial_cache(stock_code, 'all')
            if cached_financials is not None:
                debug(f"IEX Cloud: 从缓存获取财务数据: {stock_code}")
                return cached_financials

            debug(f"IEX Cloud: 获取股票 {stock_code} 的财务数据")

            financial_data = {}

            # 获取收入报表
            try:
                # 注意：IEX Cloud的财务报表API可能需要付费订阅
                endpoint = f"stock/{stock_code}/financials"
                params = {'period': 'annual'}  # 年度财务报表

                data = self._make_request(endpoint, params)

                if 'financials' in data and isinstance(data['financials'], list):
                    reports = data['financials']
                    df = pd.DataFrame(reports)

                    # 处理日期
                    df['reportDate'] = pd.to_datetime(df['reportDate'])
                    df.set_index('reportDate', inplace=True)

                    # 转换数值列
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                    financial_data['income_statement'] = df
            except Exception as e:
                warning(f"IEX Cloud: 获取收入报表失败: {str(e)}")

            # 获取资产负债表
            try:
                endpoint = f"stock/{stock_code}/balance-sheet"
                params = {'period': 'annual'}

                data = self._make_request(endpoint, params)

                if 'balancesheet' in data and isinstance(data['balancesheet'], list):
                    reports = data['balancesheet']
                    df = pd.DataFrame(reports)

                    # 处理日期
                    df['reportDate'] = pd.to_datetime(df['reportDate'])
                    df.set_index('reportDate', inplace=True)

                    # 转换数值列
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                    financial_data['balance_sheet'] = df
            except Exception as e:
                warning(f"IEX Cloud: 获取资产负债表失败: {str(e)}")

            # 获取现金流量表
            try:
                endpoint = f"stock/{stock_code}/cash-flow"
                params = {'period': 'annual'}

                data = self._make_request(endpoint, params)

                if 'cashflow' in data and isinstance(data['cashflow'], list):
                    reports = data['cashflow']
                    df = pd.DataFrame(reports)

                    # 处理日期
                    df['reportDate'] = pd.to_datetime(df['reportDate'])
                    df.set_index('reportDate', inplace=True)

                    # 转换数值列
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                    financial_data['cash_flow'] = df
            except Exception as e:
                warning(f"IEX Cloud: 获取现金流量表失败: {str(e)}")

            # 保存到缓存
            set_financial_cache(stock_code, financial_data, 'all')
            debug("IEX Cloud: 财务数据获取完成并已缓存")

            return financial_data
        except Exception as e:
            error(f"IEX Cloud: 获取财务数据失败: {str(e)}")
            raise
