#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@FileName: alltick_source.py
@Description: 股票数据获取类，用于从Alltick获取股票数据（https://alltick.co/）
@Author: HengLine
@Time: 2025/11/12
"""
import json
import time
from typing import Dict, Any, List, Optional

import pandas as pd
import requests

from hengline.logger import debug, info, warning, error


class AlltickSource:
    """
    Alltick数据源类
    提供股票价格、基本信息、新闻和财务数据获取功能
    """

    def __init__(self, api_keys: Dict[str, Any] = None):
        """
        初始化Alltick数据源
        
        Args:
            api_keys: API密钥字典
        """
        self.api_keys = api_keys or {}
        self.api_key = self.api_keys.get('alltick') or ''
        # self.base_url = "https://alltick.io"
        self.base_url = "https://alltick.io"
        self.api_version = "v1"  # API版本号
        self.cache = {}
        self.cache_ttl = 300  # 缓存5分钟
        
        # 重试机制配置
        self.max_retries = 3
        self.retry_backoff_factor = 1.0
        self.connect_timeout = 10
        self.read_timeout = 30

        # 设置请求头
        self.headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        if not self.api_key:
            warning("Alltick API密钥未配置，将使用模拟数据")
        else:
            debug(f"Alltick数据源初始化完成，API密钥: {self.api_key[:8]}...")

    def _generate_cache_key(self, method: str, *args, **kwargs) -> str:
        """
        生成缓存键
        
        Args:
            method: 方法名
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            缓存键字符串
        """
        key_parts = [method]
        key_parts.extend(str(arg) for arg in args)
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return ":".join(key_parts)

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        从缓存获取数据
        
        Args:
            cache_key: 缓存键
            
        Returns:
            缓存的数据，如果不存在或已过期则返回None
        """
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                debug(f"从缓存获取数据: {cache_key}")
                return data
            else:
                # 缓存过期，删除
                del self.cache[cache_key]
        return None

    def _set_cache(self, cache_key: str, data: Any) -> None:
        """
        设置缓存
        
        Args:
            cache_key: 缓存键
            data: 要缓存的数据
        """
        self.cache[cache_key] = (data, time.time())

    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        发送API请求
        
        Args:
            endpoint: API端点
            params: 请求参数
            
        Returns:
            API响应数据，失败则返回None
        """
        if not self.api_key:
            warning("Alltick API密钥未配置，无法发送请求")
            return None

        try:
            # 根据Alltick API文档使用正确的端点
            endpoint_mapping = {
                'quote': 'quote-stock-b-api/quote',
                'realtime': 'quote-stock-b-api/trade-tick', 
                'kline': 'quote-stock-b-api/kline',
                'news': 'quote-stock-b-api/news',
                'financial': 'quote-stock-b-api/financial'
            }
            
            actual_endpoint = endpoint_mapping.get(endpoint, endpoint)
            url = f"https://quote.alltick.co/{actual_endpoint}"
            
            # 准备请求参数
            params = params or {}
            
            # 根据不同端点验证必需参数
            if endpoint == 'kline':
                # K线数据必需参数验证
                if 'symbol' not in params:
                    error("K线数据请求缺少symbol参数")
                    return None
                if 'kline_type' not in params:
                    params['kline_type'] = '1d'  # 默认日线
                if 'limit' not in params:
                    params['limit'] = 252  # 默认1年数据
            elif endpoint == 'quote':
                # 报价数据必需参数验证
                if 'symbol' not in params:
                    error("报价数据请求缺少symbol参数")
                    return None
            elif endpoint == 'realtime':
                # 实时数据必需参数验证
                if 'symbol' not in params:
                    error("实时数据请求缺少symbol参数")
                    return None
            
            # 添加API密钥
            params['token'] = self.api_key

            info(f"发送Alltick API请求: {url}")
            info(f"请求参数: {params}")
            
            response = requests.get(
                url, 
                params=params, 
                headers=self.headers,
                timeout=10
            )

            info(f"Alltick API响应状态: {response.status_code}")
            
            if response.status_code == 429:
                warning("Alltick API请求频率超限 (HTTP 429)")
                return None
            elif response.status_code == 401:
                error("Alltick API认证失败，请检查API密钥")
                return None
            elif response.status_code == 402:
                error(f"Alltick API请求参数无效 (HTTP 402): {response.text}")
                return None
            elif response.status_code == 403:
                error("Alltick API访问被拒绝")
                return None
            elif response.status_code == 404:
                error("Alltick API端点不存在")
                return None
            elif response.status_code >= 500:
                error(f"Alltick API服务器错误: HTTP {response.status_code}")
                return None
            elif response.status_code == 200:
                try:
                    data = response.json()
                    info(f"Alltick API响应数据: {data}")
                    
                    # 检查响应格式
                    if 'code' in data:
                        if data.get('code') == 200 or data.get('code') == '200':
                            return data.get('data')
                        elif data.get('code') == 'GATEWAY_CODE_016':
                            warning(f"Alltick API不支持端点: {actual_endpoint}")
                            return None
                        else:
                            warning(f"Alltick API返回错误代码: {data.get('code')}, 消息: {data.get('msg', '未知错误')}")
                            return None
                    elif 'ok' in data and data.get('ok'):
                        return data.get('data')
                    elif 'msg' in data and data.get('msg') == 'token is required':
                        error("Alltick API需要token参数")
                        return None
                    elif 'msg' in data and data.get('msg') == 'token invalid':
                        error("Alltick API token无效")
                        return None
                    else:
                        warning(f"Alltick API返回格式异常: {data}")
                        return None
                except json.JSONDecodeError as e:
                    error(f"Alltick API响应解析失败: {str(e)}")
                    error(f"原始响应内容: {response.text[:500]}")
                    return None
            else:
                warning(f"Alltick API请求失败: HTTP {response.status_code}")
                warning(f"响应内容: {response.text[:200]}")
                return None

        except requests.exceptions.Timeout as e:
            error(f"Alltick API请求超时: {str(e)}")
            return None
        except requests.exceptions.ConnectionError as e:
            error(f"Alltick API连接错误: {str(e)}")
            return None
        except requests.exceptions.RequestException as e:
            error(f"Alltick API请求异常: {str(e)}")
            return None
        except Exception as e:
            error(f"Alltick API调用异常: {str(e)}")
            return None

    def get_stock_price_data(self, stock_code: str, period: str = "1y",
                             interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        获取股票价格数据
        
        Args:
            stock_code: 股票代码
            period: 时间周期
            interval: 数据间隔
            
        Returns:
            包含价格数据的DataFrame，失败则返回None
        """
        cache_key = self._generate_cache_key('get_stock_price_data', stock_code, period, interval)
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            # 将股票代码转换为Alltick格式
            alltick_code = self._convert_stock_code(stock_code)
            if not alltick_code:
                return None

            # 确定查询参数
            params = {
                'symbol': alltick_code,
                'kline_type': self._convert_interval(interval),
                'limit': self._convert_period_to_limit(period)
            }

            debug(f"使用Alltick获取股票价格数据: {stock_code} -> {alltick_code}")
            data = self._make_request('kline', params)

            if not data or 'kline' not in data:
                warning(f"Alltick未返回有效的价格数据: {stock_code}")
                return None

            # 转换为DataFrame
            kline_data = data['kline']
            if not kline_data:
                warning(f"Alltick返回的价格数据为空: {stock_code}")
                return None

            df = pd.DataFrame(kline_data)

            # 重命名列以匹配标准格式
            column_mapping = {
                'timestamp': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }

            # 检查必要的列是否存在
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            available_columns = df.columns.tolist()

            for col in required_columns:
                if col not in available_columns:
                    warning(f"Alltick价格数据缺少必要列: {col}")
                    return None

            df = df.rename(columns=column_mapping)

            # 转换时间戳
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], unit='s')
                df.set_index('date', inplace=True)

            # 确保数据类型正确
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # 按日期排序
            df.sort_index(inplace=True)

            debug(f"成功获取Alltick价格数据: {stock_code}, 共{len(df)}条记录")
            self._set_cache(cache_key, df)
            return df

        except Exception as e:
            error(f"获取Alltick价格数据失败: {stock_code}, 错误: {str(e)}")
            return None

    def get_stock_info(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        获取股票基本信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            包含股票信息的字典，失败则返回None
        """
        cache_key = self._generate_cache_key('get_stock_info', stock_code)
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            # 将股票代码转换为Alltick格式
            alltick_code = self._convert_stock_code(stock_code)
            if not alltick_code:
                return None

            params = {
                'symbol': alltick_code
            }

            info(f"使用Alltick获取股票基本信息: {stock_code} -> {alltick_code}")
            
            # 使用重试机制获取数据
            success, data = self._retry_api_call(
                lambda: self._make_request('quote', params),
                f"获取股票基本信息 {stock_code}"
            )
            
            if not success or not data or 'quote' not in data:
                warning(f"Alltick未返回有效的股票信息: {stock_code}")
                # 尝试备用数据
                fallback_data = self._get_fallback_data('info', stock_code)
                if fallback_data is not None:
                    self._set_cache(cache_key, fallback_data)
                    return fallback_data
                return None

            quote_data = data['quote']
            if not quote_data:
                warning(f"Alltick返回的股票信息为空: {stock_code}")
                return None

            # 构建标准格式的股票信息
            stock_info = {
                'symbol': stock_code,
                'name': quote_data.get('name', ''),
                'full_name': quote_data.get('name', ''),
                'company_name': quote_data.get('name', ''),
                'description': quote_data.get('description', ''),
                'sector': quote_data.get('sector', ''),
                'industry': quote_data.get('industry', ''),
                'market_cap': quote_data.get('market_cap', 0),
                'current_price': quote_data.get('current_price', 0),
                'change_percent': quote_data.get('change_percent', 0),
                'volume': quote_data.get('volume', 0),
                'currency': quote_data.get('currency', 'USD'),
                'exchange': quote_data.get('exchange', ''),
                'country': quote_data.get('country', ''),
                'timezone': quote_data.get('timezone', 'UTC')
            }

            info(f"成功获取Alltick股票信息: {stock_code}")
            self._set_cache(cache_key, stock_info)
            return stock_info

        except Exception as e:
            error(f"获取Alltick股票信息失败: {stock_code}, 错误: {str(e)}")
            # 尝试备用数据
            fallback_data = self._get_fallback_data('info', stock_code)
            if fallback_data is not None:
                self._set_cache(cache_key, fallback_data)
                return fallback_data
            return None

    def get_stock_realtime_data(self, stock_code: str) -> dict[str, str | Any] | None:
        """
        获取股票实时数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            实时数据字典
        """
        cache_key = self._generate_cache_key('get_stock_realtime_data', stock_code)
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            # 将股票代码转换为Alltick格式
            alltick_code = self._convert_stock_code(stock_code)
            if not alltick_code:
                return None

            params = {
                'symbol': alltick_code
            }

            info(f"使用Alltick获取实时数据: {stock_code} -> {alltick_code}")
            
            # 使用重试机制获取数据
            success, data = self._retry_api_call(
                lambda: self._make_request('realtime', params),
                f"获取实时数据 {stock_code}"
            )
            
            if not success or not data or 'realtime' not in data:
                warning(f"Alltick未返回有效的实时数据: {stock_code}")
                # 尝试备用数据
                fallback_data = self._get_fallback_data('realtime', stock_code)
                if fallback_data is not None:
                    self._set_cache(cache_key, fallback_data)
                    return fallback_data
                return None

            realtime_data = data['realtime']
            if not realtime_data:
                warning(f"Alltick返回的实时数据为空: {stock_code}")
                return None

            # 转换为标准格式
            result = {
                'symbol': stock_code,
                'name': realtime_data.get('name', ''),
                'current_price': realtime_data.get('current_price', 0),
                'open_price': realtime_data.get('open_price', 0),
                'high_price': realtime_data.get('high_price', 0),
                'low_price': realtime_data.get('low_price', 0),
                'volume': realtime_data.get('volume', 0),
                'turnover': realtime_data.get('turnover', 0),
                'change_percent': realtime_data.get('change_percent', 0),
                'change_amount': realtime_data.get('change_amount', 0),
                'amplitude': realtime_data.get('amplitude', 0),
                'turnover_rate': realtime_data.get('turnover_rate', 0),
                'pe_ratio': realtime_data.get('pe_ratio', 0),
                'market_cap': realtime_data.get('market_cap', 0),
                'update_time': realtime_data.get('update_time', pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
            }

            info(f"成功获取Alltick实时数据: {stock_code}")
            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            error(f"获取Alltick实时数据失败: {stock_code}, 错误: {str(e)}")
            # 尝试备用数据
            fallback_data = self._get_fallback_data('realtime', stock_code)
            if fallback_data is not None:
                self._set_cache(cache_key, fallback_data)
                return fallback_data
            return None

    def get_stock_news(self, stock_code: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取股票相关新闻
        
        Args:
            stock_code: 股票代码
            limit: 新闻数量限制
            
        Returns:
            新闻列表
        """
        cache_key = self._generate_cache_key('get_stock_news', stock_code, limit)
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            # 将股票代码转换为Alltick格式
            alltick_code = self._convert_stock_code(stock_code)
            if not alltick_code:
                return []

            params = {
                'symbol': alltick_code,
                'limit': limit
            }

            info(f"使用Alltick获取股票新闻: {stock_code} -> {alltick_code}")
            
            # 使用重试机制获取数据
            success, data = self._retry_api_call(
                lambda: self._make_request('news', params),
                f"获取股票新闻 {stock_code}"
            )
            
            if not success or not data or 'news' not in data:
                warning(f"Alltick未返回有效的新闻数据: {stock_code}")
                # 尝试备用数据
                fallback_data = self._get_fallback_data('news', stock_code, limit)
                if fallback_data is not None:
                    self._set_cache(cache_key, fallback_data)
                    return fallback_data
                return []

            news_data = data['news']
            if not news_data:
                warning(f"Alltick返回的新闻数据为空: {stock_code}")
                return []

            # 转换为标准格式
            news_list = []
            for item in news_data[:limit]:
                news_item = {
                    'title': item.get('title', ''),
                    'content': item.get('content', ''),
                    'summary': item.get('summary', ''),
                    'url': item.get('url', ''),
                    'source': item.get('source', 'Alltick'),
                    'published_at': item.get('published_at', ''),
                    'sentiment': item.get('sentiment', 'neutral')
                }
                news_list.append(news_item)

            info(f"成功获取Alltick新闻数据: {stock_code}, 共{len(news_list)}条")
            self._set_cache(cache_key, news_list)
            return news_list

        except Exception as e:
            error(f"获取Alltick新闻数据失败: {stock_code}, 错误: {str(e)}")
            # 尝试备用数据
            fallback_data = self._get_fallback_data('news', stock_code, limit)
            if fallback_data is not None:
                self._set_cache(cache_key, fallback_data)
                return fallback_data
            return []

    def get_financial_data(self, stock_code: str) -> Dict[str, pd.DataFrame]:
        """
        获取股票财务数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            财务数据字典
        """
        cache_key = self._generate_cache_key('get_financial_data', stock_code)
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            # 将股票代码转换为Alltick格式
            alltick_code = self._convert_stock_code(stock_code)
            if not alltick_code:
                return {}

            params = {
                'symbol': alltick_code
            }

            info(f"使用Alltick获取财务数据: {stock_code} -> {alltick_code}")
            
            # 使用重试机制获取数据
            success, data = self._retry_api_call(
                lambda: self._make_request('financial', params),
                f"获取财务数据 {stock_code}"
            )
            
            if not success or not data or 'financial' not in data:
                warning(f"Alltick未返回有效的财务数据: {stock_code}")
                # 尝试备用数据
                fallback_data = self._get_fallback_data('financial', stock_code)
                if fallback_data is not None:
                    self._set_cache(cache_key, fallback_data)
                    return fallback_data
                return {}

            financial_data = data['financial']
            if not financial_data:
                warning(f"Alltick返回的财务数据为空: {stock_code}")
                return {}

            result = {}

            # 处理收入报表
            if 'income_statement' in financial_data:
                income_data = financial_data['income_statement']
                if income_data:
                    result['income_statement'] = pd.DataFrame(income_data)

            # 处理资产负债表
            if 'balance_sheet' in financial_data:
                balance_data = financial_data['balance_sheet']
                if balance_data:
                    result['balance_sheet'] = pd.DataFrame(balance_data)

            # 处理现金流量表
            if 'cash_flow' in financial_data:
                cashflow_data = financial_data['cash_flow']
                if cashflow_data:
                    result['cash_flow'] = pd.DataFrame(cashflow_data)

            info(f"成功获取Alltick财务数据: {stock_code}, 共{len(result)}个报表")
            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            error(f"获取Alltick财务数据失败: {stock_code}, 错误: {str(e)}")
            # 尝试备用数据
            fallback_data = self._get_fallback_data('financial', stock_code)
            if fallback_data is not None:
                self._set_cache(cache_key, fallback_data)
                return fallback_data
            return {}

    def _convert_stock_code(self, stock_code: str) -> Optional[str]:
        """
        将股票代码转换为Alltick格式
        
        根据Alltick官方文档，股票代码格式为：
        - 深圳股票：000627.SZ（6位数字+.SZ后缀）
        - 上海股票：600416.SH（6位数字+.SH后缀）
        
        Args:
            stock_code: 原始股票代码
            
        Returns:
            Alltick格式的股票代码，失败则返回None
        """
        try:
            if not stock_code:
                return None
                
            # 清理股票代码
            stock_code = str(stock_code).strip().upper()
            
            # 移除可能的前缀
            if stock_code.startswith(('SZ', 'SH')):
                code = stock_code[2:]
                prefix = stock_code[:2]
                if code.isdigit() and len(code) == 6:
                    return f"{code}.{prefix}"
            elif stock_code.startswith(('0', '3', '6')):
                # 纯数字代码，需要添加交易所后缀
                if len(stock_code) == 6 and stock_code.isdigit():
                    if stock_code.startswith('6'):
                        return f"{stock_code}.SH"  # 上海证券交易所
                    elif stock_code.startswith(('0', '3')):
                        return f"{stock_code}.SZ"  # 深圳证券交易所
            elif '.' in stock_code:
                # 检查是否已经是正确格式
                parts = stock_code.split('.')
                if len(parts) == 2:
                    code = parts[0]
                    exchange = parts[1].upper()
                    if code.isdigit() and len(code) == 6 and exchange in ['SH', 'SZ']:
                        return f"{code}.{exchange}"
            
            # 如果都不匹配，返回原始代码（可能是港股、美股等）
            warning(f"无法转换股票代码为标准A股格式，使用原始代码: {stock_code}")
            return stock_code

        except Exception as e:
            error(f"转换股票代码失败: {stock_code}, 错误: {str(e)}")
            return None

    def _convert_interval(self, interval: str) -> str:
        """
        转换时间间隔参数
        
        Args:
            interval: 原始间隔字符串
            
        Returns:
            Alltick格式的间隔字符串
        """
        interval_mapping = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '1d': '1d',
            '1w': '1w',
            '1M': '1M'
        }
        return interval_mapping.get(interval, '1d')

    def _retry_api_call(self, func, description: str = "API调用"):
        """
        带重试机制的API调用
        
        Args:
            func: 要调用的函数
            description: 调用描述
            
        Returns:
            (success, result) 元组，success表示是否成功，result为结果或None
        """
        func_name = getattr(func, '__name__', 'unknown_function')
        
        for attempt in range(self.max_retries):
            try:
                info(f"Alltick API调用: {description}, 尝试次数: {attempt + 1}/{self.max_retries}")
                result = func()
                
                # 检查结果有效性
                if result is not None:
                    if isinstance(result, pd.DataFrame):
                        if not result.empty:
                            info(f"Alltick API调用成功，尝试次数: {attempt + 1}, 返回 {len(result)} 行数据")
                            return True, result
                        else:
                            warning(f"Alltick API返回空DataFrame，尝试次数: {attempt + 1}")
                    elif isinstance(result, (dict, list)):
                        if result:  # 非空字典或列表
                            info(f"Alltick API调用成功，尝试次数: {attempt + 1}")
                            return True, result
                        else:
                            warning(f"Alltick API返回空{type(result).__name__}，尝试次数: {attempt + 1}")
                    else:
                        info(f"Alltick API调用成功，尝试次数: {attempt + 1}")
                        return True, result
                else:
                    warning(f"Alltick API返回None，尝试次数: {attempt + 1}")
                    
            except requests.exceptions.ConnectionError as e:
                error(f"Alltick连接错误 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = min(30, (2 ** attempt) * self.retry_backoff_factor)  # 指数退避，最大30秒
                    info(f"连接错误，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    
            except requests.exceptions.Timeout as e:
                error(f"Alltick请求超时 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = min(30, (2 ** attempt) * self.retry_backoff_factor)
                    info(f"请求超时，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    
            except requests.exceptions.HTTPError as e:
                error(f"Alltick HTTP错误 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = min(30, (2 ** attempt) * self.retry_backoff_factor)
                    info(f"HTTP错误，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    
            except requests.exceptions.RequestException as e:
                error(f"Alltick请求异常 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = min(30, (2 ** attempt) * self.retry_backoff_factor)
                    info(f"请求异常，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    
            except ValueError as e:
                # 处理JSON解析错误或数据格式错误
                error(f"Alltick数据解析错误 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = min(30, (2 ** attempt) * self.retry_backoff_factor)
                    info(f"数据解析错误，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                error(f"Alltick未知异常 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = min(30, (2 ** attempt) * self.retry_backoff_factor)
                    info(f"未知异常，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    
        error(f"Alltick API调用失败，已重试 {self.max_retries} 次")
        return False, None

    def _get_fallback_data(self, data_type: str, symbol: str, *args, **kwargs):
        """
        获取备用数据（当Alltick服务不可用时）
        
        Args:
            data_type: 数据类型
            symbol: 股票代码
            *args: 其他参数
            **kwargs: 其他关键字参数
            
        Returns:
            备用数据
        """
        info(f"尝试获取 {symbol} 的备用数据类型: {data_type}")
        
        if data_type == "price":
            # 返回一个空的价格数据结构
            return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        elif data_type == "realtime":
            # 返回空的实时数据结构
            return pd.DataFrame(columns=['symbol', 'name', 'price', 'change', 'change_percent', 'volume'])
        elif data_type == "info":
            # 返回基本的股票信息
            return {
                'symbol': symbol,
                'name': f'股票{symbol}',
                'market': '深A' if symbol.startswith(('sz', 'SZ', '300', '000', '001', '002')) else '沪A',
                'data_source': '备用数据'
            }
        elif data_type == "news":
            # 返回空新闻列表
            return []
        elif data_type == "financial":
            # 返回空的财务数据结构
            return {
                'income_statement': pd.DataFrame(),
                'balance_sheet': pd.DataFrame(),
                'cash_flow': pd.DataFrame()
            }
        
        return None

    def _convert_period_to_limit(self, period: str) -> int:
        """
        将时间周期转换为数据条数限制
        
        Args:
            period: 时间周期字符串
            
        Returns:
            数据条数限制
        """
        period_mapping = {
            '1d': 1,
            '5d': 5,
            '1mo': 22,
            '3mo': 66,
            '6mo': 132,
            '1y': 252,
            '2y': 504,
            '5y': 1260,
            '10y': 2520,
            'max': 5000
        }
        return period_mapping.get(period, 252)
