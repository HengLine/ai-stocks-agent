#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票数据管理模块
提供统一的多源股票数据获取接口
"""

import random
import time
from typing import Dict, Any, List

import pandas as pd

# 导入配置系统
from config.config import get_api_keys_config
from hengline.logger import debug, info, error, warning
from hengline.stock.simulated.stock_data_manager import stock_data_manager
from hengline.stock.sources.akshare_source import AKShareSource
from hengline.stock.sources.alltick_source import AlltickSource
from hengline.stock.sources.alpha_vantage_source import AlphaVantageSource
from hengline.stock.sources.iex_cloud_source import IEXCloudSource
from hengline.stock.sources.yahoo_direct_source import YahooDirectSource
# 导入各个数据源类
from hengline.stock.sources.yfinance_source import YFinanceSource


class StockDataManager:
    """
    股票数据管理器，提供多源数据获取和故障转移功能
    """

    def __init__(self):
        """
        初始化股票数据管理器
        从配置系统获取API密钥
        """
        # 从配置获取API密钥
        self.api_keys: Dict[str, Any] = get_api_keys_config()
        self._data_sources = {}
        self._last_request_time = {}
        self._consecutive_failures = {}
        self._init_data_sources()

    def _init_data_sources(self):
        """
        初始化所有可用的数据源
        只有在提供了有效API密钥或不需要密钥时才初始化相应数据源
        """
        info("初始化股票数据源...")

        # 初始化各个数据源
        try:
            # AKShare不需要API密钥，直接初始化
            self._data_sources['akshare'] = AKShareSource()
            debug("成功初始化 AKShare 数据源")
        except Exception as e:
            error(f"初始化 akshare 数据源失败: {str(e)}")

        try:
            # 检查Alltick API密钥
            alltick_key = self.api_keys.get("alltick", "")
            if alltick_key and alltick_key.strip() and not alltick_key.startswith('$'):
                self._data_sources['alltick'] = AlltickSource(self.api_keys)
                info("成功初始化 alltick 数据源")
            else:
                info("未提供有效的Alltick API密钥，跳过初始化")
        except Exception as e:
            error(f"初始化 alltick 数据源失败: {str(e)}")

        try:
            # yfinance不需要API密钥，直接初始化
            self._data_sources['yfinance'] = YFinanceSource(self.api_keys)
            info("成功初始化 yfinance 数据源")
        except Exception as e:
            error(f"初始化 yfinance 数据源失败: {str(e)}")

        try:
            self._data_sources['yahoo_direct'] = YahooDirectSource(self.api_keys)
            info("成功初始化 yahoo_direct 数据源")
        except Exception as e:
            error(f"初始化 yahoo_direct 数据源失败: {str(e)}")

        try:
            # 检查Alpha Vantage API密钥
            alpha_vantage_key = self.api_keys.get("alpha_vantage", "")
            if alpha_vantage_key and alpha_vantage_key.strip() and not alpha_vantage_key.startswith('$'):
                self._data_sources['alpha_vantage'] = AlphaVantageSource(alpha_vantage_key)
                info("成功初始化 alpha_vantage 数据源")
            else:
                info("未提供有效的Alpha Vantage API密钥，跳过初始化")
        except Exception as e:
            error(f"初始化 alpha_vantage 数据源失败: {str(e)}")

        try:
            # 检查IEX Cloud API密钥
            iex_cloud_key = self.api_keys.get("iex_cloud", "")
            if iex_cloud_key and iex_cloud_key.strip() and not iex_cloud_key.startswith('$'):
                self._data_sources['iex_cloud'] = IEXCloudSource({"iex_cloud": iex_cloud_key})
                info("成功初始化 iex_cloud 数据源")
            else:
                info("未提供有效的IEX Cloud API密钥，跳过初始化")
        except Exception as e:
            error(f"初始化 iex_cloud 数据源失败: {str(e)}")

        # 初始化请求间隔和失败计数
        for source_name in self._data_sources:
            self._last_request_time[source_name] = 0
            self._consecutive_failures[source_name] = 0

        info(f"数据源初始化完成，成功加载 {len(self._data_sources)} 个数据源")

    def _get_source_instance(self, source_name: str):
        """
        获取指定名称的数据源实例
        
        Args:
            source_name: 数据源名称
            
        Returns:
            数据源实例，如果不存在则返回None
        """
        return self._data_sources.get(source_name)

    def _wait_for_rate_limit(self, source_name: str):
        """
        处理API请求频率限制
        
        Args:
            source_name: 数据源名称
        """
        # 根据数据源类型设置最小请求间隔
        min_interval = {
            'yfinance': 3.0,
            'yahoo_direct': 3.0,
            'alpha_vantage': 12.0,  # Alpha Vantage免费版限制每分钟5次请求
            'iex_cloud': 1.0
        }.get(source_name, 2.0)

        # 随机化间隔时间，避免多个请求同时发生
        actual_interval = min_interval * (0.9 + 0.2 * random.random())

        # 计算需要等待的时间
        current_time = time.time()
        elapsed = current_time - self._last_request_time.get(source_name, 0)

        if elapsed < actual_interval:
            wait_time = actual_interval - elapsed
            debug(f"等待 {wait_time:.2f} 秒以遵守 {source_name} 的速率限制")
            time.sleep(wait_time)

        # 更新最后请求时间
        self._last_request_time[source_name] = time.time()

        # 处理连续失败的冷却时间
        failures = self._consecutive_failures.get(source_name, 0)
        if failures >= 1:
            cool_down_time = 10.0
            warning(f"检测到 {source_name} 连续 {failures} 次失败，执行 {cool_down_time} 秒冷却")
            time.sleep(cool_down_time)
            # 重置连续失败计数
            self._consecutive_failures[source_name] = 0

    def _try_data_source(self, source_name: str, method_name: str, *args, **kwargs):
        """
        尝试从指定数据源获取数据
        
        Args:
            source_name: 数据源名称
            method_name: 要调用的方法名
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            获取的数据，如果失败则返回None
        """
        source = self._get_source_instance(source_name)
        if not source:
            warning(f"数据源 {source_name} 不可用")
            return None

        try:
            # 等待速率限制
            self._wait_for_rate_limit(source_name)

            # 调用指定方法
            debug(f"使用 {source_name}.{method_name} 获取数据")
            method = getattr(source, method_name)
            result = method(*args, **kwargs)

            # 检查结果是否有效
            if self._is_valid_result(method_name, result):
                debug(f"{source_name} 成功获取数据")
                # 重置失败计数
                self._consecutive_failures[source_name] = 0
                return result
            else:
                warning(f"{source_name} 返回的数据无效")
                self._consecutive_failures[source_name] += 1
                return None

        except Exception as e:
            error(f"数据源 {source_name} 调用 {method_name} 失败: {str(e)}")
            self._consecutive_failures[source_name] += 1
            return None

    def _is_valid_result(self, method_name: str, result: Any) -> bool:
        """
        检查结果是否有效
        
        Args:
            method_name: 方法名
            result: 要检查的结果
            
        Returns:
            如果结果有效则返回True，否则返回False
        """
        if result is None:
            return False

        if method_name == 'get_stock_price_data':
            return isinstance(result, pd.DataFrame) and not result.empty and len(result.columns) > 0

        elif method_name == 'get_stock_info':
            if not isinstance(result, dict) or len(result) == 0:
                return False
            # 检查关键字段是否存在，如果缺少重要字段则认为无效
            required_fields = ['market_cap', 'pe_ratio', 'eps', 'dividend_yield']
            missing_fields = [field for field in required_fields if field not in result or result[field] in ['', None]]
            if missing_fields:
                debug(f"股票信息缺少关键字段: {missing_fields}")
                return False
            return True

        elif method_name == 'get_stock_news':
            return isinstance(result, list)

        elif method_name == 'get_financial_data':
            if not isinstance(result, dict):
                debug(f"财务数据返回值类型错误，预期dict，实际为{type(result)}")
                return False
            # 确保至少有一个有效的DataFrame
            has_valid_dataframe = False
            for key, value in result.items():
                if not isinstance(value, pd.DataFrame):
                    debug(f"财务数据中键'{key}'的值类型错误，预期DataFrame，实际为{type(value)}")
                elif value.empty or len(value.columns) == 0:
                    debug(f"财务数据中键'{key}'的DataFrame为空或没有列")
                else:
                    has_valid_dataframe = True
            return has_valid_dataframe

        return True

    def _load_mock_data(self, method_name: str, *args, **kwargs):
        """
        尝试加载模拟数据，提供有效数据以避免应用无响应
        
        Args:
            method_name: 要调用的方法名
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            模拟数据，确保返回有效的数据结构
        """
        import pandas as pd

        try:
            stock_code = args[0] if args else 'Unknown'
            debug(f"尝试为 {stock_code} 生成模拟数据: {method_name}")

            if method_name == 'get_stock_price_data':
                # 从模拟数据管理器获取股票价格数据
                stock_code = args[0] if args else ''
                period = kwargs.get('period', '1mo')
                interval = kwargs.get('interval', '1d')
                return stock_data_manager.get_stock_price_data(stock_code, period, interval)

            elif method_name == 'get_stock_info':
                # 从模拟数据管理器获取股票基本信息
                stock_code = args[0] if args else ''
                return stock_data_manager.get_stock_info(stock_code)

            elif method_name == 'get_stock_news':
                # 从模拟数据管理器获取股票新闻
                stock_code = args[0] if args else ''
                return stock_data_manager.get_stock_news(stock_code)

            elif method_name == 'get_financial_data':
                # 从模拟数据管理器获取财务数据
                stock_code = args[0] if args else ''
                return stock_data_manager.get_financial_data(stock_code)

            elif method_name == 'get_stock_realtime_data':
                # 从模拟数据管理器获取实时数据
                stock_code = args[0] if args else ''
                return stock_data_manager.get_stock_realtime_data(stock_code)

            return None

        except Exception as e:
            error(f"生成模拟数据时出错: {e}")
            # 即使生成模拟数据失败，也返回对应的空数据结构
            if method_name == 'get_stock_price_data':
                return pd.DataFrame()
            elif method_name == 'get_stock_info':
                return {}
            elif method_name == 'get_stock_news':
                return []
            elif method_name == 'get_financial_data':
                return {}
            return None

    def _get_data_with_fallback(self, method_name: str, *args, **kwargs):
        """
        获取数据，自动进行故障转移并使用缓存机制避免重复调用失败的数据源
        
        Args:
            method_name: 要调用的方法名
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            获取的数据，如果在线数据源失败则立即返回模拟数据
        """
        import time

        # 生成缓存键
        cache_key = f"{method_name}:{':'.join(str(arg) for arg in args)}:{':'.join(f'{k}={v}' for k, v in sorted(kwargs.items()))}"

        # 检查缓存
        if hasattr(self, '_cache') and cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < getattr(self, '_cache_ttl', 300):  # 默认缓存5分钟
                debug(f"从缓存获取数据: {cache_key}")
                return cached_data

        stock_code = args[0] if args else ''

        # 判断是否为A股代码（简单判断：6位数字或带sh/sz前缀）
        is_a_stock = False
        stock_code_lower = stock_code.lower()
        if (stock_code_lower.startswith('sh') or stock_code_lower.startswith('sz') or
                (stock_code.isdigit() and len(stock_code) == 6)):
            is_a_stock = True

        # 根据股票类型设置数据源优先级
        if is_a_stock:
            # A股优先使用AKShare数据源，Alltick作为备选
            source_priority = ['akshare', 'alltick', 'alpha_vantage']
            info(f"检测到可能的A股代码 {stock_code}，优先使用AKShare数据源")
        else:
            # 非A股使用原来的优先级，Alltick作为高优先级备选
            source_priority = ['yfinance', 'alltick', 'alpha_vantage', 'yahoo_direct', 'iex_cloud']

        # 过滤掉冷却中的数据源
        current_time = time.time()
        available_sources = []

        if hasattr(self, '_failed_sources'):
            for source_name in source_priority:
                if source_name in self._failed_sources:
                    fail_time, cool_down = self._failed_sources[source_name]
                    if current_time - fail_time < cool_down:
                        debug(f"跳过冷却中的数据源: {source_name}")
                        continue
                    else:
                        # 冷却期已过，移除失败记录
                        del self._failed_sources[source_name]
                available_sources.append(source_name)
        else:
            available_sources = source_priority
            # 初始化失败数据源跟踪字典
            self._failed_sources = {}

        # 确保缓存机制已初始化
        if not hasattr(self, '_cache'):
            self._cache = {}
            self._cache_ttl = 300  # 缓存有效期5分钟

        # 快速检查：如果已经有多个数据源失败记录或可用数据源较少，直接返回模拟数据
        if hasattr(self, '_failed_sources') and len(self._failed_sources) >= 2:
            info("检测到多个数据源失败，立即返回模拟数据以避免阻塞")
            mock_result = self._load_mock_data(method_name, *args, **kwargs)
            if mock_result is not None:
                # 缓存模拟数据
                self._cache[cache_key] = (mock_result, time.time())
                return mock_result

        if not available_sources:
            warning("所有数据源都在冷却中，立即返回模拟数据")
            # 如果所有数据源都在冷却中，直接返回模拟数据
            mock_result = self._load_mock_data(method_name, *args, **kwargs)
            if mock_result is not None:
                # 缓存模拟数据
                self._cache[cache_key] = (mock_result, time.time())
                return mock_result

        # 限制尝试的数据源数量，避免长时间阻塞
        max_attempts = 2  # 最多尝试2个数据源
        attempts = 0

        # 遍历数据源尝试获取数据
        for source_name in available_sources:
            if source_name not in self._data_sources:
                continue

            attempts += 1
            # 尝试第一个数据源，如果失败且已达到最大尝试次数，则直接返回模拟数据
            result = self._try_data_source(source_name, method_name, *args, **kwargs)

            if result is not None:
                # 缓存成功获取的数据
                self._cache[cache_key] = (result, time.time())
                return result

            # 如果获取失败，标记该数据源进入冷却期
            cool_down_time = 10.0  # 基础冷却时间10秒
            if hasattr(self, '_failed_sources') and source_name in self._failed_sources:
                # 失败次数越多，冷却时间越长（指数退避）
                _, prev_cool_down = self._failed_sources[source_name]
                cool_down_time = min(prev_cool_down * 2, 300)  # 最多冷却5分钟

            if hasattr(self, '_failed_sources'):
                self._failed_sources[source_name] = (current_time, cool_down_time)
                warning(f"从 {source_name} 获取数据失败，进入 {cool_down_time:.1f} 秒冷却期")
            else:
                warning(f"从 {source_name} 获取数据失败")

            # 如果已经尝试了最大次数，直接返回模拟数据
            if attempts >= max_attempts:
                info(f"已尝试 {attempts} 个数据源，全部失败，立即返回模拟数据")
                mock_result = self._load_mock_data(method_name, *args, **kwargs)
                if mock_result is not None:
                    # 缓存模拟数据
                    self._cache[cache_key] = (mock_result, time.time())
                    return mock_result
                break

        # 所有数据源都失败或已达到最大尝试次数，立即返回模拟数据
        info("数据源尝试完成，立即返回模拟数据")
        mock_result = self._load_mock_data(method_name, *args, **kwargs)
        if mock_result is not None:
            return mock_result

        # 如果模拟数据也无法加载，返回安全的默认值
        # 避免递归调用，直接返回空数据结构
        default_values = {
            'get_stock_price_data': pd.DataFrame(),
            'get_stock_info': {},
            'get_stock_news': [],
            'get_financial_data': {},
            'get_stock_realtime_data': {},
            'financial': {}
        }

        error(f"所有数据源和模拟数据获取 {method_name} 失败: {args[0] if args else 'unknown'}")
        # 返回默认值
        return default_values.get(method_name, None)

    def get_stock_price_data(self, stock_code: str, period: str = "1y",
                             interval: str = "1d") -> pd.DataFrame:
        """
        获取股票价格数据
        
        Args:
            stock_code: 股票代码
            period: 时间周期，如 "1d", "1mo", "1y", "max"
            interval: 数据间隔，如 "1m", "1h", "1d"
            
        Returns:
            包含价格数据的DataFrame
        """
        return self._get_data_with_fallback('get_stock_price_data',
                                            stock_code, period, interval)

    def get_stock_info(self, stock_code: str) -> Dict[str, Any]:
        """
        获取股票基本信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            包含股票信息的字典
        """
        return self._get_data_with_fallback('get_stock_info', stock_code)

    def get_stock_news(self, stock_code: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        获取股票相关新闻
        
        Args:
            stock_code: 股票代码
            limit: 新闻数量限制
            
        Returns:
            新闻列表
        """
        return self._get_data_with_fallback('get_stock_news', stock_code, limit)

    def get_financial_data(self, stock_code: str) -> Dict[str, pd.DataFrame]:
        """
        获取股票财务数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            财务数据字典，确保只包含有效的DataFrame
        """
        financial_data = self._get_data_with_fallback('get_financial_data', stock_code)

        # 确保返回的是有效的字典，且只包含有效的DataFrame
        if not isinstance(financial_data, dict):
            debug(f"财务数据不是字典类型: {type(financial_data)}")
            return {}

        # 清理财务数据，确保每个值都是有效的DataFrame
        valid_financial_data = {}
        for key, df in financial_data.items():
            if isinstance(df, pd.DataFrame) and not df.empty and len(df.columns) > 0:
                valid_financial_data[key] = df
            else:
                debug(f"跳过无效的财务数据项: {key} = {type(df)}")

        return valid_financial_data

    def get_stock_realtime_data(self, stock_code: str) -> Dict[str, Any]:
        """
        获取股票实时数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            实时数据字典
        """
        return self._get_data_with_fallback('get_stock_realtime_data', stock_code)


# 便捷函数
def get_stock_price_data(stock_code: str, period: str = "1y",
                         interval: str = "1d") -> pd.DataFrame:
    """
    便捷函数：获取股票价格数据
    """
    return manager.get_stock_price_data(stock_code, period, interval)


def get_stock_info(stock_code: str) -> Dict[str, Any]:
    """
    便捷函数：获取股票基本信息
    """
    return manager.get_stock_info(stock_code)


def get_stock_news(stock_code: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    便捷函数：获取股票相关新闻
    """
    return manager.get_stock_news(stock_code, limit)


def get_financial_data(stock_code: str) -> Dict[str, pd.DataFrame]:
    """
    便捷函数：获取股票财务数据
    """
    return manager.get_financial_data(stock_code)


manager = StockDataManager()
