"""
@FileName: akshare_source.py
@Description: 股票数据获取类，用于从AKShare获取股票数据（https://akshare.akfamily.xyz/）
@Author: HengLine
@Time: 2025/11/13
"""
import time
import random
from typing import Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings

import akshare as ak
import pandas as pd
from pandas import DataFrame

from hengline.logger import debug, info, error, warning
from hengline.tools.cache_tool import (
    get_price_cache, set_price_cache,
    get_stock_info_cache, set_stock_info_cache,
    get_news_cache, set_news_cache,
    get_financial_cache, set_financial_cache,
    get_api_cache, set_api_cache,
    clear_all_caches,
    price_cache_manager, stock_info_cache_manager,
    news_cache_manager, financial_cache_manager, api_cache_manager
)
from utils.log_utils import print_log_exception

class AKShareSource:
    """
    AKShare数据源类，用于获取A股市场数据
    """

    def __init__(self):
        """
        初始化AKShare数据源
        """
        self.source_name = "akshare"
        
        # 检查AKShare版本
        try:
            self.akshare_version = ak.__version__
            info(f"AKShare版本: {self.akshare_version}")
        except AttributeError:
            warning("无法获取AKShare版本信息")
            self.akshare_version = "unknown"
        
        # 频率控制配置
        self.rate_limit = 1.0  # 请求频率限制（秒）
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 最小请求间隔
        
        # 重试和超时配置
        self.max_retries = 3
        self.connect_timeout = 15  # 增加连接超时时间（秒）
        self.read_timeout = 45    # 增加读取超时时间（秒）
        self.retry_backoff_factor = 1.5  # 增加重试退避因子
        
        # 连接池配置
        self.pool_connections = 10
        self.pool_maxsize = 20
        self.max_retries_adapter = 3
        
        # 配置重试策略
        try:
            # 尝试使用新版本的参数名
            self.retry_strategy = Retry(
                total=self.max_retries_adapter,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS"],
                backoff_factor=self.retry_backoff_factor,
                raise_on_status=False
            )
        except TypeError:
            # 如果失败，使用旧版本的参数名
            self.retry_strategy = Retry(
                total=self.max_retries_adapter,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS"],
                backoff_factor=self.retry_backoff_factor,
                raise_on_status=False
            )
        
        # 配置HTTP适配器
        self.adapter = HTTPAdapter(
            max_retries=self.retry_strategy,
            pool_connections=self.pool_connections,
            pool_maxsize=self.pool_maxsize
        )
        self.session = requests.Session()
        self.session.mount("http://", self.adapter)
        self.session.mount("https://", self.adapter)
        
        # 设置会话超时
        self.session.timeout = (self.connect_timeout, self.read_timeout)
        
        # 设置请求头，模拟浏览器行为
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # 抑制AKShare的警告信息
        warnings.filterwarnings('ignore', category=FutureWarning, module='akshare')
        
        info("AKShare数据源初始化完成，已配置连接超时、重试机制和频率控制")

    def _check_connection_health(self) -> bool:
        """
        检查连接健康状态
        
        Returns:
            连接是否健康
        """
        try:
            # 检查会话状态
            if not hasattr(self, 'session') or self.session is None:
                warning("HTTP会话未初始化")
                return False
            
            # 尝试简单的连接测试
            test_url = "https://httpbin.org/get"
            try:
                response = self.session.get(
                    test_url, 
                    timeout=5,  # 短超时用于健康检查
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                )
                if response.status_code == 200:
                    return True
                else:
                    warning(f"连接健康检查失败，状态码: {response.status_code}")
                    return False
            except Exception as e:
                warning(f"连接健康检查异常: {str(e)}")
                return False
                
        except Exception as e:
            warning(f"连接健康检查出错: {str(e)}")
            return False

    def _recreate_session_if_needed(self):
        """
        如果需要，重新创建HTTP会话
        """
        try:
            if not self._check_connection_health():
                warning("检测到连接问题，重新创建HTTP会话")
                
                # 关闭旧会话
                if hasattr(self, 'session') and self.session:
                    self.session.close()
                
                # 创建新会话
                self.session = requests.Session()
                self.session.mount("http://", self.adapter)
                self.session.mount("https://", self.adapter)
                self.session.timeout = (self.connect_timeout, self.read_timeout)
                
                # 重新设置请求头
                self.session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                })
                
                info("HTTP会话重新创建完成")
        except Exception as e:
            warning(f"重新创建HTTP会话失败: {str(e)}")

    def _wait_for_rate_limit(self):
        """
        等待请求频率限制
        """
        try:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            
            # 使用更严格的频率控制
            wait_time = max(0, self.min_request_interval - elapsed)
            
            if wait_time > 0:
                self._log_rate_limit(wait_time)
                time.sleep(wait_time)
            else:
                self._log_rate_limit(0)
            
            self.last_request_time = time.time()
            
        except Exception as e:
            warning(f"频率控制等待时出错: {str(e)}")
            # 即使出错也要更新最后请求时间，避免死循环
            self.last_request_time = time.time()

    def _retry_api_call(self, func, *args, **kwargs):
        """
        带重试机制的API调用 - 增强版
        
        Args:
            func: 要调用的函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            函数调用结果，失败则返回None
        """
        func_name = getattr(func, '__name__', 'unknown_function')
        
        # User-Agent轮换列表
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        ]
        
        for attempt in range(self.max_retries):
            try:
                # 在每次重试前检查连接健康状态
                if attempt > 0:
                    self._recreate_session_if_needed()
                    # 随机更换User-Agent
                    self.session.headers['User-Agent'] = random.choice(user_agents)
                
                # 添加随机延迟避免频率限制
                if attempt > 0:
                    # 随机延迟：基础延迟 + 随机因子
                    base_delay = (2 ** attempt) * self.retry_backoff_factor
                    random_delay = random.uniform(0.5, 2.0) * (attempt ** 0.5)
                    total_delay = base_delay + random_delay
                    
                    info(f"第 {attempt + 1} 次重试，随机延迟 {total_delay:.2f} 秒...")
                    time.sleep(total_delay)
                
                self._log_api_call(func_name, attempt + 1, self.max_retries, *args, **kwargs)
                
                # 频率控制
                self._wait_for_rate_limit()
                
                result = func(*args, **kwargs)
                
                # 检查结果有效性
                if result is not None:
                    if isinstance(result, pd.DataFrame):
                        if not result.empty and len(result) > 0:
                            info(f"AKShare API调用成功，尝试次数: {attempt + 1}, 返回 {len(result)} 行数据")
                            return result
                        else:
                            warning(f"AKShare API返回空DataFrame，尝试次数: {attempt + 1}")
                    elif isinstance(result, dict):
                        # 检查字典是否有效
                        if 'data' in result:
                            # 如果有data键，检查data内容
                            data = result['data']
                            if isinstance(data, pd.DataFrame):
                                if not data.empty and len(data) > 0:
                                    info(f"AKShare API调用成功，尝试次数: {attempt + 1}, 返回DataFrame包含 {len(data)} 行数据")
                                    return result
                                else:
                                    warning(f"AKShare API返回空data DataFrame，尝试次数: {attempt + 1}")
                            elif isinstance(data, list):
                                if len(data) > 0:
                                    info(f"AKShare API调用成功，尝试次数: {attempt + 1}, 返回列表包含 {len(data)} 项")
                                    return result
                                else:
                                    warning(f"AKShare API返回空data列表，尝试次数: {attempt + 1}")
                            elif isinstance(data, dict):
                                if len(data) > 0:
                                    info(f"AKShare API调用成功，尝试次数: {attempt + 1}, 返回字典包含 {len(data)} 个键")
                                    return result
                                else:
                                    warning(f"AKShare API返回空data字典，尝试次数: {attempt + 1}")
                            else:
                                # data是其他类型，只要不是None就认为有效
                                if data is not None:
                                    info(f"AKShare API调用成功，尝试次数: {attempt + 1}")
                                    return result
                                else:
                                    warning(f"AKShare API返回None data，尝试次数: {attempt + 1}")
                        else:
                            # 没有data键，检查字典本身
                            if len(result) > 0:
                                info(f"AKShare API调用成功，尝试次数: {attempt + 1}, 返回字典包含 {len(result)} 个键")
                                return result
                            else:
                                warning(f"AKShare API返回空字典，尝试次数: {attempt + 1}")
                    elif isinstance(result, list):
                        if len(result) > 0:
                            info(f"AKShare API调用成功，尝试次数: {attempt + 1}, 返回列表包含 {len(result)} 项")
                            return result
                        else:
                            warning(f"AKShare API返回空列表，尝试次数: {attempt + 1}")
                    else:
                        info(f"AKShare API调用成功，尝试次数: {attempt + 1}")
                        return result
                else:
                    warning(f"AKShare API返回None，尝试次数: {attempt + 1}")
                    
            except requests.exceptions.ConnectionError as e:
                error_msg = str(e)
                if "RemoteDisconnected" in error_msg or "Connection aborted" in error_msg:
                    error(f"AKShare远程连接中断 (尝试 {attempt + 1}/{self.max_retries}): {error_msg}")
                else:
                    error(f"AKShare连接错误 (尝试 {attempt + 1}/{self.max_retries}): {error_msg}")
                
                if attempt < self.max_retries - 1:
                    # 对于连接中断，使用更长的等待时间和随机化
                    base_wait = min(60, (2 ** attempt) * self.retry_backoff_factor * 3)
                    random_wait = random.uniform(1, 5)
                    total_wait = base_wait + random_wait
                    info(f"连接错误，等待 {total_wait:.2f} 秒后重试...")
                    time.sleep(total_wait)
                    
            except requests.exceptions.Timeout as e:
                error(f"AKShare请求超时 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    base_wait = min(45, (2 ** attempt) * self.retry_backoff_factor * 2)
                    random_wait = random.uniform(0.5, 3)
                    total_wait = base_wait + random_wait
                    info(f"请求超时，等待 {total_wait:.2f} 秒后重试...")
                    time.sleep(total_wait)
                    
            except requests.exceptions.HTTPError as e:
                error(f"AKShare HTTP错误 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    base_wait = min(30, (2 ** attempt) * self.retry_backoff_factor)
                    random_wait = random.uniform(0.5, 2)
                    total_wait = base_wait + random_wait
                    info(f"HTTP错误，等待 {total_wait:.2f} 秒后重试...")
                    time.sleep(total_wait)
                    
            except requests.exceptions.RequestException as e:
                error(f"AKShare请求异常 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    base_wait = min(30, (2 ** attempt) * self.retry_backoff_factor)
                    random_wait = random.uniform(0.5, 2)
                    total_wait = base_wait + random_wait
                    info(f"请求异常，等待 {total_wait:.2f} 秒后重试...")
                    time.sleep(total_wait)
                    
            except ConnectionResetError as e:
                error(f"AKShare连接重置错误 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    base_wait = min(60, (2 ** attempt) * self.retry_backoff_factor * 3)
                    random_wait = random.uniform(2, 8)
                    total_wait = base_wait + random_wait
                    info(f"连接重置，等待 {total_wait:.2f} 秒后重试...")
                    time.sleep(total_wait)
                    
            except ValueError as e:
                # 处理JSON解析错误或数据格式错误
                error(f"AKShare数据解析错误 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    base_wait = min(30, (2 ** attempt) * self.retry_backoff_factor)
                    random_wait = random.uniform(0.5, 2)
                    total_wait = base_wait + random_wait
                    info(f"数据解析错误，等待 {total_wait:.2f} 秒后重试...")
                    time.sleep(total_wait)
                    
            except Exception as e:
                error(f"AKShare未知异常 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    base_wait = min(30, (2 ** attempt) * self.retry_backoff_factor)
                    random_wait = random.uniform(0.5, 2)
                    total_wait = base_wait + random_wait
                    info(f"未知异常，等待 {total_wait:.2f} 秒后重试...")
                    time.sleep(total_wait)
                    
        error(f"AKShare API调用失败，已重试 {self.max_retries} 次")
        return None

    def _check_connection(self) -> bool:
        """
        检查网络连接状态
        
        Returns:
            连接是否正常
        """
        try:
            info("开始网络连接检查")
            
            # 检查多个常用网站以确保网络连接正常
            test_urls = [
                "https://www.baidu.com",
                "https://www.sina.com.cn",
                "https://finance.eastmoney.com"
            ]
            
            for url in test_urls:
                try:
                    response = self.session.get(
                        url, 
                        timeout=self.connect_timeout / 2,  # 使用较短的超时时间
                        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                    )
                    if response.status_code == 200:
                        info(f"网络检查成功 - URL: {url}, 状态码: {response.status_code}")
                        return True
                    else:
                        warning(f"网络请求失败 - URL: {url}, 状态码: {response.status_code}")
                except requests.exceptions.Timeout:
                    error(f"网络请求超时 - URL: {url}")
                except requests.exceptions.ConnectionError:
                    warning(f"网络连接错误 - URL: {url}")
                except Exception as e:
                    warning(f"网络连接检查失败 - URL: {url}, 错误: {str(e)}")

            warning("所有网络检查都失败")
            return False
            
        except Exception as e:
            warning(f"网络检查异常: {str(e)}")
            return False

    def _check_akshare_service(self) -> bool:
        """
        检查AKShare服务可用性
        
        Returns:
            AKShare服务是否可用
        """
        try:
            info("开始AKShare服务检查")
            
            # 尝试调用一个简单的API来测试服务
            test_functions = [
                ("tool_trade_date_hist_sina", "获取交易日历"),
                ("stock_zh_a_spot_em", "获取A股实时数据（仅测试）")
            ]
            
            for func_name, description in test_functions:
                try:
                    func = getattr(ak, func_name, None)
                    if func is None:
                        warning(f"AKShare函数未找到: {func_name}")
                        continue
                        
                    info(f"测试AKShare函数: {func_name} - {description}")
                    
                    if func_name == "tool_trade_date_hist_sina":
                        # 测试交易日历API，这个API比较稳定
                        result = func()
                        if result is not None and not result.empty:
                            info(f"AKShare函数测试成功: {func_name} - {description}, 记录数: {len(result)}")
                            return True
                        else:
                            warning(f"AKShare函数测试返回空数据: {func_name} - {description}")
                            
                    elif func_name == "stock_zh_a_spot_em":
                        # 测试实时数据API，但只获取前几行
                        result = func()
                        if result is not None and not result.empty:
                            info(f"AKShare函数测试成功: {func_name} - {description}, 记录数: {len(result)}")
                            return True
                        else:
                            warning(f"AKShare函数测试返回空数据: {func_name} - {description}")
                            
                except Exception as e:
                    warning(f"AKShare函数测试出错: {func_name} - {description}, 错误: {str(e)}")
                    continue
            
            warning("所有AKShare测试函数都失败")
            return False
            
        except Exception as e:
            warning(f"AKShare服务检查异常: {str(e)}")
            return False

    def _get_cached_data(self, cache_key: str, ttl_seconds: Optional[int] = None) -> Optional[Any]:
        """
        获取缓存数据的通用方法
        
        Args:
            cache_key: 缓存键
            ttl_seconds: 缓存过期时间（秒），如果不指定则使用默认值
            
        Returns:
            缓存的数据，如果缓存不存在或已过期则返回None
        """
        try:
            # 根据缓存键的类型选择合适的缓存管理器
            if cache_key.startswith("price_"):
                # 价格数据缓存
                parts = cache_key.split("_")
                if len(parts) >= 4:
                    symbol, period, start_date, end_date = parts[1], parts[2], parts[3], parts[4] if len(parts) > 4 else None
                    cached_data = get_price_cache(symbol, period, "daily")
                    if cached_data is not None:
                        info(f"从价格缓存获取数据: {cache_key}")
                        return cached_data
            elif cache_key.startswith("realtime_"):
                # 实时数据缓存
                symbol = cache_key.replace("realtime_", "")
                cached_data = get_price_cache(f"akshare_realtime_{symbol}", "1d", "realtime")
                if cached_data is not None:
                    info(f"从实时缓存获取数据: {cache_key}")
                    return cached_data
            elif cache_key.startswith("info_"):
                # 股票信息缓存
                symbol = cache_key.replace("info_", "")
                cached_data = get_stock_info_cache(symbol)
                if cached_data is not None:
                    info(f"从股票信息缓存获取数据: {cache_key}")
                    return cached_data
            elif cache_key.startswith("news_"):
                # 新闻数据缓存
                symbol = cache_key.replace("news_", "")
                cached_data = get_news_cache(symbol)
                if cached_data is not None:
                    info(f"从新闻缓存获取数据: {cache_key}")
                    return cached_data
            elif cache_key.startswith("financial_"):
                # 财务数据缓存
                parts = cache_key.split("_")
                if len(parts) >= 3:
                    symbol, report_type = parts[1], parts[2]
                    cached_data = get_financial_cache(symbol, report_type)
                    if cached_data is not None:
                        info(f"从财务缓存获取数据: {cache_key}")
                        return cached_data
            
            # 如果没有匹配到特定类型的缓存，使用通用API缓存
            cached_data = get_api_cache(cache_key)
            if cached_data is not None:
                info(f"从API缓存获取数据: {cache_key}")
                return cached_data
                
            return None
            
        except Exception as e:
            warning(f"获取缓存数据时出错: {cache_key}, 错误: {str(e)}")
            return None

    def _cache_data(self, cache_key: str, data: Any, ttl_seconds: Optional[int] = None) -> None:
        """
        设置缓存数据的通用方法
        
        Args:
            cache_key: 缓存键
            data: 要缓存的数据
            ttl_seconds: 缓存过期时间（秒），如果不指定则使用默认值
        """
        try:
            if data is None:
                warning(f"尝试缓存None数据: {cache_key}")
                return
            
            # 根据缓存键的类型选择合适的缓存管理器和TTL
            if cache_key.startswith("price_"):
                # 价格数据缓存（默认5分钟）
                parts = cache_key.split("_")
                if len(parts) >= 4:
                    symbol, period, start_date, end_date = parts[1], parts[2], parts[3], parts[4] if len(parts) > 4 else None
                    ttl = ttl_seconds or 300  # 默认5分钟
                    set_price_cache(symbol, period, "daily", data)
                    info(f"缓存价格数据: {cache_key}, TTL: {ttl}秒")
            elif cache_key.startswith("realtime_"):
                # 实时数据缓存（默认1分钟）
                symbol = cache_key.replace("realtime_", "")
                ttl = ttl_seconds or 60  # 默认1分钟
                set_price_cache(f"akshare_realtime_{symbol}", "1d", "realtime", data)
                info(f"缓存实时数据: {cache_key}, TTL: {ttl}秒")
            elif cache_key.startswith("info_"):
                # 股票信息缓存（默认1小时）
                symbol = cache_key.replace("info_", "")
                ttl = ttl_seconds or 3600  # 默认1小时
                set_stock_info_cache(symbol, data)
                info(f"缓存股票信息: {cache_key}, TTL: {ttl}秒")
            elif cache_key.startswith("news_"):
                # 新闻数据缓存（默认30分钟）
                symbol = cache_key.replace("news_", "")
                ttl = ttl_seconds or 1800  # 默认30分钟
                set_news_cache(symbol, data)
                info(f"缓存新闻数据: {cache_key}, TTL: {ttl}秒")
            elif cache_key.startswith("financial_"):
                # 财务数据缓存（默认24小时）
                parts = cache_key.split("_")
                if len(parts) >= 3:
                    symbol, report_type = parts[1], parts[2]
                    ttl = ttl_seconds or 86400  # 默认24小时
                    set_financial_cache(symbol, data, report_type)
                    info(f"缓存财务数据: {cache_key}, TTL: {ttl}秒")
            else:
                # 通用API缓存（默认1小时）
                ttl = ttl_seconds or 3600
                set_api_cache(cache_key, data, ttl)
                info(f"缓存API数据: {cache_key}, TTL: {ttl}秒")
                
        except Exception as e:
            warning(f"设置缓存数据时出错: {cache_key}, 错误: {str(e)}")

    def _invalidate_cache(self, cache_key_pattern: str) -> None:
        """
        清除匹配模式的缓存
        
        Args:
            cache_key_pattern: 缓存键模式（支持前缀匹配）
        """
        try:
            # 使用统一的缓存清理方式
            if cache_key_pattern == "all":
                clear_all_caches()
                info("清除所有缓存")
            elif cache_key_pattern.startswith("price_"):
                price_cache_manager.clear("price")
                info(f"清除价格缓存: {cache_key_pattern}")
            elif cache_key_pattern.startswith("realtime_"):
                price_cache_manager.clear("akshare_realtime")
                info(f"清除实时缓存: {cache_key_pattern}")
            elif cache_key_pattern.startswith("info_"):
                stock_info_cache_manager.clear("stock_info")
                info(f"清除股票信息缓存: {cache_key_pattern}")
            elif cache_key_pattern.startswith("news_"):
                news_cache_manager.clear("news")
                info(f"清除新闻缓存: {cache_key_pattern}")
            elif cache_key_pattern.startswith("financial_"):
                financial_cache_manager.clear("financial")
                info(f"清除财务缓存: {cache_key_pattern}")
            else:
                api_cache_manager.clear(cache_key_pattern)
                info(f"清除API缓存: {cache_key_pattern}")
                
        except Exception as e:
            warning(f"清除缓存时出错: {cache_key_pattern}, 错误: {str(e)}")

    def _get_cache_stats(self) -> dict:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计信息字典
        """
        try:
            stats = {}
            
            try:
                stats["price_cache"] = price_cache_manager.get_stats()
                stats["stock_info_cache"] = stock_info_cache_manager.get_stats()
                stats["news_cache"] = news_cache_manager.get_stats()
                stats["financial_cache"] = financial_cache_manager.get_stats()
                stats["api_cache"] = api_cache_manager.get_stats()
                
                total_items = sum(s.get("total_items", 0) for s in stats.values())
                total_size = sum(s.get("estimated_size_bytes", 0) for s in stats.values())
                
                stats["summary"] = {
                    "total_cached_items": total_items,
                    "total_estimated_size_bytes": total_size,
                    "total_size_mb": round(total_size / (1024 * 1024), 2)
                }
            except Exception as import_error:
                stats["error"] = f"缓存管理器不可用: {str(import_error)}"
            
            return stats
            
        except Exception as e:
            warning(f"获取缓存统计信息时出错: {str(e)}")
            return {"error": str(e)}

    def _log_api_call(self, operation: str, symbol: str, success: bool, 
                      duration: Optional[float] = None, data_size: Optional[int] = None,
                      error_msg: Optional[str] = None) -> None:
        """
        记录API调用日志
        
        Args:
            operation: 操作类型
            symbol: 股票代码
            success: 是否成功
            duration: 耗时（秒）
            data_size: 数据大小（行数）
            error_msg: 错误信息
        """
        try:
            if success:
                log_msg = f"AKShare API调用成功 - 操作: {operation}, 股票: {symbol}"
                if duration is not None:
                    log_msg += f", 耗时: {duration:.2f}秒"
                if data_size is not None:
                    log_msg += f", 数据量: {data_size}行"
                info(log_msg)
            else:
                log_msg = f"AKShare API调用失败 - 操作: {operation}, 股票: {symbol}"
                if duration is not None:
                    log_msg += f", 耗时: {duration:.2f}秒"
                if error_msg:
                    log_msg += f", 错误: {error_msg}"
                error(log_msg)
        except Exception as e:
            warning(f"记录API调用日志时出错: {str(e)}")

    def _log_cache_operation(self, operation: str, cache_key: str, hit: bool = None,
                           ttl: Optional[int] = None) -> None:
        """
        记录缓存操作日志
        
        Args:
            operation: 操作类型 (get, set, clear, stats)
            cache_key: 缓存键
            hit: 是否命中缓存（仅对get操作有效）
            ttl: 缓存时间（秒，仅对set操作有效）
        """
        try:
            if operation == "get":
                if hit is True:
                    info(f"缓存命中 - 键: {cache_key}")
                elif hit is False:
                    debug(f"缓存未命中 - 键: {cache_key}")
            elif operation == "set":
                log_msg = f"缓存设置 - 键: {cache_key}"
                if ttl is not None:
                    log_msg += f", TTL: {ttl}秒"
                info(log_msg)
            elif operation == "clear":
                info(f"缓存清除 - 模式: {cache_key}")
            elif operation == "stats":
                debug(f"缓存统计 - 键: {cache_key}")
        except Exception as e:
            warning(f"记录缓存操作日志时出错: {str(e)}")

    def _log_rate_limit(self, wait_time: float) -> None:
        """
        记录频率控制日志
        
        Args:
            wait_time: 等待时间（秒）
        """
        try:
            if wait_time > 0:
                info(f"频率控制 - 等待时间: {wait_time:.2f}秒")
            else:
                debug(f"频率控制 - 无需等待")
        except Exception as e:
            warning(f"记录频率控制日志时出错: {str(e)}")

    def _log_network_check(self, url: str, success: bool, duration: float,
                          status_code: Optional[int] = None) -> None:
        """
        记录网络检查日志
        
        Args:
            url: 检查的URL
            success: 是否成功
            duration: 耗时（秒）
            status_code: HTTP状态码
        """
        try:
            if success:
                info(f"网络检查成功 - URL: {url}, 耗时: {duration:.2f}秒, 状态码: {status_code}")
            else:
                warning(f"网络检查失败 - URL: {url}, 耗时: {duration:.2f}秒")
        except Exception as e:
            warning(f"记录网络检查日志时出错: {str(e)}")

    def _log_service_check(self, service_name: str, success: bool,
                          test_function: Optional[str] = None,
                          error_msg: Optional[str] = None) -> None:
        """
        记录服务检查日志
        
        Args:
            service_name: 服务名称
            success: 是否成功
            test_function: 测试函数名
            error_msg: 错误信息
        """
        try:
            if success:
                log_msg = f"服务检查成功 - 服务: {service_name}"
                if test_function:
                    log_msg += f", 测试函数: {test_function}"
                info(log_msg)
            else:
                log_msg = f"服务检查失败 - 服务: {service_name}"
                if test_function:
                    log_msg += f", 测试函数: {test_function}"
                if error_msg:
                    log_msg += f", 错误: {error_msg}"
                warning(log_msg)
        except Exception as e:
            warning(f"记录服务检查日志时出错: {str(e)}")

    def _get_fallback_data(self, data_type: str, symbol: str) -> None | DataFrame | dict[str, str] | dict[str, str | int | float]:
        """
        获取备用数据
        
        Args:
            data_type: 数据类型 (price, realtime, info, news, financial)
            symbol: 股票代码
            
        Returns:
            备用数据DataFrame，失败返回None
        """
        try:
            warning(f"获取备用数据: {data_type}, {symbol}")
            
            if data_type == "price":
                # 返回基本的价格数据结构
                return pd.DataFrame({
                    'Date': [pd.Timestamp.now()],
                    'Open': [0.0],
                    'High': [0.0],
                    'Low': [0.0],
                    'Close': [0.0],
                    'Volume': [0]
                }).set_index('Date')
            elif data_type == "realtime":
                # 返回基本的实时数据结构
                return {
                    'symbol': symbol,
                    'name': '数据获取失败',
                    'current_price': 0.0,
                    'change_percent': 0.0,
                    'volume': 0,
                    'update_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            elif data_type == "info":
                # 返回基本的股票信息结构
                return {
                    'symbol': symbol,
                    'name': '数据获取失败',
                    'industry': '未知',
                    'market': '未知'
                }
            elif data_type == "news":
                # 返回基本的新闻数据结构
                return pd.DataFrame({
                    'title': ['数据获取失败'],
                    'time': [pd.Timestamp.now()],
                    'url': ['']
                })
            elif data_type == "financial":
                # 返回基本的财务数据结构
                return pd.DataFrame({
                    'report_date': [pd.Timestamp.now()],
                    'revenue': [0.0],
                    'net_profit': [0.0]
                }).set_index('report_date')
            else:
                return None
                
        except Exception as e:
            error(f"获取备用数据失败: {data_type}, {symbol}, 错误: {str(e)}")
            return None

    def _is_a_stock(self, stock_code: str) -> bool:
        """
        判断是否为A股代码
        
        Args:
            stock_code: 股票代码
            
        Returns:
            是否为A股代码
        """
        # A股代码通常为6位数字或带有市场前缀如sh600000、sz000000
        stock_code = stock_code.lower()
        # 匹配常见的A股代码格式
        return (stock_code.startswith('sh') or stock_code.startswith('sz') or
                (stock_code.isdigit() and len(stock_code) == 6))

    def _normalize_stock_code(self, stock_code: str) -> str:
        """
        标准化A股代码格式
        
        Args:
            stock_code: 股票代码
            
        Returns:
            标准化后的股票代码（带sh/sz前缀）
        """
        stock_code = stock_code.lower().strip()
        # 如果只有6位数字，根据规则添加前缀
        if stock_code.isdigit() and len(stock_code) == 6:
            # 600000-699999: 上海证券交易所
            # 000000-009999: 深圳证券交易所主板
            # 300000-309999: 深圳证券交易所创业板
            # 688000-688999: 上海证券交易所科创板
            if stock_code.startswith(('60', '68')):
                return f'sh{stock_code}'
            else:
                return f'sz{stock_code}'
        return stock_code

    def _get_pure_stock_code(self, stock_code: str) -> str:
        """
        获取纯数字股票代码（去掉市场前缀）
        
        Args:
            stock_code: 股票代码（带或不带前缀）
            
        Returns:
            纯数字股票代码（6位）
        """
        # 先标准化代码
        normalized_code = self._normalize_stock_code(stock_code)
        
        # 去掉sh/sz前缀，返回纯数字代码
        if normalized_code.startswith(('sh', 'sz')):
            pure_code = normalized_code[2:]
        else:
            pure_code = normalized_code
            
        # 如果包含点号，去掉点号和前面的部分
        if '.' in pure_code:
            pure_code = pure_code.split('.')[-1]
            
        return pure_code

    def get_stock_price_data(self, stock_code: str, period: str = "1y",
                             interval: str = "1d") -> DataFrame | None:
        """
        获取A股价格数据
        
        Args:
            stock_code: 股票代码
            period: 时间周期
            interval: 数据间隔
            
        Returns:
            价格数据DataFrame
        """
        if not self._is_a_stock(stock_code):
            debug(f"{stock_code} 不是A股代码，跳过AKShare数据源")
            return None

        normalized_code = self._normalize_stock_code(stock_code)
        pure_code = self._get_pure_stock_code(normalized_code)

        # 生成缓存键
        cache_key = f"price_{normalized_code}_{period}_{interval}"
        
        # 尝试从缓存获取数据
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            self._log_cache_operation("hit", cache_key, hit=True)
            return cached_data

        # 检查网络连接
        if not self._check_connection():
            error("网络连接检查失败，无法获取股票价格数据")
            return self._get_fallback_data("price", stock_code)

        # 检查AKShare服务可用性
        if not self._check_akshare_service():
            error("AKShare服务不可用，无法获取股票价格数据")
            return self._get_fallback_data("price", stock_code)

        try:
            info(f"获取股票价格数据: {normalized_code}, 周期: {period}, 间隔: {interval}")
            
            # 等待频率控制
            self._wait_for_rate_limit()
            
            # 使用最新的AKShare API获取历史数据
            def fetch_stock_data():
                return ak.stock_zh_a_hist(
                    symbol=pure_code,
                    period="daily",
                    start_date="19700101",
                    end_date="20500101",
                    adjust=""
                )
            
            data = self._retry_api_call(fetch_stock_data)
            
            if data is not None and not data.empty:
                # 标准化列名，使用最新的AKShare列名映射
                column_mapping = {
                    '日期': 'Date',
                    '开盘': 'Open',
                    '收盘': 'Close',
                    '最高': 'High',
                    '最低': 'Low',
                    '成交量': 'Volume',
                    '成交额': 'Turnover',
                    '振幅': 'Amplitude',
                    '涨跌幅': 'ChangePercent',
                    '涨跌额': 'ChangeAmount',
                    '换手率': 'TurnoverRate'
                }
                
                # 重命名存在的列
                existing_columns = {k: v for k, v in column_mapping.items() if k in data.columns}
                data = data.rename(columns=existing_columns)
                
                # 确保日期列是datetime类型并设置为索引
                if 'Date' in data.columns:
                    data['Date'] = pd.to_datetime(data['Date'])
                    data = data.sort_values('Date').set_index('Date')
                
                # 根据period筛选数据
                if period == "1d":
                    data = data.tail(1)
                elif period == "1wk":
                    data = data.tail(5)
                elif period == "1mo":
                    data = data.tail(20)
                elif period == "3mo":
                    data = data.tail(60)
                elif period == "6mo":
                    data = data.tail(120)
                elif period == "1y":
                    data = data.tail(250)
                elif period == "2y":
                    data = data.tail(500)
                elif period == "5y":
                    data = data.tail(1250)
                elif period == "max":
                    pass
                
                # 缓存数据
                self._cache_data(cache_key, data, ttl_seconds=300)  # 价格数据缓存5分钟
                self._log_cache_operation("set", cache_key, hit=True, ttl=300)
                
                info(f"成功获取股票价格数据: {normalized_code}, 数据量: {len(data)} 行")
                return data
            else:
                error(f"AKShare返回空数据: {normalized_code}")
                return self._get_fallback_data("price", stock_code)
                
        except Exception as e:
            error(f"获取股票价格数据失败: {normalized_code}, 错误: {str(e)}")
            print_log_exception()
            return self._get_fallback_data("price", stock_code)

    def get_stock_realtime_data(self, stock_code: str) -> dict[str, str | Any] | None:
        """
        获取A股实时数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            实时数据字典
        """
        if not self._is_a_stock(stock_code):
            debug(f"{stock_code} 不是A股代码，跳过AKShare数据源")
            return None

        normalized_code = self._normalize_stock_code(stock_code)
        pure_code = self._get_pure_stock_code(normalized_code)

        # 生成缓存键
        cache_key = f"realtime_{normalized_code}"
        
        # 尝试从缓存获取数据
        cached_realtime = self._get_cached_data(cache_key)
        if cached_realtime is not None:
            self._log_cache_operation("hit", cache_key, hit=True)
            return cached_realtime

        try:
            # 检查网络连接
            if not self._check_connection():
                error("网络连接检查失败，无法获取股票实时数据")
                return self._get_fallback_data("realtime", stock_code)

            # 检查AKShare服务可用性
            if not self._check_akshare_service():
                error("AKShare服务不可用，无法获取股票实时数据")
                return self._get_fallback_data("realtime", stock_code)

            self._wait_for_rate_limit()

            # 获取实时行情数据
            def fetch_realtime_data():
                return ak.stock_zh_a_spot_em()
            
            result = self._retry_api_call(fetch_realtime_data)

            if result is None or result.empty:
                warning(f"未获取到实时数据: {normalized_code}，返回备用数据")
                return self._get_fallback_data("realtime", stock_code)

            # 尝试多种匹配方式查找对应股票的数据
            stock_data = None
            
            # 方式1: 直接匹配代码
            if '代码' in result.columns:
                mask = result['代码'] == pure_code
                if mask.any():
                    stock_data = result[mask]
            
            # 方式2: 匹配完整代码
            if stock_data is None or stock_data.empty:
                if '代码' in result.columns:
                    mask = result['代码'] == normalized_code
                    if mask.any():
                        stock_data = result[mask]
            
            # 方式3: 包含匹配
            if stock_data is None or stock_data.empty:
                if '代码' in result.columns:
                    mask = result['代码'].astype(str).str.contains(pure_code)
                    if mask.any():
                        stock_data = result[mask]

            if stock_data is None or stock_data.empty:
                warning(f"未找到股票 {normalized_code} 的实时数据，返回备用数据")
                return self._get_fallback_data("realtime", stock_code)

            # 提取实时数据
            row = stock_data.iloc[0]
            realtime_data = {
                'symbol': normalized_code,
                'name': row.get('名称', ''),
                'current_price': row.get('最新价', 0),
                'open_price': row.get('今开', 0),
                'high_price': row.get('最高', 0),
                'low_price': row.get('最低', 0),
                'volume': row.get('成交量', 0),
                'turnover': row.get('成交额', 0),
                'change_percent': row.get('涨跌幅', 0),
                'change_amount': row.get('涨跌额', 0),
                'amplitude': row.get('振幅', 0),
                'turnover_rate': row.get('换手率', 0),
                'pe_ratio': row.get('市盈率-动态', 0),
                'pb_ratio': row.get('市净率', 0),
                'market_cap': row.get('总市值', 0),
                'flow_market_cap': row.get('流通市值', 0),
                'update_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            info(f"成功从AKShare获取 {normalized_code} 实时数据, 当前价格: {realtime_data['current_price']}")

            # 缓存结果（实时数据缓存时间较短）
            self._cache_data(cache_key, realtime_data, ttl_seconds=60)  # 缓存1分钟
            self._log_cache_operation("set", cache_key, hit=True, ttl=60)
            return realtime_data

        except Exception as e:
            error(f"从AKShare获取 {normalized_code} 实时数据时出错: {str(e)}")
            print_log_exception()
            # 返回备用数据
            return self._get_fallback_data("realtime", stock_code)

    def get_stock_info(self, stock_code: str) -> dict[str, str | Any] | None:
        """
        获取A股基本信息（包含行业、板块等）
        
        Args:
            stock_code: 股票代码
            
        Returns:
            股票基本信息字典
        """
        if not self._is_a_stock(stock_code):
            info(f"{stock_code} 不是A股代码，跳过AKShare数据源")
            return None

        normalized_code = self._normalize_stock_code(stock_code)
        pure_code = self._get_pure_stock_code(normalized_code)

        # 尝试从缓存获取数据
        cached_data = self._get_cached_data(f"info_{normalized_code}")
        if cached_data is not None:
            self._log_cache_operation("hit", f"info_{normalized_code}", hit=True)
            debug(f"从缓存获取股票信息: {normalized_code}")
            return cached_data

        try:
            # 检查网络连接
            if not self._check_connection():
                warning(f"网络连接不可用，返回 {normalized_code} 备用数据")
                return self._get_fallback_data("info", stock_code)

            # 检查AKShare服务可用性
            if not self._check_akshare_service():
                warning(f"AKShare服务不可用，返回 {normalized_code} 备用数据")
                return self._get_fallback_data("info", stock_code)

            self._wait_for_rate_limit()

            # 构建默认的股票信息字典
            stock_info = {
                'symbol': normalized_code,
                'name': normalized_code,
                'market': 'A股',
                'industry': '',
                'sector': '',
                'full_name': ''
            }

            # 尝试多个不同的API来获取股票基本信息
            result = None
            try:
                # 方法1: 使用stock_individual_info_em
                def fetch_info_em():
                    return ak.stock_individual_info_em(symbol=pure_code, timeout=10)
                
                result = self._retry_api_call(fetch_info_em)

                # 方法2: 如果方法1失败，尝试stock_individual_basic_info_xq作为备选
                if result is None:
                    def fetch_basic_info():
                        return ak.stock_individual_basic_info_xq(symbol=normalized_code)
                    
                    result = self._retry_api_call(fetch_basic_info)
                        
            except Exception as inner_e:
                error(f"调用AKShare API时发生异常: {str(inner_e)}")
                print_log_exception()
                result = None

            # 处理API返回结果
            if result is not None:
                try:
                    if isinstance(result, pd.DataFrame) and not result.empty:
                        # 处理DataFrame格式
                        if 'item' in result.columns and 'value' in result.columns:
                            # 处理stock_individual_info_em返回的格式（item, value列）
                            info_dict = dict(zip(result['item'], result['value']))
                            stock_info['name'] = info_dict.get('股票名称', normalized_code)
                            stock_info['industry'] = info_dict.get('所属行业', '')
                            stock_info['sector'] = info_dict.get('所属板块', '')
                            stock_info['full_name'] = info_dict.get('公司名称', '')
                        else:
                            # 处理其他格式的DataFrame
                            if '股票名称' in result.columns:
                                stock_info['name'] = result['股票名称'].iloc[0] if len(result) > 0 else normalized_code
                            if '所属行业' in result.columns:
                                stock_info['industry'] = result['所属行业'].iloc[0] if len(result) > 0 else ''
                            if '所属板块' in result.columns:
                                stock_info['sector'] = result['所属板块'].iloc[0] if len(result) > 0 else ''
                            if '公司名称' in result.columns:
                                stock_info['full_name'] = result['公司名称'].iloc[0] if len(result) > 0 else ''
                    elif isinstance(result, dict):
                        # 处理字典格式
                        stock_info['name'] = result.get('股票名称', stock_info['name'])
                        stock_info['industry'] = result.get('所属行业', stock_info['industry'])
                        stock_info['sector'] = result.get('所属板块', stock_info['sector'])
                        stock_info['full_name'] = result.get('公司名称', stock_info['full_name'])
                        
                        # 尝试从data键中获取信息
                        if 'data' in result and result['data'] is not None:
                            data = result['data']
                            if isinstance(data, dict):
                                stock_info['name'] = data.get('股票名称', stock_info['name'])
                                stock_info['industry'] = data.get('所属行业', stock_info['industry'])
                                stock_info['sector'] = data.get('所属板块', stock_info['sector'])
                                stock_info['full_name'] = data.get('公司名称', stock_info['full_name'])
                            elif isinstance(data, pd.DataFrame) and not data.empty:
                                if '股票名称' in data.columns:
                                    stock_info['name'] = data['股票名称'].iloc[0]
                                if '所属行业' in data.columns:
                                    stock_info['industry'] = data['所属行业'].iloc[0]
                                if '所属板块' in data.columns:
                                    stock_info['sector'] = data['所属板块'].iloc[0]
                                if '公司名称' in data.columns:
                                    stock_info['full_name'] = data['公司名称'].iloc[0]
                            elif isinstance(data, list) and len(data) > 0:
                                # 如果data是列表，尝试从第一个元素获取信息
                                first_item = data[0]
                                if isinstance(first_item, dict):
                                    stock_info['name'] = first_item.get('股票名称', stock_info['name'])
                                    stock_info['industry'] = first_item.get('所属行业', stock_info['industry'])
                                    stock_info['sector'] = first_item.get('所属板块', stock_info['sector'])
                                    stock_info['full_name'] = first_item.get('公司名称', stock_info['full_name'])
                    else:
                        warning(f"从AKShare获取 {normalized_code} 基本信息返回未知格式: {type(result)}")
                        
                except Exception as process_e:
                    error(f"处理AKShare返回数据时发生异常: {str(process_e)}")
                    print_log_exception()
                    # 继续执行，使用已有默认值
            else:
                warning(f"AKShare返回None，返回 {normalized_code} 备用数据")
                return self._get_fallback_data("info", stock_code)

            # 获取最新行情信息以补充
            try:
                stock_spot_data = None
                
                # 尝试多个API获取实时行情
                try:
                    stock_spot_data = ak.stock_zh_a_spot()
                except Exception as spot1_e:
                    debug(f"stock_zh_a_spot API失败: {str(spot1_e)}")
                    try:
                        stock_spot_data = ak.stock_zh_a_spot_em()
                    except Exception as spot2_e:
                        debug(f"stock_zh_a_spot_em API也失败: {str(spot2_e)}")
                        stock_spot_data = None
                
                if stock_spot_data is not None and not stock_spot_data.empty:
                    stock_info_row = stock_spot_data[stock_spot_data['代码'] == pure_code]
                    if not stock_info_row.empty:
                        row = stock_info_row.iloc[0]
                        stock_info['current_price'] = row.get('现价', 0)
                        stock_info['change_percent'] = row.get('涨跌幅', 0)
                        stock_info['volume'] = row.get('成交量', 0)
                        
            except Exception as e:
                error(f"获取最新行情信息时出错: {str(e)}")
                print_log_exception()

            info(f"成功从AKShare获取 {normalized_code} 基本信息")

            # 更新缓存
            self._cache_data(f"info_{normalized_code}", stock_info, ttl_seconds=3600)  # 缓存1小时
            self._log_cache_operation("set", f"info_{normalized_code}", hit=True, ttl=3600)
            debug(f"更新缓存: {normalized_code} 股票信息")

            return stock_info

        except Exception as e:
            error(f"从AKShare获取 {normalized_code} 基本信息时出错: {str(e)}")
            print_log_exception()
            # 返回备用数据
            return self._get_fallback_data("info", stock_code)

    def get_stock_news(self, stock_code: str, limit: int = 5) -> list[dict[str, str | Any] | dict[str, str | Any]] | None:
        """
        获取A股相关新闻
        
        Args:
            stock_code: 股票代码
            limit: 返回新闻数量限制
            
        Returns:
            新闻列表
        """
        if not self._is_a_stock(stock_code):
            debug(f"{stock_code} 不是A股代码，跳过AKShare数据源")
            return None

        normalized_code = self._normalize_stock_code(stock_code)

        # 检查缓存
        cached_news = self._get_cached_data(f"news_{normalized_code}_{limit}")
        if cached_news:
            self._log_cache_operation("hit", f"news_{normalized_code}_{limit}", hit=True)
            debug(f"使用缓存的新闻数据: {normalized_code}")
            return cached_news

        try:
            # 检查网络连接
            if not self._check_connection():
                warning(f"网络连接不可用，返回 {normalized_code} 备用数据")
                return self._get_fallback_data("news", stock_code)

            # 检查AKShare服务可用性
            if not self._check_akshare_service():
                warning(f"AKShare服务不可用，返回 {normalized_code} 备用数据")
                return self._get_fallback_data("news", stock_code)

            self._wait_for_rate_limit()

            # 获取股票名称用于搜索新闻
            stock_info = self.get_stock_info(stock_code)
            stock_name = stock_info.get('name', normalized_code) if stock_info else normalized_code

            # 获取财经新闻
            news_list = []

            # 获取新浪财经滚动新闻
            def fetch_stock_news():
                return ak.stock_news_em(symbol=normalized_code)
            
            stock_news_df = self._retry_api_call(fetch_stock_news)
            
            if stock_news_df is not None and not stock_news_df.empty:
                for _, row in stock_news_df.head(limit).iterrows():
                    news_list.append({
                        'title': row.get('title', ''),
                        'summary': row.get('content', ''),
                        'url': row.get('url', ''),
                        'published_at': row.get('datetime', ''),
                        'source': '新浪财经'
                    })

            # 如果新闻数量不足，获取宏观财经新闻
            if len(news_list) < limit:
                def fetch_cctv_news():
                    return ak.news_cctv()
                
                cctv_news_df = self._retry_api_call(fetch_cctv_news)
                
                if cctv_news_df is not None and not cctv_news_df.empty:
                    for _, row in cctv_news_df.head(10).iterrows():
                        # 尝试匹配与股票相关的新闻
                        if stock_name in row.get('title', '') or stock_name in row.get('content', ''):
                            news_list.append({
                                'title': row.get('title', ''),
                                'summary': row.get('content', ''),
                                'url': row.get('url', ''),
                                'published_at': row.get('datetime', ''),
                                'source': 'CCTV'
                            })
                        if len(news_list) >= limit:
                                break

            if not news_list:
                warning(f"从AKShare获取 {normalized_code} 新闻数据为空，返回备用数据")
                return self._get_fallback_data("news", stock_code)

            info(f"成功从AKShare获取 {normalized_code} 相关新闻，返回 {len(news_list)} 条")

            # 缓存结果
            self._cache_data(f"news_{normalized_code}_{limit}", news_list[:limit], ttl_seconds=1800)  # 缓存30分钟
            self._log_cache_operation("set", f"news_{normalized_code}_{limit}", hit=True, ttl=1800)
            return news_list[:limit]

        except Exception as e:
            error(f"从AKShare获取 {normalized_code} 新闻时出错: {str(e)}")
            print_log_exception()
            # 返回备用数据
            return self._get_fallback_data("news", stock_code)

    def get_financial_data(self, stock_code: str) -> dict[str, DataFrame] | None:
        """
        获取A股财务数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            包含财务报表的字典
        """
        if not self._is_a_stock(stock_code):
            debug(f"{stock_code} 不是A股代码，跳过AKShare数据源")
            return None

        normalized_code = self._normalize_stock_code(stock_code)
        pure_code = self._get_pure_stock_code(normalized_code)

        # 检查缓存
        cached_financial = self._get_cached_data(f"financial_{normalized_code}_all_3")
        if cached_financial:
            self._log_cache_operation("hit", f"financial_{normalized_code}_all_3", hit=True)
            debug(f"使用缓存的财务数据: {normalized_code}")
            return cached_financial

        try:
            # 检查网络连接
            if not self._check_connection():
                warning(f"网络连接不可用，返回 {normalized_code} 备用数据")
                return self._get_fallback_data("financial", stock_code)

            # 检查AKShare服务可用性
            if not self._check_akshare_service():
                warning(f"AKShare服务不可用，返回 {normalized_code} 备用数据")
                return self._get_fallback_data("financial", stock_code)

            self._wait_for_rate_limit()

            financial_data = {}

            # 获取资产负债表
            def fetch_balance_sheet():
                return ak.stock_balance_sheet_by_yearly_em(symbol=pure_code)
            
            balance_sheet_data = self._retry_api_call(fetch_balance_sheet)
            
            if balance_sheet_data is not None and isinstance(balance_sheet_data, pd.DataFrame) and not balance_sheet_data.empty:
                balance_sheet_df = balance_sheet_data.copy()
                if '报表日期' in balance_sheet_df.columns:
                    balance_sheet_df['报表日期'] = pd.to_datetime(balance_sheet_df['报表日期'])
                    balance_sheet_df.set_index('报表日期', inplace=True)
                    financial_data['balance_sheet'] = balance_sheet_df

            # 获取利润表
            def fetch_income_statement():
                return ak.stock_profit_sheet_by_yearly_em(symbol=pure_code)
            
            income_statement_data = self._retry_api_call(fetch_income_statement)
            
            if income_statement_data is not None and isinstance(income_statement_data, pd.DataFrame) and not income_statement_data.empty:
                income_statement_df = income_statement_data.copy()
                if '报表日期' in income_statement_df.columns:
                    income_statement_df['报表日期'] = pd.to_datetime(income_statement_df['报表日期'])
                    income_statement_df.set_index('报表日期', inplace=True)
                    financial_data['income_statement'] = income_statement_df

            # 获取现金流量表
            def fetch_cash_flow():
                return ak.stock_cash_flow_sheet_by_yearly_em(symbol=pure_code)
            
            cash_flow_data = self._retry_api_call(fetch_cash_flow)
            
            if cash_flow_data is not None and isinstance(cash_flow_data, pd.DataFrame) and not cash_flow_data.empty:
                cash_flow_df = cash_flow_data.copy()
                if '报表日期' in cash_flow_df.columns:
                    cash_flow_df['报表日期'] = pd.to_datetime(cash_flow_df['报表日期'])
                    cash_flow_df.set_index('报表日期', inplace=True)
                    financial_data['cash_flow'] = cash_flow_df

            # 获取财务指标
            def fetch_financial_indicators():
                return ak.stock_financial_analysis_indicator(symbol=pure_code)
            
            financial_indicator_data = self._retry_api_call(fetch_financial_indicators)
            
            if financial_indicator_data is not None and isinstance(financial_indicator_data, pd.DataFrame) and not financial_indicator_data.empty:
                financial_indicator_df = financial_indicator_data.copy()
                date_col = None
                if 'trade_date' in financial_indicator_df.columns:
                    date_col = 'trade_date'
                elif '报表日期' in financial_indicator_df.columns:
                    date_col = '报表日期'
                
                if date_col:
                    financial_indicator_df[date_col] = pd.to_datetime(financial_indicator_df[date_col])
                    financial_indicator_df.set_index(date_col, inplace=True)
                    financial_data['financial_indicators'] = financial_indicator_df

            # 验证所有数据都是有效的DataFrame
            valid_financial_data = {}
            for key, df in financial_data.items():
                if isinstance(df, pd.DataFrame) and not df.empty and not df.index.empty and len(df.columns) > 0:
                    valid_financial_data[key] = df
                else:
                    debug(f"跳过无效的财务数据: {key}")

            if not valid_financial_data:
                warning(f"从AKShare获取 {normalized_code} 财务数据失败，所有报表均为空或格式无效，返回备用数据")
                return self._get_fallback_data("financial", stock_code)

            info(f"成功从AKShare获取 {normalized_code} 财务数据，包含 {len(valid_financial_data)} 种报表")

            # 缓存结果
            self._cache_data(f"financial_{normalized_code}_all_3", valid_financial_data, ttl_seconds=7200)  # 缓存2小时
            self._log_cache_operation("set", f"financial_{normalized_code}_all_3", hit=True, ttl=7200)
            return valid_financial_data

        except Exception as e:
            error(f"从AKShare获取 {normalized_code} 财务数据时出错: {str(e)}")
            print_log_exception()
            # 返回备用数据
            return self._get_fallback_data("financial", stock_code)
