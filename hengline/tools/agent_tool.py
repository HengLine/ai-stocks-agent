#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
智能体工具类
提供各智能体共用的功能，如数据获取、图表生成、格式转换等
"""

import base64
import json
import os
import time
import random
import functools
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict, Any, List, Optional, Callable, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from hengline.client.client_factory import ClientFactory
from hengline.logger import debug, error, warning
from hengline.tools.llama_index_retriever import DocumentRetriever

# 用于API请求节流的全局变量
_last_request_time = 0
_MIN_REQUEST_INTERVAL = 3.0  # 增加到3秒

# 重试装饰器配置
_MAX_RETRIES = 5
_RETRY_DELAY_BASE = 3.0  # 增加到3秒

# 连续失败计数
_consecutive_failures = 0
_MAX_CONSECUTIVE_FAILURES = 3
_COOLDOWN_PERIOD = 10.0  # 连续失败后的冷却时间（秒）

def with_retry_and_throttling(func: Callable) -> Callable:
    """
    为函数添加重试机制和请求节流的装饰器
    
    Args:
        func: 要装饰的函数
        
    Returns:
        Callable: 装饰后的函数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global _last_request_time, _consecutive_failures
        retry_count = 0
        last_exception = None
        
        # 检查是否需要冷却
        if _consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            debug(f"检测到连续失败 {_consecutive_failures} 次，执行冷却 {_COOLDOWN_PERIOD} 秒")
            time.sleep(_COOLDOWN_PERIOD)
            _consecutive_failures = 0  # 重置连续失败计数
        
        # 应用请求节流
        current_time = time.time()
        time_since_last_request = current_time - _last_request_time
        if time_since_last_request < _MIN_REQUEST_INTERVAL:
            wait_time = _MIN_REQUEST_INTERVAL - time_since_last_request
            debug(f"请求节流: 等待 {wait_time:.2f} 秒")
            time.sleep(wait_time)
        
        # 更新最后请求时间
        _last_request_time = time.time()
        
        # 重试逻辑
        while retry_count <= _MAX_RETRIES:
            try:
                result = func(*args, **kwargs)
                # 请求成功，重置连续失败计数
                _consecutive_failures = 0
                return result
            except Exception as e:
                last_exception = e
                retry_count += 1
                
                # 检查是否是速率限制错误
                error_msg = str(e)
                if "Too Many Requests" in error_msg or "Rate limited" in error_msg or "rate limit" in error_msg.lower():
                    _consecutive_failures += 1  # 增加连续失败计数
                    
                    if retry_count <= _MAX_RETRIES:
                        # 指数退避 + 随机因子，避免固定间隔导致的请求同步
                        wait_time = _RETRY_DELAY_BASE * (2 ** (retry_count - 1)) * (0.9 + 0.2 * random.random())
                        debug(f"API速率限制，{retry_count}/{_MAX_RETRIES} 次重试，等待 {wait_time:.2f} 秒: {error_msg}")
                        time.sleep(wait_time)
                    else:
                        error(f"API速率限制，重试次数已达上限: {error_msg}")
                else:
                    # 非速率限制错误，直接抛出
                    _consecutive_failures = 0  # 非速率限制错误不增加连续失败计数
                    raise
        
        # 所有重试都失败
        raise last_exception
    
    return wrapper

class AgentTools:
    """
    智能体工具类，提供共用功能
    """

    def __init__(self):
        """
        初始化工具类
        """
        self.client_factory = ClientFactory()
        self.retriever = DocumentRetriever()
        self._setup_plt_style()

    def _setup_plt_style(self):
        """
        设置绘图风格
        """
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        # 设置默认风格
        sns.set_style("whitegrid")
        sns.set_palette("husl")

    def get_llm_client(self, model_type: str = "openai") -> Any:
        """
        获取LLM客户端
        
        Args:
            model_type: 模型类型

        Returns:
            Any: LLM客户端实例
        """
        try:
            return self.client_factory.create_client(model_type)
        except Exception as e:
            error(f"获取LLM客户端失败: {str(e)}")
            raise

    def get_langchain_llm(self, model_type: str = "openai", config: Optional[Dict[str, Any]] = None) -> Any:
        """
        获取LangChain兼容的LLM实例
        
        Args:
            model_type: 模型类型
            config: 额外参数
            
        Returns:
            Any: LangChain LLM实例
        """
        try:
            return self.client_factory.get_langchain_llm(model_type, config)
        except Exception as e:
            error(f"获取LangChain LLM失败: {str(e)}")
            raise

    def search_knowledge_base(self, query: str, top_k: int = 3, knowledge_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        搜索知识库
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量
            knowledge_dir: 知识库目录
            
        Returns:
            List[Dict[str, Any]]: 搜索结果
        """
        # 检查检索器是否初始化
        if not hasattr(self, 'retriever') or self.retriever is None:
            debug("检索器未初始化，无法执行知识库搜索")
            return []

        try:
            results = self.retriever.retrieve(query, top_k=top_k)

            # 格式化结果
            formatted_results = []
            for result in results:
                # 适配不同的结果格式
                content = None
                score = 0.0
                metadata = {}

                if hasattr(result, 'node'):
                    # LlamaIndex NodeWithScore格式
                    content = result.node.get_content() if hasattr(result.node, 'get_content') else str(result.node)
                    score = float(result.score) if hasattr(result, 'score') else 0.0
                    metadata = result.node.metadata if hasattr(result.node, 'metadata') else {}
                elif isinstance(result, dict):
                    # 字典格式
                    content = result.get('content', '') or result.get('text', '') or str(result)
                    score = float(result.get('score', 0.0))
                    metadata = result.get('metadata', {})
                else:
                    # 其他格式
                    content = str(result)

                formatted_results.append({
                    "content": content,
                    "score": score,
                    "metadata": metadata
                })

            debug(f"知识库搜索完成，找到 {len(formatted_results)} 条相关信息")
            return formatted_results
        except Exception as e:
            error(f"知识库搜索失败: {str(e)}")
            return []

    @with_retry_and_throttling
    def get_stock_price_data(self, stock_code: str, period: str = "1y") -> pd.DataFrame:
        """
        获取股票价格数据
        
        Args:
            stock_code: 股票代码
            period: 时间周期
            
        Returns:
            pd.DataFrame: 价格数据
        """
        try:
            import yfinance as yf

            # 转换周期为yfinance的period参数
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
                "max": "max"
            }
            yf_period = period_map.get(period, "1y")

            # 获取数据
            ticker = yf.Ticker(stock_code)
            df = ticker.history(period=yf_period)

            # 重置索引
            if not df.empty:
                df = df.reset_index()
                # 确保列名小写
                df.columns = [col.lower() for col in df.columns]

            debug(f"获取股票 {stock_code} 价格数据完成，数据量: {len(df)} 行")
            return df
        except ImportError:
            warning("未安装yfinance库，使用模拟数据")
            return self._get_mock_price_data(stock_code, period)
        except Exception as e:
            error(f"获取股票价格数据失败: {str(e)}")
            # 重试机制会处理速率限制错误，这里返回空DataFrame
            return pd.DataFrame()

    @with_retry_and_throttling
    def get_financial_data(self, stock_code: str) -> Dict[str, Any]:
        """
        获取财务数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Dict[str, Any]: 财务数据
        """
        try:
            import yfinance as yf

            ticker = yf.Ticker(stock_code)

            financial_data = {
                "income_statement": ticker.financials,
                "balance_sheet": ticker.balance_sheet,
                "cash_flow": ticker.cashflow,
                "quarterly_income_statement": ticker.quarterly_financials,
                "quarterly_balance_sheet": ticker.quarterly_balance_sheet,
                "quarterly_cash_flow": ticker.quarterly_cashflow
            }

            # 转换为字典格式便于处理
            result = {}
            for key, df in financial_data.items():
                if not df.empty:
                    # 确保索引和列正确
                    result[key] = df.T.to_dict()
                else:
                    result[key] = {}

            debug(f"获取股票 {stock_code} 财务数据完成")
            return result
        except ImportError:
            warning("未安装yfinance库，使用模拟数据")
            return self._get_mock_financial_data(stock_code)
        except Exception as e:
            error(f"获取财务数据失败: {str(e)}")
            # 重试机制会处理速率限制错误，这里返回空字典
            return {}

    @with_retry_and_throttling
    def get_stock_info(self, stock_code: str) -> Dict[str, Any]:
        """
        获取股票基本信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Dict[str, Any]: 股票信息
        """
        try:
            import yfinance as yf

            ticker = yf.Ticker(stock_code)
            info = ticker.info

            # 过滤必要信息
            relevant_info = {
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
            
            debug(f"成功获取股票信息: {filtered_info.get('long_name', stock_code)}")
            
            return filtered_info
        except ImportError:
            warning("未安装yfinance库，使用模拟数据")
            return self._get_mock_stock_info(stock_code)
        except Exception as e:
            error(f"获取股票信息失败: {str(e)}")
            # 重试机制会处理速率限制错误，这里返回空字典
            return {}

    # 添加获取股票新闻的方法，也应用重试和节流
    @with_retry_and_throttling
    def get_stock_news(self, stock_code: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取股票新闻数据
        
        Args:
            stock_code: 股票代码
            limit: 返回新闻数量限制
            
        Returns:
            List[Dict[str, Any]]: 新闻数据列表
        """
        try:
            import yfinance as yf
            from datetime import datetime

            ticker = yf.Ticker(stock_code)
            news_list = ticker.news
            
            # 格式化新闻数据
            formatted_news = []
            for item in news_list[:limit]:
                publish_time = datetime.fromtimestamp(item.get("providerPublishTime", 0)).strftime("%Y-%m-%d %H:%M:%S")
                formatted_news.append({
                    "title": item.get("title", ""),
                    "publisher": item.get("publisher", ""),
                    "link": item.get("link", ""),
                    "publish_time": publish_time
                })
            
            debug(f"获取股票 {stock_code} 新闻数据完成，获取 {len(formatted_news)} 条新闻")
            return formatted_news
        except Exception as e:
            error(f"获取新闻数据失败: {str(e)}")
            # 重试机制会处理速率限制错误，这里返回空列表
            return []

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            df: 价格数据
            
        Returns:
            pd.DataFrame: 添加了技术指标的数据
        """
        try:
            if df.empty:
                return df

            # 深拷贝以避免修改原始数据
            result_df = df.copy()

            # 移动平均线
            result_df['sma_5'] = result_df['close'].rolling(window=5).mean()
            result_df['sma_10'] = result_df['close'].rolling(window=10).mean()
            result_df['sma_20'] = result_df['close'].rolling(window=20).mean()
            result_df['sma_50'] = result_df['close'].rolling(window=50).mean()
            result_df['sma_200'] = result_df['close'].rolling(window=200).mean()

            # RSI (相对强弱指数)
            delta = result_df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            result_df['rsi'] = 100 - (100 / (1 + rs))

            # MACD (移动平均收敛散度)
            exp1 = result_df['close'].ewm(span=12, adjust=False).mean()
            exp2 = result_df['close'].ewm(span=26, adjust=False).mean()
            result_df['macd'] = exp1 - exp2
            result_df['signal'] = result_df['macd'].ewm(span=9, adjust=False).mean()
            result_df['macd_hist'] = result_df['macd'] - result_df['signal']

            # 布林带
            result_df['bb_mid'] = result_df['close'].rolling(window=20).mean()
            result_df['bb_std'] = result_df['close'].rolling(window=20).std()
            result_df['bb_upper'] = result_df['bb_mid'] + (result_df['bb_std'] * 2)
            result_df['bb_lower'] = result_df['bb_mid'] - (result_df['bb_std'] * 2)

            # 交易量相关指标
            result_df['volume_ma_5'] = result_df['volume'].rolling(window=5).mean()
            result_df['volume_ma_20'] = result_df['volume'].rolling(window=20).mean()

            debug("技术指标计算完成")
            return result_df
        except Exception as e:
            error(f"计算技术指标失败: {str(e)}")
            return df

    def calculate_financial_ratios(self, financial_data: Dict[str, Any]) -> Dict[str, float]:
        """
        计算财务比率
        
        Args:
            financial_data: 财务数据
            
        Returns:
            Dict[str, float]: 财务比率
        """
        ratios = {}
        try:
            # 获取最新的资产负债表和利润表数据
            balance_sheet = financial_data.get("balance_sheet", {})
            income_statement = financial_data.get("income_statement", {})

            if not balance_sheet or not income_statement:
                return ratios

            # 获取最新的财务数据（第一个键）
            latest_bs_date = next(iter(balance_sheet)) if balance_sheet else None
            latest_is_date = next(iter(income_statement)) if income_statement else None

            if not latest_bs_date or not latest_is_date:
                return ratios

            latest_bs = balance_sheet[latest_bs_date]
            latest_is = income_statement[latest_is_date]

            # 盈利能力指标
            if 'Net Income' in latest_is and 'Total Assets' in latest_bs:
                ratios['roa'] = latest_is.get('Net Income', 0) / latest_bs.get('Total Assets', 1)

            if 'Net Income' in latest_is and 'Stockholders Equity' in latest_bs:
                ratios['roe'] = latest_is.get('Net Income', 0) / latest_bs.get('Stockholders Equity', 1)

            if 'Total Revenue' in latest_is and 'Gross Profit' in latest_is:
                ratios['gross_margin'] = latest_is.get('Gross Profit', 0) / latest_is.get('Total Revenue', 1)

            if 'Total Revenue' in latest_is and 'Net Income' in latest_is:
                ratios['net_margin'] = latest_is.get('Net Income', 0) / latest_is.get('Total Revenue', 1)

            # 偿债能力指标
            if 'Total Current Assets' in latest_bs and 'Total Current Liabilities' in latest_bs:
                ratios['current_ratio'] = latest_bs.get('Total Current Assets', 0) / latest_bs.get('Total Current Liabilities', 1)

            if 'Total Assets' in latest_bs and 'Total Liabilities' in latest_bs:
                ratios['debt_to_equity'] = latest_bs.get('Total Liabilities', 0) / (latest_bs.get('Total Assets', 0) - latest_bs.get('Total Liabilities', 0) or 1)

            debug("财务比率计算完成")
            return ratios
        except Exception as e:
            error(f"计算财务比率失败: {str(e)}")
            return ratios

    def create_price_chart(self, df: pd.DataFrame, stock_code: str, indicators: List[str] = None) -> str:
        """
        创建价格图表并返回base64编码
        
        Args:
            df: 价格数据
            stock_code: 股票代码
            indicators: 要显示的指标列表
            
        Returns:
            str: base64编码的图表
        """
        try:
            if df.empty:
                return ""

            plt.figure(figsize=(14, 8))

            # 绘制价格
            plt.subplot(2, 1, 1)
            plt.plot(df['date'], df['close'], label='收盘价', linewidth=2)

            # 添加技术指标
            if indicators:
                if 'sma' in indicators:
                    plt.plot(df['date'], df['sma_20'], label='SMA20', alpha=0.7)
                    plt.plot(df['date'], df['sma_50'], label='SMA50', alpha=0.7)
                if 'bb' in indicators and 'bb_upper' in df.columns:
                    plt.plot(df['date'], df['bb_upper'], '--', label='上轨', alpha=0.5)
                    plt.plot(df['date'], df['bb_lower'], '--', label='下轨', alpha=0.5)

            plt.title(f'{stock_code} 价格走势')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 绘制成交量
            plt.subplot(2, 1, 2)
            plt.bar(df['date'], df['volume'], label='成交量', alpha=0.6)
            if 'volume_ma_20' in df.columns:
                plt.plot(df['date'], df['volume_ma_20'], 'r-', label='成交量20日均线', alpha=0.8)
            plt.title('成交量')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()

            # 转换为base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()

            debug(f"创建价格图表完成: {stock_code}")
            return image_base64
        except Exception as e:
            error(f"创建价格图表失败: {str(e)}")
            plt.close()
            return ""

    def create_technical_chart(self, df: pd.DataFrame, stock_code: str) -> str:
        """
        创建技术指标图表
        
        Args:
            df: 价格数据（包含技术指标）
            stock_code: 股票代码
            
        Returns:
            str: base64编码的图表
        """
        try:
            if df.empty:
                return ""

            plt.figure(figsize=(14, 12))

            # RSI
            plt.subplot(3, 1, 1)
            if 'rsi' in df.columns:
                plt.plot(df['date'], df['rsi'], label='RSI')
                plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
                plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            plt.title('相对强弱指数 (RSI)')
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)

            # MACD
            plt.subplot(3, 1, 2)
            if 'macd' in df.columns and 'signal' in df.columns:
                plt.plot(df['date'], df['macd'], label='MACD')
                plt.plot(df['date'], df['signal'], label='信号线')
                if 'macd_hist' in df.columns:
                    plt.bar(df['date'], df['macd_hist'], label='柱状图', alpha=0.5)
            plt.title('MACD')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 布林带
            plt.subplot(3, 1, 3)
            if 'bb_upper' in df.columns:
                plt.plot(df['date'], df['close'], label='收盘价')
                plt.plot(df['date'], df['bb_upper'], '--', label='上轨', alpha=0.7)
                plt.plot(df['date'], df['bb_mid'], '--', label='中轨', alpha=0.7)
                plt.plot(df['date'], df['bb_lower'], '--', label='下轨', alpha=0.7)
            plt.title('布林带')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()

            # 转换为base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()

            debug(f"创建技术指标图表完成: {stock_code}")
            return image_base64
        except Exception as e:
            error(f"创建技术指标图表失败: {str(e)}")
            plt.close()
            return ""

    def format_number(self, num: float, decimals: int = 2) -> str:
        """
        格式化数字显示
        
        Args:
            num: 数字
            decimals: 小数位数
            
        Returns:
            str: 格式化后的字符串
        """
        try:
            if isinstance(num, (int, float)):
                if abs(num) >= 1000000000:  # 十亿
                    return f"{num / 1000000000:.{decimals}f}B"
                elif abs(num) >= 1000000:  # 百万
                    return f"{num / 1000000:.{decimals}f}M"
                elif abs(num) >= 1000:  # 千
                    return f"{num / 1000:.{decimals}f}K"
                else:
                    return f"{num:.{decimals}f}"
            return str(num)
        except Exception as e:
            error(f"格式化数字失败: {str(e)}")
            return str(num)

    def calculate_correlation(self, df1: pd.DataFrame, df2: pd.DataFrame, col1: str = 'close', col2: str = 'close') -> float:
        """
        计算两个数据系列的相关性
        
        Args:
            df1: 第一个数据框
            df2: 第二个数据框
            col1: 第一个数据框的列名
            col2: 第二个数据框的列名
            
        Returns:
            float: 相关系数
        """
        try:
            if df1.empty or df2.empty:
                return 0.0

            # 合并数据并对齐日期
            merged = pd.merge(df1[['date', col1]], df2[['date', col2]], on='date')

            if len(merged) < 2:
                return 0.0

            correlation = merged[col1].corr(merged[col2])
            return correlation
        except Exception as e:
            error(f"计算相关性失败: {str(e)}")
            return 0.0

    def save_json(self, data: Any, file_path: str):
        """
        保存数据为JSON文件
        
        Args:
            data: 要保存的数据
            file_path: 文件路径
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)

            debug(f"数据已保存到: {file_path}")
        except Exception as e:
            error(f"保存JSON文件失败: {str(e)}")

    def load_json(self, file_path: str) -> Dict[str, Any]:
        """
        从JSON文件加载数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            Dict[str, Any]: 加载的数据
        """
        try:
            if not os.path.exists(file_path):
                warning(f"文件不存在: {file_path}")
                return {}

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            debug(f"从文件加载数据完成: {file_path}")
            return data
        except Exception as e:
            error(f"加载JSON文件失败: {str(e)}")
            return {}

    # 以下为模拟数据生成方法，用于开发和测试
    def _get_mock_price_data(self, stock_code: str, period: str = "1y") -> pd.DataFrame:
        """
        生成模拟价格数据
        """
        # 根据周期生成天数
        period_days = {
            "1d": 1,
            "1w": 7,
            "1m": 30,
            "3m": 90,
            "6m": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
            "10y": 3650,
            "max": 3650
        }
        days = period_days.get(period, 365)

        # 生成日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')  # B代表工作日

        # 生成随机价格数据
        np.random.seed(hash(stock_code))  # 确保相同股票代码生成相同的数据
        base_price = np.random.uniform(50, 200)
        returns = np.random.normal(0, 0.01, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))

        # 创建DataFrame
        df = pd.DataFrame({
            'date': dates,
            'open': prices * np.random.uniform(0.98, 1.02, len(dates)),
            'high': prices * np.random.uniform(1.0, 1.03, len(dates)),
            'low': prices * np.random.uniform(0.97, 1.0, len(dates)),
            'close': prices,
            'volume': np.random.randint(100000, 10000000, len(dates))
        })

        # 确保high >= max(open, close) 和 low <= min(open, close)
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        df['low'] = df[['low', 'open', 'close']].min(axis=1)

        return df

    def _get_mock_financial_data(self, stock_code: str) -> Dict[str, Any]:
        """
        生成模拟财务数据
        """
        # 生成日期
        today = datetime.now()
        quarters = [today - timedelta(days=90 * i) for i in range(4)]

        # 模拟收入和利润
        np.random.seed(hash(stock_code))
        base_revenue = np.random.uniform(1000000000, 10000000000)

        # 构建财务数据结构
        income_statement = {}
        balance_sheet = {}

        for i, quarter in enumerate(quarters):
            quarter_str = quarter.strftime('%Y-%m-%d')

            # 收入逐年增长
            revenue = base_revenue * (1 + 0.05 * i)

            # 利润表
            income_statement[quarter_str] = {
                'Total Revenue': revenue,
                'Gross Profit': revenue * 0.4,
                'Operating Income': revenue * 0.25,
                'Net Income': revenue * 0.15,
                'EPS': revenue * 0.15 / 100000000
            }

            # 资产负债表
            balance_sheet[quarter_str] = {
                'Total Assets': revenue * 2,
                'Total Current Assets': revenue * 0.8,
                'Total Liabilities': revenue * 1.2,
                'Total Current Liabilities': revenue * 0.6,
                'Stockholders Equity': revenue * 0.8
            }

        return {
            'income_statement': income_statement,
            'balance_sheet': balance_sheet,
            'cash_flow': {},
            'quarterly_income_statement': income_statement,
            'quarterly_balance_sheet': balance_sheet,
            'quarterly_cash_flow': {}
        }

    def _get_mock_stock_info(self, stock_code: str) -> Dict[str, Any]:
        """
        生成模拟股票基本信息
        """
        np.random.seed(hash(stock_code))

        sectors = ['Technology', 'Financial Services', 'Healthcare', 'Consumer Cyclical', 'Energy', 'Consumer Defensive']
        industries = {
            'Technology': ['Software', 'Semiconductors', 'Hardware', 'Internet'],
            'Financial Services': ['Banks', 'Insurance', 'Investment Banking'],
            'Healthcare': ['Pharmaceuticals', 'Medical Devices', 'Healthcare Services'],
            'Consumer Cyclical': ['Retail', 'Automotive', 'Restaurants'],
            'Energy': ['Oil & Gas', 'Renewable Energy', 'Utilities'],
            'Consumer Defensive': ['Food & Beverages', 'Household Products', 'Personal Care']
        }

        sector = np.random.choice(sectors)
        industry = np.random.choice(industries[sector])

        return {
            'symbol': stock_code,
            'short_name': f"{sector[:4]} Corp",
            'long_name': f"{sector} Technology Corporation",
            'sector': sector,
            'industry': industry,
            'country': np.random.choice(['United States', 'China', 'Japan', 'Germany', 'United Kingdom']),
            'market_cap': np.random.uniform(1000000000, 1000000000000),
            'beta': np.random.uniform(0.8, 1.5),
            'pe_ratio': np.random.uniform(15, 40),
            'eps': np.random.uniform(1, 10),
            'dividend_yield': np.random.uniform(0, 5),
            '52_week_high': np.random.uniform(100, 500),
            '52_week_low': np.random.uniform(50, 200),
            'average_volume': np.random.uniform(1000000, 50000000),
            'currency': 'USD'
        }
