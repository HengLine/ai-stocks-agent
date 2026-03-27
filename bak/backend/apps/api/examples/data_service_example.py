#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据服务层使用示例

展示如何使用市场数据、财报、新闻舆情等数据服务组件
"""

import os
import time
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入数据服务组件
try:
    from apps.api.data.market_data_source import global_market_data_service
    from apps.api.data.financial_report_source import global_financial_report_service
    from apps.api.data.news_sentiment_source import global_news_sentiment_service
    from apps.api.data.data_cache import global_data_cache_service
    from apps.api.data.data_updater import global_data_update_service
    from apps.api.data.base import normalize_ticker
    logger.info("Successfully imported data service components")
except ImportError as e:
    logger.error(f"Failed to import data service components: {str(e)}")
    # 添加项目根目录到Python路径
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
    try:
        from backend.apps.api.data.market_data_source import global_market_data_service
        from backend.apps.api.data.financial_report_source import global_financial_report_service
        from backend.apps.api.data.news_sentiment_source import global_news_sentiment_service
        from backend.apps.api.data.data_cache import global_data_cache_service
        from backend.apps.api.data.data_updater import global_data_update_service
        from backend.apps.api.data.base import normalize_ticker
        logger.info("Successfully imported data service components after adjusting path")
    except ImportError as e2:
        logger.error(f"Failed to import data service components after adjusting path: {str(e2)}")
        raise


def example_market_data_service():
    """市场数据服务使用示例"""
    logger.info("===== 市场数据服务使用示例 =====")
    
    # 股票代码示例
    tickers = ['AAPL', 'TSLA', 'MSFT']
    
    for ticker in tickers:
        logger.info(f"\n获取股票 {ticker} 的数据：")
        
        # 1. 获取K线数据
        logger.info("1. 获取K线数据（最近3个月，日线）：")
        kline_query = {
            'type': 'kline',
            'ticker': ticker,
            'window': '3M',
            'interval': '1d'
        }
        
        # 先检查缓存
        cached_kline = global_data_cache_service.get('market_data', f'{ticker}_kline', kline_query)
        if cached_kline:
            logger.info(f"从缓存获取到K线数据，共 {len(cached_kline)} 条")
        else:
            # 从数据源获取
            kline_data = global_market_data_service.fetch_data(kline_query)
            logger.info(f"从数据源获取K线数据，共 {len(kline_data)} 条")
            
            # 保存到缓存
            if kline_data:
                global_data_cache_service.set('market_data', f'{ticker}_kline', kline_data, kline_query, ttl=60)
                logger.info("K线数据已保存到缓存")
            
        # 2. 获取实时报价
        logger.info("2. 获取实时报价：")
        quote_query = {
            'type': 'quote',
            'ticker': ticker
        }
        
        # 先检查缓存
        cached_quote = global_data_cache_service.get('quote_data', ticker, quote_query)
        if cached_quote:
            logger.info(f"从缓存获取到报价数据：价格={cached_quote.get('price')}, 成交量={cached_quote.get('volume')}")
        else:
            # 从数据源获取
            quote_data = global_market_data_service.fetch_data(quote_query)
            if quote_data and len(quote_data) > 0:
                quote = quote_data[0]
                logger.info(f"从数据源获取报价数据：价格={quote.get('price')}, 成交量={quote.get('volume')}")
                
                # 保存到缓存
                global_data_cache_service.set('quote_data', ticker, quote, quote_query, ttl=30)
                logger.info("报价数据已保存到缓存")
            
        # 3. 获取公司信息
        logger.info("3. 获取公司信息：")
        info_query = {
            'type': 'info',
            'ticker': ticker
        }
        
        # 先检查缓存
        cached_info = global_data_cache_service.get('company_info', ticker, info_query)
        if cached_info:
            logger.info(f"从缓存获取到公司信息：名称={cached_info.get('name')}, 行业={cached_info.get('industry')}")
        else:
            # 从数据源获取
            info_data = global_market_data_service.fetch_data(info_query)
            if info_data and len(info_data) > 0:
                info = info_data[0]
                logger.info(f"从数据源获取公司信息：名称={info.get('name')}, 行业={info.get('industry')}")
                
                # 保存到缓存
                global_data_cache_service.set('company_info', ticker, info, info_query, ttl=86400)  # 缓存1天
                logger.info("公司信息已保存到缓存")
    
    # 4. 切换数据源示例
    logger.info("\n4. 切换数据源示例：")
    available_sources = global_market_data_service.get_available_sources()
    logger.info(f"可用的数据源：{available_sources}")
    
    # 如果有其他数据源可用，可以切换
    if 'tushare' in available_sources:
        logger.info("尝试切换到Tushare数据源...")
        result = global_market_data_service.set_default_source('tushare')
        logger.info(f"切换结果：{result}")
        
        # 尝试获取数据
        test_data = global_market_data_service.fetch_data({'type': 'quote', 'ticker': 'AAPL'})
        logger.info(f"从Tushare获取数据：{test_data}")
        
        # 切回默认数据源
        global_market_data_service.set_default_source('yfinance')


def example_financial_report_service():
    """财报与研报服务使用示例"""
    logger.info("\n===== 财报与研报服务使用示例 =====")
    
    # 股票代码示例
    tickers = ['AAPL', 'MSFT']
    
    for ticker in tickers:
        logger.info(f"\n获取股票 {ticker} 的财报与研报数据：")
        
        # 1. 获取财报数据
        logger.info("1. 获取年度财报数据：")
        financial_query = {
            'type': 'financial_report',
            'ticker': ticker,
            'report_type': 'annual'
        }
        
        # 先检查缓存
        cached_financial = global_data_cache_service.get('financial_report', ticker, financial_query)
        if cached_financial:
            logger.info(f"从缓存获取到财报数据")
        else:
            # 从多个数据源尝试获取财报数据
            sources = ['wind', 'ifind', 'scraper']
            financial_data = None
            
            for source in sources:
                financial_data = global_financial_report_service.fetch_data(financial_query, source_name=source)
                if financial_data:
                    logger.info(f"从 {source} 获取到财报数据，共 {len(financial_data)} 条")
                    break
            
            # 如果获取到数据，保存到缓存
            if financial_data:
                global_data_cache_service.set('financial_report', ticker, financial_data, financial_query, ttl=86400)  # 缓存1天
                logger.info("财报数据已保存到缓存")
        
        # 2. 获取研报数据
        logger.info("2. 获取研报数据：")
        research_query = {
            'type': 'research_report',
            'ticker': ticker
        }
        
        # 先检查缓存
        cached_research = global_data_cache_service.get('research_report', ticker, research_query)
        if cached_research:
            logger.info(f"从缓存获取到研报数据")
        else:
            # 从多个数据源尝试获取研报数据
            sources = ['wind', 'ifind', 'scraper']
            research_data = None
            
            for source in sources:
                research_data = global_financial_report_service.fetch_data(research_query, source_name=source)
                if research_data:
                    logger.info(f"从 {source} 获取到研报数据，共 {len(research_data)} 条")
                    break
            
            # 如果获取到数据，保存到缓存
            if research_data:
                global_data_cache_service.set('research_report', ticker, research_data, research_query, ttl=3600)  # 缓存1小时
                logger.info("研报数据已保存到缓存")


def example_news_sentiment_service():
    """新闻舆情服务使用示例"""
    logger.info("\n===== 新闻舆情服务使用示例 =====")
    
    # 股票代码示例
    tickers = ['AAPL', 'TSLA']
    
    for ticker in tickers:
        logger.info(f"\n获取股票 {ticker} 的新闻舆情数据：")
        
        # 1. 获取新闻数据（不包含情感分析）
        logger.info("1. 获取新闻数据：")
        news_query = {
            'ticker': ticker,
            'limit': 10
        }
        
        # 先检查缓存
        cached_news = global_data_cache_service.get('news_data', ticker, news_query)
        if cached_news:
            logger.info(f"从缓存获取到新闻数据，共 {len(cached_news)} 条")
        else:
            # 从数据源获取新闻
            news_data = global_news_sentiment_service.fetch_news(news_query)
            logger.info(f"从数据源获取新闻数据，共 {len(news_data)} 条")
            
            # 如果获取到数据，保存到缓存
            if news_data:
                global_data_cache_service.set('news_data', ticker, news_data, news_query, ttl=3600)  # 缓存1小时
                logger.info("新闻数据已保存到缓存")
        
        # 2. 获取带情感分析的新闻数据
        logger.info("2. 获取带情感分析的新闻数据：")
        sentiment_query = {
            'ticker': ticker,
            'limit': 10
        }
        
        # 先检查缓存
        cached_sentiment = global_data_cache_service.get('sentiment_data', ticker)
        if cached_sentiment:
            logger.info(f"从缓存获取到带情感分析的新闻数据")
            if 'sentiment_summary' in cached_sentiment:
                summary = cached_sentiment['sentiment_summary']
                logger.info(f"综合情感得分：正面={summary.get('positive')}, 负面={summary.get('negative')}, 总分={summary.get('score')}")
        else:
            # 从数据源获取带情感分析的新闻
            news_with_sentiment = global_news_sentiment_service.fetch_news_with_sentiment(sentiment_query)
            logger.info(f"从数据源获取带情感分析的新闻数据，共 {len(news_with_sentiment)} 条")
            
            # 计算综合情感得分
            if news_with_sentiment:
                positive_count = sum(1 for news in news_with_sentiment if news.get('sentiment', {}).get('score', 0) > 0)
                negative_count = sum(1 for news in news_with_sentiment if news.get('sentiment', {}).get('score', 0) < 0)
                neutral_count = len(news_with_sentiment) - positive_count - negative_count
                
                logger.info(f"情感分布：正面={positive_count}, 负面={negative_count}, 中性={neutral_count}")
                
                # 保存到缓存
                cache_data = {
                    'news': news_with_sentiment,
                    'update_time': datetime.now().isoformat()
                }
                global_data_cache_service.set('sentiment_data', ticker, cache_data, ttl=1800)  # 缓存30分钟
                logger.info("带情感分析的新闻数据已保存到缓存")
        
        # 3. 选择特定的新闻数据源
        logger.info("3. 选择特定的新闻数据源：")
        available_sources = global_news_sentiment_service.get_available_sources()
        logger.info(f"可用的新闻数据源：{available_sources}")
        
        # 尝试使用特定数据源获取新闻
        if 'sina' in available_sources:
            logger.info("尝试仅从新浪财经获取新闻...")
            sina_news = global_news_sentiment_service.fetch_news(news_query, sources=['sina'])
            logger.info(f"从新浪财经获取到 {len(sina_news)} 条新闻")


def example_data_updater_service():
    """数据更新服务使用示例"""
    logger.info("\n===== 数据更新服务使用示例 =====")
    
    # 1. 检查更新器状态
    logger.info("1. 检查更新器状态：")
    for updater_name in ['realtime', 'fundamental', 'news_sentiment']:
        is_running = global_data_update_service.is_running(updater_name)
        logger.info(f"{updater_name} updater is running: {is_running}")
    
    # 2. 启动实时数据更新器
    logger.info("\n2. 启动实时数据更新器：")
    if not global_data_update_service.is_running('realtime'):
        result = global_data_update_service.start('realtime')
        logger.info(f"启动实时数据更新器：{result}")
    
    # 3. 订阅实时数据
    logger.info("3. 订阅实时数据：")
    tickers = ['AAPL', 'TSLA']
    for ticker in tickers:
        result = global_data_update_service.subscribe_realtime(ticker)
        logger.info(f"订阅 {ticker} 实时数据：{result}")
    
    # 4. 注册实时数据更新回调
    logger.info("4. 注册实时数据更新回调：")
    
    def realtime_data_callback(data):
        """实时数据更新回调函数"""
        ticker = data.get('ticker')
        price = data.get('price')
        change = data.get('change_percent')
        logger.info(f"实时数据更新 - {ticker}: ${price} ({change}%)")
    
    result = global_data_update_service.register_realtime_callback(realtime_data_callback)
    logger.info(f"注册回调函数：{result}")
    
    # 等待一段时间，观察实时数据更新
    logger.info("等待10秒，观察实时数据更新...")
    time.sleep(10)
    
    # 5. 设置基本面数据更新目标
    logger.info("\n5. 设置基本面数据更新目标：")
    fundamental_targets = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'GOOGL']
    result = global_data_update_service.set_fundamental_targets(fundamental_targets)
    logger.info(f"设置基本面数据更新目标：{result}")
    logger.info(f"目标股票：{fundamental_targets}")
    
    # 6. 设置新闻舆情数据更新目标
    logger.info("6. 设置新闻舆情数据更新目标：")
    news_targets = ['AAPL', 'TSLA', 'NVDA', 'BABA']
    result = global_data_update_service.set_news_sentiment_targets(news_targets)
    logger.info(f"设置新闻舆情数据更新目标：{result}")
    logger.info(f"目标股票：{news_targets}")
    
    # 7. 手动触发一次新闻舆情更新（仅作演示）
    logger.info("\n7. 手动触发一次新闻舆情更新（仅作演示）：")
    news_updater = global_data_update_service.get_updater('news_sentiment')
    if news_updater and hasattr(news_updater, '_update_news_sentiment'):
        logger.info("正在触发新闻舆情更新...")
        try:
            news_updater._update_news_sentiment()
            logger.info("新闻舆情更新完成")
        except Exception as e:
            logger.error(f"触发新闻舆情更新失败：{str(e)}")
    
    # 8. 停止实时数据更新器（如果不再需要）
    # logger.info("\n8. 停止实时数据更新器：")
    # result = global_data_update_service.stop('realtime')
    # logger.info(f"停止实时数据更新器：{result}")


def example_data_cache_service():
    """数据缓存服务使用示例"""
    logger.info("\n===== 数据缓存服务使用示例 =====")
    
    # 1. 设置和获取缓存
    logger.info("1. 设置和获取缓存：")
    test_key = 'test_key'
    test_value = {'name': 'test_data', 'value': 123, 'timestamp': datetime.now().isoformat()}
    
    # 设置缓存，有效期60秒
    result = global_data_cache_service.set('test_type', test_key, test_value, ttl=60)
    logger.info(f"设置缓存：{result}")
    
    # 获取缓存
    cached_value = global_data_cache_service.get('test_type', test_key)
    logger.info(f"获取缓存：{cached_value}")
    
    # 2. 检查缓存是否存在
    logger.info("2. 检查缓存是否存在：")
    exists = global_data_cache_service.exists('test_type', test_key)
    logger.info(f"缓存是否存在：{exists}")
    
    # 3. 删除缓存
    logger.info("3. 删除缓存：")
    delete_result = global_data_cache_service.delete('test_type', test_key)
    logger.info(f"删除缓存：{delete_result}")
    
    # 再次检查缓存是否存在
    exists_after_delete = global_data_cache_service.exists('test_type', test_key)
    logger.info(f"删除后缓存是否存在：{exists_after_delete}")
    
    # 4. 批量清除缓存
    logger.info("4. 批量清除缓存：")
    # 先设置一些测试缓存
    for i in range(3):
        global_data_cache_service.set('test_type', f'test_key_{i}', {'value': i}, ttl=60)
    
    # 清除特定类型的缓存
    clear_result = global_data_cache_service.clear('test_type')
    logger.info(f"清除test_type类型的缓存：{clear_result}")
    
    # 5. 设置不同数据类型的TTL
    logger.info("5. 设置不同数据类型的TTL：")
    global_data_cache_service.set_ttl('custom_type', 120)  # 设置为120秒
    logger.info("已将custom_type类型的缓存TTL设置为120秒")


def main():
    """主函数，运行所有示例"""
    try:
        # 运行市场数据服务示例
        example_market_data_service()
        
        # 运行财报与研报服务示例
        example_financial_report_service()
        
        # 运行新闻舆情服务示例
        example_news_sentiment_service()
        
        # 运行数据更新服务示例
        example_data_updater_service()
        
        # 运行数据缓存服务示例
        example_data_cache_service()
        
        logger.info("\n所有数据服务示例运行完成！")
        
    except Exception as e:
        logger.error(f"运行示例时出错：{str(e)}")
    finally:
        # 可选：关闭数据更新服务
        # global_data_update_service.shutdown()
        # logger.info("数据更新服务已关闭")
        pass


if __name__ == "__main__":
    main()