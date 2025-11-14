"""
@FileName: stock_data_manager.py
@Description: 模拟股票数据管理器。
@Author: HengLine
@Time: 2025/11/13 14:31
"""
import random
from datetime import datetime, timedelta

import pandas as pd

from utils.date_utils import format_date


class StockDataManager:
    """模拟股票数据管理器，提供基础的股票数据"""

    def get_stock_price_data(self, ticker, period='1mo', interval='1d'):
        """获取模拟的股票价格数据"""
        days = {'1d': 1, '1wk': 7, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '5y': 1825, '10y': 3650, 'max': 10950}.get(period, 30)

        dates = [datetime.now() - timedelta(days=i) for i in range(days)][::-1]
        base_price = random.uniform(100, 200)

        data = []
        current_price = base_price

        for date in dates:
            change = random.uniform(-2, 2)
            current_price = max(1, current_price + change)

            open_price = current_price * random.uniform(0.98, 1.02)
            high = max(open_price, current_price) * random.uniform(1.0, 1.01)
            low = min(open_price, current_price) * random.uniform(0.99, 1.0)
            volume = int(random.uniform(1000000, 5000000))

            data.append({
                'Date': date,
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': current_price,
                'Volume': volume
            })

        return pd.DataFrame(data)

    def get_stock_info(self, ticker):
        """获取模拟的股票基本信息"""
        # 处理股票代码格式
        ticker_code = ticker.split('.')[0] if '.' in ticker else ticker

        company_names = {
            'AAPL': '苹果公司',
            'MSFT': '微软公司',
            'GOOGL': '谷歌公司',
            '300623': '光库科技',
            '000001': '平安银行',
            '600519': '贵州茅台'
        }

        name = company_names.get(ticker_code.upper(), f'{ticker_code.upper()}公司')

        sectors = ['科技', '金融', '医疗健康', '可选消费', '能源']
        industries = ['软件', '硬件', '金融服务', '医疗设备', '零售']

        return {
            'symbol': ticker.upper(),
            'company_name': name,
            'sector': random.choice(sectors),
            'industry': random.choice(industries),
            'market_cap': f"{random.uniform(10, 3000):.2f}亿元",
            'pe_ratio': round(random.uniform(10, 40), 2),
            'eps': round(random.uniform(1, 15), 2),
            'dividend_yield': f"{random.uniform(0, 5):.2f}%",
            'description': f"{name}是{random.choice(sectors)}行业的领先企业。"
        }

    def get_stock_news(self, ticker):
        """获取模拟的股票新闻"""
        news = []
        titles = [
            f"{ticker.upper()}发布新产品，行业专家看好前景",
            f"{ticker.upper()}季度业绩超预期，股价应声上涨",
            f"{ticker.upper()}宣布重大战略合作，拓展市场布局",
            f"分析师上调{ticker.upper()}评级，目标价大幅提升",
            f"{ticker.upper()}公布最新技术突破，引领行业创新"
        ]
        sources = ['财经网', '证券时报', '中国证券报', '第一财经', '经济日报']

        for i in range(5):
            news.append({
                'title': titles[i],
                'published_date': format_date(datetime.now() - timedelta(days=i)),
                'source': sources[i % len(sources)],
                'summary': f"{ticker.upper()}今日宣布重要消息，分析师认为这将在未来几个季度推动公司业绩显著增长。"
            })
        return news

    def get_financial_data(self, ticker):
        """获取模拟的财务数据"""
        years = [str(datetime.now().year - i) for i in range(5)]

        # 返回符合预期的纯DataFrame字典结构
        income_statement = pd.DataFrame({
            'Year': years,
            'totalRevenue': [round(random.uniform(10, 100), 2) for _ in years],
            'netIncome': [round(random.uniform(2, 20), 2) for _ in years],
            'grossProfit': [round(random.uniform(5, 30), 2) for _ in years]
        })

        balance_sheet = pd.DataFrame({
            'Year': years,
            'totalAssets': [round(random.uniform(50, 200), 2) for _ in years],
            'totalLiabilities': [round(random.uniform(20, 100), 2) for _ in years],
            'totalEquity': [round(random.uniform(30, 100), 2) for _ in years]
        })

        cash_flow = pd.DataFrame({
            'Year': years,
            'operatingCashFlow': [round(random.uniform(5, 25), 2) for _ in years],
            'investingCashFlow': [round(random.uniform(-15, -5), 2) for _ in years],
            'financingCashFlow': [round(random.uniform(-10, 10), 2) for _ in years],
            'freeCashFlow': [round(random.uniform(0, 20), 2) for _ in years]
        })

        return {
            'income_statement': income_statement,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow
        }

    def get_stock_realtime_data(self, ticker):
        """获取模拟的股票实时数据"""
        # 处理股票代码格式
        ticker_code = ticker.split('.')[0] if '.' in ticker else ticker
        
        # 生成模拟的实时价格数据
        base_price = random.uniform(50, 200)
        price_change = random.uniform(-5, 5)
        current_price = base_price + price_change
        
        return {
            'symbol': ticker.upper(),
            'current_price': round(current_price, 2),
            'previous_close': round(base_price, 2),
            'price_change': round(price_change, 2),
            'price_change_percent': round((price_change / base_price) * 100, 2),
            'volume': int(random.uniform(1000000, 10000000)),
            'market_cap': f"{random.uniform(100, 5000):.2f}亿",
            'pe_ratio': round(random.uniform(10, 40), 2),
            'high': round(current_price * random.uniform(1.01, 1.05), 2),
            'low': round(current_price * random.uniform(0.95, 0.99), 2),
            'open': round(current_price * random.uniform(0.98, 1.02), 2),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


stock_data_manager = StockDataManager()
