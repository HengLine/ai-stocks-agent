#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票数据分析平台
提供股票数据查询、图表展示和分析功能
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入统一的日期处理工具
from utils.date_utils import format_date, format_date_for_filename

# 导入真实的股票数据管理器
from hengline.stock.stock_manage import get_stock_price_data, get_stock_info, get_stock_news, get_financial_data

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def setup_page_style():
    """设置页面样式"""
    st.set_page_config(
        page_title="股票数据分析平台",
        page_icon="📈",
        layout="wide"
    )

    st.markdown("""
    <style>
    .main-header {
        color: #1a5276;
        font-weight: bold;
    }
    .stock-info-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .metric-title {
        color: #3498db;
        font-size: 0.9em;
    }
    .metric-value {
        font-size: 1.2em;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


class StockDataViews:
    """简化版股票数据视图类"""

    @staticmethod
    def show_overview_view(ticker, stock_info, price_data, news_data):
        """显示股票概览视图，包含基本信息和最新新闻"""
        st.markdown(f"### {stock_info.get('name', stock_info.get('company_name', stock_info.get('full_name', stock_info.get('symbol'))))} ({stock_info['symbol']})")
        st.write(stock_info.get('description', '暂无公司简介信息'))

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("市值", stock_info.get('market_cap', 'N/A'))
        with col2:
            st.metric("市盈率(PE)", stock_info.get('pe_ratio', 'N/A'))
        with col3:
            st.metric("每股收益(EPS)", stock_info.get('eps', 'N/A'))
        with col4:
            st.metric("股息收益率", stock_info.get('dividend_yield', 'N/A'))

        st.markdown("#### 基本信息")
        st.write(f"**行业:** {stock_info.get('sector', 'N/A')} | **细分行业:** {stock_info.get('industry', 'N/A')}")

        # 添加最新新闻部分
        st.markdown("#### 最新新闻")

        for item in news_data:
            # 确保日期格式标准化
            published_date = format_date(item['published_date'])
            with st.expander(f"[{published_date}] {item['title']} - {item['source']}"):
                st.write(item['summary'])

    @staticmethod
    def show_price_chart_view(ticker, price_data):
        """显示价格图表视图"""
        # 格式化日期为标准格式
        if 'Date' in price_data.columns:
            price_data['Date'] = price_data['Date'].apply(format_date)

        fig = go.Figure(data=[go.Candlestick(x=price_data['Date'],
                                             open=price_data['Open'],
                                             high=price_data['High'],
                                             low=price_data['Low'],
                                             close=price_data['Close'])])

        fig.update_layout(title=f'{ticker} K线图',
                          xaxis_title='日期',
                          yaxis_title='价格 (¥)',
                          height=600)

        st.plotly_chart(fig, use_container_width=True)

        # 显示成交量图表
        fig_volume = px.bar(price_data, x='Date', y='Volume', title='成交量')
        fig_volume.update_layout(height=300)
        st.plotly_chart(fig_volume, use_container_width=True)

    @staticmethod
    def show_financial_analysis_view(ticker, financial_data):
        """显示财务分析视图"""
        st.markdown("### 财务概览")

        try:
            # 收入和利润图表
            if 'income_statement' in financial_data and isinstance(financial_data['income_statement'], pd.DataFrame) and not financial_data['income_statement'].empty:
                income_df = financial_data['income_statement']
                fig = go.Figure()
                if 'totalRevenue' in income_df.columns:
                    fig.add_trace(go.Bar(x=income_df['Year'], y=income_df['totalRevenue'], name='营收 (十亿元)'))
                if 'netIncome' in income_df.columns:
                    fig.add_trace(go.Bar(x=income_df['Year'], y=income_df['netIncome'], name='净利润 (十亿元)'))
                fig.update_layout(title='年度营收与净利润', barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("未找到有效的收入报表数据")

            # 资产负债表
            if 'balance_sheet' in financial_data and isinstance(financial_data['balance_sheet'], pd.DataFrame) and not financial_data['balance_sheet'].empty:
                balance_df = financial_data['balance_sheet']
                fig = go.Figure()
                if 'totalAssets' in balance_df.columns:
                    fig.add_trace(go.Bar(x=balance_df['Year'], y=balance_df['totalAssets'], name='资产总额 (十亿元)'))
                if 'totalLiabilities' in balance_df.columns:
                    fig.add_trace(go.Bar(x=balance_df['Year'], y=balance_df['totalLiabilities'], name='负债总额 (十亿元)'))
                fig.update_layout(title='资产负债表', barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("未找到有效的资产负债表数据")

            # 现金流
            if 'cash_flow' in financial_data and isinstance(financial_data['cash_flow'], pd.DataFrame) and not financial_data['cash_flow'].empty:
                cash_flow_df = financial_data['cash_flow']
                fig = px.line(cash_flow_df, x='Year', y='operatingCashFlow', title='经营现金流 (十亿元)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("未找到有效的现金流数据")
        except Exception as e:
            st.error(f"显示财务数据时出错: {str(e)}")

    @staticmethod
    def show_news_view(ticker, news_data):
        """显示新闻视图"""
        st.markdown(f"### {ticker} 最新新闻")

        for item in news_data:
            # 确保日期格式标准化
            published_date = format_date(item['published_date'])
            with st.expander(f"[{published_date}] {item['title']} - {item['source']}"):
                st.write(item['summary'])

    @staticmethod
    def show_advanced_analysis_view(ticker, price_data):
        """显示高级分析视图"""
        st.markdown("### 高级技术分析")

        # 格式化日期为标准格式
        if 'Date' in price_data.columns:
            price_data['Date'] = price_data['Date'].apply(format_date)

        # 简单的移动平均线计算
        price_data['MA5'] = price_data['Close'].rolling(window=5).mean()
        price_data['MA20'] = price_data['Close'].rolling(window=20).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price_data['Date'], y=price_data['Close'], name='收盘价'))
        fig.add_trace(go.Scatter(x=price_data['Date'], y=price_data['MA5'], name='5日均线'))
        fig.add_trace(go.Scatter(x=price_data['Date'], y=price_data['MA20'], name='20日均线'))
        fig.update_layout(title=f'{ticker} 价格与均线',
                          xaxis_title='日期',
                          yaxis_title='价格 (¥)',
                          height=600)
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def show_comparison_view(tickers, price_data_dict):
        """显示股票对比视图"""
        st.markdown("### 股票对比")

        # 价格走势对比
        st.subheader("价格走势对比")
        fig = go.Figure()
        for ticker, data in price_data_dict.items():
            # 格式化日期为标准格式
            if 'Date' in data.columns:
                data['Date'] = data['Date'].apply(format_date)
            # 归一化价格以便于比较
            norm_close = data['Close'] / data['Close'].iloc[0] * 100
            fig.add_trace(go.Scatter(x=data['Date'], y=norm_close, name=ticker))

        fig.update_layout(title='股票价格走势对比 (归一化)',
                          xaxis_title='日期',
                          yaxis_title='归一化价格 (¥) (基准=100)',
                          height=600)
        st.plotly_chart(fig, use_container_width=True)


# 设置页面样式和配置
setup_page_style()

# 侧边栏 - 股票选择和参数设置
with st.sidebar:
    st.markdown("## 股票数据查询")

    # 股票代码输入
    ticker = st.text_input("请输入股票代码", value="300000").upper()

    # 时间周期选择
    period_options = {
        "1天": "1d",
        "1周": "1wk",
        "1个月": "1mo",
        "3个月": "3mo",
        "6个月": "6mo",
        "1年": "1y",
        "5年": "5y",
        "10年": "10y",
        "全部": "max"
    }
    period_display = st.selectbox("选择时间周期", list(period_options.keys()))
    period = period_options[period_display]

    # 股票对比功能
    st.markdown("---")
    st.markdown("## 股票对比")
    comparison_enabled = st.checkbox("启用股票对比", value=False)

    compare_tickers = []
    if comparison_enabled:
        for i in range(2):  # 最多对比3只股票（包括主股票）
            compare_ticker = st.text_input(f"对比股票 {i + 1}", value="MSFT" if i == 0 else "GOOGL").upper()
            if compare_ticker:
                compare_tickers.append(compare_ticker)

    # 视图选择
    st.markdown("---")
    st.markdown("## 视图设置")
    view_mode = st.selectbox(
        "选择视图模式",
        ["股票概览", "价格图表", "财务分析", "高级分析"]
    )

# 主内容区域
st.markdown("# :blue[股票数据分析平台] :sunglasses:")
# st.markdown("## 实时股票数据可视化与分析")

# 显示加载状态
with st.spinner("正在获取数据..."):
    try:
        # 获取主股票的数据
        price_data = get_stock_price_data(ticker, period=period)
        stock_info = get_stock_info(ticker)
        # price_data = stock_data_manager.get_stock_price_data(ticker, period=period)
        # stock_info = stock_data_manager.get_stock_info(ticker)

        # 根据选择的视图显示不同内容
        if view_mode == "股票概览":
            # 获取新闻数据并传递给概览视图
            news_data = get_stock_news(ticker)
            # news_data = stock_data_manager.get_stock_news(ticker)
            StockDataViews.show_overview_view(ticker, stock_info, price_data, news_data)

        elif view_mode == "价格图表":
            StockDataViews.show_price_chart_view(ticker, price_data)

        elif view_mode == "财务分析":
            financial_data = get_financial_data(ticker)
            # financial_data = stock_data_manager.get_financial_data(ticker)
            StockDataViews.show_financial_analysis_view(ticker, financial_data)

        elif view_mode == "新闻":
            news_data = get_stock_news(ticker)
            # news_data = stock_data_manager.get_stock_news(ticker)
            StockDataViews.show_overview_view(ticker, stock_info, price_data, news_data)
            st.info("最新新闻已整合到股票概览页面中")

        elif view_mode == "高级分析":
            StockDataViews.show_advanced_analysis_view(ticker, price_data)

        # 股票对比功能
        if comparison_enabled and compare_tickers:
            st.markdown("---")

            # 获取所有对比股票的数据
            all_tickers = [ticker] + compare_tickers
            all_price_data = {}

            for t in all_tickers:
                try:
                    # all_price_data[t] = stock_data_manager.get_stock_price_data(t, period=period)
                    all_price_data[t] = get_stock_price_data(t, period=period)
                except Exception as e:
                    st.warning(f"无法获取 {t} 的数据: {str(e)}")

            if all_price_data:
                StockDataViews.show_comparison_view(list(all_price_data.keys()), all_price_data)

        # 数据导出功能
        st.markdown("---")
        st.markdown("## 数据导出")

        col1, col2 = st.columns(2)
        with col1:
            if st.download_button(
                    label="导出价格数据 (CSV)",
                    data=price_data.to_csv(index=False),
                    file_name=f"{ticker}_价格数据_{format_date_for_filename()}.csv",
                    mime="text/csv"
            ):
                st.success("价格数据导出成功")

        with col2:
            try:
                financial_data = get_financial_data(ticker)
                # financial_data = stock_manager.get_financial_data(ticker)

                # 创建一个综合的财务数据DataFrame用于导出
                years = []
                revenue_data = []
                profit_data = []
                assets_data = []
                liabilities_data = []
                cash_flow_data = []

                # 从各个DataFrame中提取数据
                if 'income_statement' in financial_data and isinstance(financial_data['income_statement'], pd.DataFrame):
                    income_df = financial_data['income_statement']
                    years = income_df['Year'].tolist()
                    revenue_data = income_df.get('totalRevenue', [None] * len(years)).tolist()
                    profit_data = income_df.get('netIncome', [None] * len(years)).tolist()

                if 'balance_sheet' in financial_data and isinstance(financial_data['balance_sheet'], pd.DataFrame):
                    balance_df = financial_data['balance_sheet']
                    assets_data = balance_df.get('totalAssets', [None] * len(years)).tolist()
                    liabilities_data = balance_df.get('totalLiabilities', [None] * len(years)).tolist()

                if 'cash_flow' in financial_data and isinstance(financial_data['cash_flow'], pd.DataFrame):
                    cash_flow_df = financial_data['cash_flow']
                    cash_flow_data = cash_flow_df.get('operatingCashFlow', [None] * len(years)).tolist()

                # 创建导出DataFrame
                financial_df = pd.DataFrame({
                    'Year': years,
                    'Revenue': revenue_data,
                    'Profit': profit_data,
                    'Assets': assets_data,
                    'Liabilities': liabilities_data,
                    'Cash Flow': cash_flow_data
                })

                if st.download_button(
                        label="导出财务数据 (CSV)",
                        data=financial_df.to_csv(index=False),
                        file_name=f"{ticker}_财务数据_{format_date_for_filename()}.csv",
                        mime="text/csv"
                ):
                    st.success("财务数据导出成功")
            except Exception as e:
                st.warning(f"无法导出财务数据: {str(e)}")

    except Exception as e:
        st.error(f"获取数据时发生错误: {str(e)}")
        st.info("请检查股票代码是否正确，或尝试其他股票代码。")

        # 显示一些示例股票代码建议
        st.markdown("### 建议的股票代码")
        st.write("- 600000-699999: 上海证券交易所")
        st.write("- 000000-009999: 深圳证券交易所主板")
        st.write("- 300000-309999: 深圳证券交易所创业板")
        st.write("- 688000-688999: 上海证券交易所科创板")
        # st.write("美股: AAPL (苹果), MSFT (微软), GOOGL (谷歌), AMZN (亚马逊), TSLA (特斯拉)")
        st.write("港股: 0700.HK (腾讯控股)")
        st.write("A股: 600519.SS (贵州茅台), 000001.SZ (平安银行)")

# 页脚信息
st.markdown("---")
st.markdown("#### :red[股票数据分析平台 | 数据仅供参考，不构成投资建议]")
