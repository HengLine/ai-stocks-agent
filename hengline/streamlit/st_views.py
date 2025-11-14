#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据分析视图模块
提供多种股票数据分析视图
"""

from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from .st_charts import StockChartGenerator
from .st_utils import COLOR_SCHEME, format_number, calculate_financial_ratios


class StockDataViews:
    """股票数据分析视图类"""

    @staticmethod
    def show_overview_view(stock_code: str, stock_info: Dict[str, Any],
                           price_data: Optional[pd.DataFrame] = None) -> None:
        """
        显示股票概览视图
        
        Args:
            stock_code: 股票代码
            stock_info: 股票基本信息
            price_data: 价格数据（可选）
        """
        st.title(f"{stock_code} 股票概览")

        # 股票基本信息卡片
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="公司名称",
                value=stock_info.get('shortName', 'N/A')
            )

        with col2:
            st.metric(
                label="行业",
                value=stock_info.get('sector', 'N/A')
            )

        with col3:
            st.metric(
                label="市值",
                value=format_number(stock_info.get('marketCap', 0))
            )

        # 价格相关指标
        if price_data is not None and not price_data.empty:
            latest_price = price_data['Close'].iloc[-1]
            prev_price = price_data['Close'].iloc[-2]
            change_percent = (latest_price - prev_price) / prev_price * 100

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="最新价格",
                    value=f"${latest_price:.2f}",
                    delta=f"{change_percent:.2f}%",
                    delta_color="normal"
                )

            with col2:
                st.metric(
                    label="52周最高",
                    value=f"${price_data['High'].max():.2f}"
                )

            with col3:
                st.metric(
                    label="52周最低",
                    value=f"${price_data['Low'].min():.2f}"
                )

        # 财务摘要
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="每股收益(EPS)",
                value=f"${stock_info.get('epsTrailingTwelveMonths', 0):.2f}"
            )

        with col2:
            st.metric(
                label="市盈率(P/E)",
                value=f"{stock_info.get('trailingPE', 0):.2f}"
            )

        with col3:
            st.metric(
                label="股息率",
                value=f"{stock_info.get('dividendYield', 0) * 100:.2f}%"
            )

        # 公司描述
        if 'longBusinessSummary' in stock_info:
            st.subheader("公司简介")
            st.write(stock_info['longBusinessSummary'])

    @staticmethod
    def show_price_chart_view(stock_code: str, price_data: pd.DataFrame) -> None:
        """
        显示价格图表视图
        
        Args:
            stock_code: 股票代码
            price_data: 价格数据
        """
        st.title(f"{stock_code} 价格走势")

        # 图表选项
        col1, col2, col3 = st.columns(3)

        with col1:
            show_ma = st.checkbox("显示均线", value=True)

        with col2:
            show_volume = st.checkbox("显示成交量", value=True)

        with col3:
            show_rsi = st.checkbox("显示RSI", value=False)

        # 生成并显示K线图
        fig = StockChartGenerator.create_enhanced_kline_chart(
            df=price_data,
            stock_code=stock_code,
            show_ma=show_ma,
            ma_periods=[5, 10, 20, 60],  # 添加默认均线周期
            show_volume=show_volume,
            show_rsi=show_rsi
        )

        st.plotly_chart(fig, use_container_width=True)

        # 价格统计摘要
        st.subheader("价格统计摘要")
        price_stats = {
            "最新价格": price_data['Close'].iloc[-1],
            "平均价格": price_data['Close'].mean(),
            "价格波动幅度": f"{price_data['Close'].pct_change().std() * 100:.2f}%",
            "最高价": price_data['High'].max(),
            "最低价": price_data['Low'].min(),
            "平均成交量": price_data['Volume'].mean()
        }

        for key, value in price_stats.items():
            st.text(f"{key}: {format_number(value) if isinstance(value, (int, float)) else value}")

    @staticmethod
    def show_financial_analysis_view(stock_code: str, financial_data: Dict[str, pd.DataFrame]) -> None:
        """
        显示财务分析视图
        
        Args:
            stock_code: 股票代码
            financial_data: 财务数据字典
        """
        st.title(f"{stock_code} 财务分析")

        # 财务数据标签页
        financial_tabs = st.tabs(["财务对比", "资产负债", "现金流", "财务比率"])

        # 1. 财务对比标签页
        with financial_tabs[0]:
            if "income_statement" in financial_data and not financial_data["income_statement"].empty:
                # 选择要对比的指标
                income_df = financial_data["income_statement"]
                available_metrics = [col for col in income_df.columns if not pd.api.types.is_numeric_dtype(income_df[col]) is False]

                if available_metrics:
                    col1, col2 = st.columns(2)

                    with col1:
                        metric1 = st.selectbox(
                            "选择第一个指标",
                            options=available_metrics,
                            index=available_metrics.index("totalRevenue") if "totalRevenue" in available_metrics else 0
                        )

                    with col2:
                        metric2 = st.selectbox(
                            "选择第二个指标",
                            options=available_metrics,
                            index=available_metrics.index("netIncome") if "netIncome" in available_metrics else 0
                        )

                    # 生成对比图表
                    fig = StockChartGenerator.create_financial_comparison_chart(
                        financial_data, metric1, metric2
                    )

                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("无法生成财务对比图表，数据可能不完整")

                    # 显示收入报表数据表格
                    st.subheader("收入报表数据")
                    st.dataframe(income_df.style.format("{:,.0f}"), use_container_width=True)
                else:
                    st.warning("收入报表中没有可比较的数值指标")
            else:
                st.warning("未找到收入报表数据")

        # 2. 资产负债标签页
        with financial_tabs[1]:
            if "balance_sheet" in financial_data and not financial_data["balance_sheet"].empty:
                # 生成资产负债结构图表
                fig = StockChartGenerator.create_balance_sheet_chart(financial_data)

                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("无法生成资产负债结构图表，数据可能不完整")

                # 显示资产负债表数据
                st.subheader("资产负债表数据")
                st.dataframe(financial_data["balance_sheet"].style.format("{:,.0f}"), use_container_width=True)
            else:
                st.warning("未找到资产负债表数据")

        # 3. 现金流标签页
        with financial_tabs[2]:
            if "cash_flow" in financial_data and not financial_data["cash_flow"].empty:
                # 生成现金流分析图表
                fig = StockChartGenerator.create_cash_flow_chart(financial_data)

                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("无法生成现金流分析图表，数据可能不完整")

                # 显示现金流量表数据
                st.subheader("现金流量表数据")
                st.dataframe(financial_data["cash_flow"].style.format("{:,.0f}"), use_container_width=True)
            else:
                st.warning("未找到现金流量表数据")

        # 4. 财务比率标签页
        with financial_tabs[3]:
            if financial_data and any(not df.empty for df in financial_data.values()):
                # 计算财务比率
                financial_ratios = calculate_financial_ratios(financial_data)

                if financial_ratios:
                    # 显示财务比率雷达图
                    fig = StockChartGenerator.create_radar_chart(financial_ratios)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                    # 显示财务比率表格
                    st.subheader("关键财务比率")
                    ratio_df = pd.DataFrame(
                        list(financial_ratios.items()),
                        columns=["指标", "值"]
                    )
                    st.table(ratio_df)
                else:
                    st.warning("无法计算财务比率，数据可能不完整")
            else:
                st.warning("财务数据不完整，无法计算财务比率")

    @staticmethod
    def show_news_view(stock_code: str, news_data: list) -> None:
        """
        显示新闻视图
        
        Args:
            stock_code: 股票代码
            news_data: 新闻数据列表
        """
        st.title(f"{stock_code} 最新资讯")

        if news_data:
            # 显示新闻列表
            for i, news in enumerate(news_data, 1):
                with st.expander(f"新闻 {i}: {news.get('title', '无标题')}"):
                    # 显示新闻详情
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.subheader(news.get('title', '无标题'))
                        st.write(f"发布时间: {news.get('publishedAt', '未知')}")
                        st.write(news.get('description', '无描述'))

                        # 如果有链接，显示链接
                        if 'url' in news:
                            st.markdown(f"[查看原文]({news['url']})")

                    with col2:
                        # 显示图片（如果有）
                        if 'image' in news:
                            st.image(news['image'], width=100)
        else:
            st.info("未找到相关新闻")

    @staticmethod
    def show_advanced_analysis_view(stock_code: str, price_data: pd.DataFrame) -> None:
        """
        显示高级分析视图
        
        Args:
            stock_code: 股票代码
            price_data: 价格数据
        """
        st.title(f"{stock_code} 高级分析")

        # 技术指标计算与显示
        st.subheader("技术指标分析")

        # 检查是否有足够的数据进行计算
        if len(price_data) >= 20:
            # 计算常用技术指标
            # 1. 移动平均线
            price_data['MA5'] = price_data['Close'].rolling(window=5).mean()
            price_data['MA10'] = price_data['Close'].rolling(window=10).mean()
            price_data['MA20'] = price_data['Close'].rolling(window=20).mean()

            # 2. 相对强弱指标(RSI)
            delta = price_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            price_data['RSI'] = 100 - (100 / (1 + rs))

            # 3. MACD
            exp1 = price_data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = price_data['Close'].ewm(span=26, adjust=False).mean()
            price_data['MACD'] = exp1 - exp2
            price_data['Signal'] = price_data['MACD'].ewm(span=9, adjust=False).mean()

            # 显示指标图表
            analysis_tabs = st.tabs(["移动平均线", "RSI指标", "MACD指标"])

            # 1. 移动平均线图表
            with analysis_tabs[0]:
                fig = go.Figure()

                # 添加收盘价
                fig.add_trace(
                    go.Scatter(
                        x=price_data.index,
                        y=price_data['Close'],
                        mode='lines',
                        name='收盘价',
                        line=dict(color=COLOR_SCHEME['primary'])
                    )
                )

                # 添加各均线
                fig.add_trace(
                    go.Scatter(
                        x=price_data.index,
                        y=price_data['MA5'],
                        mode='lines',
                        name='MA5',
                        line=dict(color=COLOR_SCHEME['success'])
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=price_data.index,
                        y=price_data['MA10'],
                        mode='lines',
                        name='MA10',
                        line=dict(color=COLOR_SCHEME['warning'])
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=price_data.index,
                        y=price_data['MA20'],
                        mode='lines',
                        name='MA20',
                        line=dict(color=COLOR_SCHEME['danger'])
                    )
                )

                fig.update_layout(
                    title="移动平均线分析",
                    xaxis_title="日期",
                    yaxis_title="价格 (¥)",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

            # 2. RSI指标图表
            with analysis_tabs[1]:
                fig = go.Figure()

                # 添加RSI线
                fig.add_trace(
                    go.Scatter(
                        x=price_data.index,
                        y=price_data['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color=COLOR_SCHEME['info'])
                    )
                )

                # 添加超买超卖线
                fig.add_shape(
                    type="line",
                    x0=price_data.index[0], x1=price_data.index[-1],
                    y0=70, y1=70,
                    line=dict(color=COLOR_SCHEME['danger'], width=1, dash="dash")
                )

                fig.add_shape(
                    type="line",
                    x0=price_data.index[0], x1=price_data.index[-1],
                    y0=30, y1=30,
                    line=dict(color=COLOR_SCHEME['success'], width=1, dash="dash")
                )

                fig.add_annotation(
                    x=price_data.index[0], y=72,
                    text="超买区域",
                    showarrow=False,
                    font=dict(color=COLOR_SCHEME['danger'])
                )

                fig.add_annotation(
                    x=price_data.index[0], y=28,
                    text="超卖区域",
                    showarrow=False,
                    font=dict(color=COLOR_SCHEME['success'])
                )

                fig.update_layout(
                    title="RSI指标分析",
                    xaxis_title="日期",
                    yaxis_title="RSI值",
                    yaxis=dict(range=[0, 100]),
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

            # 3. MACD指标图表
            with analysis_tabs[2]:
                fig = go.Figure()

                # 添加MACD线
                fig.add_trace(
                    go.Scatter(
                        x=price_data.index,
                        y=price_data['MACD'],
                        mode='lines',
                        name='MACD',
                        line=dict(color=COLOR_SCHEME['primary'])
                    )
                )

                # 添加信号线
                fig.add_trace(
                    go.Scatter(
                        x=price_data.index,
                        y=price_data['Signal'],
                        mode='lines',
                        name='Signal',
                        line=dict(color=COLOR_SCHEME['secondary'])
                    )
                )

                # 添加柱状图表示MACD与Signal的差值
                fig.add_trace(
                    go.Bar(
                        x=price_data.index,
                        y=price_data['MACD'] - price_data['Signal'],
                        name='MACD差值',
                        marker_color=COLOR_SCHEME['info'],
                        opacity=0.5
                    )
                )

                fig.update_layout(
                    title="MACD指标分析",
                    xaxis_title="日期",
                    yaxis_title="值",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("数据量不足，无法进行高级技术指标分析")

    @staticmethod
    def show_comparison_view(stock_codes: list, price_data_dict: Dict[str, pd.DataFrame]) -> None:
        """
        显示多只股票比较视图
        
        Args:
            stock_codes: 股票代码列表
            price_data_dict: 各股票的价格数据字典
        """
        st.title("多股票对比分析")

        # 检查是否有有效的价格数据
        valid_stocks = {code: data for code, data in price_data_dict.items()
                        if data is not None and not data.empty}

        if len(valid_stocks) >= 2:
            # 创建价格对比图表
            fig = go.Figure()

            # 对每只股票进行标准化处理，以百分比变化显示
            for code, data in valid_stocks.items():
                # 计算相对于起始价格的百分比变化
                norm_price = (data['Close'] / data['Close'].iloc[0] - 1) * 100

                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=norm_price,
                        mode='lines',
                        name=code
                    )
                )

            fig.update_layout(
                title="股票价格相对表现对比",
                xaxis_title="日期",
                yaxis_title="相对涨幅(%)",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # 显示各股票的统计数据对比
            st.subheader("股票表现统计对比")

            comparison_data = []
            for code, data in valid_stocks.items():
                # 计算收益率和波动率
                returns = data['Close'].pct_change().dropna()
                total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                volatility = returns.std() * np.sqrt(252) * 100  # 年化波动率

                comparison_data.append({
                    "股票代码": code,
                    "总收益率(%)": f"{total_return:.2f}",
                    "年化波动率(%)": f"{volatility:.2f}",
                    "平均日收益率(%)": f"{returns.mean() * 100:.2f}",
                    "最大回撤(%)": f"{StockDataViews._calculate_max_drawdown(data['Close']) * 100:.2f}"
                })

            # 显示对比表格
            comparison_df = pd.DataFrame(comparison_data)
            st.table(comparison_df)
        elif len(valid_stocks) == 1:
            st.info("需要至少两只股票才能进行比较")
        else:
            st.warning("没有找到有效的股票数据进行比较")

    @staticmethod
    def _calculate_max_drawdown(price_series: pd.Series) -> float:
        """
        计算最大回撤
        
        Args:
            price_series: 价格序列
        
        Returns:
            最大回撤值（小数形式）
        """
        cumulative_max = price_series.cummax()
        drawdown = (price_series - cumulative_max) / cumulative_max
        return drawdown.min()
