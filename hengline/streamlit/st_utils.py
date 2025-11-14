#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit应用的工具函数模块
提供数据处理、图表生成和UI增强功能
"""

from typing import Dict, Any, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 导入统一的日期处理工具
from utils.date_utils import format_date

# 定义颜色主题
COLOR_SCHEME = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ffbb28',
    'info': '#9467bd',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

# 样式CSS
CUSTOM_CSS = """
<style>
    .stock-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
    }
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .metric-item {
        flex: 1;
        min-width: 150px;
    }
    .highlight {
        background-color: #fff3cd;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
    }
</style>
"""


def apply_custom_css():
    """应用自定义CSS样式"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def format_number(num: Any) -> str:
    """
    格式化数字显示，以人民币为单位
    
    Args:
        num: 要格式化的数字
    
    Returns:
        格式化后的字符串，带人民币符号
    """
    try:
        num = float(num)
        if abs(num) >= 1e12:
            return f"¥{num / 1e12:.2f}万亿"
        elif abs(num) >= 1e8:
            return f"¥{num / 1e8:.2f}亿"
        elif abs(num) >= 1e4:
            return f"¥{num / 1e4:.2f}万"
        else:
            return f"¥{num:.2f}"
    except (ValueError, TypeError):
        return str(num)


def calculate_percentage_change(current: float, previous: float) -> Optional[float]:
    """
    计算百分比变化
    
    Args:
        current: 当前值
        previous: 之前的值
    
    Returns:
        百分比变化值
    """
    if previous == 0:
        return None
    return ((current - previous) / previous) * 100


def create_kline_chart(df: pd.DataFrame, stock_code: str) -> go.Figure:
    """
    创建K线图
    
    Args:
        df: 包含价格数据的DataFrame
        stock_code: 股票代码
    
    Returns:
        Plotly图表对象
    """
    fig = go.Figure()

    # 格式化日期索引
    formatted_dates = [format_date(date) for date in df.index]

    # 添加蜡烛图
    fig.add_trace(go.Candlestick(
        x=formatted_dates,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='K线'
    ))

    # 添加成交量柱状图
    if 'Volume' in df.columns:
        fig.add_trace(go.Bar(
            x=formatted_dates,
            y=df['Volume'],
            name='成交量',
            yaxis='y2',
            meta=COLOR_SCHEME['primary'],
            opacity=0.3
        ))

    # 设置双Y轴
    fig.update_layout(
        title=f"{stock_code} K线图",
        xaxis_title="日期",
        yaxis_title="价格 (¥)",
        yaxis2=dict(
            title="成交量",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        hovermode='x unified',
        height=500,
        margin=dict(l=60, r=60, t=50, b=50)
    )

    return fig


def create_line_chart(df: pd.DataFrame, title: str, x_column: str = None,
                      y_columns: List[str] = None) -> go.Figure:
    """
    创建折线图
    
    Args:
        df: 数据DataFrame
        title: 图表标题
        x_column: X轴列名，如果为None则使用索引
        y_columns: Y轴列名列表
    
    Returns:
        Plotly图表对象
    """
    fig = go.Figure()

    if x_column is None:
        # 格式化日期索引
        x_data = [format_date(date) for date in df.index]
    else:
        x_data = df[x_column]

    colors = [COLOR_SCHEME['primary'], COLOR_SCHEME['secondary'],
              COLOR_SCHEME['success'], COLOR_SCHEME['danger'],
              COLOR_SCHEME['warning']]

    for i, col in enumerate(y_columns):
        if col in df.columns:
            color_idx = i % len(colors)
            fig.add_trace(go.Scatter(
                x=x_data,
                y=df[col],
                mode='lines+markers',
                name=col,
                line=dict(color=colors[color_idx])
            ))

    fig.update_layout(
        title=title,
        xaxis_title=x_column if x_column else "日期",
        yaxis_title="数值",
        hovermode='x unified',
        height=400
    )

    return fig


def create_bar_chart(df: pd.DataFrame, title: str, x_column: str = None,
                     y_columns: List[str] = None) -> go.Figure:
    """
    创建柱状图
    
    Args:
        df: 数据DataFrame
        title: 图表标题
        x_column: X轴列名，如果为None则使用索引
        y_columns: Y轴列名列表
    
    Returns:
        Plotly图表对象
    """
    fig = go.Figure()

    if x_column is None:
        # 格式化日期索引
        x_data = [format_date(date) for date in df.index]
    else:
        x_data = df[x_column]

    colors = [COLOR_SCHEME['primary'], COLOR_SCHEME['secondary'],
              COLOR_SCHEME['success'], COLOR_SCHEME['danger']]

    for i, col in enumerate(y_columns):
        if col in df.columns:
            color_idx = i % len(colors)
            fig.add_trace(go.Bar(
                x=x_data,
                y=df[col],
                name=col,
                meta=colors[color_idx]
            ))

    fig.update_layout(
        title=title,
        xaxis_title=x_column if x_column else "日期",
        yaxis_title="数值",
        barmode='group',
        height=400
    )

    return fig


def create_pie_chart(labels: List[str], values: List[float], title: str) -> go.Figure:
    """
    创建饼图
    
    Args:
        labels: 标签列表
        values: 数值列表
        title: 图表标题
    
    Returns:
        Plotly图表对象
    """
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        textinfo='label+percent'
    )])

    fig.update_layout(
        title=title,
        height=400
    )

    return fig


def display_metrics(metrics: Dict[str, float], title: str = "关键指标"):
    """
    显示指标卡片
    
    Args:
        metrics: 指标字典 {名称: 值}
        title: 标题
    """
    st.subheader(title)
    cols = st.columns(min(len(metrics), 4))
    for i, (name, value) in enumerate(metrics.items()):
        col_idx = i % len(cols)
        with cols[col_idx]:
            st.metric(
                label=name,
                value=format_number(value) if isinstance(value, (int, float)) else value
            )


def calculate_financial_ratios(financial_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    计算财务比率
    
    Args:
        financial_data: 财务数据字典
    
    Returns:
        财务比率字典
    """
    ratios = {}

    # 首先验证输入是否为有效的字典
    if not isinstance(financial_data, dict):
        return ratios

    # 获取收入表和资产负债表，并验证它们是有效的DataFrame
    income_df = financial_data.get("income_statement")
    balance_df = financial_data.get("balance_sheet")

    # 验证DataFrame的有效性
    valid_income = isinstance(income_df, pd.DataFrame) and not income_df.empty and len(income_df.columns) > 0
    valid_balance = isinstance(balance_df, pd.DataFrame) and not balance_df.empty and len(balance_df.columns) > 0

    if valid_income and valid_balance:
        try:
            # 按日期排序，获取最新数据
            income_df = income_df.sort_index(ascending=False)
            balance_df = balance_df.sort_index(ascending=False)

            # 毛利率
            try:
                if "grossProfit" in income_df.columns and "totalRevenue" in income_df.columns:
                    revenue = income_df.iloc[0]["totalRevenue"]
                    gross_profit = income_df.iloc[0]["grossProfit"]
                    if pd.notna(revenue) and pd.notna(gross_profit) and revenue > 0:
                        ratios["毛利率"] = (gross_profit / revenue) * 100
            except Exception:
                pass

            # 净利率
            try:
                if "netIncome" in income_df.columns and "totalRevenue" in income_df.columns:
                    revenue = income_df.iloc[0]["totalRevenue"]
                    net_income = income_df.iloc[0]["netIncome"]
                    if pd.notna(revenue) and pd.notna(net_income) and revenue > 0:
                        ratios["净利率"] = (net_income / revenue) * 100
            except Exception:
                pass

            # 资产负债率
            try:
                if "totalLiabilities" in balance_df.columns and "totalAssets" in balance_df.columns:
                    total_assets = balance_df.iloc[0]["totalAssets"]
                    total_liabilities = balance_df.iloc[0]["totalLiabilities"]
                    if pd.notna(total_assets) and pd.notna(total_liabilities) and total_assets > 0:
                        ratios["资产负债率"] = (total_liabilities / total_assets) * 100
            except Exception:
                pass

            # 净资产收益率 (ROE)
            try:
                if "netIncome" in income_df.columns and "totalEquity" in balance_df.columns:
                    net_income = income_df.iloc[0]["netIncome"]
                    equity = balance_df.iloc[0]["totalEquity"]
                    if pd.notna(net_income) and pd.notna(equity) and equity > 0:
                        ratios["净资产收益率"] = (net_income / equity) * 100
            except Exception:
                pass
        except Exception:
            pass

    return ratios


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算技术指标
    
    Args:
        df: 价格数据DataFrame
    
    Returns:
        添加了技术指标的DataFrame
    """
    result_df = df.copy()

    # 计算移动平均线
    result_df['MA5'] = result_df['Close'].rolling(window=5).mean()
    result_df['MA10'] = result_df['Close'].rolling(window=10).mean()
    result_df['MA20'] = result_df['Close'].rolling(window=20).mean()

    # 计算RSI (14天)
    delta = result_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    result_df['RSI'] = 100 - (100 / (1 + rs))

    # 计算MACD
    exp1 = result_df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = result_df['Close'].ewm(span=26, adjust=False).mean()
    result_df['MACD'] = exp1 - exp2
    result_df['Signal'] = result_df['MACD'].ewm(span=9, adjust=False).mean()

    return result_df
