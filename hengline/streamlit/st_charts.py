#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图表生成模块
提供各种股票数据的可视化图表功能
"""

import os
import sys
from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.date_utils import format_date_for_chart
from .st_utils import COLOR_SCHEME, format_number


class StockChartGenerator:
    """股票图表生成器类"""

    @staticmethod
    def create_enhanced_kline_chart(df: pd.DataFrame, stock_code: str,
                                    show_ma: bool = True, ma_periods: list = None, show_volume: bool = True,
                                    show_rsi: bool = False) -> go.Figure:
        """
        创建增强版K线图，支持多种技术指标叠加
        
        Args:
            df: 价格数据DataFrame
            stock_code: 股票代码
            show_ma: 是否显示移动平均线
            ma_periods: 移动平均线周期列表，如果为空则使用默认值[5, 10, 20]
            show_volume: 是否显示成交量
            show_rsi: 是否显示RSI指标
        
        Returns:
            Plotly图表对象
        """
        # 创建图表对象
        fig = go.Figure()

        # 格式化X轴日期为x月x日格式
        formatted_dates = [format_date_for_chart(date) for date in df.index]

        # 添加主图表区域（K线图）
        fig.add_trace(go.Candlestick(
            x=formatted_dates,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='K线',
            increasing='red',
            decreasing='green',
        ))

        # 定义颜色列表
        ma_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA502', '#9370DB', '#3CB371']
        
        # 添加移动平均线
        if show_ma:
            # 如果未提供ma_periods或为空，使用默认值
            if ma_periods is None or len(ma_periods) == 0:
                ma_periods = [5, 10, 20]
            
            # 为每个周期创建移动平均线
            mas = []
            for i, period in enumerate(ma_periods):
                # 循环使用颜色列表
                color = ma_colors[i % len(ma_colors)]
                mas.append((period, f'MA{period}', color))
            
            for window, name, color in mas:
                if len(df) >= window:
                    ma = df['Close'].rolling(window=window).mean()
                    fig.add_trace(go.Scatter(
                        x=formatted_dates,
                        y=ma,
                        mode='lines',
                        name=name,
                        line=dict(color=color, width=1.5)
                    ))

        # 图表布局设置
        layout_kwargs = {
            'title': "K线图",
            'xaxis_title': "日期",
            'yaxis_title': "价格 (¥)",
            'height': 700 if show_rsi else 500,
            'margin': dict(l=60, r=60, t=50, b=20)
        }

        # 添加成交量子图
        if show_volume and 'Volume' in df.columns:
            # 创建子图
            from plotly.subplots import make_subplots
            fig = make_subplots(
                rows=3 if show_rsi else 2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.15, 0.15] if show_rsi else [0.7, 0.15],
                specs=[[{'rowspan': 1, 'secondary_y': False}],
                       [{'rowspan': 1, 'secondary_y': False}],
                       [{'rowspan': 1, 'secondary_y': False}]] if show_rsi else
                [[{'rowspan': 1, 'secondary_y': False}],
                 [{'rowspan': 1, 'secondary_y': False}]]
            )

            # 重新添加K线图和移动平均线到第一个子图
            fig.add_trace(
                go.Candlestick(
                    x=formatted_dates,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='K线',
                    increasing='red',
                    decreasing='green'
                ),
                row=1, col=1
            )

            # 重新添加移动平均线
            if show_ma:
                # 重新创建mas列表，确保与主图表一致
                if ma_periods is None or len(ma_periods) == 0:
                    ma_periods = [5, 10, 20]
                
                ma_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA502', '#9370DB', '#3CB371']
                mas = []
                for i, period in enumerate(ma_periods):
                    color = ma_colors[i % len(ma_colors)]
                    mas.append((period, f'MA{period}', color))
                
                for window, name, color in mas:
                    if len(df) >= window:
                        ma = df['Close'].rolling(window=window).mean()
                        fig.add_trace(
                            go.Scatter(
                                x=formatted_dates,
                                y=ma,
                                mode='lines',
                                name=name,
                                line=dict(color=color, width=1.5)
                            ),
                            row=1, col=1
                        )

            # 添加成交量
            # 根据涨跌设置成交量颜色
            volume_colors = []
            for i in range(len(df)):
                if i == 0:
                    volume_colors.append('gray')
                else:
                    if df.iloc[i]['Close'] >= df.iloc[i - 1]['Close']:
                        volume_colors.append('red')
                    else:
                        volume_colors.append('green')

            fig.add_trace(
                go.Bar(
                    x=formatted_dates,
                    y=df['Volume'],
                    name='成交量',
                    marker=volume_colors,
                    opacity=0.7
                ),
                row=2, col=1
            )

            fig.update_yaxes(title_text="成交量", row=2, col=1)
            fig.update_xaxes(title_text="日期", row=2, col=1)

            # 添加RSI指标
            if show_rsi:
                # 计算RSI
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))

                fig.add_trace(
                    go.Scatter(
                        x=formatted_dates,
                        y=rsi,
                        mode='lines',
                        name='RSI',
                        line=dict(color=COLOR_SCHEME['info'], width=2)
                    ),
                    row=3, col=1
                )

                # 添加超买超卖线
                fig.add_shape(
                    type="line",
                    x0=formatted_dates[0], x1=formatted_dates[-1],
                    y0=70, y1=70,
                    line=dict(color=COLOR_SCHEME['danger'], width=1, dash="dash"),
                    row=3, col=1
                )
                fig.add_shape(
                    type="line",
                    x0=formatted_dates[0], x1=formatted_dates[-1],
                    y0=30, y1=30,
                    line=dict(color=COLOR_SCHEME['success'], width=1, dash="dash"),
                    row=3, col=1
                )

                fig.add_annotation(
                    x=formatted_dates[0], y=72,
                    text="超买区域",
                    showarrow=False,
                    font=dict(color=COLOR_SCHEME['danger']),
                    row=3, col=1
                )
                fig.add_annotation(
                    x=formatted_dates[0], y=28,
                    text="超卖区域",
                    showarrow=False,
                    font=dict(color=COLOR_SCHEME['success']),
                    row=3, col=1
                )

                fig.update_xaxes(title_text="日期", row=3, col=1)

                fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])

        # 更新布局
        fig.update_layout(
            **layout_kwargs,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    @staticmethod
    def create_financial_comparison_chart(financial_data: Dict[str, pd.DataFrame],
                                          metric1: str = "totalRevenue",
                                          metric2: str = "netIncome") -> Optional[go.Figure]:
        """
        创建财务指标对比图
        
        Args:
            financial_data: 财务数据字典
            metric1: 第一个指标
            metric2: 第二个指标
        
        Returns:
            Plotly图表对象或None
        """
        # 首先验证输入是否为有效的字典
        if not isinstance(financial_data, dict):
            return None

        # 检查必要的数据是否存在
        if "income_statement" not in financial_data:
            return None

        income_df = financial_data["income_statement"]

        # 验证DataFrame的有效性
        if not (isinstance(income_df, pd.DataFrame) and not income_df.empty):
            return None

        # 指标名称映射（英文到中文）
        metric_names = {
            "totalRevenue": "总营收",
            "netIncome": "净利润",
            "grossProfit": "毛利润"
        }

        try:
            # 检查所需列是否存在
            if metric1 in income_df.columns and metric2 in income_df.columns:
                # 创建双Y轴图表
                fig = go.Figure()

                # 第一个指标（主Y轴）
                metric1_name = metric_names.get(metric1, metric1)  # 使用中文名称或保持原样
                fig.add_trace(
                    go.Bar(
                        x=income_df.index,
                        y=income_df[metric1],
                        name=metric1_name,
                        marker_color=COLOR_SCHEME['primary'],
                        yaxis="y1"
                    )
                )

                # 第二个指标（次Y轴）
                metric2_name = metric_names.get(metric2, metric2)  # 使用中文名称或保持原样
                fig.add_trace(
                    go.Scatter(
                        x=income_df.index,
                        y=income_df[metric2],
                        mode='lines+markers',
                        name=metric2_name,
                        line=dict(color=COLOR_SCHEME['secondary'], width=3),
                        yaxis="y2"
                    )
                )

                # 更新布局
                fig.update_layout(
                    title=f"{metric_names.get(metric1, metric1)} 与 {metric_names.get(metric2, metric2)} 对比",
                    xaxis_title="日期",
                    yaxis_title=metric_names.get(metric1, metric1) + " (¥)" if isinstance(metric_names.get(metric1, metric1), str) and metric_names.get(metric1, metric1) not in ['RSI', 'MA', '成交量'] else metric_names.get(metric1, metric1),
                    yaxis2=dict(
                        title=metric_names.get(metric2, metric2),
                        overlaying="y",
                        side="right"
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=500
                )

                return fig
        except Exception as e:
            # 发生任何异常时返回None
            pass

        return None

    @staticmethod
    def create_balance_sheet_chart(financial_data: Dict[str, pd.DataFrame]) -> Optional[go.Figure]:
        """
        创建资产负债结构图表
        
        Args:
            financial_data: 财务数据字典
        
        Returns:
            Plotly图表对象或None
        """
        # 首先验证输入是否为有效的字典
        if not isinstance(financial_data, dict):
            return None

        # 检查必要的数据是否存在
        if "balance_sheet" not in financial_data:
            return None

        balance_df = financial_data["balance_sheet"]

        # 验证DataFrame的有效性
        if not (isinstance(balance_df, pd.DataFrame) and not balance_df.empty):
            return None

        try:
            # 选择最新数据
            latest_data = balance_df.iloc[-1]

            # 准备资产数据
            assets = []
            asset_labels = []

            # 主要资产项目
            asset_items = [
                ("cash", "现金"),
                ("shortTermInvestments", "短期投资"),
                ("netReceivables", "应收账款净额"),
                ("inventory", "存货"),
                ("totalCurrentAssets", "流动资产合计"),
                ("propertyPlantEquipment", "固定资产"),
                ("totalNonCurrentAssets", "非流动资产合计")
            ]

            for key, label in asset_items:
                if key in latest_data.index and not pd.isna(latest_data[key]) and latest_data[key] > 0:
                    assets.append(latest_data[key])
                    asset_labels.append(label)

            # 准备负债数据
            liabilities = []
            liability_labels = []

            # 主要负债项目
            liability_items = [
                ("shortTermDebt", "短期负债"),
                ("accountsPayable", "应付账款"),
                ("totalCurrentLiabilities", "流动负债合计"),
                ("longTermDebt", "长期负债"),
                ("totalNonCurrentLiabilities", "非流动负债合计"),
                ("totalLiabilities", "总负债"),
                ("totalEquity", "股东权益")
            ]

            for key, label in liability_items:
                if key in latest_data.index and not pd.isna(latest_data[key]) and latest_data[key] > 0:
                    liabilities.append(latest_data[key])
                    liability_labels.append(label)

            # 检查是否有足够的数据创建图表
            if not assets and not liabilities:
                return None

            # 创建子图
            from plotly.subplots import make_subplots
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'pie'}, {'type': 'pie'}]],
                subplot_titles=('资产结构', '负债与权益结构')
            )

            # 添加饼图
            if assets and asset_labels:
                fig.add_trace(
                    go.Pie(
                        labels=asset_labels,
                        values=assets,
                        text=[format_number(v) for v in assets],
                        textinfo='label+text+percent',
                        insidetextorientation='radial'
                    ),
                    row=1, col=1
                )

            if liabilities and liability_labels:
                fig.add_trace(
                    go.Pie(
                        labels=liability_labels,
                        values=liabilities,
                        text=[format_number(v) for v in liabilities],
                        textinfo='label+text+percent',
                        insidetextorientation='radial'
                    ),
                    row=1, col=2
                )

            fig.update_layout(
                title="资产负债结构分析",
                height=600
            )

            return fig
        except Exception as e:
            # 发生任何异常时返回None
            pass

        return None

    @staticmethod
    def create_cash_flow_chart(financial_data: Dict[str, pd.DataFrame]) -> Optional[go.Figure]:
        """
        创建现金流分析图表
        
        Args:
            financial_data: 财务数据字典
        
        Returns:
            Plotly图表对象或None
        """
        # 首先验证输入是否为有效的字典
        if not isinstance(financial_data, dict):
            return None

        # 检查必要的数据是否存在
        if "cash_flow" not in financial_data:
            return None

        cash_df = financial_data["cash_flow"]

        # 验证DataFrame的有效性
        if not (isinstance(cash_df, pd.DataFrame) and not cash_df.empty):
            return None

        try:
            # 选择关键现金流指标
            cash_flow_items = [
                ("operatingCashFlow", "经营现金流", COLOR_SCHEME['success']),
                ("investingCashFlow", "投资现金流", COLOR_SCHEME['danger']),
                ("financingCashFlow", "筹资现金流", COLOR_SCHEME['warning']),
                ("freeCashFlow", "自由现金流", COLOR_SCHEME['primary'])
            ]

            # 创建堆叠柱状图
            fig = go.Figure()

            # 跟踪是否添加了至少一个数据系列
            added_traces = False

            # 添加各个现金流项目
            for key, label, color in cash_flow_items:
                if key in cash_df.columns:
                    # 检查该列是否有有效值
                    if not cash_df[key].isnull().all():
                        fig.add_trace(
                            go.Bar(
                                x=cash_df.index,
                                y=cash_df[key],
                                name=label,
                                marker_color=color
                            )
                        )
                        added_traces = True

            # 如果没有添加任何数据系列，则返回None
            if not added_traces:
                return None

            # 添加零线
            fig.add_shape(
                type="line",
                x0=cash_df.index[0], x1=cash_df.index[-1],
                y0=0, y1=0,
                line=dict(color="black", width=1)
            )

            fig.update_layout(
                title="现金流结构分析",
                xaxis_title="日期",
                yaxis_title="金额 (¥)",
                barmode='group',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=500
            )

            return fig
        except Exception as e:
            # 发生任何异常时返回None
            pass

        return None

    @staticmethod
    def create_radar_chart(financial_ratios: Dict[str, float]) -> Optional[go.Figure]:
        """
        创建财务比率雷达图
        
        Args:
            financial_ratios: 财务比率字典
        
        Returns:
            Plotly图表对象或None
        """
        if not financial_ratios:
            return None

        # 准备雷达图数据
        categories = list(financial_ratios.keys())
        values = list(financial_ratios.values())

        # 设置雷达图的最大值（根据指标类型调整）
        max_values = []
        for i, category in enumerate(categories):
            # 对于率类指标，设置合理的最大值
            if "率" in category:
                if "资产负债率" in category:
                    max_values.append(100)  # 资产负债率最大100%
                else:
                    # 其他比率设为当前最大值的1.2倍或50的较大值
                    max_values.append(max(values[i] * 1.2, 50))
            else:
                # 非率类指标设为当前最大值的1.2倍
                max_values.append(values[i] * 1.2)

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='当前值'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max_values)]
                )),
            showlegend=True,
            title="财务比率雷达图",
            height=500
        )

        return fig
