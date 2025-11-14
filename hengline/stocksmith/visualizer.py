import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import logging
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional

# 首先创建日志器
try:
    from utils.logger import get_logger
    logger = get_logger('stocksmith_visualizer')
except ImportError:
    # 如果无法导入自定义日志器，使用基本的logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('stocksmith_visualizer')
    logger.warning("无法导入自定义日志器")

# 临时移除当前目录，确保能导入实际的langsmith库
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if p != current_dir]

# 导入实际的langsmith库
try:
    from langsmith import Client
except ImportError:
    # 如果导入失败，设置为None
    Client = None
    logger.warning("无法导入langsmith.Client，某些功能可能受限")

# 恢复sys.path
sys.path.append(current_dir)

# 尝试导入必要的模块
try:
    # 加载环境变量
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("无法加载环境变量")

class StocksVisualizer:
    """股票分析可视化工具，用于展示评估结果和工作流执行情况"""
    
    def __init__(self):
        """初始化可视化工具"""
        self.client = Client() if os.getenv("LANGCHAIN_API_KEY") else None
        
        # 设置中文字体支持
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
        
        # 确保LangSmith已配置
        if not self.client:
            logger.warning("LangSmith未配置，将使用本地数据进行可视化")
    
    def load_evaluation_results(self, file_path: str) -> Optional[pd.DataFrame]:
        """从文件加载评估结果"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # 解析结果为DataFrame
            if isinstance(results, list):
                # 本地评估结果格式
                data = []
                for i, result in enumerate(results):
                    row = {
                        "test_case_id": i + 1,
                        "query": result["query"][:50] + "..." if len(result["query"]) > 50 else result["query"],
                        "accuracy": result["scores"].get("accuracy", 0),
                        "relevance": result["scores"].get("relevance", 0),
                        "comprehensiveness": result["scores"].get("comprehensiveness", 0),
                        "overall": result["overall_score"]
                    }
                    data.append(row)
            elif isinstance(results, dict) and "results" in results:
                # LangSmith评估结果格式
                data = []
                for i, run in enumerate(results["results"]):
                    row = {
                        "test_case_id": i + 1,
                        "query": run["inputs"]["query"][:50] + "..." if len(run["inputs"]["query"]) > 50 else run["inputs"]["query"],
                        "accuracy": run["scores"].get("accuracy", 0),
                        "relevance": run["scores"].get("relevance", 0),
                        "comprehensiveness": run["scores"].get("comprehensiveness", 0),
                        "overall": run["scores"].get("overall", 0)
                    }
                    data.append(row)
            else:
                raise ValueError("未知的结果格式")
            
            df = pd.DataFrame(data)
            logger.info(f"已加载 {len(df)} 条评估结果")
            return df
        except Exception as e:
            logger.error(f"加载评估结果失败: {str(e)}")
            return None
    
    def fetch_langsmith_results(self, project_name: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """从LangSmith获取评估结果"""
        if not self.client:
            logger.error("LangSmith未配置，无法获取结果")
            return None
        
        try:
            # 获取项目中的所有运行
            runs = list(self.client.list_runs(project_name=project_name, limit=limit))
            
            data = []
            for i, run in enumerate(runs):
                # 获取运行的评估结果
                evaluations = list(self.client.list_evaluations(run_id=run.id))
                
                # 提取评估分数
                scores = {}
                for eval_run in evaluations:
                    if hasattr(eval_run, "metrics"):
                        scores.update(eval_run.metrics)
                    elif hasattr(eval_run, "score"):
                        scores["overall"] = eval_run.score
                
                # 构建数据行
                row = {
                    "run_id": run.id,
                    "test_case_id": i + 1,
                    "query": run.inputs.get("query", "")[:50] + "..." if len(run.inputs.get("query", "")) > 50 else run.inputs.get("query", ""),
                    "accuracy": scores.get("accuracy", 0),
                    "relevance": scores.get("relevance", 0),
                    "comprehensiveness": scores.get("comprehensiveness", 0),
                    "overall": scores.get("overall", 0),
                    "start_time": run.start_time,
                    "end_time": run.end_time,
                    "duration": (run.end_time - run.start_time).total_seconds() if run.end_time and run.start_time else None
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            logger.info(f"从LangSmith获取了 {len(df)} 条评估结果")
            return df
        except Exception as e:
            logger.error(f"从LangSmith获取结果失败: {str(e)}")
            return None
    
    def plot_score_distribution(self, df: pd.DataFrame, output_file: str = None) -> None:
        """绘制评分分布图表"""
        try:
            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 评分指标列表
            metrics = ["accuracy", "relevance", "comprehensiveness", "overall"]
            titles = ["准确性评分分布", "相关性评分分布", "全面性评分分布", "综合评分分布"]
            
            # 为每个指标绘制分布图
            for i, (metric, title) in enumerate(zip(metrics, titles)):
                ax = axes[i // 2, i % 2]
                sns.histplot(df[metric], bins=10, ax=ax, kde=True)
                ax.set_title(title)
                ax.set_xlabel("评分")
                ax.set_ylabel("频数")
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表或显示
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"评分分布图已保存到 {output_file}")
            else:
                plt.show()
        except Exception as e:
            logger.error(f"绘制评分分布图失败: {str(e)}")
    
    def plot_radar_chart(self, df: pd.DataFrame, output_file: str = None) -> None:
        """绘制雷达图展示各项指标的平均得分"""
        try:
            # 计算各项指标的平均值
            avg_scores = df[["accuracy", "relevance", "comprehensiveness", "overall"]].mean()
            
            # 创建雷达图
            categories = ["准确性", "相关性", "全面性", "综合评分"]
            values = avg_scores.tolist()
            values.append(values[0])  # 闭合雷达图
            categories.append(categories[0])
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            ax.plot(np.linspace(0, 2 * np.pi, len(categories)), values, 'o-', linewidth=2)
            ax.fill(np.linspace(0, 2 * np.pi, len(categories)), values, alpha=0.25)
            ax.set_thetagrids(np.linspace(0, 360, len(categories))[:-1], categories[:-1])
            ax.set_ylim(0, 1)
            ax.set_title("模型性能雷达图")
            ax.grid(True)
            
            # 保存图表或显示
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"雷达图已保存到 {output_file}")
            else:
                plt.show()
        except Exception as e:
            logger.error(f"绘制雷达图失败: {str(e)}")
    
    def plot_interactive_comparison(self, df: pd.DataFrame, output_file: str = None) -> None:
        """绘制交互式比较图表"""
        try:
            # 创建交互式条形图
            fig = go.Figure()
            
            # 为每个测试用例添加条形
            for index, row in df.iterrows():
                fig.add_trace(go.Bar(
                    x=["准确性", "相关性", "全面性", "综合评分"],
                    y=[row["accuracy"], row["relevance"], row["comprehensiveness"], row["overall"]],
                    name=f"测试用例 {row['test_case_id']}: {row['query']}"
                ))
            
            # 更新布局
            fig.update_layout(
                title="各测试用例评分比较",
                barmode='group',
                xaxis_title="评估指标",
                yaxis_title="评分",
                yaxis_range=[0, 1],
                height=600,
                width=1000
            )
            
            # 保存图表或显示
            if output_file:
                fig.write_html(output_file)
                logger.info(f"交互式比较图表已保存到 {output_file}")
            else:
                fig.show()
        except Exception as e:
            logger.error(f"绘制交互式比较图表失败: {str(e)}")
    
    def generate_comprehensive_report(self, df: pd.DataFrame, output_file: str = "evaluation_report.html") -> None:
        """生成综合评估报告"""
        try:
            # 创建报告
            report = f"""<html>
            <head>
                <title>AI Stocks Agent 评估报告</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1, h2 {{ color: #2c3e50; }}
                    .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                    .metric {{ display: inline-block; margin: 10px; text-align: center; }}
                    .metric-value {{ font-size: 36px; font-weight: bold; color: #3498db; }}
                    .metric-label {{ font-size: 14px; color: #7f8c8d; }}
                    .chart {{ margin: 30px 0; }}
                </style>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <h1>AI Stocks Agent 评估报告</h1>
                
                <div class="summary">
                    <h2>评估摘要</h2>
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-value">{df['overall'].mean():.2f}</div>
                            <div class="metric-label">平均综合评分</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{df['accuracy'].mean():.2f}</div>
                            <div class="metric-label">平均准确性</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{df['relevance'].mean():.2f}</div>
                            <div class="metric-label">平均相关性</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{df['comprehensiveness'].mean():.2f}</div>
                            <div class="metric-label">平均全面性</div>
                        </div>
                    </div>
                    <p>测试用例总数: {len(df)}</p>
                </div>
                
                <div class="chart">
                    <h2>各测试用例评分比较</h2>
                    <div id="comparison-chart"></div>
                </div>
                
                <div class="chart">
                    <h2>评估指标分布</h2>
                    <div id="distribution-chart"></div>
                </div>
                
                <script>
                    // 比较图表数据
                    var comparisonData = [
                        {{
                            x: ['准确性', '相关性', '全面性', '综合评分'],
                            y: [{df['accuracy'].mean():.2f}, {df['relevance'].mean():.2f}, {df['comprehensiveness'].mean():.2f}, {df['overall'].mean():.2f}],
                            type: 'bar',
                            name: '平均评分',
                            marker: {{color: '#3498db'}}
                        }}
                    ];
                    
                    // 添加每个测试用例的数据
                    {test_cases_data}
                    
                    // 绘制比较图表
                    Plotly.newPlot('comparison-chart', comparisonData, {{
                        title: '各测试用例评分比较',
                        barmode: 'group',
                        xaxis: {{title: '评估指标'}}, 
                        yaxis: {{title: '评分', range: [0, 1]}},
                        height: 600,
                        width: 1000
                    }});
                    
                    // 分布图表数据
                    var distributionData = [
                        {{
                            x: {df['accuracy'].tolist()},
                            type: 'histogram',
                            name: '准确性',
                            marker: {{color: '#e74c3c'}}
                        }},
                        {{
                            x: {df['relevance'].tolist()},
                            type: 'histogram',
                            name: '相关性',
                            marker: {{color: '#3498db'}}
                        }},
                        {{
                            x: {df['comprehensiveness'].tolist()},
                            type: 'histogram',
                            name: '全面性',
                            marker: {{color: '#2ecc71'}}
                        }},
                        {{
                            x: {df['overall'].tolist()},
                            type: 'histogram',
                            name: '综合评分',
                            marker: {{color: '#9b59b6'}}
                        }}
                    ];
                    
                    // 绘制分布图表
                    Plotly.newPlot('distribution-chart', distributionData, {{
                        title: '评估指标分布',
                        xaxis: {{title: '评分'}}, 
                        yaxis: {{title: '频数'}},
                        height: 600,
                        width: 1000,
                        barmode: 'overlay',
                        histnorm: 'probability density'
                    }});
                </script>
            </body>
            </html>"""
            
            # 生成测试用例数据
            test_cases_data = ""
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#e67e22', '#1abc9c', '#34495e']
            
            for i, row in df.iterrows():
                color_index = i % len(colors)
                # 使用字典和json.dumps来避免字符串拼接的语法错误
                test_case_dict = {
                    'x': ['准确性', '相关性', '全面性', '综合评分'],
                    'y': [round(row['accuracy'], 2), round(row['relevance'], 2), 
                          round(row['comprehensiveness'], 2), round(row['overall'], 2)],
                    'type': 'bar',
                    'name': f'测试用例 {row["test_case_id"]}',
                    'marker': {'color': colors[color_index]},
                    'visible': False
                }
                # 添加逗号和缩进
                test_cases_data += '\n                        ' + json.dumps(test_case_dict) + ','
            
            # 移除最后一个逗号
            if test_cases_data.endswith(','):
                test_cases_data = test_cases_data[:-1]
            
            # 替换报告中的占位符
            report = report.replace("{test_cases_data}", test_cases_data)
            
            # 保存报告
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"综合评估报告已生成: {output_file}")
        except Exception as e:
            logger.error(f"生成综合评估报告失败: {str(e)}")

# 示例用法
if __name__ == "__main__":
    # 初始化可视化工具
    visualizer = StocksVisualizer()
    
    # 尝试加载本地评估结果
    try:
        df = visualizer.load_evaluation_results("evaluation_results.json")
        if df is not None and len(df) > 0:
            # 绘制评分分布图
            visualizer.plot_score_distribution(df, "score_distribution.png")
            
            # 绘制雷达图
            visualizer.plot_radar_chart(df, "radar_chart.png")
            
            # 绘制交互式比较图表
            visualizer.plot_interactive_comparison(df, "interactive_comparison.html")
            
            # 生成综合评估报告
            visualizer.generate_comprehensive_report(df, "evaluation_report.html")
            
            logger.info("可视化完成！")
        else:
            # 如果没有本地结果，尝试从LangSmith获取
            logger.info("没有找到本地评估结果，尝试从LangSmith获取...")
            df = visualizer.fetch_langsmith_results("ai-stocks-agent-evaluation")
            if df is not None and len(df) > 0:
                visualizer.generate_comprehensive_report(df, "langsmith_evaluation_report.html")
                logger.info("LangSmith评估结果可视化完成！")
            else:
                logger.warning("没有找到评估结果，无法进行可视化")
    except Exception as e:
        logger.error(f"可视化过程出错: {str(e)}")