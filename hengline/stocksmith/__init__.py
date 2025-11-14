"""LangSmith集成模块，提供评估与可视化功能"""

import os
import sys
from typing import Dict, Any, Optional
from utils.logger import get_logger

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 创建日志实例
logger = get_logger('stocksmith')

# 尝试导入实际的langsmith库
try:
    import langsmith
    logger.info("成功导入langsmith库")
except ImportError:
    logger.warning("无法导入langsmith库，可能需要安装")

# 导入子模块
from .evaluator import StocksEvaluator, StocksEvaluationRunner
from .visualizer import StocksVisualizer

def get_manager(model_selector):
    """获取LangSmith管理器实例"""
    return LangSmithManager(model_selector)

class LangSmithManager:
    """LangSmith管理类，整合评估和可视化功能"""
    
    def __init__(self, model_selector):
        """
        初始化LangSmith管理器
        
        Args:
            model_selector: 模型选择器实例
        """
        self.model_selector = model_selector
        self.evaluation_runner = StocksEvaluationRunner(model_selector)
        self.visualizer = StocksVisualizer()
        
    def run_evaluation(self, test_cases_file: Optional[str] = None, 
                      project_name: str = "ai-stocks-agent-evaluation",
                      output_file: str = "evaluation_results.json") -> Dict[str, Any]:
        """
        运行评估
        
        Args:
            test_cases_file: 测试用例文件路径
            project_name: LangSmith项目名称
            output_file: 评估结果输出文件路径
            
        Returns:
            评估结果
        """
        logger.info("开始运行评估...")
        
        # 加载测试用例
        if test_cases_file and os.path.exists(test_cases_file):
            self.evaluation_runner.load_test_cases(test_cases_file)
        
        # 运行评估
        results = self.evaluation_runner.run_evaluation(project_name)
        
        # 保存评估结果
        if output_file and "error" not in results:
            self.evaluation_runner.save_evaluation_results(results, output_file)
        
        logger.info("评估运行完成")
        return results
    
    def generate_visualizations(self, results_file: str = "evaluation_results.json", 
                               output_dir: str = "./data/visualizations") -> None:
        """
        生成可视化结果
        
        Args:
            results_file: 评估结果文件路径
            output_dir: 可视化输出目录
        """
        logger.info("开始生成可视化结果...")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载评估结果
        df = self.visualizer.load_evaluation_results(results_file)
        
        if df is not None and len(df) > 0:
            # 绘制评分分布图
            self.visualizer.plot_score_distribution(
                df, 
                os.path.join(output_dir, "score_distribution.png")
            )
            
            # 绘制雷达图
            self.visualizer.plot_radar_chart(
                df, 
                os.path.join(output_dir, "radar_chart.png")
            )
            
            # 绘制交互式比较图表
            self.visualizer.plot_interactive_comparison(
                df, 
                os.path.join(output_dir, "interactive_comparison.html")
            )
            
            # 生成综合评估报告
            self.visualizer.generate_comprehensive_report(
                df, 
                os.path.join(output_dir, "evaluation_report.html")
            )
            
            logger.info(f"可视化结果已生成到 {output_dir}")
        else:
            logger.warning("没有找到有效的评估结果，无法生成可视化")
    
    def run_full_analysis(self, test_cases_file: Optional[str] = None, 
                         project_name: str = "ai-stocks-agent-evaluation",
                         output_dir: str = "./data/visualizations") -> Dict[str, Any]:
        """
        运行完整的评估和可视化分析
        
        Args:
            test_cases_file: 测试用例文件路径
            project_name: LangSmith项目名称
            output_dir: 可视化输出目录
            
        Returns:
            评估结果
        """
        # 运行评估
        results = self.run_evaluation(
            test_cases_file=test_cases_file,
            project_name=project_name,
            output_file=os.path.join(output_dir, "evaluation_results.json") if output_dir else "evaluation_results.json"
        )
        
        # 生成可视化
        if "error" not in results:
            self.generate_visualizations(
                results_file=os.path.join(output_dir, "evaluation_results.json") if output_dir else "evaluation_results.json",
                output_dir=output_dir
            )
        
        return results
    
    def is_langsmith_configured(self) -> bool:
        """检查LangSmith是否已配置"""
        return self.visualizer.client is not None and hasattr(self.visualizer.client, "api_key") and self.visualizer.client.api_key is not None

# 导出主要类和函数
export = {
    "StocksEvaluator": StocksEvaluator,
    "StocksEvaluationRunner": StocksEvaluationRunner,
    "StocksVisualizer": StocksVisualizer,
    "LangSmithManager": LangSmithManager
}

# 方便导入
def get_manager(model_selector):
    """获取LangSmith管理器实例"""
    return LangSmithManager(model_selector)