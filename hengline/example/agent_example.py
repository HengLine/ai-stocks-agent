#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
智能体使用示例
展示如何使用智能体协调器执行完整的股票分析流程
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any

from hengline.agents.agent_coordinator import AgentCoordinator
from hengline.logger import debug, info, error, warning, setup_logger

# 设置日志
setup_logger(log_file="agent_example.log")


def run_complete_analysis(stock_code: str, time_range: str = "1y") -> Dict[str, Any]:
    """
    运行完整的股票分析流程
    
    Args:
        stock_code: 股票代码
        time_range: 时间范围
        
    Returns:
        Dict[str, Any]: 分析结果
    """
    try:
        info(f"开始对股票 {stock_code} 进行完整分析")
        start_time = time.time()
        
        # 创建智能体协调器
        coordinator = AgentCoordinator()
        
        # 执行分析
        result = coordinator.analyze(stock_code, time_range)
        
        elapsed_time = time.time() - start_time
        info(f"股票 {stock_code} 分析完成，耗时: {elapsed_time:.2f}秒")
        
        # 保存结果
        save_analysis_result(result, stock_code)
        
        return result
        
    except Exception as e:
        error(f"完整分析失败: {str(e)}")
        return {"success": False, "error": str(e)}


def run_single_agent_analysis(coordinator: AgentCoordinator, agent_name: str, stock_code: str, time_range: str = "1y") -> Dict[str, Any]:
    """
    运行单个智能体分析
    
    Args:
        coordinator: 智能体协调器实例
        agent_name: 智能体名称
        stock_code: 股票代码
        time_range: 时间范围
        
    Returns:
        Dict[str, Any]: 分析结果
    """
    try:
        info(f"使用 {agent_name} 对股票 {stock_code} 进行分析")
        start_time = time.time()
        
        # 执行单个智能体分析
        result = coordinator.execute_single_agent(agent_name, stock_code, time_range)
        
        elapsed_time = time.time() - start_time
        info(f"{agent_name} 分析完成，耗时: {elapsed_time:.2f}秒")
        
        return {
            "agent_name": agent_name,
            "success": result.success,
            "result": result.result,
            "confidence_score": result.confidence_score,
            "elapsed_time": elapsed_time
        }
        
    except Exception as e:
        error(f"单个智能体分析失败: {str(e)}")
        return {"success": False, "error": str(e)}


def save_analysis_result(result: Dict[str, Any], stock_code: str):
    """
    保存分析结果到文件
    
    Args:
        result: 分析结果
        stock_code: 股票代码
    """
    try:
        # 创建保存目录
        output_dir = os.path.join("..", "..", "data", "output", stock_code)
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"analysis_result_{timestamp}.json"
        file_path = os.path.join(output_dir, file_name)
        
        # 保存结果
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        
        info(f"分析结果已保存到: {file_path}")
        
    except Exception as e:
        error(f"保存分析结果失败: {str(e)}")


def print_analysis_summary(result: Dict[str, Any]):
    """
    打印分析结果摘要
    
    Args:
        result: 分析结果
    """
    try:
        print("\n======== 分析结果摘要 ========")
        
        if not result.get("success", False):
            print(f"分析失败: {result.get('error', '未知错误')}")
            return
        
        # 打印基本信息
        print(f"股票代码: {result.get('stock_code', 'N/A')}")
        print(f"分析开始时间: {result.get('analysis_start_time', 'N/A')}")
        print(f"分析完成时间: {result.get('analysis_time', 'N/A')}")
        print(f"分析耗时: {result.get('elapsed_time_seconds', 0):.2f}秒")
        
        # 打印最终建议
        final_recommendation = result.get("final_recommendation", {})
        print("\n===== 最终投资建议 =====")
        print(f"推荐类型: {final_recommendation.get('investment_recommendation', 'N/A')}")
        print(f"风险等级: {final_recommendation.get('risk_level', 'N/A')}")
        print(f"综合评分: {final_recommendation.get('overall_score', 0)}/100")
        
        # 打印关键因素
        key_factors = final_recommendation.get('key_factors', [])
        if key_factors:
            print("\n===== 关键影响因素 =====")
            for i, factor in enumerate(key_factors, 1):
                print(f"{i}. {factor}")
        
        # 打印各智能体状态
        print("\n===== 智能体执行状态 =====")
        agent_status = result.get("agent_execution_status", {})
        for agent_name, status in agent_status.items():
            status_str = "成功" if status.get("success", False) else "失败"
            confidence = status.get("confidence_score", 0)
            print(f"{agent_name}: {status_str} (置信度: {confidence:.2f})")
        
        print("\n========================\n")
        
    except Exception as e:
        error(f"打印分析摘要失败: {str(e)}")


def compare_stocks(stock_codes: list, time_range: str = "1y") -> Dict[str, Any]:
    """
    比较多只股票的分析结果
    
    Args:
        stock_codes: 股票代码列表
        time_range: 时间范围
        
    Returns:
        Dict[str, Any]: 比较结果
    """
    try:
        info(f"开始比较 {len(stock_codes)} 只股票")
        comparison_results = {}
        
        for stock_code in stock_codes:
            # 对每只股票执行完整分析
            result = run_complete_analysis(stock_code, time_range)
            comparison_results[stock_code] = {
                "success": result.get("success", False),
                "recommendation": result.get("final_recommendation", {}).get("investment_recommendation", "N/A"),
                "score": result.get("final_recommendation", {}).get("overall_score", 0),
                "risk_level": result.get("final_recommendation", {}).get("risk_level", "N/A"),
                "elapsed_time": result.get("elapsed_time_seconds", 0)
            }
        
        # 按综合评分排序
        sorted_results = sorted(
            comparison_results.items(),
            key=lambda x: x[1].get("score", 0),
            reverse=True
        )
        
        # 打印比较结果
        print("\n======== 股票比较结果 ========")
        print("排名 | 股票代码 | 推荐类型 | 综合评分 | 风险等级 | 耗时(秒)")
        print("-----|---------|---------|---------|---------|---------")
        
        for i, (stock_code, result) in enumerate(sorted_results, 1):
            print(f"{i:4d} | {stock_code:9s} | {result['recommendation']:9s} | {result['score']:9.2f} | {result['risk_level']:9s} | {result['elapsed_time']:8.2f}")
        
        print("\n=============================\n")
        
        return comparison_results
        
    except Exception as e:
        error(f"比较股票失败: {str(e)}")
        return {"success": False, "error": str(e)}


def main():
    """
    主函数，演示智能体的使用方法
    """
    try:
        print("===== 股票分析智能体演示 =====")
        
        # 示例1: 对单只股票进行完整分析
        print("\n1. 对单只股票进行完整分析")
        stock_code = "AAPL"
        result = run_complete_analysis(stock_code, "1y")
        print_analysis_summary(result)
        
        # 示例2: 使用单个智能体
        print("\n2. 使用单个智能体进行分析")
        coordinator = AgentCoordinator()
        single_result = run_single_agent_analysis(coordinator, "FundamentalAgent", stock_code, "1y")
        print(f"{single_result['agent_name']} 分析结果: {single_result['success']}")
        print(f"置信度: {single_result['confidence_score']:.2f}")
        
        # 示例3: 比较多只股票
        print("\n3. 比较多只股票")
        stock_codes = ["AAPL", "MSFT", "GOOGL"]
        compare_stocks(stock_codes, "1y")
        
        # 示例4: 查看智能体状态
        print("\n4. 查看智能体状态")
        agent_status = coordinator.get_agent_status()
        print(f"已初始化智能体数量: {len(agent_status)}")
        for agent_name, status in agent_status.items():
            print(f"- {agent_name}: {status['description']}")
        
        print("\n演示完成!")
        
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        error(f"演示过程中发生错误: {str(e)}")
        print(f"错误: {str(e)}")


if __name__ == "__main__":
    main()