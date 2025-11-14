#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票智能体API调用示例
本示例展示了如何调用stock_agent_api.py中定义的所有API接口
"""

import json
import time
import requests
from datetime import datetime
from typing import Dict, Any, List

# API基础URL
BASE_URL = "http://localhost:8000"


class StockAgentAPIClient:
    """股票智能体API客户端"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json"
        }
    
    def _print_response(self, endpoint: str, response: requests.Response):
        """打印API响应信息"""
        print(f"\n=== {endpoint} ===")
        print(f"状态码: {response.status_code}")
        try:
            response_json = response.json()
            print(f"响应内容: {json.dumps(response_json, ensure_ascii=False, indent=2)}")
        except json.JSONDecodeError:
            print(f"响应内容: {response.text}")
        print("=" * 50)
    
    def health_check(self):
        """健康检查接口示例"""
        endpoint = f"{self.base_url}/health"
        response = requests.get(endpoint, headers=self.headers)
        self._print_response("健康检查", response)
        return response.json()
    
    def get_agents(self):
        """获取智能体列表接口示例"""
        endpoint = f"{self.base_url}/agents"
        response = requests.get(endpoint, headers=self.headers)
        self._print_response("获取智能体列表", response)
        return response.json()
    
    def analyze_stock(self, stock_code: str, time_range: str = "1y", 
                      agent_params: Dict[str, Any] = None, 
                      chief_params: Dict[str, Any] = None):
        """执行股票综合分析接口示例"""
        endpoint = f"{self.base_url}/analyze"
        
        data = {
            "stock_code": stock_code,
            "time_range": time_range,
            "agent_params": agent_params or {},
            "chief_params": chief_params or {}
        }
        
        start_time = time.time()
        response = requests.post(endpoint, json=data, headers=self.headers)
        elapsed_time = time.time() - start_time
        
        self._print_response(f"股票综合分析 - {stock_code}", response)
        print(f"分析耗时: {elapsed_time:.2f}秒")
        
        if response.status_code == 200:
            return response.json()
        return None
    
    def analyze_with_single_agent(self, agent_name: str, stock_code: str, 
                                 time_range: str = "1y", 
                                 params: Dict[str, Any] = None):
        """执行单个智能体分析接口示例"""
        endpoint = f"{self.base_url}/analyze/agent"
        
        data = {
            "agent_name": agent_name,
            "stock_code": stock_code,
            "time_range": time_range,
            "params": params or {}
        }
        
        response = requests.post(endpoint, json=data, headers=self.headers)
        self._print_response(f"单个智能体分析 - {agent_name} 分析 {stock_code}", response)
        
        if response.status_code == 200:
            return response.json()
        return None
    
    def compare_stocks(self, stock_codes: List[str], time_range: str = "1y"):
        """比较多只股票接口示例"""
        endpoint = f"{self.base_url}/compare"
        
        data = {
            "stock_codes": stock_codes,
            "time_range": time_range
        }
        
        start_time = time.time()
        response = requests.post(endpoint, json=data, headers=self.headers)
        elapsed_time = time.time() - start_time
        
        self._print_response(f"股票比较 - {len(stock_codes)}只股票", response)
        print(f"比较耗时: {elapsed_time:.2f}秒")
        
        if response.status_code == 200:
            return response.json()
        return None
    
    def update_agent_config(self, agent_name: str, config: Dict[str, Any]):
        """更新智能体配置接口示例"""
        endpoint = f"{self.base_url}/agent/config"
        
        # 注意：这里使用params传递agent_name，使用json传递config
        params = {
            "agent_name": agent_name
        }
        
        response = requests.post(endpoint, params=params, json=config, headers=self.headers)
        self._print_response(f"更新智能体配置 - {agent_name}", response)
        
        if response.status_code == 200:
            return response.json()
        return None


def run_demo():
    """运行API调用示例演示"""
    print("=" * 50)
    print("股票智能体API调用示例")
    print("=" * 50)
    
    client = StockAgentAPIClient()
    
    try:
        # 1. 健康检查
        print("\n1. 执行健康检查...")
        health_response = client.health_check()
        if not health_response.get("success"):
            print("警告: 服务健康检查失败，可能无法正常使用所有功能")
        
        # 2. 获取智能体列表
        print("\n2. 获取智能体列表...")
        agents_response = client.get_agents()
        
        # 3. 执行单个智能体分析
        print("\n3. 执行单个智能体分析...")
        # 使用基本面分析智能体
        single_agent_result = client.analyze_with_single_agent(
            agent_name="FundamentalAgent",
            stock_code="AAPL",
            time_range="1y",
            params={"detail_level": "medium"}
        )
        
        # 4. 执行股票综合分析
        print("\n4. 执行股票综合分析...")
        # 注意：这个操作可能需要较长时间
        analyze_result = client.analyze_stock(
            stock_code="AAPL",
            time_range="1y",
            agent_params={"detail_level": "high"},
            chief_params={"report_format": "comprehensive"}
        )
        
        # 5. 比较多只股票
        print("\n5. 比较多只股票...")
        # 注意：这个操作可能需要很长时间
        # 为了演示，我们只比较两只股票
        compare_result = client.compare_stocks(
            stock_codes=["AAPL", "MSFT"],
            time_range="1y"
        )
        
        # 6. 更新智能体配置（可选，谨慎操作）
        print("\n6. 更新智能体配置（演示，不实际执行）...")
        # 以下代码仅作演示，实际运行可能会影响智能体行为
        # 如需测试，请取消注释
        
        # config_update_result = client.update_agent_config(
        #     agent_name="TechnicalAgent",
        #     config={
        #         "analysis_periods": ["1d", "1w", "1m"],
        #         "include_chart_data": True,
        #         "indicators": ["MA", "RSI", "MACD"]
        #     }
        # )
        
        print("\nAPI调用示例执行完成!")
        
    except Exception as e:
        print(f"执行示例时发生错误: {str(e)}")
        print("请确保API服务正在运行，并检查BASE_URL是否正确设置")


def run_selective_demo():
    """运行选择性API调用示例（更快的演示）"""
    print("=" * 50)
    print("股票智能体API快速调用示例")
    print("=" * 50)
    
    client = StockAgentAPIClient()
    
    try:
        # 只执行基础操作
        print("\n1. 执行健康检查...")
        client.health_check()
        
        print("\n2. 获取智能体列表...")
        agents = client.get_agents()
        
        # 如果有智能体，执行单个智能体分析
        if agents.get("success") and agents.get("agents"):
            agent_names = list(agents["agents"].keys())
            print(f"\n发现 {len(agent_names)} 个智能体: {', '.join(agent_names)}")
            
            if agent_names:
                print(f"\n3. 使用第一个智能体 {agent_names[0]} 分析示例股票...")
                client.analyze_with_single_agent(
                    agent_name=agent_names[0],
                    stock_code="AAPL",
                    time_range="1m"
                )
        
        print("\n快速API调用示例执行完成!")
        
    except Exception as e:
        print(f"执行示例时发生错误: {str(e)}")
        print("请确保API服务正在运行在 http://localhost:8000")


if __name__ == "__main__":
    print("请选择要运行的演示类型:")
    print("1. 完整演示（包含所有API调用，可能需要较长时间）")
    print("2. 快速演示（仅执行基础API调用）")
    
    try:
        choice = input("请输入选项 (1/2，默认2): ").strip()
        if choice == "1":
            run_demo()
        else:
            run_selective_demo()
    except KeyboardInterrupt:
        print("\n演示已取消")
    except Exception as e:
        print(f"选择时发生错误: {str(e)}")
        # 默认运行快速演示
        run_selective_demo()