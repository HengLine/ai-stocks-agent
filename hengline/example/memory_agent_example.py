#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
智能体记忆功能使用示例
演示如何使用带有VectorStoreRetrieverMemory的智能体进行分析
"""

import os
import sys
import json
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from hengline.agents.agent_coordinator import AgentCoordinator
from hengline.agents.base_agent import AgentConfig
from hengline.logger import logger


def demonstrate_memory_feature():
    """
    演示智能体的记忆功能
    """
    print("="*80)
    print("智能体记忆功能演示")
    print("="*80)
    
    # 1. 创建启用记忆功能的配置
    config = {
        "global": {
            "model_name": "gpt-4",
            "enable_memory": True  # 启用记忆功能
        },
        "agents": {
            "TechnicalAgent": {
                "memory_top_k": 4  # 为技术分析智能体设置更多的记忆检索数量
            }
        }
    }
    
    # 2. 初始化协调器
    print("\n1. 初始化智能体协调器（启用记忆功能）...")
    coordinator = AgentCoordinator(config)
    
    # 3. 执行第一次分析
    stock_code = "AAPL"
    print(f"\n2. 对股票 {stock_code} 执行第一次技术分析...")
    result1 = coordinator.execute_single_agent("TechnicalAgent", stock_code, "1y")
    
    print(f"\n   分析结果概要:")
    print(f"   - 成功状态: {result1.success}")
    if result1.success and "key_findings" in result1.result:
        print(f"   - 关键发现: {result1.result['key_findings'][:3]}...")
    
    # 4. 执行第二次相关分析，应该会利用第一次的记忆
    print(f"\n3. 对同一股票 {stock_code} 执行第二次技术分析（应利用记忆）...")
    print("   这次分析会参考第一次分析的结果，保持上下文连贯性")
    result2 = coordinator.execute_single_agent("TechnicalAgent", stock_code, "1y", use_memory=True)
    
    print(f"\n   第二次分析结果概要:")
    print(f"   - 成功状态: {result2.success}")
    if result2.success and "key_findings" in result2.result:
        print(f"   - 关键发现: {result2.result['key_findings'][:3]}...")
    
    # 5. 对不同股票执行分析，但仍然使用记忆
    another_stock = "MSFT"
    print(f"\n4. 对不同股票 {another_stock} 执行分析（仍然使用记忆）...")
    print("   这次分析会参考之前对AAPL的分析，进行对比和关联")
    result3 = coordinator.execute_single_agent("TechnicalAgent", another_stock, "1y")
    
    print(f"\n   跨股票分析结果概要:")
    print(f"   - 成功状态: {result3.success}")
    if result3.success and "key_findings" in result3.result:
        print(f"   - 关键发现: {result3.result['key_findings'][:3]}...")
    
    # 6. 演示临时禁用记忆
    print(f"\n5. 对股票 {stock_code} 执行分析（临时禁用记忆）...")
    result4 = coordinator.execute_single_agent("TechnicalAgent", stock_code, "1y", use_memory=False)
    
    print(f"\n   无记忆分析结果概要:")
    print(f"   - 成功状态: {result4.success}")
    
    # 7. 演示更新智能体配置
    print("\n6. 更新智能体配置，调整记忆参数...")
    coordinator.update_agent_config("TechnicalAgent", {"memory_top_k": 5})
    
    # 8. 再次执行分析，使用更新后的配置
    print(f"\n7. 使用更新后的配置再次分析 {stock_code}...")
    result5 = coordinator.execute_single_agent("TechnicalAgent", stock_code, "1y")
    
    print(f"\n   更新后分析结果概要:")
    print(f"   - 成功状态: {result5.success}")
    
    print("\n" + "="*80)
    print("记忆功能演示完成！")
    print("智能体现在能够:")
    print("1. 记住之前的分析结果")
    print("2. 在后续分析中利用这些记忆")
    print("3. 跨股票分析时进行对比和关联")
    print("4. 灵活启用/禁用记忆功能")
    print("5. 动态调整记忆参数")
    print("="*80)


def demonstrate_multi_agent_memory():
    """
    演示多个智能体之间的记忆协作
    """
    print("\n"*2 + "="*80)
    print("多智能体记忆协作演示")
    print("="*80)
    
    # 创建配置
    config = {
        "global": {
            "model_name": "gpt-4",
            "enable_memory": True
        }
    }
    
    coordinator = AgentCoordinator(config)
    stock_code = "TSLA"
    
    print(f"\n1. 首先使用基本面智能体分析 {stock_code}...")
    fundamental_result = coordinator.execute_single_agent("FundamentalAgent", stock_code, "1y")
    
    print(f"\n2. 然后使用技术面智能体分析同一股票...")
    print("   技术面智能体将尝试基于基本面分析的记忆进行更全面的分析")
    technical_result = coordinator.execute_single_agent("TechnicalAgent", stock_code, "1y")
    
    print(f"\n3. 最后使用首席策略官整合所有分析...")
    print("   首席策略官拥有更多的记忆容量，可以整合多个智能体的历史分析")
    strategy_result = coordinator.execute_single_agent("ChiefStrategyAgent", stock_code, "1y")
    
    print(f"\n多智能体协作完成！")
    print("每个智能体都可以利用自己的历史记忆，为最终的投资决策提供更连贯的分析。")


def main():
    """
    主函数
    """
    try:
        # 演示基本记忆功能
        demonstrate_memory_feature()
        
        # 演示多智能体记忆协作
        demonstrate_multi_agent_memory()
        
    except Exception as e:
        print(f"\n演示过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()