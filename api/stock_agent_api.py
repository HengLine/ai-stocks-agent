#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票智能体API接口
提供HTTP接口以便前端调用智能体分析功能
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import APIRouter
from fastapi import HTTPException, Query, Depends, Body
from pydantic import BaseModel, Field

from config.config import get_data_output_path
from hengline.agents.agent_coordinator import AgentCoordinator
from hengline.logger import debug, info, error

# 创建FastAPI应用
app = APIRouter()

# 全局智能体协调器实例
coordinator_instance = None


def get_coordinator() -> AgentCoordinator:
    """
    获取智能体协调器实例（单例模式）
    
    Returns:
        AgentCoordinator: 智能体协调器实例
    """
    global coordinator_instance
    if coordinator_instance is None:
        coordinator_instance = AgentCoordinator()
    return coordinator_instance


# 请求和响应模型
class StockAnalysisRequest(BaseModel):
    """
    股票分析请求模型
    """
    stock_code: str = Field(..., description="股票代码", example="AAPL")
    time_range: str = Field(default="1y", description="时间范围", example="1y")
    agent_params: Optional[Dict[str, Any]] = Field(default={}, description="智能体参数")
    chief_params: Optional[Dict[str, Any]] = Field(default={}, description="首席策略官参数")


class SingleAgentRequest(BaseModel):
    """
    单个智能体分析请求模型
    """
    agent_name: str = Field(..., description="智能体名称", example="FundamentalAgent")
    stock_code: str = Field(..., description="股票代码", example="AAPL")
    time_range: str = Field(default="1y", description="时间范围", example="1y")
    params: Optional[Dict[str, Any]] = Field(default={}, description="智能体参数")


class CompareStocksRequest(BaseModel):
    """
    比较股票请求模型
    """
    stock_codes: List[str] = Field(..., description="股票代码列表", example=["AAPL", "MSFT", "GOOGL"])
    time_range: str = Field(default="1y", description="时间范围", example="1y")


class ErrorResponse(BaseModel):
    """
    错误响应模型
    """
    success: bool = Field(default=False)
    error: str = Field(..., description="错误信息")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class SuccessResponse(BaseModel):
    """
    成功响应模型
    """
    success: bool = Field(default=True)
    message: str = Field(..., description="成功信息")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# API端点
@app.post("/analyze", response_model=Dict[str, Any], summary="执行股票综合分析")
async def analyze_stock(request: StockAnalysisRequest, coordinator: AgentCoordinator = Depends(get_coordinator)) -> Dict[str, Any]:
    """
    对指定股票执行完整的综合分析
    
    - **stock_code**: 股票代码，例如 "AAPL"
    - **time_range**: 分析的时间范围，可选值: "1d", "1w", "1m", "3m", "6m", "1y", "2y", "5y", "10y", "max"
    - **agent_params**: 传递给各专业智能体的参数
    - **chief_params**: 传递给首席策略官的参数
    
    返回包含所有智能体分析结果和最终建议的综合报告
    """
    try:
        info(f"接收到股票分析请求: {request.stock_code}, 时间范围: {request.time_range}")

        # 执行分析
        result = coordinator.analyze(
            stock_code=request.stock_code,
            time_range=request.time_range,
            agent_params=request.agent_params,
            chief_params=request.chief_params
        )

        # 保存结果到文件
        save_analysis_result(result, request.stock_code)

        return result

    except Exception as e:
        error(f"股票分析请求处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/agent", response_model=Dict[str, Any], summary="执行单个智能体分析")
async def analyze_with_single_agent(request: SingleAgentRequest, coordinator: AgentCoordinator = Depends(get_coordinator)) -> Dict[str, Any]:
    """
    使用指定的单个智能体进行股票分析
    
    - **agent_name**: 智能体名称，例如 "FundamentalAgent", "TechnicalAgent"等
    - **stock_code**: 股票代码，例如 "AAPL"
    - **time_range**: 分析的时间范围
    - **params**: 传递给智能体的参数
    
    返回单个智能体的分析结果
    """
    try:
        info(f"接收到单个智能体分析请求: {request.agent_name} 分析 {request.stock_code}")

        # 执行单个智能体分析
        result = coordinator.execute_single_agent(
            agent_name=request.agent_name,
            stock_code=request.stock_code,
            time_range=request.time_range,
            **request.params
        )

        return {
            "success": result.success,
            "agent_name": result.agent_name,
            "result": result.result,
            "confidence_score": result.confidence_score,
            "error_message": result.error_message
        }

    except Exception as e:
        error(f"单个智能体分析请求处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", response_model=Dict[str, Any], summary="比较多只股票")
async def compare_stocks(request: CompareStocksRequest, coordinator: AgentCoordinator = Depends(get_coordinator)) -> Dict[str, Any]:
    """
    比较多只股票的分析结果
    
    - **stock_codes**: 股票代码列表，例如 ["AAPL", "MSFT", "GOOGL"]
    - **time_range**: 分析的时间范围
    
    返回多只股票的分析结果比较
    """
    try:
        info(f"接收到股票比较请求: {len(request.stock_codes)} 只股票")

        comparison_results = {}

        # 对每只股票执行分析
        for stock_code in request.stock_codes:
            result = coordinator.analyze(stock_code, request.time_range)
            comparison_results[stock_code] = {
                "success": result.get("success", False),
                "recommendation": result.get("final_recommendation", {}).get("investment_recommendation", "N/A"),
                "score": result.get("final_recommendation", {}).get("overall_score", 0),
                "risk_level": result.get("final_recommendation", {}).get("risk_level", "N/A"),
                "elapsed_time": result.get("elapsed_time_seconds", 0)
            }

            # 保存每只股票的分析结果
            save_analysis_result(result, stock_code)

        # 按综合评分排序
        sorted_results = sorted(
            comparison_results.items(),
            key=lambda x: x[1].get("score", 0),
            reverse=True
        )

        return {
            "success": True,
            "comparison_results": comparison_results,
            "sorted_results": [{
                "stock_code": stock_code,
                "data": result
            } for stock_code, result in sorted_results],
            "total_stocks": len(request.stock_codes),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        error(f"股票比较请求处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents", response_model=Dict[str, Any], summary="获取智能体列表")
async def get_agents(coordinator: AgentCoordinator = Depends(get_coordinator)) -> Dict[str, Any]:
    """
    获取所有可用智能体的信息
    
    返回系统中所有已初始化的智能体及其描述
    """
    try:
        info("接收到获取智能体列表请求")

        agent_status = coordinator.get_agent_status()

        return {
            "success": True,
            "agents": agent_status,
            "total_agents": len(agent_status),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        error(f"获取智能体列表请求处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/config", response_model=Dict[str, Any], summary="更新智能体配置")
async def update_agent_config(
        agent_name: str = Query(..., description="智能体名称"),
        config: Dict[str, Any] = Body(..., description="新配置"),
        coordinator: AgentCoordinator = Depends(get_coordinator)
) -> Dict[str, Any]:
    """
    更新指定智能体的配置
    
    - **agent_name**: 智能体名称
    - **config**: 新的配置参数
    
    返回配置更新的结果
    """
    try:
        info(f"接收到智能体配置更新请求: {agent_name}")

        # 更新智能体配置
        coordinator.update_agent_config(agent_name, config)

        return SuccessResponse(
            message=f"智能体 {agent_name} 配置更新成功"
        )

    except ValueError as e:
        error(f"智能体配置更新失败: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        error(f"智能体配置更新失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def save_analysis_result(result: Dict[str, Any], stock_code: str):
    """
    保存分析结果到文件
    
    Args:
        result: 分析结果
        stock_code: 股票代码
    """
    try:
        # 创建保存目录
        output_dir = os.path.join(get_data_output_path(), stock_code)
        os.makedirs(output_dir, exist_ok=True)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"analysis_result_{timestamp}.json"
        file_path = os.path.join(output_dir, file_name)

        # 保存结果
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        debug(f"分析结果已保存到: {file_path}")

    except Exception as e:
        error(f"保存分析结果失败: {str(e)}")


@app.get("/health", response_model=Dict[str, Any], summary="健康检查")
async def health_check() -> Dict[str, Any]:
    """
    健康检查端点

    返回API服务的健康状态
    """
    try:
        # 测试智能体协调器
        coordinator = get_coordinator()
        agent_status = coordinator.get_agent_status()

        return {
            "success": True,
            "status": "healthy",
            "total_agents": len(agent_status),
            "timestamp": datetime.now().isoformat(),
            "service": "stock-agent-api"
        }

    except Exception as e:
        error(f"健康检查失败: {str(e)}")
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
