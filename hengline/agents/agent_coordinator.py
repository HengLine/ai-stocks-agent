#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@FileName: agent_coordinator.py
@Description: 智能体协调器，负责协调各专业智能体的工作流程，实现并行推理和结果整合
@Author: HengLine
@Time: 2025/11/10
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, TypedDict, Annotated, Callable

import nest_asyncio
from langgraph.graph import StateGraph, END

from config.config import get_ai_config, get_embedding_config
from hengline.agents.base_agent import AgentResult, AgentConfig
from hengline.agents.chief_strategy_agent import ChiefStrategyAgent
from hengline.agents.esg_risk_agent import ESGRiskAgent
from hengline.agents.fund_flow_agent import FundFlowAgent
from hengline.agents.fundamental_agent import FundamentalAgent
from hengline.agents.industry_macro_agent import IndustryMacroAgent
from hengline.agents.sentiment_agent import SentimentAgent
from hengline.agents.technical_agent import TechnicalAgent
from hengline.logger import debug, info, error, warning, performance_logger
from utils.log_utils import print_log_exception

# 应用nest_asyncio以支持嵌套事件循环
nest_asyncio.apply()


def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """
    自定义字典合并函数，用于处理agent_results的并行更新
    """
    merged = left.copy() if left else {}
    merged.update(right)
    return merged

class AgentState(TypedDict):
    """
    定义LangGraph的状态类型
    使用自定义合并函数处理并行更新
    """
    stock_code: str
    time_range: str
    agent_params: Dict[str, Any]
    chief_params: Dict[str, Any]
    agent_results: Annotated[Dict[str, Any], merge_dicts]
    final_result: Optional[Dict[str, Any]] = None


class AgentCoordinator:
    """
    智能体协调器，管理多智能体工作流和并行推理
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化智能体协调器
        
        Args:
            config: 协调器配置
        """
        self.config = config or {}
        self.agents = {}
        self.graph = None
        self.initialize_agents()
        self.build_workflow()

    def initialize_agents(self):
        """
        初始化所有专业智能体
        """
        try:
            debug("开始初始化专业智能体")

            # 获取全局配置
            llm_config = get_ai_config()
            embedding_config = get_embedding_config()
            model_name = llm_config.get("model_name", "gpt-4")
            model_type = llm_config.get("provider", "openai").lower()
            enable_memory = embedding_config.get("enable_memory", True)

            # 初始化各专业智能体的配置
            agent_configs = {
                "FundamentalAgent": AgentConfig(
                    agent_name="基本面分析智能体",
                    description="专业的股票基本面分析师，擅长分析公司财务状况、盈利能力和估值水平",
                    model_name=model_name,
                    model_type=model_type,
                    enable_memory=enable_memory,
                    llm_config=llm_config,
                    embedding_config=embedding_config
                ),
                "TechnicalAgent": AgentConfig(
                    agent_name="技术面分析智能体",
                    description="专业的股票技术分析师，擅长分析价格走势、交易量和技术指标",
                    model_name=model_name,
                    model_type=model_type,
                    enable_memory=enable_memory,
                    llm_config=llm_config,
                    embedding_config=embedding_config
                ),
                "IndustryMacroAgent": AgentConfig(
                    agent_name="行业宏观分析智能体",
                    description="专业的行业和宏观经济分析师，擅长分析行业趋势和宏观经济环境",
                    model_name=model_name,
                    model_type=model_type,
                    enable_memory=enable_memory,
                    llm_config=llm_config,
                    embedding_config=embedding_config
                ),
                "SentimentAgent": AgentConfig(
                    agent_name="舆情情绪分析智能体",
                    description="专业的市场情绪分析师，擅长分析新闻舆情和市场情绪",
                    model_name=model_name,
                    model_type=model_type,
                    enable_memory=enable_memory,
                    llm_config=llm_config,
                    embedding_config=embedding_config
                ),
                "FundFlowAgent": AgentConfig(
                    agent_name="资金流分析智能体",
                    description="专业的资金流向分析师，擅长分析机构持仓和资金流动",
                    model_name=model_name,
                    model_type=model_type,
                    enable_memory=enable_memory,
                    llm_config=llm_config,
                    embedding_config=embedding_config
                ),
                "ESGRiskAgent": AgentConfig(
                    agent_name="ESG风险分析智能体",
                    description="专业的ESG和公司治理分析师，擅长评估企业可持续发展风险",
                    model_name=model_name,
                    model_type=model_type,
                    enable_memory=enable_memory,
                    llm_config=llm_config,
                    embedding_config=embedding_config
                ),
                "ChiefStrategyAgent": AgentConfig(
                    agent_name="首席策略官智能体",
                    description="资深投资策略专家，负责整合各方面分析并提供最终投资建议",
                    model_name=model_name,
                    model_type=model_type,
                    enable_memory=enable_memory,
                    memory_top_k=5,  # 策略官需要更多的历史记忆
                    llm_config=llm_config,
                    embedding_config=embedding_config
                )
            }

            # 应用每个智能体的特定配置
            for agent_name, specific_config in self.config.get("agents", {}).items():
                if agent_name in agent_configs:
                    agent_configs[agent_name] = agent_configs[agent_name].model_copy(update=specific_config)

            # 初始化各专业智能体
            self.agents = {
                "FundamentalAgent": FundamentalAgent(agent_configs["FundamentalAgent"]),
                "TechnicalAgent": TechnicalAgent(agent_configs["TechnicalAgent"]),
                "IndustryMacroAgent": IndustryMacroAgent(agent_configs["IndustryMacroAgent"]),
                "SentimentAgent": SentimentAgent(agent_configs["SentimentAgent"]),
                "FundFlowAgent": FundFlowAgent(agent_configs["FundFlowAgent"]),
                "ESGRiskAgent": ESGRiskAgent(agent_configs["ESGRiskAgent"]),
                "ChiefStrategyAgent": ChiefStrategyAgent(agent_configs["ChiefStrategyAgent"])
            }

            # 记录初始化的智能体数量
            info(f"成功初始化 {len(self.agents)} 个智能体")
            info(f"记忆功能: {'已启用' if enable_memory else '已禁用'}")

        except Exception as e:
            error(f"初始化智能体失败: {str(e)}")
            raise

    def build_workflow(self):
        """
        构建智能体工作流程
        """
        try:
            debug("开始构建智能体工作流")

            # 创建LangGraph工作流，指定状态类型
            self.graph = StateGraph(AgentState)

            # 添加节点：并行执行各专业智能体
            for agent_name, agent in self.agents.items():
                if agent_name != "ChiefStrategyAgent":  # 策略官单独处理
                    self.graph.add_node(agent_name, self._create_agent_node(agent_name))

            # 添加首席策略官节点
            self.graph.add_node("ChiefStrategyAgent", self._create_chief_node())

            # 设置初始节点（所有专业智能体并行执行）
            entry_points = [name for name in self.agents.keys() if name != "ChiefStrategyAgent"]
            for entry_point in entry_points:
                self.graph.add_edge("__start__", entry_point)

            # 设置边：所有专业智能体完成后执行首席策略官
            for agent_name in entry_points:
                self.graph.add_edge(agent_name, "ChiefStrategyAgent")

            # 设置结束节点
            self.graph.add_edge("ChiefStrategyAgent", END)

            # 编译图
            self.graph = self.graph.compile()

            debug("智能体工作流构建完成")

        except Exception as e:
            error(f"构建工作流失败: {str(e)}")
            raise

    def _create_agent_node(self, agent_name: str):
        """
        创建智能体节点函数
        
        Args:
            agent_name: 智能体名称
            
        Returns:
            function: 节点函数
        """

        async def agent_node(state: AgentState) -> Dict[str, Any]:
            """
            智能体节点执行函数
            
            Args:
                state: 当前状态
                
            Returns:
                Dict[str, Any]: 需要更新的状态部分
            """
            stock_code = state.get("stock_code", "")
            time_range = state.get("time_range", "1y")
            agent_params = state.get("agent_params", {})

            @performance_logger(f"执行 {agent_name} 分析")
            async def run_agent():
                try:
                    agent = self.agents[agent_name]
                    # 同步智能体转换为异步执行
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        agent.analyze,
                        stock_code,
                        time_range,
                        **agent_params
                    )
                    return result
                except Exception as e:
                    error(f"{agent_name} 执行失败: {str(e)}")
                    return AgentResult(
                        agent_name=agent_name,
                        success=False,
                        result={},
                        error_message=str(e),
                        confidence_score=0.0
                    )

            # 执行智能体
            result = await run_agent()

            # 返回需要更新的状态部分
            debug(f"{agent_name} 分析完成，状态: {'成功' if result.success else '失败'}")
            # 使用add_messages合并策略，只返回当前智能体的结果
            return {"agent_results": {agent_name: result}}

        return agent_node

    def _create_chief_node(self):
        """
        创建首席策略官节点函数
        
        Returns:
            function: 节点函数
        """

        async def chief_node(state: AgentState) -> Dict[str, Any]:
            """
            首席策略官节点执行函数
            
            Args:
                state: 当前状态
                
            Returns:
                Dict[str, Any]: 需要更新的状态部分
            """
            stock_code = state.get("stock_code", "")
            agent_results = state.get("agent_results", {})

            @performance_logger("执行首席策略官分析")
            async def run_chief():
                try:
                    chief_agent = self.agents["ChiefStrategyAgent"]
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        chief_agent.analyze,
                        stock_code,
                        agent_results
                    )
                    return result
                except Exception as e:
                    error(f"首席策略官执行失败: {str(e)}")
                    return AgentResult(
                        agent_name="ChiefStrategyAgent",
                        success=False,
                        result={},
                        error_message=str(e),
                        confidence_score=0.0
                    )

            # 执行首席策略官
            result = await run_chief()

            # 返回需要更新的状态部分
            debug(f"首席策略官分析完成，最终建议: {result.result.get('investment_recommendation', '无') if result.success else '失败'}")
            return {"final_result": result}

        return chief_node

    async def analyze_async(self, stock_code: str, time_range: str = "1y", **kwargs) -> Dict[str, Any]:
        """
        异步执行完整分析流程
        
        Args:
            stock_code: 股票代码
            time_range: 时间范围
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            debug(f"开始对股票 {stock_code} 进行综合分析")

            # 记录分析开始时间（不放入状态中）
            analysis_start_time = datetime.now().isoformat()

            # 准备初始状态，严格符合AgentState类型定义
            initial_state = AgentState(
                stock_code=stock_code,
                time_range=time_range,
                agent_params=kwargs.get("agent_params", {}),  # 移除chief_params
                chief_params=kwargs.get("chief_params", {}),    # 添加单独的chief_params字段
                agent_results={}  # 初始化必需的agent_results字段
            )

            # 执行工作流
            @performance_logger(f"执行完整分析流程 - {stock_code}")
            async def execute_workflow():
                return await self.graph.ainvoke(initial_state)

            final_state = await execute_workflow()

            # 生成分析报告
            report = self._generate_analysis_report(final_state)

            # 记录分析完成时间
            analysis_end_time = datetime.now().isoformat()
            report["analysis_time"] = analysis_end_time
            report["elapsed_time_seconds"] = (
                    datetime.fromisoformat(analysis_end_time) -
                    datetime.fromisoformat(analysis_start_time)
            ).total_seconds()

            debug(f"股票 {stock_code} 综合分析完成，耗时: {report['elapsed_time_seconds']:.2f}秒")

            return report

        except Exception as e:
            error(f"综合分析失败: {str(e)}")
            print_log_exception()
            return {
                "success": False,
                "error": str(e),
                "stock_code": stock_code,
                "analysis_time": datetime.now().isoformat()
            }

    def analyze(self, stock_code: str, time_range: str = "1y", **kwargs) -> Dict[str, Any]:
        """
        同步执行完整分析流程
        
        Args:
            stock_code: 股票代码
            time_range: 时间范围
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        # 创建事件循环并运行异步分析
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已有运行中的事件循环，使用它
                task = loop.create_task(self.analyze_async(stock_code, time_range, **kwargs))
                return loop.run_until_complete(task)
            else:
                # 创建新的事件循环
                return loop.run_until_complete(self.analyze_async(stock_code, time_range, **kwargs))
        except Exception as e:
            error(f"同步分析失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "stock_code": stock_code
            }

    def _generate_analysis_report(self, final_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成综合分析报告
        
        Args:
            final_state: 最终状态
            
        Returns:
            Dict[str, Any]: 综合报告
        """
        report = {
            "success": True,
            "stock_code": final_state.get("stock_code", ""),
            "analysis_start_time": final_state.get("analysis_start_time", ""),
            "agent_execution_status": {},
            "final_recommendation": {},
            "detailed_results": {}
        }

        # 记录各智能体执行状态
        agent_results = final_state.get("agent_results", {})
        for agent_name, result in agent_results.items():
            report["agent_execution_status"][agent_name] = {
                "success": result.success,
                "confidence_score": result.confidence_score,
                "error": result.error_message if not result.success else None
            }

            # 保存详细结果
            if result.success:
                report["detailed_results"][agent_name] = result.result

        # 添加最终建议
        final_result = final_state.get("final_result")
        if final_result and final_result.success:
            report["final_recommendation"] = final_result.result
            report["success"] = True
        else:
            report["success"] = False
            report["error"] = final_result.error_message if final_result else "首席策略官执行失败"

        return report

    def get_agent_status(self) -> Dict[str, Any]:
        """
        获取各智能体状态
        
        Returns:
            Dict[str, Any]: 智能体状态信息
        """
        status = {}
        for agent_name, agent in self.agents.items():
            status[agent_name] = {
                "name": agent.agent_name,
                "description": agent.description,
                "is_initialized": True
            }
        return status

    def update_agent_config(self, agent_name: str, config: Dict[str, Any]):
        """
        更新指定智能体的配置
        
        Args:
            agent_name: 智能体名称
            config: 新的配置参数
        """
        try:
            if agent_name in self.agents:
                agent = self.agents[agent_name]

                # 如果智能体有config属性，使用Pydantic的model_copy更新
                if hasattr(agent, 'config') and isinstance(agent.config, AgentConfig):
                    new_config = agent.config.model_copy(update=config)
                    agent.config = new_config

                    # 如果启用/禁用了记忆，需要重新初始化记忆
                    if "enable_memory" in config:
                        if config["enable_memory"] and not agent.memory:
                            agent.memory = agent._init_memory()
                            info(f"为智能体 {agent_name} 启用了记忆功能")
                        elif not config["enable_memory"] and agent.memory:
                            agent.memory = None
                            info(f"为智能体 {agent_name} 禁用了记忆功能")

                    # 如果更新了模型或嵌入模型，重新获取LLM
                    if "model_name" in config or "embedding_model" in config:
                        agent.langchain_llm = agent._get_langchain_llm()
                        if agent.memory:
                            agent.memory = agent._init_memory()

                # 如果是关键配置变更，重建工作流
                if "model_name" in config or "enable_memory" in config:
                    self.build_workflow()

                info(f"更新智能体配置成功: {agent_name}")
            else:
                warning(f"智能体不存在: {agent_name}")
        except Exception as e:
            error(f"更新智能体配置失败: {str(e)}")
            raise

    def execute_single_agent(self, agent_name: str, stock_code: str, time_range: str = "1y", **kwargs) -> AgentResult:
        """
        执行单个智能体分析
        
        Args:
            agent_name: 智能体名称
            stock_code: 股票代码
            time_range: 时间范围
            **kwargs: 额外参数
            
        Returns:
            AgentResult: 分析结果
        """
        try:
            if agent_name in self.agents:
                agent = self.agents[agent_name]

                # 可以从kwargs中获取特定的记忆相关参数
                use_memory = kwargs.pop("use_memory", agent.config.enable_memory if hasattr(agent, 'config') else True)

                # 保存原始的enable_memory设置
                original_enable_memory = None
                if hasattr(agent, 'config') and hasattr(agent, 'memory'):
                    original_enable_memory = agent.config.enable_memory
                    agent.config.enable_memory = use_memory

                    # 如果临时启用记忆但当前没有，初始化记忆
                    if use_memory and not agent.memory:
                        agent.memory = agent._init_memory()

                # 执行分析
                result = agent.analyze(stock_code, time_range, **kwargs)

                # 恢复原始设置
                if original_enable_memory is not None:
                    agent.config.enable_memory = original_enable_memory
                    if not original_enable_memory and agent.memory:
                        # 可选：是否在分析后清理临时启用的记忆
                        # agent.memory = None
                        pass

                # 记录执行结果
                status = "成功" if result.success else "失败"
                memory_status = "使用记忆" if use_memory else "不使用记忆"
                info(f"执行 {agent_name} 分析完成，状态: {status}，{memory_status}")

                return result
            else:
                warning(f"智能体不存在: {agent_name}")
                return AgentResult(
                    agent_name=agent_name,
                    success=False,
                    result={},
                    error_message="智能体不存在",
                    confidence_score=0.0
                )
        except Exception as e:
            error(f"执行智能体分析失败: {str(e)}")
            return AgentResult(
                agent_name=agent_name,
                success=False,
                result={},
                error_message=str(e),
                confidence_score=0.0
            )
