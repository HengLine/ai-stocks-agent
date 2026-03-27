from typing import Any, Dict, List, Optional, Callable, Union
import asyncio
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from .agents.base import BaseAgent, AgentResult
from .services.vector_store import global_vector_store

logger = logging.getLogger(__name__)


class TaskNode:
    """任务节点，表示一个Agent的执行任务"""
    def __init__(self,
                 agent_name: str,
                 params: Optional[Dict[str, Any]] = None,
                 depends_on: Optional[List[str]] = None,
                 node_id: Optional[str] = None):
        self.node_id = node_id or str(uuid.uuid4())[:8]
        self.agent_name = agent_name
        self.params = params or {}
        self.depends_on = depends_on or []
        self.result: Optional[AgentResult] = None
        self.status = "pending"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "agent_name": self.agent_name,
            "params": self.params,
            "depends_on": self.depends_on,
            "status": self.status
        }


class Workflow:
    """工作流定义，包含一系列任务节点及其关系"""
    def __init__(self,
                 name: str,
                 nodes: Optional[List[TaskNode]] = None,
                 conditions: Optional[Dict[str, Dict[str, Any]]] = None,
                 loops: Optional[Dict[str, Dict[str, Any]]] = None):
        self.name = name
        self.nodes = nodes or []
        self.conditions = conditions or {}
        self.loops = loops or {}
        self.node_map = {node.node_id: node for node in self.nodes}

    def add_node(self, node: TaskNode) -> "Workflow":
        """添加任务节点"""
        self.nodes.append(node)
        self.node_map[node.node_id] = node
        return self

    def add_condition(self, node_id: str, condition: Dict[str, Any]) -> "Workflow":
        """添加条件分支"""
        self.conditions[node_id] = condition
        return self

    def add_loop(self, node_id: str, loop_config: Dict[str, Any]) -> "Workflow":
        """添加循环配置"""
        self.loops[node_id] = loop_config
        return self

    def get_node(self, node_id: str) -> Optional[TaskNode]:
        """根据node_id获取任务节点"""
        return self.node_map.get(node_id)


class AgentRegistry:
    """Agent注册中心，管理所有可用的Agent"""
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> "AgentRegistry":
        """注册Agent"""
        self.agents[agent.name] = agent
        logger.info(f"Agent registered: {agent.name}")
        return self

    def get(self, name: str) -> Optional[BaseAgent]:
        """获取Agent实例"""
        return self.agents.get(name)

    def list_agents(self) -> List[str]:
        """列出所有可用的Agent"""
        return list(self.agents.keys())


class ContextManager:
    """上下文管理器，负责管理任务执行过程中的上下文和记忆"""
    def __init__(self, vector_store=None):
        self.global_context: Dict[str, Any] = {}
        self.vector_store = vector_store if vector_store is not None else global_vector_store
        self.session_id: Optional[str] = None
        self.user_preferences: Dict[str, Any] = {}

    def set_session_id(self, session_id: str) -> "ContextManager":
        """设置会话ID"""
        self.session_id = session_id
        return self

    def update_global_context(self, key: str, value: Any) -> "ContextManager":
        """更新全局上下文"""
        self.global_context[key] = value
        return self

    def merge_context(self, context: Dict[str, Any]) -> "ContextManager":
        """合并上下文"""
        self.global_context.update(context)
        return self

    def get_context(self, keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """获取上下文，可以指定键列表"""
        if keys:
            return {k: self.global_context.get(k) for k in keys}
        return self.global_context.copy()

    async def store_memory(self, content: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
        """存储记忆到向量数据库"""
        if not self.vector_store:
            logger.warning("Vector store not available, cannot store memory")
            return ""
        
        metadata = metadata or {}
        if self.session_id:
            metadata["session_id"] = self.session_id
        
        metadata["timestamp"] = datetime.now().isoformat()
        
        try:
            memory_id = await self.vector_store.add_document(content, metadata)
            logger.info(f"Memory stored with id: {memory_id}")
            return memory_id
        except Exception as e:
            logger.error(f"Failed to store memory: {str(e)}")
            return ""

    async def retrieve_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """根据查询检索相关记忆"""
        if not self.vector_store:
            logger.warning("Vector store not available, cannot retrieve memories")
            return []
        
        try:
            filters = {"session_id": self.session_id} if self.session_id else {}
            results = await self.vector_store.search(query, filters=filters, limit=limit)
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {str(e)}")
            return []

    def set_user_preferences(self, preferences: Dict[str, Any]) -> "ContextManager":
        """设置用户偏好"""
        self.user_preferences.update(preferences)
        # 将用户偏好也放入全局上下文
        self.global_context["user_preferences"] = self.user_preferences
        return self


class IntelligentOrchestrator:
    """智能体协作调度引擎"""
    def __init__(self, vector_store=None):
        self.agent_registry = AgentRegistry()
        self.context_manager = ContextManager(vector_store)
        self.execution_history: List[Dict[str, Any]] = []

    def register_agent(self, agent: BaseAgent) -> "IntelligentOrchestrator":
        """注册Agent"""
        self.agent_registry.register(agent)
        return self

    def set_session_id(self, session_id: str) -> "IntelligentOrchestrator":
        """设置会话ID"""
        self.context_manager.set_session_id(session_id)
        return self

    def set_user_preferences(self, preferences: Dict[str, Any]) -> "IntelligentOrchestrator":
        """设置用户偏好"""
        self.context_manager.set_user_preferences(preferences)
        return self

    async def execute_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """执行工作流"""
        start_time = datetime.now()
        execution_id = str(uuid.uuid4())[:12]
        
        logger.info(f"Starting workflow execution: {workflow.name}, execution_id: {execution_id}")
        
        # 初始化执行状态
        execution_state = {
            "execution_id": execution_id,
            "workflow_name": workflow.name,
            "start_time": start_time.isoformat(),
            "nodes": {node.node_id: node.to_dict() for node in workflow.nodes},
            "results": {},
            "status": "running"
        }
        
        try:
            # 执行工作流
            await self._execute_nodes(workflow, execution_state)
            
            # 更新执行状态
            execution_state["status"] = "completed"
            execution_state["end_time"] = datetime.now().isoformat()
            execution_state["duration_seconds"] = (datetime.now() - start_time).total_seconds()
            
            # 存储执行历史
            self.execution_history.append(execution_state)
            
            logger.info(f"Workflow execution completed: {workflow.name}, execution_id: {execution_id}")
            
            return execution_state
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            execution_state["status"] = "failed"
            execution_state["error"] = str(e)
            execution_state["end_time"] = datetime.now().isoformat()
            execution_state["duration_seconds"] = (datetime.now() - start_time).total_seconds()
            
            # 存储执行历史
            self.execution_history.append(execution_state)
            
            return execution_state

    async def _execute_nodes(self, workflow: Workflow, execution_state: Dict[str, Any]) -> None:
        """执行工作流中的节点"""
        # 首先执行没有依赖的节点（根节点）
        root_nodes = [node for node in workflow.nodes if not node.depends_on]
        
        # 使用队列来管理待执行的节点
        pending_nodes = root_nodes.copy()
        completed_nodes = set()
        
        while pending_nodes:
            # 找出当前可以执行的节点（所有依赖都已完成）
            executable_nodes = []
            remaining_nodes = []
            
            for node in pending_nodes:
                if all(dep in completed_nodes for dep in node.depends_on):
                    executable_nodes.append(node)
                else:
                    remaining_nodes.append(node)
            
            # 更新待执行队列
            pending_nodes = remaining_nodes
            
            # 并行执行当前可执行的节点
            if executable_nodes:
                # 对于某些节点，检查是否有条件执行
                tasks = []
                for node in executable_nodes:
                    # 检查条件
                    if not self._check_condition(node.node_id, workflow.conditions):
                        logger.info(f"Skipping node {node.node_id} due to condition not met")
                        continue
                    
                    # 检查循环配置
                    loop_config = workflow.loops.get(node.node_id)
                    if loop_config:
                        # 执行循环
                        await self._execute_loop(node, loop_config)
                    else:
                        # 单个执行
                        tasks.append(self._execute_node(node))
                
                # 等待所有并行任务完成
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # 处理执行结果
                    for i, result in enumerate(results):
                        node = executable_nodes[i]
                        if isinstance(result, Exception):
                            logger.error(f"Node {node.node_id} execution failed: {str(result)}")
                            node.status = "failed"
                            node.result = AgentResult(
                                name=node.agent_name,
                                success=False,
                                data={},
                                error=str(result)
                            )
                        else:
                            node.status = "completed"
                            node.result = result
                            completed_nodes.add(node.node_id)
                            
                            # 更新上下文
                            self._update_context_with_result(node, result)
                            
                        # 更新执行状态
                        execution_state["nodes"][node.node_id] = node.to_dict()
                        execution_state["results"][node.node_id] = {
                            "success": result.success if isinstance(result, AgentResult) else False,
                            "data": result.data if isinstance(result, AgentResult) else {},
                            "error": result.error if isinstance(result, AgentResult) else str(result) if isinstance(result, Exception) else None
                        }
    
    async def _execute_node(self, node: TaskNode) -> AgentResult:
        """执行单个任务节点"""
        logger.info(f"Executing node: {node.node_id}, agent: {node.agent_name}")
        
        # 获取Agent实例
        agent = self.agent_registry.get(node.agent_name)
        if not agent:
            raise ValueError(f"Agent not found: {node.agent_name}")
        
        # 准备上下文
        context = self.context_manager.get_context()
        
        # 执行Agent
        result = agent.run(params=node.params, context=context)
        
        logger.info(f"Node execution completed: {node.node_id}")
        return result
    
    async def _execute_loop(self, node: TaskNode, loop_config: Dict[str, Any]) -> None:
        """执行循环任务"""
        iterations = loop_config.get("iterations", 1)
        condition_func = loop_config.get("condition")
        
        logger.info(f"Starting loop for node: {node.node_id}, iterations: {iterations}")
        
        for i in range(iterations):
            # 如果有条件函数，检查是否继续循环
            if condition_func and not condition_func(self.context_manager.get_context()):
                logger.info(f"Loop condition not met, breaking loop for node: {node.node_id}")
                break
            
            # 执行节点
            result = await self._execute_node(node)
            
            # 更新上下文
            self._update_context_with_result(node, result)
            
            # 如果有延迟配置，等待
            delay = loop_config.get("delay", 0)
            if delay > 0:
                await asyncio.sleep(delay)
    
    def _check_condition(self, node_id: str, conditions: Dict[str, Dict[str, Any]]) -> bool:
        """检查节点的执行条件"""
        condition = conditions.get(node_id)
        if not condition:
            return True
        
        condition_type = condition.get("type", "always")
        
        if condition_type == "always":
            return True
        elif condition_type == "context":
            # 基于上下文的条件
            context_path = condition.get("context_path")
            expected_value = condition.get("value")
            
            if context_path:
                # 简单的上下文路径解析，如 "analysis.signal" -> context["analysis"]["signal"]
                parts = context_path.split(".")
                value = self.context_manager.get_context()
                
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        return False
                
                return value == expected_value
            
        elif condition_type == "function":
            # 基于函数的条件
            func = condition.get("function")
            if callable(func):
                return func(self.context_manager.get_context())
        
        return True
    
    def _update_context_with_result(self, node: TaskNode, result: AgentResult) -> None:
        """用Agent执行结果更新上下文"""
        if not result.success:
            return
        
        # 将Agent结果整体放入上下文
        self.context_manager.update_global_context(node.agent_name, {
            "success": result.success,
            "data": result.data,
            "error": result.error
        })
        
        # 对于常用字段，直接放入上下文根级别
        if node.agent_name == "DataAgent" and result.data:
            self.context_manager.update_global_context("prices", result.data.get("prices", []))
            self.context_manager.update_global_context("ticker", result.data.get("ticker"))
        elif node.agent_name == "AnalysisAgent" and result.data:
            self.context_manager.update_global_context("current_price", result.data.get("current_price"))
            self.context_manager.update_global_context("technical_analysis", result.data.get("technical_analysis"))
        elif node.agent_name == "WritingAgent" and result.data:
            self.context_manager.update_global_context("article", result.data.get("article"))
    
    async def create_workflow_from_query(self, query: Dict[str, Any]) -> Workflow:
        """从用户查询动态创建工作流"""
        # 示例实现：根据用户查询创建工作流
        workflow_name = f"stock_analysis_{query.get('ticker', 'unknown')}"
        workflow = Workflow(name=workflow_name)
        
        # 添加DataAgent节点
        data_node = TaskNode(
            agent_name="DataAgent",
            params={
                "ticker": query.get("ticker"),
                "time_window": query.get("time_window", "3M")
            }
        )
        workflow.add_node(data_node)
        
        # 添加AnalysisAgent节点，依赖DataAgent
        analysis_node = TaskNode(
            agent_name="AnalysisAgent",
            depends_on=[data_node.node_id]
        )
        workflow.add_node(analysis_node)
        
        # 根据输出类型添加相应的Agent节点
        output_type = query.get("output", "article")
        if output_type == "article":
            # 添加WritingAgent节点
            writing_node = TaskNode(
                agent_name="WritingAgent",
                params={
                    "style": query.get("style", "professional"),
                    "format": "markdown"
                },
                depends_on=[analysis_node.node_id]
            )
            workflow.add_node(writing_node)
        elif output_type == "video":
            # 添加VideoAgent节点
            video_node = TaskNode(
                agent_name="VideoAgent",
                params={
                    "style": query.get("style", "professional"),
                    "duration": "short"
                },
                depends_on=[analysis_node.node_id]
            )
            workflow.add_node(video_node)
        
        return workflow

    async def handle_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """处理用户查询，创建并执行工作流"""
        # 存储用户查询到上下文
        self.context_manager.merge_context(query)
        
        # 创建工作流
        workflow = await self.create_workflow_from_query(query)
        
        # 执行工作流
        result = await self.execute_workflow(workflow)
        
        # 存储结果到向量数据库作为记忆
        if result["status"] == "completed":
            await self.context_manager.store_memory(
                content={
                    "query": query,
                    "result": result["results"]
                },
                metadata={
                    "type": "query_result"
                }
            )
        
        return result


# 创建全局调度引擎实例
def get_orchestrator() -> IntelligentOrchestrator:
    """获取全局调度引擎实例"""
    # 初始化调度引擎，使用全局向量数据库客户端
    orchestrator = IntelligentOrchestrator(global_vector_store)
    
    # 注册所有已知的Agent
    try:
        from .agents.data_agent import DataAgent
        from .agents.analysis_agent import AnalysisAgent
        from .agents.writing_agent import WritingAgent
        from .agents.video_agent import VideoAgent
        from .agents.qa_agent import QAAgent
        
        orchestrator.register_agent(DataAgent())
        orchestrator.register_agent(AnalysisAgent())
        orchestrator.register_agent(WritingAgent())
        orchestrator.register_agent(VideoAgent())
        orchestrator.register_agent(QAAgent())
        
        logger.info(f"Successfully registered {len(orchestrator.agent_registry.list_agents())} agents")
    except ImportError as e:
        logger.error(f"Failed to import or register some agents: {str(e)}")
    
    return orchestrator


# 全局调度引擎实例
global_orchestrator = get_orchestrator()