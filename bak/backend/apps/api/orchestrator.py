from __future__ import annotations

from typing import Any, Dict, List
from .agents.base import BaseAgent, AgentResult
from .agents.data_agent import DataAgent
from .agents.analysis_agent import AnalysisAgent
from .agents.writing_agent import WritingAgent


AGENT_REGISTRY: Dict[str, BaseAgent] = {
    "DataAgent": DataAgent(),
    "AnalysisAgent": AnalysisAgent(),
    "WritingAgent": WritingAgent(),
}


def run_plan(plan: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    name_to_result: Dict[str, AgentResult] = {}
    for node in plan:
        agent_name = node.get("agent")
        params = node.get("params", {})
        depends_on = node.get("depends_on", [])
        # 简单的依赖检查
        if any(dep not in name_to_result for dep in depends_on):
            raise ValueError(f"依赖未满足: {depends_on}")
        agent = AGENT_REGISTRY.get(agent_name)
        if not agent:
            raise ValueError(f"未知的 Agent: {agent_name}")
        # 合并上下文：前序结果可放入 context[AgentName]
        for dep in depends_on:
            context[dep] = {
                "success": name_to_result[dep].success,
                "data": name_to_result[dep].data,
                "error": name_to_result[dep].error,
            }
        result = agent.run(params=params, context=context)
        name_to_result[agent_name] = result
        results[agent_name] = {
            "success": result.success,
            "data": result.data,
            "error": result.error,
        }
        # 将常用字段沉入 context，便于下游 agent 使用
        if agent_name == "DataAgent":
            context["prices"] = [p.get("close") for p in result.data.get("prices", [])]
            context["ticker"] = result.data.get("ticker") or context.get("ticker")
    return {"context": context, "results": results}


