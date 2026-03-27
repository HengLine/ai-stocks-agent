from typing import Dict, Any, List, Optional
import logging
from .base import BaseAgent, AgentResult
from ..services.context_memory import global_context_manager

logger = logging.getLogger(__name__)


class QAAgent(BaseAgent):
    """QA智能体，用于回答用户追问，支持多轮对话"""
    
    def __init__(self):
        """初始化QA智能体"""
        self.name = "QAAgent"
        logger.info("QAAgent initialized")
    
    def run(self, params: Dict[str, Any], context: Dict[str, Any] = None) -> AgentResult:
        """执行QA任务，回答用户问题"""
        try:
            logger.info(f"QAAgent running with params: {params}")
            
            # 获取必要参数
            user_query = params.get("query", "")
            user_id = params.get("user_id", "default_user")
            session_id = params.get("session_id", None)
            
            if not user_query:
                error_msg = "No query provided for QAAgent"
                logger.error(error_msg)
                return AgentResult(
                    name=self.name,
                    success=False,
                    data={},
                    error=error_msg
                )
            
            # 获取用户上下文
            context_memory = global_context_manager.get_context(user_id, session_id)
            
            # 搜索相关记忆
            relevant_memory = self._search_relevant_memory(context_memory, user_query)
            
            # 获取对话历史
            dialogue_history = context_memory.get_dialogue_history(limit=5)  # 获取最近5轮对话
            
            # 构建回答
            answer = self._generate_answer(user_query, relevant_memory, dialogue_history, context)
            
            # 保存对话到历史记录
            context_memory.add_dialogue_turn(user_query, answer)
            
            # 保存上下文
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(context_memory.save_context())
            else:
                loop.run_until_complete(context_memory.save_context())
            
            # 返回结果
            result_data = {
                "answer": answer,
                "query": user_query,
                "session_id": context_memory.session_id,
                "relevant_memory_count": len(relevant_memory)
            }
            
            return AgentResult(
                name=self.name,
                success=True,
                data=result_data,
                error=""
            )
        except Exception as e:
            error_msg = f"QAAgent failed: {str(e)}"
            logger.error(error_msg)
            return AgentResult(
                name=self.name,
                success=False,
                data={},
                error=error_msg
            )
    
    def _search_relevant_memory(self, context_memory: Any, query: str) -> List[Dict[str, Any]]:
        """搜索相关记忆"""
        try:
            # 构建搜索过滤条件
            filters = {"type": {"$in": ["long_term_memory", "analysis_result", "data_query"]}}
            
            # 执行搜索
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果事件循环已经在运行，使用run_coroutine_threadsafe
                future = asyncio.run_coroutine_threadsafe(
                    context_memory.search_memory(query, filters, limit=3),
                    loop
                )
                relevant_memory = future.result(timeout=5)  # 5秒超时
            else:
                # 否则直接运行
                relevant_memory = loop.run_until_complete(
                    context_memory.search_memory(query, filters, limit=3)
                )
            
            logger.debug(f"Found {len(relevant_memory)} relevant memories")
            return relevant_memory
        except Exception as e:
            logger.error(f"Failed to search relevant memory: {str(e)}")
            return []
    
    def _generate_answer(self, query: str, relevant_memory: List[Dict[str, Any]], 
                         dialogue_history: List[Dict[str, Any]], context: Dict[str, Any] = None) -> str:
        """生成回答"""
        try:
            # 构建回答内容
            # 在实际应用中，这里应该调用LLM服务来生成回答
            # 目前使用基于规则的简单回答生成
            
            # 检查是否是常见问题
            common_answers = self._get_common_answers(query)
            if common_answers:
                return common_answers
            
            # 检查是否有相关记忆可以使用
            if relevant_memory:
                # 基于相关记忆生成回答
                memory_content = "\n".join([str(m["content"]) for m in relevant_memory])
                return self._generate_answer_based_on_memory(query, memory_content)
            
            # 检查对话历史
            if dialogue_history:
                # 基于对话历史生成回答
                history_content = "\n".join([f"User: {h['user_query']}\nAgent: {h['agent_response']}" for h in dialogue_history])
                return self._generate_answer_based_on_history(query, history_content)
            
            # 如果没有相关信息，返回默认回答
            return self._get_default_answer(query)
        except Exception as e:
            logger.error(f"Failed to generate answer: {str(e)}")
            return "I'm sorry, I couldn't generate a response to your question. Please try again."
    
    def _get_common_answers(self, query: str) -> Optional[str]:
        """获取常见问题的回答"""
        query_lower = query.lower()
        
        # 定义一些常见问题和回答
        common_qa = {
            "你是谁": "我是股票分析智能助手，能够为您提供股票行情、分析报告和投资建议。",
            "你能做什么": "我可以为您提供股票行情查询、技术分析、基本面分析、生成分析报告，还能回答您的投资问题。",
            "如何使用你": "您可以直接问我关于股票的问题，比如'帮我分析一下贵州茅台的行情'或者'腾讯控股最近的走势如何'。",
            "什么是股票": "股票是股份公司发行的所有权凭证，是股份公司为筹集资金而发行给各个股东作为持股凭证并借以取得股息和红利的一种有价证券。",
            "什么是技术分析": "技术分析是通过研究过去的市场数据（价格、成交量等）来预测未来价格走势的方法。"
        }
        
        # 检查是否匹配常见问题
        for question, answer in common_qa.items():
            if question in query_lower:
                return answer
        
        # 检查是否是关于风险偏好的问题
        if any(keyword in query_lower for keyword in ["风险偏好", "风险等级", "能承受多大风险"]):
            return "您可以告诉我您的风险偏好（低、中、高），我会根据您的风险偏好为您提供更合适的投资建议。"
        
        # 检查是否是关于关注股票的问题
        if any(keyword in query_lower for keyword in ["关注的股票", "我的股票池", "自选股"]):
            return "您可以告诉我股票代码，我会帮您添加到关注列表中，方便您随时查看。"
        
        return None
    
    def _generate_answer_based_on_memory(self, query: str, memory_content: str) -> str:
        """基于记忆生成回答"""
        # 在实际应用中，这里应该调用LLM服务
        # 目前使用模板生成简单回答
        return f"根据我的记录，{query}相关的信息如下：\n{memory_content}\n\n如果您需要更详细的解释或有其他问题，请随时告诉我。"
    
    def _generate_answer_based_on_history(self, query: str, history_content: str) -> str:
        """基于对话历史生成回答"""
        # 在实际应用中，这里应该调用LLM服务
        # 目前使用模板生成简单回答
        return f"关于您的问题{query}，结合我们之前的对话，我认为：\n\n这是一个关于我们之前讨论过的话题的后续问题。为了给您更准确的回答，我需要了解更多细节。您可以提供更多信息吗？"
    
    def _get_default_answer(self, query: str) -> str:
        """获取默认回答"""
        # 在实际应用中，这里应该调用LLM服务
        # 目前使用模板生成简单回答
        return f"感谢您的提问。关于'{query}'，我需要获取更多信息来为您提供准确的回答。您可以提供更多细节，或者我可以为您查询相关的股票数据和分析报告。"
    
    def _extract_stock_code_from_query(self, query: str) -> Optional[str]:
        """从查询中提取股票代码"""
        import re
        
        # 匹配A股代码（6位数字）
        a_stock_match = re.search(r'[0-9]{6}', query)
        if a_stock_match:
            return a_stock_match.group()
        
        # 匹配港股代码（例如：00700）
        hk_stock_match = re.search(r'0[0-9]{4,5}', query)
        if hk_stock_match:
            return hk_stock_match.group()
        
        # 匹配美股代码（字母）
        us_stock_match = re.search(r'\b[A-Za-z]{1,5}\b', query)
        if us_stock_match:
            return us_stock_match.group().upper()
        
        return None
    
    def _get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """获取用户偏好"""
        context_memory = global_context_manager.get_context(user_id)
        return context_memory.user_preferences


# 注册QA智能体
def register_qa_agent():
    """注册QA智能体"""
    from ..orchestrator import AGENT_REGISTRY
    
    # 检查是否已经注册
    if "QAAgent" not in AGENT_REGISTRY:
        AGENT_REGISTRY["QAAgent"] = QAAgent()
        logger.info("QAAgent registered successfully")
    else:
        logger.warning("QAAgent is already registered")


# 在模块加载时自动注册
register_qa_agent()