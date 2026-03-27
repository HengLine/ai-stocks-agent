from typing import Any, Dict, List, Optional, Union
import logging
import time
from datetime import datetime, timedelta
from ..services.vector_store import global_vector_store

logger = logging.getLogger(__name__)


class ContextMemory:
    """上下文记忆库，用于管理智能体交互过程中的上下文信息"""
    
    def __init__(self, user_id: str, session_id: str = None):
        """初始化上下文记忆库"""
        self.user_id = user_id
        self.session_id = session_id or self._generate_session_id()
        self.context_data = {}
        self.dialogue_history = []
        self.user_preferences = {}
        self.long_term_memory = {}
        
        # 加载用户数据
        self._load_user_data()
        
        logger.info(f"Context memory initialized for user: {user_id}, session: {session_id}")
    
    def _generate_session_id(self) -> str:
        """生成会话ID"""
        import uuid
        return str(uuid.uuid4())
    
    async def _load_user_data(self):
        """从向量数据库加载用户数据"""
        try:
            # 加载用户偏好
            preferences_filter = {"user_id": self.user_id, "type": "user_preferences"}
            preferences_results = await global_vector_store.search("user preferences", preferences_filter, limit=1)
            if preferences_results:
                self.user_preferences = preferences_results[0]["content"]
                logger.info(f"User preferences loaded for {self.user_id}")
            
            # 加载长期记忆
            memory_filter = {"user_id": self.user_id, "type": "long_term_memory"}
            memory_results = await global_vector_store.search("long term memory", memory_filter, limit=20)
            for result in memory_results:
                memory_key = result["metadata"].get("memory_key", "")
                if memory_key:
                    self.long_term_memory[memory_key] = result["content"]
            
            logger.info(f"Long term memory loaded for {self.user_id}, count: {len(self.long_term_memory)}")
        except Exception as e:
            logger.error(f"Failed to load user data: {str(e)}")
    
    async def save_context(self):
        """保存上下文数据到向量数据库"""
        try:
            # 保存用户偏好
            preferences_doc = {
                "content": self.user_preferences,
                "metadata": {
                    "user_id": self.user_id,
                    "type": "user_preferences",
                    "updated_at": datetime.now().isoformat()
                }
            }
            
            # 保存长期记忆
            memory_docs = []
            for key, value in self.long_term_memory.items():
                memory_docs.append({
                    "content": value,
                    "metadata": {
                        "user_id": self.user_id,
                        "type": "long_term_memory",
                        "memory_key": key,
                        "updated_at": datetime.now().isoformat()
                    }
                })
            
            # 批量添加文档
            if memory_docs:
                await global_vector_store.batch_add_documents(memory_docs)
            
            # 添加用户偏好文档
            await global_vector_store.add_document(**preferences_doc)
            
            logger.info(f"Context saved for user: {self.user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save context: {str(e)}")
            return False
    
    def add_dialogue_turn(self, user_query: str, agent_response: str, metadata: Dict[str, Any] = None):
        """添加对话轮次到历史记录"""
        try:
            dialogue_turn = {
                "timestamp": datetime.now().isoformat(),
                "user_query": user_query,
                "agent_response": agent_response,
                "metadata": metadata or {}
            }
            
            self.dialogue_history.append(dialogue_turn)
            
            # 限制历史记录长度，防止内存溢出
            max_history_length = 100  # 可配置
            if len(self.dialogue_history) > max_history_length:
                self.dialogue_history = self.dialogue_history[-max_history_length:]
            
            logger.debug(f"Dialogue turn added for user: {self.user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add dialogue turn: {str(e)}")
            return False
    
    def get_dialogue_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取对话历史记录"""
        return self.dialogue_history[-limit:]
    
    def set_user_preference(self, key: str, value: Any):
        """设置用户偏好"""
        self.user_preferences[key] = value
        logger.debug(f"User preference set: {key} = {value}")
    
    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """获取用户偏好"""
        return self.user_preferences.get(key, default)
    
    def add_long_term_memory(self, key: str, value: Any, metadata: Dict[str, Any] = None):
        """添加长期记忆"""
        memory_item = {
            "value": value,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        self.long_term_memory[key] = memory_item
        logger.debug(f"Long term memory added: {key}")
    
    def update_long_term_memory(self, key: str, value: Any, metadata: Dict[str, Any] = None):
        """更新长期记忆"""
        if key in self.long_term_memory:
            self.long_term_memory[key]["value"] = value
            self.long_term_memory[key]["updated_at"] = datetime.now().isoformat()
            if metadata:
                self.long_term_memory[key]["metadata"].update(metadata)
            logger.debug(f"Long term memory updated: {key}")
        else:
            self.add_long_term_memory(key, value, metadata)
    
    def get_long_term_memory(self, key: str, default: Any = None) -> Any:
        """获取长期记忆"""
        if key in self.long_term_memory:
            return self.long_term_memory[key]["value"]
        return default
    
    def delete_long_term_memory(self, key: str) -> bool:
        """删除长期记忆"""
        if key in self.long_term_memory:
            del self.long_term_memory[key]
            logger.debug(f"Long term memory deleted: {key}")
            return True
        return False
    
    async def search_memory(self, query: str, filters: Dict[str, Any] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """搜索记忆"""
        try:
            # 构建过滤条件，确保只搜索当前用户的记忆
            memory_filters = {"user_id": self.user_id}
            if filters:
                memory_filters.update(filters)
            
            # 搜索向量数据库
            results = await global_vector_store.search(query, memory_filters, limit)
            
            logger.debug(f"Memory search completed for query: {query}, results: {len(results)}")
            return results
        except Exception as e:
            logger.error(f"Failed to search memory: {str(e)}")
            return []
    
    def set_context_value(self, key: str, value: Any):
        """设置上下文值"""
        self.context_data[key] = value
        logger.debug(f"Context value set: {key}")
    
    def get_context_value(self, key: str, default: Any = None) -> Any:
        """获取上下文值"""
        return self.context_data.get(key, default)
    
    def clear_context(self):
        """清空上下文数据（保留历史记录和长期记忆）"""
        self.context_data.clear()
        logger.debug(f"Context cleared for user: {self.user_id}")
    
    def get_recent_interactions(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取最近的交互记录"""
        recent_interactions = []
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        for turn in self.dialogue_history:
            turn_time = datetime.fromisoformat(turn["timestamp"])
            if turn_time >= cutoff_time:
                recent_interactions.append(turn)
        
        return recent_interactions
    
    def get_stock_watchlist(self) -> List[str]:
        """获取用户关注的股票池"""
        return self.get_user_preference("stock_watchlist", [])
    
    def add_to_stock_watchlist(self, stock_code: str) -> bool:
        """添加股票到关注列表"""
        watchlist = self.get_stock_watchlist()
        if stock_code not in watchlist:
            watchlist.append(stock_code)
            self.set_user_preference("stock_watchlist", watchlist)
            logger.info(f"Added stock {stock_code} to watchlist for user {self.user_id}")
            return True
        return False
    
    def remove_from_stock_watchlist(self, stock_code: str) -> bool:
        """从关注列表移除股票"""
        watchlist = self.get_stock_watchlist()
        if stock_code in watchlist:
            watchlist.remove(stock_code)
            self.set_user_preference("stock_watchlist", watchlist)
            logger.info(f"Removed stock {stock_code} from watchlist for user {self.user_id}")
            return True
        return False
    
    def get_risk_preference(self) -> str:
        """获取用户风险偏好"""
        return self.get_user_preference("risk_preference", "medium")  # low, medium, high
    
    def set_risk_preference(self, risk_level: str) -> bool:
        """设置用户风险偏好"""
        valid_levels = ["low", "medium", "high"]
        if risk_level.lower() in valid_levels:
            self.set_user_preference("risk_preference", risk_level.lower())
            logger.info(f"Risk preference set to {risk_level} for user {self.user_id}")
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """将上下文记忆转换为字典"""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "context_data": self.context_data,
            "dialogue_history": self.dialogue_history,
            "user_preferences": self.user_preferences,
            "long_term_memory": self.long_term_memory
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextMemory":
        """从字典创建上下文记忆对象"""
        instance = cls(data["user_id"], data["session_id"])
        instance.context_data = data.get("context_data", {})
        instance.dialogue_history = data.get("dialogue_history", [])
        instance.user_preferences = data.get("user_preferences", {})
        instance.long_term_memory = data.get("long_term_memory", {})
        return instance


# 上下文记忆管理类
class ContextManager:
    """上下文记忆管理器，管理多个用户的上下文记忆"""
    
    def __init__(self):
        """初始化上下文管理器"""
        self.contexts: Dict[str, ContextMemory] = {}
        self.session_to_user: Dict[str, str] = {}
        
        logger.info("Context manager initialized")
    
    def get_context(self, user_id: str, session_id: str = None) -> ContextMemory:
        """获取用户的上下文记忆"""
        # 如果提供了会话ID，检查是否已经存在
        if session_id and session_id in self.session_to_user:
            existing_user_id = self.session_to_user[session_id]
            context_key = f"{existing_user_id}_{session_id}"
            if context_key in self.contexts:
                return self.contexts[context_key]
        
        # 如果没有提供会话ID，或者会话ID不存在，创建新的上下文
        if not session_id:
            session_id = self._generate_session_id()
        
        context_key = f"{user_id}_{session_id}"
        
        # 检查是否已经存在该用户和会话的上下文
        if context_key not in self.contexts:
            self.contexts[context_key] = ContextMemory(user_id, session_id)
            self.session_to_user[session_id] = user_id
        
        return self.contexts[context_key]
    
    def _generate_session_id(self) -> str:
        """生成会话ID"""
        import uuid
        return str(uuid.uuid4())
    
    async def save_all_contexts(self) -> bool:
        """保存所有上下文"""
        success = True
        for context in self.contexts.values():
            if not await context.save_context():
                success = False
        
        if success:
            logger.info("All contexts saved successfully")
        else:
            logger.warning("Some contexts failed to save")
        
        return success
    
    def clear_user_contexts(self, user_id: str) -> bool:
        """清除用户的所有上下文"""
        user_context_keys = [key for key in self.contexts.keys() if key.startswith(f"{user_id}_")]
        
        for key in user_context_keys:
            _, session_id = key.split("_", 1)
            if session_id in self.session_to_user:
                del self.session_to_user[session_id]
            del self.contexts[key]
        
        logger.info(f"Cleared {len(user_context_keys)} contexts for user: {user_id}")
        return True
    
    def get_user_sessions(self, user_id: str) -> List[str]:
        """获取用户的所有会话ID"""
        return [key.split("_", 1)[1] for key in self.contexts.keys() if key.startswith(f"{user_id}_")]
    
    def get_active_users_count(self) -> int:
        """获取活跃用户数量"""
        return len(set(self.session_to_user.values()))
    
    def get_total_contexts_count(self) -> int:
        """获取总上下文数量"""
        return len(self.contexts)


# 创建全局上下文管理器实例
global_context_manager = ContextManager()


def get_context_manager() -> ContextManager:
    """获取上下文管理器实例"""
    return global_context_manager