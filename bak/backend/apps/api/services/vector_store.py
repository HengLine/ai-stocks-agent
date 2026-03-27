from typing import Any, Dict, List, Optional, Union
import logging
import json

logger = logging.getLogger(__name__)


class VectorStoreClient:
    """向量数据库客户端，用于存储和检索向量数据（如对话历史、用户偏好等）"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化向量数据库客户端"""
        self.config = config or {}
        self.embedding_model = self.config.get("embedding_model", "default")
        self.vector_store_type = self.config.get("type", "mock")  # mock, milvus, pinecone, chromadb
        
        # 模拟存储，实际应用中应该替换为真实的向量数据库客户端
        self.mock_store = {}
        
        logger.info(f"Vector store client initialized with type: {self.vector_store_type}")
    
    async def initialize(self) -> bool:
        """初始化向量数据库连接"""
        try:
            if self.vector_store_type == "milvus":
                # 实际应用中，这里应该初始化Milvus客户端
                logger.info("Initializing Milvus vector store")
                # 示例：
                # from pymilvus import connections, db
                # connections.connect(**self.config.get('connection_params', {}))
            elif self.vector_store_type == "pinecone":
                # 实际应用中，这里应该初始化Pinecone客户端
                logger.info("Initializing Pinecone vector store")
                # 示例：
                # import pinecone
                # pinecone.init(**self.config.get('connection_params', {}))
            elif self.vector_store_type == "chromadb":
                # 实际应用中，这里应该初始化ChromaDB客户端
                logger.info("Initializing ChromaDB vector store")
                # 示例：
                # import chromadb
                # self.client = chromadb.Client()
            else:
                # 使用模拟存储
                logger.info("Using mock vector store")
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            return False
    
    async def add_document(self, content: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
        """添加文档到向量数据库"""
        try:
            metadata = metadata or {}
            document_id = self._generate_document_id()
            
            if self.vector_store_type == "mock":
                # 模拟存储实现
                document = {
                    "id": document_id,
                    "content": content,
                    "metadata": metadata,
                    "vector": self._generate_mock_vector(content)
                }
                self.mock_store[document_id] = document
            else:
                # 实际应用中，这里应该调用相应向量数据库的API添加文档
                # 例如：
                # vector = self._generate_embedding(content)
                # self.client.insert(collection_name, [document_id], [vector], [metadata])
                pass
            
            logger.info(f"Document added with id: {document_id}")
            return document_id
        except Exception as e:
            logger.error(f"Failed to add document: {str(e)}")
            return ""
    
    async def search(self, query: str, filters: Dict[str, Any] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """根据查询检索相关文档"""
        try:
            filters = filters or {}
            
            if self.vector_store_type == "mock":
                # 模拟搜索实现
                results = []
                
                # 简单的基于关键词的搜索
                query_lower = query.lower()
                
                for doc_id, doc in self.mock_store.items():
                    # 检查过滤条件
                    if not self._check_filters(doc["metadata"], filters):
                        continue
                    
                    # 简单的文本匹配评分
                    content_str = json.dumps(doc["content"]).lower()
                    score = 0
                    if query_lower in content_str:
                        score = 0.8 + (content_str.count(query_lower) * 0.05)
                    
                    if score > 0:
                        results.append({
                            "id": doc_id,
                            "content": doc["content"],
                            "metadata": doc["metadata"],
                            "score": min(score, 1.0)
                        })
                
                # 按评分排序并限制结果数量
                results.sort(key=lambda x: x["score"], reverse=True)
                return results[:limit]
            else:
                # 实际应用中，这里应该调用相应向量数据库的搜索API
                # 例如：
                # query_vector = self._generate_embedding(query)
                # results = self.client.search(collection_name, [query_vector], filter=filters, limit=limit)
                return []
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取文档"""
        try:
            if self.vector_store_type == "mock":
                return self.mock_store.get(document_id)
            else:
                # 实际应用中，这里应该调用相应向量数据库的API获取文档
                # 例如：
                # result = self.client.get(collection_name, [document_id])
                # return result[0] if result else None
                return None
        except Exception as e:
            logger.error(f"Failed to get document: {str(e)}")
            return None
    
    async def update_document(self, document_id: str, content: Dict[str, Any] = None, metadata: Dict[str, Any] = None) -> bool:
        """更新文档"""
        try:
            if self.vector_store_type == "mock":
                if document_id not in self.mock_store:
                    return False
                
                if content is not None:
                    self.mock_store[document_id]["content"] = content
                    # 更新向量
                    self.mock_store[document_id]["vector"] = self._generate_mock_vector(content)
                
                if metadata is not None:
                    self.mock_store[document_id]["metadata"].update(metadata)
                
                return True
            else:
                # 实际应用中，这里应该调用相应向量数据库的API更新文档
                # 例如：
                # if content is not None:
                #     vector = self._generate_embedding(content)
                #     self.client.update(collection_name, [document_id], [vector], [metadata])
                return False
        except Exception as e:
            logger.error(f"Failed to update document: {str(e)}")
            return False
    
    async def delete_document(self, document_id: str) -> bool:
        """删除文档"""
        try:
            if self.vector_store_type == "mock":
                if document_id in self.mock_store:
                    del self.mock_store[document_id]
                    return True
                return False
            else:
                # 实际应用中，这里应该调用相应向量数据库的API删除文档
                # 例如：
                # self.client.delete(collection_name, [document_id])
                return False
        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
            return False
    
    async def batch_add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """批量添加文档"""
        try:
            document_ids = []
            
            for doc in documents:
                content = doc.get("content", {})
                metadata = doc.get("metadata", {})
                doc_id = await self.add_document(content, metadata)
                if doc_id:
                    document_ids.append(doc_id)
            
            return document_ids
        except Exception as e:
            logger.error(f"Batch add documents failed: {str(e)}")
            return []
    
    def _generate_document_id(self) -> str:
        """生成文档ID"""
        import uuid
        return str(uuid.uuid4())
    
    def _generate_mock_vector(self, content: Dict[str, Any]) -> List[float]:
        """生成模拟向量（实际应用中应该使用真实的嵌入模型）"""
        import hashlib
        import random
        
        # 将内容转换为字符串并哈希
        content_str = json.dumps(content, sort_keys=True)
        hash_obj = hashlib.md5(content_str.encode())
        hash_hex = hash_obj.hexdigest()
        
        # 从哈希值生成固定长度的向量
        vector_length = 128  # 模拟向量长度
        vector = []
        
        # 使用哈希值的每两个字符生成一个浮点数
        for i in range(0, min(len(hash_hex), vector_length * 2), 2):
            # 将两个十六进制字符转换为0-1之间的浮点数
            value = int(hash_hex[i:i+2], 16) / 255.0
            vector.append(value)
        
        # 如果向量长度不足，用随机数填充
        while len(vector) < vector_length:
            vector.append(random.random())
        
        return vector
    
    def _check_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """检查元数据是否满足过滤条件"""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    def _generate_embedding(self, content: Union[str, Dict[str, Any]]) -> List[float]:
        """生成内容的向量嵌入（实际应用中应该调用真实的嵌入模型）"""
        # 这个方法在使用真实向量数据库时需要实现
        # 例如：
        # if isinstance(content, dict):
        #     content_str = json.dumps(content, sort_keys=True)
        # else:
        #     content_str = str(content)
        # 
        # # 调用嵌入模型API
        # response = requests.post(embedding_api_url, json={"text": content_str})
        # return response.json().get("embedding", [])
        
        # 目前返回模拟向量
        return self._generate_mock_vector(content if isinstance(content, dict) else {"text": content})


# 创建全局向量数据库客户端实例
def get_vector_store_client(config: Dict[str, Any] = None) -> VectorStoreClient:
    """获取向量数据库客户端实例"""
    client = VectorStoreClient(config)
    # 异步初始化，但这里直接返回，实际应用中应该等待初始化完成
    import asyncio
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # 如果事件循环已经在运行，创建一个新的任务
        loop.create_task(client.initialize())
    else:
        # 否则直接运行初始化
        loop.run_until_complete(client.initialize())
    
    return client


# 全局向量数据库客户端实例
global_vector_store = get_vector_store_client()