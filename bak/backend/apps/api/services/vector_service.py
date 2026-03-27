from __future__ import annotations

from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer


class VectorService:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="conversations",
            metadata={"hnsw:space": "cosine"}
        )
        # 使用轻量级中文嵌入模型
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def add_conversation(self, user_id: str, message: str, response: str, metadata: Dict[str, Any] = None) -> str:
        """添加对话记录到向量数据库"""
        content = f"用户: {message}\n助手: {response}"
        embedding = self.encoder.encode(content).tolist()
        doc_id = f"{user_id}_{len(self.collection.get()['ids'])}"
        
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{
                "user_id": user_id,
                "message": message,
                "response": response,
                **(metadata or {})
            }]
        )
        return doc_id

    def search_similar(self, query: str, user_id: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """搜索相似对话"""
        query_embedding = self.encoder.encode(query).tolist()
        where_clause = {"user_id": user_id} if user_id else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_clause
        )
        
        conversations = []
        for i, doc_id in enumerate(results['ids'][0]):
            conversations.append({
                "id": doc_id,
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
        return conversations

    def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取用户对话历史"""
        results = self.collection.get(
            where={"user_id": user_id},
            limit=limit
        )
        
        conversations = []
        for i, doc_id in enumerate(results['ids']):
            conversations.append({
                "id": doc_id,
                "content": results['documents'][i],
                "metadata": results['metadatas'][i]
            })
        return conversations
