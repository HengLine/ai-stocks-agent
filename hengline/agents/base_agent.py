#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@FileName: base_agent.py
@Description: 智能体基础抽象类，定义所有智能体共有的接口和功能，包括记忆管理和知识库检索
@Author: HengLine
@Time: 2025/11/10
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from langchain_classic.memory import VectorStoreRetrieverMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel

from hengline.client.client_factory import ClientFactory
from hengline.client.embedding_client import get_embedding_client
from hengline.logger import debug, error, warning, info
from config.config import get_ai_config, get_embedding_config
from hengline.tools.llama_index_retriever import DocumentRetriever
from utils.log_utils import print_log_exception
# 尝试导入FAISS，如果失败则提供替代方案
try:
    from langchain_community.vectorstores import FAISS
    FAISS_AVAILABLE = True
except ImportError as e:
    if 'faiss.swigfaiss_avx512' in str(e):
        # 如果是AVX512相关错误，设置FAISS不可用并使用替代方案
        FAISS_AVAILABLE = False
        warning("FAISS导入失败（AVX512相关错误），将使用替代方案")
    else:
        # 其他导入错误则重新抛出
        raise

class AgentConfig(BaseModel):
    """智能体配置"""
    agent_name: str
    description: str

    llm_config: Optional[Dict[str, Any]] = get_ai_config()  # LLM配置参数
    embedding_config: Optional[Dict[str, Any]] = get_embedding_config()  # 嵌入模型配置参数
    model_type: str = llm_config.get("provider", "openai") if llm_config else "openai"
    model_name: str = llm_config.get("model_name", "gpt-4") if llm_config else "gpt-4"
    temperature: float = llm_config.get("temperature", 0.3) if llm_config else 0.3
    max_tokens: int = llm_config.get("max_tokens", 2000) if llm_config else 2000
    memory_top_k: int = 3  # 从记忆中检索的最大文档数
    enable_memory: bool = embedding_config.get("enable_memory", True) if embedding_config else True  # 是否启用记忆功能
    embedding_type: str = embedding_config.get("provider", "openai") if embedding_config else "openai"  # 嵌入模型类型
    embedding_model: str = embedding_config.get("model_name", "BAAI/bge-small-zh-v1.5") if embedding_config else "BAAI/bge-small-zh-v1.5"  # 用于向量存储的嵌入模型


class AgentResult(BaseModel):
    """智能体执行结果"""
    agent_name: str
    success: bool
    result: Dict[str, Any]
    error_message: Optional[str] = None
    confidence_score: float = 0.0  # 0-1之间的置信度分数


class BaseAgent(ABC):
    """智能体基础抽象类"""

    def __init__(self, config: AgentConfig):
        """
        初始化智能体
        
        Args:
            config: 智能体配置
        """
        self.config = config
        self.agent_name = config.agent_name
        self.description = config.description

        # 初始化LLM客户端
        self.client = ClientFactory.create_client(
            provider=config.model_type,
            config=config.llm_config
        )

        # 获取LangChain兼容的LLM
        self.langchain_llm = self._get_langchain_llm()

        # 初始化文档检索器
        try:
            # 导入必要的组件
            from hengline.tools.llama_index_tool import create_index_from_directory
            from config.config import get_data_embeddings_path
            import os

            # 获取知识库目录
            knowledge_base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'knowledge_base')
            embedding_dir = get_data_embeddings_path()

            # 从知识库目录创建或加载向量索引
            vector_index = create_index_from_directory(
                directory_path=knowledge_base_dir,
                index_name="stock_knowledge_base",
                storage_dir=embedding_dir,
                embedding_model=None,
                recursive=True,
                rebuild=False
            )

            # 初始化文档检索器
            self.retriever = DocumentRetriever(
                index=vector_index,
                similarity_top_k=3,
                search_type="similarity",
                similarity_threshold=0.5
            )
            debug(f"成功初始化文档检索器，使用知识库: {knowledge_base_dir}")
        except Exception as e:
            error(f"初始化文档检索器失败: {str(e)}")
            # 创建一个默认的DocumentRetriever（如果可能）
            self.retriever = None

        # 初始化记忆系统
        self.memory = None
        if config.enable_memory:
            self.memory = self._init_memory()

        debug(f"初始化智能体: {self.agent_name}")
        if config.enable_memory:
            debug(f"智能体记忆功能已启用")

    @abstractmethod
    def analyze(self, stock_code: str, time_range: str = "1y", **kwargs) -> AgentResult:
        """
        执行分析任务
        
        Args:
            stock_code: 股票代码
            time_range: 时间范围，如 "1y"、"6m"、"3m"等
            **kwargs: 其他参数
            
        Returns:
            AgentResult: 分析结果
        """
        pass

    def _get_langchain_llm(self):
        """
        获取LangChain兼容的LLM实例
        
        Returns:
            LangChain LLM实例
        """
        try:
            # 这里使用ClientFactory获取LangChain LLM
            from hengline.tools.agent_tool import AgentTools
            tools = AgentTools()
            return tools.get_langchain_llm(model_type=self.config.model_type,
                                           config=self.config.llm_config)
        except Exception as e:
            error(f"获取LangChain LLM失败: {str(e)}")
            print_log_exception()
            # 回退方案：创建一个简单的包装器
            class SimpleLLMWrapper:
                def __init__(self, client, model):
                    self.client = client
                    self.model = model

                def invoke(self, messages):
                    # 转换为客户端需要的格式
                    formatted_messages = []
                    for msg in messages:
                        if isinstance(msg, dict):
                            formatted_messages.append(msg)
                        else:
                            # 处理LangChain格式的消息
                            role = "user" if msg.type == "human" else "system"
                            formatted_messages.append({"role": role, "content": msg.content})

                    response = self.client.chat_completion(
                        model=self.model,
                        messages=formatted_messages,
                        temperature=0.3
                    )
                    return response.json()['content'] if isinstance(response.json(), dict) else response.json()

            return SimpleLLMWrapper(self.client, self.config.model_name)

    def _init_memory(self):
        """
        初始化向量记忆系统
        
        Returns:
            VectorStoreRetrieverMemory实例
        """
        try:
            # 首先检查FAISS是否可用
            if not FAISS_AVAILABLE:
                warning("FAISS不可用，直接使用虚拟向量存储作为替代方案")
                # 创建一个空的索引作为后备
                class DummyVectorStore:
                    def as_retriever(self, **kwargs):
                        class DummyRetriever:
                            def get_relevant_documents(self, query):
                                return []
                        return DummyRetriever()
                vectorstore = DummyVectorStore()
                warning("使用虚拟向量存储作为后备")

            else:
                # 创建一个适配层，使llama_index的embedding兼容langchain的FAISS
                class LlamaIndexToLangChainEmbeddingAdapter:
                    def __init__(self, llama_embedding):
                        self.llama_embedding = llama_embedding

                    def embed_documents(self, texts):
                        # 适配llama_index的嵌入方法到langchain的接口，添加异常处理
                        results = []
                        for text in texts:
                            try:
                                # 尝试不同的嵌入方法
                                if hasattr(self.llama_embedding, 'get_text_embedding'):
                                    results.append(self.llama_embedding.get_text_embedding(text))
                                elif callable(self.llama_embedding):
                                    results.append(self.llama_embedding(text))
                                else:
                                    # 返回默认嵌入向量
                                    results.append([0.1] * 768)  # 假设768维向量
                            except Exception as e:
                                warning(f"嵌入文档失败: {str(e)}")
                                results.append([0.1] * 768)
                        return results

                    def embed_query(self, text):
                        # 实现query嵌入方法，确保适配器可调用
                        try:
                            if hasattr(self.llama_embedding, 'get_text_embedding'):
                                return self.llama_embedding.get_text_embedding(text)
                            elif callable(self.llama_embedding):
                                return self.llama_embedding(text)
                            else:
                                return [0.1] * 768
                        except Exception as e:
                            warning(f"嵌入查询失败: {str(e)}")
                            return [0.1] * 768

                    def __call__(self, text):
                        # 添加__call__方法，确保适配器可以直接被调用
                        return self.embed_query(text)

                # 从embedding_client获取嵌入模型
                llama_embedding = get_embedding_client(
                    model_type=self.config.embedding_type,
                    model_name=self.config.embedding_model
                )

                # 创建适配的embedding对象
                embedding = LlamaIndexToLangChainEmbeddingAdapter(llama_embedding)

                # 创建向量存储 - 使用一个默认文档避免空列表错误
                from langchain_core.documents import Document
                default_doc = Document(page_content="Empty memory initialization document")
                
                try:
                    vectorstore = FAISS.from_documents([default_doc], embedding)
                    info("向量存储创建成功")
                except Exception as vec_error:
                    warning(f"创建向量存储失败，尝试使用备选方法: {str(vec_error)}")
                    # 备选方法：直接创建FAISS索引
                    try:
                        # 创建一个简单的嵌入向量
                        import numpy as np
                        from langchain_community.vectorstores.utils import maximal_marginal_relevance
                        
                        # 创建一个空的索引作为后备
                        class DummyVectorStore:
                            def as_retriever(self, **kwargs):
                                class DummyRetriever:
                                    def get_relevant_documents(self, query):
                                        return []
                                return DummyRetriever()
                        vectorstore = DummyVectorStore()
                        warning("使用虚拟向量存储作为后备")
                    except Exception as fallback_error:
                        error(f"创建后备向量存储也失败: {str(fallback_error)}")
                        return None

            # 创建检索器
            retriever = vectorstore.as_retriever(search_kwargs={"k": self.config.memory_top_k})

            # 创建向量记忆
            memory = VectorStoreRetrieverMemory(
                retriever=retriever,
                memory_key="chat_history",
                input_key="input",
                return_docs=True
            )

            debug(f"向量记忆初始化成功，使用模型: {self.config.embedding_model}")
            return memory
        except Exception as e:
            error(f"初始化向量记忆失败: {str(e)}")
            print_log_exception()
            return None

    def _retrieve_knowledge(self, query: str, top_k: int = 3) -> List[str]:
        """
        从知识库检索相关知识
        
        Args:
            query: 检索查询
            top_k: 返回的最大文档数
            
        Returns:
            List[str]: 检索到的知识片段列表
        """
        # 检查检索器是否初始化
        if self.retriever is None:
            debug("检索器未初始化，无法执行知识库检索")
            return []

        try:
            results = self.retriever.retrieve(query, top_k=top_k)
            knowledge = []
            for result in results:
                if hasattr(result, 'text'):
                    knowledge.append(result.text)
                elif hasattr(result, 'content'):
                    knowledge.append(result.content)
                elif isinstance(result, dict) and 'text' in result:
                    knowledge.append(result['text'])
                elif isinstance(result, dict) and 'node' in result and hasattr(result['node'], 'get_content'):
                    # 处理llama_index的NodeWithScore格式
                    knowledge.append(result['node'].get_content())
            return knowledge
        except Exception as e:
            error(f"知识库检索失败: {str(e)}")
            return []

    def _generate_analysis(self, prompt: str, knowledge: List[str] = None) -> Dict[str, Any]:
        """
        使用LLM生成分析结果，统一使用langchain框架的提示词模板
        
        Args:
            prompt: 提示词
            knowledge: 相关知识片段
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            # 构建完整提示词
            knowledge_text = chr(10).join(knowledge) if knowledge else "无"

            # 如果启用了记忆，使用记忆增强的LLM调用
            if self.memory:
                try:
                    # 使用langchain的ChatPromptTemplate创建带记忆的提示模板
                    template = """
                    你是{{description}}。请基于提供的信息和历史对话进行专业、客观的分析。
                    
                    相关知识：
                    {{knowledge_text}}
                    
                    历史对话：
                    {{chat_history}}
                    
                    当前任务：
                    {{input}}
                    
                    请输出JSON格式的分析结果。
                    """

                    prompt_template = ChatPromptTemplate.from_template(template)

                    # 创建链
                    chain = RunnablePassthrough.assign(
                        chat_history=self.memory.load_memory_variables,
                        description=lambda _: self.description,
                        knowledge_text=lambda _: knowledge_text
                    ) | prompt_template | self.langchain_llm | StrOutputParser()

                    # 执行链
                    response_text = chain.invoke({"input": prompt})

                    # 保存到记忆
                    self.memory.save_context(
                        {"input": prompt},
                        {"output": response_text}
                    )

                    # 解析JSON响应
                    import json
                    return json.loads(response_text)
                except Exception as mem_e:
                    warning(f"使用记忆系统失败，回退到普通调用: {str(mem_e)}")

            # 普通调用方式（备用）- 统一使用langchain框架
            template = """
            你是{{description}}。请基于提供的信息进行专业、客观的分析。
            
            相关知识：
            {{knowledge_text}}
            
            任务：
            {{input}}
            
            请输出JSON格式的分析结果。
            """

            # 使用langchain的ChatPromptTemplate
            prompt_template = ChatPromptTemplate.from_template(template)

            # 创建链
            chain = RunnablePassthrough.assign(
                description=lambda _: self.description,
                knowledge_text=lambda _: knowledge_text
            ) | prompt_template | self.langchain_llm | StrOutputParser()

            # 执行链
            response_text = chain.invoke({"input": prompt})

            # 解析JSON响应
            import json
            return json.loads(response_text)

        except Exception as e:
            error(f"LLM分析生成失败: {str(e)}")
            raise

    def add_memory(self, input_text: str, output_text: str):
        """
        添加内容到智能体记忆
        
        Args:
            input_text: 输入文本
            output_text: 输出文本
        """
        if self.memory:
            try:
                self.memory.save_context({"input": input_text}, {"output": output_text})
                debug(f"成功添加内容到智能体记忆")
            except Exception as e:
                error(f"添加记忆失败: {str(e)}")

    def get_memory_context(self, query: str) -> List[str]:
        """
        获取与查询相关的记忆上下文
        
        Args:
            query: 查询文本
            
        Returns:
            List[str]: 相关的记忆内容列表
        """
        if self.memory:
            try:
                memory_vars = self.memory.load_memory_variables({"prompt": query})
                if "chat_history" in memory_vars:
                    # 格式化记忆消息为文本
                    history_texts = []
                    for msg in memory_vars["chat_history"]:
                        if hasattr(msg, 'content'):
                            history_texts.append(f"{msg.type}: {msg.content}")
                    return history_texts
            except Exception as e:
                error(f"获取记忆上下文失败: {str(e)}")
        return []

    def get_result_template(self) -> Dict[str, Any]:
        """
        获取结果模板
        
        Returns:
            Dict[str, Any]: 结果模板
        """
        return {
            "agent_name": self.agent_name,
            "analysis_time": "",
            "confidence_score": 0.0,
            "key_findings": [],
            "detailed_analysis": {}
        }

    def validate_result(self, result: Dict[str, Any]) -> bool:
        """
        验证结果的有效性
        
        Args:
            result: 分析结果
            
        Returns:
            bool: 是否有效
        """
        if not isinstance(result, dict):
            return False

        required_fields = ["agent_name", "key_findings", "detailed_analysis"]
        for field in required_fields:
            if field not in result:
                return False

        return True
