#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LlamaIndex 工具函数模块
提供向量存储创建、嵌入模型获取等实用功能
"""

import os
from typing import Optional, List, Any

# 设置环境变量以避免默认使用OpenAI
os.environ["OPENAI_API_KEY"] = "sk-dummy-key"  # 设置一个占位符，防止系统尝试使用真实的OpenAI API

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.embeddings import BaseEmbedding, resolve_embed_model
from llama_index.core.storage import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore

from config.config import get_data_embeddings_path, get_embedding_config
from hengline.logger import info, debug, error, warning


def get_local_embedding_model(embedding_config=None):
    """获取本地嵌入模型，优先使用Ollama，然后使用本地模型"""
    if embedding_config is None:
        embedding_config = get_embedding_config()
    
    provider = embedding_config.get('provider', 'ollama')
    model = embedding_config.get('model_name', 'quentinz/bge-large-zh-v1.5:latest')
    
    try:
        if provider == 'ollama':
            from llama_index.embeddings.ollama import OllamaEmbedding
            embedding_model = OllamaEmbedding(
                model_name=model,
                base_url=embedding_config.get('base_url', 'http://localhost:11434'),
                request_timeout=embedding_config.get('timeout', 60)
            )
            info(f"使用Ollama嵌入模型: {model}")
        else:
            # 尝试使用local嵌入模型
            embedding_model = resolve_embed_model('local')
            info("使用本地嵌入模型")
        return embedding_model
    except Exception as e:
        warning(f"无法创建嵌入模型，将使用null嵌入模型: {str(e)}")
        # 尝试使用一个非常简单的null嵌入模型作为最后的后备
        try:
            return resolve_embed_model('null')  # 这是一个无操作模型，不会实际生成嵌入
        except:
            # 如果null也不可用，返回None
            warning("null嵌入模型也不可用，将使用None")
            return None


def create_vector_store(
        documents: Optional[List] = None,
        index_name: str = "default_index",
        storage_dir: Optional[str] = None,
        embedding_model: Optional[BaseEmbedding] = None,
        rebuild: bool = False
) -> VectorStoreIndex:
    # 获取嵌入模型配置
    embedding_config = get_embedding_config()
    provider = embedding_config.get('provider', 'ollama')
    model = embedding_config.get('model_name', 'bge-large-zh-v1.5')
    """
    创建或加载向量存储索引
    
    Args:
        documents: 要索引的文档列表，如果为None则尝试加载现有索引
        index_name: 索引名称
        storage_dir: 存储目录路径
        embedding_model: 嵌入模型实例
        rebuild: 是否重建索引，即使已存在也会删除旧索引
        
    Returns:
        VectorStoreIndex实例
    """
    try:
        # 获取存储目录路径
        if storage_dir is None:
            storage_dir = get_data_embeddings_path()
        
        # 确保存储目录存在 - 这是关键修复点
        storage_dir = os.path.normpath(storage_dir)
        
        # 如果提供了存储目录，配置存储上下文
        storage_context = None

        if storage_dir:
            try:
                # 尝试从持久化目录加载，如果失败则创建新的存储
                vector_store = SimpleVectorStore()
                doc_store = SimpleDocumentStore()
                index_store = SimpleIndexStore()
                
                # 只有当不重建且目录存在时才尝试加载
                if not rebuild and os.path.exists(storage_dir):
                    # 分别尝试加载每个组件，如果加载失败则使用空实例
                    try:
                        vector_store_file = os.path.join(storage_dir, "vector_store.json")
                        if os.path.exists(vector_store_file):
                            vector_store = SimpleVectorStore.from_persist_dir(storage_dir)
                            info("向量存储加载成功")
                        else:
                            debug("向量存储文件不存在，将创建新的")
                    except Exception as e:
                        debug(f"无法加载向量存储，创建新的: {str(e)}")
                        vector_store = SimpleVectorStore()
                    
                    try:
                        doc_store = SimpleDocumentStore.from_persist_dir(storage_dir)
                    except Exception as e:
                        debug(f"无法加载文档存储，创建新的: {str(e)}")
                        doc_store = SimpleDocumentStore()
                    
                    try:
                        index_store = SimpleIndexStore.from_persist_dir(storage_dir)
                    except Exception as e:
                        debug(f"无法加载索引存储，创建新的: {str(e)}")
                        index_store = SimpleIndexStore()
                else:
                    info("重建索引或目录不存在，创建新的存储组件")
                    vector_store = SimpleVectorStore()
                    doc_store = SimpleDocumentStore()
                    index_store = SimpleIndexStore()

                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store,
                    docstore=doc_store,
                    index_store=index_store
                )
            except Exception as e:
                debug(f"创建存储上下文时出错，使用默认配置: {str(e)}")
                # 如果出现问题，使用默认存储上下文
                storage_context = StorageContext.from_defaults(
                    vector_store=SimpleVectorStore(),
                    docstore=SimpleDocumentStore(),
                    index_store=SimpleIndexStore()
                )

        # 如果需要重建、有文档，从文档创建新索引
        if documents or rebuild:
            debug(f"需要创建新索引: rebuild={rebuild}, 文档数量={len(documents) if documents else 0}")
            
            # 确保使用本地嵌入模型，避免依赖OpenAI
            if embedding_model is None:
                embedding_model = get_local_embedding_model()
            
            # 准备文档列表
            if documents is None:
                documents = []
                info("没有提供文档，创建空索引")

            # 创建索引 - 添加额外的错误处理
            try:
                # 对于空文档列表，直接创建空索引
                if not documents:
                    info("创建空向量存储索引")
                    # 创建存储上下文
                    storage_context = StorageContext.from_defaults(vector_store=SimpleVectorStore())
                    # 创建空索引
                    index = VectorStoreIndex(
                        [],  # 空文档列表
                        storage_context=storage_context,
                        embed_model=embedding_model
                    )
                else:
                    info(f"从 {len(documents)} 个文档创建向量存储索引")
                    index = VectorStoreIndex.from_documents(
                        documents,
                        storage_context=storage_context,
                        embed_model=embedding_model,
                        show_progress=True
                    )
                info("向量存储索引创建成功")
            except Exception as create_error:
                error(f"创建索引失败: {str(create_error)}")
                debug(f"异常详情: {str(create_error)}")
                
                # 尝试使用更简单的方式创建空索引
                try:
                    info("尝试创建最小空索引")
                    # 使用null嵌入模型创建最基本的索引
                    try:
                        null_embed_model = resolve_embed_model('null')  # 使用无操作嵌入模型
                    except:
                        null_embed_model = None  # 如果null不可用，使用None
                    index = VectorStoreIndex([], embed_model=null_embed_model)
                    info("最小空索引创建成功")
                except Exception as fallback_error:
                    error(f"创建空索引也失败: {str(fallback_error)}")
                    # 如果所有尝试都失败，返回一个标记对象
                    class DummyVectorStore:
                        """虚拟向量存储对象，用于处理创建失败的情况"""
                        def __init__(self):
                            self.vector_store = None
                    return DummyVectorStore()

            # 保存索引到存储目录
            if storage_dir:
                try:
                    index.storage_context.persist(persist_dir=storage_dir)
                    info(f"索引已保存到: {storage_dir}")
                    
                    # 验证保存是否成功
                    saved_file = os.path.join(storage_dir, "vector_store.json")
                    if os.path.exists(saved_file):
                        info(f"验证: 向量存储文件已保存: {saved_file}")
                    else:
                        warning(f"验证: 向量存储文件似乎未保存: {saved_file}")
                except Exception as e:
                    warning(f"保存索引失败，但索引已创建: {str(e)}")

            return index
        # 如果有存储目录但没有文档，尝试从存储加载现有索引
        elif storage_dir:
            try:
                debug(f"尝试从存储目录加载向量存储索引: {storage_dir}")
                # 检查是否有实际的向量存储内容
                vector_store_path = os.path.join(storage_dir, "vector_store.json")
                if os.path.exists(vector_store_path) and os.path.getsize(vector_store_path) > 0:
                    # 确保使用本地嵌入模型
                    if embedding_model is None:
                        embedding_model = get_local_embedding_model()
                    index = VectorStoreIndex.from_vector_store(
                        storage_context.vector_store,
                        storage_context=storage_context,
                        embed_model=embedding_model
                    )
                    info(f"成功加载向量存储索引: {index_name}")
                    return index
                else:
                    debug("向量存储文件不存在或为空，创建空索引")
                    # 确保使用本地嵌入模型
                    if embedding_model is None:
                        embedding_model = get_local_embedding_model()
                    # 创建空索引
                    index = VectorStoreIndex.from_documents(
                        [],
                        storage_context=storage_context,
                        embed_model=embedding_model
                    )
                    # 保存空索引到存储
                    if storage_dir:
                        try:
                            index.storage_context.persist(persist_dir=storage_dir)
                            info(f"空索引已保存到: {storage_dir}")
                        except Exception as e:
                            warning(f"保存空索引失败: {str(e)}")
                    return index
            except Exception as e:
                debug(f"加载现有索引失败，创建空索引: {str(e)}")
                # 创建空索引作为后备
                try:
                    # 确保使用本地嵌入模型
                    if embedding_model is None:
                        embedding_model = get_local_embedding_model()
                    index = VectorStoreIndex.from_documents(
                        [],
                        storage_context=storage_context,
                        embed_model=embedding_model
                    )
                except Exception as fallback_e:
                    # 尝试使用null嵌入模型
                    info("使用null嵌入模型作为最后后备")
                    try:
                        null_embed_model = resolve_embed_model('null')
                    except:
                        null_embed_model = None
                    index = VectorStoreIndex([], embed_model=null_embed_model)
                return index
        else:
            # 没有存储目录且没有文档，创建临时索引
            debug("没有存储目录和文档，创建临时索引")
            try:
                # 确保使用本地嵌入模型
                if embedding_model is None:
                    embedding_model = get_local_embedding_model()
                index = VectorStoreIndex.from_documents(
                    [],
                    storage_context=storage_context,
                    embed_model=embedding_model
                )
            except Exception as temp_error:
                # 使用null嵌入模型作为后备
                info("使用null嵌入模型创建临时索引")
                try:
                    null_embed_model = resolve_embed_model('null')
                except:
                    null_embed_model = None
                index = VectorStoreIndex([], embed_model=null_embed_model)
            return index

    except Exception as e:
        error(f"创建向量存储索引失败: {str(e)}")
        debug(f"异常类型: {type(e).__name__}")
        import traceback
        debug(f"堆栈跟踪: {traceback.format_exc()}")
        # 如果所有尝试都失败，返回一个最小的空索引
        try:
            warning("创建空索引作为最终后备方案")
            # 使用null嵌入模型确保不依赖OpenAI
            try:
                null_embed_model = resolve_embed_model('null')
            except:
                null_embed_model = None
            return VectorStoreIndex([], embed_model=null_embed_model)
        except Exception as final_error:
            error(f"创建空索引也失败: {str(final_error)}")
            raise


def create_index_from_directory(
        directory_path: str,
        index_name: str = "directory_index",
        storage_dir: Optional[str] = get_data_embeddings_path(),
        embedding_model: Optional[BaseEmbedding] = None,
        recursive: bool = True,
        required_exts: Optional[List[str]] = None,
        rebuild: bool = False
) -> VectorStoreIndex:
    """
    从目录创建向量存储索引
    
    Args:
        directory_path: 包含文档的目录路径
        index_name: 索引名称
        storage_dir: 存储目录路径
        embedding_model: 嵌入模型实例
        recursive: 是否递归加载子目录
        required_exts: 必需的文件扩展名列表
        rebuild: 是否重建索引
        
    Returns:
        VectorStoreIndex实例
    """
    try:
        # 先检查是否可以直接加载现有索引
        if storage_dir and os.path.exists(storage_dir) and not rebuild:
            try:
                return create_vector_store(
                    index_name=index_name,
                    storage_dir=storage_dir,
                    embedding_model=embedding_model
                )
            except Exception as e:
                debug(f"加载现有索引失败，将创建新索引: {str(e)}")

        # 加载目录中的文档
        debug(f"从目录加载文档: {directory_path}")
        
        # 检查目录是否存在
        if not os.path.exists(directory_path):
            raise ValueError(f"目录不存在: {directory_path}")
        
        # 尝试创建文档加载器并加载文档，处理可能的空目录情况
        documents = []
        try:
            loader = SimpleDirectoryReader(
                input_dir=directory_path,
                recursive=recursive,
                required_exts=required_exts
            )

            documents = loader.load_data()
            
            # 处理空目录情况
            if not documents:
                info(f"目录中没有找到文档: {directory_path}")
            else:
                info(f"从目录加载了{len(documents)}个文档")
        except ValueError as ve:
            # 捕获目录为空的错误
            if "No files found" in str(ve):
                info(f"目录为空，将创建空索引: {directory_path}")
            else:
                raise
        except Exception as load_error:
            warning(f"加载文档时出错，但将继续创建空索引: {str(load_error)}")

        # 创建向量存储索引，即使文档为空也创建索引
        return create_vector_store(
            documents=documents if documents else None,
            index_name=index_name,
            storage_dir=storage_dir,
            embedding_model=embedding_model,
            rebuild=rebuild or bool(documents)
        )

    except Exception as e:
        error(f"从目录创建索引失败: {str(e)}")
        debug(f"异常类型: {type(e).__name__}")
        import traceback
        debug(f"堆栈跟踪: {traceback.format_exc()}")
        
        # 作为后备，尝试创建空索引
        try:
            warning("尝试创建空索引作为后备方案")
            return create_vector_store(
                documents=None,
                index_name=index_name,
                storage_dir=storage_dir,
                embedding_model=embedding_model,
                rebuild=True
            )
        except Exception as fallback_error:
            error(f"创建后备索引也失败: {str(fallback_error)}")
            raise


def get_retriever_from_index(
        index: VectorStoreIndex,
        similarity_top_k: int = 3,
        search_type: str = "similarity",
        **kwargs
) -> Any:
    """
    从索引获取检索器
    
    Args:
        index: 向量存储索引
        similarity_top_k: 返回的最相似文档数量
        search_type: 搜索类型，支持 "similarity", "mmr"
        **kwargs: 额外参数
        
    Returns:
        检索器实例
    """
    try:
        debug(f"获取检索器: search_type={search_type}, top_k={similarity_top_k}")
        
        # 添加错误处理，确保检索器可以正常工作
        try:
            if search_type == "mmr":
                # 使用最大边际相关性搜索
                retriever = index.as_retriever(
                    retriever_mode="mmr",
                    similarity_top_k=similarity_top_k,
                    **kwargs
                )
            else:
                # 默认使用相似度搜索
                retriever = index.as_retriever(
                    retriever_mode="similarity",
                    similarity_top_k=similarity_top_k,
                    **kwargs
                )
            return retriever
        except Exception as retriever_error:
            # 如果常规检索器创建失败，尝试使用更简单的配置
            warning(f"常规检索器创建失败，尝试使用简化配置: {str(retriever_error)}")
            # 使用最小配置创建检索器
            retriever = index.as_retriever(
                retriever_mode="similarity",
                similarity_top_k=similarity_top_k
            )
            return retriever

    except Exception as e:
        error(f"初始化检索器失败: {str(e)}")
        # 创建一个简单的虚拟检索器作为后备
        class DummyRetriever:
            """虚拟检索器，在检索失败时返回空结果"""
            def retrieve(self, query):
                return []
        
        warning("返回虚拟检索器作为后备")
        return DummyRetriever()
