from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import os
import shutil
from dotenv import load_dotenv

# 导入日志工具
from utils.logger import get_logger
logger = get_logger('vector_store')

# 加载环境变量
load_dotenv()

class VectorStoreManager:
    """向量存储管理器，用于加载文档、创建向量存储和进行检索"""
    
    def __init__(self, vector_store_path=None, embedding_model=None):
        """
        初始化向量存储管理器
        
        Args:
            vector_store_path: 向量存储的保存路径
            embedding_model: 使用的嵌入模型名称
        """
        self.vector_store_path = vector_store_path or os.getenv('VECTOR_STORE_PATH', 'data/vector_store')
        self.embedding_model = embedding_model or os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '200'))
        
        # 初始化嵌入模型
        try:
            # 首先尝试不强制本地文件，允许自动下载模型（如果有网络连接）
            logger.debug(f"尝试加载嵌入模型: {self.embedding_model}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True},
                cache_folder=os.getenv('CACHE_DIR', './cache')
                # 不设置local_files_only=True，允许自动下载
            )
            logger.debug(f"成功加载嵌入模型: {self.embedding_model}")
        except Exception as e:
            # 如果模型加载失败，提供友好的错误消息和指导
            error_info = f"\n{'='*80}\n" + \
                        f"错误: 无法加载嵌入模型: {str(e)}\n" + \
                        f"模型名称: {self.embedding_model}\n" +\
                        "\n可能的原因:\n" +\
                        "1. 没有网络连接，无法下载模型\n" +\
                        "2. 本地缓存中没有预先下载的模型文件\n" +\
                        "3. 模型名称不正确或模型不存在\n" +\
                        "\n解决方案:\n" +\
                        "1. 确保您已连接到互联网，首次运行时会自动下载模型\n" +\
                        "2. 或在有网络的环境下使用download_embedding_model.py脚本预先下载模型\n" +\
                        "3. 或在.env文件中设置OFFLINE_MODE=true以强制使用离线模式\n" +\
                        "4. 或设置一个已经下载到本地的模型路径到EMBEDDING_MODEL环境变量\n" +\
                        f"{'='*80}"
            logger.error(error_info)
            
            # 尝试在离线模式下创建一个简单的本地嵌入模型
            try:
                logger.debug("尝试创建简单的本地嵌入模型...")
                
                # 首先尝试使用GPT4AllEmbeddings
                try:
                    from langchain_community.embeddings import GPT4AllEmbeddings
                    self.embeddings = GPT4AllEmbeddings()
                    logger.debug("成功创建GPT4AllEmbeddings作为替代")
                except Exception as gpt4all_err:
                    logger.warning(f"GPT4AllEmbeddings创建失败: {str(gpt4all_err)}")
                    
                    # 如果GPT4All也失败，尝试使用另一个可能的替代方案
                    try:
                        from langchain_community.embeddings import OllamaEmbeddings
                        # 尝试使用本地Ollama服务器提供的嵌入功能
                        self.embeddings = OllamaEmbeddings(
                            model=os.getenv('OLLAMA_MODEL', 'llama3'),
                            base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
                        )
                        logger.debug("成功创建OllamaEmbeddings作为替代")
                    except Exception as ollama_err:
                        logger.warning(f"OllamaEmbeddings创建失败: {str(ollama_err)}")
                        
                        # 如果所有替代方案都失败，创建一个更强大的模拟嵌入类
                        logger.warning("无法创建替代嵌入模型。创建增强的模拟嵌入模型...")
                        from langchain_core.embeddings import Embeddings
                        import hashlib
                        import numpy as np
                        
                        class EnhancedMockEmbeddings(Embeddings):
                            """增强的模拟嵌入类，提供更稳定的结果"""
                            def __init__(self, embedding_dim=384):
                                self.embedding_dim = embedding_dim
                                logger.debug(f"创建增强模拟嵌入模型 (维度: {embedding_dim})")
                                
                            def _get_hash_vector(self, text):
                                """基于文本哈希生成确定性向量"""
                                # 使用哈希算法生成固定长度的向量
                                hash_obj = hashlib.md5(text.encode('utf-8'))
                                hash_bytes = hash_obj.digest()
                                
                                # 从哈希值创建浮点数向量
                                vector = []
                                for i in range(0, min(len(hash_bytes), self.embedding_dim)):
                                    vector.append(float(hash_bytes[i]) / 255.0)
                                
                                # 如果向量长度不足，用随机数填充，但保持确定性
                                if len(vector) < self.embedding_dim:
                                    # 使用文本本身作为随机种子
                                    seed = int.from_bytes(hash_obj.digest(), byteorder='big')
                                    np.random.seed(seed)
                                    additional_values = np.random.rand(self.embedding_dim - len(vector)).tolist()
                                    vector.extend(additional_values)
                                
                                return vector
                                
                            def embed_documents(self, texts):
                                """为文档列表生成嵌入向量"""
                                return [self._get_hash_vector(text) for text in texts]
                            
                            def embed_query(self, text):
                                """为查询文本生成嵌入向量"""
                                return self._get_hash_vector(text)
                            
                        self.embeddings = EnhancedMockEmbeddings()
                        logger.debug("已创建增强的模拟嵌入模型，用于在离线环境下提供基本功能")
            except Exception as e:
                logger.error(f"创建替代嵌入模型时出现意外错误: {str(e)}")
                # 作为最后的后备方案，创建一个非常简单的模拟嵌入
                from langchain_core.embeddings import Embeddings
                class MinimalMockEmbeddings(Embeddings):
                    def embed_documents(self, texts):
                        import random
                        return [[random.random() for _ in range(384)] for _ in texts]
                    
                    def embed_query(self, text):
                        import random
                        return [random.random() for _ in range(384)]
                
                self.embeddings = MinimalMockEmbeddings()
                logger.debug("已创建最小化的模拟嵌入模型")
        
        # 初始化向量存储
        self.vector_store = None
    
    def load_documents(self, data_dir='data'):
        """
        从目录加载文档
        
        Args:
            data_dir: 数据目录路径
            
        Returns:
            加载的文档列表
        """
        # 在较新版本的LangChain中，encoding参数应传递给TextLoader而不是DirectoryLoader
        # 使用recursive=True递归加载所有子目录中的txt文件
        loader = DirectoryLoader(
            data_dir,
            glob="**/*.txt",  # 递归匹配所有子目录中的txt文件
            loader_cls=lambda path: TextLoader(path, encoding='utf-8'),
            recursive=True  # 递归加载子目录
        )
        documents = loader.load()
        logger.debug(f"加载了 {len(documents)} 个文档")
        
        # 确保至少加载了一个文档
        if not documents:
            logger.warning(f"在 {data_dir} 目录下未找到任何txt文件")
        
        return documents
    
    def split_documents(self, documents):
        """
        将文档分割成小块
        
        Args:
            documents: 文档列表
            
        Returns:
            分割后的文档块列表
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        logger.debug(f"正在将 {len(documents)} 个文档分割成小块...")
        splits = text_splitter.split_documents(documents)
        logger.debug(f"分割完成，共生成 {len(splits)} 个文档块")
        return splits
    
    def create_vector_store(self, splits):
        """
        创建向量存储
        
        Args:
            splits: 文档块列表
            
        Returns:
            创建的向量存储
        """
        # 创建向量存储
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        
        # 保存向量存储
        if not os.path.exists(self.vector_store_path):
            os.makedirs(self.vector_store_path)
        self.vector_store.save_local(self.vector_store_path)
        logger.debug(f"向量存储已保存到 {self.vector_store_path}")
        
        return self.vector_store
    
    def load_vector_store(self):
        """
        加载已保存的向量存储
        
        Returns:
            加载的向量存储
        """
        # 检查向量存储目录是否存在以及是否包含必要的文件
        index_faiss_path = os.path.join(self.vector_store_path, 'index.faiss')
        index_pkl_path = os.path.join(self.vector_store_path, 'index.pkl')
        
        if os.path.exists(index_faiss_path) and os.path.exists(index_pkl_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.debug(f"向量存储已从 {self.vector_store_path} 加载")
            except Exception as e:
                logger.error(f"加载向量存储时出错: {str(e)}")
                logger.debug("将重新创建向量存储...")
                # 尝试重新创建向量存储
                documents = self.load_documents()
                if documents:
                    splits = self.split_documents(documents)
                    self.create_vector_store(splits)
        else:
            logger.warning(f"向量存储文件不存在: {self.vector_store_path}")
            # 确保向量存储目录存在
            if not os.path.exists(self.vector_store_path):
                os.makedirs(self.vector_store_path)
                logger.debug(f"已创建向量存储目录: {self.vector_store_path}")
            
            # 尝试从数据目录创建向量存储
            documents = self.load_documents()
            if documents:
                splits = self.split_documents(documents)
                self.create_vector_store(splits)
            else:
                logger.warning("没有找到文档来创建向量存储。请确保data目录下有文本文件。")
                # 创建一个空的向量存储作为占位符
                self._create_empty_vector_store()
        
        return self.vector_store
        
    def _create_empty_vector_store(self):
        """
        创建一个空的向量存储作为占位符
        """
        try:
            # 创建一个简单的文档来初始化向量存储
            from langchain_core.documents import Document
            empty_docs = [Document(page_content="占位文档，用于初始化向量存储", metadata={"source": "empty"})]
            
            # 使用模拟嵌入或实际嵌入创建向量存储
            if hasattr(self.embeddings, 'embed_documents'):
                self.vector_store = FAISS.from_documents(empty_docs, self.embeddings)
                self.vector_store.save_local(self.vector_store_path)
                logger.debug(f"已创建空向量存储作为占位符: {self.vector_store_path}")
        except Exception as e:
            logger.error(f"创建空向量存储时出错: {str(e)}")
            logger.debug("将创建一个简单的模拟向量存储类")
            
            # 创建一个模拟向量存储类
            class MockVectorStore:
                def similarity_search(self, query, k=4):
                    return []
                
                def save_local(self, path):
                    pass
            
            self.vector_store = MockVectorStore()
    
    def retrieve(self, query, k=4):
        """
        检索与查询相关的文档
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            相关文档列表
        """
        if not self.vector_store:
            logger.debug("向量存储未初始化，正在加载...")
            self.load_vector_store()
        
        if self.vector_store:
            # 安全地记录查询文本，避免对非字符串类型进行切片操作
            query_text = str(query)[:50] if isinstance(query, (str, bytes)) else str(query)
            logger.debug(f"正在进行相似度搜索，查询: {query_text}...")
            
            # 确保查询是字符串类型
            if not isinstance(query, (str, bytes)):
                query = str(query)
                logger.debug("查询参数已转换为字符串类型")
                
            relevant_docs = self.vector_store.similarity_search(query, k=k)
            logger.debug(f"检索完成，找到 {len(relevant_docs)} 个相关文档")
            return relevant_docs
        else:
            logger.warning("向量存储未初始化，无法进行检索")
            return []
    
    def update_vector_store(self, data_dir='data'):
        """
        更新向量存储
        
        Args:
            data_dir: 数据目录路径
            
        Returns:
            更新后的向量存储
        """
        logger.debug(f"开始更新向量存储，数据目录: {data_dir}")
        documents = self.load_documents(data_dir)
        splits = self.split_documents(documents)
        vector_store = self.create_vector_store(splits)
        logger.debug("向量存储更新完成")
        return vector_store

# 示例用法
if __name__ == "__main__":
    # 初始化向量存储管理器
    vector_store_manager = VectorStoreManager()
    
    # 加载或创建向量存储
    vector_store = vector_store_manager.load_vector_store()
    
    # 如果向量存储不存在或需要更新，则创建或更新
    if not vector_store or input("是否更新向量存储？(y/n): ").lower() == 'y':
        vector_store_manager.update_vector_store()
    
    # 测试检索功能
    query = "什么是股票的市盈率？"
    relevant_docs = vector_store_manager.retrieve(query)
    
    logger.debug(f"\n检索到的相关文档 ({len(relevant_docs)}):")
    for i, doc in enumerate(relevant_docs):
        logger.debug(f"\n文档 {i+1}: ")
        logger.debug(f"内容: {doc.page_content[:200]}...")
        if 'source' in doc.metadata:
            logger.debug(f"来源: {doc.metadata['source']}")
        else:
            logger.debug("来源: 未知")