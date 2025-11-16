from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.runnables import RunnablePassthrough

from hengline.logger import debug, info, error
from .vector_store import VectorStoreManager


class RAGChain:
    """检索增强生成(RAG)链，将文档检索与语言模型结合起来"""

    def __init__(self, model, vector_store_manager=None):
        """
        初始化RAG链
        
        Args:
            model: 用于生成回答的语言模型
            vector_store_manager: 向量存储管理器实例
        """
        debug("初始化RAG链...")

        self.model = model
        self.vector_store_manager = vector_store_manager or VectorStoreManager()

        # 初始化向量存储
        self.vector_store_manager.load_vector_store()

        # 创建RAG链
        self.rag_chain = self._create_rag_chain()
        debug("RAG链初始化完成")

    def _create_rag_chain(self):
        """
        创建RAG链
        
        Returns:
            创建的RAG链
        """
        # 定义提示模板
        template = """
        你是一位专业的金融分析师，你的任务是基于提供的上下文信息回答用户的问题。
        
        上下文信息:
        {context}
        
        用户问题:
        {question}
        
        请根据上下文信息，用专业但易懂的语言回答用户的问题。
        如果上下文信息不足以回答问题，请明确说明，并表示无法提供相关信息。
        不要添加上下文信息中没有的内容。
        """

        prompt = ChatPromptTemplate.from_template(template)

        # 定义检索函数
        def retrieve_docs(query):
            docs = self.vector_store_manager.retrieve(query)
            return "\n\n".join([doc.page_content for doc in docs])

        # 创建RAG链
        rag_chain = RunnableMap({
            "context": RunnablePassthrough.assign(query=lambda x: x["question"]) | retrieve_docs,
            "question": RunnablePassthrough()
        }) | prompt | self.model | StrOutputParser()

        return rag_chain

    def invoke(self, question, k=4):
        """
        调用RAG链回答问题
        
        Args:
            question: 用户的问题
            k: 检索的文档数量
            
        Returns:
            模型生成的回答
        """
        # 安全地记录查询文本，避免对非字符串类型进行切片操作
        question_text = str(question)[:50] if isinstance(question, (str, bytes)) else str(question)
        info(f"执行RAG查询: {question_text}...")

        try:
            # 确保问题是字符串类型
            if not isinstance(question, (str, bytes)):
                question = str(question)

            # 更新检索参数
            if k != 4:  # 4是默认值
                # 重新创建RAG链以使用新的k值
                def retrieve_docs_with_k(query):
                    docs = self.vector_store_manager.retrieve(query, k=k)
                    return "\n\n".join([doc.page_content for doc in docs])

                # 定义提示模板
                template = """
                你是一位专业的金融分析师，你的任务是基于提供的上下文信息回答用户的问题。
                
                上下文信息:
                {context}
                
                用户问题:
                {question}
                
                请根据上下文信息，用专业但易懂的语言回答用户的问题。
                如果上下文信息不足以回答问题，请明确说明，并表示无法提供相关信息。
                不要添加上下文信息中没有的内容。
                """

                prompt = ChatPromptTemplate.from_template(template)

                # 创建临时RAG链
                temp_rag_chain = RunnableMap({
                    "context": RunnablePassthrough.assign(query=lambda x: x["question"]) | retrieve_docs_with_k,
                    "question": RunnablePassthrough()
                }) | prompt | self.model | StrOutputParser()

                result = temp_rag_chain.invoke({"question": question})
            else:
                # 使用默认的RAG链
                result = self.rag_chain.invoke({"question": question})

            debug("RAG查询执行成功")
            return result
        except Exception as e:
            error(f"RAG查询执行失败: {str(e)}")
            raise

    def get_relevant_docs(self, question, k=4):
        """
        获取与问题相关的文档
        
        Args:
            question: 用户的问题
            k: 检索的文档数量
            
        Returns:
            相关文档列表
        """
        # 安全地记录查询文本，避免对非字符串类型进行切片操作
        question_text = str(question)[:50] if isinstance(question, (str, bytes)) else str(question)
        info(f"检索相关文档，查询: {question_text}..., 数量: {k}")

        try:
            # 确保问题是字符串类型
            if not isinstance(question, (str, bytes)):
                question = str(question)

            # 从向量存储中检索相关文档
            docs = self.vector_store_manager.retrieve(question, k=k)

            debug(f"检索到 {len(docs)} 篇相关文档")
            return docs
        except Exception as e:
            error(f"文档检索失败: {str(e)}")
            raise

    def update_vector_store(self):
        """
        更新向量存储
        
        Returns:
            更新后的向量存储
        """
        return self.vector_store_manager.update_vector_store()
