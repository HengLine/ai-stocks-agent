# 完全使用模拟实现，避免langgraph导入冲突

# 模拟MemorySaver类
class MockMemorySaver:
    def __init__(self):
        self.memories = {}
    
    def get(self, **kwargs):
        key = frozenset(kwargs.items())
        return self.memories.get(key)
    
    def put(self, value, **kwargs):
        key = frozenset(kwargs.items())
        self.memories[key] = value

# 模拟Graph类
class MockGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.entry_point = None
        self.memory = MockMemorySaver()
        
    def add_node(self, name, func):
        self.nodes[name] = func
        
    def add_edge(self, from_node, to_node):
        if from_node not in self.edges:
            self.edges[from_node] = []
        self.edges[from_node].append(to_node)
        
    def set_entry_point(self, node_name):
        self.entry_point = node_name
        
    def compile(self):
        return self

# 定义常量和模拟对象
Graph = MockGraph
END = "END"
MemorySaver = MockMemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from utils.logger import get_logger

# 加载环境变量
load_dotenv()

# 创建日志实例
logger = get_logger('workflow')

class StockAnalysisWorkflow:
    """股票分析工作流，使用LangGraph构建多代理协作的股票分析系统"""
    
    def __init__(self, model_selector):
        """
        初始化股票分析工作流
        
        Args:
            model_selector: 模型选择器实例
        """
        self.model_selector = model_selector
        self.workflow = None
        
        # 初始化工作流
        logger.info("初始化股票分析工作流")
        self._build_workflow()
        logger.debug("股票分析工作流初始化完成")
    
    def _build_workflow(self):
        """\构建LangGraph工作流"""
        # 创建内存检查点
        memory = MemorySaver()
        
        # 创建工作流图
        workflow = Graph()
        
        # 添加节点
        workflow.add_node("query_analyzer", self._query_analyzer)
        workflow.add_node("fundamental_analyst", self._fundamental_analyst)
        workflow.add_node("technical_analyst", self._technical_analyst)
        workflow.add_node("market_sentiment_analyst", self._market_sentiment_analyst)
        workflow.add_node("report_generator", self._report_generator)
        
        # 添加边
        workflow.set_entry_point("query_analyzer")
        workflow.add_edge("query_analyzer", "fundamental_analyst")
        workflow.add_edge("fundamental_analyst", "technical_analyst")
        workflow.add_edge("technical_analyst", "market_sentiment_analyst")
        workflow.add_edge("market_sentiment_analyst", "report_generator")
        workflow.add_edge("report_generator", END)
        
        # 编译工作流
        self.workflow = workflow.compile(checkpointer=memory)
    
    def _query_analyzer(self, state):
        """
        分析用户查询，确定需要分析的股票和分析类型
        
        Args:
            state: 当前工作流状态
            
        Returns:
            更新后的状态
        """
        logger.debug("[查询分析器] 正在分析用户查询...")
        
        # 获取用户查询
        query = state.get("query", "")
        
        # 使用模型分析查询
        model = self.model_selector.get_model()
        prompt = ChatPromptTemplate.from_template(
            """
            你是一个查询分析器，你的任务是分析用户的股票相关查询，并提取出需要分析的股票代码或名称，以及用户希望进行的分析类型。
            
            用户查询: {query}
            
            请以JSON格式输出分析结果，包含以下字段:
            - stock: 需要分析的股票代码或名称，如果无法确定，请设为"未知"
            - analysis_type: 分析类型，可以是"基本面分析"、"技术分析"、"市场情绪分析"或"综合分析"
            - requirements: 用户的具体需求描述
            """
        )
        
        chain = prompt | model | StrOutputParser()
        result = chain.invoke({"query": query})
        
        # 更新状态
        state["query_analysis"] = result
        logger.debug(f"[查询分析器] 分析结果: {result}")
        
        return state
    
    def _fundamental_analyst(self, state):
        """
        基本面分析师节点
        
        Args:
            state: 当前工作流状态
            
        Returns:
            更新后的状态
        """
        logger.debug("[基本面分析师] 正在进行基本面分析...")
        
        # 获取查询分析结果
        query_analysis = state.get("query_analysis", "")
        
        # 使用模型进行基本面分析
        model = self.model_selector.get_model()
        prompt = ChatPromptTemplate.from_template(
            """
            你是一位资深的基本面分析师，你的任务是根据提供的查询分析结果，对股票进行基本面分析。
            
            查询分析结果: {query_analysis}
            
            请提供以下方面的分析:
            1. 公司业务和行业地位
            2. 财务状况分析（收入、利润、资产负债等）
            3. 估值分析（市盈率、市净率等）
            4. 竞争优势和风险
            
            请用专业但清晰的语言进行分析，如果信息不足，请说明需要哪些额外信息。
            """
        )
        
        chain = prompt | model | StrOutputParser()
        fundamental_analysis = chain.invoke({"query_analysis": query_analysis})
        
        # 更新状态
        state["fundamental_analysis"] = fundamental_analysis
        logger.debug("[基本面分析师] 分析完成")
        
        return state
    
    def _technical_analyst(self, state):
        """
        技术分析师节点
        
        Args:
            state: 当前工作流状态
            
        Returns:
            更新后的状态
        """
        logger.debug("[技术分析师] 正在进行技术分析...")
        
        # 获取查询分析结果
        query_analysis = state.get("query_analysis", "")
        
        # 使用模型进行技术分析
        model = self.model_selector.get_model()
        prompt = ChatPromptTemplate.from_template(
            """
            你是一位资深的技术分析师，你的任务是根据提供的查询分析结果，对股票进行技术分析。
            
            查询分析结果: {query_analysis}
            
            请提供以下方面的分析:
            1. 价格走势和趋势判断
            2. 关键技术指标分析（如均线、RSI、MACD等）
            3. 支撑位和阻力位
            4. 技术形态识别
            5. 短期和中期技术展望
            
            请用专业但清晰的语言进行分析，如果信息不足，请说明需要哪些额外信息。
            """
        )
        
        chain = prompt | model | StrOutputParser()
        technical_analysis = chain.invoke({"query_analysis": query_analysis})
        
        # 更新状态
        state["technical_analysis"] = technical_analysis
        logger.debug("[技术分析师] 分析完成")
        
        return state
    
    def _market_sentiment_analyst(self, state):
        """
        市场情绪分析师节点
        
        Args:
            state: 当前工作流状态
            
        Returns:
            更新后的状态
        """
        logger.debug("[市场情绪分析师] 正在进行市场情绪分析...")
        
        # 获取查询分析结果
        query_analysis = state.get("query_analysis", "")
        
        # 使用模型进行市场情绪分析
        model = self.model_selector.get_model()
        prompt = ChatPromptTemplate.from_template(
            """
            你是一位市场情绪分析师，你的任务是根据提供的查询分析结果，对股票的市场情绪进行分析。
            
            查询分析结果: {query_analysis}
            
            请提供以下方面的分析:
            1. 市场关注度和交易量变化
            2. 投资者情绪指标
            3. 新闻和社交媒体情绪
            4. 机构持仓变化
            5. 市场情绪对股价的潜在影响
            
            请用专业但清晰的语言进行分析，如果信息不足，请说明需要哪些额外信息。
            """
        )
        
        chain = prompt | model | StrOutputParser()
        sentiment_analysis = chain.invoke({"query_analysis": query_analysis})
        
        # 更新状态
        state["sentiment_analysis"] = sentiment_analysis
        logger.debug("[市场情绪分析师] 分析完成")
        
        return state
    
    def _report_generator(self, state):
        """
        报告生成器节点
        
        Args:
            state: 当前工作流状态
            
        Returns:
            更新后的状态
        """
        logger.debug("[报告生成器] 正在生成综合分析报告...")
        
        # 获取各分析师的分析结果
        query_analysis = state.get("query_analysis", "")
        fundamental_analysis = state.get("fundamental_analysis", "")
        technical_analysis = state.get("technical_analysis", "")
        sentiment_analysis = state.get("sentiment_analysis", "")
        
        # 使用模型生成综合报告
        model = self.model_selector.get_model()
        prompt = ChatPromptTemplate.from_template(
            """
            你是一位高级金融分析师，你的任务是根据各专业分析师的分析结果，生成一份综合的股票分析报告。
            
            查询分析结果: {query_analysis}
            
            基本面分析:
            {fundamental_analysis}
            
            技术分析:
            {technical_analysis}
            
            市场情绪分析:
            {sentiment_analysis}
            
            请生成一份结构清晰、专业且易懂的综合分析报告，包含以下部分:
            1. 股票概况
            2. 各方面分析摘要
            3. 综合结论
            4. 投资建议（如适用）
            
            请确保报告逻辑连贯，内容全面，并在信息不足的部分明确说明。
            """
        )
        
        chain = prompt | model | StrOutputParser()
        report = chain.invoke({
            "query_analysis": query_analysis,
            "fundamental_analysis": fundamental_analysis,
            "technical_analysis": technical_analysis,
            "sentiment_analysis": sentiment_analysis
        })
        
        # 更新状态
        state["final_report"] = report
        logger.debug("[报告生成器] 报告生成完成")
        
        return state
    
    def invoke(self, query, config=None):
        """
        调用工作流进行股票分析
        
        Args:
            query: 用户的查询
            config: 工作流配置
            
        Returns:
            工作流执行结果
        """
        # 如果没有提供配置，创建一个默认配置
        if config is None:
            config = {"configurable": {"thread_id": "stock_analysis_thread_1"}}
        
        # 执行工作流
        logger.debug(f"开始执行工作流，查询: {query[:50]}...")
        result = self.workflow.invoke({"query": query}, config=config)
        logger.debug("工作流执行完成")
        
        return result

# 示例用法
if __name__ == "__main__":
    from stockschain.model_selector import ModelSelector
    
    # 初始化模型选择器
    model_selector = ModelSelector(default_model_type='ollama')
    
    # 初始化股票分析工作流
    workflow = StockAnalysisWorkflow(model_selector)
    
    # 测试工作流
    queries = [
        "请分析一下苹果公司(AAPL)的投资价值",
        "特斯拉(TSLA)的股票技术面如何？"
    ]
    
    for query in queries:
        logger.debug(f"\n\n===== 处理查询: {query} =====")
        try:
            # 执行工作流
            result = workflow.invoke(query)
            
            # 打印最终报告
            logger.debug("\n\n===== 综合分析报告 =====")
            logger.debug(result["final_report"])
        except Exception as e:
            logger.error(f"错误: {str(e)}")
        
        logger.debug("\n\n" + "="*50 + "\n")