from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import Runnable
import os
from dotenv import load_dotenv
from utils.logger import get_logger

# 加载环境变量
load_dotenv()

# 创建日志实例
logger = get_logger('ollama_llm')

# 针对弃用警告的兼容性处理 - 检查是否需要导入替代方案
# try:
#     from langchain_ollama.chat_models import ChatOllama
# except ImportError:
#     # 回退到旧的导入方式
#     from langchain.chat_models import ChatOllama

class OllamaLLM(Runnable):
    """Ollama本地模型推理封装类"""
    
    def __init__(self, model=None, base_url=None, temperature=0.7):
        """
        初始化Ollama LLM实例
        
        Args:
            model: 使用的模型名称，默认为环境变量中的OLLAMA_MODEL
            base_url: Ollama API的基础URL，默认为环境变量中的OLLAMA_BASE_URL
            temperature: 生成文本的随机性，默认为0.7
        """
        self.model = model or os.getenv('OLLAMA_MODEL', 'deepseek-r1')
        self.base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.temperature = temperature
        
        # 初始化ChatOllama实例
        logger.info(f"初始化Ollama LLM: 模型={self.model}, 基础URL={self.base_url}")
        self.llm = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            keep_alive="1h"  # 保持模型活跃1小时
        )
        logger.debug("Ollama LLM初始化完成")
        
        # 创建输出解析器
        self.output_parser = StrOutputParser()
    
    def invoke(self, input_data, config=None, **kwargs):
        """实现Runnable接口的invoke方法"""
        try:
            logger.debug(f"开始调用Ollama模型: {self.model}")
            if isinstance(input_data, str):
                # 如果输入是字符串，直接调用内部的LLM
                return self._invoke_internal(input_data, kwargs.get('system_prompt'))
            elif isinstance(input_data, dict) and 'input' in input_data:
                # 如果输入是包含'input'键的字典（LangChain标准格式）
                system_prompt = kwargs.get('system_prompt') or input_data.get('system_prompt')
                return self._invoke_internal(input_data['input'], system_prompt)
            else:
                # 其他情况尝试将输入转换为字符串
                return self._invoke_internal(str(input_data), kwargs.get('system_prompt'))
        except Exception as e:
            logger.error(f"Ollama模型调用失败: {str(e)}")
            raise
    
    def _invoke_internal(self, input_text, system_prompt=None):
        """内部调用方法，避免递归"""
        try:
            logger.debug(f"内部调用，输入文本长度: {len(input_text)}字符")
            if system_prompt:
                logger.debug(f"使用系统提示: {system_prompt[:50]}...")
            
            chain = self.create_chain(system_prompt)
            result = chain.invoke({"input": input_text})
            logger.debug(f"内部调用成功，返回结果长度: {len(result)}字符")
            return result
        except Exception as e:
            logger.error(f"内部调用失败: {str(e)}")
            raise
    
    def create_chain(self, system_prompt=None):
        """
        创建一个LLM链
        
        Args:
            system_prompt: 系统提示词
            
        Returns:
            一个可调用的链对象
        """
        if system_prompt:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", "{input}")
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("user", "{input}")
            ])
        
        chain = prompt | self.llm | self.output_parser
        return chain
    
    def direct_invoke(self, input_text, system_prompt=None):
        """
        直接调用模型生成文本
        
        Args:
            input_text: 用户输入文本
            system_prompt: 系统提示词
            
        Returns:
            模型生成的文本
        """
        return self._invoke_internal(input_text, system_prompt)
    
    def batch_invoke(self, inputs, system_prompt=None):
        """
        批量调用模型生成文本
        
        Args:
            inputs: 输入文本列表
            system_prompt: 系统提示词
            
        Returns:
            模型生成的文本列表
        """
        try:
            logger.debug(f"开始批量调用Ollama模型，共{len(inputs)}个输入")
            if system_prompt:
                logger.debug(f"使用系统提示: {system_prompt[:50]}...")
            
            chain = self.create_chain(system_prompt)
            results = chain.batch([{"input": input_text} for input_text in inputs])
            logger.debug(f"批量调用完成，成功处理{len(results)}个输入")
            return results
        except Exception as e:
            logger.error(f"批量调用失败: {str(e)}")
            raise

# 示例用法
if __name__ == "__main__":
    # 初始化Ollama LLM
    ollama_llm = OllamaLLM()
    
    # 测试基本调用
    response = ollama_llm.invoke("什么是股票基本面分析？")
    logger.debug("基本调用结果:")
    logger.debug(response)
    logger.debug("\n" + "="*50 + "\n")
    
    # 测试带系统提示的调用
    system_prompt = "你是一位专业的金融分析师，请用简单易懂的语言解释复杂的金融概念。"
    response_with_system = ollama_llm.invoke("什么是股票基本面分析？", system_prompt)
    logger.debug("带系统提示的调用结果:")
    logger.debug(response_with_system)