import requests
import json
import os
from dotenv import load_dotenv
from utils.logger import get_logger

# 加载环境变量
load_dotenv()

# 创建日志实例
logger = get_logger('api_llm')

class APILLM:
    """第三方模型API调用封装类"""
    
    def __init__(self, api_url=None, api_key=None, model=None, temperature=0.7, max_tokens=1024, headers=None):
        """
        初始化第三方模型API实例
        
        Args:
            api_url: API的URL，默认为环境变量中的API_URL
            api_key: API密钥，默认为环境变量中的API_KEY
            model: 使用的模型名称，默认为环境变量中的API_MODEL
            temperature: 生成文本的随机性，默认为0.7
            max_tokens: 最大生成的token数，默认为1024
            headers: 请求头信息，如果为None则使用默认请求头
        """
        logger.info(f"初始化第三方模型API实例，API URL: {api_url or '默认'}")
        
        self.api_url = api_url or os.getenv('API_URL', 'https://api.example.com/v1/chat/completions')
        self.api_key = api_key or os.getenv('API_KEY', '')
        self.model = model or os.getenv('API_MODEL', 'gpt-4o-mini')
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 默认请求头
        self.headers = headers or {
            "Content-Type": "application/json",
        }
        
        # 如果提供了API密钥，添加到请求头
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        
        logger.debug(f"API配置: URL={self.api_url}, 模型={self.model}, 温度={self.temperature}, 最大token={self.max_tokens}")
        
        # 检查API服务是否可用
        try:
            self._check_service()
            logger.debug(f"API服务已连接: {self.api_url}")
        except Exception as e:
            logger.warning(f"无法连接到API服务 - {str(e)}")
            logger.warning("请确保API URL和API密钥配置正确")
    
    def _check_service(self):
        """检查API服务是否可用"""
        logger.debug("开始检查API服务可用性")
        try:
            # 发送一个简单的测试请求来检查服务可用性
            headers = self.headers.copy()
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "test"}],
                "temperature": 0,
                "max_tokens": 1
            }
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload), timeout=5)
            response.raise_for_status()
            logger.debug("API服务检查成功")
        except requests.exceptions.RequestException as e:
            logger.error(f"服务检查失败: {str(e)}")
            # 不抛出异常，仅记录警告，因为API可能在实际使用时可用
            logger.warning(f"API服务检查失败，但将继续初始化: {str(e)}")
    
    def invoke(self, prompt, system_prompt=None, temperature=None, max_tokens=None):
        """
        调用API生成文本
        
        Args:
            prompt: 用户输入文本
            system_prompt: 系统提示词
            temperature: 生成文本的随机性，如果为None则使用初始化时的值
            max_tokens: 最大生成的token数，如果为None则使用初始化时的值
            
        Returns:
            模型生成的文本
        """
        logger.debug(f"开始API调用，输入文本长度: {len(prompt)}字符")
        if system_prompt:
            logger.debug(f"使用系统提示: {system_prompt[:50]}...")
        
        # 准备消息列表
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # 准备请求参数
        headers = self.headers.copy()
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens
        }
        
        # 发送请求
        try:
            logger.debug(f"发送请求到API: {self.api_url}")
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload), timeout=60)
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            
            # 尝试从常见的响应格式中提取文本
            if "choices" in result and len(result["choices"]) > 0:
                if "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                    text_result = result["choices"][0]["message"]["content"]
                elif "text" in result["choices"][0]:
                    text_result = result["choices"][0]["text"]
                else:
                    text_result = str(result["choices"][0])
            else:
                text_result = str(result)
                logger.warning(f"API响应格式不标准: {result}")
            
            logger.debug(f"API调用成功，返回结果长度: {len(text_result)}字符")
            return text_result
        except requests.exceptions.RequestException as e:
            logger.error(f"API调用失败: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"API响应状态码: {e.response.status_code}")
                logger.error(f"API响应内容: {e.response.text}")
            raise Exception(f"API调用失败: {str(e)}")
        except json.JSONDecodeError:
            logger.error(f"API响应解析失败: {response.text}")
            raise Exception(f"API响应解析失败: {response.text}")
    
    def batch_invoke(self, prompts, system_prompt=None, temperature=None, max_tokens=None):
        """
        批量调用API生成文本
        
        Args:
            prompts: 输入文本列表
            system_prompt: 系统提示词，将应用于所有输入
            temperature: 生成文本的随机性
            max_tokens: 最大生成的token数
            
        Returns:
            模型生成的文本列表
        """
        logger.debug(f"开始API批量调用，共{len(prompts)}个输入")
        if system_prompt:
            logger.debug(f"批量调用使用系统提示: {system_prompt[:50]}...")
        
        results = []
        try:
            for i, prompt in enumerate(prompts):
                logger.debug(f"处理第{i+1}/{len(prompts)}个输入")
                results.append(self.invoke(prompt, system_prompt, temperature, max_tokens))
            
            logger.debug(f"API批量调用完成，成功处理{len(results)}个输入")
            return results
        except Exception as e:
            logger.error(f"API批量调用失败: {str(e)}")
            raise

# 示例用法
if __name__ == "__main__":
    # 初始化API LLM
    try:
        api_llm = APILLM()
        
        # 测试基本调用
        response = api_llm.invoke("什么是股票基本面分析？")
        logger.debug("基本调用结果:")
        logger.debug(response)
        logger.debug("\n" + "="*50 + "\n")
        
        # 测试带系统提示的调用
        system_prompt = "你是一位专业的金融分析师，请用简单易懂的语言解释复杂的金融概念。"
        response_with_system = api_llm.invoke("什么是股票基本面分析？", system_prompt)
        logger.debug("带系统提示的调用结果:")
        logger.debug(response_with_system)
    except Exception as e:
        logger.error(f"错误: {str(e)}")