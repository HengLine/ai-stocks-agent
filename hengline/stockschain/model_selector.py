from .vllm.vllm_llm import VLLM
from .ollama.ollama_llm import OllamaLLM
from .api.api_llm import APILLM
import os
from dotenv import load_dotenv

# 导入日志工具
from utils.logger import get_logger
logger = get_logger('model_selector')

# 加载环境变量
load_dotenv()

class ModelSelector:
    """模型选择器，用于在不同的模型推理后端之间进行选择"""
    
    # 支持的模型类型
    MODEL_TYPES = {
        'ollama': OllamaLLM,
        'vllm': VLLM,
        'api': APILLM
    }
    
    def __init__(self, default_model_type=None):
        """
        初始化模型选择器
        
        Args:
            default_model_type: 默认使用的模型类型，可选值为'ollama'或'vllm'
        """
        self.default_model_type = default_model_type or os.getenv('DEFAULT_MODEL_TYPE', 'ollama')
        
        logger.debug(f"初始化模型选择器，默认模型类型: {self.default_model_type}")
        
        # 确保默认模型类型有效
        if self.default_model_type not in self.MODEL_TYPES:
            error_msg = f"不支持的模型类型: {self.default_model_type}，支持的类型有: {list(self.MODEL_TYPES.keys())}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 存储已初始化的模型实例
        self.models = {}
    
    def get_model(self, model_type=None, **kwargs):
        """
        获取指定类型的模型实例
        
        Args:
            model_type: 模型类型，可选值为'ollama'或'vllm'
            **kwargs: 传递给模型初始化的参数
            
        Returns:
            模型实例
        """
        model_type = model_type or self.default_model_type
        
        logger.debug(f"获取模型实例，类型: {model_type}")
        
        # 检查模型类型是否有效
        if model_type not in self.MODEL_TYPES:
            error_msg = f"不支持的模型类型: {model_type}，支持的类型有: {list(self.MODEL_TYPES.keys())}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 如果模型实例已存在且没有提供新的参数，则返回已存在的实例
        key = f"{model_type}_{hash(frozenset(kwargs.items()))}"
        if key not in self.models:
            # 创建新的模型实例
            logger.debug(f"创建新的模型实例，类型: {model_type}")
            model_class = self.MODEL_TYPES[model_type]
            self.models[key] = model_class(**kwargs)
        
        return self.models[key]
    
    def invoke(self, prompt, model_type=None, system_prompt=None, **kwargs):
        """
        调用指定类型的模型生成文本
        
        Args:
            prompt: 用户输入文本
            model_type: 模型类型
            system_prompt: 系统提示词
            **kwargs: 传递给模型invoke方法的参数
            
        Returns:
            模型生成的文本
        """
        # 确定实际使用的模型类型
        actual_model_type = model_type or self.default_model_type
        logger.debug(f"调用模型进行推理，类型: {actual_model_type}")
        
        try:
            model = self.get_model(actual_model_type)
            
            # 根据模型类型调用相应的方法
            if actual_model_type == 'ollama':
                # 对于Ollama模型，使用direct_invoke方法
                result = model.direct_invoke(prompt, system_prompt=system_prompt, **kwargs)
            else:
                # 对于其他模型，保持原有调用方式
                result = model.invoke(prompt, system_prompt=system_prompt, **kwargs)
            
            logger.debug(f"模型推理成功，类型: {actual_model_type}")
            return result
        except Exception as e:
            logger.error(f"模型推理失败，类型: {actual_model_type}，错误: {str(e)}", exc_info=True)
            raise
    
    def batch_invoke(self, prompts, model_type=None, system_prompt=None, **kwargs):
        """
        批量调用指定类型的模型生成文本
        
        Args:
            prompts: 输入文本列表
            model_type: 模型类型
            system_prompt: 系统提示词
            **kwargs: 传递给模型batch_invoke方法的参数
            
        Returns:
            模型生成的文本列表
        """
        actual_model_type = model_type or self.default_model_type
        logger.debug(f"批量调用模型进行推理，类型: {actual_model_type}，数量: {len(prompts)}")
        
        try:
            model = self.get_model(actual_model_type)
            results = model.batch_invoke(prompts, system_prompt=system_prompt, **kwargs)
            logger.debug(f"批量模型推理成功，类型: {actual_model_type}")
            return results
        except Exception as e:
            logger.error(f"批量模型推理失败，类型: {actual_model_type}，错误: {str(e)}", exc_info=True)
            raise

# 示例用法
if __name__ == "__main__":
    # 初始化模型选择器
    model_selector = ModelSelector(default_model_type='ollama')
    
    # 使用默认模型(ollama)进行调用
    try:
        response_ollama = model_selector.invoke("解释一下什么是股票技术分析？")
        logger.debug("Ollama模型调用结果:")
        logger.debug(response_ollama)
        logger.debug("\n" + "="*50 + "\n")
    except Exception as e:
        logger.error(f"Ollama调用失败: {str(e)}")
    
    # 使用vllm模型进行调用
    try:
        response_vllm = model_selector.invoke("解释一下什么是股票技术分析？", model_type='vllm')
        logger.debug("VLLM模型调用结果:")
        logger.debug(response_vllm)
        logger.debug("\n" + "="*50 + "\n")
    except Exception as e:
        logger.error(f"VLLM调用失败: {str(e)}")
    
    # 使用API模型进行调用
    try:
        response_api = model_selector.invoke("解释一下什么是股票技术分析？", model_type='api')
        logger.debug("API模型调用结果:")
        logger.debug(response_api)
    except Exception as e:
        logger.error(f"API调用失败: {str(e)}")