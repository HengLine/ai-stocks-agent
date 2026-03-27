from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class IntentRecognizer(ABC):
    """
    意图识别器的抽象基类，定义了意图识别的通用接口
    """
    
    @abstractmethod
    def recognize_intent(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        识别用户输入的意图并提取关键参数
        
        Args:
            text: 用户输入的文本
            **kwargs: 其他可能的参数
        
        Returns:
            包含识别结果的字典，至少包含以下键:
            - ticker: 股票代码
            - time_window: 时间窗口
            - dimensions: 分析维度列表
            - output: 输出格式
        """
        pass


class IntentParser:
    """
    意图解析器，将识别到的意图转换为结构化查询指令
    """
    
    def parse_to_query(self, intent_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        将意图识别结果转换为结构化查询指令
        
        Args:
            intent_result: 意图识别结果
        
        Returns:
            结构化查询指令
        """
        # 默认实现：直接返回识别结果
        return intent_result.copy()


class IntentRecognizerFactory:
    """
    意图识别器工厂，用于创建不同类型的意图识别器
    """
    
    _recognizers: Dict[str, IntentRecognizer] = {}
    
    @classmethod
    def register(cls, name: str, recognizer: IntentRecognizer) -> None:
        """
        注册意图识别器
        
        Args:
            name: 识别器名称
            recognizer: 意图识别器实例
        """
        cls._recognizers[name] = recognizer
        
    @classmethod
    def get(cls, name: str) -> Optional[IntentRecognizer]:
        """
        获取指定名称的意图识别器
        
        Args:
            name: 识别器名称
        
        Returns:
            意图识别器实例，如果不存在则返回None
        """
        return cls._recognizers.get(name)