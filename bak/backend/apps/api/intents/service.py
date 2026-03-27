from typing import Dict, Any, Optional
import logging
from .base import IntentRecognizerFactory, IntentParser
from .rule_based_recognizer import RuleBasedIntentRecognizer
from .nlp_based_recognizer import NLPBasedIntentRecognizer, HybridIntentRecognizer


class IntentRecognitionService:
    """
    意图识别服务，提供统一的意图识别接口
    """
    
    def __init__(self):
        """初始化意图识别服务"""
        self.logger = logging.getLogger(__name__)
        
        # 初始化并注册意图识别器
        self._initialize_recognizers()
        
        # 初始化意图解析器
        self.intent_parser = IntentParser()
        
        # 默认使用混合识别器
        self.default_recognizer_name = "hybrid"
    
    def _initialize_recognizers(self) -> None:
        """初始化并注册意图识别器"""
        # 创建基于规则的识别器
        rule_based_recognizer = RuleBasedIntentRecognizer()
        IntentRecognizerFactory.register("rule_based", rule_based_recognizer)
        
        # 创建基于NLP的识别器
        # 注意：这里暂时不传入LLM服务，实际使用时需要传入
        nlp_based_recognizer = NLPBasedIntentRecognizer()
        IntentRecognizerFactory.register("nlp_based", nlp_based_recognizer)
        
        # 创建混合识别器
        hybrid_recognizer = HybridIntentRecognizer(
            rule_based_recognizer=rule_based_recognizer,
            nlp_based_recognizer=nlp_based_recognizer
        )
        IntentRecognizerFactory.register("hybrid", hybrid_recognizer)
    
    def set_llm_service(self, llm_service) -> None:
        """
        设置LLM服务实例
        
        Args:
            llm_service: LLM服务实例
        """
        nlp_recognizer = IntentRecognizerFactory.get("nlp_based")
        if nlp_recognizer and hasattr(nlp_recognizer, "llm_service"):
            nlp_recognizer.llm_service = llm_service
    
    def recognize_intent(self, 
                         text: str, 
                         recognizer_name: Optional[str] = None, 
                         **kwargs) -> Dict[str, Any]:
        """
        识别用户意图
        
        Args:
            text: 用户输入的文本
            recognizer_name: 使用的识别器名称，默认为None（使用默认识别器）
            **kwargs: 其他可能的参数
        
        Returns:
            识别结果
        """
        # 选择识别器
        recognizer_name = recognizer_name or self.default_recognizer_name
        recognizer = IntentRecognizerFactory.get(recognizer_name)
        
        if not recognizer:
            self.logger.error(f"未找到名为'{recognizer_name}'的意图识别器")
            # 回退到默认识别器
            recognizer = IntentRecognizerFactory.get(self.default_recognizer_name)
            
            if not recognizer:
                self.logger.error("默认意图识别器也不存在，使用规则基础识别器")
                recognizer = RuleBasedIntentRecognizer()
        
        try:
            # 调用识别器进行意图识别
            result = recognizer.recognize_intent(text, **kwargs)
            
            # 将识别结果转换为结构化查询指令
            query = self.intent_parser.parse_to_query(result)
            
            # 添加元信息
            query["recognition_info"] = {
                "recognizer": recognizer_name,
                "raw_text": text
            }
            
            return query
        except Exception as e:
            self.logger.error(f"意图识别失败: {str(e)}")
            # 返回默认结果
            return {
                "ticker": kwargs.get("ticker", ""),
                "time_window": kwargs.get("time_window", "3M"),
                "dimensions": kwargs.get("dimensions", ["technical"]),
                "output": kwargs.get("output", "article"),
                "raw": {"text": text},
                "recognition_info": {
                    "recognizer": recognizer_name,
                    "error": str(e)
                }
            }
    
    def parse_structured_input(self, structured_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析结构化输入
        
        Args:
            structured_data: 结构化输入数据
        
        Returns:
            结构化查询指令
        """
        # 提取结构化数据中的参数
        ticker = structured_data.get("ticker", "")
        time_window = structured_data.get("time_window", "3M")
        dimensions = structured_data.get("dimensions", ["technical"])
        output = structured_data.get("output", "article")
        
        # 确保dimensions是列表格式
        if isinstance(dimensions, str):
            dimensions = [dimensions]
        
        result = {
            "ticker": ticker,
            "time_window": time_window,
            "dimensions": dimensions,
            "output": output,
            "raw": structured_data,
            "recognition_info": {
                "source": "structured_input"
            }
        }
        
        # 转换为结构化查询指令
        return self.intent_parser.parse_to_query(result)
    
    def parse_voice_input(self, voice_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析语音输入
        
        Args:
            voice_data: 语音输入数据，包含转写后的文本等信息
        
        Returns:
            结构化查询指令
        """
        # 从语音数据中提取转写后的文本
        text = voice_data.get("transcribed_text", "")
        
        # 使用意图识别器处理转写后的文本
        result = self.recognize_intent(text)
        
        # 添加语音相关元信息
        result["recognition_info"]["source"] = "voice_input"
        result["recognition_info"]["voice_details"] = {
            "duration": voice_data.get("duration"),
            "confidence": voice_data.get("confidence")
        }
        
        return result


# 创建全局服务实例，方便其他模块使用
global_intent_service = IntentRecognitionService()