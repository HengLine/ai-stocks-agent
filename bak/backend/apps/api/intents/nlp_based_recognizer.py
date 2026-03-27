from typing import Dict, Any, List, Optional
import logging
from .base import IntentRecognizer


class NLPBasedIntentRecognizer(IntentRecognizer):
    """
    基于NLP模型的意图识别器，使用预训练模型进行更复杂的意图理解
    """
    
    def __init__(self, model_name: str = "default", llm_service=None):
        """
        初始化NLP意图识别器
        
        Args:
            model_name: 使用的模型名称
            llm_service: LLM服务实例，用于调用预训练模型
        """
        self.model_name = model_name
        self.llm_service = llm_service
        self.logger = logging.getLogger(__name__)
        
        # 设置默认的提示模板
        self.prompt_template = """
        请分析用户输入并提取以下信息：
        1. 股票代码（如果有，格式如：SH:600519或SZ:300750）
        2. 时间窗口（如：3M、6M、1Y等）
        3. 分析维度（可选值：technical、fundamental、sentiment，可多选）
        4. 输出格式（可选值：article、video，单选）
        
        如果用户输入中没有明确的信息，请提供合理的默认值或留空。
        请以JSON格式返回结果，不要包含其他解释性文本。
        
        用户输入：{text}
        """
    
    def _call_llm_service(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        调用LLM服务获取意图识别结果
        
        Args:
            prompt: 提示文本
        
        Returns:
            LLM返回的结构化结果，或None如果调用失败
        """
        if not self.llm_service:
            self.logger.warning("LLM服务未初始化，无法使用NLP模型进行意图识别")
            return None
        
        try:
            # 调用LLM服务获取结构化结果
            response = self.llm_service.generate_structured_output(prompt)
            return response
        except Exception as e:
            self.logger.error(f"调用LLM服务失败: {str(e)}")
            return None
    
    def _extract_stock_info(self, text: str) -> str:
        """
        从文本中提取股票信息
        这个方法可以通过NLP模型或规则结合的方式实现
        """
        # 简化实现，实际应用中可以结合NLP模型进行更复杂的实体识别
        # 这里可以根据需要扩展，例如使用命名实体识别(NER)来提取股票名称或代码
        return ""
    
    def _determine_intent(self, text: str) -> Dict[str, Any]:
        """
        使用NLP模型确定用户意图
        """
        # 生成提示文本
        prompt = self.prompt_template.format(text=text)
        
        # 调用LLM服务
        result = self._call_llm_service(prompt)
        
        # 如果调用失败，返回默认值
        if not result:
            return {
                "ticker": "",
                "time_window": "3M",
                "dimensions": ["technical"],
                "output": "article",
            }
        
        return result
    
    def recognize_intent(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        实现抽象方法，使用NLP模型识别用户意图
        """
        if not text:
            return {
                "ticker": "",
                "time_window": "3M",
                "dimensions": ["technical"],
                "output": "article",
                "raw": {"text": text}
            }
        
        # 使用NLP模型确定意图
        nlp_result = self._determine_intent(text)
        
        # 优先使用传入的参数，其次使用NLP模型的结果
        ticker = kwargs.get("ticker") or nlp_result.get("ticker", "")
        time_window = kwargs.get("time_window") or nlp_result.get("time_window", "3M")
        dimensions = kwargs.get("dimensions") or nlp_result.get("dimensions", ["technical"])
        output = kwargs.get("output") or nlp_result.get("output", "article")
        
        # 确保dimensions是列表格式
        if isinstance(dimensions, str):
            dimensions = [dimensions]
        
        return {
            "ticker": ticker,
            "time_window": time_window,
            "dimensions": dimensions,
            "output": output,
            "raw": {"text": text, "nlp_result": nlp_result}
        }


class HybridIntentRecognizer(IntentRecognizer):
    """
    混合意图识别器，结合规则基础和NLP基础的识别器
    """
    
    def __init__(self, rule_based_recognizer: IntentRecognizer, 
                 nlp_based_recognizer: IntentRecognizer):
        """
        初始化混合意图识别器
        
        Args:
            rule_based_recognizer: 基于规则的识别器实例
            nlp_based_recognizer: 基于NLP的识别器实例
        """
        self.rule_based_recognizer = rule_based_recognizer
        self.nlp_based_recognizer = nlp_based_recognizer
    
    def recognize_intent(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        实现抽象方法，使用混合策略识别用户意图
        """
        # 先用规则基础识别器进行识别
        rule_result = self.rule_based_recognizer.recognize_intent(text, **kwargs)
        
        # 如果规则基础识别器无法完全识别，使用NLP基础识别器进行补充
        # 这里可以根据实际需求定义判断条件
        if not rule_result.get("ticker") or not rule_result.get("time_window"):
            nlp_result = self.nlp_based_recognizer.recognize_intent(text, **kwargs)
            
            # 合并结果，优先使用NLP结果补充缺失的信息
            rule_result["ticker"] = rule_result["ticker"] or nlp_result.get("ticker", "")
            rule_result["time_window"] = rule_result["time_window"] or nlp_result.get("time_window", "3M")
            rule_result["dimensions"] = rule_result["dimensions"] or nlp_result.get("dimensions", ["technical"])
            rule_result["output"] = rule_result["output"] or nlp_result.get("output", "article")
            rule_result["raw"]["nlp_result"] = nlp_result.get("raw", {})
        
        return rule_result