# 意图识别模块的初始化文件

# 导出主要类和全局实例
from .base import IntentRecognizer, IntentParser, IntentRecognizerFactory
from .rule_based_recognizer import RuleBasedIntentRecognizer
from .nlp_based_recognizer import NLPBasedIntentRecognizer, HybridIntentRecognizer
from .service import IntentRecognitionService, global_intent_service

__all__ = [
    'IntentRecognizer',
    'IntentParser', 
    'IntentRecognizerFactory',
    'RuleBasedIntentRecognizer',
    'NLPBasedIntentRecognizer',
    'HybridIntentRecognizer',
    'IntentRecognitionService',
    'global_intent_service'
]