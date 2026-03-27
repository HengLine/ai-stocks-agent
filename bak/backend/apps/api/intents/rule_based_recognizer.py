import re
from typing import Dict, Any, List
from .base import IntentRecognizer


class RuleBasedIntentRecognizer(IntentRecognizer):
    """
    基于规则的意图识别器，使用关键词匹配和正则表达式解析用户输入
    """
    
    def __init__(self):
        # 初始化关键词和正则表达式模式
        self.time_patterns = [
            (r"(\d+)\s*个月?", "M"),
            (r"(\d+)\s*月", "M"),
            (r"(\d+)\s*周", "W"),
            (r"(\d+)\s*天", "D"),
            (r"(\d+)\s*年", "Y"),
            # 增加更多模式以匹配"未来3个月"、"近一个月"等表述
            (r"未来(\d+)\s*个月?", "M"),
            (r"近(\d+)\s*个月?", "M"),
            (r"最近(\d+)\s*个月?", "M"),
            (r"过去(\d+)\s*个月?", "M"),
        ]
        
        # 分析维度关键词映射
        self.dimension_mapping = {
            "technical": ["技术", "指标", "K线", "均线", "MACD", "RSI", "布林", 
                           "技术面", "走势", "图形", "形态"],
            "fundamental": ["基本面", "财报", "盈利", "估值", "ROE", "营收",
                             "财务", "业绩", "市盈率", "市净率"],
            "sentiment": ["情绪", "舆情", "新闻", "热度", "微博", "论坛",
                          "市场情绪", "投资者情绪"],
        }
        
        # 输出格式关键词映射
        self.output_mapping = {
            "article": ["文章", "报告", "文档", "文本"],
            "video": ["视频", "配音", "字幕", "视频讲解"],
        }
        
        # 股票代码相关正则
        self.stock_code_patterns = [
            r"(SH|SZ)[:]?\s?(\d{6})",  # 匹配SH:600519或SZ:300750格式
            r"(\d{6})",  # 匹配纯数字股票代码
        ]
        
        # 常见股票名称与代码的映射（可扩展）
        self.stock_name_to_code = {
            "宁德时代": "SZ:300750",
            "贵州茅台": "SH:600519",
            "比亚迪": "SZ:002594",
            "腾讯控股": "HK:00700",
            "阿里巴巴": "US:BABA",
        }
    
    def _parse_time_window(self, text: str) -> str:
        """解析时间窗口"""
        if not text:
            return ""
        
        for pattern, suffix in self.time_patterns:
            match = re.search(pattern, text)
            if match:
                return f"{match.group(1)}{suffix}"
        
        # 默认返回3个月
        return "3M"
    
    def _parse_dimensions(self, text: str) -> List[str]:
        """解析分析维度"""
        dimensions = []
        
        for dim_name, keywords in self.dimension_mapping.items():
            if any(keyword in text for keyword in keywords):
                dimensions.append(dim_name)
        
        # 如果没有识别到维度，默认为技术面
        return dimensions or ["technical"]
    
    def _parse_output(self, text: str) -> str:
        """解析输出格式"""
        for output_type, keywords in self.output_mapping.items():
            if any(keyword in text for keyword in keywords):
                return output_type
        
        # 默认返回文章格式
        return "article"
    
    def _parse_ticker(self, text: str) -> str:
        """解析股票代码"""
        # 1. 首先尝试从股票名称映射中获取代码
        for name, code in self.stock_name_to_code.items():
            if name in text:
                return code
        
        # 2. 使用正则表达式匹配股票代码
        for pattern in self.stock_code_patterns:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) == 2:
                    # 格式为SH:600519
                    return f"{match.group(1).upper()}:{match.group(2)}"
                else:
                    # 纯数字代码
                    code = match.group(1)
                    # 尝试根据代码前几位判断市场（简化逻辑）
                    if code.startswith('6'):
                        return f"SH:{code}"
                    elif code.startswith('0') or code.startswith('3'):
                        return f"SZ:{code}"
                    else:
                        return code
        
        return ""
    
    def recognize_intent(self, text: str, **kwargs) -> Dict[str, Any]:
        """实现抽象方法，识别用户意图"""
        if not text:
            return {
                "ticker": "",
                "time_window": "3M",
                "dimensions": ["technical"],
                "output": "article",
                "raw": {"text": text}
            }
        
        # 提取参数
        ticker = kwargs.get("ticker") or self._parse_ticker(text)
        time_window = kwargs.get("time_window") or self._parse_time_window(text)
        dimensions = kwargs.get("dimensions") or self._parse_dimensions(text)
        output = kwargs.get("output") or self._parse_output(text)
        
        return {
            "ticker": ticker,
            "time_window": time_window,
            "dimensions": dimensions,
            "output": output,
            "raw": {"text": text}
        }