from __future__ import annotations

from typing import List, Dict, Any, Optional
import openai
import os


class LLMService:
    def __init__(self):
        # 可以配置 OpenAI API 或使用本地模型
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        
        if self.api_key:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        else:
            self.client = None

    def generate_article(self, ticker: str, metrics: Dict[str, Any], style: str = "professional") -> str:
        """生成股票分析文章"""
        if not self.client:
            return self._fallback_article(ticker, metrics, style)
        
        prompt = f"""
        请基于以下股票数据生成一篇{style}风格的分析文章：
        
        股票代码: {ticker}
        技术指标: {metrics}
        
        要求：
        1. 分析当前技术指标的含义
        2. 给出投资建议
        3. 风格要求: {style}
        4. 字数控制在500字以内
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return self._fallback_article(ticker, metrics, style)

    def chat_response(self, message: str, context: List[Dict[str, Any]] = None) -> str:
        """生成对话回复"""
        if not self.client:
            return self._fallback_chat(message)
        
        messages = [{"role": "system", "content": "你是一个专业的股票分析助手，能够回答股票相关问题。"}]
        
        # 添加上下文
        if context:
            for ctx in context[-3:]:  # 只取最近3条
                messages.append({"role": "user", "content": ctx.get("message", "")})
                messages.append({"role": "assistant", "content": ctx.get("response", "")})
        
        messages.append({"role": "user", "content": message})
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return self._fallback_chat(message)

    def _fallback_article(self, ticker: str, metrics: Dict[str, Any], style: str) -> str:
        """备用文章生成（无API时）"""
        close = metrics.get("close", "未知")
        ma5 = metrics.get("ma5", "未知")
        rsi = metrics.get("rsi", "未知")
        
        if style == "humor":
            return f"""
            【{ticker}】股票分析报告（幽默版）
            
            各位股民朋友，今天我们来聊聊{ticker}这只股票。
            当前收盘价：{close}元，MA5均线：{ma5}，RSI指标：{rsi}。
            
            从技术面看，这只股票就像过山车一样刺激！
            建议：系好安全带，准备好晕车药，投资有风险，入市需谨慎！
            
            免责声明：以上内容纯属娱乐，不构成投资建议。
            """
        else:
            return f"""
            【{ticker}】技术分析报告
            
            股票代码：{ticker}
            当前收盘价：{close}元
            MA5均线：{ma5}
            RSI指标：{rsi}
            
            技术分析：
            1. 价格走势需要结合更多指标综合判断
            2. 建议关注成交量变化
            3. 注意风险控制
            
            投资建议：仅供参考，请结合基本面分析。
            """

    def _fallback_chat(self, message: str) -> str:
        """备用对话回复"""
        if "股票" in message or "分析" in message:
            return "我是股票分析助手，可以帮您分析股票技术指标。请提供股票代码，我会为您生成分析报告。"
        elif "你好" in message or "hello" in message.lower():
            return "您好！我是AIGC股票分析助手，可以帮您分析股票、生成报告。有什么可以帮您的吗？"
        else:
            return "我主要专注于股票分析，请告诉我您想了解哪只股票的情况。"
