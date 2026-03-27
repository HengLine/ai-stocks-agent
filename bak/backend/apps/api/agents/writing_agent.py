from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime
from .base import BaseAgent, AgentResult
from ..services.llm_service import LLMService


class WritingAgent(BaseAgent):
    name = "WritingAgent"

    def __init__(self):
        self.llm_service = LLMService()
        self.templates = self._load_templates()

    def run(self, params: Dict[str, Any], context: Dict[str, Any]) -> AgentResult:
        ticker = context.get("ticker") or params.get("ticker", "UNKNOWN")
        analysis_data = (
            context.get("AnalysisAgent", {})
            or context.get("analysis", {})
            or {}
        )
        
        # 获取分析结果
        if isinstance(analysis_data, dict) and "data" in analysis_data:
            analysis_result = analysis_data["data"]
        else:
            analysis_result = analysis_data
        
        # 获取参数
        template_name = params.get("template", "professional")
        output_format = params.get("format", "markdown")
        style = params.get("style", "professional")
        time_period = params.get("time_period", "未来一周")
        
        # 生成文章
        article_result = self._generate_article(ticker, analysis_result, template_name, style, time_period)
        
        # 转换为指定格式
        formatted_content = self._format_content(article_result, output_format)
        
        return AgentResult(
            name=self.name, 
            success=True, 
            data={
                "article": formatted_content,
                "template": template_name,
                "format": output_format,
                "style": style,
                "metadata": {
                    "ticker": ticker,
                    "generated_at": datetime.now().isoformat(),
                    "word_count": len(formatted_content.split()) if isinstance(formatted_content, str) else 0
                }
            }
        )
    
    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """加载文章模板，支持股票名称、时间、技术面、基本面、风险提示等占位符"""
        return {
            "professional": {
                "title": "{ticker}未来{time_period}走势分析",
                "structure": {
                    "summary": "基于技术指标分析，{ticker}当前{signal}信号明显，{trend}趋势持续，建议{action}。",
                    "technical": "技术面分析：{technical_analysis}",
                    "fundamental": "基本面分析：{fundamental_analysis}",
                    "risk": "风险提示：{risk_analysis}",
                    "conclusion": "投资建议：{investment_suggestion}"
                }
            },
            "humor": {
                "title": "【{ticker}】{time_period}股市过山车体验报告",
                "structure": {
                    "summary": "各位股民朋友，今天我们来聊聊{ticker}这只股票。{summary_humor}",
                    "technical": "技术面就像过山车：{technical_humor}",
                    "fundamental": "基本面分析：{fundamental_humor}",
                    "risk": "风险提示：{risk_humor}",
                    "conclusion": "总结：{conclusion_humor}"
                }
            },
            "brief": {
                "title": "{ticker}快速分析：{time_period}展望",
                "structure": {
                    "summary": "{ticker} {signal}，{action}建议",
                    "key_points": "关键要点：{key_points}",
                    "conclusion": "结论：{investment_suggestion}"
                }
            },
            "detailed": {
                "title": "{ticker}深度技术分析报告：{time_period}预测",
                "structure": {
                    "summary": "本报告对{ticker}进行全面的技术分析，包括趋势、指标、风险等多个维度。",
                    "technical": "技术指标分析：{detailed_technical}",
                    "trend": "趋势分析：{trend_analysis}",
                    "risk": "风险评估：{risk_analysis}",
                    "fundamental": "基本面分析：{fundamental_analysis}",
                    "conclusion": "投资建议：{investment_suggestion}"
                }
            }
        }
    
    def _generate_article(self, ticker: str, analysis_result: Dict[str, Any], 
                         template_name: str, style: str, time_period: str) -> Dict[str, str]:
        """生成文章内容，支持股票名称、时间、技术面、基本面、风险提示等占位符"""
        template = self.templates.get(template_name, self.templates["professional"])
        
        # 提取分析数据
        technical_analysis = analysis_result.get("technical_analysis", {})
        trend_analysis = analysis_result.get("trend_analysis", {})
        investment_suggestion = analysis_result.get("investment_suggestion", {})
        key_points = analysis_result.get("key_points", [])
        risk_level = analysis_result.get("risk_level", "medium")
        
        # 生成各部分内容
        article_parts = {}
        
        # 标题 - 包含股票名称、时间、技术面、基本面、风险提示等占位符
        article_parts["title"] = template["title"].format(
            ticker=ticker,
            time_period=time_period
        )
        
        # 根据模板生成各部分内容
        for section, template_text in template["structure"].items():
            if section == "summary":
                article_parts[section] = self._generate_summary(ticker, technical_analysis, investment_suggestion, template_text, style)
            elif section == "technical":
                article_parts[section] = self._generate_technical_section(technical_analysis, key_points, template_text, style)
            elif section == "fundamental":
                article_parts[section] = self._generate_fundamental_section(analysis_result, template_text, style)
            elif section == "risk":
                article_parts[section] = self._generate_risk_section(risk_level, technical_analysis, template_text, style)
            elif section == "conclusion":
                article_parts[section] = self._generate_conclusion(investment_suggestion, template_text, style)
            elif section == "key_points":
                article_parts[section] = self._format_key_points(key_points, template_text)
            elif section == "trend":
                article_parts[section] = self._generate_trend_section(trend_analysis, template_text, style)
            else:
                article_parts[section] = template_text
        
        return article_parts
    
    def _generate_summary(self, ticker: str, technical_analysis: Dict, investment_suggestion: Dict, 
                         template: str, style: str) -> str:
        """生成摘要"""
        signal = technical_analysis.get("overall_signal", "neutral")
        action = investment_suggestion.get("action", "hold")
        
        if style == "humor":
            return template.format(
                ticker=ticker,
                signal="看涨" if signal == "bullish" else "看跌" if signal == "bearish" else "震荡",
                action="买入" if action == "buy" else "卖出" if action == "sell" else "观望",
                summary_humor=f"当前价格就像坐过山车，{signal}信号明显，建议{action}！"
            )
        else:
            return template.format(
                ticker=ticker,
                signal="看涨" if signal == "bullish" else "看跌" if signal == "bearish" else "震荡",
                trend="上涨" if signal == "bullish" else "下跌" if signal == "bearish" else "横盘",
                action="买入" if action == "buy" else "卖出" if action == "sell" else "观望"
            )
    
    def _generate_technical_section(self, technical_analysis: Dict, key_points: List[str], 
                                   template: str, style: str) -> str:
        """生成技术面分析"""
        ma_signal = technical_analysis.get("ma_signal", "neutral")
        rsi_signal = technical_analysis.get("rsi_signal", "neutral")
        macd_signal = technical_analysis.get("macd_signal", "neutral")
        
        if style == "humor":
            technical_text = f"MA指标{ma_signal}，RSI{rsi_signal}，MACD{macd_signal}。"
            if rsi_signal == "overbought":
                technical_text += "RSI超买，就像吃撑了需要消化！"
            elif rsi_signal == "oversold":
                technical_text += "RSI超卖，就像饿坏了需要补充能量！"
        else:
            technical_text = f"移动平均线显示{ma_signal}信号，RSI指标处于{rsi_signal}状态，MACD指标呈现{macd_signal}趋势。"
            if key_points:
                technical_text += "关键要点：" + "；".join(key_points[:3])
        
        return template.format(technical_analysis=technical_text)
    
    def _generate_fundamental_section(self, analysis_result: Dict, template: str, style: str) -> str:
        """生成基本面分析"""
        if style == "humor":
            return template.format(
                fundamental_humor="基本面就像公司的体检报告，需要定期检查。建议关注财报数据和行业动态。"
            )
        else:
            return template.format(
                fundamental_analysis="建议关注公司财报、行业趋势、宏观经济环境等基本面因素，结合技术分析做出投资决策。"
            )
    
    def _generate_risk_section(self, risk_level: str, technical_analysis: Dict, 
                              template: str, style: str) -> str:
        """生成风险提示"""
        if style == "humor":
            risk_text = f"风险等级：{risk_level}。"
            if risk_level == "high":
                risk_text += "高风险就像高空走钢丝，需要格外小心！"
            elif risk_level == "medium":
                risk_text += "中等风险就像开车，需要遵守交通规则！"
            else:
                risk_text += "低风险就像散步，相对安全但也要看路！"
        else:
            risk_text = f"当前风险等级为{risk_level}。"
            if risk_level == "high":
                risk_text += "建议谨慎操作，严格控制仓位，设置止损。"
            elif risk_level == "medium":
                risk_text += "建议适度参与，注意风险控制。"
            else:
                risk_text += "风险相对较低，但仍需注意市场变化。"
        
        return template.format(risk_analysis=risk_text)
    
    def _generate_conclusion(self, investment_suggestion: Dict, template: str, style: str) -> str:
        """生成结论"""
        action = investment_suggestion.get("action", "hold")
        confidence = investment_suggestion.get("confidence", "low")
        reasoning = investment_suggestion.get("reasoning", "")
        
        if style == "humor":
            action_text = "买入" if action == "buy" else "卖出" if action == "sell" else "观望"
            confidence_text = "高" if confidence == "high" else "中" if confidence == "medium" else "低"
            return template.format(
                conclusion_humor=f"综合判断建议{action_text}，信心度{confidence_text}。记住：投资有风险，入市需谨慎！"
            )
        else:
            return template.format(
                investment_suggestion=f"基于技术分析，建议{action}操作，信心度{confidence}。{reasoning}。投资有风险，决策需谨慎。"
            )
    
    def _format_key_points(self, key_points: List[str], template: str) -> str:
        """格式化关键要点"""
        points_text = "；".join(key_points) if key_points else "暂无特殊要点"
        return template.format(key_points=points_text)
    
    def _generate_trend_section(self, trend_analysis: Dict, template: str, style: str) -> str:
        """生成趋势分析"""
        short_term = trend_analysis.get("short_term", "unknown")
        long_term = trend_analysis.get("long_term", "unknown")
        strength = trend_analysis.get("strength", "weak")
        price_change = trend_analysis.get("price_change_percent", 0)
        
        if style == "humor":
            trend_text = f"短期趋势{short_term}，长期趋势{long_term}，趋势强度{strength}，价格变化{price_change}%。"
        else:
            trend_text = f"短期趋势呈现{short_term}态势，长期趋势为{long_term}，趋势强度{strength}，期间价格变化{price_change}%。"
        
        return template.format(trend_analysis=trend_text)
    
    def _format_content(self, article_parts: Dict[str, str], output_format: str) -> str:
        """将文章内容转换为指定格式"""
        if output_format == "markdown":
            return self._to_markdown(article_parts)
        elif output_format == "html":
            return self._to_html(article_parts)
        elif output_format == "plain":
            return self._to_plain_text(article_parts)
        else:
            return self._to_markdown(article_parts)  # 默认markdown
    
    def _to_markdown(self, article_parts: Dict[str, str]) -> str:
        """转换为Markdown格式"""
        content = f"# {article_parts.get('title', '')}\n\n"
        
        for section, text in article_parts.items():
            if section != "title":
                content += f"## {self._get_section_title(section)}\n\n{text}\n\n"
        
        return content
    
    def _to_html(self, article_parts: Dict[str, str]) -> str:
        """转换为HTML格式"""
        content = f"<h1>{article_parts.get('title', '')}</h1>\n"
        
        for section, text in article_parts.items():
            if section != "title":
                content += f"<h2>{self._get_section_title(section)}</h2>\n<p>{text}</p>\n"
        
        return f"<html><body>{content}</body></html>"
    
    def _to_plain_text(self, article_parts: Dict[str, str]) -> str:
        """转换为纯文本格式"""
        content = f"{article_parts.get('title', '')}\n\n"
        
        for section, text in article_parts.items():
            if section != "title":
                content += f"{self._get_section_title(section)}\n{text}\n\n"
        
        return content
    
    def _get_section_title(self, section: str) -> str:
        """获取章节标题"""
        titles = {
            "summary": "摘要",
            "technical": "技术面分析",
            "fundamental": "基本面分析",
            "risk": "风险提示",
            "conclusion": "投资建议",
            "key_points": "关键要点",
            "trend": "趋势分析"
        }
        return titles.get(section, section)


