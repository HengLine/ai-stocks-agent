from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime
import os
import json
from .base import BaseAgent, AgentResult
from ..services.llm_service import LLMService


class VideoAgent(BaseAgent):
    name = "VideoAgent"

    def __init__(self):
        self.llm_service = LLMService()
        self.video_templates = self._load_video_templates()

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
        video_style = params.get("style", "professional")
        duration = params.get("duration", "short")  # short, medium, long
        aspect_ratio = params.get("aspect_ratio", "16:9")  # 16:9, 9:16, 1:1
        language = params.get("language", "zh-CN")
        
        # 生成视频脚本
        script_result = self._generate_video_script(ticker, analysis_result, video_style, duration)
        
        # 生成视频制作计划
        production_plan = self._create_production_plan(script_result, aspect_ratio, language)
        
        return AgentResult(
            name=self.name,
            success=True,
            data={
                "script": script_result,
                "production_plan": production_plan,
                "metadata": {
                    "ticker": ticker,
                    "style": video_style,
                    "duration": duration,
                    "aspect_ratio": aspect_ratio,
                    "language": language,
                    "generated_at": datetime.now().isoformat()
                }
            }
        )
    
    def _load_video_templates(self) -> Dict[str, Dict[str, Any]]:
        """加载视频模板"""
        return {
            "professional": {
                "opening": "大家好，今天我们来分析{ticker}的技术走势。",
                "structure": {
                    "introduction": "让我们先看看{ticker}的基本情况。",
                    "technical_analysis": "从技术面来看，{technical_summary}",
                    "trend_analysis": "趋势分析显示，{trend_summary}",
                    "risk_assessment": "风险方面，{risk_summary}",
                    "conclusion": "综合以上分析，{investment_conclusion}"
                },
                "closing": "以上就是对{ticker}的分析，感谢观看，我们下期再见。",
                "bgm_style": "calm_professional",
                "visual_style": "charts_and_graphs"
            },
            "humor": {
                "opening": "各位股民朋友，今天我们来聊聊{ticker}这只股票，看看它到底在玩什么花样！",
                "structure": {
                    "introduction": "首先让我们认识一下{ticker}这位'演员'。",
                    "technical_analysis": "技术面就像过山车，{technical_humor}",
                    "trend_analysis": "趋势分析告诉我们，{trend_humor}",
                    "risk_assessment": "风险提示：{risk_humor}",
                    "conclusion": "总结一下，{conclusion_humor}"
                },
                "closing": "好了，{ticker}的表演就到这里，记住投资有风险，入市需谨慎！",
                "bgm_style": "upbeat_fun",
                "visual_style": "animated_charts"
            },
            "brief": {
                "opening": "快速分析{ticker}。",
                "structure": {
                    "key_points": "关键要点：{key_points}",
                    "conclusion": "结论：{investment_conclusion}"
                },
                "closing": "分析完毕。",
                "bgm_style": "minimal",
                "visual_style": "simple_charts"
            }
        }
    
    def _generate_video_script(self, ticker: str, analysis_result: Dict[str, Any], 
                              style: str, duration: str) -> Dict[str, Any]:
        """生成视频脚本"""
        template = self.video_templates.get(style, self.video_templates["professional"])
        
        # 提取分析数据
        technical_analysis = analysis_result.get("technical_analysis", {})
        trend_analysis = analysis_result.get("trend_analysis", {})
        investment_suggestion = analysis_result.get("investment_suggestion", {})
        key_points = analysis_result.get("key_points", [])
        risk_level = analysis_result.get("risk_level", "medium")
        
        # 生成脚本内容
        script = {
            "title": f"{ticker}股票分析视频",
            "opening": template["opening"].format(ticker=ticker),
            "scenes": [],
            "closing": template["closing"].format(ticker=ticker),
            "total_duration": self._estimate_duration(duration),
            "bgm_style": template["bgm_style"],
            "visual_style": template["visual_style"]
        }
        
        # 生成各个场景
        scene_duration = self._get_scene_duration(duration)
        
        for section, template_text in template["structure"].items():
            scene_content = self._generate_scene_content(
                section, template_text, ticker, technical_analysis, 
                trend_analysis, investment_suggestion, key_points, risk_level, style
            )
            
            scene = {
                "scene_id": len(script["scenes"]) + 1,
                "section": section,
                "content": scene_content["text"],
                "duration": scene_duration,
                "visual_elements": scene_content["visuals"],
                "subtitle_timing": scene_content["timing"]
            }
            script["scenes"].append(scene)
        
        return script
    
    def _generate_scene_content(self, section: str, template: str, ticker: str,
                               technical_analysis: Dict, trend_analysis: Dict,
                               investment_suggestion: Dict, key_points: List[str],
                               risk_level: str, style: str) -> Dict[str, Any]:
        """生成场景内容"""
        if section == "introduction":
            text = template.format(ticker=ticker)
            visuals = ["stock_logo", "price_chart"]
            timing = [(0, 2), (2, 5)]
        
        elif section == "technical_analysis":
            if style == "humor":
                technical_summary = self._generate_technical_humor(technical_analysis)
                text = template.format(technical_humor=technical_summary)
            else:
                technical_summary = self._generate_technical_summary(technical_analysis, key_points)
                text = template.format(technical_summary=technical_summary)
            visuals = ["ma_chart", "rsi_chart", "macd_chart"]
            timing = [(0, 3), (3, 6), (6, 9)]
        
        elif section == "trend_analysis":
            if style == "humor":
                trend_summary = self._generate_trend_humor(trend_analysis)
                text = template.format(trend_humor=trend_summary)
            else:
                trend_summary = self._generate_trend_summary(trend_analysis)
                text = template.format(trend_summary=trend_summary)
            visuals = ["trend_chart", "volume_chart"]
            timing = [(0, 4), (4, 7)]
        
        elif section == "risk_assessment":
            if style == "humor":
                risk_summary = self._generate_risk_humor(risk_level)
                text = template.format(risk_humor=risk_summary)
            else:
                risk_summary = self._generate_risk_summary(risk_level, technical_analysis)
                text = template.format(risk_summary=risk_summary)
            visuals = ["risk_meter", "warning_icon"]
            timing = [(0, 3), (3, 5)]
        
        elif section == "conclusion":
            if style == "humor":
                conclusion = self._generate_conclusion_humor(investment_suggestion)
                text = template.format(conclusion_humor=conclusion)
            else:
                conclusion = self._generate_investment_conclusion(investment_suggestion)
                text = template.format(investment_conclusion=conclusion)
            visuals = ["summary_chart", "action_icon"]
            timing = [(0, 4), (4, 6)]
        
        elif section == "key_points":
            points_text = "；".join(key_points[:3]) if key_points else "暂无特殊要点"
            text = template.format(key_points=points_text)
            visuals = ["bullet_points", "highlight_chart"]
            timing = [(0, 2), (2, 5)]
        
        else:
            text = template
            visuals = ["default_chart"]
            timing = [(0, 3)]
        
        return {
            "text": text,
            "visuals": visuals,
            "timing": timing
        }
    
    def _generate_technical_summary(self, technical_analysis: Dict, key_points: List[str]) -> str:
        """生成技术面摘要"""
        ma_signal = technical_analysis.get("ma_signal", "neutral")
        rsi_signal = technical_analysis.get("rsi_signal", "neutral")
        macd_signal = technical_analysis.get("macd_signal", "neutral")
        
        summary = f"移动平均线显示{ma_signal}信号，RSI指标处于{rsi_signal}状态，MACD指标呈现{macd_signal}趋势。"
        
        if key_points:
            summary += f"关键要点包括：{key_points[0]}"
        
        return summary
    
    def _generate_technical_humor(self, technical_analysis: Dict) -> str:
        """生成幽默的技术面分析"""
        ma_signal = technical_analysis.get("ma_signal", "neutral")
        rsi_signal = technical_analysis.get("rsi_signal", "neutral")
        
        if rsi_signal == "overbought":
            return f"MA指标{ma_signal}，RSI超买就像吃撑了需要消化！"
        elif rsi_signal == "oversold":
            return f"MA指标{ma_signal}，RSI超卖就像饿坏了需要补充能量！"
        else:
            return f"MA指标{ma_signal}，RSI{rsi_signal}，整体还算正常！"
    
    def _generate_trend_summary(self, trend_analysis: Dict) -> str:
        """生成趋势摘要"""
        short_term = trend_analysis.get("short_term", "unknown")
        long_term = trend_analysis.get("long_term", "unknown")
        strength = trend_analysis.get("strength", "weak")
        price_change = trend_analysis.get("price_change_percent", 0)
        
        return f"短期趋势{short_term}，长期趋势{long_term}，趋势强度{strength}，价格变化{price_change}%。"
    
    def _generate_trend_humor(self, trend_analysis: Dict) -> str:
        """生成幽默的趋势分析"""
        short_term = trend_analysis.get("short_term", "unknown")
        strength = trend_analysis.get("strength", "weak")
        
        if strength == "strong":
            return f"短期趋势{short_term}，而且力度很强，就像开足马力的跑车！"
        else:
            return f"短期趋势{short_term}，力度一般，就像慢慢散步。"
    
    def _generate_risk_summary(self, risk_level: str, technical_analysis: Dict) -> str:
        """生成风险摘要"""
        if risk_level == "high":
            return "当前风险等级较高，建议谨慎操作，严格控制仓位。"
        elif risk_level == "medium":
            return "风险等级中等，建议适度参与，注意风险控制。"
        else:
            return "风险相对较低，但仍需注意市场变化。"
    
    def _generate_risk_humor(self, risk_level: str) -> str:
        """生成幽默的风险提示"""
        if risk_level == "high":
            return "高风险就像高空走钢丝，需要格外小心！"
        elif risk_level == "medium":
            return "中等风险就像开车，需要遵守交通规则！"
        else:
            return "低风险就像散步，相对安全但也要看路！"
    
    def _generate_investment_conclusion(self, investment_suggestion: Dict) -> str:
        """生成投资结论"""
        action = investment_suggestion.get("action", "hold")
        confidence = investment_suggestion.get("confidence", "low")
        reasoning = investment_suggestion.get("reasoning", "")
        
        action_text = "买入" if action == "buy" else "卖出" if action == "sell" else "观望"
        confidence_text = "高" if confidence == "high" else "中" if confidence == "medium" else "低"
        
        return f"综合判断建议{action_text}操作，信心度{confidence_text}。{reasoning}。"
    
    def _generate_conclusion_humor(self, investment_suggestion: Dict) -> str:
        """生成幽默的结论"""
        action = investment_suggestion.get("action", "hold")
        confidence = investment_suggestion.get("confidence", "low")
        
        action_text = "买入" if action == "buy" else "卖出" if action == "sell" else "观望"
        confidence_text = "高" if confidence == "high" else "中" if confidence == "medium" else "低"
        
        return f"综合判断建议{action_text}，信心度{confidence_text}。记住：投资有风险，入市需谨慎！"
    
    def _estimate_duration(self, duration_type: str) -> int:
        """估算视频总时长（秒）"""
        durations = {
            "short": 30,
            "medium": 60,
            "long": 120
        }
        return durations.get(duration_type, 60)
    
    def _get_scene_duration(self, duration_type: str) -> int:
        """获取每个场景的时长（秒）"""
        scene_durations = {
            "short": 5,
            "medium": 10,
            "long": 20
        }
        return scene_durations.get(duration_type, 10)
    
    def _create_production_plan(self, script: Dict[str, Any], aspect_ratio: str, language: str) -> Dict[str, Any]:
        """创建视频制作计划"""
        return {
            "tts_requirements": {
                "voice_type": "professional" if script["visual_style"] == "charts_and_graphs" else "friendly",
                "language": language,
                "speed": 1.0,
                "pitch": 1.0
            },
            "visual_requirements": {
                "aspect_ratio": aspect_ratio,
                "resolution": self._get_resolution(aspect_ratio),
                "style": script["visual_style"],
                "charts_needed": self._extract_chart_requirements(script),
                "animations": self._get_animation_requirements(script)
            },
            "audio_requirements": {
                "bgm_style": script["bgm_style"],
                "volume_level": 0.3,
                "fade_in_out": True
            },
            "subtitle_requirements": {
                "language": language,
                "font_size": "medium",
                "position": "bottom",
                "color": "white",
                "background": "black"
            },
            "output_specifications": {
                "format": "mp4",
                "codec": "h264",
                "bitrate": "2000k",
                "fps": 30
            }
        }
    
    def _get_resolution(self, aspect_ratio: str) -> str:
        """根据宽高比获取分辨率"""
        resolutions = {
            "16:9": "1920x1080",
            "9:16": "1080x1920",
            "1:1": "1080x1080"
        }
        return resolutions.get(aspect_ratio, "1920x1080")
    
    def _extract_chart_requirements(self, script: Dict[str, Any]) -> List[str]:
        """提取图表需求"""
        charts = set()
        for scene in script["scenes"]:
            for visual in scene["visual_elements"]:
                if "chart" in visual:
                    charts.add(visual)
        return list(charts)
    
    def _get_animation_requirements(self, script: Dict[str, Any]) -> List[str]:
        """获取动画需求"""
        animations = []
        for scene in script["scenes"]:
            if scene["section"] in ["technical_analysis", "trend_analysis"]:
                animations.extend(["fade_in", "slide_up", "highlight"])
            elif scene["section"] == "conclusion":
                animations.extend(["zoom_in", "pulse"])
        return list(set(animations))
