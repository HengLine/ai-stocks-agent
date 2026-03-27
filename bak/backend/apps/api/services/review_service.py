from __future__ import annotations

from typing import Any, Dict, List

from .llm_service import LLMService


class ReviewService:
    """应用服务：负责文本润色和AI审核"""

    def __init__(self) -> None:
        self.llm = LLMService()

    def refine(self, text: str, tone: str = "professional", language: str = "zh-CN") -> Dict[str, Any]:
        prompt = (
            f"请以{tone}语气，用{language}对以下内容进行润色，使其更通顺、简洁、逻辑清晰，保留关键信息，不杜撰：\n\n"
            f"原文：\n{text}\n\n"
            "要求：\n- 不改变事实\n- 提升可读性\n- 给出改写后的完整文本"
        )
        refined = self.llm.chat_response(prompt, context=None)
        return {"success": True, "refined_text": refined, "metadata": {"tone": tone, "language": language}}

    def ai_review(self, text: str, focus: List[str] | None = None) -> Dict[str, Any]:
        focus = focus or ["事实准确性", "逻辑一致性", "用词风险"]
        prompt = (
            "你是内容审核专家，请审阅以下文本，从以下角度标注风险并提出修改建议："+", ".join(focus)+
            "。输出JSON，字段：issues(问题数组: {type,detail,level}), suggestions(修改建议数组)。\n\n文本：\n"+text
        )
        try:
            raw = self.llm.chat_response(prompt, context=None)
            import json as _json
            data = _json.loads(raw)
        except Exception:
            data = {"issues": [], "suggestions": ["解析失败，原始输出："+raw]}
        return {"success": True, "review": data}


