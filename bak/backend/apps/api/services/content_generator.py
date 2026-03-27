from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import os
from enum import Enum

from ..agents.writing_agent import WritingAgent
from ..agents.video_agent import VideoAgent
from ..agents.base import AgentResult
from .review_service import ReviewService
from .vector_store import global_vector_store


class ContentType(Enum):
    ARTICLE = "article"
    VIDEO = "video"


class ContentStatus(Enum):
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    REVIEWED = "reviewed"
    REFINED = "refined"
    COMPLETED = "completed"
    PUBLISHED = "published"


class ContentFormat(Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    MP4 = "mp4"


class ContentGenerator:
    """内容生成器：整合文章和视频生成、审核、润色流程"""

    def __init__(self):
        self.writing_agent = WritingAgent()
        self.video_agent = VideoAgent()
        self.review_service = ReviewService()
        self.vector_store = global_vector_store

    def generate_content(self,
                         content_type: str,
                         ticker: str,
                         analysis_data: Dict[str, Any],
                         params: Dict[str, Any] = None,
                         context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        生成内容的主入口函数
        
        Args:
            content_type: 内容类型，article或video
            ticker: 股票代码
            analysis_data: 分析数据
            params: 生成参数
            context: 上下文数据
        
        Returns:
            生成结果，包含内容、状态和元数据
        """
        params = params or {}
        context = context or {}
        
        # 确保上下文包含必要信息
        context = {
            **context,
            "ticker": ticker,
            "analysis": analysis_data,
            "timestamp": datetime.now().isoformat()
        }
        
        # 根据内容类型调用不同的生成器
        if content_type.lower() == ContentType.ARTICLE.value:
            return self._generate_article(ticker, analysis_data, params, context)
        elif content_type.lower() == ContentType.VIDEO.value:
            return self._generate_video(ticker, analysis_data, params, context)
        else:
            raise ValueError(f"不支持的内容类型: {content_type}")

    def _generate_article(self, ticker: str, analysis_data: Dict[str, Any],
                         params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """生成文章内容"""
        # 调用WritingAgent生成初稿
        result = self.writing_agent.run(params, context)
        
        if not result.success:
            return {
                "success": False,
                "error": result.error,
                "status": ContentStatus.DRAFT.value,
                "content_id": self._generate_content_id(ticker, ContentType.ARTICLE.value)
            }
        
        article_data = result.data
        
        # 生成内容ID
        content_id = self._generate_content_id(ticker, ContentType.ARTICLE.value)
        
        # 保存到向量数据库
        self._save_content_to_memory(content_id, article_data, ContentType.ARTICLE.value)
        
        # 准备返回结果
        return {
            "success": True,
            "content_id": content_id,
            "content": article_data,
            "status": ContentStatus.DRAFT.value,
            "metadata": {
                "ticker": ticker,
                "type": ContentType.ARTICLE.value,
                "template": article_data.get("template", "professional"),
                "format": article_data.get("format", "markdown"),
                "generated_at": datetime.now().isoformat()
            }
        }

    def _generate_video(self, ticker: str, analysis_data: Dict[str, Any],
                        params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """生成视频脚本和制作计划"""
        # 调用VideoAgent生成视频脚本
        result = self.video_agent.run(params, context)
        
        if not result.success:
            return {
                "success": False,
                "error": result.error,
                "status": ContentStatus.DRAFT.value,
                "content_id": self._generate_content_id(ticker, ContentType.VIDEO.value)
            }
        
        video_data = result.data
        
        # 生成内容ID
        content_id = self._generate_content_id(ticker, ContentType.VIDEO.value)
        
        # 保存到向量数据库
        self._save_content_to_memory(content_id, video_data, ContentType.VIDEO.value)
        
        # 准备返回结果
        return {
            "success": True,
            "content_id": content_id,
            "content": video_data,
            "status": ContentStatus.DRAFT.value,
            "metadata": {
                "ticker": ticker,
                "type": ContentType.VIDEO.value,
                "style": video_data.get("metadata", {}).get("style", "professional"),
                "duration": video_data.get("metadata", {}).get("duration", "medium"),
                "aspect_ratio": video_data.get("metadata", {}).get("aspect_ratio", "16:9"),
                "generated_at": datetime.now().isoformat()
            }
        }

    def review_content(self, content_id: str, focus: List[str] = None) -> Dict[str, Any]:
        """
        AI审核内容
        
        Args:
            content_id: 内容ID
            focus: 审核重点
        
        Returns:
            审核结果
        """
        # 从向量数据库获取内容
        content = self._get_content_from_memory(content_id)
        
        if not content:
            return {
                "success": False,
                "error": f"找不到内容: {content_id}"
            }
        
        # 提取文本内容进行审核
        text_to_review = self._extract_text_for_review(content)
        
        # 调用审核服务
        review_result = self.review_service.ai_review(text_to_review, focus)
        
        # 更新内容状态
        self._update_content_status(content_id, ContentStatus.REVIEWED.value)
        
        return {
            "success": True,
            "content_id": content_id,
            "review": review_result["review"],
            "status": ContentStatus.REVIEWED.value
        }

    def refine_content(self, content_id: str, tone: str = "professional", 
                      language: str = "zh-CN") -> Dict[str, Any]:
        """
        润色内容
        
        Args:
            content_id: 内容ID
            tone: 语气
            language: 语言
        
        Returns:
            润色结果
        """
        # 从向量数据库获取内容
        content = self._get_content_from_memory(content_id)
        
        if not content:
            return {
                "success": False,
                "error": f"找不到内容: {content_id}"
            }
        
        # 提取文本内容进行润色
        text_to_refine = self._extract_text_for_review(content)
        
        # 调用润色服务
        refine_result = self.review_service.refine(text_to_refine, tone, language)
        
        # 更新润色后的内容
        updated_content = self._update_content_with_refined_text(content, refine_result["refined_text"])
        
        # 保存更新后的内容
        self._update_content_in_memory(content_id, updated_content)
        
        # 更新内容状态
        self._update_content_status(content_id, ContentStatus.REFINED.value)
        
        return {
            "success": True,
            "content_id": content_id,
            "refined_text": refine_result["refined_text"],
            "status": ContentStatus.REFINED.value
        }

    def format_content(self, content_id: str, target_format: str) -> Dict[str, Any]:
        """
        格式化内容为指定格式
        
        Args:
            content_id: 内容ID
            target_format: 目标格式
        
        Returns:
            格式化后的内容
        """
        # 从向量数据库获取内容
        content = self._get_content_from_memory(content_id)
        
        if not content:
            return {
                "success": False,
                "error": f"找不到内容: {content_id}"
            }
        
        # 检查内容类型
        content_type = content.get("metadata", {}).get("type", ContentType.ARTICLE.value)
        
        if content_type == ContentType.ARTICLE.value:
            # 文章格式化
            return self._format_article(content, target_format)
        elif content_type == ContentType.VIDEO.value:
            # 视频格式处理
            return self._format_video(content, target_format)
        
        return {
            "success": False,
            "error": f"不支持的内容类型格式化: {content_type}"
        }

    def _format_article(self, content: Dict[str, Any], target_format: str) -> Dict[str, Any]:
        """格式化文章内容"""
        # 确保目标格式有效
        if target_format not in [f.value for f in ContentFormat if f.value != ContentFormat.MP4.value]:
            return {
                "success": False,
                "error": f"不支持的文章格式: {target_format}"
            }
        
        # 获取文章内容
        article_content = content.get("content", {})
        
        # 调用WritingAgent的格式转换功能
        if hasattr(self.writing_agent, "_format_content"):
            formatted_content = self.writing_agent._format_content(
                article_content.get("content", ""), 
                target_format
            )
        else:
            # 如果没有内置的格式转换，使用简单的处理
            formatted_content = self._simple_format_conversion(
                article_content.get("content", ""),
                target_format
            )
        
        return {
            "success": True,
            "formatted_content": formatted_content,
            "format": target_format
        }

    def _format_video(self, content: Dict[str, Any], target_format: str) -> Dict[str, Any]:
        """处理视频格式需求"""
        if target_format != ContentFormat.MP4.value:
            return {
                "success": False,
                "error": f"视频只支持MP4格式，请求的格式: {target_format}"
            }
        
        # 提取视频制作计划
        production_plan = content.get("content", {}).get("production_plan", {})
        
        return {
            "success": True,
            "production_plan": production_plan,
            "format": ContentFormat.MP4.value
        }

    def _simple_format_conversion(self, text: str, target_format: str) -> str:
        """简单的格式转换"""
        if target_format == ContentFormat.HTML.value:
            # 简单的Markdown到HTML转换
            html = text.replace("# ", "<h1>")
            html = html.replace("## ", "<h2>")
            html = html.replace("### ", "<h3>")
            html = html.replace("\n", "<br>")
            return html
        elif target_format == ContentFormat.PDF.value:
            # 返回原始文本，实际PDF生成需要额外的库支持
            return text
        else:
            return text

    def _generate_content_id(self, ticker: str, content_type: str) -> str:
        """生成唯一的内容ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{content_type}_{ticker}_{timestamp}"

    def _save_content_to_memory(self, content_id: str, content: Dict[str, Any], content_type: str) -> None:
        """保存内容到向量数据库"""
        try:
            # 提取关键词用于向量存储
            keywords = self._extract_keywords(content, content_type)
            
            # 构建文档
            document = {
                "id": content_id,
                "content": str(content),
                "metadata": {
                    "type": content_type,
                    "keywords": keywords,
                    "created_at": datetime.now().isoformat()
                }
            }
            
            # 保存到向量数据库
            self.vector_store.add_documents([document])
        except Exception as e:
            print(f"保存内容到向量数据库失败: {e}")

    def _get_content_from_memory(self, content_id: str) -> Optional[Dict[str, Any]]:
        """从向量数据库获取内容"""
        try:
            # 根据ID检索文档
            results = self.vector_store.search_documents(
                query=f"content_id:{content_id}",
                limit=1
            )
            
            if results and len(results) > 0:
                import json
                # 尝试解析内容
                try:
                    content = json.loads(results[0].get("content", "{}"))
                    return {
                        "content": content,
                        "metadata": results[0].get("metadata", {})
                    }
                except:
                    return {
                        "content": results[0].get("content", {}),
                        "metadata": results[0].get("metadata", {})
                    }
        except Exception as e:
            print(f"从向量数据库获取内容失败: {e}")
        
        return None

    def _update_content_in_memory(self, content_id: str, updated_content: Dict[str, Any]) -> None:
        """更新向量数据库中的内容"""
        try:
            # 先删除旧内容
            self.vector_store.delete_documents([content_id])
            
            # 保存更新后的内容
            content_type = updated_content.get("metadata", {}).get("type", ContentType.ARTICLE.value)
            self._save_content_to_memory(content_id, updated_content.get("content", {}), content_type)
        except Exception as e:
            print(f"更新向量数据库内容失败: {e}")

    def _update_content_status(self, content_id: str, status: str) -> None:
        """更新内容状态"""
        try:
            content = self._get_content_from_memory(content_id)
            if content:
                content["metadata"]["status"] = status
                content_type = content.get("metadata", {}).get("type", ContentType.ARTICLE.value)
                self._update_content_in_memory(content_id, content)
        except Exception as e:
            print(f"更新内容状态失败: {e}")

    def _extract_text_for_review(self, content: Dict[str, Any]) -> str:
        """提取用于审核的文本内容"""
        content_data = content.get("content", {})
        content_type = content.get("metadata", {}).get("type", ContentType.ARTICLE.value)
        
        if content_type == ContentType.ARTICLE.value:
            return content_data.get("content", "")
        elif content_type == ContentType.VIDEO.value:
            # 提取视频脚本文本
            script = content_data.get("script", {})
            text_parts = [
                script.get("title", ""),
                script.get("opening", ""),
            ]
            
            for scene in script.get("scenes", []):
                text_parts.append(scene.get("content", ""))
            
            text_parts.append(script.get("closing", ""))
            
            return "\n".join(text_parts)
        
        return str(content_data)

    def _update_content_with_refined_text(self, content: Dict[str, Any], 
                                        refined_text: str) -> Dict[str, Any]:
        """更新内容为润色后的文本"""
        content_data = content.get("content", {})
        content_type = content.get("metadata", {}).get("type", ContentType.ARTICLE.value)
        
        if content_type == ContentType.ARTICLE.value:
            # 更新文章内容
            content_data["content"] = refined_text
            content_data["refined"] = True
        elif content_type == ContentType.VIDEO.value:
            # 更新视频脚本
            if "script" in content_data:
                # 这里简化处理，实际可能需要更复杂的逻辑来更新视频脚本的各个部分
                # 简单起见，我们假设润色后的文本是整个脚本的组合
                content_data["script"]["refined_text"] = refined_text
                content_data["refined"] = True
        
        return {
            "content": content_data,
            "metadata": content.get("metadata", {})
        }

    def _extract_keywords(self, content: Dict[str, Any], content_type: str) -> List[str]:
        """提取内容关键词"""
        keywords = []
        
        # 从内容中提取关键词
        if content_type == ContentType.ARTICLE.value:
            # 提取文章关键词
            if isinstance(content, dict):
                # 从元数据中获取股票代码
                metadata = content.get("metadata", {})
                ticker = metadata.get("ticker", "")
                if ticker:
                    keywords.append(ticker)
                
                # 从分析数据中提取关键词
                analysis_data = content.get("analysis_data", {})
                if isinstance(analysis_data, dict):
                    # 提取技术指标关键词
                    technical = analysis_data.get("technical_analysis", {})
                    if isinstance(technical, dict):
                        for key, value in technical.items():
                            if isinstance(value, str) and value.lower() != "neutral":
                                keywords.append(value)
        elif content_type == ContentType.VIDEO.value:
            # 提取视频关键词
            if isinstance(content, dict):
                # 从元数据中获取
                metadata = content.get("metadata", {})
                ticker = metadata.get("ticker", "")
                if ticker:
                    keywords.append(ticker)
                
                # 从视频风格获取
                style = metadata.get("style", "")
                if style:
                    keywords.append(style)
        
        # 去重并返回
        return list(set(keywords))[:10]  # 限制最多10个关键词


# 创建全局内容生成器实例
global_content_generator = ContentGenerator()