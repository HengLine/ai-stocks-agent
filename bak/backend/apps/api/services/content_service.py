#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
内容服务模块

负责管理文章和视频的生成流程
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from datetime import datetime
import os
import uuid
import json

from ..models import GeneratedContent, VideoProductionJob, ContentReview
from .review_service import ReviewService
from .vector_service import VectorStoreClient
from .content_generator import global_content_generator
from .video_processor import global_video_processor

# 创建全局实例
global_vector_store = VectorStoreClient().client


class ContentService:
    """应用服务：负责文章/视频生成编排，整合内容生成、审核、润色流程，支持人工审核接口"""

    def __init__(self) -> None:
        self.content_generator = global_content_generator
        self.review_service = ReviewService()
        self.vector_store = global_vector_store

    def generate_content(self,
                        content_type: str,  # article 或 video
                        ticker: str,
                        analysis_data: Dict[str, Any] = None,
                        params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        生成内容（文章或视频）的统一入口
        
        Args:
            content_type: 内容类型，article或video
            ticker: 股票代码
            analysis_data: 分析数据，可选，如果不提供会自动获取
            params: 生成参数
        
        Returns:
            生成结果
        """
        # 如果没有提供分析数据，这里可以添加获取分析数据的逻辑
        # 为了简化，这里假设调用方已经提供了分析数据
        if analysis_data is None:
            analysis_data = self._get_analysis_data(ticker)
            
            if analysis_data is None:
                return {"success": False, "error": "获取分析数据失败"}
        
        # 调用内容生成器生成内容
        result = self.content_generator.generate_content(
            content_type,
            ticker,
            analysis_data,
            params
        )
        
        if result["success"]:
            # 创建或更新数据库记录
            content_obj = self._save_content_to_db(result)
            
            # 返回包含数据库ID的结果
            result["db_id"] = content_obj.id
        
        return result
    
    def _get_analysis_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """获取分析数据（简化实现）"""
        # 这里可以实现获取分析数据的逻辑
        # 为了简化，返回一个模拟的分析数据
        return {
            "time_frame": "3个月",
            "technical_analysis": {
                "ma_signal": "金叉",
                "rsi_signal": "超买",
                "macd_signal": "看涨"
            },
            "fundamental_analysis": {
                "revenue_growth": "30%",
                "gross_margin": "稳定"
            },
            "risk_assessment": {
                "risk_level": "中等"
            }
        }
    
    def _save_content_to_db(self, content_result: Dict[str, Any]) -> GeneratedContent:
        """保存内容到数据库"""
        # 创建或更新数据库记录
        content_id = content_result["content_id"]
        metadata = content_result["metadata"]
        
        # 提取内容中的标题和内容
        content_data = content_result["content"]
        if metadata["type"] == "article":
            title = content_data.get("title", "")
            text_content = content_data.get("content", "")
        else:
            title = content_data.get("script", {}).get("title", "")
            text_content = str(content_data)
        
        # 创建数据库记录
        content_obj, created = GeneratedContent.objects.get_or_create(
            content_id=content_id,
            defaults={
                "ticker": metadata["ticker"],
                "title": title,
                "content": text_content,
                "content_type": metadata["type"],
                "status": content_result["status"],
                "metadata": metadata,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
        )
        
        if not created:
            # 更新现有记录
            content_obj.title = title
            content_obj.content = text_content
            content_obj.status = content_result["status"]
            content_obj.metadata = metadata
            content_obj.updated_at = datetime.now()
            content_obj.save()
        
        return content_obj
    
    def review_content_with_ai(self,
                             content_id: str,
                             focus: List[str] = None) -> Dict[str, Any]:
        """
        使用AI审核内容
        
        Args:
            content_id: 内容ID
            focus: 审核重点
        
        Returns:
            审核结果
        """
        # 调用内容生成器的AI审核功能
        result = self.content_generator.review_content(content_id, focus)
        
        if result["success"]:
            # 更新数据库记录
            self._update_content_status(content_id, result["status"])
            
            # 创建审核记录
            self._save_review_to_db(content_id, result["review"], "ai")
        
        return result
    
    def review_content_manually(self,
                              content_id: str,
                              reviewer_id: int,
                              issues: List[Dict[str, Any]],
                              suggestions: List[str],
                              approved: bool = False) -> Dict[str, Any]:
        """
        人工审核内容接口
        
        Args:
            content_id: 内容ID
            reviewer_id: 审核人ID
            issues: 问题列表
            suggestions: 修改建议
            approved: 是否通过审核
        
        Returns:
            审核结果
        """
        try:
            # 保存人工审核记录
            review_obj = self._save_review_to_db(
                content_id,
                {"issues": issues, "suggestions": suggestions},
                "manual",
                reviewer_id,
                approved
            )
            
            # 更新内容状态
            status = "approved" if approved else "rejected"
            self._update_content_status(content_id, status)
            
            return {
                "success": True,
                "review_id": review_obj.id,
                "status": status,
                "message": "人工审核记录已保存"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "保存人工审核记录失败"
            }
    
    def refine_content(self,
                      content_id: str,
                      tone: str = "professional",
                      language: str = "zh-CN") -> Dict[str, Any]:
        """
        润色内容（使用Grammarly类模型）
        
        Args:
            content_id: 内容ID
            tone: 语气
            language: 语言
        
        Returns:
            润色结果
        """
        # 调用内容生成器的润色功能
        result = self.content_generator.refine_content(content_id, tone, language)
        
        if result["success"]:
            # 更新数据库记录
            self._update_content_status(content_id, result["status"])
            
            # 更新内容文本
            self._update_content_text(content_id, result["refined_text"])
        
        return result
    
    def format_content(self,
                      content_id: str,
                      target_format: str) -> Dict[str, Any]:
        """
        格式化内容为指定格式（Markdown/HTML/PDF）
        
        Args:
            content_id: 内容ID
            target_format: 目标格式
        
        Returns:
            格式化结果
        """
        # 调用内容生成器的格式化功能
        result = self.content_generator.format_content(content_id, target_format)
        
        if result["success"]:
            # 更新内容状态为已完成
            self._update_content_status(content_id, "completed")
            
            # 保存格式化后的内容
            self._save_formatted_content(content_id, result["formatted_content"], target_format)
        
        return result
    
    def _save_review_to_db(self,
                         content_id: str,
                         review_data: Dict[str, Any],
                         review_type: str,  # ai 或 manual
                         reviewer_id: int = None,
                         approved: bool = False) -> ContentReview:
        """保存审核记录到数据库"""
        # 获取内容对象
        try:
            content_obj = GeneratedContent.objects.get(content_id=content_id)
        except GeneratedContent.DoesNotExist:
            raise ValueError(f"内容不存在: {content_id}")
        
        # 创建审核记录
        review_obj = ContentReview.objects.create(
            content=content_obj,
            review_data=review_data,
            review_type=review_type,
            reviewer_id=reviewer_id,
            approved=approved,
            reviewed_at=datetime.now()
        )
        
        return review_obj
    
    def _update_content_status(self,
                             content_id: str,
                             status: str) -> None:
        """更新内容状态"""
        try:
            content_obj = GeneratedContent.objects.get(content_id=content_id)
            content_obj.status = status
            content_obj.updated_at = datetime.now()
            content_obj.save()
        except GeneratedContent.DoesNotExist:
            # 如果内容不存在，可以记录日志或抛出异常
            pass
    
    def _update_content_text(self,
                           content_id: str,
                           refined_text: str) -> None:
        """更新内容文本"""
        try:
            content_obj = GeneratedContent.objects.get(content_id=content_id)
            content_obj.content = refined_text
            content_obj.updated_at = datetime.now()
            content_obj.save()
        except GeneratedContent.DoesNotExist:
            # 如果内容不存在，可以记录日志或抛出异常
            pass
    
    def _save_formatted_content(self,
                              content_id: str,
                              formatted_content: str,
                              format_type: str) -> None:
        """保存格式化后的内容"""
        try:
            content_obj = GeneratedContent.objects.get(content_id=content_id)
            content_obj.formatted_content = formatted_content
            content_obj.format_type = format_type
            content_obj.updated_at = datetime.now()
            content_obj.save()
        except GeneratedContent.DoesNotExist:
            # 如果内容不存在，可以记录日志或抛出异常
            pass
    
    # 视频相关方法保持，但整合到新的流程中
    def enqueue_video_job(self, ticker: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """将视频生成任务加入队列"""
        try:
            # 获取分析数据（从向量数据库）
            analysis_data = self._get_analysis_data_from_vector_store(ticker)
            if not analysis_data:
                # 如果向量数据库中没有，生成新的分析数据
                # 注意：这里简化处理，实际应该调用AnalysisAgent
                analysis_data = self._generate_default_analysis_data(ticker)
                
                # 保存到向量数据库
                self._save_analysis_data_to_vector_store(ticker, analysis_data)
            
            # 生成视频内容（脚本）
            generate_result = self.generate_content(
                content_type="video",
                ticker=ticker,
                analysis_data=analysis_data,
                params=params
            )
            
            if generate_result["success"]:
                content_id = generate_result["content_id"]
                
                # 创建视频制作任务
                video_job = {
                    "job_id": str(uuid.uuid4()),
                    "content_id": content_id,
                    "ticker": ticker,
                    "status": "pending",
                    "created_at": datetime.now().isoformat(),
                    "params": params
                }
                
                # 处理视频制作任务
                self._process_video_job(video_job, analysis_data)
                
                return {
                    "success": True,
                    "job_id": video_job["job_id"],
                    "content_id": content_id,
                    "status": "processing",
                    "message": "视频生成任务已开始处理"
                }
            else:
                return {
                    "success": False,
                    "error": generate_result["error"],
                    "message": "生成视频脚本失败"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "视频任务创建失败"
            }
    
    def _process_video_job(self, video_job: Dict[str, Any], analysis_data: Dict[str, Any]):
        """处理视频制作任务，调用VideoProcessor完成完整视频生成流程"""
        try:
            # 获取视频内容（脚本）
            content = GeneratedContent.objects.get(content_id=video_job["content_id"])
            content_data = {
                "script": content.content,
                "production_plan": {}
            }
            
            # 更新任务状态为处理中
            video_job["status"] = "processing"
            video_job["started_at"] = datetime.now().isoformat()
            
            # 使用VideoProcessor处理完整视频生成流程
            video_result = global_video_processor.process_video_generation(
                ticker=video_job["ticker"],
                analysis_data=analysis_data,
                params=video_job["params"]
            )
            
            if video_result["success"]:
                # 更新任务状态为已完成
                video_job["status"] = "completed"
                video_job["completed_at"] = datetime.now().isoformat()
                video_job["video_path"] = video_result["video_path"]
                video_job["thumbnail_path"] = video_result["thumbnail_path"]
                video_job["cdn_url"] = video_result["cdn_url"]
                video_job["thumbnail_cdn_url"] = video_result["thumbnail_cdn_url"]
                video_job["duration"] = video_result["duration"]
                
                # 更新内容数据
                content_data["video_result"] = video_result
                
                # 更新内容状态
                content.status = "completed"
                content.metadata = json.dumps({
                    **json.loads(content.metadata if content.metadata else "{}"),
                    "video_job": video_job,
                    "cdn_url": video_result["cdn_url"],
                    "thumbnail_cdn_url": video_result["thumbnail_cdn_url"]
                })
                content.save()
                
                print(f"视频任务处理成功: {video_job['job_id']}")
            else:
                # 视频处理失败
                video_job["status"] = "failed"
                video_job["error"] = video_result["error"]
                video_job["failed_step"] = video_result.get("step", "unknown")
                
                # 更新内容状态
                content.status = "failed"
                content.metadata = json.dumps({
                    **json.loads(content.metadata if content.metadata else "{}"),
                    "video_job": video_job,
                    "error": video_result["error"]
                })
                content.save()
                
                print(f"视频任务处理失败: {video_job['job_id']}, 错误: {video_result['error']}, 步骤: {video_result.get('step', 'unknown')}")
                
        except Exception as e:
            video_job["status"] = "failed"
            video_job["error"] = str(e)
            
            # 更新内容状态
            content = GeneratedContent.objects.get(content_id=video_job["content_id"])
            content.status = "failed"
            content.metadata = json.dumps({
                **json.loads(content.metadata if content.metadata else "{}"),
                "video_job": video_job,
                "error": str(e)
            })
            content.save()
            
            print(f"视频任务处理失败: {e}")
            
    def _get_analysis_data_from_vector_store(self, ticker: str) -> Optional[Dict[str, Any]]:
        """从向量数据库获取分析数据"""
        try:
            results = global_vector_store.search({
                "query": f"{ticker} analysis data",
                "top_k": 1
            })
            if results and len(results) > 0:
                return results[0].get("metadata", {})
            return None
        except Exception:
            return None
            
    def _generate_default_analysis_data(self, ticker: str) -> Dict[str, Any]:
        """生成默认分析数据"""
        return {
            "ticker": ticker,
            "time_frame": "3个月",
            "technical_analysis": {
                "ma_signal": "金叉",
                "rsi_signal": "超买",
                "macd_signal": "看涨"
            },
            "fundamental_analysis": {
                "revenue_growth": "30%",
                "gross_margin": "稳定"
            },
            "risk_assessment": {
                "risk_level": "中等"
            }
        }
        
    def _save_analysis_data_to_vector_store(self, ticker: str, analysis_data: Dict[str, Any]) -> None:
        """保存分析数据到向量数据库"""
        try:
            global_vector_store.insert({
                "id": f"analysis_{ticker}_{datetime.now().timestamp()}",
                "text": f"{ticker} analysis data",
                "metadata": analysis_data
            })
        except Exception:
            # 记录错误但不中断流程
            pass


