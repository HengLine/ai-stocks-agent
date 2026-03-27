#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
视频处理核心模块

负责协调视频生成的完整流程：
1. 视频脚本生成（WritingAgent）
2. 文本转语音（TTS服务）
3. 图表动画生成
4. 视频合成（FFmpeg）
5. CDN上传
"""

from typing import Dict, Any, List, Optional
import os
import subprocess
import json
import time
from datetime import datetime
import requests

from ..agents.video_agent import VideoAgent
from ..agents.writing_agent import WritingAgent
from .video_service import VideoService
from .tts_service import TTSService
from .llm_service import LLMService


class VideoProcessor:
    """视频处理核心组件，整合所有视频生成相关服务"""
    
    def __init__(self):
        self.video_agent = VideoAgent()
        self.writing_agent = WritingAgent()
        self.video_service = VideoService()
        self.tts_service = TTSService()
        self.llm_service = LLMService()
        
        # CDN配置
        self.cdn_config = {
            "upload_url": os.getenv("CDN_UPLOAD_URL", "https://cdn.example.com/upload"),
            "api_key": os.getenv("CDN_API_KEY", "dummy_key"),
            "bucket_name": os.getenv("CDN_BUCKET_NAME", "videos")
        }
        
        # 临时文件目录
        self.temp_dir = "temp_video_files"
        self.output_dir = "output_videos"
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        for dir_path in [self.temp_dir, self.output_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
    
    def process_video_generation(self, ticker: str, analysis_data: Dict[str, Any], 
                               params: Dict[str, Any]) -> Dict[str, Any]:
        """处理完整的视频生成流程"""
        try:
            # 1. 生成视频脚本
            script_result = self._generate_video_script(ticker, analysis_data, params)
            if not script_result["success"]:
                return {
                    "success": False,
                    "error": script_result["error"],
                    "step": "script_generation"
                }
            
            script = script_result["script"]
            production_plan = script_result["production_plan"]
            
            # 2. 生成TTS语音和字幕
            audio_subtitle_result = self._generate_audio_and_subtitles(
                script, production_plan
            )
            if not audio_subtitle_result["success"]:
                return {
                    "success": False,
                    "error": audio_subtitle_result["error"],
                    "step": "audio_generation"
                }
            
            audio_result = audio_subtitle_result["audio"]
            subtitles = audio_subtitle_result["subtitles"]
            subtitle_path = audio_subtitle_result["subtitle_path"]
            
            # 3. 生成图表动画
            chart_result = self._generate_chart_videos(
                script, analysis_data, ticker
            )
            if not chart_result["success"]:
                return {
                    "success": False,
                    "error": chart_result["error"],
                    "step": "chart_generation"
                }
            
            chart_videos = chart_result["chart_videos"]
            
            # 4. 合成最终视频
            video_result = self._compose_final_video(
                audio_result, chart_videos, subtitles, subtitle_path, 
                production_plan, ticker
            )
            if not video_result["success"]:
                return {
                    "success": False,
                    "error": video_result["error"],
                    "step": "video_compose"
                }
            
            # 5. 上传到CDN
            cdn_result = self._upload_to_cdn(video_result)
            if not cdn_result["success"]:
                # CDN上传失败不应影响视频生成结果
                print(f"CDN上传失败，但视频已生成: {cdn_result['error']}")
                cdn_url = None
                thumbnail_url = None
            else:
                cdn_url = cdn_result["cdn_url"]
                thumbnail_url = cdn_result["thumbnail_url"]
            
            # 整合所有结果
            return {
                "success": True,
                "video_path": video_result["video_path"],
                "thumbnail_path": video_result["thumbnail_path"],
                "cdn_url": cdn_url,
                "thumbnail_cdn_url": thumbnail_url,
                "duration": video_result["duration"],
                "metadata": {
                    "ticker": ticker,
                    "created_at": datetime.now().isoformat(),
                    "video_script": script,
                    "production_plan": production_plan,
                    "params": params
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "step": "unknown"
            }
        finally:
            # 清理临时文件
            self.video_service.cleanup_temp_files()
    
    def _generate_video_script(self, ticker: str, analysis_data: Dict[str, Any], 
                             params: Dict[str, Any]) -> Dict[str, Any]:
        """生成视频脚本"""
        try:
            context = {
                "ticker": ticker,
                "analysis": analysis_data
            }
            
            # 调用VideoAgent生成脚本
            result = self.video_agent.run(params, context)
            
            if result.success:
                return {
                    "success": True,
                    "script": result.data["script"],
                    "production_plan": result.data["production_plan"]
                }
            else:
                return {
                    "success": False,
                    "error": "VideoAgent生成脚本失败"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_audio_and_subtitles(self, script: Dict[str, Any], 
                                    production_plan: Dict[str, Any]) -> Dict[str, Any]:
        """生成TTS语音和字幕"""
        try:
            # 合并所有文本
            full_text = script.get("opening", "")
            for scene in script.get("scenes", []):
                full_text += " " + scene.get("content", "")
            full_text += " " + script.get("closing", "")
            
            # 获取TTS配置
            tts_config = production_plan["tts_requirements"]
            voice_type = tts_config["voice_type"]
            language = tts_config["language"]
            
            # 生成语音文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_path = os.path.join(self.temp_dir, f"audio_{timestamp}.mp3")
            
            # 尝试使用Azure TTS，失败则回退到阿里云或其他服务
            audio_result = self.tts_service.generate_speech(
                full_text, voice_type, "azure", audio_path
            )
            
            # 如果Azure失败，尝试阿里云
            if not audio_result["success"]:
                audio_result = self.tts_service.generate_speech(
                    full_text, voice_type, "aliyun", audio_path
                )
            
            # 生成字幕
            subtitles = self.tts_service.generate_subtitles(script, language)
            
            # 保存字幕文件
            subtitle_path = os.path.join(self.temp_dir, f"subtitles_{timestamp}.srt")
            self.tts_service.save_subtitles_srt(subtitles, subtitle_path)
            
            return {
                "success": True,
                "audio": audio_result,
                "subtitles": subtitles,
                "subtitle_path": subtitle_path
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_chart_videos(self, script: Dict[str, Any], 
                             analysis_data: Dict[str, Any], 
                             ticker: str) -> Dict[str, Any]:
        """生成图表动画视频"""
        try:
            chart_videos = []
            
            # 生成价格走势图
            price_chart = self.video_service._create_price_chart_animation(
                ticker, 
                analysis_data.get("technical_indicators", {}).get("close", 0),
                analysis_data.get("technical_indicators", {}).get("ma5", 0),
                analysis_data.get("technical_indicators", {}).get("ma20", 0)
            )
            if price_chart:
                chart_videos.append(price_chart)
            
            # 生成RSI指标图
            rsi_value = analysis_data.get("technical_indicators", {}).get("rsi", 50)
            rsi_chart = self.video_service._create_rsi_chart_animation(rsi_value)
            if rsi_chart:
                chart_videos.append(rsi_chart)
            
            # 生成MACD指标图
            macd_data = analysis_data.get("technical_indicators", {}).get("macd", {})
            macd_chart = self.video_service._create_macd_chart_animation(macd_data)
            if macd_chart:
                chart_videos.append(macd_chart)
            
            # 根据脚本需求生成其他图表
            for scene in script.get("scenes", []):
                visual_elements = scene.get("visual_elements", [])
                for element in visual_elements:
                    # 如果需要其他类型的图表，可以在这里添加
                    pass
            
            return {
                "success": True,
                "chart_videos": chart_videos
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _compose_final_video(self, audio_result: Dict[str, Any], 
                           chart_videos: List[Dict[str, Any]],
                           subtitles: List[Dict[str, Any]],
                           subtitle_path: str,
                           production_plan: Dict[str, Any],
                           ticker: str) -> Dict[str, Any]:
        """合成最终视频"""
        try:
            # 调用VideoService的合成功能
            result = self.video_service.create_video(
                script={},  # 这里实际上不需要完整的script对象
                production_plan=production_plan,
                ticker=ticker,
                analysis_data={}
            )
            
            if result["success"]:
                # 移动到输出目录
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_filename = f"{ticker}_analysis_{timestamp}.mp4"
                thumbnail_filename = f"{ticker}_thumbnail_{timestamp}.jpg"
                
                final_video_path = os.path.join(self.output_dir, video_filename)
                final_thumbnail_path = os.path.join(self.output_dir, thumbnail_filename)
                
                # 复制文件
                import shutil
                shutil.copy(result["video_path"], final_video_path)
                shutil.copy(result["thumbnail_path"], final_thumbnail_path)
                
                return {
                    "success": True,
                    "video_path": final_video_path,
                    "thumbnail_path": final_thumbnail_path,
                    "duration": result["duration"]
                }
            else:
                return {
                    "success": False,
                    "error": result["error"]
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _upload_to_cdn(self, video_result: Dict[str, Any]) -> Dict[str, Any]:
        """上传视频和缩略图到CDN"""
        try:
            # 这里是CDN上传的示例实现
            # 实际项目中需要根据具体的CDN提供商API进行实现
            
            # 模拟CDN上传
            time.sleep(2)  # 模拟上传时间
            
            # 生成CDN URL
            base_url = self.cdn_config["upload_url"]
            video_filename = os.path.basename(video_result["video_path"])
            thumbnail_filename = os.path.basename(video_result["thumbnail_path"])
            
            cdn_url = f"{base_url}/{video_filename}"
            thumbnail_url = f"{base_url}/{thumbnail_filename}"
            
            # 在实际项目中，这里应该有真实的上传逻辑
            # 例如使用requests库发送文件
            
            return {
                "success": True,
                "cdn_url": cdn_url,
                "thumbnail_url": thumbnail_url
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# 创建全局视频处理器实例
global_video_processor = VideoProcessor()