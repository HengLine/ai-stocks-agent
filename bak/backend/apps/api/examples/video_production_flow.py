#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
视频生成流程示例

演示如何使用ContentService和VideoProcessor生成完整的股票分析视频
"""

import os
import json
import time
from datetime import datetime

# 导入必要的服务和模型
from backend.apps.api.services.content_service import ContentService
from backend.apps.api.services.video_processor import VideoProcessor
from backend.apps.api.models import GeneratedContent


def main():
    """视频生成流程的主函数"""
    print("===== 股票分析视频生成流程示例 =====")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n")
    
    try:
        # 初始化内容服务
        content_service = ContentService()
        
        # 示例股票代码和参数
        ticker = "AAPL"  # 苹果公司股票代码
        video_params = {
            "style": "professional",  # 专业风格
            "duration": "medium",     # 中等时长
            "aspect_ratio": "16:9",    # 16:9宽高比
            "voice_type": "professional",  # 专业语音
            "tts_provider": "azure",   # 使用Azure TTS服务
            "chart_type": "candlestick" # 蜡烛图
        }
        
        # 步骤1: 提交视频生成任务
        print(f"步骤1: 为股票 {ticker} 提交视频生成任务...")
        result = content_service.enqueue_video_job(
            ticker=ticker,
            params=video_params
        )
        
        if not result["success"]:
            print(f"❌ 提交视频生成任务失败: {result['error']}")
            return
        
        job_id = result["job_id"]
        content_id = result["content_id"]
        print(f"✅ 视频生成任务已提交成功")
        print(f"   任务ID: {job_id}")
        print(f"   内容ID: {content_id}")
        print(f"   任务状态: {result['status']}")
        print("\n")
        
        # 步骤2: 等待视频生成完成
        print("步骤2: 等待视频生成完成...")
        # 注意：在实际应用中，应该使用异步任务或回调来处理，这里简化为轮询检查
        # 模拟等待时间
        print("   视频生成中，请稍候...")
        
        # 步骤3: 获取视频生成结果
        print("步骤3: 获取视频生成结果...")
        # 查询数据库获取内容信息
        try:
            content = GeneratedContent.objects.get(content_id=content_id)
            print(f"   内容状态: {content.status}")
            print(f"   标题: {content.title}")
            print(f"   创建时间: {content.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 如果内容元数据存在
            if content.metadata:
                metadata = json.loads(content.metadata)
                if "video_job" in metadata:
                    video_job = metadata["video_job"]
                    print(f"   视频任务状态: {video_job.get('status', 'unknown')}")
                    
                    if video_job.get('status') == 'completed':
                        print(f"   视频文件路径: {video_job.get('video_path', 'N/A')}")
                        print(f"   视频时长: {video_job.get('duration', 'N/A')}秒")
                        print(f"   缩略图路径: {video_job.get('thumbnail_path', 'N/A')}")
                        print(f"   CDN视频链接: {video_job.get('cdn_url', 'N/A')}")
                        print(f"   CDN缩略图链接: {video_job.get('thumbnail_cdn_url', 'N/A')}")
        except GeneratedContent.DoesNotExist:
            print("❌ 未找到内容记录")
        
        print("\n")
        print(f"视频生成流程示例完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("===== 视频生成流程示例结束 =====")
        
    except Exception as e:
        print(f"❌ 处理过程中发生错误: {str(e)}")


def direct_video_generation_example():
    """直接使用VideoProcessor生成视频的示例"""
    print("\n===== 直接使用VideoProcessor生成视频示例 =====")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 初始化VideoProcessor
        video_processor = VideoProcessor()
        
        # 示例股票代码和参数
        ticker = "TSLA"  # 特斯拉公司股票代码
        
        # 示例分析数据
        analysis_data = {
            "ticker": ticker,
            "time_frame": "3个月",
            "technical_analysis": {
                "ma_signal": "金叉",
                "rsi_signal": "超买",
                "macd_signal": "看涨"
            },
            "fundamental_analysis": {
                "revenue_growth": "40%",
                "gross_margin": "稳定"
            },
            "risk_assessment": {
                "risk_level": "中等"
            }
        }
        
        # 视频生成参数
        video_params = {
            "style": "professional",
            "duration": "medium",
            "aspect_ratio": "16:9",
            "voice_type": "professional",
            "tts_provider": "aliyun",
            "chart_type": "line"
        }
        
        print(f"直接为股票 {ticker} 生成视频...")
        
        # 步骤1: 生成视频脚本
        print("步骤1: 生成视频脚本...")
        script_result = video_processor.generate_video_script(
            ticker=ticker,
            analysis_data=analysis_data,
            params=video_params
        )
        
        if not script_result["success"]:
            print(f"❌ 生成视频脚本失败: {script_result['error']}")
            return
        
        script = script_result["script"]
        production_plan = script_result["production_plan"]
        print(f"✅ 视频脚本生成成功")
        print(f"   视频标题: {script.get('title', 'N/A')}")
        
        # 步骤2: 生成TTS语音和字幕
        print("步骤2: 生成TTS语音和字幕...")
        tts_result = video_processor.generate_tts_and_subtitles(
            script=script,
            voice_type=video_params["voice_type"],
            provider=video_params["tts_provider"]
        )
        
        if not tts_result["success"]:
            print(f"❌ 生成TTS语音失败: {tts_result['error']}")
            return
        
        audio_path = tts_result["audio_path"]
        subtitles_path = tts_result["subtitles_path"]
        print(f"✅ TTS语音和字幕生成成功")
        print(f"   音频文件: {audio_path}")
        print(f"   字幕文件: {subtitles_path}")
        
        # 步骤3: 生成图表动画
        print("步骤3: 生成图表动画...")
        chart_result = video_processor.generate_chart_animations(
            ticker=ticker,
            analysis_data=analysis_data,
            chart_type=video_params["chart_type"],
            aspect_ratio=video_params["aspect_ratio"]
        )
        
        if not chart_result["success"]:
            print(f"❌ 生成图表动画失败: {chart_result['error']}")
            return
        
        chart_videos = chart_result["chart_videos"]
        print(f"✅ 图表动画生成成功")
        print(f"   生成的图表数量: {len(chart_videos)}")
        
        # 步骤4: 合成视频
        print("步骤4: 合成视频...")
        video_result = video_processor.compose_video(
            script=script,
            audio_path=audio_path,
            subtitles_path=subtitles_path,
            chart_videos=chart_videos,
            production_plan=production_plan
        )
        
        if not video_result["success"]:
            print(f"❌ 合成视频失败: {video_result['error']}")
            return
        
        video_path = video_result["video_path"]
        thumbnail_path = video_result["thumbnail_path"]
        duration = video_result["duration"]
        print(f"✅ 视频合成成功")
        print(f"   视频文件: {video_path}")
        print(f"   缩略图文件: {thumbnail_path}")
        print(f"   视频时长: {duration}秒")
        
        # 步骤5: 上传到CDN
        print("步骤5: 上传到CDN...")
        cdn_result = video_processor.upload_to_cdn(
            video_path=video_path,
            thumbnail_path=thumbnail_path,
            ticker=ticker
        )
        
        if not cdn_result["success"]:
            print(f"❌ 上传CDN失败: {cdn_result['error']}")
            return
        
        cdn_url = cdn_result["cdn_url"]
        thumbnail_cdn_url = cdn_result["thumbnail_cdn_url"]
        print(f"✅ CDN上传成功")
        print(f"   CDN视频链接: {cdn_url}")
        print(f"   CDN缩略图链接: {thumbnail_cdn_url}")
        
        print("\n")
        print(f"直接视频生成示例完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("===== 直接视频生成示例结束 =====")
        
    except Exception as e:
        print(f"❌ 处理过程中发生错误: {str(e)}")


if __name__ == "__main__":
    # 运行完整的视频生成流程示例
    main()
    
    # 运行直接使用VideoProcessor的示例
    direct_video_generation_example()