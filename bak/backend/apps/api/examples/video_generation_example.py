#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
视频生成模块使用示例

展示如何使用VideoAgent和VideoService生成完整的股票分析视频，包括：
1. 生成视频脚本（含分镜、字幕、BGM建议）
2. 文本转语音（TTS）
3. 图表动画生成
4. 视频合成（FFmpeg）
5. 输出MP4文件和缩略图
"""

from typing import Dict, Any
import sys
import os
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from backend.apps.api.agents.video_agent import VideoAgent
from backend.apps.api.services.video_service import VideoService
from backend.apps.api.services.tts_service import TTSService
from backend.apps.api.services.content_service import ContentService


def generate_video_script_example():
    """视频脚本生成示例"""
    print("\n=== 步骤1: 生成视频脚本 ===")
    
    # 初始化VideoAgent
    video_agent = VideoAgent()
    
    # 准备分析数据（通常来自AnalysisAgent）
    analysis_data = {
        "ticker": "AAPL",
        "time_frame": "1个月",
        "technical_analysis": {
            "ma_signal": "金叉",
            "rsi_signal": "超买",
            "macd_signal": "看涨"
        },
        "trend_analysis": {
            "short_term": "上涨",
            "long_term": "震荡上行",
            "strength": "strong",
            "price_change_percent": 5.2
        },
        "risk_level": "medium",
        "investment_suggestion": {
            "action": "buy",
            "confidence": "high",
            "reasoning": "综合技术面和基本面分析，AAPL具有良好的上涨潜力"
        },
        "key_points": [
            "MACD形成金叉，显示买入信号",
            "成交量放大，市场关注度提升",
            "RSI处于超买区域，短期可能回调"
        ],
        "technical_indicators": {
            "close": 187.23,
            "ma5": 184.56,
            "ma20": 180.34,
            "rsi": 72,
            "macd": {
                "line": 2.34,
                "signal": 1.98,
                "histogram": 0.36
            }
        }
    }
    
    # 准备参数
    params = {
        "style": "professional",   # 视频风格
        "duration": "medium",      # 视频时长
        "aspect_ratio": "16:9",    # 宽高比
        "language": "zh-CN"        # 语言
    }
    
    context = {
        "ticker": "AAPL",
        "analysis": analysis_data
    }
    
    # 生成视频脚本
    result = video_agent.run(params, context)
    
    if result.success:
        script = result.data["script"]
        production_plan = result.data["production_plan"]
        
        print(f"视频脚本生成成功！")
        print(f"标题: {script['title']}")
        print(f"时长: {script['total_duration']}秒")
        print(f"BGM风格: {script['bgm_style']}")
        print(f"视觉风格: {script['visual_style']}")
        
        # 打印脚本结构
        print(f"\n脚本结构:")
        print(f"开场: {script['opening']}")
        print(f"场景数量: {len(script['scenes'])}")
        
        # 打印前两个场景详情
        print(f"\n场景详情:")
        for i, scene in enumerate(script['scenes'][:2]):
            print(f"场景 {i+1} ({scene['section']}):")
            print(f"  内容: {scene['content']}")
            print(f"  时长: {scene['duration']}秒")
            print(f"  视觉元素: {scene['visual_elements']}")
        
        print(f"\n结尾: {script['closing']}")
        
        # 打印制作计划
        print(f"\n制作计划详情:")
        print(f"TTS需求: {production_plan['tts_requirements']}")
        print(f"视觉需求: {production_plan['visual_requirements']['aspect_ratio']}, {production_plan['visual_requirements']['resolution']}")
        print(f"音频需求: {production_plan['audio_requirements']}")
        print(f"字幕需求: {production_plan['subtitle_requirements']}")
        print(f"输出规格: {production_plan['output_specifications']}")
        
        return analysis_data, script, production_plan
    else:
        print(f"视频脚本生成失败: {result.data}")
        return None, None, None


def tts_generation_example(script: Dict[str, Any]):
    """文本转语音（TTS）示例"""
    print("\n=== 步骤2: 文本转语音（TTS） ===")
    
    # 初始化TTSService
    tts_service = TTSService()
    
    # 合并脚本文本
    full_text = script.get("opening", "")
    for scene in script.get("scenes", []):
        full_text += " " + scene.get("content", "")
    full_text += " " + script.get("closing", "")
    
    # 生成语音
    # 注意：实际使用时需要设置环境变量中的API密钥
    print(f"生成语音文件，文本长度: {len(full_text)}字符")
    print(f"尝试使用Azure TTS服务...")
    
    # 由于是示例，我们会生成一个模拟的音频结果
    # 在实际环境中，这里会调用真实的TTS服务
    audio_result = {
        "success": True,
        "file_path": "sample_audio.mp3",
        "duration": tts_service._estimate_audio_duration(full_text),
        "provider": "azure",
        "voice_type": "professional"
    }
    
    if audio_result["success"]:
        print(f"TTS生成成功！")
        print(f"音频文件: {audio_result['file_path']}")
        print(f"音频时长: {audio_result['duration']:.2f}秒")
        print(f"TTS提供商: {audio_result['provider']}")
        print(f"语音类型: {audio_result['voice_type']}")
    else:
        print(f"TTS生成失败: {audio_result.get('message', '未知错误')}")
    
    # 生成字幕
    print("\n生成字幕文件...")
    subtitles = tts_service.generate_subtitles(script)
    
    if subtitles:
        print(f"字幕生成成功！")
        print(f"字幕数量: {len(subtitles)}")
        print(f"前两条字幕预览:")
        for i, subtitle in enumerate(subtitles[:2]):
            print(f"  {i+1}. [{subtitle['start']:.2f}-{subtitle['end']:.2f}s] {subtitle['text']}")
    
    return audio_result, subtitles


def chart_animation_example(analysis_data: Dict[str, Any], ticker: str):
    """图表动画生成示例"""
    print("\n=== 步骤3: 图表动画生成 ===")
    
    # 初始化VideoService
    video_service = VideoService()
    
    # 生成价格走势图动画
    print(f"生成{tticker}价格走势图动画...")
    price_chart = video_service._create_price_chart_animation(
        ticker, 
        analysis_data['technical_indicators']['close'],
        analysis_data['technical_indicators']['ma5'],
        analysis_data['technical_indicators']['ma20']
    )
    
    # 生成RSI指标图动画
    print(f"生成RSI指标图动画...")
    rsi_chart = video_service._create_rsi_chart_animation(
        analysis_data['technical_indicators']['rsi']
    )
    
    # 生成MACD指标图动画
    print(f"生成MACD指标图动画...")
    macd_chart = video_service._create_macd_chart_animation(
        analysis_data['technical_indicators']['macd']
    )
    
    chart_videos = []
    for chart in [price_chart, rsi_chart, macd_chart]:
        if chart:
            chart_videos.append(chart)
    
    if chart_videos:
        print(f"图表动画生成成功！")
        print(f"生成的图表数量: {len(chart_videos)}")
        for chart in chart_videos:
            print(f"  - {chart['type']}: {chart['path']} ({chart['duration']:.2f}秒)")
    
    return chart_videos


def video_compose_example(script: Dict[str, Any], production_plan: Dict[str, Any], 
                          analysis_data: Dict[str, Any]):
    """视频合成示例"""
    print("\n=== 步骤4: 视频合成 ===")
    
    # 初始化VideoService
    video_service = VideoService()
    
    print(f"开始合成视频，包含音频、图表动画和字幕...")
    
    # 注意：实际使用时，这里会调用真实的视频合成功能
    # 由于是示例，我们会生成一个模拟的视频结果
    video_result = {
        "success": True,
        "video_path": "output_video.mp4",
        "thumbnail_path": "thumbnail.jpg",
        "duration": script['total_duration'],
        "metadata": {
            "ticker": "AAPL",
            "created_at": "2023-11-15T12:30:45",
            "format": "mp4",
            "resolution": "1920x1080"
        }
    }
    
    if video_result["success"]:
        print(f"视频合成成功！")
        print(f"视频文件: {video_result['video_path']}")
        print(f"缩略图文件: {video_result['thumbnail_path']}")
        print(f"视频时长: {video_result['duration']}秒")
        print(f"视频分辨率: {video_result['metadata']['resolution']}")
        
        # 模拟CDN上传
        print("\n=== 步骤5: 上传至CDN ===")
        cdn_url = "https://cdn.example.com/videos/output_video.mp4"
        thumbnail_cdn_url = "https://cdn.example.com/videos/thumbnail.jpg"
        
        print(f"视频已上传至CDN: {cdn_url}")
        print(f"缩略图已上传至CDN: {thumbnail_cdn_url}")
    
    return video_result


def full_video_generation_flow():
    """完整的视频生成流程示例"""
    print("===== AIGC股票分析视频生成模块示例 =====")
    
    try:
        # 步骤1: 生成视频脚本
        analysis_data, script, production_plan = generate_video_script_example()
        if not script:
            print("脚本生成失败，流程终止。")
            return
        
        # 步骤2: 文本转语音和字幕生成
        audio_result, subtitles = tts_generation_example(script)
        
        # 步骤3: 图表动画生成
        chart_videos = chart_animation_example(analysis_data, "AAPL")
        
        # 步骤4: 视频合成
        video_result = video_compose_example(script, production_plan, analysis_data)
        
        # 清理临时文件
        print("\n清理临时文件...")
        video_service = VideoService()
        video_service.cleanup_temp_files()
        
        print("\n===== 视频生成流程完成 =====")
        print("\n视频生成摘要:")
        print(f"- 脚本生成: 成功")
        print(f"- TTS生成: {'成功' if audio_result['success'] else '失败'}")
        print(f"- 图表动画: 生成了{len(chart_videos)}个图表")
        print(f"- 视频合成: {'成功' if video_result['success'] else '失败'}")
        if video_result['success']:
            print(f"- CDN上传: 完成")
            print(f"- 最终输出: MP4文件 + 缩略图")
            
    except Exception as e:
        print(f"视频生成过程中发生错误: {e}")


def content_service_integration_example():
    """使用ContentService集成视频生成示例"""
    print("\n\n===== 使用ContentService集成视频生成示例 =====")
    
    # 初始化ContentService
    content_service = ContentService()
    
    # 准备分析数据
    analysis_data = {
        "ticker": "TSLA",
        "time_frame": "1周",
        "technical_analysis": {
            "ma_signal": "死叉",
            "rsi_signal": "超卖",
            "macd_signal": "看跌"
        },
        "trend_analysis": {
            "short_term": "下跌",
            "long_term": "震荡",
            "strength": "medium",
            "price_change_percent": -3.8
        },
        "risk_level": "high",
        "investment_suggestion": {
            "action": "sell",
            "confidence": "medium",
            "reasoning": "短期内存在下行风险"
        },
        "key_points": [
            "MACD形成死叉",
            "RSI进入超卖区间",
            "成交量萎缩"
        ]
    }
    
    # 生成视频内容
    print("使用ContentService生成视频内容...")
    result = content_service.generate_content(
        content_type="video",
        ticker="TSLA",
        analysis_data=analysis_data,
        params={
            "style": "professional",
            "duration": "short",
            "aspect_ratio": "16:9",
            "language": "zh-CN"
        }
    )
    
    if result["success"]:
        print(f"视频内容生成成功！")
        print(f"内容ID: {result['content_id']}")
        print(f"视频标题: {result['content']['script']['title']}")
        print(f"场景数量: {len(result['content']['script']['scenes'])}")
    else:
        print(f"视频内容生成失败: {result['error']}")


def main():
    """主函数，运行所有示例"""
    # 运行完整的视频生成流程示例
    full_video_generation_flow()
    
    # 运行ContentService集成示例
    content_service_integration_example()


if __name__ == "__main__":
    main()