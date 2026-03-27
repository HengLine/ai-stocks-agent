from __future__ import annotations

from typing import Dict, Any, Optional, List
import os
import requests
import json
from datetime import datetime


class TTSService:
    """文本转语音服务"""
    
    def __init__(self):
        self.azure_key = os.getenv('AZURE_TTS_KEY')
        self.azure_region = os.getenv('AZURE_TTS_REGION', 'eastus')
        self.aliyun_key = os.getenv('ALIYUN_TTS_KEY')
        self.aliyun_secret = os.getenv('ALIYUN_TTS_SECRET')
        
        # 支持的语音类型
        self.voice_types = {
            "professional": {
                "azure": "zh-CN-XiaoxiaoNeural",
                "aliyun": "xiaoxiao"
            },
            "friendly": {
                "azure": "zh-CN-YunxiNeural", 
                "aliyun": "yunxi"
            },
            "humor": {
                "azure": "zh-CN-YunyangNeural",
                "aliyun": "yunyang"
            }
        }
    
    def generate_speech(self, text: str, voice_type: str = "professional", 
                       provider: str = "azure", output_path: str = None) -> Dict[str, Any]:
        """生成语音文件"""
        if provider == "azure":
            return self._azure_tts(text, voice_type, output_path)
        elif provider == "aliyun":
            return self._aliyun_tts(text, voice_type, output_path)
        else:
            return self._fallback_tts(text, voice_type, output_path)
    
    def _azure_tts(self, text: str, voice_type: str, output_path: str = None) -> Dict[str, Any]:
        """使用Azure TTS服务"""
        if not self.azure_key:
            return self._fallback_tts(text, voice_type, output_path)
        
        voice_name = self.voice_types.get(voice_type, {}).get("azure", "zh-CN-XiaoxiaoNeural")
        
        # Azure TTS API配置
        url = f"https://{self.azure_region}.tts.speech.microsoft.com/cognitiveservices/v1"
        headers = {
            "Ocp-Apim-Subscription-Key": self.azure_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3"
        }
        
        # SSML格式的文本
        ssml = f"""
        <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='zh-CN'>
            <voice name='{voice_name}'>
                <prosody rate='1.0' pitch='1.0'>
                    {text}
                </prosody>
            </voice>
        </speak>
        """
        
        try:
            response = requests.post(url, headers=headers, data=ssml.encode('utf-8'))
            if response.status_code == 200:
                # 保存音频文件
                if not output_path:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"temp_audio_{timestamp}.mp3"
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                return {
                    "success": True,
                    "file_path": output_path,
                    "duration": self._estimate_audio_duration(text),
                    "provider": "azure",
                    "voice_type": voice_type
                }
            else:
                return self._fallback_tts(text, voice_type, output_path)
        except Exception as e:
            return self._fallback_tts(text, voice_type, output_path)
    
    def _aliyun_tts(self, text: str, voice_type: str, output_path: str = None) -> Dict[str, Any]:
        """使用阿里云TTS服务"""
        if not self.aliyun_key or not self.aliyun_secret:
            return self._fallback_tts(text, voice_type, output_path)
        
        voice_name = self.voice_types.get(voice_type, {}).get("aliyun", "xiaoxiao")
        
        # 阿里云TTS API配置
        url = "https://nls-gateway-cn-shanghai.aliyuncs.com/stream/v1/tts"
        
        params = {
            "appkey": self.aliyun_key,
            "token": self._get_aliyun_token(),
            "text": text,
            "format": "mp3",
            "voice": voice_name,
            "volume": 50,
            "speech_rate": 0,
            "pitch_rate": 0
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                # 保存音频文件
                if not output_path:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"temp_audio_{timestamp}.mp3"
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                return {
                    "success": True,
                    "file_path": output_path,
                    "duration": self._estimate_audio_duration(text),
                    "provider": "aliyun",
                    "voice_type": voice_type
                }
            else:
                return self._fallback_tts(text, voice_type, output_path)
        except Exception as e:
            return self._fallback_tts(text, voice_type, output_path)
    
    def _get_aliyun_token(self) -> str:
        """获取阿里云访问令牌"""
        # 这里需要实现阿里云的token获取逻辑
        # 简化实现，实际需要调用阿里云的token API
        return "dummy_token"
    
    def _fallback_tts(self, text: str, voice_type: str, output_path: str = None) -> Dict[str, Any]:
        """备用TTS实现（使用系统TTS或返回文本）"""
        # 这里可以实现本地TTS或返回文本供其他工具处理
        return {
            "success": False,
            "text": text,
            "message": "TTS服务不可用，返回文本内容",
            "provider": "fallback",
            "voice_type": voice_type
        }
    
    def _estimate_audio_duration(self, text: str) -> float:
        """估算音频时长（秒）"""
        # 中文平均语速约200字/分钟
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        english_chars = len([c for c in text if c.isalpha()])
        
        # 估算时长
        duration = (chinese_chars * 0.3 + english_chars * 0.1)  # 秒
        return max(duration, 1.0)  # 最少1秒
    
    def generate_subtitles(self, script: Dict[str, Any], language: str = "zh-CN") -> List[Dict[str, Any]]:
        """生成字幕文件"""
        subtitles = []
        current_time = 0.0
        
        # 处理开场白
        if "opening" in script:
            duration = self._estimate_audio_duration(script["opening"])
            subtitles.append({
                "start": current_time,
                "end": current_time + duration,
                "text": script["opening"]
            })
            current_time += duration
        
        # 处理各个场景
        for scene in script.get("scenes", []):
            scene_duration = scene.get("duration", 10)
            content = scene.get("content", "")
            
            # 将场景内容分段
            segments = self._split_text_for_subtitles(content, scene_duration)
            
            for segment in segments:
                segment_duration = self._estimate_audio_duration(segment)
                subtitles.append({
                    "start": current_time,
                    "end": current_time + segment_duration,
                    "text": segment
                })
                current_time += segment_duration
        
        # 处理结束语
        if "closing" in script:
            duration = self._estimate_audio_duration(script["closing"])
            subtitles.append({
                "start": current_time,
                "end": current_time + duration,
                "text": script["closing"]
            })
        
        return subtitles
    
    def _split_text_for_subtitles(self, text: str, max_duration: float) -> List[str]:
        """将文本分段以适应字幕显示"""
        # 每段字幕最多显示3-4秒
        max_chars_per_segment = 20  # 每段最多20个字符
        
        if len(text) <= max_chars_per_segment:
            return [text]
        
        segments = []
        sentences = text.split('。')
        
        current_segment = ""
        for sentence in sentences:
            if len(current_segment + sentence) <= max_chars_per_segment:
                current_segment += sentence + "。"
            else:
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = sentence + "。"
        
        if current_segment:
            segments.append(current_segment.strip())
        
        return segments
    
    def save_subtitles_srt(self, subtitles: List[Dict[str, Any]], output_path: str) -> bool:
        """保存为SRT字幕格式"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, subtitle in enumerate(subtitles, 1):
                    start_time = self._format_srt_time(subtitle["start"])
                    end_time = self._format_srt_time(subtitle["end"])
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{subtitle['text']}\n\n")
            
            return True
        except Exception as e:
            return False
    
    def _format_srt_time(self, seconds: float) -> str:
        """格式化SRT时间格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
