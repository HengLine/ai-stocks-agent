from __future__ import annotations

from typing import Dict, Any, List, Optional
import os
import subprocess
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from ..services.tts_service import TTSService


class VideoService:
    """视频处理和合成服务"""
    
    def __init__(self):
        self.tts_service = TTSService()
        self.temp_dir = "temp_video_files"
        self._ensure_temp_dir()
    
    def _ensure_temp_dir(self):
        """确保临时目录存在"""
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
    
    def create_video(self, script: Dict[str, Any], production_plan: Dict[str, Any], 
                    ticker: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建完整视频"""
        try:
            # 1. 生成音频
            audio_result = self._generate_audio(script, production_plan)
            
            # 2. 生成图表动画
            chart_videos = self._generate_chart_videos(script, analysis_data, ticker)
            
            # 3. 生成字幕
            subtitles = self._generate_subtitles(script, production_plan)
            
            # 4. 合成最终视频
            final_video = self._compose_final_video(
                audio_result, chart_videos, subtitles, production_plan
            )
            
            return {
                "success": True,
                "video_path": final_video["path"],
                "thumbnail_path": final_video["thumbnail"],
                "duration": final_video["duration"],
                "metadata": {
                    "ticker": ticker,
                    "created_at": datetime.now().isoformat(),
                    "format": production_plan["output_specifications"]["format"],
                    "resolution": production_plan["visual_requirements"]["resolution"]
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "视频生成失败"
            }
    
    def _generate_audio(self, script: Dict[str, Any], production_plan: Dict[str, Any]) -> Dict[str, Any]:
        """生成音频文件"""
        # 合并所有文本
        full_text = script.get("opening", "")
        
        for scene in script.get("scenes", []):
            full_text += " " + scene.get("content", "")
        
        full_text += " " + script.get("closing", "")
        
        # 生成TTS音频
        tts_requirements = production_plan["tts_requirements"]
        voice_type = tts_requirements["voice_type"]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = os.path.join(self.temp_dir, f"audio_{timestamp}.mp3")
        
        audio_result = self.tts_service.generate_speech(
            full_text, voice_type, "azure", audio_path
        )
        
        return audio_result
    
    def _generate_chart_videos(self, script: Dict[str, Any], analysis_data: Dict[str, Any], 
                              ticker: str) -> List[Dict[str, Any]]:
        """生成图表动画视频"""
        chart_videos = []
        
        # 获取价格数据
        prices = analysis_data.get("technical_indicators", {})
        current_price = prices.get("close", 0)
        ma5 = prices.get("ma5", 0)
        ma20 = prices.get("ma20", 0)
        rsi = prices.get("rsi", 50)
        
        # 生成价格走势图
        price_chart = self._create_price_chart_animation(ticker, current_price, ma5, ma20)
        if price_chart:
            chart_videos.append(price_chart)
        
        # 生成RSI指标图
        rsi_chart = self._create_rsi_chart_animation(rsi)
        if rsi_chart:
            chart_videos.append(rsi_chart)
        
        # 生成MACD指标图
        macd_chart = self._create_macd_chart_animation(analysis_data.get("technical_indicators", {}).get("macd", {}))
        if macd_chart:
            chart_videos.append(macd_chart)
        
        return chart_videos
    
    def _create_price_chart_animation(self, ticker: str, current_price: float, 
                                    ma5: float, ma20: float) -> Optional[Dict[str, Any]]:
        """创建价格走势图动画"""
        try:
            # 生成模拟价格数据
            days = 30
            base_price = current_price * 0.9
            prices = [base_price + np.random.normal(0, current_price * 0.02) for _ in range(days)]
            prices[-1] = current_price  # 确保最后一天是当前价格
            
            # 计算移动平均线
            ma5_values = [np.mean(prices[max(0, i-4):i+1]) for i in range(days)]
            ma20_values = [np.mean(prices[max(0, i-19):i+1]) for i in range(days)]
            
            # 创建动画
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_title(f"{ticker} 价格走势图", fontsize=16, fontproperties='SimHei')
            ax.set_xlabel("交易日", fontproperties='SimHei')
            ax.set_ylabel("价格", fontproperties='SimHei')
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            
            line_price, = ax.plot([], [], 'b-', linewidth=2, label='收盘价')
            line_ma5, = ax.plot([], [], 'r-', linewidth=1, label='MA5')
            line_ma20, = ax.plot([], [], 'g-', linewidth=1, label='MA20')
            
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            def animate(frame):
                if frame < days:
                    x_data = list(range(frame + 1))
                    line_price.set_data(x_data, prices[:frame + 1])
                    line_ma5.set_data(x_data, ma5_values[:frame + 1])
                    line_ma20.set_data(x_data, ma20_values[:frame + 1])
                    
                    ax.set_xlim(0, days)
                    ax.set_ylim(min(prices) * 0.95, max(prices) * 1.05)
                
                return line_price, line_ma5, line_ma20
            
            # 保存动画
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(self.temp_dir, f"price_chart_{timestamp}.mp4")
            
            anim = animation.FuncAnimation(fig, animate, frames=days, interval=100, blit=True)
            anim.save(video_path, writer='ffmpeg', fps=10)
            plt.close(fig)
            
            return {
                "type": "price_chart",
                "path": video_path,
                "duration": days * 0.1,  # 每帧0.1秒
                "scene_timing": (0, 10)  # 在视频中的时间段
            }
        except Exception as e:
            print(f"价格图表动画生成失败: {e}")
            return None
    
    def _create_rsi_chart_animation(self, rsi_value: float) -> Optional[Dict[str, Any]]:
        """创建RSI指标图动画"""
        try:
            # 生成模拟RSI数据
            days = 30
            rsi_values = [50 + np.random.normal(0, 10) for _ in range(days)]
            rsi_values = [max(0, min(100, rsi)) for rsi in rsi_values]
            rsi_values[-1] = rsi_value  # 确保最后一天是当前RSI
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title("RSI 相对强弱指标", fontsize=14, fontproperties='SimHei')
            ax.set_xlabel("交易日", fontproperties='SimHei')
            ax.set_ylabel("RSI值", fontproperties='SimHei')
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            
            line_rsi, = ax.plot([], [], 'purple', linewidth=2, label='RSI')
            ax.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='超买线(70)')
            ax.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='超卖线(30)')
            ax.axhline(y=50, color='gray', linestyle='-', alpha=0.5, label='中线(50)')
            
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            
            def animate(frame):
                if frame < days:
                    x_data = list(range(frame + 1))
                    line_rsi.set_data(x_data, rsi_values[:frame + 1])
                    ax.set_xlim(0, days)
                
                return line_rsi,
            
            # 保存动画
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(self.temp_dir, f"rsi_chart_{timestamp}.mp4")
            
            anim = animation.FuncAnimation(fig, animate, frames=days, interval=100, blit=True)
            anim.save(video_path, writer='ffmpeg', fps=10)
            plt.close(fig)
            
            return {
                "type": "rsi_chart",
                "path": video_path,
                "duration": days * 0.1,
                "scene_timing": (10, 20)
            }
        except Exception as e:
            print(f"RSI图表动画生成失败: {e}")
            return None
    
    def _create_macd_chart_animation(self, macd_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """创建MACD指标图动画"""
        try:
            # 生成模拟MACD数据
            days = 30
            macd_line = [np.random.normal(0, 0.5) for _ in range(days)]
            signal_line = [np.random.normal(0, 0.3) for _ in range(days)]
            histogram = [macd_line[i] - signal_line[i] for i in range(days)]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # MACD线图
            ax1.set_title("MACD 指标", fontsize=14, fontproperties='SimHei')
            ax1.set_ylabel("MACD值", fontproperties='SimHei')
            
            line_macd, = ax1.plot([], [], 'blue', linewidth=2, label='MACD')
            line_signal, = ax1.plot([], [], 'red', linewidth=2, label='Signal')
            ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 柱状图
            ax2.set_xlabel("交易日", fontproperties='SimHei')
            ax2.set_ylabel("MACD柱", fontproperties='SimHei')
            bars = ax2.bar([], [], color='green', alpha=0.7)
            ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax2.grid(True, alpha=0.3)
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            
            def animate(frame):
                if frame < days:
                    x_data = list(range(frame + 1))
                    line_macd.set_data(x_data, macd_line[:frame + 1])
                    line_signal.set_data(x_data, signal_line[:frame + 1])
                    
                    ax1.set_xlim(0, days)
                    ax1.set_ylim(min(min(macd_line), min(signal_line)) - 0.5, 
                                max(max(macd_line), max(signal_line)) + 0.5)
                    
                    # 更新柱状图
                    ax2.clear()
                    ax2.set_xlabel("交易日", fontproperties='SimHei')
                    ax2.set_ylabel("MACD柱", fontproperties='SimHei')
                    colors = ['green' if h >= 0 else 'red' for h in histogram[:frame + 1]]
                    ax2.bar(x_data, histogram[:frame + 1], color=colors, alpha=0.7)
                    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
                    ax2.grid(True, alpha=0.3)
                    ax2.set_xlim(0, days)
                
                return line_macd, line_signal
            
            # 保存动画
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(self.temp_dir, f"macd_chart_{timestamp}.mp4")
            
            anim = animation.FuncAnimation(fig, animate, frames=days, interval=100, blit=True)
            anim.save(video_path, writer='ffmpeg', fps=10)
            plt.close(fig)
            
            return {
                "type": "macd_chart",
                "path": video_path,
                "duration": days * 0.1,
                "scene_timing": (20, 30)
            }
        except Exception as e:
            print(f"MACD图表动画生成失败: {e}")
            return None
    
    def _generate_subtitles(self, script: Dict[str, Any], production_plan: Dict[str, Any]) -> Dict[str, Any]:
        """生成字幕文件"""
        subtitles = self.tts_service.generate_subtitles(script)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subtitle_path = os.path.join(self.temp_dir, f"subtitles_{timestamp}.srt")
        
        success = self.tts_service.save_subtitles_srt(subtitles, subtitle_path)
        
        return {
            "path": subtitle_path,
            "success": success,
            "subtitles": subtitles
        }
    
    def _compose_final_video(self, audio_result: Dict[str, Any], chart_videos: List[Dict[str, Any]], 
                           subtitles: Dict[str, Any], production_plan: Dict[str, Any]) -> Dict[str, Any]:
        """合成最终视频"""
        try:
            # 使用FFmpeg合成视频
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"output_video_{timestamp}.mp4"
            thumbnail_path = f"thumbnail_{timestamp}.jpg"
            
            # 构建FFmpeg命令
            cmd = self._build_ffmpeg_command(
                audio_result, chart_videos, subtitles, production_plan, output_path
            )
            
            # 执行FFmpeg命令
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                # 生成缩略图
                self._generate_thumbnail(output_path, thumbnail_path)
                
                # 获取视频时长
                duration = self._get_video_duration(output_path)
                
                return {
                    "path": output_path,
                    "thumbnail": thumbnail_path,
                    "duration": duration
                }
            else:
                raise Exception(f"FFmpeg执行失败: {result.stderr}")
                
        except Exception as e:
            raise Exception(f"视频合成失败: {e}")
    
    def _build_ffmpeg_command(self, audio_result: Dict[str, Any], chart_videos: List[Dict[str, Any]], 
                            subtitles: Dict[str, Any], production_plan: Dict[str, Any], 
                            output_path: str) -> str:
        """构建FFmpeg命令"""
        # 获取输出规格
        output_specs = production_plan["output_specifications"]
        visual_reqs = production_plan["visual_requirements"]
        
        # 基础命令
        cmd = "ffmpeg -y"
        
        # 添加音频输入
        if audio_result.get("success") and audio_result.get("file_path"):
            cmd += f" -i {audio_result['file_path']}"
        
        # 添加视频输入（如果有图表视频）
        for chart_video in chart_videos:
            cmd += f" -i {chart_video['path']}"
        
        # 视频滤镜
        filters = []
        
        # 如果有多个视频输入，需要合并
        if len(chart_videos) > 1:
            # 创建视频合并滤镜
            concat_filter = "concat=n=" + str(len(chart_videos)) + ":v=1:a=0"
            filters.append(concat_filter)
        
        # 添加字幕
        if subtitles.get("success") and subtitles.get("path"):
            filters.append(f"subtitles={subtitles['path']}")
        
        # 设置输出格式和编码
        cmd += f" -c:v {output_specs['codec']}"
        cmd += f" -b:v {output_specs['bitrate']}"
        cmd += f" -r {output_specs['fps']}"
        cmd += f" -s {visual_reqs['resolution']}"
        
        # 添加音频编码
        if audio_result.get("success"):
            cmd += " -c:a aac -b:a 128k"
        
        # 应用滤镜
        if filters:
            cmd += f" -filter_complex \"{';'.join(filters)}\""
        
        cmd += f" {output_path}"
        
        return cmd
    
    def _generate_thumbnail(self, video_path: str, thumbnail_path: str):
        """生成视频缩略图"""
        cmd = f"ffmpeg -i {video_path} -ss 00:00:01 -vframes 1 {thumbnail_path}"
        subprocess.run(cmd, shell=True, capture_output=True)
    
    def _get_video_duration(self, video_path: str) -> float:
        """获取视频时长"""
        cmd = f"ffprobe -v quiet -show_entries format=duration -of csv=p=0 {video_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        try:
            return float(result.stdout.strip())
        except:
            return 0.0
    
    def cleanup_temp_files(self):
        """清理临时文件"""
        try:
            for filename in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except Exception as e:
            print(f"清理临时文件失败: {e}")
