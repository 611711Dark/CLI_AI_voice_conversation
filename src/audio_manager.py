"""
音频管理模块
处理音频录制和播放功能
"""

import sounddevice as sd
import numpy as np
import tempfile
import os
import subprocess
import threading
import asyncio
from config import SAMPLE_RATE, BUFFER_SIZE, SILENCE_THRESHOLD, SILENCE_DURATION, VERBOSE

def log(msg):
    if VERBOSE:
        print(msg)

class AudioManager:
    """音频管理器"""
    
    def __init__(self):
        self.is_recording = False
        self.audio_buffer = []
        self.silence_timer = 0.0
        self.should_stop = False
    
    def _rms(self, audio: np.ndarray) -> float:
        """计算音频片段的 RMS 能量"""
        return np.sqrt(np.mean(np.square(audio), axis=0))
    
    def _callback(self, indata, frames, time, status):
        """音频输入回调函数"""
        chunk = indata[:, 0].astype(np.float32)
        level = self._rms(chunk)

        if not self.is_recording:
            if level > SILENCE_THRESHOLD:
                self.is_recording = True
                log("🎤 检测到语音，开始录音...")
                self.silence_timer = 0.0
            else:
                return
        else:
            # 录音中
            self.audio_buffer.append(chunk)
            if level <= SILENCE_THRESHOLD:
                self.silence_timer += frames / SAMPLE_RATE
                if self.silence_timer >= SILENCE_DURATION:
                    log("🔇 检测到静音，停止录音。")
                    self.should_stop = True
                    raise sd.CallbackStop()
            else:
                self.silence_timer = 0.0
    
    def record_audio(self) -> np.ndarray:
        """录制音频直到检测到静音"""
        self.is_recording = False
        self.audio_buffer = []
        self.silence_timer = 0.0
        self.should_stop = False

        log("👂 等待语音输入...")
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                blocksize=BUFFER_SIZE,
                callback=self._callback
            ):
                while not self.should_stop:
                    sd.sleep(100)
        except sd.CallbackStop:
            pass

        if self.audio_buffer:
            return np.concatenate(self.audio_buffer)
        else:
            log("⚠️ 未检测到有效音频。")
            return None
    
    async def play_audio_with_interrupt(self, audio_file_path: str, signal_handler) -> bool:
        """播放音频文件，支持打断功能"""
        signal_handler.set_playing_state(True)
        
        print("🔊 正在播放... (按 Ctrl+C 可打断)")
        
        try:
            # 尝试使用不同的播放器
            if os.name == 'nt':  # Windows
                process = subprocess.Popen(['start', '/wait', audio_file_path], 
                                         shell=True, 
                                         stdout=subprocess.DEVNULL, 
                                         stderr=subprocess.DEVNULL)
            else:  # Linux/Mac
                # 尝试使用 ffplay
                try:
                    process = subprocess.Popen(['ffplay', '-nodisp', '-autoexit', audio_file_path], 
                                             stdout=subprocess.DEVNULL, 
                                             stderr=subprocess.DEVNULL)
                except FileNotFoundError:
                    # 如果没有 ffplay，尝试使用 mpg123
                    try:
                        process = subprocess.Popen(['mpg123', '-q', audio_file_path], 
                                                 stdout=subprocess.DEVNULL, 
                                                 stderr=subprocess.DEVNULL)
                    except FileNotFoundError:
                        # 回退到 playsound
                        return await self._fallback_play_audio(audio_file_path, signal_handler)
            
            signal_handler.set_playing_state(True, process)
            
            # 监控播放过程
            while process.poll() is None:
                if signal_handler.should_interrupt:
                    log("🔇 播放被 Ctrl+C 打断")
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
                await asyncio.sleep(0.1)
            
            return signal_handler.should_interrupt
            
        except Exception as e:
            log(f"❌ 播放过程出错: {e}")
            return False
        finally:
            signal_handler.set_playing_state(False)
    
    async def _fallback_play_audio(self, audio_file_path: str, signal_handler) -> bool:
        """回退的音频播放方法"""
        try:
            from playsound import playsound
            
            def play_audio():
                try:
                    playsound(audio_file_path)
                except:
                    pass
            
            play_thread = threading.Thread(target=play_audio)
            play_thread.daemon = True
            play_thread.start()
            
            # 等待播放完成或被打断
            while play_thread.is_alive():
                if signal_handler.should_interrupt:
                    log("🔇 播放被 Ctrl+C 打断")
                    break
                await asyncio.sleep(0.1)
            
            return signal_handler.should_interrupt
            
        except ImportError:
            log("❌ 没有可用的音频播放方法。")
            return False
