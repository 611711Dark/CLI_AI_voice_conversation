"""
文本转语音模块
使用 Edge TTS 进行语音合成
"""

import asyncio
import tempfile
import os
import edge_tts
from config import TTS_VOICE, VERBOSE

def log(msg):
    if VERBOSE:
        print(msg)

class TextToSpeech:
    """文本转语音器"""
    
    def __init__(self):
        self.voice = TTS_VOICE
    
    async def synthesize(self, text: str) -> str:
        """合成语音并返回音频文件路径"""
        try:
            # 创建 TTS 通信对象
            communicate = edge_tts.Communicate(text, self.voice)
            
            # 收集音频数据
            audio_chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])
            
            if not audio_chunks:
                log("⚠️ TTS 未生成音频。")
                return None
            
            # 保存为临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                for chunk in audio_chunks:
                    f.write(chunk)
                temp_path = f.name
            
            return temp_path
            
        except Exception as e:
            log(f"❌ TTS 合成失败: {e}")
            return None
    
    def cleanup_temp_file(self, file_path: str):
        """清理临时文件"""
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
        except:
            pass
