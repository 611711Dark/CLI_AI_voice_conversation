"""
语音识别模块
使用 Whisper 模型进行语音转文字
"""

import io
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel
from config import (
    WHISPER_MODEL_PATH, 
    WHISPER_DEVICE, 
    WHISPER_COMPUTE_TYPE,
    SAMPLE_RATE,
    SILENCE_DURATION,
    VERBOSE
)

def log(msg):
    if VERBOSE:
        print(msg)

class SpeechRecognizer:
    """语音识别器"""
    
    def __init__(self):
        log("🔄 正在加载 Whisper 模型...")
        self.model = WhisperModel(
            WHISPER_MODEL_PATH, 
            device=WHISPER_DEVICE, 
            compute_type=WHISPER_COMPUTE_TYPE
        )
        log("✅ Whisper 模型加载完成")
    
    def transcribe(self, audio_data: np.ndarray) -> str:
        """将音频数据转录为文本"""
        if audio_data is None or len(audio_data) == 0:
            return ""
        
        try:
            # 将音频数据转换为 WAV 格式
            buf = io.BytesIO()
            sf.write(buf, audio_data, SAMPLE_RATE, format="WAV")
            buf.seek(0)
            
            # 使用 Whisper 进行转录
            segments, _ = self.model.transcribe(
                buf,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": int(SILENCE_DURATION * 1000)},
                beam_size=5
            )
            
            # 合并所有片段的文本
            text = " ".join(segment.text for segment in segments)
            return text.strip()
            
        except Exception as e:
            log(f"❌ 转录错误: {e}")
            return ""
