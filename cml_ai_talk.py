import asyncio
import edge_tts
import signal
import sys
from playsound import playsound
from openai import OpenAI
import sounddevice as sd
import numpy as np
import wave
from faster_whisper import WhisperModel
import librosa
import soundfile as sf
from fuzzywuzzy import fuzz

# 初始化 OpenAI 客户端
client = OpenAI(api_key="<your-api>", base_url="<base-url>")

# 初始化 Whisper 模型
whisper_model = WhisperModel("faster-whisper-base", device="cpu", compute_type="int8")

# 对话历史，用于存储用户和AI的对话记录
conversation_history = [{"role": "system", "content": "You are a super intelligent artificial intelligence assistant, and you are currently in a command-line communication environment."}]
MAX_HISTORY_LENGTH = 20  # 对话历史的最大长度

# 录音参数
SAMPLE_RATE = 44100  # 设备支持的采样率
DURATION = 7  # 每次录音的时长（秒）

async def get_model_response(prompt):
    """
    获取AI模型的响应。

    Args:
        prompt (str): 用户输入的文本。

    Returns:
        str: AI模型的响应文本。
    """
    global conversation_history

    # 将用户输入添加到对话历史
    conversation_history.append({"role": "user", "content": prompt})

    # 如果对话历史超过最大长度，删除最早的对话（保留系统消息）
    if len(conversation_history) > MAX_HISTORY_LENGTH:
        conversation_history.pop(1)

    try:
        # 调用DeepSeek的ChatCompletion API获取AI响应
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=conversation_history,
            max_tokens=100,
            stream=False
        )
        model_response = response.choices[0].message.content
        # 将AI响应添加到对话历史
        conversation_history.append({"role": "assistant", "content": model_response})
        return model_response
    except Exception as e:
        print(f"模型请求出错: {e}")
        return None

async def text_to_speech(text):
    """
    将文本转换为语音并播放。

    Args:
        text (str): 需要转换为语音的文本。
    """
    communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
    audio_chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])

    if audio_chunks:
        # 将音频数据保存为临时文件
        with open("temp_audio.mp3", "wb") as f:
            for chunk in audio_chunks:
                f.write(chunk)
        # 播放音频文件
        playsound("temp_audio.mp3")

def record_audio(duration, sample_rate):
    """
    录制音频并保存为WAV文件。

    Args:
        duration (int): 录音时长（秒）。
        sample_rate (int): 采样率。
    """
    print("请开始说话...")
    # 录制音频
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()  # 等待录音完成
    print("录音结束。")
    
    # 保存原始音频
    with wave.open("temp_input_raw.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    
    # 重采样到16000 Hz（Whisper模型推荐采样率）
    audio_data, _ = librosa.load("temp_input_raw.wav", sr=16000)
    sf.write("temp_input.wav", audio_data, 16000)

def transcribe_audio(file_path):
    """
    将音频文件转录为文本。

    Args:
        file_path (str): 音频文件路径。

    Returns:
        str: 转录的文本。
    """
    segments, info = whisper_model.transcribe(file_path, beam_size=5)
    print(f"检测到语言: '{info.language}', 置信度: {info.language_probability}")
    transcription = " ".join(segment.text for segment in segments)
    print(f"转录结果: {transcription}")  # 打印转录结果
    return transcription

def signal_handler(sig, frame):
    """
    处理Ctrl+C信号，优雅退出程序。
    """
    print("\n退出程序。")
    sys.exit(0)

async def main():
    """
    主函数，处理用户交互和AI对话。
    """
    global conversation_history

    # 设置信号处理函数，捕获Ctrl+C信号
    signal.signal(signal.SIGINT, signal_handler)

    while True:
        input("按下回车键开始录音，或按下 Ctrl+C 退出: ")
        record_audio(DURATION, SAMPLE_RATE)  # 录制音频
        user_input = transcribe_audio("temp_input.wav")  # 转录音频为文本
        print(f"你说: {user_input}")

        # 模糊匹配退出命令
        if fuzz.partial_ratio(user_input.lower(), "quit") > 80 or \
           fuzz.partial_ratio(user_input.lower(), "退出") > 80 or \
           fuzz.partial_ratio(user_input.lower(), "结束") > 80:
            print("退出程序。")
            break

        model_response = await get_model_response(user_input)  # 获取 AI 响应
        if model_response:
            print(f"AI 回复: {model_response}")
            await text_to_speech(model_response)  # 将 AI 响应转换为语音

if __name__ == "__main__":
    asyncio.run(main())
