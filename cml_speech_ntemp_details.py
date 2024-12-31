import asyncio
import edge_tts
import signal
import sys
from openai import OpenAI
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from fuzzywuzzy import fuzz
import io
import soundfile as sf
import librosa

# 常量定义
API_KEY = "<your-api>"
BASE_URL = "<base-url>"
WHISPER_MODEL_PATH = "faster-whisper-base"
SAMPLE_RATE = 44100  # 采样率
SILENCE_THRESHOLD = 0.02  # 静音阈值（根据音频能量调整）
SILENCE_DURATION = 1.5  # 静音时长（秒），超过该时长停止录音
BUFFER_SIZE = 1024  # 每次读取的音频帧数
MAX_HISTORY_LENGTH = 30  # 对话历史的最大长度

# 初始化
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
whisper_model = WhisperModel(WHISPER_MODEL_PATH, device="cpu", compute_type="int8")
conversation_history = [{"role": "system", "content": "You are a super intelligent artificial intelligence assistant, and you are currently in an oral communication environment."}]

# 全局变量，用于控制录音状态
is_recording = False
audio_buffer = []
silence_counter = 0
should_stop = False  # 用于控制录音是否停止

# 详细模式
verbose = False  # 设置为 True 输出详细日志，False 只输出用户和 AI 的对话

def signal_handler(sig, frame):
    """
    处理Ctrl+C信号，优雅退出程序。
    """
    print("\n退出程序。")
    sys.exit(0)

def log(message):
    """
    根据详细模式输出日志。

    Args:
        message (str): 需要输出的日志信息。
    """
    if verbose:
        print(message)

def is_silent(audio_data, threshold):
    """
    检测音频数据是否为静音。

    Args:
        audio_data (numpy.ndarray): 音频数据。
        threshold (float): 静音阈值。

    Returns:
        bool: 是否为静音。
    """
    if audio_data.size == 0:  # 如果音频数据为空，直接返回 True
        return True
    return np.max(np.abs(audio_data)) < threshold

def callback(indata, frames, time, status):
    """
    录音回调函数，处理每一帧音频数据。
    """
    global is_recording, audio_buffer, silence_counter, should_stop
    audio_data = indata[:, 0]  # 单声道
    if is_silent(audio_data, SILENCE_THRESHOLD):
        silence_counter += frames / SAMPLE_RATE  # 计算静音时长
        if is_recording and silence_counter >= SILENCE_DURATION:
            log("检测到静音，停止录音。")
            should_stop = True  # 设置停止标志
            raise sd.CallbackStop  # 停止录音
    else:
        silence_counter = 0  # 重置静音计数器
        if not is_recording:
            log("检测到语音，开始录音。")
            is_recording = True
    if is_recording:
        audio_buffer.append(audio_data.copy())  # 保存音频数据

def record_audio_on_speech():
    """
    实时录音，检测到语音开始录音，静音超过阈值后停止录音。
    """
    global is_recording, audio_buffer, silence_counter, should_stop
    log("等待语音输入...")

    # 重置状态
    is_recording = False
    audio_buffer = []
    silence_counter = 0
    should_stop = False

    try:
        # 开始录音
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=BUFFER_SIZE,
            callback=callback
        ):
            log("录音中...")
            while not should_stop:  # 检查停止标志
                sd.sleep(100)  # 短暂睡眠，避免占用 CPU
    except sd.CallbackStop:
        pass  # 正常停止录音
    except KeyboardInterrupt:
        log("\n检测到 Ctrl+C，停止录音。")
    finally:
        # 返回录制的音频数据
        if audio_buffer:
            return np.concatenate(audio_buffer)
        else:
            log("未检测到语音，未保存录音。")
            return None

def transcribe_audio(audio_data):
    """
    将音频数据转录为文本。

    Args:
        audio_data (numpy.ndarray): 音频数据。

    Returns:
        str: 转录的文本。
    """
    if audio_data is None or audio_data.size == 0:  # 如果音频数据为空，直接返回空字符串
        return ""
    try:
        # 将音频数据转换为字节流
        audio_stream = io.BytesIO()
        sf.write(audio_stream, audio_data, SAMPLE_RATE, format='WAV')
        audio_stream.seek(0)

        # 使用 Whisper 转录
        segments, _ = whisper_model.transcribe(
            audio_stream,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=1000),
            beam_size=8
        )
        return " ".join(segment.text for segment in segments)
    except Exception as e:
        log(f"转录出错: {e}")
        return ""

async def get_model_response(prompt):
    """
    获取AI模型的响应。

    Args:
        prompt (str): 用户输入的文本。

    Returns:
        str: AI模型的响应文本。
    """
    global conversation_history
    conversation_history.append({"role": "user", "content": prompt})
    if len(conversation_history) > MAX_HISTORY_LENGTH:
        conversation_history.pop(1)
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=conversation_history,
            max_tokens=300,
            stream=False
        )
        model_response = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": model_response})
        return model_response
    except Exception as e:
        log(f"模型请求出错: {e}")
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
        # 将音频数据拼接
        audio_data = b"".join(audio_chunks)
        audio_stream = io.BytesIO(audio_data)
        audio_stream.seek(0)

        # 读取音频数据并重采样
        audio, sr = sf.read(audio_stream)
        target_sr = 44100  # 目标采样率
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        # 使用 sounddevice 播放音频
        sd.play(audio, samplerate=target_sr)
        sd.wait()

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
    signal.signal(signal.SIGINT, signal_handler)
    print("程序已启动。按下 Ctrl+C 退出。")
    while True:
        audio_data = record_audio_on_speech()  # 使用 VAD 录音
        user_input = transcribe_audio(audio_data)
        if not user_input:
            log("未检测到语音，请重新说话。")
            continue
        print(f"你说: {user_input}")
        if fuzz.partial_ratio(user_input.lower(), "quit") > 80 or \
           fuzz.partial_ratio(user_input.lower(), "退出") > 80 or \
           fuzz.partial_ratio(user_input.lower(), "结束") > 80:
            print("退出程序。")
            break
        model_response = await get_model_response(user_input)
        if model_response:
            print(f"AI 回复: {model_response}")
            await text_to_speech(model_response)

if __name__ == "__main__":
    asyncio.run(main())
