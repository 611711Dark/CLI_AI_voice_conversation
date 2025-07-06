import asyncio
import edge_tts
import signal
import sys
from openai import OpenAI
import sounddevice as sd
import numpy as np
import io
import soundfile as sf
import librosa
from faster_whisper import WhisperModel
from fuzzywuzzy import fuzz
import tempfile
import threading
import time
import os
import subprocess

# Constants
API_KEY = "api_key"
BASE_URL = "url"
WHISPER_MODEL_PATH = "faster-whisper-base"
SAMPLE_RATE = 44100
SILENCE_THRESHOLD = 0.1      # RMS 阈值
SILENCE_DURATION = 1.5       # 连续静音超过 1.5 秒停止
BUFFER_SIZE = 1024
MAX_HISTORY_LENGTH = 30

# 初始化客户端与模型
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
whisper_model = WhisperModel(WHISPER_MODEL_PATH, device="cpu", compute_type="int8")
conversation_history = [
    {"role": "system", "content": "You are a super intelligent artificial intelligence assistant, and you are currently in an oral communication environment."}
]

# 全局状态
is_recording = False
audio_buffer = []
silence_timer = 0.0
should_stop = False
is_playing = False
should_interrupt = False
playback_process = None

verbose = True
def log(msg):
    if verbose:
        print(msg)

def signal_handler(sig, frame):
    global playback_process, should_interrupt, is_playing
    
    if is_playing:
        # 如果正在播放，打断播放
        should_interrupt = True
        if playback_process:
            try:
                playback_process.terminate()
                playback_process.wait(timeout=1)
            except:
                pass
        print("\n播放被打断，等待语音输入...")
        return
    else:
        # 如果不在播放，退出程序
        print("\n退出程序。")
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def rms(audio: np.ndarray) -> float:
    """计算音频片段的 RMS 能量。"""
    return np.sqrt(np.mean(np.square(audio), axis=0))

def callback(indata, frames, time, status):
    global is_recording, audio_buffer, silence_timer, should_stop

    chunk = indata[:, 0].astype(np.float32)
    level = rms(chunk)

    if not is_recording:
        if level > SILENCE_THRESHOLD:
            is_recording = True
            log("检测到语音，开始录音。")
            silence_timer = 0.0
        else:
            return
    else:
        # 录音中
        audio_buffer.append(chunk)
        if level <= SILENCE_THRESHOLD:
            silence_timer += frames / SAMPLE_RATE
            if silence_timer >= SILENCE_DURATION:
                log("检测到静音，停止录音。")
                should_stop = True
                raise sd.CallbackStop()
        else:
            silence_timer = 0.0

def record_audio_on_speech() -> np.ndarray:
    """实时录音：检测到语音开始录音，静音超过阈值后停止。"""
    global is_recording, audio_buffer, silence_timer, should_stop

    is_recording = False
    audio_buffer = []
    silence_timer = 0.0
    should_stop = False

    log("等待语音输入...")
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=BUFFER_SIZE,
            callback=callback
        ):
            while not should_stop:
                sd.sleep(100)
    except sd.CallbackStop:
        pass

    if audio_buffer:
        return np.concatenate(audio_buffer)
    else:
        log("未检测到有效音频。")
        return None

def transcribe_audio(audio_data: np.ndarray) -> str:
    """将录制的音频数据转写为文本。"""
    if audio_data is None or len(audio_data) == 0:
        return ""
    try:
        buf = io.BytesIO()
        sf.write(buf, audio_data, SAMPLE_RATE, format="WAV")
        buf.seek(0)
        segments, _ = whisper_model.transcribe(
            buf,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": int(SILENCE_DURATION * 1000)},
            beam_size=5
        )
        return " ".join(segment.text for segment in segments)
    except Exception as e:
        log(f"转录错误: {e}")
        return ""

async def text_to_speech_with_interrupt(text: str):
    """将文本转换为语音并播放，支持 Ctrl+C 打断功能。"""
    global is_playing, should_interrupt, playback_process
    
    try:
        # 生成语音
        communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        if not audio_chunks:
            log("TTS 未生成音频。")
            return False

        # 保存为临时 MP3 文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            for c in audio_chunks:
                f.write(c)
            temp_path = f.name

        # 设置播放状态
        is_playing = True
        should_interrupt = False
        
        print("正在播放... (按 Ctrl+C 可打断)")

        # 使用 subprocess 播放音频
        try:
            # 尝试使用不同的播放器
            if os.name == 'nt':  # Windows
                playback_process = subprocess.Popen(['start', '/wait', temp_path], 
                                                   shell=True, 
                                                   stdout=subprocess.DEVNULL, 
                                                   stderr=subprocess.DEVNULL)
            else:  # Linux/Mac
                # 尝试使用 ffplay (ffmpeg 套件的一部分)
                try:
                    playback_process = subprocess.Popen(['ffplay', '-nodisp', '-autoexit', temp_path], 
                                                       stdout=subprocess.DEVNULL, 
                                                       stderr=subprocess.DEVNULL)
                except FileNotFoundError:
                    # 如果没有 ffplay，尝试使用 mpg123
                    try:
                        playback_process = subprocess.Popen(['mpg123', '-q', temp_path], 
                                                           stdout=subprocess.DEVNULL, 
                                                           stderr=subprocess.DEVNULL)
                    except FileNotFoundError:
                        # 最后尝试使用系统默认播放器
                        log("警告: 未找到 ffplay 或 mpg123，尝试使用系统默认播放器")
                        playback_process = subprocess.Popen(['python', '-c', 
                                                           f'from playsound import playsound; playsound("{temp_path}")'], 
                                                           stdout=subprocess.DEVNULL, 
                                                           stderr=subprocess.DEVNULL)

            # 监控播放过程
            while playback_process.poll() is None:
                if should_interrupt:
                    log("播放被 Ctrl+C 打断")
                    playback_process.terminate()
                    try:
                        playback_process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        playback_process.kill()
                    break
                await asyncio.sleep(0.1)
            
            # 清理临时文件
            try:
                os.unlink(temp_path)
            except:
                pass
            
            return should_interrupt  # 返回是否被打断
            
        except FileNotFoundError:
            log("错误: 未找到合适的音频播放器。请安装 ffmpeg、mpg123 或确保有 playsound 库。")
            # 回退到原来的方法
            try:
                from playsound import playsound
                # 在单独的线程中播放，这样可以被打断
                def play_audio():
                    try:
                        playsound(temp_path)
                    except:
                        pass
                
                play_thread = threading.Thread(target=play_audio)
                play_thread.daemon = True
                play_thread.start()
                
                # 等待播放完成或被打断
                while play_thread.is_alive():
                    if should_interrupt:
                        log("播放被 Ctrl+C 打断")
                        break
                    await asyncio.sleep(0.1)
                
                # 清理临时文件
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
                return should_interrupt
            except ImportError:
                log("错误: 没有可用的音频播放方法。")
                return False
        except Exception as e:
            log(f"播放过程出错: {e}")
            return False
            
    except Exception as e:
        log(f"TTS 播放失败: {e}")
        return False
    finally:
        is_playing = False
        playback_process = None

async def get_model_response(prompt: str) -> str:
    """调用 OpenAI 接口获取 AI 响应。"""
    conversation_history.append({"role": "user", "content": prompt})
    if len(conversation_history) > MAX_HISTORY_LENGTH:
        conversation_history.pop(1)
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b",
            messages=conversation_history,
            max_tokens=300,
            stream=False
        )
        text = resp.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": text})
        return text
    except Exception as e:
        log(f"模型请求出错: {e}")
        return ""

async def main():
    print("程序已启动，开始语音对话。")
    print("提示：在 AI 播放语音时，按 Ctrl+C 可以打断播放并继续对话。")
    print("      在等待输入时，按 Ctrl+C 可以退出程序。")
    
    while True:
        # 等待用户输入
        audio_data = record_audio_on_speech()
        user_input = transcribe_audio(audio_data)
        if not user_input:
            log("请再说一遍。")
            continue
        print(f"你说: {user_input}")

        # 检查退出命令
        if any(fuzz.partial_ratio(user_input.lower(), cmd) > 80
               for cmd in ("quit", "退出", "结束")):
            print("退出程序。")
            break

        # 获取 AI 响应
        reply = await get_model_response(user_input)
        if reply:
            print(f"AI 回复: {reply}")
            
            # 播放响应，支持 Ctrl+C 打断
            was_interrupted = await text_to_speech_with_interrupt(reply)
            
            if was_interrupted:
                print("继续对话...")
                # 继续下一轮对话循环

if __name__ == "__main__":
    asyncio.run(main())
