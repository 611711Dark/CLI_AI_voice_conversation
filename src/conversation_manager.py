"""
对话管理模块
管理整个对话流程和 AI 交互
"""

import asyncio
from openai import OpenAI
from fuzzywuzzy import fuzz

from .audio_manager import AudioManager
from .speech_recognition import SpeechRecognizer
from .text_to_speech import TextToSpeech
from config import (
    API_KEY, BASE_URL, MODEL_NAME, MAX_TOKENS,
    MAX_HISTORY_LENGTH, SYSTEM_PROMPT,
    EXIT_COMMANDS, EXIT_FUZZY_THRESHOLD,
    VERBOSE
)

def log(msg):
    if VERBOSE:
        print(msg)

class ConversationManager:
    """对话管理器"""
    
    def __init__(self, signal_handler):
        self.signal_handler = signal_handler
        
        # 初始化各个组件
        self.audio_manager = AudioManager()
        self.speech_recognizer = SpeechRecognizer()
        self.tts = TextToSpeech()
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        
        # 初始化对话历史
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        
        log("✅ 对话管理器初始化完成")
    
    def _should_exit(self, user_input: str) -> bool:
        """检查用户是否想要退出"""
        return any(
            fuzz.partial_ratio(user_input.lower(), cmd) > EXIT_FUZZY_THRESHOLD
            for cmd in EXIT_COMMANDS
        )
    
    async def _get_ai_response(self, user_input: str) -> str:
        """获取 AI 响应"""
        # 添加用户输入到历史
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # 保持历史长度
        if len(self.conversation_history) > MAX_HISTORY_LENGTH:
            self.conversation_history.pop(1)  # 保留系统提示
        
        try:
            log("🤖 正在获取 AI 响应...")
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=self.conversation_history,
                max_tokens=MAX_TOKENS,
                stream=False
            )
            
            ai_response = response.choices[0].message.content
            
            # 添加 AI 响应到历史
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            return ai_response
            
        except Exception as e:
            log(f"❌ 模型请求出错: {e}")
            return ""
    
    async def start_conversation(self):
        """启动对话循环"""
        print("🎯 开始语音对话...")
        
        while True:
            try:
                # 重置打断标志
                self.signal_handler.reset_interrupt_flag()
                
                # 录制用户语音
                audio_data = self.audio_manager.record_audio()
                
                # 转录语音
                user_input = self.speech_recognizer.transcribe(audio_data)
                
                if not user_input:
                    log("⚠️ 请再说一遍。")
                    continue
                
                print(f"👤 你说: {user_input}")
                
                # 检查是否退出
                if self._should_exit(user_input):
                    print("👋 再见!")
                    break
                
                # 获取 AI 响应
                ai_response = await self._get_ai_response(user_input)
                
                if not ai_response:
                    log("⚠️ 未获得有效回复，请重试。")
                    continue
                
                print(f"🤖 AI 回复: {ai_response}")
                
                # 合成语音
                audio_file = await self.tts.synthesize(ai_response)
                
                if audio_file:
                    # 播放语音（支持打断）
                    was_interrupted = await self.audio_manager.play_audio_with_interrupt(
                        audio_file, self.signal_handler
                    )
                    
                    # 清理临时文件
                    self.tts.cleanup_temp_file(audio_file)
                    
                    if was_interrupted:
                        print("🔄 继续对话...")
                
            except Exception as e:
                log(f"❌ 对话过程中发生错误: {e}")
                continue
