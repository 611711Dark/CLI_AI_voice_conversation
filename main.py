"""
智能语音对话助手主程序
支持实时语音识别、AI对话和语音合成
"""

import asyncio
import sys
from src.conversation_manager import ConversationManager
from src.signal_handler import SignalHandler
from config import VERBOSE

def log(msg):
    if VERBOSE:
        print(msg)

async def main():
    print("🎤 智能语音对话助手启动中...")
    print("📝 提示：在 AI 播放语音时，按 Ctrl+C 可以打断播放并继续对话")
    print("🚪 提示：在等待输入时，按 Ctrl+C 可以退出程序")
    print("🎯 提示：也可以说 '退出'、'结束' 或 'quit' 来退出程序")
    print("-" * 50)
    
    # 初始化信号处理器
    signal_handler = SignalHandler()
    
    # 初始化对话管理器
    conversation_manager = ConversationManager(signal_handler)
    
    try:
        # 启动对话循环
        await conversation_manager.start_conversation()
    except KeyboardInterrupt:
        print("\n程序被用户中断退出。")
    except Exception as e:
        print(f"程序发生错误: {e}")
        sys.exit(1)
    finally:
        print("程序已退出。")

if __name__ == "__main__":
    asyncio.run(main())
