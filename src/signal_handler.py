"""
系统信号处理模块
处理 Ctrl+C 等系统信号，实现智能打断功能
"""

import signal
import sys
from config import VERBOSE

def log(msg):
    if VERBOSE:
        print(msg)

class SignalHandler:
    """系统信号处理器"""
    
    def __init__(self):
        self.is_playing = False
        self.should_interrupt = False
        self.playback_process = None
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """处理 Ctrl+C 信号"""
        if self.is_playing:
            # 如果正在播放，打断播放
            self.should_interrupt = True
            if self.playback_process:
                try:
                    self.playback_process.terminate()
                    self.playback_process.wait(timeout=1)
                except:
                    pass
            print("\n🔇 播放被打断，等待语音输入...")
            return
        else:
            # 如果不在播放，退出程序
            print("\n👋 退出程序。")
            sys.exit(0)
    
    def set_playing_state(self, is_playing, process=None):
        """设置播放状态"""
        self.is_playing = is_playing
        self.playback_process = process
        self.should_interrupt = False
    
    def reset_interrupt_flag(self):
        """重置打断标志"""
        self.should_interrupt = False
