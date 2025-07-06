# 智能语音对话助手

一个基于 AI 的实时语音对话系统，支持语音识别、文本生成和语音合成，并具有 Ctrl+C 打断播放功能。

## 功能特点

- 🎤 **实时语音识别**：使用 Whisper 模型进行高精度语音转文字
- 🤖 **智能对话**：集成 AI 模型进行智能回复
- 🔊 **语音合成**：使用 Edge TTS 进行自然语音播放
- ⚡ **打断功能**：支持 Ctrl+C 打断 AI 播放并继续对话
- 🎯 **VAD 检测**：智能语音活动检测，自动开始和停止录音

## 系统要求

- Python 3.7+
- 支持的操作系统：Windows、Linux、macOS
- 麦克风设备
- 扬声器或耳机

## 安装说明

### 1. 克隆项目
```bash
git clone https://github.com/611711Dark/CLI_AI_voice_conversation.git
cd  CLI_AI_voice_conversation
```
以及按需要安装whisper模型
```
git clone https://huggingface.co/guillaumekln/faster-whisper-base
```

### 2. 安装 Python 依赖
```bash
pip install -r requirements.txt
```

### 3. 安装音频播放器（可选，推荐）
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# 或者安装 mpg123
# Ubuntu/Debian
sudo apt-get install mpg123

# macOS
brew install mpg123
```

### 4. 配置 API 密钥
编辑 `config.py` 文件，设置您的 API 密钥和baseurl：
```python
API_KEY = "your-api-key-here"
BASE_URL = "your-api-base-url-here"
```

## 使用方法

### 启动程序
```bash
python main.py
```

### 操作说明
1. **开始对话**：程序启动后，等待语音输入提示，直接说话即可
2. **打断播放**：AI 回复时按 `Ctrl+C` 可立即打断播放
3. **退出程序**：在等待输入时按 `Ctrl+C` 退出程序
4. **语音退出**：说 "退出"、"结束" 或 "quit" 也可退出

### 配置选项
在 `config.py` 中可以调整以下参数：
- `SILENCE_THRESHOLD`：语音检测灵敏度
- `SILENCE_DURATION`：静音检测时长
- `TTS_VOICE`：语音合成音色
- `MAX_HISTORY_LENGTH`：对话历史长度

## 项目结构

```
voice_chat/
├── README.md              # 项目说明文档
├── requirements.txt       # Python 依赖列表
├── main.py               # 主程序入口
├── config.py             # 配置文件
└── src/                  # 源代码目录
    ├── audio_manager.py      # 音频管理
    ├── speech_recognition.py # 语音识别
    ├── text_to_speech.py     # 语音合成
    ├── conversation_manager.py # 对话管理
    └── signal_handler.py     # 信号处理
```

## 故障排除

### 常见问题

1. **音频播放失败**
   - 确保安装了 ffmpeg 或 mpg123
   - 检查系统音频设备是否正常

2. **语音识别不准确**
   - 调整 `SILENCE_THRESHOLD` 参数
   - 确保麦克风设备正常工作
   - 在安静环境中使用

3. **API 调用失败**
   - 检查 API 密钥是否正确
   - 确保网络连接正常
   - 检查 API 配额是否充足

4. **依赖安装失败**
   - 使用 Python 虚拟环境
   - 确保 Python 版本兼容
   - 尝试使用国内 pip 源

### 调试模式
设置 `config.py` 中的 `VERBOSE = True` 可以看到详细的调试信息。

## 开发说明

### 代码结构
- `audio_manager.py`：处理音频录制和播放
- `speech_recognition.py`：语音转文字功能
- `text_to_speech.py`：文字转语音功能
- `conversation_manager.py`：对话逻辑管理
- `signal_handler.py`：系统信号处理

### 扩展功能
- 可以轻松替换不同的 AI 模型
- 支持自定义 TTS 语音
- 可以添加更多的退出命令
- 支持多语言识别和合成

## 许可证

MIT License

