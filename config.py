# API 配置
API_KEY = "your_api_key"
BASE_URL = "your_base_url"
MODEL_NAME = "your_model_name"
MAX_TOKENS = 300

# Whisper 模型配置
WHISPER_MODEL_PATH = "faster-whisper-base"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# 音频录制配置
SAMPLE_RATE = 44100            # 采样率
SILENCE_THRESHOLD = 0.1        # 静音阈值
SILENCE_DURATION = 1.5         # 静音时长（秒）
BUFFER_SIZE = 1024             # 读取帧大小

# TTS 配置
TTS_VOICE = "zh-CN-XiaoxiaoNeural"

# 对话配置
MAX_HISTORY_LENGTH = 30        # 历史长度
SYSTEM_PROMPT = (
    "You are a super intelligent artificial intelligence assistant,"
    " and you are currently in an oral communication environment."
)

# 退出命令与阈值
EXIT_COMMANDS = ["quit", "退出", "结束","exit"]
EXIT_FUZZY_THRESHOLD = 80

# 调试模式
VERBOSE = False
