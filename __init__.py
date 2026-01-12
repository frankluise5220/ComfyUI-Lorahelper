import os

# Define environment variable for language (default: English)
# Priority: 1. Environment Variable 2. Config File 3. Default (en_US)
LANG = os.environ.get("COMFYUI_LORAHELPER_LANG")

if not LANG:
    config_path = os.path.join(os.path.dirname(__file__), "lang_config.txt")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                LANG = f.read().strip()
        except Exception:
            pass

if not LANG:
    LANG = "en_US"

from .LH_Chat import UniversalAIChat, UniversalGGUFLoader, LH_History_Monitor
from .LH_Utils import Qwen3TextSplitter, LoRA_AllInOne_Saver

NODE_CLASS_MAPPINGS = {
    "UniversalGGUFLoader": UniversalGGUFLoader,
    "UniversalAIChat": UniversalAIChat,
    "LH_History_Monitor": LH_History_Monitor,
    "Qwen3TextSplitter": Qwen3TextSplitter,
    "LoRA_AllInOne_Saver": LoRA_AllInOne_Saver
}

# Mapping: English Display Name -> Chinese Display Name
ZH_CN_DISPLAY_MAP = {
    "Qwen3_GGUF_loader": "大语言模型加载器(GGUF)",
    "LoraHelper_Chat": "Lora训练助手_核心对话",
    "LoraHelper_Monitor": "Lora训练助手_历史看板",
    "LoraHelper_Splitter": "Lora训练助手_文本切分",
    "LoraHelper_Saver": "Lora训练助手_数据集保存"
}

# Default English Mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalGGUFLoader": "Qwen3_GGUF_loader",
    "UniversalAIChat": "LoraHelper_Chat",
    "LH_History_Monitor": "LoraHelper_Monitor",
    "Qwen3TextSplitter": "LoraHelper_Splitter",
    "LoRA_AllInOne_Saver": "LoraHelper_Saver"
}

# Apply Chinese mappings if language is set to zh_CN or zh
if LANG.startswith("zh"):
    for key, en_name in NODE_DISPLAY_NAME_MAPPINGS.items():
        if en_name in ZH_CN_DISPLAY_MAP:
            NODE_DISPLAY_NAME_MAPPINGS[key] = ZH_CN_DISPLAY_MAP[en_name]

WEB_DIRECTORY = "./web"
