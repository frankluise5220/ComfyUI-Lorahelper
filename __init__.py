from .LH_Chat import UniversalAIChat, UniversalGGUFLoader, LH_History_Monitor
from .LH_Utils import Qwen3TextSplitter, LoRA_AllInOne_Saver

NODE_CLASS_MAPPINGS = {
    "UniversalGGUFLoader": UniversalGGUFLoader,
    "UniversalAIChat": UniversalAIChat,
    "LH_History_Monitor": LH_History_Monitor,
    "Qwen3TextSplitter": Qwen3TextSplitter,
    "LoRA_AllInOne_Saver": LoRA_AllInOne_Saver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalGGUFLoader": "Qwen3_GGUF_loader",
    "UniversalAIChat": "LoraHelper_Chat",
    "LH_History_Monitor": "LoraHelper_Monitor",
    "Qwen3TextSplitter": "LoraHelper_Splitter",
    "LoRA_AllInOne_Saver": "LoraHelper_Saver"
}

WEB_DIRECTORY = "./web"
