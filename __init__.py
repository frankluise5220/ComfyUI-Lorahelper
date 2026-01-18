from .LH_Chat import UniversalAIChat, UniversalGGUFLoader, LH_History_Monitor, LH_MultiTextSelector
from .LH_LlamaInstruct import LH_LlamaInstruct
from .LH_Utils import LoRA_AllInOne_Saver, Qwen3TextSplitter

NODE_CLASS_MAPPINGS = {
    "UniversalGGUFLoader": UniversalGGUFLoader,
    "UniversalAIChat": UniversalAIChat,
    "LH_LlamaInstruct": LH_LlamaInstruct,
    "LH_History_Monitor": LH_History_Monitor,
    "LH_MultiTextSelector": LH_MultiTextSelector,
    "LoRA_AllInOne_Saver": LoRA_AllInOne_Saver,
    "Qwen3TextSplitter": Qwen3TextSplitter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalGGUFLoader": "LH_GGUFLoader",
    "UniversalAIChat": "LH_AIChat",
    "LH_LlamaInstruct": "LH_LlamaInstruct",
    "LH_History_Monitor": "LH_History_Monitor",
    "LH_MultiTextSelector": "LH_MultiTextSelector",
    "LoRA_AllInOne_Saver": "LH_AllInOne_Saver",
    "Qwen3TextSplitter": "LH_TextSplitter (Legacy)",
}

WEB_DIRECTORY = "./web"
