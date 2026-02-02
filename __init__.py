from .LH_Chat import UniversalAIChat, UniversalGGUFLoader, UniversalOllamaLoader, LH_History_Monitor, LH_KeywordLoraLoader
from .LH_LlamaInstruct import LH_LlamaInstruct
from .LH_Utils import LoRA_AllInOne_Saver, LH_AutoRatio
from .LH_Text import LH_SuperText, LH_MultiTextSelector

NODE_CLASS_MAPPINGS = {
    "UniversalGGUFLoader": UniversalGGUFLoader,
    "UniversalOllamaLoader": UniversalOllamaLoader,
    "UniversalAIChat": UniversalAIChat,
    "LH_LlamaInstruct": LH_LlamaInstruct,
    "LH_History_Monitor": LH_History_Monitor,
    "LH_MultiTextSelector": LH_MultiTextSelector,
    "LoRA_AllInOne_Saver": LoRA_AllInOne_Saver,
    "LH_SuperText": LH_SuperText,
    "LH_LoraLoader": LH_KeywordLoraLoader,
    "LH_AutoRatio": LH_AutoRatio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalGGUFLoader": "LH_GGUFLoader",
    "UniversalOllamaLoader": "LH_OllamaLoader",
    "UniversalAIChat": "LH_AIChat",
    "LH_LlamaInstruct": "LH_LlamaInstruct",
    "LH_History_Monitor": "LH_History_Monitor",
    "LH_MultiTextSelector": "LH_MultiTextSelector",
    "LoRA_AllInOne_Saver": "LH_AllInOne_Saver",
    "LH_SuperText": "LH_SuperText",
    "LH_LoraLoader": "LH_LoraLoader",
    "LH_AutoRatio": "LH_AutoRatio",
}

WEB_DIRECTORY = "./web"
