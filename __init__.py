from .LH_Chat import UniversalAIChat, UniversalGGUFLoader
from .LH_Utils import Qwen3TextSplitter, LoRA_AllInOne_Saver

NODE_CLASS_MAPPINGS = {
    "UniversalGGUFLoader": UniversalGGUFLoader,
    "UniversalAIChat": UniversalAIChat,
    "Qwen3TextSplitter": Qwen3TextSplitter,
    "LoRA_AllInOne_Saver": LoRA_AllInOne_Saver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalGGUFLoader": "LoraHelper_Loader",
    "UniversalAIChat": "LoraHelper_Chat",
    "Qwen3TextSplitter": "LoraHelper_Splitter",
    "LoRA_AllInOne_Saver": "LoraHelper_Saver"
}