from .LH_Chat import UniversalAIChat, UniversalGGUFLoader, UniversalOllamaLoader, LH_History_Monitor, LH_KeywordLoraLoader, LH_TextDirectoryLoader
from .LH_LlamaInstruct import LH_LlamaInstruct
from .LH_Utils import LoRA_AllInOne_Saver, LH_AutoRatio
from .LH_Text import LH_SuperText, LH_MultiTextSelector
import os
import json
from aiohttp import web

# Try to import PromptServer from server (ComfyUI standard)
try:
    from server import PromptServer
except ImportError:
    # Fallback or dummy if running outside ComfyUI (unlikely)
    PromptServer = None

# --- Config Management ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "lh_config.json")

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[ComfyUI-Lorahelper] Error loading config: {e}")
    return {}

# Register API Route
if PromptServer:
    @PromptServer.instance.routes.get("/lorahelper/get_config")
    async def get_config(request):
        config = load_config()
        return web.json_response(config)

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
    "LH_TextDirectoryLoader": LH_TextDirectoryLoader,
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
    "LH_TextDirectoryLoader": "LH_TextDirectoryLoader",
    "LH_AutoRatio": "LH_AutoRatio",
}

WEB_DIRECTORY = "./web"
