import os
os.environ.setdefault("GGML_LOG_LEVEL", "ERROR")
os.environ.setdefault("LLAMA_LOG_LEVEL", "ERROR")
import torch
import gc
import folder_paths
import re
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from datetime import datetime
import json
import requests
import random

# Import guard for llama_cpp
try:
    import llama_cpp as _llama_cpp
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
    try:
        from llama_cpp.llama_chat_format import Llava16ChatHandler
    except ImportError:
        Llava16ChatHandler = None
    try:
        from llama_cpp.llama_chat_format import MoondreamChatHandler
    except ImportError:
        MoondreamChatHandler = None
    try:
        from llama_cpp.llama_chat_format import NanoLlavaChatHandler
    except ImportError:
        NanoLlavaChatHandler = None
    try:
        # Attempt to import Qwen2VLChatHandler (Available in newer llama-cpp-python)
        from llama_cpp.llama_chat_format import Qwen2VLChatHandler
    except ImportError:
        Qwen2VLChatHandler = None
    from llama_cpp.llama_grammar import LlamaGrammar
    try:
        if hasattr(_llama_cpp, "llama_log_set"):
            def _lh_silent_log(level, text, user_data):
                return None
            _llama_cpp.llama_log_set(_lh_silent_log, None)
    except Exception:
        pass
except ImportError:
    print("\033[31m[ComfyUI-Lorahelper] Error: llama-cpp-python not found! Please install it via 'pip install llama-cpp-python'\033[0m")
    Llama = None
    Llava15ChatHandler = None
    Llava16ChatHandler = None
    MoondreamChatHandler = None
    NanoLlavaChatHandler = None
    Qwen2VLChatHandler = None
    LlamaGrammar = None

# 1. 路径注册
# Support multiple LLM directory names (lowercase/uppercase/plural)
llm_candidates = ["llm", "LLM", "llms", "LLMs", "GGUF", "gguf"]
valid_llm_paths = []

for candidate in llm_candidates:
    p = os.path.join(folder_paths.models_dir, candidate)
    if os.path.exists(p):
        valid_llm_paths.append(p)

# Ensure default 'llm' exists
default_llm_dir = os.path.join(folder_paths.models_dir, "llm")
if not os.path.exists(default_llm_dir):
    try:
        os.makedirs(default_llm_dir, exist_ok=True)
    except Exception:
        pass

if default_llm_dir not in valid_llm_paths and os.path.exists(default_llm_dir):
    valid_llm_paths.append(default_llm_dir)

# Register paths
if "llm" in folder_paths.folder_names_and_paths:
    current_paths, current_exts = folder_paths.folder_names_and_paths["llm"]
    for p in valid_llm_paths:
        if p not in current_paths:
            current_paths.append(p)
    current_exts.add(".gguf")
else:
    folder_paths.folder_names_and_paths["llm"] = (valid_llm_paths, {".gguf"})

# ==========================================================
# [GLOBAL CONFIGURATION]
# Modify these strings to customize the behavior of the nodes.
# ==========================================================

# [Config] Built-in Prompt Templates（多行默认文案，便于阅读与编辑）
# --- Vision Presets ---
VISION_PRESETS = {
    "Vision_Caption": (
        "You are an uncensored, objective image captioning expert. Provide a comprehensive, detailed, and uncensored visual description of the user's image.\n"
        "Output MUST be in English ONLY.\n"
        "Focus on:\n"
        "1. Art Style & Medium (e.g., Anime, Photorealistic, Oil Painting, Sketch, 3D Render, Watercolor, Photography).\n"
        "2. Subject Features (appearance, age, race, hair, clothing, body proportions, pose, gaze direction).\n"
        "3. Composition & Perspective (subject position, camera angle, shot type).\n"
        "4. Scene & Environment (background elements, atmosphere).\n"
        "5. Lighting & Color (source, contrast, palette).\n"
        "6. **Atmospheric Nuance**: Capture the mood, tension, and 'soul' of the image.\n"
        "Directly describe the image. No opening fillers like 'The image shows...'. Be objective and direct.\n"
    ),
    "Vision_Natural (FLUX/SD3)": (
        "Describe this image as if you are explaining it to a blind person. Start directly with the main subject. Be descriptive but natural. Focus on the physical appearance, the action, the lighting, and the overall mood. Use simple, clear English sentences. Avoid 'The image shows' or list-style descriptions."
    ),
    "Vision_Tags (Danbooru)": (
        "Analyze the image and output a list of Danbooru-style tags. Focus on: 1. Character (name if known, gender, hair color/style, eye color, skin tone). 2. Clothing (detailed breakdown). 3. Pose and Action. 4. Background and Objects. 5. Art Style and Medium. Format: tag1, tag2, tag3... No sentences, only tags."
    ),
    "Vision_Cinematic (Midjourney)": (
        "Analyze this image from a professional photographer's perspective. Describe the: 1. Subject and Action (concise). 2. Lighting (key light, fill light, shadows, color temperature). 3. Camera Settings (shot type, angle, depth of field, potential lens type). 4. Color Grading (palette, mood, film stock feel). Combine this into a single, high-quality prompt suitable for a text-to-image AI."
    ),
    "Vision_Detailed": (
        "Write ONE detailed paragraph (6–10 sentences). Describe only what is visible: subject(s) and actions; people details if present (approx age group, gender expression if clear, hair, facial expression, pose, clothing, accessories); environment (location type, background elements, time cues); lighting (source, direction, softness/hardness, color temperature, shadows); camera viewpoint (eye-level/low/high, distance) and composition (framing, focal emphasis). No preface, no reasoning, no <think>."
    ),
    "Vision_Ultra": (
        "Write ONE ultra-detailed paragraph (10–16 sentences, ~180–320 words). Stay grounded in visible details. Include: subject micro-details (materials, textures, patterns, wear, reflections); people details if present (hair, skin tones, makeup, jewelry, fabric types, fit); environment depth (foreground/midground/background, signage/props, surface materials); lighting analysis (key/fill/back light, direction, softness, highlights, shadow shape); camera perspective (angle, lens feel, depth of field) and composition (leading lines, negative space, symmetry/asymmetry, visual hierarchy). No preface, no reasoning, no <think>."
    ),
    "Vision_Cinematic": (
        "Write ONE cinematic paragraph (8–12 sentences). Describe the scene like a film still: subject(s) and action; environment and atmosphere; lighting design (practical lights vs ambient, direction, contrast); camera language (shot type, angle, lens feel, depth of field, motion implied); composition and mood. Keep it vivid but factual. No preface, no reasoning, no <think>."
    ),
    "Vision_Analysis": (
        "Output ONLY these sections with short labels (no bullets): Subject; People (if any); Environment; Lighting; Camera/Composition; Color/Texture. In each section, write 2–4 sentences of concrete visible details. If something is not visible, write 'not visible'. No preface, no reasoning, no <think>."
    )
}

# --- Text Presets ---
TEXT_PRESETS = {
    "Enhance_Prompt (Creative)": (
        "Refine and enhance the following user prompt for creative text-to-image generation (Stable Diffusion / Flux).\n"
        "Keep the core meaning and keywords, but make it extremely expressive, visually rich, and detailed.\n"
        "Expand on:\n"
        "1. **Intricate Details**: Clothing, accessories, textures.\n"
        "2. **Environment & Atmosphere**: Lighting, weather, mood.\n"
        "3. **Character**: Appearance, pose, expression.\n"
        "4. **Style**: Medium, camera angle, art style.\n"
        "5. **Atmospheric Nuance**: Capture the 'soul' and mood.\n"
        "Output **only the improved prompt text** in English. No reasoning, no explanations. 300+ words, 20+ descriptors.\n"
    ),
    "Text_Refine": (
        "Write ONE clear, concise photography prompt paragraph (120–200 words) that preserves the user’s intent and subject details. Focus on visual facts: subject, action, environment, lighting, and camera. Remove redundancy. Output in English only. No preface, no reasoning, no <think>."
    ),
    "Text_Translation": (
        "You are a professional prompt translator. Translate the user's input into high-quality English for text-to-image generation. Ensure accurate terminology for art styles, lighting, and visual elements. Maintain the original meaning but optimize phrasing for AI comprehension. Output ONLY the English translation. No explanations."
    ),
    "Text_Creative_Rewrite": (
        "You are a creative photography prompt writer. Rewrite the user’s scene into ONE fresh, imaginative photography prompt paragraph (150–250 words).\n"
        "Strict output rules:\n"
        "- Output ONLY the prompt paragraph. Start immediately with the scene.\n"
        "- No reasoning, no planning, no meta text.\n"
        "- No <think>, no quotes, no markdown.\n"
        "Preserve the core intent while adding vivid imagery and cohesive narrative flair. Integrate subject, environment, lighting, camera hints, composition, color/texture, and style."
    ),
    "Text_Artistic": (
        "You craft artistic photography prompts. Write ONE artistic photography prompt paragraph (180–260 words).\n"
        "Strict output rules:\n"
        "- Output ONLY the prompt paragraph. Start immediately with the scene.\n"
        "- No reasoning, no planning, no meta text.\n"
        "- No <think>, no quotes, no markdown.\n"
        "Weave in subject, scene, and lighting with explicit style references (e.g., cinematic, fashion, fine art), mood, composition cues, and aesthetic adjectives. Keep it cohesive and visually rich."
    ),
    "Text_Technical": (
        "You convert scenes into technical photography directives. Write ONE clear, actionable photography prompt paragraph (130–210 words).\n"
        "Strict output rules:\n"
        "- Output ONLY the prompt paragraph. Start immediately with the scene.\n"
        "- No reasoning, no planning, no meta text.\n"
        "- No <think>, no quotes, no markdown.\n"
        "Cover: subject and scene plus focal length, aperture, depth of field, shooting angle, lighting type/direction, color temperature, focus target, and composition priorities as sentences."
    )
}

FALLBACK_DEBUG = (
    "The previous round of conversation is above. Please analyze the reason for this result.\n"
)

# [Config] Widget Default Values (Appears in the UI text boxes)
DEFAULT_USER_MATERIAL = ""
DEFAULT_INSTRUCTION = ""
# [Config] Tag & Filename Instructions
PROMPT_TAGS = (
    "[tags]: Generate a detailed list of Danbooru-style tags based on the visual information. **Must be in English ONLY**.\n"
    "Focus on extracting:\n"
    "1. Art Style (e.g., anime, photorealistic, oil painting, sketch, 3d render, greyscale, monochrome);\n"
    "2. Quality & Medium (e.g., masterpiece, best quality, 4k, film grain, traditional media);\n"
    "3. Character Features (clothing, action, expression, gaze);\n"
    "4. Background, Environment, Lighting (e.g., cinematic lighting, ray tracing).\n"
    "**NO Chinese allowed**. Separate tags with commas.\n"
)
PROMPT_FILENAME = (
    "[filename]: Generate a filename for the prompt, max 3 English words separated by underscores. No special characters. Enclose in square brackets, on a new line.\n"
)
PROMPT_SYSTEM_DEFAULT = "You are a helpful assistant.\n" 

# [Config] Constraint Strings
CONSTRAINT_HEADER = "\n\n[Strictly follow these generation rules:]\n"

# rules are now lists of strings, numbering will be dynamic
CONSTRAINT_NO_COT = [
    "[description]: Process the user material according to instructions. Strictly follow word count requirements. Output ONLY the prompt text for image generation. Do NOT output thinking process, analysis, or conversational fillers.\n"
]

CONSTRAINT_ALLOW_COT = [
    "[description]: Process the user material according to instructions. Strictly follow word count requirements. You MAY output your thinking process, but MUST include the final prompt text.\n"
]

CONSTRAINT_NO_REPEAT = [
    "Do NOT repeat the instructions. Output the content ONLY ONCE. Do not output multiple variations.\n"
]

# [Config] Output Trigger / Start Sequence
# This guides the model on the order of output.
TRIGGER_PREFIX = "\nNow output your final content. Output ONLY the following items in order:\n"
TRIGGER_ORDER_DESC = "**description**:\n[description]\n"
TRIGGER_ORDER_TAGS = "\n**tags**:\n[tags]\n"
TRIGGER_ORDER_FILENAME = "**filename**:\n[filename]\n"
TRIGGER_SUFFIX = "\n"

# [Config] Input Labels
# Used to wrap the user's input so the model knows what it is.
LABEL_USER_INPUT = "[User Material]:"


# 2. 模型加载节点
# ==========================================================
# PROJECT: Qwen3_GGUF_loader (GGUF Model Loader)
# MANDATORY UI ORDER (INPUT_TYPES):
#   1. gguf_model (File List) -> 2. clip_model (MMProj) -> 3. n_gpu_layers -> 4. n_ctx
#
# LOGIC DEFINITION:
#   - Loads .gguf models from ComfyUI/models/llm
#   - Supports CLIP/MMProj for Vision Models (Required for image analysis)
# ==========================================================
class UniversalGGUFLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gguf_model": (
                    folder_paths.get_filename_list("llm"),
                    {
                        "tooltip": "必选：LLM GGUF 模型文件，支持 ComfyUI/models/ 下的 llm, LLM, GGUF 等目录",
                    },
                ),
                "clip_model": (
                    ["None"] + folder_paths.get_filename_list("llm"),
                    {
                        "tooltip": "可选：Vision mmproj/CLIP 模型；为 None 时仅加载纯文本模型",
                    },
                ),
                "n_gpu_layers": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 100,
                        "tooltip": "-1 表示自动分配 GPU 层数；0 为纯 CPU；遇到显存不足时可调小",
                    },
                ),
                "n_ctx": (
                    "INT",
                    {
                        "default": 8192,
                        "min": 2048,
                        "max": 32768,
                        "tooltip": "上下文长度（token 数）。越大可处理的对话越长，但显存占用越高",
                    },
                ),
            }
        }
    RETURN_TYPES = ("LLM_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "custom_nodes/MyLoraNodes"

    def load_model(self, gguf_model, clip_model, n_gpu_layers, n_ctx):
        if Llama is None:
            raise ImportError("llama-cpp-python is not installed. Please install it to use this node.")
        
        model_path = folder_paths.get_full_path("llm", gguf_model)
        if not model_path or not os.path.exists(model_path):
             raise FileNotFoundError(f"Model file not found: {gguf_model}")

        # Setup Chat Handler for Vision (CLIP/MMProj)
        # Loader 直接加载 CLIP，保持逻辑统一
        chat_handler = None
        if clip_model != "None":
            clip_path = folder_paths.get_full_path("llm", clip_model)
            if clip_path and os.path.exists(clip_path):
                print(f"\033[34m[UniversalGGUFLoader] Attempting to load Vision Projector: {clip_model}\033[0m")
                
                # Helper function to try loading a handler
                def try_load_handler(HandlerClass, name):
                    if not HandlerClass: return None
                    try:
                        # verbose=True can be helpful but let's keep it simple
                        h = HandlerClass(clip_model_path=clip_path)
                        print(f"\033[32m[UniversalGGUFLoader] Success: {name} Vision Adapter Loaded.\033[0m")
                        return h
                    except Exception as e:
                        # Don't print stack trace for expected failures, just the error
                        print(f"\033[33m[UniversalGGUFLoader] Info: {name} handler failed ({str(e)}). Trying next...\033[0m")
                        return None
                
                # 0. Try Qwen (High Priority)
                if not chat_handler and ("qwen" in model_path.lower() or "qwen" in clip_model.lower()):
                     chat_handler = try_load_handler(Qwen2VLChatHandler, "Qwen2-VL")

                # 1. Try Llava 1.5 (Standard for many models)
                if not chat_handler:
                    chat_handler = try_load_handler(Llava15ChatHandler, "Llava 1.5")
                
                # 2. Try Llava 1.6 (Vicuna/Mistral based)
                if not chat_handler:
                    chat_handler = try_load_handler(Llava16ChatHandler, "Llava 1.6")

                # 3. Try Moondream (Specific architecture)
                if not chat_handler and "moondream" in clip_model.lower():
                     chat_handler = try_load_handler(MoondreamChatHandler, "Moondream")
                
                # 4. Try NanoLlava
                if not chat_handler and "nano" in clip_model.lower():
                     chat_handler = try_load_handler(NanoLlavaChatHandler, "NanoLlava")

                # Final Check
                if chat_handler:
                    print(f"\033[32m[UniversalGGUFLoader] Vision Model Ready.\033[0m")
                else:
                    print(f"\033[31m[UniversalGGUFLoader] Error: Failed to load ANY compatible Vision Handler for: {clip_model}\033[0m")
                    print("\033[33m[UniversalGGUFLoader] Possible reasons:\n"
                          "1. Mismatched Version: You are trying to use a 2B mmproj with a 7B model (or vice versa). MUST match exactly!\n"
                          "2. The 'mmproj' file is corrupted or incompatible with installed llama-cpp-python.\n"
                          "3. You are using a model type (e.g. Qwen-VL) that requires a specific handler not yet auto-detected.\n"
                          "4. Update llama-cpp-python to the latest version.\033[0m")
                    print("\033[33m[UniversalGGUFLoader] Continuing in Text-Only mode...\033[0m")
                    chat_handler = None
            else:
                print(f"\033[33m[UniversalGGUFLoader] CLIP model not found: {clip_model}\033[0m")

        # [Auto-Detect Chat Format]
        # 针对 Qwen 等模型，自动应用 chatml 格式，避免 llama-cpp-python 猜错。
        # 这里进行简单的文件名启发式检测。
        chat_format = None
        model_name = os.path.basename(model_path).lower()
        
        if "qwen" in model_name:
            chat_format = "chatml"
            print(f"\033[36m[UniversalGGUFLoader] Auto-detected Qwen model. Enforcing chat_format='chatml'.\033[0m")
        elif "llama-3" in model_name or "llama3" in model_name:
             chat_format = "llama-3"
        elif "vicuna" in model_name:
             chat_format = "vicuna"
        
        # 实例化模型
        model = Llama(
            model_path=model_path, 
            chat_handler=chat_handler,
            n_gpu_layers=n_gpu_layers, 
            n_ctx=n_ctx, 
            n_batch=512,
            chat_format=chat_format # 注入自动识别的格式
        )
        # 标记是否加载了 CLIP，供 Chat 节点参考
        model._loaded_clip_path = folder_paths.get_full_path("llm", clip_model) if clip_model != "None" else None
        # [Smart Vision Check] 标记模型是否拥有有效的 Vision Handler
        # 这允许 Chat 节点在用户误连图片但使用纯文本模型时，自动回退到纯文本模式，避免报错。
        model._has_vision_handler = chat_handler is not None
        # [Model Name] 记录模型文件名，用于后续的智能判断
        model._model_filename = os.path.basename(model_path)
        # [Smart Detection] Check if model is Qwen-based (for special prompt handling)
        model._is_qwen = "qwen" in os.path.basename(model_path).lower()
        
        # [Auto-Reload Support] Save init params to allow Chat node to reload the model if closed
        model._init_params = {
            "model_path": model_path,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
            "n_batch": 512,
            "chat_format": chat_format,
            "clip_path": folder_paths.get_full_path("llm", clip_model) if clip_model != "None" else None,
            "verbose": False
        }
        
        return (model,)

# ==========================================================
# 2.5 UniversalOllamaLoader (New - Ollama Support)
# ==========================================================
class OllamaModelWrapper:
    def __init__(self, model_name, base_url, timeout=120):
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._is_closed = False
        self._has_vision_handler = False 
        self._model_filename = model_name
        self._init_params = {} # Dummy

    def n_ctx(self):
        return 8192 

    def reload(self):
        # Ollama is a service, no need to "reload" strictly, but we can check connection
        try:
            requests.get(self.base_url, timeout=5)
            self._is_closed = False
        except:
            raise RuntimeError("Ollama service unreachable during reload.")

    def create_chat_completion(self, messages, max_tokens=None, temperature=0.7, top_p=0.9, stop=None, **kwargs):
        url = f"{self.base_url}/api/chat"
        
        ollama_messages = []
        for msg in messages:
            o_msg = {"role": msg["role"], "content": ""}
            if isinstance(msg["content"], list):
                text_content = ""
                images = []
                for part in msg["content"]:
                    if part["type"] == "text":
                        text_content += part["text"]
                    elif part["type"] == "image_url":
                        url_str = part["image_url"]["url"]
                        if url_str.startswith("data:image/"):
                            base64_img = url_str.split(",")[1]
                            images.append(base64_img)
                o_msg["content"] = text_content
                if images:
                    o_msg["images"] = images
            else:
                o_msg["content"] = msg["content"]
            ollama_messages.append(o_msg)

        payload = {
            "model": self.model_name,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        if stop:
            payload["options"]["stop"] = stop
            
        if "repeat_penalty" in kwargs:
            payload["options"]["repeat_penalty"] = kwargs["repeat_penalty"]
        if "seed" in kwargs and kwargs["seed"] != -1:
            payload["options"]["seed"] = kwargs["seed"]
        
        if "min_p" in kwargs:
             payload["options"]["min_p"] = kwargs["min_p"]

        if "mirostat" in kwargs:
             payload["options"]["mirostat"] = kwargs["mirostat"]
             
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            res_json = response.json()
            
            content = res_json.get("message", {}).get("content", "")
            return {
                "choices": [
                    {
                        "message": {
                            "content": content
                        },
                        "finish_reason": "stop" if res_json.get("done") else "length"
                    }
                ],
                "usage": {}
            }
            
        except Exception as e:
            raise RuntimeError(f"Ollama API Error: {e}")

class UniversalOllamaLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "model_name": ("STRING", {"default": "deepseek-r1:8b"}), 
                "is_vision_model": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("LLM_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_ollama"
    CATEGORY = "custom_nodes/MyLoraNodes"

    def load_ollama(self, ollama_url, model_name, is_vision_model):
        model = OllamaModelWrapper(model_name, ollama_url)
        model._has_vision_handler = is_vision_model
        return (model,)

# ==========================================================
# 3. UniversalAIChat (NEW - Formerly LH_LlamaInstruct)
# ==========================================================
# PROJECT: UniversalAIChat
# LOGIC DEFINITION:
#   - Advanced version of UniversalAIChat (Replaces old Logic)
#   - Supports GBNF Grammar for structured output
#   - Supports Advanced Samplers (Mirostat, Min-P)
# ==========================================================
class UniversalAIChat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    "LLM_MODEL",
                    {
                        "tooltip": "来自 UniversalGGUFLoader 的已加载 LLM 模型",
                    },
                ),
                "user_material": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": DEFAULT_USER_MATERIAL,
                        "tooltip": "用户素材文本。反推图片时会被忽略，仅在扩写/调试模式中使用",
                    },
                ),
                "instruction": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": DEFAULT_INSTRUCTION,
                        "tooltip": "系统指令/风格设定。留空时使用内置默认说明",
                    },
                ),
                "chat_mode": (
                    [
                        "Auto_Mode (Default)",
                        "Vision_Caption (Standard)",
                        "Vision_Natural (FLUX/SD3)",
                        "Vision_Tags (Danbooru)",
                        "Vision_Cinematic (Midjourney)",
                        "Enhance_Prompt (Creative)",
                        "Debug_Chat (Raw)"
                    ],
                    {
                        "default": "Auto_Mode (Default)",
                        "tooltip": "Auto_Mode: 自动模式 (连图用 Vision_Caption, 没图用 Enhance_Prompt)\nVision_Caption: 标准反推，详尽客观\nVision_Natural: 自然语言风格，适合FLUX\nVision_Tags: 仅输出标签，适合二次元\nVision_Cinematic: 摄影师视角，重光影氛围\nEnhance_Prompt: 文本扩写润色\nDebug_Chat: 纯指令模式",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": 8192,
                        "tooltip": "本次回答的最大片段长度（token）。注意：数值越大，生成内容越长，耗时也会显著增加（尤其是开启思维链的模型）",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "采样温度。数值越高越随机，越低越保守。推荐 0.6–0.9",
                    },
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {
                        "default": 1.1,
                        "min": 1.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "重复惩罚系数。>1 会减少重复句子。常用范围 1.05–1.2",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 0xffffffffffffffff,
                        "tooltip": "-1 表示随机种子；固定某个值可复现相同输出",
                    },
                ),
                "release_vram": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "勾选后每次生成结束都会关闭模型释放显存，但下次调用会重新加载模型，速度较慢",
                    },
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "连接图片后自动进入 Vision 反推模式，忽略文本素材，仅使用图像+指令",
                    },
                ),
                "min_p": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Min-P 采样阈值，控制低概率词的截断。推荐 0.05–0.15",
                    },
                ),
                "mirostat_mode": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2,
                        "tooltip": "Mirostat 采样模式：0=关闭，1/2=自适应采样。一般保持 0 即可",
                    },
                ),
                "mirostat_tau": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Mirostat 目标困惑度参数。仅在开启 Mirostat 时生效，常用 5",
                    },
                ),
                "mirostat_eta": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Mirostat 学习率参数。仅在开启 Mirostat 时生效，常用 0.1",
                    },
                ),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "tags", "filename", "raw_output")
    FUNCTION = "chat"
    CATEGORY = "custom_nodes/MyLoraNodes"

    # 强制每次运行 (Force Execution)
    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def _build_grammar(self, enable_tag, enable_filename):
        """
        Builds a GBNF grammar string based on enabled features.
        """
        # GBNF Definition
        # root ::= description tags? filename?
        # description ::= "**description**:" space text
        # tags ::= "**tags**:" space text
        # filename ::= "**filename**:" space text
        
        # Basic text definition (allows newlines)
        # We need to be careful not to consume the next keyword.
        # But GBNF for "everything until keyword" is tricky.
        # Simple approach: standard text.
        
        gbnf = r"""
root ::= description
"""
        if enable_tag:
            gbnf += " tags"
        if enable_filename:
            gbnf += " filename"
            
        gbnf += r"""
description ::= "**description**:" "\n" content
tags ::= "**tags**:" "\n" tag_content
filename ::= "**filename**:" "\n" filename_content

content ::= [^#*]+ 
tag_content ::= [^#*]+
filename_content ::= "[" [a-zA-Z0-9_]+ "]"

"""
        # Note: The regex [^#*]+ is a simplification. 
        # Ideally we want "anything that doesn't look like a start tag".
        # For now, let's try a simpler approach without strict GBNF first?
        # Or use a very loose GBNF that just mandates the headers.
        
        # Let's try a structure that enforces the headers but allows free text in between.
        
        grammar_str = r"""
root ::= description
"""
        if enable_tag:
            grammar_str += " tags"
        if enable_filename:
            grammar_str += " filename"
            
        grammar_str += r"""
description ::= "**description**:\n" text
tags ::= "**tags**:\n" text
filename ::= "**filename**:\n" filename_pattern

text ::= ( [^#] | "#" [^*] )+ 
filename_pattern ::= "[" [a-zA-Z0-9_]+ "]"
"""
        return None
    
    def chat(self, model, user_material, instruction, chat_mode, max_tokens, temperature, repetition_penalty, seed, release_vram, min_p=0.05, mirostat_mode=0, mirostat_tau=5.0, mirostat_eta=0.1, image=None):
        # 0. 基础防御性处理 (Defensive Check)
        if user_material is None: user_material = ""
        if instruction is None: instruction = ""

        # Ensure model is loaded
        if model is None:
             raise ValueError("Model is not loaded.")
        
        # Check and Reload if needed
        # Priority: Check _is_closed flag first
        need_reload = False
        if getattr(model, '_is_closed', False):
            need_reload = True
        else:
            try:
                 # Check if model context is valid
                 _ = model.n_ctx() 
            except:
                 need_reload = True

        if need_reload:
             if hasattr(model, 'reload'):
                 try:
                     model.reload()
                 except Exception as e:
                     print(f"\033[31m[UniversalAIChat] Reload failed: {e}\033[0m")
                     raise ValueError(f"Model reload failed: {e}")
             elif hasattr(model, '_init_params'):
                 print("\033[33m[UniversalAIChat] Model is closed or invalid. Reloading...\033[0m")
                 from llama_cpp import Llama
                 init_p = model._init_params
                 
                 # Re-instantiate model locally
                 try:
                     model = Llama(
                         model_path=init_p["model_path"],
                         n_gpu_layers=init_p["n_gpu_layers"],
                         n_ctx=init_p["n_ctx"],
                         n_batch=init_p["n_batch"],
                         chat_format=init_p["chat_format"],
                         verbose=init_p["verbose"],
                     )
                     # Restore attributes
                     model._init_params = init_p
                     model._loaded_clip_path = init_p.get("clip_path")
                     model._has_vision_handler = False 
                     model._model_filename = os.path.basename(init_p["model_path"])
                     model._is_closed = False # Reset flag for the new instance
                     
                     # Restore Vision Handler if needed
                     if model._loaded_clip_path:
                         try:
                            from llama_cpp.llama_chat_format import Llava15ChatHandler
                            clip_path = model._loaded_clip_path
                            if clip_path:
                                chat_handler = Llava15ChatHandler(clip_model_path=clip_path)
                                model.chat_handler = chat_handler
                                model._has_vision_handler = True
                         except:
                            pass
                 except Exception as e:
                     print(f"\033[31m[UniversalAIChat] Reload failed: {e}\033[0m")
                     raise ValueError(f"Model reload failed: {e}")
             else:
                 pass # Cannot reload, hope for the best
        
        # ==========================================================
        # 1. 模式判定与默认指令定义 (Mode Determination & Defaults)
        # ==========================================================
        
        # Widget Default Value (视为“空”)
        WIDGET_DEFAULT_SC = ""

        enable_tag = True
        enable_filename = True

        is_vision_task = image is not None
        
        # Check SC status
        sc_stripped = instruction.strip()
        is_sc_empty = (not sc_stripped) or (sc_stripped == WIDGET_DEFAULT_SC.strip())
        
        # Prepare Variables
        final_system_command = instruction
        final_user_content = ""
        apply_template = False

        eff_max_tokens = max_tokens

        # [Auto-Adjust for Vision]
        # If image is connected, vision tasks generally require more tokens.
        # We ensure a safe minimum (1024).
        if is_vision_task and eff_max_tokens < 1024:
            eff_max_tokens = 1024

        # [Safety Cap] Ensure max_tokens doesn't exceed model context limit
        try:
            ctx_limit = model.n_ctx()
            # Reserve tokens for Input (Image + System Prompt + User Text)
            # Vision models use ~1024 tokens for image embeddings typically
            reserved_input = 1536 if is_vision_task else 512
            
            safe_max = ctx_limit - reserved_input
            if safe_max < 256: safe_max = 256 # Minimum floor
            
            if eff_max_tokens > safe_max:
                print(f"\033[33m[UniversalAIChat] Auto-Adjust: max_tokens ({eff_max_tokens}) reduced to {safe_max} to fit within context window ({ctx_limit}).\033[0m")
                eff_max_tokens = safe_max
        except:
            pass

        # [Auto Mode Logic]
        # If instruction is EMPTY -> Use Preset (and apply template)
        # If instruction is CUSTOM -> Only apply template if tags/filename are requested
        
        if is_sc_empty:
             apply_template = True
        else:
             if enable_tag or enable_filename:
                 apply_template = True
             else:
                 apply_template = False

        if is_vision_task:
            if not getattr(model, '_has_vision_handler', False):
                 err_msg = "[SYSTEM ERROR] Vision Task requested but no Vision Handler (CLIP/MMProj) is loaded.\nPlease make sure you selected a CLIP/Vision model in the Loader node."
                 print(f"\033[31m[{datetime.now().strftime('%H:%M:%S')}] {err_msg}\033[0m")
                 return (err_msg, "", "", err_msg)
            
            # [Vision Mode Logic]
            current_mode = "VISION"
            
            # Determine Preset
            preset_key = "Vision_Caption" # Default
            if chat_mode in VISION_PRESETS:
                preset_key = chat_mode
            elif chat_mode == "Auto_Mode (Default)":
                preset_key = "Vision_Caption"
            
            # If user provided custom instruction, use it. Otherwise use preset.
            if not is_sc_empty:
                final_system_command = instruction
            else:
                final_system_command = VISION_PRESETS.get(preset_key, VISION_PRESETS["Vision_Caption"])
            
            final_user_content = "Analyze the image and generate the content according to the following rules:\n"
            
        else:
            # [Text/Enhance Mode Logic]
            current_mode = "TEXT"
            final_user_content = f"{LABEL_USER_INPUT}\n{user_material}"
            
            # Determine Preset
            preset_key = "Enhance_Prompt (Creative)" # Default
            if chat_mode in TEXT_PRESETS:
                preset_key = chat_mode
            elif chat_mode == "Auto_Mode (Default)":
                preset_key = "Enhance_Prompt (Creative)"
                
            if not is_sc_empty:
                final_system_command = instruction
            else:
                final_system_command = TEXT_PRESETS.get(preset_key, TEXT_PRESETS["Enhance_Prompt (Creative)"])

        if chat_mode == "Debug_Chat (Raw)":
             if not is_sc_empty:
                 final_system_command = instruction
             else:
                 final_system_command = FALLBACK_DEBUG
             # In Debug mode, we usually don't force templates unless user asks
             apply_template = False

            
        # ==========================================================
        # 2. 模板构建 (Template Construction)
        # ==========================================================
        template_instructions = ""
        needs_structure = enable_tag or enable_filename
        
        if apply_template:
            rules = []
            rules.extend(CONSTRAINT_NO_REPEAT)
            rules.extend(CONSTRAINT_ALLOW_COT)

            if enable_tag:
                rules.append(PROMPT_TAGS)
            
            if enable_filename:
                rules.append(PROMPT_FILENAME)

            strict_constraints = CONSTRAINT_HEADER
            for i, rule in enumerate(rules, 1):
                strict_constraints += f"{i}. {rule}\n"
            
            output_order = [TRIGGER_ORDER_DESC]
            if enable_tag:
                output_order.append(TRIGGER_ORDER_TAGS)
            if enable_filename:
                output_order.append(TRIGGER_ORDER_FILENAME)
            
            start_sequence = f"{TRIGGER_PREFIX}{chr(10).join(output_order)}{TRIGGER_SUFFIX}"
            strict_constraints += start_sequence
            template_instructions += strict_constraints
            
        # ==========================================================
        # 3. 消息组装 (Message Assembly)
        # ==========================================================
        messages = []
        if final_system_command:
            messages.append({"role": "system", "content": final_system_command})
    
        # 3.2 User Message
        if is_vision_task:
            # [Vision Mode]
            i = 255. * image[0].cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            if img.mode == "RGBA":
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3]) 
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")
                
            buffered = BytesIO()
            max_dimension = 1536
            if max(img.size) > max_dimension:
                scale_factor = max_dimension / max(img.size)
                new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                # print(f"\033[36m[UniversalAIChat] Image Resized to {img.size}\033[0m")

            img.save(buffered, format="JPEG", quality=95) 
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            user_text_content = f"{final_user_content}{template_instructions}"
            
            user_content_list = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}},
                {"type": "text", "text": user_text_content}
            ]
            
            messages.append({"role": "user", "content": user_content_list})
            display_up = f"[IMAGE]\n{LABEL_USER_INPUT}\n{user_material}"
            
        else:
            # [Text Mode]
            final_text_content = f"{final_user_content}{template_instructions}"
            messages.append({"role": "user", "content": final_text_content})
            display_up = f"{LABEL_USER_INPUT}\n{user_material}"

        # ==========================================================
        # 4. 推理执行 (Inference Execution)
        # ==========================================================
        
        try:
            stop_tokens = ["<|im_end|>", "<|endoftext|>", "User:", "\nUser:"] 
            safe_temperature = min(max(temperature, 0.0), 2.0)
            grammar = None
            if apply_template and needs_structure:
                 pass

            output = None
            full_res = ""
            finish_reason = "unknown"
            usage = {}

            max_attempts = 2 if is_vision_task else 1
            attempt = 0
            sampler_used = "text-default" if not is_vision_task else "vision-advanced"

            # [HOTFIX] Temporary disable vision handler for text-only tasks
            # This prevents "Failed to load mtmd context" errors when mmproj is selected but no image provided.
            original_handler = None
            if not is_vision_task and hasattr(model, 'chat_handler'):
                original_handler = model.chat_handler
                model.chat_handler = None

            try:
                while attempt < max_attempts:
                    attempt += 1

                    if is_vision_task:
                        if attempt == 1:
                            eff_min_p = min_p
                            eff_mirostat_mode = mirostat_mode
                            sampler_used = "vision-advanced"
                        else:
                            eff_min_p = 0.0
                            eff_mirostat_mode = 0
                            sampler_used = "vision-safe"
                        eff_mirostat_tau = mirostat_tau
                        eff_mirostat_eta = mirostat_eta
                    else:
                        eff_min_p = min_p
                        eff_mirostat_mode = mirostat_mode
                        eff_mirostat_tau = mirostat_tau
                        eff_mirostat_eta = mirostat_eta
                        sampler_used = "text-default"

                    local_error = None
                    try:
                        # [Fix] top_p should not be assigned min_p value. We use default top_p=0.9 and let min_p handle truncation.
                        output = model.create_chat_completion(
                            messages=messages, 
                            max_tokens=eff_max_tokens, 
                            temperature=safe_temperature, 
                            repeat_penalty=repetition_penalty, 
                            top_p=0.9, 
                            min_p=eff_min_p,
                            mirostat_mode=eff_mirostat_mode,
                            mirostat_tau=eff_mirostat_tau,
                            mirostat_eta=eff_mirostat_eta,
                            seed=seed,
                            stop=stop_tokens,
                            grammar=grammar
                        )
                    except Exception as e_inner:
                        local_error = e_inner

                    if local_error is not None:
                        # If error is about vision/embedding, try fallback
                        err_str = str(local_error).lower()
                        if is_vision_task and attempt < max_attempts:
                             print(f"\033[33m[UniversalAIChat] Vision attempt {attempt} failed ({err_str}). Retrying with SAFE samplers...\033[0m")
                             continue
                        else:
                            raise local_error
                    else:
                        # Success
                        break
            finally:
                # Restore original handler if it was removed
                if original_handler is not None and hasattr(model, 'chat_handler'):
                    model.chat_handler = original_handler

            if not output or 'choices' not in output or not output['choices']:
                 raise ValueError("Empty response from model.")
            full_res = output['choices'][0]['message']['content']
            finish_reason = output['choices'][0].get('finish_reason', 'unknown')
            usage = output.get('usage', {})

            if finish_reason == 'length':
                full_res += "\n\n[SYSTEM: Output Truncated. Max Tokens Reached. Increase 'max_tokens' in widget.]"
            
            # [Post-Processing]
            if full_res:
                 for token in ["[/INST]", "[INST]", "<|im_end|>", "<|endoftext|>", "<|im_start|>", "User:"]:
                     full_res = full_res.replace(token, "")
            
        except Exception as e:
            error_msg = str(e)
            full_res = f"Error: {error_msg}"
            # print(f"\033[31m[UniversalAIChat] Generation Error: {error_msg}\033[0m")
            # Minimal error logging
            if "No KV slot available" in error_msg:
                 full_res += "\n\n[SYSTEM ERROR]: Context Window Full (n_ctx too small). Please increase 'n_ctx' in the Loader node."

        # 4. 输出解析 (Output Parsing)
        # ==========================================================
        
        # Log to Console (Raw Output)
        if is_vision_task:
            user_log = f"🛡️ [System Instruction]:\n{final_system_command}\n\n[IMAGE INPUT PROVIDED]\n(Text Input Ignored in Vision Mode)\nOriginal Text: {user_material}\n\n[Template Constraints]:\n{template_instructions}"
        else:
            user_log = f"🛡️ [System Instruction]:\n{final_system_command}\n\n{LABEL_USER_INPUT}\n{user_material}\n\n[Template Constraints]:\n{template_instructions}"
            
        debug_meta = f"[MODE: {current_mode}, SAMPLER: {sampler_used}, Temp: {safe_temperature:.2f}, Min_P: {eff_min_p:.2f}]"
        raw_output = f"User: {user_log}\n\nAI:\n{full_res}\n\n{debug_meta}"

        # Release VRAM if requested
        if release_vram:
             try:
                if hasattr(model, "close"):
                    model.close()
                print("\033[33m[UniversalAIChat] Model VRAM Released (Closed).\033[0m")
             except:
                pass
             model._is_closed = True


        # 5. 分割逻辑 (Simple Splitter)
        clean_res = full_res.strip()
            
        marker_desc = "**description**:"
        marker_tags = "**tags**:"
        marker_filename = "**filename**:"
        
        def get_pos(marker, text):
            think_spans = [(m.start(), m.end()) for m in re.finditer(r'<think>.*?</think>', text, re.DOTALL)]
            def in_think(pos):
                for s, e in think_spans:
                    if s <= pos < e:
                        return True
                return False
            matches = [m for m in re.finditer(re.escape(marker), text, re.IGNORECASE)]
            if not matches:
                return -1
            valid_matches = []
            for m in matches:
                start = m.start()
                snippet = text[start + len(marker):start + len(marker) + 20]
                if in_think(start):
                    continue
                if "[description]" in snippet or "[tags]" in snippet or "[filename]" in snippet:
                    continue
                valid_matches.append(m)
            if not valid_matches:
                 if len(matches) > 1:
                     return matches[-1].start()
                 return matches[0].start()
            return valid_matches[-1].start()
            
        pos_desc = get_pos(marker_desc, clean_res)
        pos_tags = get_pos(marker_tags, clean_res)
        pos_filename = get_pos(marker_filename, clean_res)
        
        start_desc = 0
        if pos_desc != -1:
            start_desc = pos_desc + len(marker_desc)
            
        end_desc = len(clean_res)
        candidates = []
        if pos_tags != -1 and pos_tags > start_desc: candidates.append(pos_tags)
        if pos_filename != -1 and pos_filename > start_desc: candidates.append(pos_filename)
        
        if candidates:
            end_desc = min(candidates)
            
        out_desc = clean_res[start_desc:end_desc].strip()
        
        # Clean <think> tags from prompt output (out_desc)
        # We don't want the thinking process in the actual prompt.
        out_desc = re.sub(r'<think>.*?</think>', '', out_desc, flags=re.DOTALL).strip()
        
        out_tags = ""
        if enable_tag:
            if pos_tags != -1:
                start_tags = pos_tags + len(marker_tags)
                end_tags = len(clean_res)
                if pos_filename != -1 and pos_filename > start_tags:
                    end_tags = pos_filename
                # [Robustness] Also stop at description if it appears after tags (out of order)
                if pos_desc != -1 and pos_desc > start_tags:
                    end_tags = min(end_tags, pos_desc)
                    
                raw_tags = clean_res[start_tags:end_tags].strip()
                out_tags = raw_tags.replace("\n", ",")
        else:
            out_tags = ""
            
        out_filename = ""
        if enable_filename:
            if pos_filename != -1:
                 start_fn = pos_filename + len(marker_filename)
                 raw_fn = clean_res[start_fn:].strip()
                 m = re.search(r'\[(.*?)\]', raw_fn)
                 if m:
                     out_filename = m.group(1)
                 else:
                     out_filename = raw_fn.split('\n')[0]
        else:
            out_filename = ""

        # ==========================================================
        # 6. 输出结果 (Return)
        # ==========================================================
        return (out_desc, out_tags, out_filename, raw_output)


# ==========================================================
# 4. UniversalAIChat_Legacy (Old AIChat Node - Shelved)
# ==========================================================
class UniversalAIChat_Legacy:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("LLM_MODEL",), 
                "user_material": ("STRING", {"multiline": True, "default": DEFAULT_USER_MATERIAL}), 
                "instruction": ("STRING", {"multiline": True, "default": DEFAULT_INSTRUCTION}),
                "chat_mode": (["Enhance_Prompt", "Debug_Chat"],),
                "enable_tag": ("BOOLEAN", {"default": False, "label_on": "Enable Tags", "label_off": "Disable Tags"}),
                "enable_filename": ("BOOLEAN", {"default": False, "label_on": "Enable Filename", "label_off": "Disable Filename"}),
                "enable_cot": ("BOOLEAN", {"default": False, "label_on": "Enable Thinking (CoT)", "label_off": "Disable Thinking"}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "release_vram": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "tags", "filename", "raw_output")
    FUNCTION = "chat"
    CATEGORY = "custom_nodes/MyLoraNodes/Legacy"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def chat(self, model, user_material, instruction, chat_mode, enable_tag, enable_filename, enable_cot, max_tokens, temperature, repetition_penalty, seed, release_vram, image=None):
        return ("Legacy Node - Shelved", "", "", "This node is deprecated. Please use the new UniversalAIChat node.")


class LH_MultiTextSelector:
    def __init__(self):
        self.index = 0
        self._spintax_pattern = re.compile(r"\{([^{}]+)\}")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (
                    ["Sequential", "Random"],
                    {
                        "tooltip": "多文本选择模式：Sequential=按顺序轮流；Random=每次随机选择一个文本",
                    },
                ),
                "text_1": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "第一个候选文本",
                    },
                ),
                "text_2": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "第二个候选文本",
                    },
                ),
                "text_3": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "第三个候选文本",
                    },
                ),
                "text_4": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "第四个候选文本",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "select"
    CATEGORY = "custom_nodes/MyLoraNodes"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def _apply_spintax(self, text):
        if not isinstance(text, str):
            return text

        def repl(match):
            raw = match.group(1)
            tokens = [p for p in raw.split("|") if p]
            if not tokens:
                return ""

            weighted = []
            total = 0.0
            for token in tokens:
                value = token
                weight = 1.0
                if "::" in token:
                    w_str, val = token.split("::", 1)
                    w_str = w_str.strip()
                    value = val
                    try:
                        weight = float(w_str)
                    except Exception:
                        weight = 1.0
                value = value
                if weight <= 0:
                    continue
                weighted.append((value, weight))
                total += weight

            if not weighted:
                return ""

            r = random.random() * total
            acc = 0.0
            for val, w in weighted:
                acc += w
                if r <= acc:
                    return val
            return weighted[-1][0]

        prev = None
        while prev != text and self._spintax_pattern.search(text):
            prev = text
            text = self._spintax_pattern.sub(repl, text)
        return text

    def select(self, mode, text_1, text_2, text_3, text_4):
        items = []
        for t in (text_1, text_2, text_3, text_4):
            if isinstance(t, str) and t.strip() != "":
                items.append(t)
        if not items:
            return ("",)
        if mode == "Random":
            chosen = random.choice(items)
        else:
            idx = self.index % len(items)
            chosen = items[idx]
            self.index += 1
        chosen = self._apply_spintax(chosen)
        return (chosen,)


# 4. 历史监控节点 (流水线排序)
# ==========================================================
# PROJECT: LoraHelper_Monitor (History Viewer)
# MANDATORY UI ORDER (INPUT_TYPES):
#   1. raw_output (Raw Text Input)
#
# LOGIC DEFINITION:
#   - Maintains a rolling buffer of last 5 chat interactions
#   - Output 1: context (Raw text for LLM)
#   - UI Display: Formatted cards (Old -> New)
# ==========================================================
class LH_History_Monitor:
    def __init__(self):
        self.history = []

    @classmethod
    def INPUT_TYPES(s):
        return { 
            "required": { 
                "raw_input": ("STRING", {"forceInput": True}),
                "clear_history": ("BOOLEAN", {"default": False, "label_on": "Clear History", "label_off": "Keep History"})
            } 
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("context",)
    OUTPUT_NODE = True
    FUNCTION = "update"
    CATEGORY = "custom_nodes/MyLoraNodes"

    def update(self, raw_input, clear_history):
        # 0. Clear History Check
        if clear_history:
            self.history = []
            # We still process the current input, but it will be the ONLY item in history.
            print("\033[36m[LH_History_Monitor] History Cleared by User.\033[0m")
        # 1. 解析输入 (支持 JSON 或 纯文本)
        import json
        user_msg = ""
        ai_msg = ""
        
        # 尝试解析特定格式 "User: ... \nAI: ..."
        if isinstance(raw_input, str) and raw_input.startswith("User:"):
             # 使用 split 分割，注意只分割第一个 "\nAI: "
             parts = raw_input.split("\nAI: ", 1)
             if len(parts) == 2:
                 user_msg = parts[0][5:].strip() # 去掉 "User: "
                 ai_msg = parts[1].strip()
             else:
                 user_msg = "Raw Input"
                 ai_msg = str(raw_input)
        else:
            try:
                data = json.loads(raw_input)
                if isinstance(data, dict):
                    user_msg = data.get("user", "")
                    ai_msg = data.get("ai", "")
                else:
                    user_msg = "Raw Input"
                    ai_msg = str(raw_input)
            except:
                 user_msg = "Raw Input"
                 ai_msg = str(raw_input)
        
        # 2. 更新历史 (去重)
        # 构造一个结构化对象存储
        new_entry = {"user": user_msg, "ai": ai_msg}
        
        # 简单去重：检查上一条是否完全一致
        if self.history:
            last = self.history[-1]
            if last["user"] == user_msg and last["ai"] == ai_msg:
                pass # 重复，忽略
            else:
                self.history.append(new_entry)
        else:
            self.history.append(new_entry)
            
        # 保持 5 轮
        if len(self.history) > 5:
            self.history.pop(0)

        # 3. 构造 Context (用于回传给 Chat)
        # 格式：Round X User: ... \n Round X AI: ...
        context_parts = []
        for i, h in enumerate(self.history):
            context_parts.append(f"Round {i+1} User: {h['user']}")
            context_parts.append(f"Round {i+1} AI: {h['ai']}")
        context = "\n\n".join(context_parts)

        # 4. 构造 UI 显示 —— 关键修改：拆分成多个短文本块
        ui_text = []
        ui_text.append("═════════ 👀 Visual History (Latest 5 Rounds) ═════════\n")
        
        for i, h in enumerate(reversed(self.history)): # 最新轮在上
            idx = len(self.history) - i
            ui_text.append(f"🔻 Round {idx} — User 输入")
            ui_text.append(h["user"] or "(空)")  # 单独一块，用户输入
            
            ui_text.append(f"🔹 Round {idx} — AI 输出")
            ui_text.append(h["ai"] or "(空)")    # 单独一块，AI输出
            
            ui_text.append("──────────────────────────\n")  # 分隔线

        # 如果历史为空
        if len(ui_text) <= 1:
             ui_text.append("（暂无对话历史）")
        
        return {"ui": {"text": ui_text}, "result": (context,)}
