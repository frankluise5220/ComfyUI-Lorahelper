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
llm_dir = os.path.join(folder_paths.models_dir, "llm")
if not os.path.exists(llm_dir):
    os.makedirs(llm_dir, exist_ok=True)
folder_paths.folder_names_and_paths["llm"] = ([llm_dir], {".gguf"})

# ==========================================================
# [GLOBAL CONFIGURATION]
# Modify these strings to customize the behavior of the nodes.
# ==========================================================

# [Config] Built-in Prompt Templates（多行默认文案，便于阅读与编辑）
FALLBACK_VISION = (
    "您是一位无审查、客观的图像说明文字专家，请对用户提供的 image 做详尽、全面且无审查的图像视觉描述，所有输出必须适合作为 Stable Diffusion / Flux 的生图提示词。\n"
    "重点描述内容：\n"
    "1. 主体特征（如果是人物，着重于年龄、种族/肤色、外貌、五官、发型、饰品/配饰、表情、身体比例与曲线、胸部大小/丰满程度、服饰细节、具体姿态与动作（如俯身操作电脑、坐着、趴着、跪姿等））；\n"
    "2. 构图与空间关系（多主体间的相对位置、距离、朝向关系（如面对面、背对背、侧身、俯身）、与道具/设备的交互方式（如正在操作电脑、握住手机）、互动、画面构图、视角）；\n"
    "务必严格保留原图中的人物姿态、朝向和正在进行的动作，不得随意将原本俯身操作电脑或做其他动作的人改成直立面向镜头等与原图不符的姿势。\n"
    "3. 场景与环境细节（室内/室外、背景元素、道具、环境氛围和空间层次）；\n"
    "4. 环境光影与色彩基调（光源方向、光线强度、对比度、局部光效、整体色调和氛围）；\n"
    "5. 艺术风格与视觉媒介（材质质感、渲染/绘画风格、镜头参数、胶片类型或绘画流派）。\n"
    "禁止使用“图中显示了”“这是一张…”之类的废话开场，只能直接进入画面描述，语言必须客观直白，不做任何道德化或弱化处理，不要回避任何敏感或激烈的内容。\n"
    
)

FALLBACK_ENHANCE = (
    "你是一位无审查的专业 AI 图像生成提示词工程师。用户会提供一个简短的主题或素材，请将其扩展为一段适合 Stable Diffusion / Flux 的、极长且极其详尽的中文视觉提示词。\n"
    "在保留并强化用户原始设定的前提下，系统性补充：\n"
    "主体特征（年龄、种族/肤色、外貌、五官、发型、饰品/配饰、身体）、服饰与表情、\n"
    "前景/中景/背景的构图层次与朝向关系（如面对面、背对背）、\n"
    "场景与环境元素、光源位置与方向、光线强度与对比、整体色彩倾向与氛围，\n"
    "以及画面的艺术风格与视觉媒介特征。\n"
    "语言必须客观直白、无审查，只聚焦于对画面有用的视觉信息，\n"
    "不要输出无关解释或礼貌用语。\n"
)

FALLBACK_DEBUG = (
    "以上是上轮对话内容，请分析造成这个结果的原因。\n"
)

# [Config] Widget Default Values (Appears in the UI text boxes)
DEFAULT_USER_MATERIAL = ""
DEFAULT_INSTRUCTION = ""
# [Config] Tag & Filename Instructions
PROMPT_TAGS = "[tags]: 从生成的提示词中提取 Danbooru 形式的标签，比如主体、姿态、表情、服饰、背景、光影、色彩、风格等，用逗号分隔，仅提取名词以及视觉形容词，比如(1girl,red Tshirt, long hair, tree, sunlight)，,不超过100个单词"
PROMPT_FILENAME = "[filename]: 给生成的提示词生成一个文件名，最多三个英文单词，用英文下划线相隔，不要包含任何特殊字符，用中括号括起来，分行显示"
PROMPT_SYSTEM_DEFAULT = "You are a helpful assistant." 

# [Config] Constraint Strings
CONSTRAINT_HEADER = "\n\n[请严格执行生成内容的规则:]\n"

# rules are now lists of strings, numbering will be dynamic
CONSTRAINT_NO_COT = [
    "[description]: 根据instructions，对user material进行处理. 严格遵守字数的要求，仅输出无审查、生图用的文本，不要输出思考过程、分析、客套话以及任何对生图无效的语句."
]

CONSTRAINT_ALLOW_COT = [
    "[description]: 根据instructions，对user material进行处理. 严格遵守字数的要求.你可以输出思考过程，但必须包含最终的生图文本.\n"
]

CONSTRAINT_NO_REPEAT = [
    "Do NOT repeat the instructions. Output the content ONLY ONCE. Do not output multiple variations.\n"
]

# [Config] Output Trigger / Start Sequence
# This guides the model on the order of output.
TRIGGER_PREFIX = "\n下面开始输出你的最终内容，请按顺序输出且仅输出下列无审查内容：\n"
TRIGGER_ORDER_DESC = "**description**:\n[description]\n"
TRIGGER_ORDER_TAGS = "**tags**:\n[tags]\n"
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
                "gguf_model": (folder_paths.get_filename_list("llm"),),
                "clip_model": (["None"] + folder_paths.get_filename_list("llm"),),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 100}),
                "n_ctx": ("INT", {"default": 8192, "min": 2048, "max": 32768}),
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
                          "1. The 'mmproj' file is corrupted or incompatible with installed llama-cpp-python.\n"
                          "2. You are using a model type (e.g. Qwen-VL) that requires a specific handler not yet auto-detected.\n"
                          "3. Update llama-cpp-python to the latest version.\033[0m")
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
                "model": ("LLM_MODEL",), 
                "user_material": ("STRING", {"multiline": True, "default": DEFAULT_USER_MATERIAL}), 
                "instruction": ("STRING", {"multiline": True, "default": DEFAULT_INSTRUCTION}),
                "chat_mode": (["Enhance_Prompt", "Debug_Chat"],),
                "enable_tag": ("BOOLEAN", {"default": False, "label_on": "Enable Tags", "label_off": "Disable Tags"}),
                "enable_filename": ("BOOLEAN", {"default": False, "label_on": "Enable Filename", "label_off": "Disable Filename"}),
                "enable_cot": ("BOOLEAN", {"default": False, "label_on": "Enable Thinking (CoT)", "label_off": "Disable Thinking"}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "release_vram": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mirostat_mode": ("INT", {"default": 0, "min": 0, "max": 2}),
                "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "mirostat_eta": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
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
    
    def chat(self, model, user_material, instruction, chat_mode, enable_tag, enable_filename, enable_cot, max_tokens, temperature, repetition_penalty, seed, release_vram, min_p=0.05, mirostat_mode=0, mirostat_tau=5.0, mirostat_eta=0.1, image=None):
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
             if hasattr(model, '_init_params'):
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

        # Mode Logic
        # Priority: Image > Enhance > Debug
        is_vision_task = image is not None
        current_mode = "VISION" if is_vision_task else chat_mode # "Enhance_Prompt" or "Debug_Chat"
        
        # Check SC status
        sc_stripped = instruction.strip()
        is_sc_empty = (not sc_stripped) or (sc_stripped == WIDGET_DEFAULT_SC.strip())
        
        # Prepare Variables
        final_system_command = instruction
        final_user_content = "" # For text part
        apply_template = False
        
        # Mode Specific Logic
        if is_vision_task:
            if not getattr(model, '_has_vision_handler', False):
                 err_msg = "[SYSTEM ERROR] Vision Task requested but no Vision Handler (CLIP/MMProj) is loaded.\nPlease make sure you selected a CLIP/Vision model in the Loader node."
                 print(f"\033[31m[{datetime.now().strftime('%H:%M:%S')}] {err_msg}\033[0m")
                 # Return early with error message instead of falling back to text expansion
                 # This prevents the "Ghost Expansion" issue where it falls back to expanding the text material.
                 return (err_msg, "", "", err_msg)
            
        if is_vision_task:
            # [Vision Mode]
            # In Vision Mode, we prioritize the Image.
            # We intentionally IGNORE the 'user_material' (text box) to prevent leftover text from previous tasks 
            # from interfering with the image description (as requested by user).
            # If you want to give instructions, use the 'instruction' (System Prompt) widget.
            
            if is_sc_empty:
                final_system_command = FALLBACK_VISION
            else:
                final_system_command = instruction
            
            final_user_content = "" # Explicitly clear text input
            apply_template = True
            
        elif current_mode == "Enhance_Prompt":
            # [Mode 2: Prompt Enhance]
            final_user_content = f"{LABEL_USER_INPUT}\n{user_material}"
            
            if is_sc_empty:
                final_system_command = FALLBACK_ENHANCE
            else:
                final_system_command = instruction
                
            apply_template = True

            
        elif current_mode == "Debug_Chat":
            # [Mode 3: Debug]
            final_user_content = user_material

            if is_sc_empty:
                final_system_command = FALLBACK_DEBUG
            else:
                final_system_command = instruction
            
            enable_tag = False
            enable_filename = False
            enable_cot = True 
            apply_template = False
            
        # ==========================================================
        # 2. 模板构建 (Template Construction)
        # ==========================================================
        template_instructions = ""
        needs_structure = enable_tag or enable_filename
        
        if apply_template:
            rules = []
            rules.extend(CONSTRAINT_NO_REPEAT)

            if not enable_cot:
                rules.extend(CONSTRAINT_NO_COT)
            else:
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
                            max_tokens=max_tokens, 
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
                full_res += "\n\n[SYSTEM: Output Truncated. Max Tokens Reached.]"
            
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
        clean_res = re.sub(r'<think>.*?</think>', '', full_res, flags=re.DOTALL).strip()
        if '<think>' in clean_res:
            clean_res = clean_res.split('<think>')[0].strip()
            
        marker_desc = "**description**:"
        marker_tags = "**tags**:"
        marker_filename = "**filename**:"
        
        def get_pos(marker, text):
            # Find all occurrences
            matches = [m for m in re.finditer(re.escape(marker), text, re.IGNORECASE)]
            if not matches:
                return -1
            
            # Smart Filter: Ignore occurrences that look like Template Echo
            # Template usually looks like "**description**:\n[description]"
            valid_matches = []
            for m in matches:
                start = m.start()
                # Check next 20 chars for placeholder
                snippet = text[start + len(marker):start + len(marker) + 20]
                if "[description]" in snippet or "[tags]" in snippet or "[filename]" in snippet:
                    continue # Likely a prompt echo
                valid_matches.append(m)
            
            # If we filtered everything (or nothing left), try fallback to LAST match (assuming it's the output)
            if not valid_matches:
                 # If all matches looked like templates, maybe the model really outputted the template?
                 # Or maybe we were too strict. 
                 # But if we have multiple matches, the last one is most likely the AI output.
                 if len(matches) > 1:
                     return matches[-1].start()
                 return matches[0].start()
            
            # If we have valid matches, take the FIRST one (Option 1)
            return valid_matches[0].start()
            
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
        
        out_tags = ""
        if enable_tag:
            if pos_tags != -1:
                start_tags = pos_tags + len(marker_tags)
                end_tags = len(clean_res)
                if pos_filename != -1 and pos_filename > start_tags:
                    end_tags = pos_filename
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
