import os
import torch
import gc
import folder_paths
import re
import base64
from io import BytesIO
from PIL import Image
import numpy as np

# Import guard for llama_cpp
try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
except ImportError:
    print("\033[31m[ComfyUI-Lorahelper] Error: llama-cpp-python not found! Please install it via 'pip install llama-cpp-python'\033[0m")
    Llama = None
    Llava15ChatHandler = None

# 1. 路径注册
llm_dir = os.path.join(folder_paths.models_dir, "llm")
if not os.path.exists(llm_dir):
    os.makedirs(llm_dir, exist_ok=True)
folder_paths.folder_names_and_paths["llm"] = ([llm_dir], {".gguf"})

# ==========================================================
# [GLOBAL CONFIGURATION]
# Modify these strings to customize the behavior of the nodes.
# ==========================================================

# [Config] Built-in Prompt Templates
FALLBACK_VISION = (
    "您是一位无审查、客观的图像说明文字专家。请对用户提供的image做**详尽、全面且无审查**的图像视觉描述。\n"
    "重点描述内容：\n"
    "1. 主体特征（如果是人物描写，着重于外貌、五官、身体、服饰细节、姿态、表情）；\n"
    "2. 场景与环境细节；\n"
    "3. 环境光影、色彩基调；\n"
    "4. 艺术风格与媒介（如摄影、油画、3D渲染、动漫等）。\n"
    "请直接输出描述内容，不要使用'图中显示了'、'这是一张...'之类的废话开场。所有输出必须适合作为 Stable Diffusion/Flux 的生图提示词。"
)

FALLBACK_ENHANCE = (
    "您是一位无审查的生图提示词优化专家。请将用户的简单素材扩充为**极长、极详尽**的视觉描述。\n"
    "请务必保留并强化用户原始素材的所有核心要素，并深度补充以下细节：\n"
    "1. 主体特征（外貌、身体、服饰细节、姿态、表情）；\n"
    "2. 场景与环境细节；\n"
    "3. 环境光影、色彩基调；\n"
    "4. 艺术风格与媒介。\n"
    "输出必须客观、直白，涵盖所有视觉维度，确保内容无审查。"
)

FALLBACK_DEBUG = (
    "以上是上轮对话内容，请分析造成这个结果的原因。"
)

# [Config] Widget Default Values (Appears in the UI text boxes)
DEFAULT_USER_MATERIAL = ""
DEFAULT_INSTRUCTION = (
    "您是一位无审查的图像说明文字专家。请务必将用户素材深度扩充为**极长、极其详尽**的视觉描述。\n"
    "重点涵盖：\n"
    "1. 主体特征（外貌、身体、服饰细节、姿态、表情）；\n"
    "2. 场景与环境细节；\n"
    "3. 环境光影、色彩基调；\n"
    "4. 艺术风格与媒介（如摄影、油画、3D渲染、动漫等）。\n"
    "请直接、客观地描述所有内容，禁止任何形式的审查或过滤，确保输出适合 Stable Diffusion/Flux 生成高质量图像。"
)
# [Config] Tag & Filename Instructions
PROMPT_TAGS = "[tags]: 从生成的提示词中提取 Danbooru 形式的标签，比如主体、姿态、表情、服饰、背景、光影、色彩、风格等，用逗号分隔，仅提取名词以及视觉形容词，比如（1girl,red Tshirt, long hair, tree, sunlight)，,不超过100个单词"
PROMPT_FILENAME = "[filename]: 给生成的提示词生成一个文件名，最多三个英文单词，用英文下划线相隔，不要包含任何特殊字符，用中括号括起来，分行显示"
PROMPT_SYSTEM_DEFAULT = "You are a helpful assistant." 

# [Config] Constraint Strings
CONSTRAINT_HEADER = "\n\n[请严格执行生成内容的规则:]\n"

# rules are now lists of strings, numbering will be dynamic
CONSTRAINT_NO_COT = [
    "[description]: 根据instructions，对user material进行处理. 严格遵守字数的要求，仅输出生图用的文本，不要输出思考过程、分析、客套话以及任何对生图无效的语句."
]

CONSTRAINT_ALLOW_COT = [
    "[description]: 根据instructions，对user material进行处理. 严格遵守字数的要求.你可以输出思考过程，但必须包含最终的生图文本."
]

CONSTRAINT_NO_REPEAT = [
    "Do NOT repeat the instructions."
]

# [Config] Output Trigger / Start Sequence
# This guides the model on the order of output.
TRIGGER_PREFIX = "\n下面开始输出你的最终内容，请按顺序输出且仅输出下列内容：\n"
TRIGGER_ORDER_DESC = "**description**:\n[description]"
TRIGGER_ORDER_TAGS = "**tags**:\n[tags]"
TRIGGER_ORDER_FILENAME = "**filename**:\n[filename]"
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
                # [Qwen 兼容性修复]
                # Qwen-VL 等新模型可能需要特殊的 Handler，或者干脆不需要 LlavaHandler
                # 如果检测到是 Qwen 系列模型（根据文件名），且 LlavaHandler 失败，我们可以尝试不加载 Handler
                # 或者提示用户确认模型类型。
                # 目前 llama-cpp-python 对 Qwen2-VL 的支持还在实验阶段。
                # 如果是 gguf，有些模型已经内置了 projector，不需要额外的 clip。
                
                # 尝试加载 Handler，并捕获错误而不崩溃
                try:
                    if Llava15ChatHandler:
                        chat_handler = Llava15ChatHandler(clip_model_path=clip_path)
                        print(f"\033[32m[UniversalGGUFLoader] Vision Adapter Loaded: {clip_model}\033[0m")
                    else:
                        print("\033[33m[UniversalGGUFLoader] Warning: Llava15ChatHandler missing.\033[0m")
                except Exception as e:
                    print(f"\033[31m[UniversalGGUFLoader] Failed to load CLIP handler (Llava15): {str(e)}\033[0m")
                    print("\033[33m[UniversalGGUFLoader] Attempting to continue without CLIP handler (for models with built-in vision support or incompatible mmproj)...\033[0m")
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
        
        return (model,)

# 3. 核心对话节点
# ==========================================================
# PROJECT: LoraHelper_Chat (DeepBlue Architecture)
# MANDATORY UI ORDER (INPUT_TYPES):
#   1. model (Loader) -> 2. image (Optional)
#   3. user_material (Material) -> 4. instruction (Command)
#   5. chat_mode (Logic Switch) -> 6. max_tokens -> 7. temperature
#   8. repetition_penalty -> 9. seed -> 10. release_vram
#
# LOGIC DEFINITION:
#   - user_material = Input Material
#   - instruction = Executive Instructions
#   - chat_mode = [Enhance_Prompt, Debug_Chat]
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
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "release_vram": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "tags", "filename", "raw_output")
    FUNCTION = "chat"
    CATEGORY = "custom_nodes/MyLoraNodes"

    # 强制每次运行 (Force Execution)
    # 防止 ComfyUI 因为输入未变（如固定 Seed）而跳过执行，导致用户以为“没反应”
    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def chat(self, model, user_material, instruction, chat_mode, enable_tag, enable_filename, enable_cot, max_tokens, temperature, repetition_penalty, seed, release_vram, image=None):
        # 0. 基础防御性处理 (Defensive Check)
        if user_material is None: user_material = ""
        if instruction is None: instruction = ""
        
        # ==========================================================
        # 1. 模式判定与默认指令定义 (Mode Determination & Defaults)
        # ==========================================================
        
        # Widget Default Value (视为“空”)
        WIDGET_DEFAULT_SC = ""

        # [Config] Constants moved to Global Scope (Top of file) for easy access.

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
            # 强制给一个简短的 System 角色，有时能激活 Qwen 的回复逻辑
            #messages.insert(0, {"role": "system", "content": "You are a helpful assistant that describes images in detail."})
            # [Vision Mode Guard]
            if not getattr(model, '_has_vision_handler', False):
                 raise ValueError("Vision Task detected (Image Input), but the loaded model does not have a Vision Handler (CLIP/MMProj). Please load a CLIP model in the Loader node.")

            # [Mode 1: Vision / Reverse Engineering]
            # Ignore UP (User Prompt is ignored as per request)
            # But we need an INSTRUCTION for the image.
            
            # Handle SC (System Command acts as the Instruction)
            if is_sc_empty:
                # No SC provided -> Use Fallback Instruction
                instruction_content = FALLBACK_VISION
                # Also set system command to this fallback for consistency? 
                # Or keep system command empty?
                # Usually system prompt defines "Who you are", User prompt defines "What to do".
                # For simplicity and effectiveness, we put the instruction in USER prompt.
                final_system_command = "" # Disable System Message for Vision
            else:
                # User provided SC -> Use it as the Instruction
                instruction_content = instruction
                # [FIX] Avoid duplication and System Role confusion in Vision Mode!
                # For Llama.cpp Vision, it's safest to use a SINGLE User Message containing [Image, Text].
                # We disable the System Message entirely for Vision tasks to prevent "0-token output" or handler errors.
                final_system_command = "" 
            
            # Set the content that goes into User Message
            final_user_content = instruction_content
            
            # Enable Template
            apply_template = True
            
        elif current_mode == "Enhance_Prompt":
            # [Mode 2: Prompt Enhance]
            # Use UP, wrapped with label
            final_user_content = f"{LABEL_USER_INPUT}\n{user_material}"
            
            # Handle SC
            if is_sc_empty:
                final_system_command = FALLBACK_ENHANCE
            else:
                final_system_command = instruction
                
            # Enable Template
            apply_template = True
            
        elif current_mode == "Debug_Chat":
            # [Mode 3: Debug]
            # Use UP directly (User should provide Context in UP if needed)
            final_user_content = user_material

            # Handle SC
            if is_sc_empty:
                final_system_command = FALLBACK_DEBUG
            else:
                final_system_command = instruction
            
            # Force Disable Switches
            enable_tag = False
            enable_filename = False
            enable_cot = True # Debug mode defaults to allowing thinking
            apply_template = False
            
        # ==========================================================
        # 2. 模板构建 (Template Construction)
        # ==========================================================
        template_instructions = ""
        
        # [Smart Template Logic]
        # Only apply rigid template if we actually need to extract specific parts (Tag/Filename).
        # If both are disabled, we should allow the model to flow naturally.
        needs_structure = enable_tag or enable_filename
        
        # We ignore 'apply_template' flag for content decision, only use it as a gate for modes that support it.
        # But effectively, if needs_structure is False, we append NOTHING.
        
        if apply_template:
            # [Strict Instruction Injection]
            # 用户要求：无论是用户指令还是默认指令，都要加上“仅输出最终描述”、“不要输出思考过程”、“不要生成无效文字”。
            # 这必须作为系统级的强制约束，追加在 System Command 或 User Prompt 的末尾。
            
            # [CoT Switch Logic]
            # If enable_cot is True, we SKIP the "No Thinking" constraint.
            # If enable_cot is False (default), we ENFORCE it.
            
            # [Smart Constraint] Dynamically append specific format instructions as Rules
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
            
            # [Smart Constraint] Dynamically append output trigger based on switches
            output_order = [TRIGGER_ORDER_DESC]
            if enable_tag:
                output_order.append(TRIGGER_ORDER_TAGS)
            if enable_filename:
                output_order.append(TRIGGER_ORDER_FILENAME)
            
            start_sequence = f"\n下面开始输出你的最终内容，内容包含以下{len(output_order)}个部分，请按顺序输出且仅输出下列内容：\n{chr(10).join(output_order)}{TRIGGER_SUFFIX}"
            strict_constraints += start_sequence

            # Append constraints to template_instructions (which is appended to User Message)
            # This ensures it's the LAST thing the model sees.
            template_instructions += strict_constraints
            
            # If not needs_structure, template_instructions only contains strict_constraints (if apply_template is True).
            
        # ==========================================================
        # 3. 消息组装 (Message Assembly)
        # ==========================================================
        is_qwen_model = getattr(model, '_is_qwen', False)
        
        messages = []
        # 3.1 System Message
        # [Fix] Some models require a System Message to initialize the context correctly, even if empty.
        # Especially Qwen-VL or Llama-3-Vision might expect the chat template to start with System.
        # If final_system_command is empty (Vision Mode), we skip adding it to avoid confusing the Handler?
        # User feedback suggests MISSING System message might be the cause of 0-token output.
        # Let's try adding a generic System Message if it's empty but we are in Vision Mode?
        # OR: Restore the generic system persona for Vision Mode, but keep it very simple.
        
        if final_system_command:
            messages.append({"role": "system", "content": final_system_command})
        # elif is_vision_task:
             # [Vision Fix] Qwen/Llama Vision often fail if System message is present
             # We strictly omit System message for Vision tasks to prevent 0-token output.
             # messages.append({"role": "system", "content": PROMPT_SYSTEM_DEFAULT})
    
        # 3.2 User Message
        if is_vision_task:
            # [Vision Mode]
            # Image Processing
            i = 255. * image[0].cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # [Format Handling]
            # Ensure RGB. Handle RGBA by pasting on white background (better for vision models than black default)
            if img.mode == "RGBA":
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")
                
            buffered = BytesIO()
            # [Optimization] Use JPEG for better compatibility and smaller size
            # PNG can sometimes cause issues with certain VLM tokenizers or just be too large.
            # JPEG quality 95 is virtually lossless for vision tasks.
            # [Resize Logic]
            # If the image is too large, we should resize it to avoid OOM or excessive token usage.
            # Standard VLM limit is often around 1024x1024 or 2048x2048 (depending on model).
            # Qwen-VL handles high res well, but >2048 is usually diminishing returns for simple captioning.
            # Let's cap at 1536px on the long edge to be safe and fast.
            max_dimension = 1536
            if max(img.size) > max_dimension:
                scale_factor = max_dimension / max(img.size)
                new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                print(f"\033[36m[UniversalAIChat] Image Resized to {img.size}\033[0m")

            img.save(buffered, format="JPEG", quality=95) 
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            print(f"\033[36m[UniversalAIChat] Image Processed. Size: {img.size}, Mode: {img.mode} -> RGB/JPEG\033[0m")
            
            # User Content Construction
            # [Simplicity First]
            
            if is_sc_empty:
                prefix = "Instructions for the image above:\n"
            else:
                prefix = "\n" 
            
            user_text_content = f"{prefix}{final_user_content}\n{template_instructions}"
            
            # === DEBUG PROMPT DIFFERENCE ===
            print(f"\n\033[33m[UniversalAIChat] === PROMPT CONTENT DEBUG ===\033[0m")
            print(f"\033[33m[UniversalAIChat] SWITCHES: Tag={enable_tag}, Filename={enable_filename}\033[0m")
            print(f"\033[33m[UniversalAIChat] FINAL PROMPT SENT TO MODEL:\n----------------------------------------\n{user_text_content}\n----------------------------------------\033[0m\n")
            
            # Standard Multimodal Message Structure
            # Works for Llama-3-Vision, Qwen-VL, MiniCPM-V via llama-cpp-python
            user_content_list = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}},
                {"type": "text", "text": user_text_content}
            ]
            
            messages.append({"role": "user", "content": user_content_list})
            display_up = f"[IMAGE]\n{user_text_content}"
            
            print(f"\033[36m[UniversalAIChat] Vision Prompt Constructed. Messages: {len(messages)}\033[0m")
            
        else:
            # [Text Mode (Enhance / Debug)]
            final_text_content = f"{final_user_content}{template_instructions}"
            messages.append({"role": "user", "content": final_text_content})
            
            display_up = f"🛡️ [System Instruction]:\n{final_system_command}\n\n{final_text_content}"

        # ==========================================================
        # 4. 推理执行 (Inference Execution)
        # ==========================================================
        
        # [State Management]
        # Vision models with adapters are sensitive to KV cache state.
        # We MUST reset the model state before each generation to prevent:
        # 1. "Turn off switch but still broken" (Cache corruption)
        # 2. Interference from previous turns
        # if hasattr(model, 'reset'):
        #     model.reset()
        
        # [Check for Released Model]
        if getattr(model, '_is_closed', False):
             print(f"\033[31m[UniversalAIChat] 🔴 模型已释放 (Model Released)\033[0m")
             print(f"\033[33m[UniversalAIChat] 💡 您上次运行开启了 'release_vram'，导致模型从显存中卸载。\033[0m")
             print(f"\033[33m[UniversalAIChat] 💡 请修改 [LH_GGUFLoader] 节点的任意参数（例如改变 n_ctx 或 n_gpu_layers），以触发模型重新加载。\033[0m")
             raise ValueError("Model is closed (release_vram was active). Please reload the model by changing Loader parameters.")

        # Print Debug Info
        # print(f"\033[36m[UniversalAIChat] Mode: {current_mode}\033[0m")
        # print(f"\033[36m[UniversalAIChat] System Command: {final_system_command[:50]}...\033[0m")
        
        try:
            # [Optimization] Standard Stop Tokens
            # We strictly stick to standard EOS tokens to avoid "0-token output" caused by false positives.
            # Removing custom tokens like "[PART 1: Description]" because they might be the START of the generation!
            
            # [CRITICAL FIX] Stop Token Strategy for Vision
            # User diagnosis: "Stop Token hitting start" or "0 output".
            # For Vision tasks, especially with Qwen2-VL or Llama-3-Vision via GGUF,
            # explicit stop tokens might be triggering false positives if the chat template is slightly mismatched.
            # We will disable explicit stop tokens for Vision tasks and rely on the model's EOS.
            # [Unified Inference Setup]
            if is_vision_task:
                # Vision Mode
                # 1. Stop Tokens: Disable explicit stop tokens to prevent false positives (0-token output).
                #    Vision models (like Qwen-VL) often trigger stop tokens prematurely if we force them.
                stop_tokens = None
                
                # 2. Repetition Penalty:
                # - If structure is needed (Tags/Filename), use mild penalty (1.05) to prevent infinite tag loops.
                # - If no structure, use 1.0 (no penalty) to allow natural captioning flow.
                repetition_penalty = 1.05 if (apply_template and needs_structure) else 1.0
                
                # Debug
                if apply_template and needs_structure:
                     print(f"\033[36m[UniversalAIChat] Vision + Structure: Penalty={repetition_penalty}, StopTokens=None\033[0m")

            else:
                # Text Mode (Enhance / Chat)
                # 1. Stop Tokens: Use standard ChatML/Llama stop tokens. Text models rely on these to stop.
                #    Without this, complex instructions (like "5 sections") cause the model to loop or hallucinate.
                stop_tokens = ["<|im_end|>", "<|endoftext|>"]
                
                # 2. Repetition Penalty:
                # - Always use mild penalty (1.1) for text enhancement to prevent loops.
                repetition_penalty = 1.1
                
                print(f"\033[36m[UniversalAIChat] Text Mode: Penalty={repetition_penalty}, StopTokens={stop_tokens}\033[0m")

            safe_temperature = min(max(temperature, 0.0), 2.0)
            
            # Vision Task uses create_chat_completion (mandatory for image handler)
            # Text Task also uses it now for consistency, unless specific Qwen issues arise.
            # (Previously we switched to manual ChatML for Text to avoid Llama template errors, but Qwen handles standard messages well if chat_format is set)
            
            # [Unified Inference]
            # Use create_chat_completion for both Text and Vision tasks.
            # This ensures compatibility with whatever chat_format is detected (ChatML, Llama-3, Vicuna, etc.)
            
            output = model.create_chat_completion(
                messages=messages, 
                max_tokens=max_tokens, 
                temperature=safe_temperature, 
                repeat_penalty=repetition_penalty, 
                seed=seed,
                stop=stop_tokens
            )
            if not output or 'choices' not in output or not output['choices']:
                 raise ValueError("Empty response from model.")
            full_res = output['choices'][0]['message']['content']
            finish_reason = output['choices'][0].get('finish_reason', 'unknown')
            usage = output.get('usage', {})



            
            print(f"\033[36m[UniversalAIChat] Usage: {usage}, Finish Reason: {finish_reason}\033[0m")
            
            if finish_reason == 'length':
                print(f"\033[31m[UniversalAIChat] WARNING: Output Truncated! Max Tokens or Context Limit Reached.\033[0m")
                print(f"\033[33m[UniversalAIChat] Solution 1: Increase 'max_tokens' in THIS node (Chat) - likely the cause.\033[0m")
                print(f"\033[33m[UniversalAIChat] Solution 2: Increase 'n_ctx' in Loader node (if input is very long).\033[0m")
                full_res += "\n\n[SYSTEM: Output Truncated. Please increase 'max_tokens' (Chat Node) or 'n_ctx' (Loader Node).]"
            
            # [Post-Processing] 清理可能残留的 Token
            if full_res:
                 for token in ["[/INST]", "[INST]", "<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
                     full_res = full_res.replace(token, "")
            
            # [Anti-Repetition Guard]
            # 检测并移除 System Command 复读
            # 如果 full_res 以 system_command 开头（允许少量差异），则移除
            if final_system_command and len(final_system_command) > 10:
                # 简单的前缀检查
                if full_res.strip().startswith(final_system_command.strip()[:20]):
                    print(f"\033[33m[UniversalAIChat] Warning: System Command repetition detected at start. Attempting to clean...\033[0m")
                    # 尝试找到 System Command 的结束位置
                    # 这里假设 System Command 是完整的
                    if final_system_command.strip() in full_res:
                        temp_res = full_res.replace(final_system_command.strip(), "", 1).strip()
                        if temp_res:
                            full_res = temp_res
                        else:
                            # 如果移除后为空，说明模型只是复读了指令
                            # 这种情况下，保留原内容可能更好，让用户看到“它复读了”，而不是“它没说话”
                            print(f"\033[31m[UniversalAIChat] Warning: Model only repeated the instruction!\033[0m")
                            # full_res = "[Error: Model only repeated the instruction]" # Optional
                            pass 
                    else:
                        # 如果找不到完全匹配，可能是因为 Tokenization 导致的微小差异
                        pass
            
            if not full_res:
                 # 尝试获取 finish_reason，看是否是因为 token 限制或其他原因截断
                 print(f"\033[33m[UniversalAIChat] Empty Content. Finish Reason: {finish_reason}\033[0m")
                 
        except Exception as e:
            error_msg = str(e)
            full_res = f"Error: {error_msg}"
            print(f"\033[31m[UniversalAIChat] Generation Error: {error_msg}\033[0m")
            
            # [Friendly Error Handler]
            # 针对常见的 "No KV slot available" 错误给出中文建议
            if "No KV slot available" in error_msg:
                 print(f"\033[31m[UniversalAIChat] 🔴 错误诊断: 上下文长度 (n_ctx) 不足！\033[0m")
                 print(f"\033[33m[UniversalAIChat] 💡 解决方案: 请在 [Qwen3_GGUF_loader] 节点中，将 'n_ctx' 的值调大。\033[0m")
                 print(f"\033[33m[UniversalAIChat]    - 当前可能设置过小，建议尝试 8192, 16384 或 32768。\033[0m")
                 print(f"\033[33m[UniversalAIChat]    - 视觉任务(Vision)通常需要更大的上下文空间。\033[0m")
                 
                 full_res += "\n\n[SYSTEM ERROR]: Context Window Full (n_ctx too small). Please increase 'n_ctx' in the Loader node."

        # 4. 输出解析 (Output Parsing)
        # ==========================================================
        
        # [Critical Correction] Monitor 数据流
        # Chat 应该把“文本原样不动”给 Monitor，连 think 过程都要保留。
        raw_output = f"User: {display_up}\nAI: {full_res}"

        if release_vram:
            # [Fix] Explicitly close the llama.cpp model to release VRAM
            # Simply gc.collect() is NOT enough for C++ bound objects.
            try:
                if hasattr(model, 'close'):
                    model.close()
                    print("\033[36m[UniversalAIChat] 🧹 Model Closed (VRAM Released).\033[0m")
            except Exception as e:
                print(f"\033[33m[UniversalAIChat] Warning during model close: {e}\033[0m")
            
            # Mark as closed so we can warn user next time
            model._is_closed = True
            
            gc.collect()
            torch.cuda.empty_cache()

        # 5. 简单分割逻辑 (Simple Splitter)
        # 既然 Splitter 节点已删除，这里必须承担起分割的任务。
        # 配合新的 Trigger 格式：**description**:, **tags**:, **filename**:
        
        # Step A: 清理 <think> 标签 (仅针对结构化输出端口)
        clean_res = re.sub(r'<think>.*?</think>', '', full_res, flags=re.DOTALL).strip()
        
        # 处理未闭合的 <think>
        if '<think>' in clean_res:
            clean_res = clean_res.split('<think>')[0].strip()
            
        # Step B: 定义标记 (Markers)
        # 必须与 Trigger 定义保持一致 (忽略大小写)
        marker_desc = "**description**:"
        marker_tags = "**tags**:"
        marker_filename = "**filename**:"
        
        # 辅助函数：查找位置
        def get_pos(marker, text):
            m = re.search(re.escape(marker), text, re.IGNORECASE)
            return m.start() if m else -1
            
        pos_desc = get_pos(marker_desc, clean_res)
        pos_tags = get_pos(marker_tags, clean_res)
        pos_filename = get_pos(marker_filename, clean_res)
        
        # Step C: 提取 Description
        # 逻辑：
        # 1. 如果找到 **description**:，从它后面开始。
        # 2. 如果没找到，默认从头开始。
        # 3. 截止到 **tags**: 或 **filename**: (谁在前算谁)。
        
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
        
        # Step D: 提取 Tags
        out_tags = ""
        if enable_tag and pos_tags != -1:
            start_tags = pos_tags + len(marker_tags)
            end_tags = len(clean_res)
            # 如果 filename 在 tags 后面，则截止到 filename
            if pos_filename != -1 and pos_filename > start_tags:
                end_tags = pos_filename
            
            raw_tags = clean_res[start_tags:end_tags].strip()
            # 简单清理：换行变逗号
            out_tags = raw_tags.replace("\n", ",")
            
        # Step E: 提取 Filename
        out_filename = ""
        if enable_filename and pos_filename != -1:
             start_fn = pos_filename + len(marker_filename)
             raw_fn = clean_res[start_fn:].strip()
             # 提取中括号内容
             m = re.search(r'\[(.*?)\]', raw_fn)
             if m:
                 out_filename = m.group(1)
             else:
                 out_filename = raw_fn.split('\n')[0] # 没括号就取第一行

        # ==========================================================
        # 6. 输出结果 (Return)
        # ==========================================================
        return (out_desc, out_tags, out_filename, raw_output)

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