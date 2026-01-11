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
        
        return (model,)

# 3. 核心对话节点
# ==========================================================
# PROJECT: LoraHelper_Chat (DeepBlue Architecture)
# MANDATORY UI ORDER (INPUT_TYPES):
#   1. model (Loader) -> 2. image (Optional)
#   3. context (History/Top) -> 4. user_prompt (Material/UP) -> 5. system_command (Command/SC)
#   6. chat_mode (Logic Switch) -> 7. max_tokens -> 8. temperature
#   9. repetition_penalty -> 10. seed -> 11. release_vram
#
# LOGIC DEFINITION:
#   - user_prompt = Input Material (UP)
#   - system_command = Executive Instructions (SC)
#   - chat_mode = [Enhance_Prompt, Debug_Chat]
# ==========================================================
class UniversalAIChat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("LLM_MODEL",), 
                "context": ("STRING", {"multiline": True, "default": ""}), 
                "user_prompt": ("STRING", {"multiline": True, "default": "在此输入素材内容 (UP)..."}), 
                "system_command": ("STRING", {"multiline": True, "default": "你是一个AI提示词大师。请严格按照格式输出：\nSECTION 1:\n请用连贯的自然语言详细描述图片内容，包括主体、表情、头饰、服饰、动作、场景和氛围。不要使用列表（不少于300个单词）。\nSECTION 2:\n请输出标准 Danbooru 风格标签，用英文逗号分隔。范围：1.主体与数量(如 1girl, solo)；2.外貌特征(保留颜色/形态，如 long hair, blue eyes)；3.衣着配饰(如 white dress, glasses)；4.动作姿态(如 sitting, hand on hip)；5.构图视角(如 upper body, close-up, from side)；6.环境背景。禁止：主观评价词(beautiful, amazing)及权重语法。\nSECTION 3:\n用职业视角为内容取一个简短的英文标题（由三个代表性名词组成，用空格分隔），用方括号括起来，例如：[woman bed lamp]。不要包含后缀或数字。"}),
                "chat_mode": (["Enhance_Prompt", "Debug_Chat"],),
                "enable_tags_extraction": ("BOOLEAN", {"default": False, "label_on": "Enable Tags", "label_off": "Disable Tags"}),
                "enable_filename_extraction": ("BOOLEAN", {"default": False, "label_on": "Enable Filename", "label_off": "Disable Filename"}),
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
    RETURN_NAMES = ("description", "tags", "filename", "chat_history")
    FUNCTION = "chat"
    CATEGORY = "custom_nodes/MyLoraNodes"

    # 强制每次运行 (Force Execution)
    # 防止 ComfyUI 因为输入未变（如固定 Seed）而跳过执行，导致用户以为“没反应”
    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def chat(self, model, context, user_prompt, system_command, chat_mode, enable_tags_extraction, enable_filename_extraction, max_tokens, temperature, repetition_penalty, seed, release_vram, image=None):
        # 0. 基础防御性处理 (Defensive Check)
        # 确保输入不为 None，即使 ComfyUI 传了空值
        if user_prompt is None: user_prompt = ""
        if system_command is None: system_command = ""
        if context is None: context = ""
        
        # 1. 隐形反推模式判定 (Implicit Vision Mode)
        # 逻辑：
        # A. 必须有图片输入 (image is not None)
        # B. 模型必须支持视觉 (model._has_vision_handler is True)
        
        has_vision_handler = getattr(model, '_has_vision_handler', False)
        
        # [Strict Logic per User Request]
        # 1. 接 image 仍然是最高优先级，接了就反推。
        # 2. 如果没接 image，就 enhance (除非调成 debug 模式)。
        
        is_vision_task = image is not None
        
        if is_vision_task and not has_vision_handler:
             # 用户连了 image，但模型不支持
             print("\033[31m[UniversalAIChat] CRITICAL WARNING: Image input detected but model has no Vision Handler!\033[0m")
             print("\033[33m[UniversalAIChat] System will attempt to run in Text-Only mode, but results may be unexpected as Vision Logic was requested.\033[0m")
             pass
        
        # 智能处理默认占位符
        if user_prompt.strip() == "在此输入素材内容 (UP)...":
            user_prompt = ""

        # ==========================================================
        # 2. 动态指令构建 (Dynamic Instruction Construction)
        # ==========================================================
        # 核心逻辑：
        # - SECTION 1 (主任务): 由 system_command 决定。如果用户没写，则使用内部默认值。
        # - SECTION 2/3 (附加任务): 由开关 (enable_tags/filename) 强制决定，硬性追加。
        
        # [Debug] 打印开关状态
        # print(f"\033[36m[UniversalAIChat] Tags Extraction: {enable_tags_extraction}, Filename Extraction: {enable_filename_extraction}\033[0m")

        # 2.1 确定基础指令 (SECTION 1)
        # 检查是否为默认 SC (空，或者使用了已知的默认模板)
        
        # 定义已知的默认模板 (用于智能切换)
        # 1. 中文默认 (INPUT_TYPES 中的默认值)
        DEFAULT_CN_VISION = "你是一个AI提示词大师。请严格按照格式输出：\nSECTION 1:\n请用连贯的自然语言详细描述图片内容，包括主体、表情、头饰、服饰、动作、场景和氛围。不要使用列表（不少于300个单词）。\nSECTION 2:\n请输出标准 Danbooru 风格标签，用英文逗号分隔。范围：1.主体与数量(如 1girl, solo)；2.外貌特征(保留颜色/形态，如 long hair, blue eyes)；3.衣着配饰(如 white dress, glasses)；4.动作姿态(如 sitting, hand on hip)；5.构图视角(如 upper body, close-up, from side)；6.环境背景。禁止：主观评价词(beautiful, amazing)及权重语法。\nSECTION 3:\n用职业视角为内容取一个简短的英文标题（由三个代表性名词组成，用空格分隔），用方括号括起来，例如：[woman bed lamp]。不要包含后缀或数字。"
        
        # 2. 英文默认 (Vision)
        DEFAULT_EN_VISION = (
            "Describe the image in detail.\n"
            "SECTION 1:\n"
            "Provide a detailed, natural language description of the image content, including subject, action, scene, and atmosphere. (Min 300 words)."
        )
        
        # 3. 英文默认 (Text)
        DEFAULT_EN_TEXT = (
            "Refine the following text.\n"
            "SECTION 1:\n"
            "Provide a refined, detailed version of the input text."
        )

        sc_stripped = system_command.strip()
        
        # 判定当前 SC 是否为某种默认值
        is_cn_vision_default = (sc_stripped == DEFAULT_CN_VISION.strip())
        is_en_vision_default = (sc_stripped == DEFAULT_EN_VISION.strip())
        is_en_text_default = (sc_stripped == DEFAULT_EN_TEXT.strip())
        is_empty = (not sc_stripped)

        # 智能切换逻辑：
        # 1. 如果为空 -> 填补默认值 (Vision/Text 对应)
        # 2. 如果是 Vision 任务，但 SC 是 Text 默认值 -> 切换为 Vision 默认
        # 3. 如果是 Text 任务，但 SC 是 Vision 默认值 -> 切换为 Text 默认
        # 4. 如果是 Vision 任务，且 SC 是 Vision 默认值 (无论是 CN 还是 EN) -> 保持不变 (尊重用户选择的语言)
        
        new_sc = None
        
        if is_empty:
             new_sc = DEFAULT_EN_VISION if is_vision_task else DEFAULT_EN_TEXT
             # print(f"\033[36m[UniversalAIChat] System Command is empty. Using default {'Vision' if is_vision_task else 'Text'} Prompt.\033[0m")
        
        elif is_vision_task:
             if is_en_text_default:
                 new_sc = DEFAULT_EN_VISION
                 # print(f"\033[36m[UniversalAIChat] Auto-switched from Text Default to Vision Default.\033[0m")
             # 如果是 CN_VISION_DEFAULT，虽然是默认值，但适用于 Vision，所以保留，不强制转 EN
        
        else: # Text Task
             if is_cn_vision_default or is_en_vision_default:
                 new_sc = DEFAULT_EN_TEXT
                 # print(f"\033[36m[UniversalAIChat] Auto-switched from Vision Default to Text Default.\033[0m")

        if new_sc:
            system_command = new_sc

        # 2.2 构建附加指令 (SECTION 2 & 3)
        extra_instructions = ""
        required_sections = ["SECTION 1"]
        
        if enable_tags_extraction:
            extra_instructions += (
                "\n\nSECTION 2:\n"
                "Extract Danbooru-style tags based on the generated description in SECTION 1. Comma-separated.\n"
                "Rule 1: MUST start with subject tags (e.g., 1girl, solo, man, 2boys).\n"
                "Rule 2: Followed by appearance, clothes, pose, background.\n"
                "Rule 3: No weights. No subjective words."
            )
            required_sections.append("SECTION 2")
            
        if enable_filename_extraction:
            extra_instructions += (
                "\n\nSECTION 3:\n"
                "Create a short title (3 words max, lower_case_with_underscores) for the generated description in SECTION 1. Output in brackets, e.g., [morning_coffee]."
            )
            required_sections.append("SECTION 3")

        # 2.3 构建格式约束 (Footer)
        # 极简版 Footer，仅列出清单
        footer_instruction = ""
        if len(required_sections) > 1:
            req_str = ", ".join(required_sections)
            footer_instruction = (
                f"\n\nOUTPUT FORMAT REQUIRED:\n"
                f"You must output {req_str} in order.\n"
                f"Do not output anything else."
            )
        
        # ==========================================================
        # 3. 构造消息内容 (Message Content Construction)
        # ==========================================================
        
        current_user_content = None
        display_up = ""

        if is_vision_task:
            # [Vision Mode]
            # 组合：[Image] + [User Prompt (Hints)] + [System Command (Task)] + [Extra (Tags/File)] + [Footer]
            
            # 注意：对于 Vision 模型，通常建议把 Task 放在 Image 之后
            
            # 如果是默认 SC，system_command 已经是完整的 Base Instruction
            # 如果是自定义 SC，system_command 是用户的指令
            
            # 组合 Text 部分
            # 结构：[SC/Base] + [User Prompt] + [Extra] + [Footer]
            
            final_text_parts = []
            
            # Part 1: System Command (Base Task)
            final_text_parts.append(system_command)
            
            # Part 2: User Prompt (Hints) - 如果有的话
            if user_prompt:
                final_text_parts.append(f"\n[User Hint/Input]: {user_prompt}")
            
            # Part 3: Extra Sections
            if extra_instructions:
                final_text_parts.append(extra_instructions)
            
            # Part 4: Footer
            if footer_instruction:
                final_text_parts.append(footer_instruction)
            
            final_vision_text = "\n\n".join(final_text_parts)
            display_up = f"[IMAGE]\n{final_vision_text}"
            
            # 图像处理
            i = 255. * image[0].cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            if img.mode != "RGB": img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            current_user_content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}},
                {"type": "text", "text": final_vision_text}
            ]
            
            # Vision 任务不使用独立的 System Message (合并到 User Text)
            system_command_for_msg = "" 
            
        else:
            # [Text Mode]
            # 结构：System Message = system_command
            # User Message = user_prompt + [Extra] + [Footer]
            
            # System Message 保持为 system_command
            system_command_for_msg = system_command

            # 如果是默认 SC，我们已经把它改写成了 "你是一个...专家...SECTION 1..."
            # 如果是自定义 SC，保持原样
            
            # 处理 User Prompt 空值 fallback
            if not user_prompt.strip():
                if chat_mode == "Enhance_Prompt":
                    user_prompt = "Please proceed with the task."
                else:
                    user_prompt = "Hello."
            
            # 构建 User Message 的后缀 (Extra + Footer)
            user_suffix_parts = []
            if extra_instructions:
                user_suffix_parts.append(extra_instructions)
            if footer_instruction:
                user_suffix_parts.append(footer_instruction)
            
            user_suffix = "\n\n".join(user_suffix_parts)
            
            final_user_text = f"{user_prompt}\n{user_suffix}" if user_suffix else user_prompt
            
            current_user_content = final_user_text
            
            # [Display Logic Improvement]
            # 让 Monitor 显示完整上下文 (包含 System Command)，消除用户对“指令是否生效”的疑虑
            if system_command_for_msg:
                 display_up = f"🛡️ [System Instruction]:\n{system_command_for_msg}\n\n👤 [User Input]:\n{final_user_text}"
            else:
                 display_up = final_user_text

        # 4. 构造完整消息链 (Messages List)
        messages = []
        
        # System Message (仅 Text Mode)
        if system_command_for_msg:
             messages.append({"role": "system", "content": system_command_for_msg})

        
        # Rule 2: Context 注入
        # [Universal Support - Modified]
        # 用户纠正：Monitor 只是存储，通常不连线给 Chat。
        # 只有在 Debug 模式下，用户才会手动连线或复制内容。
        # 但如果用户在 Enhance_Prompt 模式下也连了线呢？
        # 用户原话：“不需要，除非 我在debug模式的情况下，我才需要手动复制，或者连线给chat.”
        # 这意味着：如果连了线，我们应该尊重连线。
        # 但如果 Context 导致了截断，说明 Context 太长了。
        
        # 既然用户说“这是错的，并没有”，那说明刚才的“第二轮截断”并非因为 Context（因为用户可能根本没连 Context）。
        # 如果没连 Context，为什么会截断？
        # 1. 第一轮生成的太长，导致 System Command + User Prompt + Output > 2048？
        # 2. 或者用户其实连了 Context 但自己没意识到？
        # 3. 或者模型自己在发疯？
        
        # 无论如何，我们先把 Context 注入逻辑改回“尊重连线”。
        # 只要 context 有值，就注入。这没问题。
        
        # 关键是，用户说“第二轮输出不完整”，如果没连 Context，那第二轮和第一轮应该是一模一样的（假设 Prompt 没变）。
        # 如果第二轮是针对第一轮的润色（比如把第一轮的输出作为第二轮的输入），那么输入确实变长了。
        
        if context and context.strip():
            # ... (保持注入逻辑不变，因为只有连了线 context 才有值)
            pass
            
            context_header = "\n\n## Historical Context (Reference Only):\n"
            
            if is_vision_task:
                 current_text = current_user_content[1]["text"]
                 new_text = f"{context_header}{context}\n\n{current_text}"
                 current_user_content[1]["text"] = new_text
            else:
                 messages.append({"role": "user", "content": f"{context_header}{context}"})

        
        # Rule 3: User Input (Text + Image or Text only)
        # [Critical Fix for VL Models]
        # 对于某些 VL 模型 (如 Qwen-VL, Llava)，如果 content 是 list 格式，必须确保格式完全符合 llama-cpp-python 的预期。
        # 调试信息：打印消息结构
        print(f"\033[36m[UniversalAIChat] Input Messages: {len(messages)} items\033[0m")
        if is_vision_task:
             print(f"\033[36m[UniversalAIChat] Vision Task Detected. Image Size: {len(img_str)} chars\033[0m")
             
        messages.append({"role": "user", "content": current_user_content})

        # 推理执行
        try:
            # [Vision Mode Context Warning]
            # 视觉任务通常需要较长的 Context (图片 Token + 生成内容)
            if is_vision_task:
                 # 获取当前 n_ctx
                 current_n_ctx = model.n_ctx() if hasattr(model, 'n_ctx') else 0
                 if current_n_ctx < 4096:
                     print(f"\033[31m[UniversalAIChat] CRITICAL WARNING: Vision task requires at least 4096 ctx. Current: {current_n_ctx}.\033[0m")
                     print(f"\033[33m[UniversalAIChat] Please increase 'n_ctx' in Loader node to avoid truncation or errors.\033[0m")

            # [Stop Token Handling]
            # 强制锁定停止词，防止模型无限生成或吐出特殊标记
            # User Suggestion: Add <|im_start|> to stop tokens to prevent hallucinating new turns.
            stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]
            
            # [Temperature Guard]
            safe_temperature = min(max(temperature, 0.0), 2.0)
            if safe_temperature > 1.5:
                print(f"\033[33m[UniversalAIChat] Warning: High temperature ({safe_temperature}) detected. Output may be incoherent.\033[0m")

            # [Execution Logic Split]
            # User Request: 强制 ChatML 格式，且在 Text 模式下不使用 create_chat_completion (避免错误模板)
            
            if is_vision_task:
                # [Vision Task]
                # 必须使用 create_chat_completion，因为 image 处理逻辑封装在 chat_handler 中
                # 我们尝试强制修正 chat_format，但主要依赖 handler
                
                # 尝试临时覆盖 format (如果支持)
                # model.chat_format = 'chatml' 
                
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
                
                finish_reason = output['choices'][0].get('finish_reason', 'unknown')
                usage = output.get('usage', {})
                full_res = output['choices'][0]['message']['content']
                
            else:
                # [Text Task]
                # User Request (via AI Advice): 
                # 1. Abandon messages/create_chat_completion to avoid Llama-2 template errors.
                # 2. Manually construct ChatML string with System/User roles.
                # 3. Use create_completion (basic inference).
                
                prompt_parts = []
                
                # Part 1: System
                if system_command_for_msg:
                    prompt_parts.append(f"<|im_start|>system\n{system_command_for_msg}<|im_end|>\n")
                
                # Part 2: User
                # current_user_content includes User Prompt + Extra Instructions + Footer
                prompt_parts.append(f"<|im_start|>user\n{current_user_content}<|im_end|>\n")
                
                # Part 3: Assistant Start
                prompt_parts.append("<|im_start|>assistant\n")
                
                final_prompt = "".join(prompt_parts)
                
                print(f"\033[36m[UniversalAIChat] Manual ChatML Prompt Constructed ({len(final_prompt)} chars)\033[0m")
                # Debug: Print first 100 chars to verify format
                print(f"\033[36m[UniversalAIChat] Prompt Head: {final_prompt[:100].replace(chr(10), '\\n')}...\033[0m")
                
                # 3. 调用 create_completion (Raw)
                output = model.create_completion(
                    prompt=final_prompt,
                    max_tokens=max_tokens,
                    temperature=safe_temperature,
                    repeat_penalty=repetition_penalty,
                    seed=seed,
                    stop=stop_tokens
                )
                
                if not output or 'choices' not in output or not output['choices']:
                     raise ValueError("Empty response from model.")

                finish_reason = output['choices'][0].get('finish_reason', 'unknown')
                usage = output.get('usage', {})
                full_res = output['choices'][0]['text'] # create_completion 返回 'text' 字段

            
            print(f"\033[36m[UniversalAIChat] Usage: {usage}, Finish Reason: {finish_reason}\033[0m")
            
            if finish_reason == 'length':
                print(f"\033[31m[UniversalAIChat] WARNING: Output Truncated! Context Limit Reached.\033[0m")
                print(f"\033[33m[UniversalAIChat] Solution: Please increase 'n_ctx' in LoraHelper_Loader node (Current default is 2048, try 8192 or 16384).\033[0m")
                full_res += "\n\n[SYSTEM: Output Truncated due to Context Limit (n_ctx). Please increase it in Loader node.]"
            
            # [Post-Processing] 清理可能残留的 Token
            if full_res:
                 for token in ["[/INST]", "[INST]", "<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
                     full_res = full_res.replace(token, "")
            
            # [Anti-Repetition Guard]
            # 检测并移除 System Command 复读
            # 如果 full_res 以 system_command 开头（允许少量差异），则移除
            if system_command and len(system_command) > 10:
                # 简单的前缀检查
                if full_res.strip().startswith(system_command.strip()[:20]):
                    print(f"\033[33m[UniversalAIChat] Warning: System Command repetition detected at start. Attempting to clean...\033[0m")
                    # 尝试找到 System Command 的结束位置
                    # 这里假设 System Command 是完整的
                    if system_command.strip() in full_res:
                        full_res = full_res.replace(system_command.strip(), "", 1).strip()
                    else:
                        # 如果找不到完全匹配，可能是因为 Tokenization 导致的微小差异
                        # 尝试移除前 N 个字符？风险较大。
                        # 尝试匹配 SECTION 1 之前的内容
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

        # 4. 智能截取 (Smart Truncation)
        # 用户要求 gen_text 只包含 SECTION 1, 2, 3。
        # 无论是否有 <think> 标签，如果检测到 "SECTION 1:"，则丢弃其之前的所有内容。
        
        # Step A: 标准 think 标签清理 (针对闭合的标签)
        clean_text = re.sub(r'<think>.*?</think>', '', full_res, flags=re.DOTALL).strip()
        
        # [强化清理] 如果清理后仍以 <think> 开头 (说明没有闭合)，尝试暴力移除直到真正的正文
        # 策略：如果找不到 </think>，但能找到 SECTION 1，则丢弃 SECTION 1 之前的所有内容
        if clean_text.startswith('<think>'):
             # 尝试寻找 </think> 的变体
             end_think = clean_text.find('</think>')
             if end_think != -1:
                 clean_text = clean_text[end_think+8:].strip()
             else:
                 # 没找到闭合标签，依赖下面的 SECTION 锚点截取
                 pass

        # Step B: 智能锚点截取 (Smart Anchor Truncation)
        # 策略升级：优先匹配“行首”的 SECTION 1，以避免匹配到文中引用的 SECTION 1。
        # 同时放弃 rpartition (从后往前找)，改回从前往后找，防止因文末总结包含 SECTION 1 而导致整个正文被截断。
        
        # [Refined Logic] 排除 System Command 中的 "SECTION 1: 自然语言描述"
        # 我们可以查找 "SECTION 1:" 且后面不紧跟 " 自然语言描述" 的情况
        # 或者更通用地，查找 SECTION 1: 后面有换行或者非指令文本
        
        target_anchor_pattern = r'(?:^|\n)SECTION 1:(?!\s*自然语言描述)'
        match = re.search(target_anchor_pattern, clean_text)
        
        if match:
            # 从匹配到的位置开始截取
            start_index = match.start()
            # 如果匹配到的是 \nSECTION 1:，start_index 会包含 \n，我们需要保留 SECTION 1:
            # match.group() 是 "\nSECTION 1:" 或 "SECTION 1:"
            # 我们直接从 match.start() + (1 if match.group().startswith('\n') else 0) 开始?
            # 不，直接取 match.start() 之后的内容即可，保留 \n 也没关系，反正后面有 strip()
            
            # 精确处理：找到 "SECTION 1:" 的起始位置
            real_start = clean_text.find("SECTION 1:", start_index)
            clean_text = clean_text[real_start:]
        else:
            # Fallback: 如果没找到标准格式，尝试宽容匹配 (不带换行符限制)
            # 但仍然使用 find (从左往右)，以防误删
            first_anchor = clean_text.find("SECTION 1:")
            if first_anchor != -1:
                clean_text = clean_text[first_anchor:]
            else:
                # 容错：如果没找到 SECTION 1，但找到了 SECTION 2 (极罕见情况)
                # 同样使用 find
                pass
            target_anchor_2 = "SECTION 2:"
            if target_anchor_2 in clean_text:
                start_pos = clean_text.find(target_anchor_2)
                clean_text = clean_text[start_pos:]
            
            # 如果都没找到，说明可能不是标准格式输出，或者用户用了自定义 Prompt。
            # 此时再尝试处理 "未闭合的 <think>" (即只有 <think> 没有 </think>)
            # 策略：如果开头是 <think>，且找不到 </think>，这通常意味着整个输出都是思考过程或者被截断了。
            # 但既然没找到 SECTION，我们最好还是保留它，或者是给个提示？
            # [Aggressive Fix] 如果还是以 <think> 开头，说明整个回复都是思考过程，或者正文被吞了
            if clean_text.startswith('<think>'):
                # 尝试只保留最后一部分文本（风险较大），或者提示错误
                # 这里我们选择保留原样，但在 Monitor 里可能会比较难看
                pass


        # [Critical Correction] Monitor 数据流
        # 用户纠正：Chat 应该把“文本原样不动”给 Monitor，连 think 过程都要保留。
        # Monitor 负责整理原始对话历史。
        # 而 Description/Tags/Filename 端口输出的是经过智能截取和分割的内容。
        
        # 因此，chat_history 使用 full_res (保留 <think> 标签和完整内容)
        raw_clean_text = full_res
            
        chat_history = f"User: {display_up}\nAI: {raw_clean_text}"

        if release_vram:
            gc.collect()
            torch.cuda.empty_cache()
            
        # 5. 内置 Splitter 逻辑 (Built-in Splitter)
        # 将 clean_text (已截取 SECTION 1 之后的内容) 切分为 description, tags, filename
        # 默认值
        out_desc = clean_text
        out_tags = ""
        out_filename = ""
        
        # 尝试解析 SECTION 格式
        # 格式预期:
        # SECTION 1: xxx
        # SECTION 2: xxx
        # SECTION 3: xxx
        
        # [User Correction] 截取动作要在 context 之前？
        # 用户说：“我们做截取的时候，不是从这个五轮的文字里截取，是从每一次输出的文本里截取，截取动作要在context这个动作之前。”
        # 理解：用户可能是在纠正我之前的回复（我之前说把 Monitor 的历史全塞给 Chat）。
        # 现在的代码逻辑正是如此：
        # 1. full_res (Model Output) -> 2. clean_text (Smart Truncation/截取) -> 3. Splitter (解析) -> 4. Return
        # 而 chat_history 也是基于 clean_text 生成的。
        # 所以目前的截取逻辑是作用于“单次输出”的，符合用户要求。

        
        # 查找各个 Section 的位置
        # 注意：由于之前做过“从右向左查找 SECTION 1”，所以 clean_text 理论上是从 SECTION 1 开始的
        
        # 如果 clean_text 不包含 SECTION 1 字样，说明可能模型没按格式输出，或者已经被截断了
        # 我们用更通用的正则来提取
        
        # 提取 SECTION 1 (Description)
        # 逻辑升级：无论是否有 "SECTION 1:" 标签，只要是在 SECTION 2/3 之前的内容，都算作 Description
        # 先尝试标准匹配 (注意：增加了对换行的容错，且允许冒号丢失)
        # 正则含义：查找 SECTION 1(可选冒号) 后面，直到遇到 SECTION 2 或 SECTION 3 或结束
        
        match_s1 = re.search(r'SECTION 1[:：]?\s*(.*?)(?=\n\s*SECTION 2|\n\s*SECTION 3|SECTION 2|SECTION 3|$)', clean_text, re.DOTALL | re.IGNORECASE)
        if match_s1:
            out_desc = match_s1.group(1).strip()
        else:
            # Fallback: 如果没找到 SECTION 1 标签，尝试截取开头到第一个其他 SECTION 的位置
            # 找到最早出现的 SECTION 2 或 SECTION 3
            end_pos = len(clean_text)
            
            # 使用更严格的正则查找 SECTION 2/3，允许冒号丢失
            match_s2_start = re.search(r'(?:\n|^)\s*SECTION 2', clean_text, re.IGNORECASE)
            if match_s2_start:
                end_pos = min(end_pos, match_s2_start.start())
                
            match_s3_start = re.search(r'(?:\n|^)\s*SECTION 3', clean_text, re.IGNORECASE)
            if match_s3_start:
                end_pos = min(end_pos, match_s3_start.start())
            
            # 截取
            candidate_desc = clean_text[:end_pos].strip()
            if candidate_desc:
                out_desc = candidate_desc
        
        # 提取 SECTION 2 (Tags)
        # [Fix] 增强对冒号的容错，有些模型可能漏写冒号，或者写成中文冒号
        match_s2 = re.search(r'SECTION 2[:：]?\s*(.*?)(?=\nSECTION 3:|SECTION 3:|$)', clean_text, re.DOTALL | re.IGNORECASE)
        if match_s2:
            raw_tags = match_s2.group(1).strip()
            # 清理 tags: 移除可能的 markdown 列表符，统一逗号
            raw_tags = raw_tags.replace('\n', ',').replace('、', ',')
            # 简单的去重和清理
            tags_list = [t.strip() for t in raw_tags.split(',') if t.strip()]
            out_tags = ", ".join(tags_list)
            
        # 提取 SECTION 3 (Filename)
        match_s3 = re.search(r'SECTION 3[:：]?\s*(.*?)(?=$)', clean_text, re.DOTALL | re.IGNORECASE)
        if match_s3:
            raw_fn = match_s3.group(1).strip()
            # 尝试提取方括号内的内容
            match_bracket = re.search(r'\[(.*?)\]', raw_fn)
            if match_bracket:
                out_filename = match_bracket.group(1).strip()
            else:
                out_filename = raw_fn
                
        # 返回切分后的结果
        return (out_desc, out_tags, out_filename, chat_history)

# 4. 历史监控节点 (流水线排序)
# ==========================================================
# PROJECT: LoraHelper_Monitor (History Viewer)
# MANDATORY UI ORDER (INPUT_TYPES):
#   1. chat_history (Raw Text Input)
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
        return { "required": { "chat_history": ("STRING", {"forceInput": True}) } }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("context",)
    OUTPUT_NODE = True
    FUNCTION = "update"
    CATEGORY = "custom_nodes/MyLoraNodes"

    def update(self, chat_history):
        # 1. 解析输入 (支持 JSON 或 纯文本)
        import json
        user_msg = ""
        ai_msg = ""
        
        # 尝试解析特定格式 "User: ... \nAI: ..."
        if isinstance(chat_history, str) and chat_history.startswith("User:"):
             # 使用 split 分割，注意只分割第一个 "\nAI: "
             parts = chat_history.split("\nAI: ", 1)
             if len(parts) == 2:
                 user_msg = parts[0][5:].strip() # 去掉 "User: "
                 ai_msg = parts[1].strip()
             else:
                 user_msg = "Raw Input"
                 ai_msg = str(chat_history)
        else:
            try:
                data = json.loads(chat_history)
                if isinstance(data, dict):
                    user_msg = data.get("user", "")
                    ai_msg = data.get("ai", "")
                else:
                    user_msg = "Raw Input"
                    ai_msg = str(chat_history)
            except:
                 user_msg = "Raw Input"
                 ai_msg = str(chat_history)
        
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