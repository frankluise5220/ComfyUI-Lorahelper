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
except ImportError:
    print("\033[31m[ComfyUI-Lorahelper] Error: llama-cpp-python not found! Please install it via 'pip install llama-cpp-python'\033[0m")
    Llama = None

# 1. 路径注册
llm_dir = os.path.join(folder_paths.models_dir, "llm")
if not os.path.exists(llm_dir):
    os.makedirs(llm_dir, exist_ok=True)
folder_paths.folder_names_and_paths["llm"] = ([llm_dir], {".gguf"})

# 2. 模型加载节点
# ==========================================================
# PROJECT: LoraHelper_Loader (GGUF Model Loader)
# MANDATORY UI ORDER (INPUT_TYPES):
#   1. model_name (File List) -> 2. n_gpu_layers (-1=Auto) -> 3. n_ctx (Context Window)
#
# LOGIC DEFINITION:
#   - Loads .gguf models from ComfyUI/models/llm
#   - Requires llama-cpp-python
# ==========================================================
class UniversalGGUFLoader:
    @classmethod
    def INPUT_TYPES(s):
        all_files = folder_paths.get_filename_list("llm")
        valid_files = [f for f in all_files if f.lower().endswith(".gguf")]
        return {
            "required": {
                "model_name": (sorted(valid_files) if valid_files else ["No .gguf found"], ),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 128}),
                "n_ctx": ("INT", {"default": 8192, "min": 512, "max": 32768}), 
            },
        }
    RETURN_TYPES = ("LLM_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "custom_nodes/MyLoraNodes"

    def load_model(self, model_name, n_gpu_layers, n_ctx):
        if Llama is None:
            raise ImportError("llama-cpp-python is not installed. Please install it to use this node.")
        
        model_path = folder_paths.get_full_path("llm", model_name)
        # Ensure model exists
        if not model_path or not os.path.exists(model_path):
             raise FileNotFoundError(f"Model file not found: {model_name}")

        model = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, n_batch=512)
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
                "system_command": ("STRING", {"multiline": True, "default": "你是一个AI提示词大师。请严格按照格式输出：\nSECTION 1: [提示词]\nSECTION 2: [标签]\nSECTION 3: [文件名]"}),
                "chat_mode": (["Enhance_Prompt", "Debug_Chat"],),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "release_vram": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE",), 
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("gen_text", "response_think", "chat_history")
    FUNCTION = "chat"
    CATEGORY = "custom_nodes/MyLoraNodes"

    def chat(self, model, context, user_prompt, system_command, chat_mode, max_tokens, temperature, repetition_penalty, seed, release_vram, image=None):
        # 素材标记处理
        display_up = "[IMAGE + Text] " + user_prompt if image is not None else user_prompt
        
        if image is not None:
            i = 255. * image[0].cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            current_user_content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}]
        else:
            current_user_content = user_prompt

        # 构造消息链 (Context > UP > SC)
        messages = []
        if chat_mode == "Debug_Chat" and context and context.strip():
            messages.append({"role": "user", "content": f"## 背景参考 (Context):\n{context}"})
        
        messages.append({"role": "user", "content": current_user_content})
        messages.append({"role": "system", "content": system_command})

        # 推理执行
        try:
            output = model.create_chat_completion(
                messages=messages, max_tokens=max_tokens, temperature=temperature, 
                repeat_penalty=repetition_penalty, seed=seed
            )
            full_res = output['choices'][0]['message']['content']
        except Exception as e:
            full_res = f"Error: {str(e)}"
            print(f"\033[31m[UniversalAIChat] Generation Error: {str(e)}\033[0m")

        clean_text = re.sub(r'<think>.*?</think>', '', full_res, flags=re.DOTALL).strip()

        # 生成本轮格式化卡片
        chat_history = (
            f"┏━━━━━━ USER MATERIAL (UP) ━━━━━━┓\n{display_up}\n"
            f"┣━━━━━━━ AI RESPONSE (SC) ━━━━━━━┫\n{clean_text}\n"
            f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
        )

        if release_vram:
            gc.collect()
            torch.cuda.empty_cache()
            
        return (clean_text, full_res, chat_history)

# 4. 历史监控节点 (流水线排序)
# ==========================================================
# PROJECT: LoraHelper_Monitor (History Viewer)
# MANDATORY UI ORDER (INPUT_TYPES):
#   1. input_history_card (String Input)
#
# LOGIC DEFINITION:
#   - Maintains a rolling buffer of last 5 chat interactions
#   - Merges them into a scrollable view
# ==========================================================
class LH_History_Monitor:
    _slots = []
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "input_history_card": ("STRING", {"forceInput": True}) } }
    
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("full_scroll_view", "count")
    FUNCTION = "update"
    CATEGORY = "custom_nodes/MyLoraNodes"

    def update(self, input_history_card):
        if input_history_card and (not self._slots or input_history_card != self._slots[-1]):
            self._slots.append(input_history_card)
        if len(self._slots) > 5:
            self._slots.pop(0)
        
        full_scroll_view = "\n\n".join(self._slots)
        return (full_scroll_view, len(self._slots))
