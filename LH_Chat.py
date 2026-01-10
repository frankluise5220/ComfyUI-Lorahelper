import os
import torch
import gc
import folder_paths
from llama_cpp import Llama
import re

# 路径注册
llm_dir = os.path.join(folder_paths.models_dir, "llm")
if not os.path.exists(llm_dir):
    os.makedirs(llm_dir, exist_ok=True)
folder_paths.folder_names_and_paths["llm"] = ([llm_dir], {".gguf"})

class UniversalGGUFLoader:
    @classmethod
    def INPUT_TYPES(s):
        all_files = folder_paths.get_filename_list("llm")
        valid_files = [f for f in all_files if f.lower().endswith(".gguf")]
        return {
            "required": {
                "model_name": (sorted(valid_files) if valid_files else ["No .gguf found"], ),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 100}),
                "n_ctx": ("INT", {"default": 8192, "min": 512, "max": 32768}), 
            },
        }
    RETURN_TYPES = ("LLM_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "custom_nodes/MyLoraNodes"

    def load_model(self, model_name, n_gpu_layers, n_ctx):
        model_path = folder_paths.get_full_path("llm", model_name)
        model = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, n_batch=512)
        return (model,)

class UniversalAIChat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("LLM_MODEL",),
                "user_prompt": ("STRING", {"multiline": True, "default": "请在这里存放素材"}),
                "system_prompt": ("STRING", {"multiline": True, "default": "你是一个AI提示词大师"}),
                "mode": (["Enhance_Prompt", "Debug_Chat"],),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.8}),
                "repetition_penalty": ("FLOAT", {"default": 1.1}),
                "seed": ("INT", {"default": -1}),
                "release_vram": ("BOOLEAN", {"default": True}),
            },
        }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("gen_text", "response_or_think")
    FUNCTION = "chat"
    CATEGORY = "custom_nodes/MyLoraNodes"

    def chat(self, model, system_prompt, user_prompt, mode, max_tokens, temperature, repetition_penalty, seed, release_vram):
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        # --- 确认：参数已绑定 ---
        output = model.create_chat_completion(
            messages=messages, 
            max_tokens=max_tokens, 
            temperature=temperature, 
            repeat_penalty=repetition_penalty, 
            seed=seed
        )
        full_res = output['choices'][0]['message']['content']
        clean_text = re.sub(r'<think>.*?</think>', '', full_res, flags=re.DOTALL).strip()
        if release_vram:
            gc.collect()
            torch.cuda.empty_cache()
        return (clean_text, full_res)