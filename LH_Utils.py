import os
import time
import re
import json
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths

class Qwen3TextSplitter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"forceInput": True, "multiline": True}), "user_prefix": ("STRING", {"default": "Anran"})}}
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("gen_prompt", "lora_tags", "filename_final")
    FUNCTION = "split"
    CATEGORY = "custom_nodes/MyLoraNodes/Legacy"

    def split(self, text, user_prefix):
        raw = str(text).strip() if text else ""
        p_match = re.search(r'SECTION 1.*?[:：]\s*(.*?)(?=SECTION 2|SECTION 3|\[|$)', raw, re.DOTALL | re.IGNORECASE)
        gen_p = p_match.group(1).strip() if p_match else raw.split('\n\n')[0].strip()
        
        # 优化 SECTION 2 提取逻辑，防止吞掉 SECTION 3
        # 使用更严格的断言，并二次清洗
        tags_match = re.search(r'SECTION 2.*?[:：]\s*(.*?)(?=\s*SECTION 3|\s*\[|$)', raw, re.DOTALL | re.IGNORECASE)
        lora_tags = ""
        if tags_match:
            raw_tags = tags_match.group(1)
            # 二次保障：如果正则漏了，手动截断
            if "SECTION 3" in raw_tags.upper():
                raw_tags = re.split(r'SECTION 3', raw_tags, flags=re.IGNORECASE)[0]
            
            tags_src = re.sub(r':\s*\d+\.?\d*|[()\[\]{}]', '', raw_tags)
            tokens = [t.strip() for t in re.split(r'[,\n;，；]', tags_src) if 2 < len(t.strip()) < 50]
            # 过滤掉包含 "SECTION" 的 token
            tokens = [t for t in tokens if "SECTION" not in t.upper()]
            lora_tags = ", ".join(list(dict.fromkeys(tokens)))

        t_match = re.search(r'(?:SECTION 3.*?[:：]\s*)?\[([^\]]+)\]', raw, re.DOTALL | re.IGNORECASE)
        title = re.sub(r"[\[\]{}'\"`’]", '', t_match.group(1)).strip() if t_match else f"Auto_{int(time.time())}"
        return (gen_p, lora_tags, f"{user_prefix}_{title[:50]}")

# ==========================================================
# PROJECT: LoraHelper_Saver (Dataset Saver)
# MANDATORY UI ORDER (INPUT_TYPES):
#   1. images -> 2. gen_prompt -> 3. lora_tags
#   4. filename_final -> 5. folder_path -> 6. trigger_word
#   7. save_workflow
#
# LOGIC DEFINITION:
#   - Saves images to output/folder_path
#   - Creates .txt caption files (trigger_word + tags)
#   - Creates _log.txt with full description
# ==========================================================
class LoRA_AllInOne_Saver:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "gen_prompt": ("STRING", {"forceInput": True}),
                "lora_tags": ("STRING", {"forceInput": True}),
                "filename_final": ("STRING", {"forceInput": True}),
                "folder_path": ("STRING", {"default": "LoRA_Train_Data"}),
                "filename_prefix": ("STRING", {"default": "Anran"}),
                "trigger_word": ("STRING", {"default": "ChenAnran"}), 
                "save_workflow": ("BOOLEAN", {"default": True}), # 功能 2：开关
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}
        }
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True  
    CATEGORY = "custom_nodes/MyLoraNodes"

    def save(self, images, gen_prompt, lora_tags, filename_final, folder_path, filename_prefix, trigger_word, save_workflow, prompt=None, extra_pnginfo=None):
        # 0. 基础清理与智能验证
        s_file = str(filename_final).strip()
        
        # [智能修正] 如果输入的文件名过长（说明可能是误连了 Prompt 或模型输出了句子），则丢弃它
        # 阈值设定：长度 > 80 或 单词数 > 8 (放宽一点以免误杀长单词)
        if len(s_file) > 80 or len(s_file.split()) > 8:
             print(f"\033[33m[LoRA_Saver] Warning: Input filename is too long (Likely a sentence). Fallback to prefix only.\033[0m")
             s_file = ""

        # 应用 filename_prefix
        # 逻辑：最终文件名 = 前缀 + (如果 s_file 有效且不包含前缀则加上 s_file)
        prefix = filename_prefix.strip() if filename_prefix else ""
        
        if not s_file:
            # 如果没有有效文件名，直接用前缀
            final_name_str = prefix
        else:
            # 如果有文件名，检查是否已包含前缀
            if prefix and not s_file.startswith(prefix):
                final_name_str = f"{prefix}_{s_file}"
            else:
                final_name_str = s_file
        
        # 如果连前缀都没有，给个默认值
        if not final_name_str:
            final_name_str = "ComfyUI_Output"

        clean_name = re.sub(r"[\[\]{}'\"`’]", '', final_name_str)
        safe_name = re.sub(r'[\\/:*?"<>|\n\r\t]', '', clean_name).strip().replace(' ', '_')[:100] # 放宽一点限制
        
        # [最后一道防线] 如果清理后safe_name为空（例如前缀全是非法字符），强制给默认值
        if not safe_name:
            safe_name = "LH_AutoSave"
        
        full_path = os.path.join(self.output_dir, folder_path) if not os.path.isabs(folder_path) else folder_path
        os.makedirs(full_path, exist_ok=True)
        
        timestamp = int(time.time())
        results = [] # 这里开始修复
        
        for i, image in enumerate(images):
            img = Image.fromarray((255. * image.cpu().numpy()).clip(0, 255).astype(np.uint8))
            file_basename = f"{safe_name}_{timestamp}_{i:02d}"
            save_filename = f"{file_basename}.png"
            
            metadata = PngInfo()
            if save_workflow:
                if prompt is not None: metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for k, v in extra_pnginfo.items(): metadata.add_text(k, json.dumps(v))
            
            img.save(os.path.join(full_path, save_filename), pnginfo=metadata)
            
            # --- 关键修复：这就是解决资产栏名字的那几行 ---
            results.append({
                "filename": save_filename,
                "subfolder": folder_path,
                "type": "output"
            })
            # ------------------------------------------

            if i == 0:
                with open(os.path.join(full_path, f"{safe_name}_{timestamp}.txt"), "w", encoding="utf-8") as f:
                    f.write(f"{trigger_word}, {lora_tags}".strip(", "))
                with open(os.path.join(full_path, f"{safe_name}_{timestamp}_log.txt"), "w", encoding="utf-8") as f:
                    f.write(str(gen_prompt))
                    
        return {"ui": {"images": results}} # 这里也从 [] 修复为了 results
