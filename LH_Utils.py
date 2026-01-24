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
        
        # --- 1. Generate Prompt (Description) ---
        # Supports both new (**description**) and legacy (SECTION 1) formats
        p_match = re.search(r'(?:\*\*description\*\*|SECTION 1).*?[:：]\s*(.*?)(?=\*\*tags\*\*|SECTION 2|SECTION 3|\*\*filename\*\*|\[|$)', raw, re.DOTALL | re.IGNORECASE)
        gen_p = p_match.group(1).strip() if p_match else raw.split('\n\n')[0].strip()
        
        # --- 2. Lora Tags ---
        # Supports both new (**tags**) and legacy (SECTION 2) formats
        tags_match = re.search(r'(?:\*\*tags\*\*|SECTION 2).*?[:：]\s*(.*?)(?=\*\*filename\*\*|SECTION 3|\s*\[|$)', raw, re.DOTALL | re.IGNORECASE)
        lora_tags = ""
        if tags_match:
            raw_tags = tags_match.group(1)
            # Cleanup: Ensure we didn't capture the next section header
            if "SECTION 3" in raw_tags.upper() or "**FILENAME**" in raw_tags.upper():
                raw_tags = re.split(r'SECTION 3|\*\*filename\*\*', raw_tags, flags=re.IGNORECASE)[0]
            
            tags_src = re.sub(r':\s*\d+\.?\d*|[()\[\]{}]', '', raw_tags)
            tokens = [t.strip() for t in re.split(r'[,\n;，；]', tags_src) if 2 < len(t.strip()) < 50]
            # Filter out keywords
            tokens = [t for t in tokens if "SECTION" not in t.upper() and "**" not in t]
            lora_tags = ", ".join(list(dict.fromkeys(tokens)))

        # --- 3. Filename ---
        # Supports [filename] pattern, optionally prefixed by headers
        t_match = re.search(r'(?:(?:\*\*filename\*\*|SECTION 3).*?[:：]\s*)?\[([^\]]+)\]', raw, re.DOTALL | re.IGNORECASE)
        
        title = ""
        if t_match:
             title = t_match.group(1)
        else:
             # Fallback: Try to catch filename without brackets if header exists
             f_match = re.search(r'(?:\*\*filename\*\*|SECTION 3).*?[:：]\s*([^\n]+)', raw, re.IGNORECASE)
             if f_match:
                 title = f_match.group(1)

        title = re.sub(r"[\[\]{}'\"`’]", '', title).strip()
        if not title: 
            title = f"Auto_{int(time.time())}"
            
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
                "folder_path": ("STRING", {"default": "LoRA_Train_Data"}),
                "filename_prefix": ("STRING", {"default": "Anran"}),
                "trigger_word": ("STRING", {"default": "ChenAnran"}), 
                "save_workflow": ("BOOLEAN", {"default": True}), # 功能 2：开关
            },
            "optional": {
                "gen_prompt": ("STRING", {"forceInput": True}),
                "lora_tags": ("STRING", {"forceInput": True}),
                "filename_final": ("STRING", {"forceInput": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}
        }
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True  
    CATEGORY = "custom_nodes/MyLoraNodes"

    def save(self, images, folder_path, filename_prefix, trigger_word, save_workflow, gen_prompt=None, lora_tags=None, filename_final=None, prompt=None, extra_pnginfo=None):
        # 0. 路径清理 (Sanitization) - 修复维护者提出的安全问题
        # 移除可能导致路径穿越的 ".." 字符，并清理前后空格
        folder_path = folder_path.strip().replace("..", "")
        # 移除非法字符，防止系统报错
        folder_path = re.sub(r'[\\:*?"<>|]', '', folder_path)
        
        # 如果清理后为空，给个默认名
        if not folder_path:
            folder_path = "LoRA_Train_Data"
        # Handle optional inputs being None
        # [CHANGE] We do NOT convert lora_tags/gen_prompt to "" here, because we need to detect None for file skipping.
        
        s_file = str(filename_final).strip() if filename_final else ""
        
        # 如果输入的文件名过长（说明可能是误连了 Prompt 或模型输出了句子），则丢弃它
        # 智能修正 1：太长/太多词，丢弃
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
        
        # [SECURITY FIX] 强制使用相对路径，禁止绝对路径，防止路径穿越
        # Force relative path logic to ensure we stay within ComfyUI output directory
        if os.path.isabs(folder_path):
            # 如果用户传了绝对路径，我们只取最后一部分作为子文件夹名
            # 或者直接忽略盘符，强制变为相对路径
            # 这里选择简单粗暴的方案：把冒号和斜杠都去掉，变成一个长文件夹名，确保安全
            print(f"\033[33m[LoRA_Saver] Warning: Absolute path detected. Converting '{folder_path}' to relative path for security.\033[0m")
            folder_path = re.sub(r'[:]', '', folder_path).lstrip('/\\')
            
        full_path = os.path.join(self.output_dir, folder_path)
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
                # [Fix] Only save text files if inputs are connected (not None)
                if lora_tags is not None:
                    with open(os.path.join(full_path, f"{safe_name}_{timestamp}.txt"), "w", encoding="utf-8") as f:
                        f.write(f"{trigger_word}, {lora_tags}".strip(", "))
                
                if gen_prompt is not None:
                    with open(os.path.join(full_path, f"{safe_name}_{timestamp}_log.txt"), "w", encoding="utf-8") as f:
                        f.write(str(gen_prompt))
                    
        return {"ui": {"images": results}} # 这里也从 [] 修复为了 results
