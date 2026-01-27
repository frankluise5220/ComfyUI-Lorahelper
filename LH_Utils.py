import os
import time
import re
import json
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths

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
        # 0. 路径清理 (Sanitization) - 修复安全漏洞 (Security Fix)
        # 使用 os.path.commonpath 进行严格的路径遍历检测
        
        # 获取绝对路径的输出基准目录
        base_output_dir = os.path.abspath(self.output_dir)
        
        # 处理用户输入的 folder_path
        if not folder_path:
            folder_path = "LoRA_Train_Data"
        folder_path = folder_path.strip()
        
        # 简单清理非法字符 (Windows不支持的字符)，但保留路径分隔符 / 和 \ 以支持子目录
        # 仅移除 * ? " < > | : 
        folder_path = re.sub(r'[*?"<>|:]', '', folder_path)
        
        # 防止绝对路径被 os.path.join 处理（绝对路径会丢弃前面的基准路径）
        if os.path.isabs(folder_path):
             # 简单的处理方式：去掉盘符，去掉开头的斜杠，强制转为相对路径
             drive, tail = os.path.splitdrive(folder_path)
             folder_path = tail.lstrip(os.sep + '/')
        
        # 构造目标绝对路径
        # os.path.normpath 会处理 .. 和多余的斜杠
        target_path = os.path.abspath(os.path.join(base_output_dir, folder_path))
        
        # [SECURITY CHECK] 确保 target_path 是 base_output_dir 的子目录
        try:
            common = os.path.commonpath([base_output_dir, target_path])
        except ValueError:
            common = ""
            
        if common != base_output_dir:
            print(f"\033[31m[LoRA_Saver] Security Alert: Path traversal attempt detected! '{folder_path}' -> '{target_path}'. Fallback to default.\033[0m")
            full_path = os.path.join(base_output_dir, "LoRA_Train_Data")
            folder_path = "LoRA_Train_Data"
        else:
            full_path = target_path
            # 更新 folder_path 为相对路径，确保 metadata 记录整洁
            folder_path = os.path.relpath(full_path, base_output_dir)

        os.makedirs(full_path, exist_ok=True)
        
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
