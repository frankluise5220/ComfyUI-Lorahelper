import os
import time
import re
import json
import random
try:
    import numpy as np
except ImportError:
    np = None
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
        # 使用 os.path.commonpath 进行严格的路径遍历检测 (Refined Logic)
        
        # 1. 获取规范化的根目录
        base_output_dir = os.path.abspath(self.output_dir)
        
        # 处理用户输入的 folder_path
        if not folder_path:
            folder_path = "LoRA_Train_Data"
        folder_path = folder_path.strip()
        
        # 简单清理非法字符 (Windows不支持的字符)，但保留路径分隔符 / 和 \ 以支持子目录
        folder_path = re.sub(r'[*?"<>|:]', '', folder_path)
        
        # 2. 计算并规范化目标路径
        # 无论用户输入是相对还是绝对，都先合并。
        # 注意：如果 folder_path 是绝对路径，os.path.join 会丢弃 base_output_dir，
        # 但这没关系，因为下面的 commonpath 检查会拦截这种情况。
        requested_path = os.path.join(base_output_dir, folder_path)
        normalized_dest = os.path.abspath(os.path.normpath(requested_path))
        
        # 3. 核心安全检查：验证最终路径是否在 base_output_dir 之内
        try:
            if os.path.commonpath([base_output_dir, normalized_dest]) != base_output_dir:
                print(f"\033[31m[LoRA_Saver] Security Alert: Access to path '{folder_path}' rejected (Out of bounds). Fallback to default.\033[0m")
                # 如果越界，强制回退到默认安全目录
                full_path = os.path.join(base_output_dir, "LoRA_Train_Data")
                folder_path = "LoRA_Train_Data"
            else:
                full_path = normalized_dest
                # 更新 folder_path 为相对路径，确保 metadata 记录整洁
                folder_path = os.path.relpath(full_path, base_output_dir)
        except Exception:
             # 发生任何路径解析错误，回退到安全目录
             full_path = os.path.join(base_output_dir, "LoRA_Train_Data")
             folder_path = "LoRA_Train_Data"

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
            if np:
                img = Image.fromarray((255. * image.cpu().numpy()).clip(0, 255).astype(np.uint8))
            else:
                 # Fallback if numpy is missing (though torch usually requires numpy, this is safe)
                 import torch
                 img = Image.fromarray((255. * image.cpu()).clip(0, 255).to(torch.uint8).numpy())

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

# ==========================================================
# Helper: Dynamic Prompts Processor
# ==========================================================
def _get_wildcard_content(name, wildcard_dirs, rng):
    # Security: Prevent path traversal
    if any(x in name for x in ["..", ":"]) or name.startswith(("/", "\\")): return None
    
    # Allow alphanumeric, underscore, hyphen, space, slash, backslash, dot
    safe_name = re.sub(r'[^\w\-\s/.\\]', '', name)
    
    for wd in wildcard_dirs:
        p = os.path.normpath(os.path.join(wd, f"{safe_name}.txt"))
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    # Allow empty lines (empty choices) but skip comments
                    lines = [line.strip() for line in f if not line.strip().startswith("#")]
                if lines: return rng.choice(lines)
            except Exception: pass
    return None

def process_dynamic_prompts(text, seed=None, process_random=True):
    """
    Process Dynamic Prompts syntax:
    1. Wildcards: __name__ -> reads from wildcards/name.txt
    2. Inline Random: {a|b|c} -> random choice (Optional)
    """
    if not text: return ""

    # Global cleanup of invisible characters that might cause regex failure
    # \u200b: Zero-width space, \uFEFF: BOM, \u200c: ZWNJ, \u200d: ZWJ, \u2060: Word Joiner
    for char in ['\u200b', '\uFEFF', '\u200c', '\u200d', '\u2060']:
        text = text.replace(char, '')
    
    rng = random.Random(seed) if seed is not None and seed != -1 else random.Random()
    
    # Define Wildcard Search Paths
    base_path = folder_paths.base_path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    wildcard_dirs = [
        os.path.join(base_path, "wildcards"), 
        os.path.join(current_dir, "wildcards"), 
        os.path.join(base_path, "custom_nodes", "ComfyUI-DynamicPrompts", "wildcards")
    ]
    
    # 1. Wildcards Processing (__name__)
    for _ in range(10): # Max recursive depth
        def _repl_wildcard(m):
            content = _get_wildcard_content(m.group(1), wildcard_dirs, rng)
            return content if content is not None else m.group(0)
            
        # Support Unicode characters (including Chinese) by using \w
        new_text = re.sub(r"__([\w\-\s/.\\]+)__", _repl_wildcard, text)
        if new_text == text: break
        text = new_text

    # 2. Inline Random Processing ({a|b|c})
    if process_random:
        # Loop to handle nested brackets (innermost first)
        # Using a loop allows handling structures like {A|{B|C}}
        for _ in range(20): # Increased depth limit for complex nesting
            found_match = False
            
            def _repl_inline(m):
                nonlocal found_match
                found_match = True
                content = m.group(1).strip()
                
                # Use simple split since we process innermost first, meaning no nested braces should exist here.
                # If they do, it's malformed input, and smart split won't save it logically anyway.
                opts = content.split("|")
                choices, weights = [], []
                
                for opt in opts:
                    opt = opt.strip()
                    # Allow empty options (e.g. {A|}) -> empty string choice
                    if not opt and len(opts) > 1:
                        weights.append(1.0)
                        choices.append("")
                        continue
                    if not opt: continue
                    
                    # Regex: Allow leading whitespace, Capture Weight, Match ::, Capture Content
                    match = re.match(r'^\s*(\d+(?:\.\d+)?)\s*::\s*(.*)$', opt, re.DOTALL)
                    
                    parsed = False
                    if match:
                        try:
                            w_val = float(match.group(1))
                            c_val = match.group(2).strip()
                            weights.append(w_val)
                            choices.append(c_val)
                            parsed = True
                        except ValueError: pass
                    
                    if not parsed:
                        # Fallback: Check if :: exists and try manual parsing for robustness
                        # This handles cases where regex fails due to weird invisible chars or format issues
                        if "::" in opt:
                            try:
                                parts = opt.split("::", 1)
                                w_candidate = parts[0].strip()
                                # Allow simple number check
                                if re.match(r'^\d+(\.\d+)?$', w_candidate):
                                    w_val = float(w_candidate)
                                    c_val = parts[1].strip()
                                    weights.append(w_val)
                                    choices.append(c_val)
                                    parsed = True
                            except: pass
                    
                    if not parsed:
                        weights.append(1.0)
                        choices.append(opt)
                
                if not choices: return ""
                
                # Weighted Random Selection
                if sum(weights) <= 0: return rng.choice(choices)
                return rng.choices(choices, weights=weights, k=1)[0]

            # Regex to find innermost braces: { followed by anything not { or } followed by }
            # [^{}] matches newlines automatically, so DOTALL is not strictly needed for the outer find,
            # but we use it for safety.
            new_text = re.sub(r"\{([^{}]+)\}", _repl_inline, text)
            
            if new_text == text:
                break
            text = new_text
            
    return text

# ==========================================================
# New: Simple Text Node (Raw String)
# ==========================================================
class LH_SimpleText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "force_input": ("STRING", {"forceInput": True, "tooltip": "Optional input to override text"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "custom_nodes/MyLoraNodes"
    OUTPUT_NODE = True

    def execute(self, text, force_input=None):
        # Prioritize connected input if available
        final_text = force_input if force_input is not None else text
        
        # Return both the string for connection and the UI update
        return {"ui": {"text": [final_text]}, "result": (final_text,)}
