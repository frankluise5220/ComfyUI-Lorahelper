import os
import time
import re
import json
import random
try:
    import numpy as np
except ImportError:
    np = None
try:
    import torch
except ImportError:
    torch = None
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
from datetime import datetime

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
                "folder_path": ("STRING", {"default": "output"}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "save_workflow": ("BOOLEAN", {"default": True}),
                "save_method": (["timestamp", "sequential", "时间戳", "序号"], {"default": "timestamp"}),
            },
            "optional": {
                "images": ("IMAGE", ),
                "text1": ("STRING", {"forceInput": True}),
                "text2": ("STRING", {"forceInput": True}),
                "filename_final": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "LoraHelper"

    def save(self, folder_path, filename_prefix, save_workflow, save_method, images=None, text1=None, text2=None, filename_final=None, prompt=None, extra_pnginfo=None):
        
        # [Localization Support] Normalize save_method
        if save_method == "时间戳":
            save_method = "timestamp"
        elif save_method == "序号":
            save_method = "sequential"

        # 0. Path Security Check & ComfyUI Standard Path Resolution
        if folder_paths and hasattr(folder_paths, 'get_output_directory'):
            self.output_dir = folder_paths.get_output_directory()
        else:
            self.output_dir = "output"
        
        # Construct the prefix properly
        if folder_path and folder_path.strip():
             full_prefix_arg = os.path.join(folder_path, filename_prefix)
        else:
             full_prefix_arg = filename_prefix
        
        # Determine image dimensions for counter logic
        width = 0
        height = 0
        if images is not None and len(images) > 0:
             width = images[0].shape[1]
             height = images[0].shape[0]

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(full_prefix_arg, self.output_dir, width, height)

        # 1. Prepare Content (Caption)
        caption_content = ""
        if text2 is not None:
             clean_tags = text2.replace("\n", ", ").replace("  ", " ")
             caption_content = clean_tags.strip().strip(",")

        results = []
        
        # 2. Pre-Loop: Determine Base Filename and Suffix (Timestamp or Sequential)
        # We calculate the suffix ONCE for the entire batch to keep them grouped.
        
        # A. Base Filename Construction
        base_file_parts = [filename] # This is the prefix part returned by get_save_image_path
        
        # Add filename_final (custom part) if exists
        if filename_final:
            # 1. Basic Sanitization
            cleaned_name = re.sub(r'[<>:"/\\|?*\n\r\t]', "", filename_final).strip()
            # 2. Length Check
            if len(cleaned_name) > 100:
                print(f"\033[33m[LoraHelper Saver] Warning: Filename too long. Truncated.\033[0m")
                cleaned_name = cleaned_name[:100]
            # 3. Final Cleanup
            cleaned_name = cleaned_name.strip(". ")
            if not cleaned_name and filename_final.strip():
                cleaned_name = "INVALID_NAME"
            # Remove extension
            name_no_ext, ext = os.path.splitext(cleaned_name)
            if ext.lower() in ['.png', '.jpg', '.jpeg', '.webp', '.txt']:
                cleaned_name = name_no_ext
            
            if cleaned_name:
                base_file_parts.append(cleaned_name)

        # B. Suffix Generation (Timestamp or Sequential)
        common_suffix = ""
        if "timestamp" in save_method:
            common_suffix = str(int(time.time()))
        else:
            # Sequential Mode: Find next available index
            # Logic: Scan directory for files matching base_fname_XXXXX(_batch).ext
            base_fname_str = "_".join(base_file_parts)
            max_idx = 0
            
            # Regex to match: base_fname_XXXXX(_Y).ext
            # We match 5 digits at the end of the main part
            # Example: Anran_00001.png or Anran_00001_0.png
            # Pattern: ^base_fname_(\d{5})(?:_\d+)?\.(?:png|txt|json)$
            try:
                # Optimized Regex: Matches base_fname_ followed by digits, optionally followed by _anything (batch), ending with extension
                pattern = re.compile(r"^" + re.escape(base_fname_str) + r"_(\d{5})(?:_.*)?\.(?:png|txt|json|jpeg|jpg|webp)$", re.IGNORECASE)
                
                if os.path.exists(full_output_folder):
                    for f in os.listdir(full_output_folder):
                        if f.startswith(base_fname_str):
                            match = pattern.match(f)
                            if match:
                                try:
                                    idx = int(match.group(1))
                                    if idx > max_idx:
                                        max_idx = idx
                                except:
                                    pass
            except Exception as e:
                print(f"[LoraHelper Saver] Error scanning directory: {e}")
            
            next_idx = max_idx + 1
            common_suffix = f"{next_idx:05d}"

        # 3. Iterate Logic
        loop_count = 1
        if images is not None and len(images) > 0:
            loop_count = len(images)
            
        for i in range(loop_count):
            image = None
            if images is not None and len(images) > i:
                image = images[i]
            
            # Construct Final Filename
            file_parts = base_file_parts.copy()
            file_parts.append(common_suffix)
            
            fname = "_".join(file_parts)

            # Handle batch index ONLY if we have multiple images
            if images is not None and len(images) > 1:
                fname += f"_{i}"
            
            # Save Image
            if image is not None:
                img_tensor = image
                i_np = 255. * img_tensor.cpu().numpy()
                img = Image.fromarray(np.clip(i_np, 0, 255).astype(np.uint8))
                
                metadata = None
                if save_workflow:
                    metadata = PngInfo()
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        metadata.add_text("workflow", json.dumps(extra_pnginfo))

                img_path = os.path.join(full_output_folder, f"{fname}.png")
                img.save(img_path, pnginfo=metadata, compress_level=4)
                
                results.append({
                    "filename": f"{fname}.png",
                    "subfolder": subfolder,
                    "type": "output"
                })
            else:
                pass

            # Save Caption
            if text2 is not None:
                txt_path = os.path.join(full_output_folder, f"{fname}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption_content)
            
            # Save Log
            if text1 is not None:
                log_path = os.path.join(full_output_folder, f"{fname}_log.txt")
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(f"{text1}")
            
        return {"ui": {"images": results}}



class LH_AutoRatio:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "max_edge": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 32, "label": "最大边长"}),
                "default_ratio": (["16:9", "1:1", "9:16", "3:2", "2:3"], {"default": "16:9", "label": "默认比例(无图时)"}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT")
    RETURN_NAMES = ("width", "height", "raw_ratio")
    FUNCTION = "calculate"
    CATEGORY = "LoraHelper"

    def calculate(self, max_edge, default_ratio, image=None):
        # 1. Default Fallback Logic (Initialize first)
        ratio_map = {
            "16:9": 16/9,
            "9:16": 9/16,
            "1:1": 1.0,
            "3:2": 3/2,
            "2:3": 2/3
        }
        target_ratio = ratio_map.get(default_ratio, 1.0)
        raw_ratio = target_ratio

        # 2. Try to use Image if available
        if image is not None:
            try:
                # Verify shape validity [B, H, W, C]
                if hasattr(image, "shape") and len(image.shape) >= 3:
                    # Handle both [B,H,W,C] and [H,W,C] just in case, though Comfy is usually [B,H,W,C]
                    if len(image.shape) == 4:
                        _, h, w, _ = image.shape
                    else:
                        h, w, _ = image.shape
                        
                    if h > 0 and w > 0:
                        raw_ratio = w / h
                        
                        # Apply smart matching logic
                        if raw_ratio > 1.635:   target_ratio = 16/9
                        elif raw_ratio > 1.25:  target_ratio = 3/2
                        elif raw_ratio > 0.833: target_ratio = 1.0
                        elif raw_ratio > 0.614: target_ratio = 2/3
                        else:                   target_ratio = 9/16
            except Exception as e:
                print(f"LH_AutoRatio Warning: Failed to process image, using default ratio. Error: {e}")

        # 3. 根据 target_ratio 和 max_edge 计算最终宽高
        # 逻辑：长边 = max_edge
        if target_ratio >= 1.0: # 横图或方图：宽是长边
            width = max_edge
            height = int(max_edge / target_ratio)
        else: # 竖图：高是长边
            height = max_edge
            width = int(max_edge * target_ratio)

        # 强迫症对齐：确保结果是 8 的倍数（防止 VAE 解码模糊）
        width = (width // 8) * 8
        height = (height // 8) * 8

        # print(f"📏 LH_AutoRatio: 原始比例 {raw_ratio:.2f} -> 归一化输出 {width}x{height}")
        return (width, height, raw_ratio)

# Helper function to process dynamic prompts (wildcards)
def process_dynamic_prompts(text, seed=None):
    if not text:
        return ""
    
    # Clean Zero Width Space (ZWSP) which often causes regex failure
    text = text.replace("\u200b", "")

    # Handle seed=-1 (Random) explicitly
    if seed == -1:
        seed = None

    rng = random.Random(seed) if seed is not None else random
    
    # Simple recursive dynamic prompt processor
    # 1. {a|b|c} with support for weights {2::a|1::b}
    # 2. __wildcard__ (search in multiple directories)
    
    # Limit recursion depth
    MAX_DEPTH = 5
    
    # Prepare wildcard search paths
    # Priority: 
    # 1. ComfyUI-Lorahelper/wildcards (Local)
    # 2. ComfyUI/wildcards (Global)
    # 3. ComfyUI/custom_nodes/comfyui-dynamicprompts/wildcards (DynamicPrompts)
    search_dirs = []
    
    # 1. Local
    base_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs.append(os.path.join(base_dir, "wildcards"))
    
    # 2. Global (Standard ComfyUI Path)
    try:
        if hasattr(folder_paths, "base_path"):
             search_dirs.append(os.path.join(folder_paths.base_path, "wildcards"))
             search_dirs.append(os.path.join(folder_paths.base_path, "custom_nodes", "comfyui-dynamicprompts", "wildcards"))
             search_dirs.append(os.path.join(folder_paths.base_path, "custom_nodes", "ComfyUI-DynamicPrompts", "wildcards"))
    except:
        pass
        
    # 3. Global (Relative Path Fallback)
    # In case folder_paths.base_path fails or is not set
    try:
        comfy_root_guess = os.path.abspath(os.path.join(base_dir, "..", ".."))
        wildcards_guess = os.path.join(comfy_root_guess, "wildcards")
        if os.path.exists(wildcards_guess) and wildcards_guess not in search_dirs:
             search_dirs.append(wildcards_guess)
    except:
        pass

    # [DEBUG LOG] Print search directories to console
    # print(f"\033[33m[LH_Utils] Wildcard Search Dirs: {search_dirs}\033[0m")


    def process(current_text, depth):
        if depth > MAX_DEPTH:
            return current_text
            
        # 1. Handle {option1|option2} with weights
        while True:
            # Find innermost braces \{[^{}]*\}
            match = re.search(r"\{([^{}]+)\}", current_text)
            if not match:
                break
            
            full_match = match.group(0)
            options = match.group(1).split("|")
            
            choices = []
            weights = []
            
            for opt in options:
                opt = opt.strip()
                # Support both English "::" and Chinese "：" (just in case)
                if "::" in opt:
                    sep = "::"
                elif "：" in opt: # Handle potential Chinese colon typo
                    sep = "："
                else:
                    sep = None

                if sep:
                    try:
                        w_str, content = opt.split(sep, 1)
                        w = float(w_str)
                        choices.append(content)
                        weights.append(w)
                    except:
                        # Fallback if parsing fails
                        choices.append(opt)
                        weights.append(1.0)
                else:
                    choices.append(opt)
                    weights.append(1.0)
            
            if choices:
                # Weighted random choice
                # rng.choices returns a list, we take the first element
                choice = rng.choices(choices, weights=weights, k=1)[0].strip()
                current_text = current_text.replace(full_match, choice, 1)
            else:
                # Empty braces {} -> remove
                current_text = current_text.replace(full_match, "", 1)
            
        # 2. Handle __wildcard__
        wildcard_matches = list(re.finditer(r"__([\w\-\./\\]+)__", current_text))
        if wildcard_matches:
             # Process from end to start to maintain indices
             for m in reversed(wildcard_matches):
                 w_name = m.group(1)
                 replacement = m.group(0)
                 found = False
                 
                 # Search in all directories
                 for w_dir in search_dirs:
                     if not os.path.exists(w_dir):
                         continue
                         
                     # Support subdirectories in wildcards e.g. __folder/name__
                     # Check if w_name contains path separators? 
                     # The regex allows / and \ now.
                     
                     w_file = os.path.join(w_dir, f"{w_name}.txt")
                     if os.path.exists(w_file):
                         try:
                             with open(w_file, "r", encoding="utf-8") as f:
                                 # Ignore empty lines and comments starting with #
                                 lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
                                 if lines:
                                     replacement = rng.choice(lines)
                                     found = True
                         except:
                             pass
                     
                     if found:
                         break
                 
                 # Apply replacement (if found, it's the random line; if not, it's the original __name__)
                 # Wait, if we want to recurse, we should probably process the replacement too?
                 # The outer loop calls process(text, 0) which handles recursion via the return call?
                 # No, the recursion is: return process(text, 0) - wait, that's infinite loop if not careful?
                 # Ah, the original code was: return process(text, 0) at the end?
                 # No, the original code returned current_text at the end of function, and then outside called process(text,0) ONCE?
                 # Wait, looking at original code:
                 # line 290: return process(text, 0)
                 # inside process:
                 #   ... handles braces ...
                 #   ... handles wildcards ...
                 #   return current_text (NO RECURSION in the function body!)
                 
                 # To support recursive wildcards (wildcards inside wildcards), we need to recurse.
                 # If replacement contains { } or __ __, we need to process it.
                 # But we must decrement depth.
                 
                 current_text = current_text[:m.start()] + replacement + current_text[m.end():]
        
        # Check if we need another pass (recursion)
        # Only recurse if we actually changed something AND we see more braces/wildcards
        if (re.search(r"\{[^{}]+\}", current_text) or re.search(r"__([\w\-\./\\]+)__", current_text)) and depth < MAX_DEPTH:
             # Prevent infinite loops by only recursing if depth allows
             # Note: logic here is slightly different from typical recursion. 
             # Usually you process the *replacement* recursively.
             # Here we process the *whole string* again.
             return process(current_text, depth + 1)

        return current_text

    return process(text, 0)
