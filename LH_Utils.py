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
    CATEGORY = "LoraHelper"

    def save(self, images, folder_path, filename_prefix, trigger_word, save_workflow, gen_prompt=None, lora_tags=None, filename_final=None, prompt=None, extra_pnginfo=None):
        
        # 0. Path Security Check & ComfyUI Standard Path Resolution
        # We use ComfyUI's standard method to handle %date% and auto-increment counters correctly.
        
        # If user provided a specific folder_path in the widget, we should try to respect it
        # BUT ComfyUI's get_save_image_path uses 'output_dir' as base. 
        # If we want a subfolder, we should prepend it to filename_prefix or handle it manually.
        
        # Let's align with the user's existing logic but use standard numbering.
        # Existing logic: base_output_dir/folder_path/filename_prefix...
        
        # Construct the prefix properly:
        if folder_path and folder_path.strip():
             # Combine folder_path and filename_prefix for the standard function
             # e.g. "LoRA_Train/Anran"
             full_prefix_arg = os.path.join(folder_path, filename_prefix)
        else:
             full_prefix_arg = filename_prefix
             
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(full_prefix_arg, self.output_dir, images[0].shape[1], images[0].shape[0])

        # 1. Prepare Content (Caption)
        if gen_prompt is None: gen_prompt = ""
        if lora_tags is None: lora_tags = ""
        
        # Process tags
        clean_tags = lora_tags.replace("\n", ", ").replace("  ", " ")
        if trigger_word and trigger_word.strip() != "":
            t_word = trigger_word.strip()
            if not clean_tags.strip().lower().startswith(t_word.lower()):
                 caption_content = f"{t_word}, {clean_tags}"
            else:
                 caption_content = clean_tags
        else:
            caption_content = clean_tags

        caption_content = caption_content.strip().strip(",")

        results = []
        # 2. Iterate Images
        for i, image in enumerate(images):
            # Determine Filename
            # User Requirement: Prefix + [FilenameFinal] + Timestamp + Batch
            
            file_parts = [filename] # This is the prefix part returned by get_save_image_path (e.g. "Anran")
            
            # Add filename_final (custom part) if exists
            if filename_final:
                # Relaxed sanitization: Allow brackets [] () {} but remove illegal Windows chars
                cleaned_name = re.sub(r'[<>:"/\\|?*]', "", filename_final).strip()
                # Remove extension if user typed it manually
                cleaned_name = os.path.splitext(cleaned_name)[0]
                if cleaned_name:
                    file_parts.append(cleaned_name)

            # Add Timestamp (Requested by user)
            timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
            file_parts.append(timestamp_str)
            
            # Add counter
            # We use the standard counter + loop index
            current_count = counter + i
            
            # Construct final filename
            # e.g. "Anran_[CustomName]_20250201_120000_00005"
            fname = "_".join(file_parts) + f"_{current_count:05}"
            
            # Save Image
            img_tensor = image
            i_np = 255. * img_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(i_np, 0, 255).astype(np.uint8))
            
            metadata = None
            if save_workflow:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            img_path = os.path.join(full_output_folder, f"{fname}.png")
            img.save(img_path, pnginfo=metadata, compress_level=4)

            # Save Caption
            txt_path = os.path.join(full_output_folder, f"{fname}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption_content)
            
            # Save Log (optional, keeping as per original)
            log_path = os.path.join(full_output_folder, f"{fname}_log.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"{gen_prompt}")

            results.append({
                "filename": f"{fname}.png",
                "subfolder": subfolder,
                "type": "output"
            })

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

    if seed is not None:
        random.seed(seed)
    
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
    
    # 2. Global
    try:
        if folder_paths.base_path:
             search_dirs.append(os.path.join(folder_paths.base_path, "wildcards"))
             # 3. DynamicPrompts (Standard install location)
             search_dirs.append(os.path.join(folder_paths.base_path, "custom_nodes", "comfyui-dynamicprompts", "wildcards"))
             search_dirs.append(os.path.join(folder_paths.base_path, "custom_nodes", "ComfyUI-DynamicPrompts", "wildcards"))
    except:
        pass

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
                # random.choices returns a list, we take the first element
                choice = random.choices(choices, weights=weights, k=1)[0].strip()
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
                                     replacement = random.choice(lines)
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
