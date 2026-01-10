import os
import time
import re
import numpy as np
from PIL import Image
import folder_paths

class Qwen3TextSplitter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"forceInput": True, "multiline": True}), "user_prefix": ("STRING", {"default": "Anran"})}}
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("gen_prompt", "lora_tags", "filename_final")
    FUNCTION = "split"
    CATEGORY = "custom_nodes/MyLoraNodes"

    def split(self, text, user_prefix):
        raw = str(text).strip() if text else ""
        p_match = re.search(r'SECTION 1.*?[:：]\s*(.*?)(?=SECTION 2|SECTION 3|\[|$)', raw, re.DOTALL | re.IGNORECASE)
        gen_p = p_match.group(1).strip() if p_match else raw.split('\n\n')[0].strip()
        
        tags_match = re.search(r'SECTION 2.*?[:：]\s*(.*?)(?=SECTION 3|\[|$)', raw, re.DOTALL | re.IGNORECASE)
        lora_tags = ""
        if tags_match:
            tags_src = re.sub(r':\s*\d+\.?\d*|[()\[\]{}]', '', tags_match.group(1))
            tokens = [t.strip() for t in re.split(r'[,\n;，；]', tags_src) if 2 < len(t.strip()) < 50]
            lora_tags = ", ".join(list(dict.fromkeys(tokens)))

        t_match = re.search(r'(?:SECTION 3.*?[:：]\s*)?\[([^\]]+)\]', raw, re.DOTALL | re.IGNORECASE)
        title = re.sub(r"[\[\]{}'\"`’]", '', t_match.group(1)).strip() if t_match else f"Auto_{int(time.time())}"
        return (gen_p, lora_tags, f"{user_prefix}_{title[:50]}")

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
                "trigger_word": ("STRING", {"default": "ChenAnran"}), 
            }
        }
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True  
    CATEGORY = "custom_nodes/MyLoraNodes"

    def save(self, images, gen_prompt, lora_tags, filename_final, folder_path, trigger_word):
        # 还原：文件名过滤与路径拼接
        s_file = str(filename_final)
        clean_name = re.sub(r"[\[\]{}'\"`’]", '', s_file)
        safe_name = re.sub(r'[\\/:*?"<>|\n\r\t]', '', clean_name).strip().replace(' ', '_')[:50]
        
        full_path = os.path.join(self.output_dir, folder_path) if not os.path.isabs(folder_path) else folder_path
        os.makedirs(full_path, exist_ok=True)
        
        timestamp = int(time.time())
        for i, image in enumerate(images):
            img = Image.fromarray((255. * image.cpu().numpy()).clip(0, 255).astype(np.uint8))
            base = f"{safe_name}_{timestamp}_{i:02d}"
            img.save(os.path.join(full_path, f"{base}.png"))
            if i == 0:
                with open(os.path.join(full_path, f"{safe_name}_{timestamp}.txt"), "w", encoding="utf-8") as f:
                    f.write(f"{trigger_word}, {lora_tags}".strip(", "))
                with open(os.path.join(full_path, f"{safe_name}_{timestamp}_log.txt"), "w", encoding="utf-8") as f:
                    f.write(str(gen_prompt))
        return {"ui": {"images": []}}