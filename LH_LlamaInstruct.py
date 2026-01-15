import folder_paths

class LH_LlamaInstruct:
    @classmethod
    def INPUT_TYPES(s):
        return { 
            "required": { 
                "model": ("LLM_MODEL",),
                "user_material": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Deprecated Node"}),
                "instruction": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Deprecated Node"}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "tags", "filename", "raw_output")
    FUNCTION = "chat"
    CATEGORY = "custom_nodes/MyLoraNodes/Legacy"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def chat(self, model, user_material, instruction, image=None, **kwargs):
        warning_msg = "⚠️ This node (LH_LlamaInstruct) is DEPRECATED. Please replace it with the new 'LH_AIChat' (UniversalAIChat) node which includes all features."
        print(f"\033[31m[ComfyUI-Lorahelper] {warning_msg}\033[0m")
        return (warning_msg, "", "", warning_msg)
