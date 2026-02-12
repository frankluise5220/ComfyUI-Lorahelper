import re
import random
from .LH_Utils import process_dynamic_prompts

class LH_SuperText:
    """
    Finalized: 2026-02-02
    DO NOT MODIFY unless absolutely necessary.
    This node serves as both an input and an output (display) node.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "showtext": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "force_text": ("STRING", {"forceInput": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "tooltip": "Seed for Wildcards"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_NODE = True
    FUNCTION = "process"
    CATEGORY = "LoraHelper"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def process(self, showtext, force_text=None, seed=-1):
        # Prioritize force_text if connected and valid
        text_to_process = showtext
        if force_text is not None and isinstance(force_text, str) and force_text.strip() != "":
            text_to_process = force_text

        # Apply Dynamic Prompts processing
        # Note: If seed is -1, process_dynamic_prompts treats it as None (fully random) if not handled, 
        # but let's check LH_Utils implementation.
        # LH_Utils: rng = random.Random(seed) if seed is not None else random
        # So passing -1 will fix the seed to -1 (which is valid but static).
        # We usually want fully random if seed is not connected or set to control widget default?
        # Standard ComfyUI "control_after_generate" handles the seed change.
        
        final_text = process_dynamic_prompts(text_to_process, seed)
        # Only return the processed text as result, do NOT update the UI widget
        # We keep showtext in UI to avoid confusion
        return {"ui": {"text": [showtext]}, "result": (final_text,)}

class LH_MultiTextSelector:
    def __init__(self):
        self.index = 0
        self._spintax_pattern = re.compile(r"\{([^{}]+)\}")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (
                    ["Sequential", "Random"],
                    {
                        "tooltip": "多文本选择模式：Sequential=按顺序批量运行；Random=每次随机选择一行",
                    },
                ),
            },
            "optional": {
                "batch_text": ("STRING", {"forceInput": True}),
                "widget_text": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "tooltip": "随机种子 (用于控制Wildcards选择)"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "select"
    CATEGORY = "LoraHelper"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def _apply_spintax(self, text):
        if not isinstance(text, str):
            return text

        def repl(match):
            raw = match.group(1)
            tokens = [p for p in raw.split("|") if p]
            if not tokens:
                return ""

            weighted = []
            total = 0.0
            for token in tokens:
                value = token
                weight = 1.0
                if "::" in token:
                    w_str, val = token.split("::", 1)
                    w_str = w_str.strip()
                    value = val
                    try:
                        weight = float(w_str)
                    except Exception:
                        weight = 1.0
                value = value
                if weight <= 0:
                    continue
                weighted.append((value, weight))
                total += weight

            if not weighted:
                return ""

            r = random.random() * total
            acc = 0.0
            for val, w in weighted:
                acc += w
                if r <= acc:
                    return val
            return weighted[-1][0]

        prev = None
        while prev != text and self._spintax_pattern.search(text):
            prev = text
            text = self._spintax_pattern.sub(repl, text)
        return text

    def select(self, mode, batch_text=None, widget_text=None, seed=-1):
        # 1. Determine source
        raw_text = ""
        if batch_text is not None:
            raw_text = "\n".join(batch_text) if isinstance(batch_text, list) else str(batch_text)
        elif widget_text is not None:
            raw_text = widget_text
            
        items = [line.strip() for line in raw_text.split('\n') if line.strip()]
        
        if not items:
            return ([""],)
            
        final_list = []
        
        if mode == "Random":
            # Random Mode: Return 1 random item (List of 1)
            rng = random.Random(seed) if seed != -1 else random.Random()
            chosen = rng.choice(items)
            final_list = [chosen]
        else:
            # Sequential Mode: Return ALL items (List of N)
            # This triggers ComfyUI batch processing (one run per item)
            final_list = items

        # Process each item in the list
        processed_list = []
        for item in final_list:
            # 1. Process Wildcards (Dynamic Prompts)
            item = process_dynamic_prompts(item, seed)
            # 2. Process Spintax (Inline Random with weights)
            item = self._apply_spintax(item)
            processed_list.append(item)
        
        return (processed_list,)
