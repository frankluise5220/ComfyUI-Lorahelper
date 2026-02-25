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
                # Nothing required, allowing the node to be a pure generator/display node
            },
            "optional": {
                # 'text' is a standard Widget now (forceInput removed to avoid mandatory connection error)
                # But placed FIRST to capture auto-connect.
                "text": ("STRING", {"multiline": True, "forceInput": True, "default": ""}), 
                # Wait, if forceInput is True, it demands connection? 
                # Let's try forceInput: False or just omit it. 
                # If omitted, it's a widget. But we want it to be an input point primarily?
                # Actually, forceInput: True in optional SHOULD be fine. 
                # The user report "must be connected" suggests it's treated as required.
                # Let's try making it a simple STRING widget that auto-converts.
                # BUT auto-convert only happens if it's NOT forceInput.
                # If we remove forceInput, it becomes a text box.
                # Let's keep forceInput: True but ensure it's optional.
                # Maybe the issue was the lack of default value?
                # Let's add default value to text.
                "text": ("STRING", {"forceInput": True, "default": ""}), 

                "showtext": ("STRING", {"multiline": True, "default": "", "forceInput": False}),
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

    def process(self, text="", showtext="", seed=-1):
        if not text:
            # If text is empty (e.g. initial state), fallback to showtext (manual input)
            text_to_process = showtext
        else:
            # If text has input (from upstream), use it and update showtext
            if isinstance(text, str):
                text_to_process = text
            elif isinstance(text, (int, float, bool)):
                text_to_process = str(text)
            elif isinstance(text, (list, tuple)):
                # Join list items with newline
                text_to_process = "\n".join([str(item) for item in text])
            elif isinstance(text, dict):
                 import json
                 try:
                     text_to_process = json.dumps(text, indent=4, ensure_ascii=False)
                 except:
                     text_to_process = str(text)
            else:
                text_to_process = str(text)

        # Apply Dynamic Prompts processing
        # Use seed if provided, default to 0 if not
        eff_seed = seed if seed != -1 else 0
        final_text = process_dynamic_prompts(text_to_process, eff_seed)

        # Update showtext widget in frontend with the RAW text (before dynamic prompts?) 
        # Or processed? Usually user wants to see what's being processed.
        # But if it's dynamic, maybe show raw? 
        # Let's show the raw input text so user knows what came in.
        return {"ui": {"showtext": [text_to_process]}, "result": (final_text,)}

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
