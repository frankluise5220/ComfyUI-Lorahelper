import re
import random
import json
import time  # Import time for IS_CHANGED
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
                # 'text' is the ONLY input port (The Dot).
                # 'showtext' is the display widget (The Box) and should NOT be connectable.
                
                # Input Dot (Top connection point)
                "text": ("STRING", {"forceInput": True, "default": ""}), 

                # Display Widget (Text Box, not connectable)
                "showtext": ("STRING", {"multiline": True, "default": "", "forceInput": False}),
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

    def process(self, text="", showtext="", seed=None):
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
                 try:
                     text_to_process = json.dumps(text, indent=4, ensure_ascii=False)
                 except:
                     text_to_process = str(text)
            else:
                text_to_process = str(text)

        # Apply Dynamic Prompts processing
        # Auto-generate seed internally since widget is hidden
        if seed is None:
            seed = random.randint(0, 0xffffffffffffffff)
        
        final_text = process_dynamic_prompts(text_to_process, seed)

        # Update showtext widget in frontend with the RAW text (before dynamic prompts?) 
        # Or processed? Usually user wants to see what's being processed.
        # But if it's dynamic, maybe show raw? 
        # Let's show the raw input text so user knows what came in.
        return {"ui": {"showtext": [text_to_process]}, "result": (final_text,)}

# Global cache for LH_MultiTextSelector to persist history across executions
_MULTI_TEXT_HISTORY = {}

class LH_MultiTextSelector:
    def __init__(self):
        self.index = 0
        self._spintax_pattern = re.compile(r"\{([^{}]+)\}")
        # self.history_texts is now managed via _MULTI_TEXT_HISTORY using unique_id

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
                "clear_history": ("BOOLEAN", {"default": False, "tooltip": "是否在每次运行时清空历史记录"}),
            },
            "optional": {
                "batch_text": ("STRING", {"forceInput": True, "tooltip": "连接此处以将文本'推送'到列表末尾"}),
                "widget_text": ("STRING", {"default": "", "multiline": True}),
                "showtext": ("STRING", {"default": "", "multiline": True, "forceInput": False}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "tooltip": "随机种子 (用于控制Wildcards选择)"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (False,)
    OUTPUT_NODE = True
    FUNCTION = "select"
    CATEGORY = "LoraHelper"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        # Use timestamp to ensure execution on every run, avoiding cache issues
        return float(time.time())

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

    def select(self, mode, clear_history, unique_id=None, batch_text=None, widget_text=None, showtext=None, seed=-1):
        global _MULTI_TEXT_HISTORY
        
        unique_id = unique_id if unique_id is not None else "default"
        
        # Initialize history
        if unique_id not in _MULTI_TEXT_HISTORY or clear_history:
             if clear_history:
                 print(f"[LoraHelper] Node {unique_id}: History cleared.")
             _MULTI_TEXT_HISTORY[unique_id] = []

        # 1. Collect NEW inputs (Push to Stack)
        if batch_text:
            new_items = []
            if isinstance(batch_text, list):
                new_items = [str(item) for item in batch_text]
            else:
                # Handle string input (split lines)
                s_text = str(batch_text).strip()
                if s_text:
                    new_items = [line.strip() for line in s_text.split('\n') if line.strip()]
            
            if new_items:
                _MULTI_TEXT_HISTORY[unique_id].extend(new_items)
                print(f"[LoraHelper] Node {unique_id}: Pushed {len(new_items)} items. Total: {len(_MULTI_TEXT_HISTORY[unique_id])}")

        # 2. Get Current Stack
        items = [x for x in _MULTI_TEXT_HISTORY[unique_id] if x.strip()]
        full_text_display = "\n".join(items)

        if not items:
            return {"ui": {"widget_text": [""], "showtext": [""]}, "result": ("",)}

        # 3. Mode Selection
        try:
            seed_int = int(seed)
        except (ValueError, TypeError):
            seed_int = random.randint(0, 0xffffffffffffffff)

        final_list = []
        if mode == "Random":
            rng = random.Random(seed_int) if seed_int != -1 else random.Random()
            final_list = [rng.choice(items)]
        else:
            # Sequential (All items)
            final_list = items

        # 4. Process Content (Wildcards & Spintax)
        processed_list = []
        dp_seed = seed_int if seed_int != -1 else random.randint(0, 0xffffffffffffffff)
        
        for item in final_list:
            item = process_dynamic_prompts(item, dp_seed)
            item = self._apply_spintax(item)
            processed_list.append(item)

        # 5. Output
        final_output_string = "\n".join(processed_list)
        
        return {"ui": {"widget_text": [full_text_display], "showtext": [full_text_display]}, "result": (final_output_string,)}
