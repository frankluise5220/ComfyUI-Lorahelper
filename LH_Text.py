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
                 import json
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

class LH_MultiTextSelector:
    def __init__(self):
        self.index = 0
        self._spintax_pattern = re.compile(r"\{([^{}]+)\}")
        self.history_texts = []  # Store pushed texts

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

    def select(self, mode, clear_history, batch_text=None, widget_text=None, seed=-1):
        # Clear history if requested
        if clear_history:
            self.history_texts = []

        # 1. Collect inputs
        new_items = []

        # Process batch_text (multiline string or list)
        if batch_text is not None:
            if isinstance(batch_text, list):
                # If upstream sends a list, use it directly (flattening)
                for item in batch_text:
                     new_items.append(str(item))
            else:
                # If string, split by newline? Or treat as single item to push?
                # "Push" logic usually implies adding one item (or a batch) to the existing queue.
                # If it contains newlines, should we split? 
                # Let's assume standard behavior: split by lines if it looks like a batch, 
                # or treat as one if it's a single prompt.
                # To be safe and flexible: Split by lines.
                lines = [line.strip() for line in str(batch_text).split('\n') if line.strip()]
                new_items.extend(lines)

        # Process widget_text (multiline string) - always treated as base/static list
        static_items = []
        if widget_text is not None:
             lines = [line.strip() for line in widget_text.split('\n') if line.strip()]
             static_items.extend(lines)

        # Update history with new items (Push logic)
        # We only add to history if there are new items from batch_text
        if new_items:
            self.history_texts.extend(new_items)
            # Limit history to prevent infinite growth? Let's say 100 max for now, or keep all?
            # User said "push 6 times", so probably wants to accumulate.
            # Let's keep it unbounded for this session but provide clear_history to reset.
            
        # Combine static items (from widget) and history items (from push)
        # Priority: Widget text + History text
        combined_items = static_items + self.history_texts
        
        # Filter out empty strings
        items = [item for item in combined_items if item.strip()]
        
        # Remove duplicates? Maybe user wants duplicates. Let's keep them.
        
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
