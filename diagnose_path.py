
import os
import re
import random

# 模拟环境路径
base_dir = r"e:\ComfyUI_windows\ComfyUI\custom_nodes\ComfyUI-Lorahelper"
comfy_root_guess = os.path.abspath(os.path.join(base_dir, "..", ".."))
wildcards_guess = os.path.join(comfy_root_guess, "wildcards")

print(f"Base Dir: {base_dir}")
print(f"Comfy Root Guess: {comfy_root_guess}")
print(f"Wildcards Guess: {wildcards_guess}")
print(f"Wildcards Exists: {os.path.exists(wildcards_guess)}")

search_dirs = [wildcards_guess]

# 模拟查找文件
w_name = "anran_scene_home"
found_file = False

for w_dir in search_dirs:
    w_file = os.path.join(w_dir, f"{w_name}.txt")
    print(f"Checking: {w_file}")
    if os.path.exists(w_file):
        print(f"  -> FILE FOUND!")
        found_file = True
        try:
            with open(w_file, "r", encoding="utf-8") as f:
                content = f.read()
                print(f"  -> Content Length: {len(content)}")
                print(f"  -> First 50 chars: {content[:50]}")
        except Exception as e:
            print(f"  -> Read Error: {e}")
    else:
        print(f"  -> File not found")

# 模拟正则匹配
text = "__anran_scene_home__"
matches = list(re.finditer(r"__([\w\-\./\\]+)__", text))
print(f"Regex Matches: {len(matches)}")
for m in matches:
    print(f"  Group 1: {m.group(1)}")
