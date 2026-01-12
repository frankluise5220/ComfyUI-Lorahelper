import os
import sys

def switch_language():
    current_lang = os.environ.get("COMFYUI_LORAHELPER_LANG", "en_US")
    print("----------------------------------------------------------------")
    print(f"Current Language Setting: {current_lang}")
    print("----------------------------------------------------------------")
    print("Please select language:")
    print("1. English (en_US)")
    print("2. Chinese (zh_CN)")
    print("----------------------------------------------------------------")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        new_lang = "en_US"
        print("Switched to English.")
    elif choice == "2":
        new_lang = "zh_CN"
        print("已切换至中文。")
    else:
        print("Invalid choice. No changes made.")
        return

    # In a real ComfyUI environment, changing os.environ here only affects the current process.
    # To make it persistent, users usually need to set it in their run.bat or system env.
    # However, since we are inside a custom node, we can't easily change the parent process env permanently.
    # A common workaround is to write a config file that __init__.py reads.
    
    config_path = os.path.join(os.path.dirname(__file__), "lang_config.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(new_lang)
    
    print(f"Language preference saved to {config_path}")
    print("Please restart ComfyUI for changes to take effect.")

if __name__ == "__main__":
    switch_language()
