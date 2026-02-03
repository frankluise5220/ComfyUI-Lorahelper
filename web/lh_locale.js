import { app } from "../../scripts/app.js";

const TRANSLATIONS = {
    "zh-CN": {
        "LoraHelper_Chat": {
            "title": "LH_AI对话助手",
            "model": "模型 (Model)",
            "image": "图像 (Image)",
            "max_tokens": "最大生成长度",
            "temperature": "温度 (Temperature)",
            "repetition_penalty": "重复惩罚",
            "instruction": "系统指令/执行指令 (Instruction)",
            "user_material": "用户素材 (User Material)",
            "chat_mode": "对话模式",
            "enable_tags": "启用标签提取",
            "enable_filename": "启用文件名生成",
            "seed": "种子 (Seed)",
            "release_vram": "自动释放显存",
            "force_chinese": "强制中文输出"
        },
        "Qwen3_GGUF_loader": {
            "title": "LH_GGUF模型加载器",
            "gguf_model": "GGUF模型",
            "clip_model": "CLIP视觉模型",
            "n_gpu_layers": "GPU层数 (-1为全部)",
            "n_ctx": "最大上下文 (n_ctx)"
        },
        "LoraHelper_OllamaLoader": {
            "title": "LH_Ollama加载器",
            "ollama_model": "Ollama模型",
            "ollama_url": "Ollama地址"
        },
        "LoraHelper_CloudLoader": {
            "title": "LH_云端模型加载器",
            "api_key": "API 密钥 (API Key)",
            "base_url": "API 地址 (Base URL)",
            "model_name": "模型名称 (Model Name)"
        },
        "LoraHelper_Monitor": {
            "title": "LH_对话历史监控",
            "raw_input": "原始输出 (Raw Input)",
            "clear_history": "清除历史记录"
        },
        "LoraHelper_MultiTextSelector": {
            "title": "LH_多路文本选择器",
            "mode": "选择模式 (顺序/随机)",
            "text_1": "文本 1",
            "text_2": "文本 2",
            "text_3": "文本 3",
            "text_4": "文本 4"
        },
        "LoraHelper_Splitter": {
            "title": "LH_提示词切分器",
            "text": "文本输入",
            "user_prefix": "用户前缀"
        },
        "LoraHelper_Saver": {
            "title": "LH_全功能保存器",
            "images": "图像",
            "gen_prompt": "生成提示词",
            "lora_tags": "LoRA标签",
            "filename_final": "最终文件名",
            "folder_path": "保存路径",
            "filename_prefix": "文件前缀",
            "trigger_word": "触发词",
            "save_workflow": "保存工作流"
        },
        "LoraHelper_SuperText": {
            "title": "LH_超级文本框",
            "showtext": "显示文本",
            "widget_text": "编辑文本"
        },
        "LoraHelper_LoraLoader": {
            "title": "LH_关键字Lora加载器",
            "model": "模型",
            "lora_name": "LoRA名称",
            "strength_model": "模型强度",
            "strength_clip": "CLIP强度",
            "prompt_in": "提示词输入",
            "prompt_out": "提示词输出",
            "trigger_keywords": "触发关键字",
            "clip": "CLIP"
        },
        "LoraHelper_LlamaInstruct": {
            "title": "LH_Llama指令助手 (旧版)",
            "model": "模型",
            "instruction": "指令"
        },
        "LoraHelper_AutoRatio": {
            "title": "LH_智能比例归一化",
            "image": "图像",
            "max_edge": "最大边长",
            "default_ratio": "默认比例(无图时)",
            "width": "宽",
            "height": "高",
            "raw_ratio": "原始比例"
        }
    }
};

const DEFAULT_LANG = "zh-CN";

// Helper to get global ComfyUI language
function getComfyLanguage() {
    return app.ui.settings.getSettingValue("Comfy.Language", "zh-CN");
}

app.registerExtension({
    name: "LoraHelper.Translation",
    async setup() {
        console.log("[LoraHelper] Translation extension loaded.");
        const settings = app.ui.settings;
        
        // 1. Fetch Config from Backend API
        let backendConfig = {};
        try {
            const resp = await fetch("/lorahelper/get_config");
            if (resp.ok) {
                backendConfig = await resp.json();
                console.log("[LoraHelper] Loaded backend config:", backendConfig);
            }
        } catch (e) {
            console.warn("[LoraHelper] Failed to fetch backend config:", e);
        }

        // 2. Determine Initial Language
        // Priority: User Setting > Backend Config > ComfyUI Global > Default "en-US"
        let defaultLang = getComfyLanguage();
        if (backendConfig.locale) {
            // If backend config has a locale set (e.g. "zh-CN"), use it as the preferred default
            // BUT, if user has manually set "LoraHelper.Language" in the settings before, we should respect that?
            // Actually, if user went to trouble to set config.json, they probably want it enforced.
            // Let's use it as the fallback if setting is not present.
            defaultLang = backendConfig.locale;
        }

        const initialLang = settings.getSettingValue("LoraHelper.Language", defaultLang);
        
        // If the backend config differs from current setting and user hasn't manually set it (hard to detect), 
        // we might want to auto-switch. But for now, using it as the default for getSettingValue is safe.
        // However, if the setting already exists in localStorage, getSettingValue will return that.
        // If user wants to force switch via config.json, they might need to clear setting or we logic it here.
        
        // Logic: If backend config "locale" is present, and it's different from what we thought,
        // we could hint or just let the default value handle new users.
        
        settings.addSetting({
            id: "LoraHelper.Language",
            name: "LoraHelper Language (中英文切换)",
            type: "combo",
            options: [
                { value: "en-US", text: "English" },
                { value: "zh-CN", text: "Chinese (简体中文)" }
            ],
            defaultValue: initialLang,
            onChange: (newVal) => {
                updateAllNodes(newVal);
            }
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Add a context menu option to switch language quickly
        const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function(canvas, options) {
            if (origGetExtraMenuOptions) {
                origGetExtraMenuOptions.apply(this, arguments);
            }
            
            // Debug log
            console.log("[LoraHelper] Checking node:", this.comfyClass, nodeData.category);
            
            // Check if this node belongs to LoraHelper category
            let isLoraHelperNode = false;
            
            // 1. Check category from nodeData
            if (nodeData.category && nodeData.category.includes("LoraHelper")) {
                isLoraHelperNode = true;
            }
            
            // 2. Fallback: Check hardcoded list (for safety)
            if (!isLoraHelperNode) {
                 const NODE_TYPE_MAP = {
                     "UniversalAIChat": true, "UniversalGGUFLoader": true, "UniversalOllamaLoader": true,
                     "UniversalCloudLoader": true, "LH_History_Monitor": true, "LH_MultiTextSelector": true,
                     "LoRA_AllInOne_Saver": true, "LH_SimpleText": true,
                     "LH_LoraLoader": true, "LH_KeywordLoraLoader": true, "LH_LlamaInstruct": true,
                     "LH_AutoRatio": true
                 };
                 // Check both comfyClass and type
                 if (NODE_TYPE_MAP[this.comfyClass] || NODE_TYPE_MAP[this.type]) {
                     isLoraHelperNode = true;
                 }
            }
            
            if (isLoraHelperNode) {
                console.log("[LoraHelper] Adding menu to:", this.comfyClass);
                options.push({
                    content: "Switch Language (中英文切换)",
                    callback: () => {
                        const current = app.ui.settings.getSettingValue("LoraHelper.Language", DEFAULT_LANG);
                        const next = current === "en-US" ? "zh-CN" : "en-US";
                        app.ui.settings.setSettingValue("LoraHelper.Language", next);
                        updateAllNodes(next);
                    }
                });
            } else {
                console.log("[LoraHelper] Skipped node:", this.comfyClass);
            }
        };
    },
    async nodeCreated(node) {
        const lang = app.ui.settings.getSettingValue("LoraHelper.Language", getComfyLanguage());
        updateSingleNode(node, lang);
    }
});

function updateAllNodes(lang) {
    const graph = app.graph;
    if (!graph) return;
    for (const node of graph._nodes) {
        updateSingleNode(node, lang);
    }
}

function updateSingleNode(node, lang) {
    // Only process LoraHelper nodes
    // We check based on the node title or type
    // Since we reverted node names, we can check node.type or node.comfyClass
    
    // Mapping of node types (comfyClass) to translation keys
    const NODE_TYPE_MAP = {
        "UniversalAIChat": "LoraHelper_Chat",
        "UniversalGGUFLoader": "Qwen3_GGUF_loader",
        "UniversalOllamaLoader": "LoraHelper_OllamaLoader",
        "UniversalCloudLoader": "LoraHelper_CloudLoader",
        "LH_History_Monitor": "LoraHelper_Monitor",
        "LH_MultiTextSelector": "LoraHelper_MultiTextSelector",
        "LoRA_AllInOne_Saver": "LoraHelper_Saver",
        "LH_SuperText": "LoraHelper_SuperText",
        "LH_LoraLoader": "LoraHelper_LoraLoader",
        "LH_LlamaInstruct": "LoraHelper_LlamaInstruct",
        "LH_AutoRatio": "LoraHelper_AutoRatio"
    };

    const translationKey = NODE_TYPE_MAP[node.comfyClass];
    if (!translationKey) return;

    const dict = TRANSLATIONS[lang]?.[translationKey];
    if (!dict && lang !== "en-US") return; // If no translation found for non-EN, skip
    
    // Process Node Title - Disabled as per user request
    /*
    if (!node.originalTitle) {
        node.originalTitle = node.title || node.type;
    }
    if (lang === "en-US") {
        node.title = node.originalTitle;
    } else if (dict && dict["title"]) {
        node.title = dict["title"];
    }
    */

    // Process Inputs (Connections)
    if (node.inputs) {
        for (const input of node.inputs) {
            // Store original name if not stored
            if (!input.originalLabel) {
                // [Fix] Always use input.name as the source of truth to avoid pollution from saved workflows or previous sessions
                input.originalLabel = input.name;
            }
            
            if (lang === "en-US") {
                input.label = input.originalLabel;
            } else if (dict && dict[input.name]) {
                input.label = dict[input.name];
            }
        }
    }

    // Process Widgets
    if (node.widgets) {
        for (const widget of node.widgets) {
             // Store original name if not stored
            if (!widget.originalLabel) {
                // [Fix] Always use widget.name as the source of truth
                widget.originalLabel = widget.name;
            }

            if (lang === "en-US") {
                widget.label = widget.originalLabel;
            } else if (dict && dict[widget.name]) {
                widget.label = dict[widget.name];
            }
        }
    }
    
    // Process Outputs (New Feature)
    if (node.outputs) {
        for (const output of node.outputs) {
             // Store original name if not stored
            if (!output.originalLabel) {
                // [Fix] Always use output.name as the source of truth
                output.originalLabel = output.name;
            }

            if (lang === "en-US") {
                output.label = output.originalLabel;
            } else if (dict && dict[output.name]) {
                output.label = dict[output.name];
            }
        }
    }
    
    // Force redraw
    node.setDirtyCanvas(true, true);
}
