import { app } from "../../scripts/app.js";

const TRANSLATIONS = {
    "zh-CN": {
        "LoraHelper_Chat": {
            "model": "模型 (Model)",
            "image": "图像 (Image)",
            "max_tokens": "最大生成长度",
            "temperature": "温度 (Temperature)",
            "repetition_penalty": "重复惩罚",
            "instruction": "系统指令/执行指令 (Instruction)",
            "user_material": "用户素材 (User Material)",
            "chat_mode": "对话模式",
            "enable_tag": "启用标签提取",
            "enable_filename": "启用文件名生成",
            "seed": "种子 (Seed)",
            "release_vram": "自动释放显存"
        },
        "Qwen3_GGUF_loader": {
            "gguf_model": "GGUF模型",
            "clip_model": "CLIP视觉模型",
            "n_gpu_layers": "GPU层数 (-1为全部)",
            "n_ctx": "最大上下文 (n_ctx)"
        },
        "LoraHelper_Monitor": {
            "raw_input": "原始输出 (Raw Input)",
            "clear_history": "清除历史记录"
        },
        "LoraHelper_Splitter": {
            "text": "文本输入",
            "user_prefix": "用户前缀"
        },
        "LoraHelper_Saver": {
            "images": "图像",
            "gen_prompt": "生成提示词",
            "lora_tags": "LoRA标签",
            "filename_final": "最终文件名",
            "folder_path": "保存路径",
            "filename_prefix": "文件前缀",
            "trigger_word": "触发词",
            "save_workflow": "保存工作流"
        },
        "TestVisionWorkflow": {
            "model": "模型",
            "image": "图像",
            "instruction": "指令",
            "enable_tag": "启用标签",
            "enable_filename": "启用文件名",
            "max_tokens": "最大长度",
            "temperature": "温度",
            "simple_mode": "极简调试模式"
        }
    }
};

const DEFAULT_LANG = "en-US";

app.registerExtension({
    name: "LoraHelper.Translation",
    async setup() {
        const settings = app.ui.settings;
        settings.addSetting({
            id: "LoraHelper.Language",
            name: "LoraHelper Language",
            type: "combo",
            options: [
                { value: "en-US", text: "English" },
                { value: "zh-CN", text: "Chinese" }
            ],
            defaultValue: DEFAULT_LANG,
            onChange: (newVal) => {
                updateAllNodes(newVal);
            }
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // No action needed here usually, but we could patch nodeType.prototype.onNodeCreated
    },
    async nodeCreated(node) {
        const lang = app.ui.settings.getSettingValue("LoraHelper.Language", DEFAULT_LANG);
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
        "LH_History_Monitor": "LoraHelper_Monitor",
        "Qwen3TextSplitter": "LoraHelper_Splitter",
        "LoRA_AllInOne_Saver": "LoraHelper_Saver",
        "TestVisionWorkflow": "TestVisionWorkflow"
    };

    const translationKey = NODE_TYPE_MAP[node.comfyClass];
    if (!translationKey) return;

    const dict = TRANSLATIONS[lang]?.[translationKey];
    if (!dict && lang !== "en-US") return; // If no translation found for non-EN, skip
    
    // Process Inputs (Connections)
    if (node.inputs) {
        for (const input of node.inputs) {
            // Store original name if not stored
            if (!input.originalLabel) {
                input.originalLabel = input.label || input.name;
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
                // Some widgets might have a label property, others rely on name
                widget.originalLabel = widget.label || widget.name;
            }

            if (lang === "en-US") {
                widget.label = widget.originalLabel;
            } else if (dict && dict[widget.name]) {
                widget.label = dict[widget.name];
            }
        }
    }
    
    // Force redraw
    node.setDirtyCanvas(true, true);
}
