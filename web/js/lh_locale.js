import { app } from "../../scripts/app.js";

const TRANSLATIONS = {
    "zh-CN": {
        "LoraHelper_Chat": {
            "model": "模型 (Model)",
            "image": "图像 (Image)",
            "max_new_tokens": "最大生成长度",
            "temperature": "温度 (Temperature)",
            "top_p": "Top-P",
            "top_k": "Top-K",
            "repetition_penalty": "重复惩罚",
            "system_command": "系统指令 (System Command)",
            "user_prompt": "用户素材 (User Prompt)",
            "chat_mode": "对话模式",
            "enable_tags_extraction": "启用标签提取",
            "print_debug_info": "打印调试信息",
            "seed": "种子 (Seed)"
        },
        "Qwen3_GGUF_loader": {
            "model_path": "模型路径",
            "max_ctx": "最大上下文"
        },
        "LoraHelper_Monitor": {
            "history_data": "历史数据"
        },
        "LoraHelper_Splitter": {
            "text_input": "文本输入",
            "chunk_size": "分块大小",
            "chunk_overlap": "重叠大小"
        },
        "LoraHelper_Saver": {
            "images": "图像",
            "captions": "描述文本",
            "output_path": "输出路径",
            "folder_name": "文件夹名",
            "filename_prefix": "文件前缀"
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
        "LoRA_AllInOne_Saver": "LoraHelper_Saver"
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
