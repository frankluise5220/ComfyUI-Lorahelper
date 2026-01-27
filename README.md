# 🚀 ComfyUI-LoraHelper

一个专为 ComfyUI 打造的 **全能型 Prompt 引擎与 LoRA 炼丹助手**。

它不仅是一个简单的 GGUF 模型加载器，更内置了 **“专家级” Prompt 工程师逻辑**：
*   ✨ **灵魂反推 (Soul-Deep Captioning)**：超越简单的看图说话，精准捕捉画面氛围与意境，转化为 Flux/SD 可用的高质量 Prompt。
*   🚀 **智能扩写 (Intelligent Expansion)**：让简单的“女孩在海边”瞬间变成包含光影、材质、构图的 300 字大师级描述。
*   💾 **一键炼丹 (One-Click Dataset)**：自动整理 LoRA 训练素材，图片、标签、工作流元数据一键打包，拖入即复现。

**无需昂贵显卡**，本地 GGUF 模型即可驱动这一切，让你的工作流彻底自动化。

An **All-in-One Prompt Engine & LoRA Training Assistant** for ComfyUI.

More than just a GGUF loader, it embeds **Expert-Level Prompt Engineering logic**:
*   ✨ **Soul-Deep Captioning**: Goes beyond object detection to capture atmospheric nuance and artistic essence, converting images into production-ready Flux/SD prompts.
*   🚀 **Intelligent Expansion**: Instantly transforms simple inputs like "girl at beach" into 300-word masterworks rich in lighting, texture, and composition.
*   💾 **One-Click Dataset Prep**: Automates LoRA training data organization—images, tags, and workflow metadata saved in one click. Drag-and-drop to reproduce.

**No expensive GPU required.** Power your automated workflow with local GGUF models.

---

[English Version](#-english-version) | [中文说明](#-中文说明)

<a name="-english-version"></a>
## 🌏 English Version

### 📦 Installation
1. Clone this repository into your `ComfyUI/custom_nodes/` directory:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/yourusername/ComfyUI-Lorahelper.git
   ```
2. Install the required dependencies:
   ```bash
   cd ComfyUI-Lorahelper
   pip install -r requirements.txt
   ```
   *Note: This project requires `llama-cpp-python` for GGUF model support. For vision capabilities, ensure your installation supports CLIP/MMProj.*

### 🧩 Node Overview

#### 1. LH_GGUFLoader (GGUF Model Loader)
*   **Function**: Loads `.gguf` format LLM models.
*   **Supported Models**: Extensive support for mainstream VLM/LLM GGUF models, including **Qwen2.5-VL / Qwen2-VL**, **Llama 3.2 Vision**, **Yi-VL**, **Llava 1.5/1.6**, and other GGUF-compatible models.
*   **Path**: Place your models in `ComfyUI/models/llm/`.
*   **Features**:
    *   **GGUF Model**: Select your main LLM.
    *   **CLIP Model**: (Optional) Load a CLIP/MMProj model to enable vision capabilities for image analysis.
    *   **GPU Layers**: Supports auto-offloading VRAM.
    *   **n_ctx**: Maximum context window size (default: 8192).

#### 2. LH_AIChat (DeepBlue Architecture)
The core intelligence node. [View Logic Flowchart](./Logic_Flowchart.md)
*   **Inputs**:
    *   `model`: The loaded LLM.
    *   `image` (Optional): Connecting an image automatically triggers **Implicit Vision Mode**.
    *   `user_material`: Input material/text (Ignored in Vision Mode).
    *   `instruction`: Executive instructions for the AI.
    *   `max_tokens`: Max generation length.
    *   `temperature`: Creativity (higher = more random).
    *   `repetition_penalty`: Penalty for repeating text.
    *   `seed`: Random seed for reproducibility.
    *   `release_vram`: Auto-release VRAM after generation.
*   **Outputs**:
    *   `prompt`: The main generated text (SECTION 1).
    *   `tags`: Extracted tags (if enabled).
    *   `filename`: Extracted filename (if enabled).
    *   `raw_output`: Raw history for Monitor.
*   **Vision Mode (Implicit)**:
    *   Triggered automatically when an image is connected.
    *   **Auto-Instruction**: If `instruction` is left default/empty, it uses a built-in **JoyCaption-style Uncensored** prompt for detailed image captioning.
    *   **Customizable**: You can override the built-in behavior by providing your own `instruction`.
*   **Modes (Text-only)**:
    *   **Enhance_Prompt**: Creatively expands on user inputs.
    *   **Debug_Chat**: Analyzes prompts based on instructions.

#### 3. LH_History_Monitor (History Viewer)
*   **Function**: Manages conversation history and context.
*   **Features**:
    *   **Visual History**: Displays the last 5 rounds in clear "Round X" cards.
    *   **Auto-Resize**: Automatically adjusts size to fit content.
    *   **Context Loop**: Outputs formatted context to be copied into `user_material` for multi-turn debugging.

#### 4. LH_TextSplitter (Legacy)
*   **Status**: This node has been removed as `LH_AIChat` now fully handles output splitting internally.

#### 5. LH_AllInOne_Saver (Dataset Saver)
*   **Function**: One-click solution for saving training data, prompts, tags, and workflows.
*   **Features**:
    *   **One-Click Save**: Simultaneously saves Images, Caption text, Log files, and ComfyUI Workflow metadata.
    *   **Workflow Embedding**: Supports saving the full ComfyUI workflow into the PNG, allowing drag-and-drop to reproduce the generation.
    *   **Flexible Naming**: Supports custom prefixes, filename overrides, and auto-incrementing.
*   **Inputs**:
    *   `images`: Input images to save.
    *   `folder_path`: Subfolder path in output directory (default: "LoRA_Train_Data").
    *   `filename_prefix`: Prefix for filenames (default: "Anran").
    *   `trigger_word`: Trigger word added to the start of caption files (default: "ChenAnran").
    *   `save_workflow`: Toggle to save ComfyUI workflow metadata in PNG.
    *   `gen_prompt`: (Optional) Connect full description text to save in `_log.txt`.
    *   `lora_tags`: (Optional) Connect tags to save in `.txt`.
    *   `filename_final`: (Optional) Override specific filename (will be combined with prefix).
*   **Outputs**:
    *   **Image**: `.png` with metadata (Workflow).
    *   **Tags**: `.txt` file with format `trigger_word, tags`.
    *   **Log**: `_log.txt` with the full raw AI response.
*   **Path**: Default saves to `ComfyUI/output/LoRA_Train_Data/`.

#### 6. LH_MultiTextSelector (Dynamic Prompt Generator)
*   **Function**: A powerful text selector with support for Dynamic Prompts syntax.
*   **Features**:
    *   **4-Slot Selection**: Input up to 4 text candidates.
    *   **Mode**: `Random` (select one randomly) or `Sequential` (cycle through them).
    *   **Dynamic Syntax**:
        *   **Inline Random**: `{cat|dog|bird}` - Randomly chooses one option.
        *   **Weighted Random**: `{0.8::cat|0.2::dog}` - Weighted probability.
        *   **Wildcards**: `__colors__` - Reads from `wildcards/colors.txt` (supports recursive lookup in `ComfyUI/wildcards` or plugin's `wildcards` folder).
    *   **Seed Control**: Ensure reproducible results for your random prompts.

## 💡 Best Practice

It is recommended to use this tool with **[Dynamic Prompts (DP)](https://github.com/adieyal/comfyui-dynamicprompts)**:
1.  **DP Randomization**: Use DP nodes to generate random combinations (e.g., `{white dress|red cheongsam}, {black hair|blonde hair}`).
2.  **AI Refinement**: Feed the random output from DP into this plugin as `user_material`.
3.  **Deep Expansion**: This plugin will automatically add lighting, composition, and scene details based on the random attributes.

**Core Advantage**: Combines the "breadth" of randomness with the "depth" of AI to quickly generate high-quality, diverse LoRA training datasets.

---

<a name="-中文说明"></a>
## 🇨🇳 中文说明

### 📦 安装指南
1. 将本项目克隆到 `ComfyUI/custom_nodes/` 目录：
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/yourusername/ComfyUI-Lorahelper.git
   ```
2. 安装必要的依赖库：
   ```bash
   cd ComfyUI-Lorahelper
   pip install -r requirements.txt
   ```
   *注意：本项目依赖 `llama-cpp-python` 来加载 GGUF 模型。如需使用视觉反推功能，请确保安装版本支持 CLIP/MMProj。*

### 🧩 节点详解

#### 1. LH_GGUFLoader (模型加载器)
*   **功能**: 加载 `.gguf` 格式的大语言模型。
*   **路径**: 请将模型文件放入 `ComfyUI/models/llm/` 目录。
*   **特性**:
    *   **GGUF Model**: 选择主 LLM 模型。
    *   **CLIP Model**: (可选) 加载 CLIP/MMProj 模型以启用视觉能力。
    *   **GPU Layers**: 支持自动显存分流 (Offload)。
    *   **n_ctx**: 最大上下文窗口大小 (默认: 8192)。

#### 2. LH_AIChat (DeepBlue Architecture)
核心智能节点。[查看逻辑流程图](./Logic_Flowchart.md)
*   **输入参数**:
    *   `model`: 已加载的 LLM 模型。
    *   `image` (可选): 连接图片后自动触发 **隐形反推模式**。
    *   `user_material`: 用户输入的素材/文本 (反推模式下忽略)。
    *   `instruction`: 给 AI 的执行指令。
    *   `max_tokens`: 最大生成长度。
    *   `temperature`: 温度 (创造力，越高越随机)。
    *   `repetition_penalty`: 重复惩罚系数。
    *   `seed`: 随机种子 (控制结果一致性)。
    *   `release_vram`: 生成后自动释放显存。
*   **输出端口**:
    *   `prompt`: 核心描述文本 (description).
    *   `tags`: 提取的标签 (需开启tag开关)。
    *   `filename`: 提取的文件名 (需开启开关)。
    *   `raw_output`: 原始输出 (连入 Monitor)。
*   **隐形反推模式**:
    *   **自动触发**: 只要连接图片，无需输入任何用户指令，即刻生效。
    *   **智能指令**: 若 `instruction` 保持默认或留空，将使用内置的 **JoyCaption 同款无审查** 强力反推指令，生成极详尽的视觉描述。
    *   **自定义**: 您也可以输入自定义 `instruction` 来覆盖内置行为。
*   **运行模式 (纯文本)**:
    *   **Enhance_Prompt**: 对用户素材（包括image和文本）进行创意扩写。
    *   **Debug_Chat**: 根据指令分析素材，输出思考过程。

#### 3. LH_History_Monitor (历史看板)
*   **功能**: 维护并显示最近 5 轮的对话历史。
*   **特性**:
    *   **可视化显示**: 以 "Round X" 卡片形式清晰展示对话内容，自动调整窗口大小。
    *   **上下文循环**: 输出格式化后的 `context` 文本，可复制到 `user_material` 实现多轮对话调试。

#### 4. LH_TextSplitter (Legacy)
*   **状态**: 该节点已被移除，`LH_AIChat` 现已内置完整的自动切分功能。

#### 5. LH_AllInOne_Saver (数据集保存器)
*   **功能**: 一键保存 LoRA 训练所需的所有文件（Prompt、Tags、工作流）。
*   **特性**:
    *   **一键保存**: 同时保存图片、标签文本、日志文件和 ComfyUI 工作流元数据。
    *   **工作流嵌入**: 支持将完整的 ComfyUI 工作流嵌入 PNG 图片，拖入界面即可完美复现。
    *   **灵活命名**: 支持自定义前缀、覆盖文件名和自动递增。
*   **输入参数**:
    *   `images`: 需保存的图片输入。
    *   `folder_path`: 保存路径子文件夹 (默认: "LoRA_Train_Data")。
    *   `filename_prefix`: 文件名前缀 (默认: "Anran").
    *   `trigger_word`: 触发词，自动添加在 caption 文件的最开头 (默认: "ChenAnran").
    *   `save_workflow`: 开关，决定是否将 ComfyUI 工作流元数据写入图片 (支持拖入复现)。
    *   `gen_prompt`: (可选) 连接完整描述文本，保存到 `_log.txt`。
    *   `lora_tags`: (可选) 连接标签文本，保存到 `.txt` (位于触发词之后)。
    *   `filename_final`: (可选) 覆盖具体文件名 (会自动拼接前缀)。
*   **输出内容**:
    *   **图片**: `.png` 格式，包含完整元数据（工作流）。
    *   **标签**: `.txt` 文件，格式为 `触发词, 标签1, 标签2...`。
    *   **日志**: `_log.txt` 文件，记录 AI 的原始完整回复（可以接入任何想要保存的文本）。
*   **路径**: 默认保存在 `ComfyUI/output/LoRA_Train_Data/`，支持自定义子文件夹。

#### 6. LH_MultiTextSelector (动态提示词生成器)
*   **功能**: 支持动态语法 (Dynamic Prompts) 的多功能文本选择器。
*   **特性**:
    *   **4路选择**: 支持输入 4 个候选文本框。
    *   **模式切换**: `Random` (随机选择) 或 `Sequential` (顺序循环)。
    *   **动态语法支持**:
        *   **内联随机**: `{cat|dog|bird}` - 随机选择其中一个。
        *   **权重随机**: `{0.8::cat|0.2::dog}` - 按概率选择。
        *   **通配符 (Wildcards)**: `__colors__` - 读取 `wildcards/colors.txt` 文件的内容 (支持递归查找 `ComfyUI/wildcards` 或本插件内置目录)。
    *   **Seed 控制**: 通过种子固定随机结果，方便复现。

---

## 💡 使用建议 (Best Practice)

建议配合 **[Dynamic Prompts (DP)](https://github.com/adieyal/comfyui-dynamicprompts)** 插件使用：
1.  **DP 抽签**: 使用 DP 节点生成随机组合（如 `{白色长裙|红色旗袍}, {黑发|金发}`）。
2.  **AI 润色**: 将 DP 的随机输出作为 `user_material` 输入给本插件。
3.  **深度扩写**: 本插件会基于随机属性，自动补充灯光、构图及场景细节。

**核心优势**: 结合了随机性的“广度”和 AI 的“深度”，能够快速生成高质量、多样化的 LoRA 训练数据集。
