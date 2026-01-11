# 🚀 ComfyUI-LoraHelper

一个专为 ComfyUI 设计的 AI 自动生成提示词、自动化批量生图、 LoRA 训练素材整理工具。通过集成本地大语言模型（GGUF），实现从原始素材到结构化训练数据的自动化转化。

An AI-powered tool designed for ComfyUI to automate prompt generation, batch image creation, and LoRA training dataset organization. By integrating local Large Language Models (GGUF), it achieves a seamless, automated transformation from raw materials to structured training datasets.

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

#### 1. Qwen3_GGUF_loader (GGUF Model Loader)
*   **Function**: Loads `.gguf` format LLM models.
*   **Path**: Place your models in `ComfyUI/models/llm/`.
*   **Features**:
    *   **GGUF Model**: Select your main LLM.
    *   **CLIP Model**: (Optional) Load a CLIP/MMProj model to enable vision capabilities for image analysis.
    *   **GPU Layers**: Supports auto-offloading VRAM.

#### 2. LoraHelper_Chat (DeepBlue Architecture)
The core intelligence node.
*   **Inputs**:
    *   `model`: The loaded LLM.
    *   `image` (Optional): Connecting an image automatically triggers **Implicit Vision Mode**.
    *   `context`: Connects to history for multi-turn conversations.
    *   `user_prompt` (UP): Input material/text.
    *   `system_command` (SC): Executive instructions for the AI.
*   **Vision Mode (Implicit)**:
    *   Triggered automatically when an image is connected.
    *   **Auto-Instruction**: Ignores `user_prompt` and uses a built-in optimized instruction to generate structured outputs (Caption, Tags, Filename).
    *   **Tagging**: Generates Danbooru-style tags, comma-separated, covering subject, appearance, attire, pose, view, and background.
*   **Modes (Text-only)**:
    *   **Enhance_Prompt**: Creatively expands on user inputs.
    *   **Debug_Chat**: Analyzes prompts based on instructions.

#### 3. LoraHelper_Monitor (History Viewer)
*   **Function**: Manages conversation history and context.
*   **Features**:
    *   **Rolling Buffer**: Maintains the last 5 rounds of conversation.
    *   **Built-in Display**: Directly shows the chat history on the node (no external `ShowText` needed).
    *   **Context Loop**: Outputs context to be fed back into the Chat node.

#### 4. LoraHelper_Splitter (Text Parser)
*   **Function**: Parses the LLM output into structured data.
*   **Logic**: Looks for specific markers:
    *   `SECTION 1`: Natural Language Description (Caption)
    *   `SECTION 2`: LoRA Tags (Comma-separated)
    *   `SECTION 3`: Filename

#### 5. LoraHelper_Saver (Dataset Saver)
*   **Function**: One-click solution for saving training data.
*   **Outputs**:
    *   **Image**: `.png` with metadata.
    *   **Tags**: `.txt` file with trigger word and tags.
    *   **Log**: `_log.txt` with the full raw AI response.
*   **Path**: Default saves to `ComfyUI/output/LoRA_Train_Data/`.

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

#### 1. LoraHelper_Loader (模型加载器)
*   **功能**: 加载 `.gguf` 格式的大语言模型。
*   **路径**: 请将模型文件放入 `ComfyUI/models/llm/` 目录。
*   **特性**:
    *   **GGUF Model**: 选择主 LLM 模型。
    *   **CLIP Model**: (可选) 加载 CLIP/MMProj 模型，为 Chat 节点提供视觉分析能力。
    *   支持 VRAM 自动卸载，优化显存占用。

#### 2. LoraHelper_Chat (核心对话节点)
基于 DeepBlue 架构的智能核心。
*   **输入参数**:
    *   `model`: 已加载的 LLM 模型。
    *   `image` (可选): 接入图片后自动进入**隐形反推模式 (Implicit Vision Mode)**。
    *   `context`: 上下文输入，用于多轮对话记忆。
    *   `user_prompt` (UP): 用户素材或原始提示词。
    *   `system_command` (SC): 给 AI 的系统级指令。
*   **隐形反推模式**:
    *   **自动触发**: 只要连接图片，即刻生效。
    *   **智能指令**: 自动忽略 `user_prompt`，使用内置的强指令生成结构化输出（自然语言描述、LoRA 标签、文件名）。
    *   **打标优化**: 自动生成标准 Danbooru 风格标签，逗号分隔，涵盖主体、外貌、衣着、动作、视角、背景等核心要素。
*   **运行模式 (纯文本)**:
    *   **Enhance_Prompt**: 对用户素材进行创意扩写。
    *   **Debug_Chat**: 根据指令分析素材，输出思考过程。

#### 3. LoraHelper_Monitor (历史看板)
*   **功能**: 维护并显示最近 5 轮的对话历史。
*   **特性**:
    *   **可视化显示**: 以 "Round X" 卡片形式清晰展示对话内容，便于阅读。
    *   **上下文循环**: 输出原始 `context` 文本，可回传给 Chat 节点实现多轮对话记忆。
    *   **滚动缓存**: 自动保留最新的 5 条记录。

#### 4. LoraHelper_Splitter (文本切分器)
*   **功能**: 将 AI 的输出解析为结构化数据。
*   **逻辑**: 自动识别以下标记进行提取：
    *   `SECTION 1`: 自然语言描述 (Caption)
    *   `SECTION 2`: LoRA 训练标签 (Tags - 逗号分隔)
    *   `SECTION 3`: 最终文件名 (Filename)

#### 5. LoraHelper_Saver (数据集保存器)
*   **功能**: 一键保存 LoRA 训练所需的所有文件。
*   **输出内容**:
    *   **图片**: `.png` 格式，包含完整元数据。
    *   **标签**: `.txt` 文件，格式为 `触发词, 标签1, 标签2...`。
    *   **日志**: `_log.txt` 文件，记录 AI 的原始完整回复。
*   **路径**: 默认保存在 `ComfyUI/output/LoRA_Train_Data/`，支持自定义子文件夹。

---

## 💡 使用建议 (Best Practice)

建议配合 **[Dynamic Prompts (DP)](https://github.com/adieyal/comfyui-dynamicprompts)** 插件使用：
1.  **DP 抽签**: 使用 DP 节点生成随机组合（如 `{白色长裙|红色旗袍}, {黑发|金发}`）。
2.  **AI 润色**: 将 DP 的随机输出作为 `user_prompt` 输入给本插件。
3.  **深度扩写**: 本插件会基于随机属性，自动补充灯光、构图及场景细节。

**核心优势**: 结合了随机性的“广度”和 AI 的“深度”，能够快速生成高质量、多样化的 LoRA 训练数据集。
