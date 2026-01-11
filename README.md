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
   *Note: This project requires `llama-cpp-python` for GGUF model support.*

### 🧩 Node Overview

#### 1. LoraHelper_Loader (GGUF Model Loader)
*   **Function**: Loads `.gguf` format LLM models.
*   **Path**: Place your models in `ComfyUI/models/llm/`.
*   **Features**: Supports auto-offloading VRAM.

#### 2. LoraHelper_Chat (DeepBlue Architecture)
The core intelligence node.
*   **Inputs**:
    *   `model`: The loaded LLM.
    *   `image` (Optional): Connecting an image enables **Vision Mode**.
    *   `context`: Connects to history for multi-turn conversations.
    *   `user_prompt` (UP): Input material/text.
    *   `system_command` (SC): Executive instructions for the AI.
*   **Modes**:
    *   **Enhance_Prompt**: Creatively expands on user inputs.
    *   **Debug_Chat**: Analyzes prompts or images based on instructions.

#### 3. LoraHelper_Monitor (History Viewer)
*   **Function**: Displays a rolling buffer of the last 5 chat interactions.
*   **Usage**: Connect to a `ShowText` node to visualize the conversation history.

#### 4. LoraHelper_Splitter (Text Parser)
*   **Function**: Parses the LLM output into structured data.
*   **Logic**: Looks for specific markers:
    *   `SECTION 1`: Generation Prompt
    *   `SECTION 2`: LoRA Tags
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
   *注意：本项目依赖 `llama-cpp-python` 来加载 GGUF 模型，请确保正确安装。*

### 🧩 节点详解

#### 1. LoraHelper_Loader (模型加载器)
*   **功能**: 加载 `.gguf` 格式的大语言模型。
*   **路径**: 请将模型文件放入 `ComfyUI/models/llm/` 目录。
*   **特性**: 支持 VRAM 自动卸载，优化显存占用。

#### 2. LoraHelper_Chat (核心对话节点)
基于 DeepBlue 架构的智能核心。
*   **输入参数**:
    *   `model`: 已加载的 LLM 模型。
    *   `image` (可选): 接入图片后自动进入**视觉模式 (Vision Mode)**，忽略文本输入，仅根据指令分析图片。
    *   `context`: 上下文输入，用于多轮对话记忆。
    *   `user_prompt` (UP): 用户素材或原始提示词。
    *   `system_command` (SC): 给 AI 的系统级指令。
*   **运行模式**:
    *   **Enhance_Prompt**: 对用户素材进行创意扩写。
    *   **Debug_Chat**: 根据指令分析素材或图片，输出思考过程。

#### 3. LoraHelper_Monitor (历史看板)
*   **功能**: 维护并显示最近 5 轮的对话历史。
*   **用法**: 输出连接到 `ShowText` 节点，方便实时监控 AI 的回复和上下文。

#### 4. LoraHelper_Splitter (文本切分器)
*   **功能**: 将 AI 的输出解析为结构化数据。
*   **逻辑**: 自动识别以下标记进行提取：
    *   `SECTION 1`: 生图提示词 (Gen Prompt)
    *   `SECTION 2`: LoRA 训练标签 (Tags)
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
