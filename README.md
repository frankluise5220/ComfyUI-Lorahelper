# 🚀 ComfyUI-LoraHelper

一个专为 ComfyUI 设计的 AI 自动生成提示词、自动化批量生图、 LoRA 训练素材整理工具。通过集成大语言模型（LLM），实现从原始素材到结构化训练数据的自动化转化。

An AI-powered tool designed for ComfyUI to automate prompt generation, batch image creation, and LoRA training dataset organization. By integrating Large Language Models (LLM), it achieves a seamless, automated transformation from raw materials to structured training datasets.

---

[English Version](#-core-features-en) | [中文说明](#-核心功能)

<details>
<summary>🌐 Click to expand English Version / 点击展开英文版</summary>

## 📦 Core Features (EN)

- **Model Loader (GGUF_Loader)**: A dedicated loader optimized for GGUF architectures (e.g., Qwen3), featuring an integrated **VRAM Auto-Offload** mechanism to maximize generation efficiency.
- **User Interaction (Debug_Chat)**: Supports dynamic adjustment of core AI parameters such as `max_tokens`, `temperature`, etc.
    - Includes the following two primary modes:
    - **Debug Mode**: Analyzes `user_prompt` based on `system_command` instructions to output logical reasoning and thought processes, facilitating easier prompt debugging.
    - **Prompt_Enhance Mode**: AI creatively expands on user-provided materials following specific system instructions to generate high-quality, detail-rich visual descriptions.
- **Script Parsing (Output_Splitter)**: An automated extraction tool that leverages specific identifiers (**SECTION 1 / SECTION 2 / SECTION 3**) to parse prompts, LoRA training tags, and custom filenames from AI responses.
- **Automated Storage (All-In-One_Saver)**: A one-click solution to synchronize the saving of images, matching tag files (standardized for LoRA training), and comprehensive prompt logs.

## 📂 Directory & Storage Specifications

- **LLM Models**: Please place your `.gguf` model files into the `ComfyUI/models/llm/` directory.
- **Asset Storage**: Files are saved to `ComfyUI/output/LoRA_Train_Data/` by default. Custom paths are supported.

## ✂️ Splitter Execution Mechanism

The node identifies and segments AI output by recognizing specific semantic markers:
- `SECTION 1`: Extracted as the Image Generation Prompt (`gen_prompt`).
- `SECTION 2`: Extracted as LoRA Training Tags (`lora_tags`).
- `SECTION 3`: Extracted as the final Filename (`filename_final`).
*Fallback Mechanism: If no markers are detected, the system automatically captures the first natural paragraph to ensure the workflow remains uninterrupted.*

## 💾 Saving Mechanism (Three-In-One)

Every save operation generates three synchronized files:
1. **Image (.png)**: Contains full generation metadata (workflow embedding is optional).
2. **Tags (.txt)**: Formatted as `trigger_word, tag1, tag2...`, ready for training.
3. **Logs (_log.txt)**: Records the original, complete AI response to preserve all raw prompt information for future reference.

## 🛠️ Modular Installation

This project utilizes a decoupled architecture. Ensure the following files are present in the plugin folder:
- `__init__.py`: Plugin entry point and node registration.
- `LH_Chat.py`: Handles model loading and AI dialogue/enhancement logic.
- `LH_Utils.py`: Handles text splitting and file storage nodes.

## 💡 Recommendation

**Combine with the Dynamic Prompts extension for maximum efficiency:**
- **Workflow**: Connect the `gen_prompt` output of this plugin to the input of a Dynamic Prompts node.
- **Advantage**: While the AI generates high-level scene descriptions, Dynamic Prompts can handle micro-variables via wildcards (e.g., `{red|blue} dress`), enabling infinite variations for batch generation from a single AI script.

</details>

---

## 📦 核心功能 (CN)

- **模型加载 (GGUF_Loader)**: 专为 Qwen3 等 GGUF 架构设计的加载器，内置 VRAM 自动卸载机制。
- **用户交互 (Debug_Chat)**: 支持动态调节 `max_tokens`、`temperature` 等 AI 核心参数。
    - 并且包括以下两个功能：
    - **Debug Mode**: 根据 system_command 的指令，对 user_prompt 进行分析，给出思考结果，方便调试。
    - **Prompt_Enhance Mode**: AI 将根据用户提供素材进行创意扩写，生成更丰富的视觉描述提示词。
- **剧本切分 (Output_Splitter)**: 基于特定的分段词（SECTION 1/2/3）从输出中截取提示词、LoRA 标签和自定义文件名。
- **自动化存盘 (All-In-One_Saver)**: 一键保存图片、同名标签文件（LoRA 训练打标用）以及详细的 prompt 日志。

## 📂 目录存放规范

- **LLM 模型**: 请将 `.gguf` 文件放入 `ComfyUI/models/llm/` 目录下。
- **素材存盘**: 默认保存在 `ComfyUI/output/LoRA_Train_Data/`，支持自定义路径。

## ✂️ Splitter 运行机制

节点通过识别 AI 输出中的特定标记进行切分：
- `SECTION 1`: 提取为生图提示词 (gen_prompt)。
- `SECTION 2`: 提取为 LoRA 训练标签 (lora_tags)。
- `SECTION 3`: 提取为最终文件名 (filename_final)。
*若未发现标记，系统会自动抓取首个自然段进行保底，确保流程不中断。*

## 💾 保存机制 (三位一体)

每次保存将生成：
1. **图片 (.png)**: 包含完整生图元数据 (工作流保存可选)。
2. **标签 (.txt)**: 格式为 `触发词, 标签1, 标签2...`。
3. **日志 (_log.txt)**: 记录 AI 的原始完整描述，方便整理文生图原始信息。

## 🛠️ 模块化安装

本项目采用解耦架构，请确保文件夹内包含以下文件：
- `__init__.py`: 插件入口与节点注册。
- `LH_Chat.py`: 处理模型加载与 AI 对话及增强逻辑。
- `LH_Utils.py`: 处理文本切分与文件存盘节点。

## 💡 使用建议

**建议配合 Dynamic Prompts 插件使用：**
- **操作方式**: 将本插件输出的 `gen_prompt` 接入 Dynamic Prompts 节点的输入端。
- **核心优势**: AI 负责生成场景描述，Dynamic Prompts 负责对通配符变量进行替换（如 `{red|blue} dress`），实现单次 AI 剧本下的无限变体批量生图。