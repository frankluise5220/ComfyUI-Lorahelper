# 🚀 ComfyUI-LoraHelper

[English](#-core-features-en) | [中文](#-核心功能)

<details>
<summary>🌐 Click to expand English Version / 点击展开英文版</summary>

一个专为 ComfyUI 设计的AI自动生成提示词、自动化批量生图、 LoRA 训练素材整理工具。通过集成大语言模型（LLM），实现从原始素材到结构化训练数据的自动化转化。

## 📦 核心功能

- **模型加载 (GGUF_Loader)**: 专为 Qwen3 等 GGUF 架构设计的加载器，内置 VRAM 自动卸载机制。
- **用户交互 (debug_Chat)**: 支持动态调节 `max_tokens`、`temperature` 等 AI 核心参数。
-   并且包括以下两个功能：
-     Debug Mode: 根据system_command的指令，对user_prompt进行分析，给出思考结果，方便调试。
-     Prompt_Enhance Mode: AI 将根据用户提供素材，按照用户的system_command的指令进行创意扩写，生成更丰富的视觉描述提示词。
- **剧本切分 (output_Splitter)**: 基于特定的分段词（SECTION1/SECTION2/SECTION3)从 gen_text输出中截取提示词、LoRA 标签和自定义文件名。
- **自动化存盘 (All-In-One_Saver)**: 一键保存图片、同名标签文件（LoRA训练打标用）以及详细的prompt日志。

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
1. **图片 (.png)**: 包含完整生图元数据(工作流保存可选）。
2. **标签 (.txt)**: 格式为 `触发词, 标签1, 标签2...`。
3. **日志 (_log.txt)**: 记录 AI 的原始完整描述，方便整理文生图原始信息。

## 🛠️ 模块化安装

本项目采用解耦架构，请确保文件夹内包含以下文件：
- `__init__.py`: 插件入口与节点注册。
- `LH_Chat.py`: 处理模型加载与 AI 对话//提示词增加逻辑节点。
- `LH_Utils.py`: 处理文本切分与文件存盘节点。

**建议：**
建议配合dynamic prompt插件使用，效果更佳，配置如配图。

-------------------------------------------------------------------------------------
## 📦 Core Features (EN)
Model Loader (GGUF_Loader): A dedicated loader designed for GGUF architectures (e.g., Qwen3), featuring an integrated VRAM Auto-Offload mechanism to optimize memory usage.

AI Interaction (Debug_Chat): Supports dynamic adjustment of core AI parameters such as max_tokens and temperature.

Debug Mode: Analyzes user_prompt based on system_command to output logical reasoning and thought processes, facilitating prompt debugging.

Prompt_Enhance Mode: Generates creative expansions based on raw user materials, following system instructions to produce rich, detailed visual descriptions.

Output Splitter (Output_Splitter): An automated parser that leverages specific identifiers (SECTION 1 / SECTION 2 / SECTION 3) to extract image prompts, LoRA training tags, and custom filenames from the AI response.

All-In-One Saver (All-In-One_Saver): A one-click solution for synchronized data preservation, saving the generated image, a standardized tag file (.txt for LoRA training), and a detailed prompt log (_log.txt).
