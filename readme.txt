🎨 LoraHelper 插件系统使用指南 (V5.0)
本插件是一套专为 ComfyUI 设计的自动化模型调用与素材整理工具。通过将大语言模型（LLM）与图片保存机制结合，实现从“提示词优化”到“数据自动分类存盘”的全流程自动化。

1. 📂 环境部署与模型存放
  物理路径：将 GGUF 格式的大模型存放在 ComfyUI/models/llm/ 文件夹下。
2. 🧠 核心节点功能说明
① LoraHelper_Loader (模型加载器)
    n_gpu_layers：设置显卡加速层数（-1 为全量加速）。
    n_ctx：模型上下文长度，建议根据显存大小设置在 4096-8192 之间。

② LoraHelper_Chat (导演节点)
    User_Prompt (素材输入)：上方文本框。用于存放原始描述、灵感或抓取的素材内容。
    System_Prompt (创作指令)：下方文本框。用于定义 AI 的角色（如“提示词专家”）以及输出的格式要求。
    动态控制：支持实时调节 max_tokens（最大生成长度）、temperature（随机性）以及 seed（生成种子）。

③ LoraHelper_Splitter (智能切分器)
    运作逻辑：自动解析 AI 生成的长文本。

    精准截取：
    识别 SECTION 1 作为生图提示词。
    识别 SECTION 2 作为训练标签。
    识别 [方括号] 内内容作为文件名。
    容错机制：若 AI 未按特定格式输出，节点会自动抓取首个有效段落作为提示词，确保工作流不中断。

④ LoraHelper_Saver (全能存盘器)
    三位一体存盘：每次生成会自动在目标文件夹创建三个关联文件：
    图片文件 (.png)：包含完整元数据的成品图。
    标签文件 (.txt)：由“触发词+AI提取标签”组成，可直接用于 LoRA 训练。
    日志文件 (_log.txt)：保存 AI 创作的原始全文本，方便日后追溯场景、服装等详细细节。
    文件名脱敏：系统会自动清理非法字符，确保在 Windows 环境下文件名合法。

3. 🛠️ 推荐工作流架构
    加载：使用 Loader 挂载 Qwen3 或其他 GGUF 模型。
    创作：在 Chat 节点输入素材，由 AI 进行扩写润色。
    解析：通过 Splitter 将润色后的内容拆分为“生图词”和“归档词”。
    执行：生图完成后，由 Saver 自动完成文件命名、标签提取及分文件夹归档。

📝 文件结构参考 (安装位置)
为保证系统稳定性，建议将代码拆分为以下模块存放：
custom_nodes/LoraHelper/__init__.py —— 插件入口
custom_nodes/LoraHelper/LH_Chat.py —— 逻辑处理模块
custom_nodes/LoraHelper/LH_Utils.py —— 存盘与工具模块