### 🧠 核心逻辑流程图 (Core Logic Flow)

```mermaid
graph TD
    Start[开始: 运行 LH_AIChat 节点] --> CheckImage{是否连接 Image?<br/>(图像输入)}

    %% 视觉模式分支
    CheckImage -- 是 --> VisionMode[**隐形反推模式**<br/>(解析图片)]
    VisionMode --> IgnoreMaterial[忽略 user_material (用户素材)]
    IgnoreMaterial --> CheckInstVision{是否反推模式?<br/>(自定义指令)}
    
    CheckInstVision -- 是 --> UseUserInstVision[使用用户 instruction (指令)<br/>处理图片]
    CheckInstVision -- 否 --> UseBuiltinVision[使用 **内置反推指令**<br/>(描述主体, 细节, >300字)]
    
    UseUserInstVision --> AddTagsVision[添加分段标签]
    UseBuiltinVision --> AddTagsVision
    
    AddTagsVision --> OutputVision[生成内容]

    %% 文本模式分支
    CheckImage -- 否 --> CheckMode{检查 chat_mode<br/>(聊天模式)}
    
    %% 扩写模式分支
    CheckMode -- Enhance_Prompt<br/>(扩写) --> EnhanceMode[**扩写模式**<br/>(创意扩充)]
    EnhanceMode --> ProcessMaterialEnhance[处理 user_material (用户素材)]
    ProcessMaterialEnhance --> CheckInstEnhance{是否有自定义<br/>instruction (指令)?}
    
    CheckInstEnhance -- 是 --> UseUserInstEnhance[使用用户 instruction (指令)<br/>处理素材]
    CheckInstEnhance -- 否 --> UseBuiltinEnhance[使用 **内置扩写指令**<br/>(扩写细节, 风格, >300字)]
    
    UseUserInstEnhance --> AddTagsEnhance[添加分段标签]
    UseBuiltinEnhance --> AddTagsEnhance
    AddTagsEnhance --> OutputEnhance[生成内容]

    %% Debug模式分支
    CheckMode -- Debug_Chat<br/>(调试) --> DebugMode[**Debug 模式**<br/>(分析原因)]
    DebugMode --> ProcessMaterialDebug[处理 user_material (用户素材)]
    ProcessMaterialDebug --> CheckInstDebug{是否有自定义<br/>instruction (指令)?}
    
    CheckInstDebug -- 是 --> UseUserInstDebug[使用用户 instruction (指令)]
    CheckInstDebug -- 否 --> UseBuiltinDebug[使用 **内置分析指令**<br/>(分析上轮结果)]
    
    UseUserInstDebug --> OutputDebug[生成内容]
    UseBuiltinDebug --> OutputDebug

    %% 输出逻辑
    OutputVision --> CheckSwitches{检查开关:<br/>enable_tag (标签)<br/>enable_filename (文件名)}
    OutputEnhance --> CheckSwitches
    
    CheckSwitches -- 根据开关处理 --> FormatOutput[按顺序输出:<br/>1. Prompt (主要内容)<br/>2. Tags (标签 - 如开启)<br/>3. Filename (文件名 - 如开启)]
    
    OutputDebug --> ForceIgnoreSwitches[**强制忽略开关**<br/>(无 Tags/Filename)]
    ForceIgnoreSwitches --> FinalOutput[最终输出 (Splitter识别)]
    FormatOutput --> FinalOutput

    FinalOutput --> End[结束]
```
