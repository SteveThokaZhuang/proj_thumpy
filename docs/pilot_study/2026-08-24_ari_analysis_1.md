这两个指标是本次实验**最核心的发现**，它们揭示了合成数据“形似而神不似”的本质。

简单来说：
- **Generation ARI (0.31)**：衡量声学特征与**“LLM 原本想生成的意图”**的一致性。（结果：中等一致，说明渲染引擎把“样子”做出来了）
- **Realized ARI (0.08)**：衡量声学特征与**“物理上真正同时发声的事实”**的一致性。（结果：几乎不一致，说明渲染引擎把“灵魂”弄丢了）

下面是详细的技术定义和解读：

---

## 1. Generation Label (生成标签) vs. Realized Label (实现标签)

在 Behavior-SD 数据集中，同一个重叠事件有两种不同的“真值”定义：

### A. Generation Label (生成标签)
- **来源**：数据集构建时，LLM 根据上下文语义决定的**意图标签**。
- **定义逻辑**：
  - LLM 判断：“这里需要插入一个 Backchannel 表示支持”或“这里需要插入一个 Interruption 表示打断”。
  - 元数据记录：`behaviors[].backchannels > 0` → 标签为 **BC**；`behaviors[].interruptions > 0` → 标签为 **Int**。
- **物理含义**：代表了**“设计者的意图”**。即：“我想让你听到一个 BC”。
- **时间戳**：通常来自 LLM 规划的理想时间窗口（嵌套在 `backchannels[]` 数组中）。

### B. Realized Label (实现标签/声学事实标签)
- **来源**：本次实验中，通过**声道级 RMS 能量分析**客观计算出的**物理事实**。
- **定义逻辑**（与 `verify_backchannel.py` 同口径）：
  - 提取重叠时间段内，**说话人 A 声道**和**说话人 B 声道**的能量。
  - **判定规则**：
    - 如果 **双声道同时活跃** (RMS > 阈值) → 标签为 **Real-Overlap (真正重叠)**。
    - 如果 **只有宿主声道活跃**，听众声道静音 → 标签为 **Silent/Gap (无重叠)**。
  - **映射到 BC/Int**：
    - 如果 Generation 说是 BC，但物理上双声道未同时活跃 → Realized Label = **None (或 Pseudo-BC)**。
    - 如果 Generation 说是 BC，且物理上双声道同时活跃 → Realized Label = **BC**。
- **物理含义**：代表了**“耳朵实际听到的物理现实”**。即：“实际上两个人有没有同时说话”。

---

## 2. 两个 ARI 指标的计算逻辑

实验中，我们提取了同一组声学特征（能量比、F0 相关性等），然后分别计算它们与这两种标签的聚类一致性（ARI）。

### 📊 Generation ARI = 0.31
- **计算公式**：`ARI(声学特征聚类结果, Generation Label)`
- **含义**：声学特征能否区分出“LLM 意图是 BC”还是“LLM 意图是 Int”的事件？
- **解读**：
  - **0.31 (中等一致)** 说明：当 LLM 意图是 BC 时，渲染引擎通常会生成一个**听起来像 BC 的声音片段**（短促、音高下降、能量低）。
  - **结论**：渲染引擎在**局部声学特征（Timbre/Prosody）**上是成功的。它知道“BC 应该听起来什么样”。

### 📉 Realized ARI = 0.08
- **计算公式**：`ARI(声学特征聚类结果, Realized Label)`
- **含义**：声学特征能否区分出“物理上真正双声道重叠”还是“物理上无重叠”的事件？
- **解读**：
  - **0.08 (几乎随机)** 说明：声学特征**无法区分**哪些事件是真正重叠的，哪些是假重叠。
  - **原因**：因为大部分 Generation Label 为 BC 的事件，在物理上**并没有真正重叠**（被渲染进了宿主的停顿里）。
  - **结论**：渲染引擎在**全局交互本质（Interaction）**上是失败的。它虽然生成了“像 BC 的声音”，但**放错了位置**（没放在重叠区，而是放在了停顿区）。

---

## 3. 通俗类比：画一只“老虎”

想象一个画家（渲染引擎）接到指令（Generation Label）画一只“老虎”：

| 维度 | Generation ARI (0.31) | Realized ARI (0.08) |
| :--- | :--- | :--- |
| **指令** | “画一只老虎” | “画一只**正在捕猎**的老虎” |
| **画家表现** | 画家画了一只猫科动物，有**条纹、尖牙、尾巴**（局部特征像老虎）。 | 但画家把老虎画在了**笼子里睡觉**，而不是在捕猎（全局交互状态不对）。 |
| **评测** | **问**：这看起来像老虎吗？<br>**答**：像！（ARI 高） | **问**：这只老虎在捕猎吗？<br>**答**：完全不像，它在睡觉。（ARI 低） |
| **对应 Behavior-SD** | **问**：这个声音片段听起来像 BC 吗？<br>**答**：像！（短促、音高降） | **问**：这个 BC 是和宿主同时发声的吗？<br>**答**：不是，宿主当时停顿了。 |

---

## 4. 为什么这个发现至关重要？

这个对比直接证明了你的核心论点：

> **Behavior-SD 的渲染失真，不是“声学特征”的失真，而是“交互时机”的失真。**

1. **如果 Generation ARI 也低 (如 0.08)**：说明渲染引擎连 BC 的**声音**都渲染错了（比如把 BC 渲染成了长句）。那问题出在 TTS 或 LLM 文本生成。
2. **现状 (Gen=0.31, Real=0.08)**：说明渲染引擎**声音渲染对了**，但**时机渲染错了**。
   - 它把本该重叠的 BC，移到了宿主的停顿里（为了规避 TTS 重叠合成的技术难点）。
   - 这导致**标称的“重叠事件”在物理上变成了“填空事件”**。

**论文金句**：
> "The high Generation ARI (0.31) confirms that the rendering engine successfully synthesizes the **local acoustic properties** of backchannels (e.g., short duration, falling pitch). However, the near-zero Realized ARI (0.08) reveals a catastrophic failure in synthesizing the **global interaction dynamics**: the system places these 'backchannel-shaped' sounds into silent gaps rather than overlapping with the host speech, effectively turning 'concurrent feedback' into 'filler tokens'."

---

## 5. 总结

| 指标 | 标签来源 | 物理意义 | 数值 | 结论 |
| :--- | :--- | :--- | :--- | :--- |
| **Generation ARI** | LLM 意图 (元数据) | "我想生成 BC" | **0.31** | 渲染引擎**形似**：声音听起来像 BC |
| **Realized ARI** | 声道级 RMS (物理事实) | "实际上双声道重叠" | **0.08** | 渲染引擎**神离**：实际上没重叠 |

**这组对比是你论文中最强有力的证据**，它比单纯的“重叠时长缩短”更深刻地揭示了合成数据的缺陷：**它学会了 BC 的“声音”，却没学会 BC 的“行为”。**