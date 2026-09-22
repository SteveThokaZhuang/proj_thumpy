# 扩展论文故事：全双工评测的"标签从哪来"问题

> **日期**：2026-08-25
> **背景**：ARI 实验链（主实验 + A1/A2/B/C/D/E/F，见
> [主报告](2026-08-23_ari_results.md) §3.5）发现合成数据标注存在"静音捷径"、
> "重叠不写入韵律"、"韵律只随文本模板"三个结构性问题。本文档把发现串成扩大的
> 论文故事：**评测指标的标签层级审计 + 模型化标注器**。
> 三个第三方仓库的调研结论：FD-Bench（指标 = VAD 时间戳区间规则）、
> X2-Turn（4B，80ms/帧 6 类状态含 backchannel+概率）、SoulX-Duplug（0.6B，
> 160ms/块 5 类用户状态含 backchannel，需 training-code 分支才能拿到）。

---

## 1. 核心命题

**评测全双工模型的行为指标（打断/附和/轮转），标签从哪来？** 现在只有三条路径，
而我们恰好把每条路径的失效模式都测出来了：

| 层级 | 标签源 | 代表 | 已测失效模式 |
|---|---|---|---|
| L1 生成端 GT | 合成数据的 metadata | Behavior-SD、FD-Bench 的 GPT 对话 | 完美但捷径化：标签-声学一致性 100% 由"对方声道静音"承载（B）；重叠状态不写入 token 韵律（A1/A2）；韵律只是词条模板（E） |
| L2 声学派生 | VAD/ASR 时间戳 + 阈值规则 | CANDOR backbiter、FD-Bench 的 Silero VAD 区间规则 | 噪声大：无监督结构上限 ≈0.07（D）；标签空间坍塌（realized-ARI 0.08，A1）；阈值敏感（A2） |
| L3 模型化标注器 | 全双工话轮检测插件 | X2-Turn、SoulX-Duplug | **未测——本文档的核心新实验** |

**论文命题**：全双工评测的可信度 = 所用标签层级的可信度。任何声称测"打断处理"的
指标，都该明确它吃的是哪一层标签，并接受对应失效模式的审计。

## 2. 两个 idea 与已有发现的接点

### Idea 1：标注问题 → benchmark 指标设计（FD-Bench 审计）

调研确认 FD-Bench 的标签是 **L2**（输入/输出两条音频流各自跑 Silero VAD，再用
硬编码区间规则判"打断成功/失败/噪声"；无声道分离、无混音重叠检测）。这带来三个
可直接用我们的发现批判的点：

1. **它回避了真实重叠**：输入输出是两条独立流，模型行为相对"合成输入的轮边界"
   被度量——真实混音中的重叠处理从未被测到。我们项目的声道级 realized 分析
   （verify_backchannel + A1/A2）正是它缺的那一环。
2. **干净输入分布**：合成单声道 TTS + 受控静音间距（easy/med/hard）——模型面对的是
   人造干净分布。我们的 C 实验已量化干净→真实分布的迁移折扣（F1 0.72→0.45，
   去能量比归零）。FD-Bench 的分数在真实录音上会有同款折扣，且无人报告过。
3. **标签空间坍塌**：VAD 时间戳区间规则在真实音频上会遭遇与 realized 标签同样的
   问题（A1 已证：阈值裁决无法区分"真重叠"与"轮边界邻接"），而 FD-Bench 的
   0.5s/2.5s 阈值从未做过敏感性分析（我们的 A2 方法学可直接搬用）。

**新实验 E1/E2**（见 §4）把这三个批判做成数据。

### Idea 2：LLM 辅助标注 → 插件标注器（X2-Turn / SoulX-Duplug）

两个插件恰好覆盖 L3 层级，且**互补**：

| | X2-Turn | SoulX-Duplug |
|---|---|---|
| 输出 | 80ms/帧，6 类（idle/noidle/speaking/turn_end/**backchannel**/uncertain）+ 各类概率 | 160ms/块，5 类（idle/nonidle/speak/**backchannel**/wait），需 training-code 分支 |
| 输入 | 16k mono，文件/波形，离线整段推理 | 16k mono 流式（1.16s 窗口），单说话人单流 |
| 显存 | ≥24GB（4B） | ~4-6GB（0.6B） |
| 依赖 | 自身 ASR 双头（两遍推理） | 级联外部 ASR（Paraformer/SenseVoice） |
| 限制 | backchannel 为帧级语义；无官方评测脚本 | 只预测"用户"侧状态，标注双人对话需**按声道拆两条流** |

**关键的独享红利**：我们手上有 **Behavior-SD 的完美 GT**——这是唯一能对标注器做
金标准校准的地方；且两个数据集都是**真立体声每声道一人**——正好满足 SoulX-Duplug
"单说话人单流"的结构要求。故事闭环：

```
合成 GT（完美）──校准──▶ 插件标注器 ──应用──▶ 真实数据/真实模型输出
      ▲                                        │
      └───────── 验证：ARI 管线重跑 ◀───────────┘
```

如果校准后的插件标签在 CANDOR 上把 ARI 从 0.058（L2）拉到显著更高（接近合成数据的
标注质量），就证明了 **L3 能把真实数据的标注质量提升到接近 L1**——这是评测真实
模型可行性的直接证据，也是论文的第三贡献。

## 3. 论文结构（三幕）

**Act 1 — 标签层级审计**（已完成 + E1 补强）：三种标签源的声学一致性、捷径构成、
失效模式。核心数据：主实验 ΔARI=−0.25、B 消融、A1/A2、C/D/E。
**Act 2 — benchmark 指标审计**（E1/E2）：FD-Bench 的 L2 指标在被审计时的表现；
提出"**无捷径对照协议**"：每个 turn-taking 指标须附带 (a) 无能量比消融下界、
(b) 真实分布压力测试（串扰混音）、(c) 标签源一致性矩阵。
**Act 3 — 模型化标注器**（E3/E4/E5）：合成 GT 校准 → 真实数据标注 → 标注质量
闭环验证；双插件交叉验证出高置信子集。

**论文一句话**：*"Full-duplex benchmarks measure the label they inherit, not the
behavior they claim: we audit three annotation layers, show the synthetic shortcut
and the acoustic-label noise, and close the gap with a synthetic-GT-calibrated
model-as-annotator pipeline."*

## 4. 新实验清单（按依赖顺序）

| # | 实验 | 输入 | 产出 | 可行性 |
|---|---|---|---|---|
| **E1** 标签源一致性矩阵 | CANDOR 同一批事件的三套标签：backbiter(L2-ASR) vs Silero VAD 区间规则（复刻 FD-Bench 逻辑）vs 声道 realized | 三种标签的两两一致率/κ + 各类冲突分布 | 事件表已有，写复刻 VAD 区间规则的脚本即可，~1 天 | 无需新资源 |
| **E2** FD-Bench 指标审计 | 在 Behavior-SD 上复刻 FD-Bench 的 SIR/EIR/NIR 区间规则 → 与 metadata GT 与 realized 对照；再加"串扰混合"压力测试（人为把两声道路标混合后重跑区间规则） | FD-Bench 指标在"干净 vs 串扰"下的漂移量 | 事件表已有，区间规则逻辑已由 agent 摘出（0.5s/2.5s/5.5s 阈值），~1-2 天 | 无需新资源 |
| **E3** 标注器金标准校准 | X2-Turn 跑 Behavior-SD 验证集子集（~200 文件，切段推理）→ 帧级状态 → 事件级映射（backchannel 帧聚类 vs BC 事件窗口）→ 帧级/事件级 precision/recall | X2-Turn 的校准曲线 + 最优置信阈值 | 需 ≥24GB 卡（gpu02 有 4×4090D）+ HF 权重下载；~2-3 天 | 权重需 HF 下载（镜像可行） |
| **E4** 真实数据标注 | 校准后的 X2-Turn 标注 CANDOR 子集（~300 会话）→ 插件派生标签 → 重跑 ARI 管线 | **插件标签的 ARI vs 0.058（L2）**——故事成败关键数字 | 依赖 E3；推理时长取决于子集规模 | 同上 |
| **E5** 双插件交叉验证 | SoulX-Duplug（training-code 分支）按声道拆流标注同一 CANDOR 子集 → 与 X2-Turn 一致性 + 分歧分析 → 高置信子集 | 双标注器一致率 + 共识标签集 | SoulX 依赖链长（FunASR/modelScope 权重），~2-3 天 | 需装 FunASR 环境 |
| **E6** 论文收口 | 全部结果 → 标签层级框架 + benchmark 协议建议 + 校准标注数据发布说明 | 论文章节 + 建议稿 | — | — |

**优先级**：E1/E2 无新资源、直接强化 Act 1/2，先做；E3/E4 是 Act 3 的成败手，
需 GPU+权重，与 E1/E2 并行准备；E5 视 E4 结果决定。

## 5. 风险与开放问题

1. **X2-Turn 的帧级 backchannel ≠ 事件级 BC 标签**：帧级→事件级映射（连续帧聚类、
   置信阈值）是设计空间，映射不当会低估标注器；E3 需同时报告多组映射。
2. **标注器受训数据与 CANDOR 的分布差**：两个插件主要面向电话/近场语音，CANDOR 是
   视频通话录音（串扰）——E4 的标注质量可能受此拖累，需要先报"域内（Behavior-SD）
   vs 域外（CANDOR）"两组校准数字。
3. **ASR 依赖链**：两插件状态预测都吃 ASR 文本，CANDOR 的 64kbps MP3 + 串扰会让
   ASR 变差进而污染状态标签——E4 的失败模式要能归因（对比插件的 ASR WER）。
4. **"接近合成数据的标注质量"的验收标准**：需要预设 E4 的量化目标（例如插件标签
   ARI ≥ 0.15 即显著优于 L2 的 0.058；或按 C 迁移的 F1 口径）。
5. FD-Bench 审计的**措辞边界**：对方是已发表工作，Act 2 的写法应定位为
   "标注层级框架 + 协议建议"，避免攻击性表述。

## 6. 与团队文档的关系

- 本文档为故事总纲；每个实验完成后在 docs/pilot_study/ 写独立报告，更新
  [实验计划](2026-08-25_ari_followup_plan.md)（新增 E1-E6 章节）。
- 数据产物沿用 `pilot_study/real_data/results/analysis/ari/`；插件推理产物放
  `pilot_study/real_data/results/annotator/`。
- 调研细节（三个仓库的接口/阈值/限制）见本条消息的三个 agent 报告，写实验脚本前
  建议再读对应源码确认。
