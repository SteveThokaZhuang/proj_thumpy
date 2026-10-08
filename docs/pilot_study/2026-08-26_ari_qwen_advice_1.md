# 全双工评测论文：下一步行动完整指南

**当前时间**：2026-08-26 12:22  
**目标**：在 2-3 周内完成 E1-E4 核心实验，产出论文初稿  
**投稿目标**：Interspeech 2027 / ICASSP 2027（截稿通常在 9-10 月）

---

## 📅 第一阶段：零成本快速产出（第 1-3 天）

### 任务 1.1：E1 标签源一致性矩阵（优先级：🔴 最高）

**目标**：量化 CANDOR 上三种标签（L2-ASR / L2-VAD 规则 / L2-realized）的一致性上限

**具体步骤**：

```bash
# 1. 创建实验脚本（基于已有事件表）
cd pilot_study/scripts
touch e1_label_consistency.py
```

**脚本核心逻辑**：
1. 加载 CANDOR 事件表（`backbiter` BC 事件 + `audiophile` Int/None 事件）
2. 复现 FD-Bench 的 VAD 区间规则：
   - 对每个事件窗口，提取输入/输出声道的 Silero VAD 时间戳
   - 应用 0.5s/2.5s/5.5s 阈值规则（从 agent 报告中提取的逻辑）
   - 生成 L2-VAD 标签
3. 计算两两一致性：
   - backbiter (L2-ASR) vs VAD 规则 → Cohen's κ
   - backbiter vs realized → κ
   - VAD 规则 vs realized → κ
4. 输出冲突分布桑基图（Sankey Diagram）

**预期产出**：
- 表格：3×3 一致性矩阵（κ值 + 95% CI）
- 图表：桑基图展示标签流转
- **关键洞察**：若 κ<0.4，说明 L2 标签本身就不稳定，直接支撑 Act 1

**时间估算**：1 天（事件表已有，只需写 VAD 规则复现逻辑）

---

### 任务 1.2：E2 FD-Bench 指标审计（优先级：🔴 最高）

**目标**：量化 FD-Bench 风格指标在"干净 vs 串扰"下的漂移

**具体步骤**：

```bash
cd pilot_study/scripts
touch e2_fdbench_audit.py
```

**实验设计**：
1. **基线**：在 Behavior-SD 验证集（干净单声道）上复现 SIR/EIR/NIR 指标
   - 输入：原始单声道 TTS 音频
   - 输出：SIR（成功打断率）、EIR（早期打断率）、NIR（噪声打断率）
2. **压力测试**：人为制造串扰混合
   - 随机选取两个无关会话的音频，按 1:1 能量比混合
   - 保持标签不变（仍用原 metadata）
   - 重跑 SIR/EIR/NIR
3. **对照**：用 realized 标签重跑上述两步

**预期产出**：
- 表格：干净 vs 串扰下的指标漂移量（ΔSIR, ΔEIR, ΔNIR）
- 图表：阈值敏感性曲线（横轴=VAD 阈值，纵轴=SIR）
- **关键数字**：若 ΔSIR > 0.2，直接证明 FD-Bench 指标在真实分布上不可靠

**复用 ARI 数据**：
- 直接引用 ARI 报告 §3.2 的"能量比偏移 d=1.57"作为理论预测
- 引用 C 实验的迁移折扣（F1 0.72→0.45）作为预期漂移量级

**时间估算**：1-2 天（区间规则逻辑已提取，只需批量推理）

---

### 任务 1.3：论文草稿 Act 1 & Act 2 初稿（并行进行）

**目标**：在实验跑的同时，先写文字部分

**写作清单**：
- [ ] **§1 Introduction**：用 ΔARI=-0.251 作为开篇钩子
- [ ] **§3.1 标签层级框架**：画 L1/L2/L3 框架图（参考本文档 §1）
- [ ] **§3.2 ARI 主结果**：直接插入 ARI 报告的 Fig 1-3
- [ ] **§3.3 消融分析**：插入 ARI 报告 §3.5 的 B/C/D/E 实验摘要
- [ ] **§4.1 FD-Bench 审计**：预留 E1/E2 图表位置，先写方法论

**关键句子模板**：
> *"Our analysis reveals that synthetic data exhibits 5× higher acoustic-label consistency than human conversation (ARI 0.31 vs 0.06), but 84% of this advantage is carried by a silence shortcut: removing the energy ratio feature collapses the ARI to 0.05."*

---

## 🖥️ 第二阶段：GPU 实验准备（第 2-4 天，与第一阶段并行）

### 任务 2.1：X2-Turn 环境搭建（优先级：🟠 高）

**硬件确认**：
- 目标机器：`gpu02`（4×4090D，每卡 24GB 显存）
- 显存需求：X2-Turn 4B 模型需 ≥24GB（单卡可跑）[X2-Turn paper](https://arxiv.org/html/2608.10878v2)

**具体步骤**：

```bash
# 1. 创建独立 conda 环境
conda create -n x2turn python=3.10 -y
conda activate x2turn

# 2. 克隆仓库（确认代码结构）
git clone https://github.com/X-Square-Robot/X2-Turn.git
cd X2-Turn

# 3. 安装依赖（参考 requirements.txt）
pip install -r requirements.txt
# 关键依赖：torch, transformers, torchaudio, silero-vad

# 4. 下载权重（HuggingFace 或镜像）
# 若 HF 下载慢，用 modelscope 镜像或团队内部缓存
huggingface-cli download X-Square-Robot/X2-Turn-4B --local-dir ./models/x2turn-4b
```

**验证脚本**（minimal_demo.py）：
```python
import torch
from x2turn import X2TurnModel

model = X2TurnModel.from_pretrained("./models/x2turn-4b")
model.to("cuda:0")

# 单文件推理测试（16k mono, 5s 音频）
audio, sr = torchaudio.load("test_5s.wav")
assert sr == 16000
audio = audio.to("cuda:0")

with torch.no_grad():
    states = model.infer(audio)  # 输出：80ms/帧，6 类状态+概率

print(f"Input: {audio.shape[-1]/16000:.2f}s, Output: {len(states)} frames")
# 预期：5s → ~62 帧
```

**预期问题与解决**：
- **问题 1**：HF 下载超时 → 用 `HF_ENDPOINT=https://hf-mirror.com`
- **问题 2**：显存 OOM → 用 `--batch-size 1` 或 `--chunk-size 30s`
- **问题 3**：依赖冲突 → 优先保证 `torch==2.1.0` + `transformers>=4.35`

**时间估算**：1 天（含调试）

---

### 任务 2.2：SoulX-Duplug 环境搭建（优先级：🟡 中，视 E4 结果决定）

**特点**：0.6B 小模型（4-6GB 显存），但依赖链长（FunASR + Paraformer）

```bash
# 1. 克隆仓库
git clone https://github.com/Soul-AILab/SoulX-Duplug.git
cd SoulX-Duplug

# 2. 安装 FunASR（关键依赖）
pip install funasr>=1.1.0

# 3. 下载权重（ModelScope 更快）
modelscope download Soul-AILab/SoulX-Duplug-0.6B --local_dir ./models

# 4. 验证（需按声道拆分立体声）
# 注意：SoulX 只预测"用户侧"状态，双人对话需拆成两条单声道流
```

**时间估算**：0.5 天（若 FunASR 安装顺利）

---

## 🚀 第三阶段：核心实验 E3/E4（第 5-10 天）

### 任务 3.1：E3 标注器金标准校准（优先级：🟠 高）

**目标**：在 Behavior-SD 验证集上校准 X2-Turn，找到最优置信阈值

**实验设计**：

```bash
cd pilot_study/scripts
touch e3_annotator_calibration.py
```

**步骤**：
1. **数据准备**：
   - 选取 Behavior-SD 验证集子集（200 文件，覆盖 BC/Int/None 均衡）
   - 提取每文件的 GT 标签（metadata + realized 双版本）
2. **推理**：
   - 用 X2-Turn 逐文件推理，输出 80ms/帧的 6 类状态概率
   - 保存为 `frame_states.parquet`（列：`file_id, frame_idx, state, prob`）
3. **帧级→事件级映射**（关键设计空间）：
   - **策略 A**：连续 N 帧 > 阈值 → 合并为一个事件
   - **策略 B**：峰值检测（局部最大 prob）
   - **策略 C**：滑动窗口投票（window=5 帧，majority vote）
4. **评估**：
   - 帧级：precision/recall/F1（6 类多分类）
   - 事件级：与 GT 事件窗口对齐（IoU>0.5 视为匹配）
   - 输出：校准曲线（横轴=置信阈值，纵轴=F1）

**预期产出**：
- 图表：校准曲线（3 种映射策略对比）
- 数字：最优阈值（如 `prob>0.65`）+ 对应 F1
- **关键洞察**：若帧级 F1>0.7 但事件级 F1<0.5，说明映射策略是瓶颈

**时间估算**：2-3 天（推理耗时 + 映射策略调优）

---

### 任务 3.2：E4 真实数据标注（优先级：🔴 成败手）

**目标**：用校准后的 X2-Turn 标注 CANDOR，重跑 ARI 管线

**实验设计**：

```bash
cd pilot_study/scripts
touch e4_real_data_annotation.py
```

**步骤**：
1. **数据准备**：
   - 选取 CANDOR 子集（300 会话，覆盖不同场景/噪声水平）
   - 按声道拆分立体声（`channel_map.json` 已有）
2. **推理**：
   - 用 E3 最优阈值跑 X2-Turn
   - 输出：每声道的帧级状态 → 事件级标签
3. **重跑 ARI**：
   - 复用 `ari_analyze.py`，替换标签源为 X2-Turn 预测
   - 计算：三分类 ARI（K=3）、BC-vs-rest ARI
4. **归因分析**（关键！）：
   - 计算 ASR WER（若 X2-Turn 输出文本）
   - 分析：WER vs ARI 的相关性（若强相关，说明瓶颈在 ASR）
   - 对照：去能量比后的 ARI（证明学到非捷径特征）

**验收标准**：
| 指标 | L2 基线 | E4 成功线 | E4 理想线 |
|---|---|---|---|
| 三分类 ARI | 0.058 | ≥0.10 | ≥0.15 |
| BC-vs-rest ARI | 0.048 | ≥0.08 | ≥0.12 |
| 去能量比 ARI | ≈0.00 | ≥0.03 | ≥0.05 |

**若失败（ARI<0.10）的应对**：
- **方案 A**：转向"域差异分析"故事（证明 CANDOR 与训练集分布差是核心挑战）
- **方案 B**：用 Behavior-SD GT 微调 X2-Turn 的分类头（Probe 实验）
- **方案 C**：报告"高置信子集"（X2-Turn prob>0.9 的片段）的 ARI，证明潜力

**时间估算**：3-4 天（300 会话推理 + ARI 重跑）

---

## 📝 第四阶段：论文收口（第 11-15 天）

### 任务 4.1：E5 双插件交叉验证（可选，视 E4 结果）

**目标**：用 SoulX-Duplug 验证 X2-Turn 的高置信预测

**步骤**：
1. 对同一 CANDOR 子集跑 SoulX-Duplug（按声道拆流）
2. 计算两标注器一致率（κ值）
3. 提取"共识标签集"（两模型 prob 均>0.8 的事件）
4. 在共识集上重跑 ARI（预期更高）

**时间估算**：2 天（若 E4 成功则做，否则跳过）

---

### 任务 4.2：论文完整草稿（优先级：🔴 最高）

**结构建议**（8-10 页，双栏）：

| 章节 | 内容 | 图表 |
|---|---|---|
| **Abstract** | ΔARI=-0.251 + 84% 捷径 + L3 提升 | — |
| **1. Intro** | 全双工评测危机 + 三层框架 + 贡献 | Fig 1（框架图） |
| **2. Related Work** | FD-Bench, X2-Turn, 话轮检测 | — |
| **3. Label Taxonomy** | L1/L2/L3 定义 + 失效模式 | — |
| **4. Audit Study** | ARI 主结果 + 消融（B/C/D/E） | Fig 2-4（ARI 对比/Bootstrap/消融） |
| **5. Benchmark Audit** | E1/E2 结果（FD-Bench 审计） | Fig 5-6（一致性矩阵/阈值曲线） |
| **6. Model Annotator** | E3/E4 结果（校准 + 真实标注） | Fig 7-8（校准曲线/L3 ARI 对比） |
| **7. Discussion** | 局限 + 协议建议 | Table 1（无捷径对照协议） |
| **8. Conclusion** | 总结 + 未来工作 | — |

**关键图表清单**（共 8 图 1 表）：
1. Fig 1: L1/L2/L3 框架图
2. Fig 2: ARI 对比柱状图（CANDOR 0.058 vs Behavior-SD 0.309）
3. Fig 3: ΔARI Bootstrap 分布
4. Fig 4: 特征消融瀑布图
5. Fig 5: 标签一致性桑基图（E1）
6. Fig 6: FD-Bench 指标漂移（E2）
7. Fig 7: X2-Turn 校准曲线（E3）
8. Fig 8: L3 vs L2 ARI 对比（E4，成败图）
9. Table 1: "无捷径对照协议"建议

**时间估算**：3-4 天（与实验并行写文字部分）

---

## 📋 每日检查清单

### 第 1 天
- [ ] E1 脚本完成，跑通 CANDOR 前 10 会话
- [ ] X2-Turn 仓库克隆，依赖安装完成
- [ ] 论文 §1-§3 草稿完成

### 第 2 天
- [ ] E1 全量跑完，产出桑基图
- [ ] E2 脚本完成，基线（干净）推理完成
- [ ] X2-Turn minimal demo 跑通（单文件）

### 第 3 天
- [ ] E2 压力测试（串扰混合）完成
- [ ] 论文 §4（ARI 审计）草稿完成
- [ ] X2-Turn 权重下载完成

### 第 4-5 天
- [ ] E3 推理完成（200 文件 Behavior-SD）
- [ ] E3 映射策略调优，产出校准曲线
- [ ] 论文 §5（FD-Bench 审计）草稿完成

### 第 6-8 天
- [ ] E4 推理完成（300 会话 CANDOR）
- [ ] E4 ARI 重跑，产出 Fig 8
- [ ] 若 ARI≥0.10：开始写 §6；若失败：启动应对方案

### 第 9-10 天
- [ ] E5（可选）完成
- [ ] 论文 §6-§8 草稿完成
- [ ] 全文通读，补充参考文献

---

## ⚠️ 风险应对预案

| 风险 | 触发条件 | 应对方案 |
|---|---|---|
| **X2-Turn 推理 OOM** | 单卡 24GB 不够 | 改用 `--chunk-size 10s` 分段推理，或换 40GB 卡（gpu01） |
| **E4 ARI < 0.08** | L3 未显著超越 L2 | 转向"域差异分析"故事，报告 Behavior-SD 域内结果作为潜力证明 |
| **FD-Bench 代码不开源** | 无法复现评判逻辑 | 在论文中声明"复现典型 L2 规则"，引用 Silero VAD+ 阈值规则文献 |
| **ASR WER 过高** | CANDOR 64kbps 导致 ASR 崩溃 | 增加"Oracle ASR"对照（用 GT 文本），分离 ASR 误差与状态预测误差 |

> 🔴 **2026-10-08 更正（不改上表原文，保持建议记录原样）**：上表「**FD-Bench 代码不开源**」**不成立** ——
> 本地 `third_party/FD-Bench/` 有完整 `benchmarking.py`（E2 审计正是复刻它跑的）。该触发条件的前提已消失；
> 「复现典型 L2 规则」的应对方案本身仍有效。出处：`docs/related_works/fd_activities.md` §7 D。

---

## 🎯 最终交付物

1. **实验代码**：`pilot_study/scripts/e1_*.py` ~ `e5_*.py`
2. **数据产物**：`pilot_study/real_data/results/annotator/`（X2-Turn 预测结果）
3. **论文草稿**：`docs/paper/full_duplex_label_audit_v1.tex`
4. **可视化图表**：8 图 1 表（可直接插入论文）
5. **复现指南**：`README.md`（含环境配置 + 命令）

---

**立即行动**：
```bash
# 1. 创建实验目录
mkdir -p pilot_study/scripts/{e1,e2,e3,e4}
mkdir -p pilot_study/real_data/results/annotator/{x2turn,soulx}

# 2. 开始 E1 脚本
cd pilot_study/scripts
vim e1_label_consistency.py  # 从加载事件表开始

# 3. 并行启动 X2-Turn 环境
conda create -n x2turn python=3.10 -y
```

**祝实验顺利！这个故事线有潜力成为全双工评测领域的标志性工作。** 🚀