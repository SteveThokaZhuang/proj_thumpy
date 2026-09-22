# A1：类内重叠判别——"神不似"的独立检验

> **日期**：2026-08-25
> **动机**：[qwen 的分析](2026-08-24_ari_analysis_1.md) 用 generation-ARI (0.31) vs realized-ARI (0.08)
> 论证"形似而神不似"，但 realized-ARI 存在**标签空间坍塌**问题（BC 渲染进停顿与真正的
> None 事件同被标为 realized=None），0.08 部分是标签设计的产物而非纯声学证据。
> A1 用**类内判别**绕开该缺陷：只取同一事件类型（BC 或 Int），判别其 realized 状态。
>
> **脚本**：`pilot_study/scripts/ari_analyze_a1.py`；结果 `real_data/results/analysis/ari/a1_results.json`

---

## 1. 方法与数据

对 Behavior-SD 的 BC 事件与 Int 事件分别做二分类：

- **y=1（真重叠）**：事件窗口内存在双声道同时活跃段（realized = Backchannel 或 Interruption）
- **y=0（停顿内）**：无真实重叠（realized = None）

| 事件类型 | 停顿内 (y=0) | 真重叠 (y=1) | 重叠占比 |
|---|---|---|---|
| BC | 21,947 | 10,901 | 33.2% |
| Int | 3,753 | 6,651 | 63.9% |

即：**2/3 的标称 BC 与 1/3 的标称 Int 实际被渲染进了宿主停顿**（与主报告的 realized×cls 交叉表一致）。

**关键设计——分离"构造性耦合"与"真信号"**：`energy_ratio` 与 realized 标签同源于
"窗口内双方声道的能量"（同一测量的两个函数），二者相关近乎必然。真正非循环的问题是：
**只用事件自身声道的韵律特征（F0 斜率、频谱质心、有声帧占比）能否判别？**
即"渲染在重叠里 vs 停顿里的 BC，token 自身的音色/韵律是否不同"。

指标：重复分层 5 折 CV × 20 种子 的 AUC（均值±std）；单特征 AUC；组间 Cohen's d + bootstrap 95% CI；
K=2 KMeans ARI（平衡抽样）。分类特征沿用主实验 5 特征（不含 duration；dur 仅作描述性对照）。

## 2. 结果

### 2.1 判别力（AUC，随机水平 = 0.50）

| 特征组 | BC | Int |
|---|---|---|
| 全 5 特征 | **0.761 ± 0.005** | **0.730 ± 0.011** |
| 去掉 energy_ratio | 0.580 ± 0.007 | 0.527 ± 0.012 |
| 仅事件声道韵律特征 | 0.579 ± 0.007 | 0.528 ± 0.012 |
| K=2 ARI（平衡） | 0.071 | 0.004 |

单特征 AUC：

| 特征 | BC | Int |
|---|---|---|
| energy_ratio | **0.708** | **0.726** |
| f0_correlation | 0.504 | 0.498 |
| f0_slope | 0.502 | 0.509 |
| spectral_centroid | 0.555 | 0.527 |
| voiced_ratio | 0.550 | 0.507 |

### 2.2 组间差异（Cohen's d，正 = 真重叠组更大；bootstrap 95% CI）

| 特征 | BC | Int |
|---|---|---|
| energy_ratio | **−0.78** [−0.80, −0.77] | **−0.88** [−0.93, −0.83] |
| f0_correlation | +0.03 | +0.02 |
| **f0_slope** | **+0.002** [−0.02, +0.02] | −0.04 [−0.08, 0.00] |
| spectral_centroid | −0.19 [−0.21, −0.17] | −0.10 [−0.14, −0.06] |
| voiced_ratio | +0.18 [−0.16, +0.21] | +0.03 [−0.01, +0.07] |
| dur（描述性） | +0.46 [0.43, 0.48]（0.72s vs 0.62s） | +0.04（3.71s vs 3.59s） |

## 3. 裁决：对"神不似"的结论

1. **全特征 AUC ≈ 0.75 几乎全部由 energy_ratio 单特征贡献**（0.71/0.73），而 energy_ratio
   与 realized 标签是同一物理量的两个函数——"真重叠时对方声道更响"是构造性事实，
   **不能作为渲染引擎有意识行为的证据**。
2. **非循环检验在随机水平附近**（BC 0.58 / Int 0.53）：事件 token 自身的韵律——尤其
   **F0 斜率（d=0.002，完全无差异）**——不随"是否真的与宿主同时发声"而变化。
   引擎把同一个 BC 形 token 放进重叠区或停顿区时，token 本身听起来**完全一样**。
3. 因此 qwen 的"形似而神不似"**方向成立，但证据要升级**：
   - 旧证据（realized-ARI=0.08）有标签空间坍塌缺陷，且与 verify_backchannel 的
     "91%/90% 的 BC 确有 ~0.6s 短暂重叠"存在表述张力；
   - **新证据（A1）**：重叠与否在事件自身韵律上**无迹可循**（AUC≈0.58/0.53，f0_slope d≈0）。
     这是"渲染层的交互属性没有进入 token 的声学"的最直接证据。
4. 描述性小发现：真重叠的 BC 比停顿内的 BC 略长（0.72s vs 0.62s，d=0.46）且质心略低、
   有声占比略高——方向合理但效应小，不足以支撑判别。

## 4. 建议的论文表述（替代"catastrophic failure"版）

> "Within the same event category, whether a backchannel truly overlaps host speech
> leaves almost no trace in the token's own prosody (AUC ≈ 0.58 with pitch-shape,
> spectral, and voicing features; pitch slope Cohen's d ≈ 0.0). The rendering engine
> produces context-invariant tokens: the interactional property of being concurrent
> with host speech is carried only by the (trivial) presence of energy in the host
> channel, not by any prosodic adaptation of the token itself."

一句话：**引擎学会了 BC 的形状，但"与宿主并发"这一交互属性没有写入 token 的声音**——
所以站在特征/模型视角，"真重叠的 BC"与"停顿里的 BC"是同一个声学对象。这正是
"形似神不似"的可验证、可引用的表述。

## 5. 局限

- energy_ratio 与 realized 的构造性耦合只能通过"排除它"来隔离；其他特征与重叠状态
  的关联可能有微弱且未测量到的间接效应（如宿主语音对 token 的掩蔽）。
- realized 裁决的阈值（0.1×声道最大 RMS、25ms 窗）会影响分组；连续量版本（A2，
  both_active_frac 与特征的相关分析）可作为敏感性检验。
- 类内判别未用 duration（与主实验口径一致），dur 的组间差异（BC: d=0.46）值得后续单独分析
  （是渲染时的时长补偿还是窗口效应）。

## 6. 复现

```bash
srun --jobid=<kimi作业号> --overlap bash -c \
  'cd /share/workspace3/zhuangruicen/proj-thumpy/pilot_study && \
   /share/home/zhuangruicen/miniconda3/envs/fd_analysis/bin/python scripts/ari_analyze_a1.py'
```
