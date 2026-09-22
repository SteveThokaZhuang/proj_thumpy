# 论文骨架 v1：Full-Duplex Benchmark 标签层级审计

> **日期**：2026-08-26
> **投稿目标**：Interspeech 2027 / ICASSP 2027（8-10 页双栏，8 图 1 表）
> **依据**：主报告 + A1/A2/B/C/D/E + E1-E5 全部证据链；与
> [路线图](2026-08-26_ari_roadmap_updated.md) 配套。
> **工作标题候选**：
> - *"Who Labels the Labels? Auditing Annotation Layers in Full-Duplex Dialogue Evaluation"*
> - *"Benchmarks Measure the Label They Inherit: A Three-Layer Audit of Full-Duplex Dialogue Evaluation"*

---

## Abstract（4 句版）

1. 问题：全双工对话模型的评测依赖行为标签（BC/Int/None），标签来源有三层——生成端 metadata GT（L1）、声学派生（L2，VAD/ASR 时间戳+阈值规则）、模型化标注器（L3）——每层有各自的失效模式，但从未被系统审计。
2. 主结果：匹配口径下，合成数据的标签-声学一致性是人类的 5 倍（ARI 0.309 vs 0.058，Δ=−0.251, p<0.001），但 84% 的优势由"对方声道静音"捷径承载（去能量比后坍缩至 0.050）。
3. Benchmark 审计：FD-Bench 风格 VAD 区间规则在同一批数据上 EIR 漂移 25 倍（0.018→0.449）、10% 声道串扰即令指标集归零；三标签源在真实音频上两两 κ 仅 0.06-0.37。
4. 出路：模型化标注器（X2-Turn）在真实数据上帧级 AUC 0.636、窗口级 AUROC 0.667（追踪真实重叠），显著优于任何 L2 标签源；提出"无捷径对照协议"。

**数据/报告**：主报告 §3.1/§3.5、B、E2、E1、E4、E5。

## 1. Introduction

- 钩子：合成数据 ΔARI=−0.251 + 捷径（qwen 模板句，数字已验证：5×、84%）
- 危机陈述：全双工评测（FD-Bench 等）的指标吃什么标签？→ 三层框架
- 贡献三点：① 首次对三层标签源做统一声学一致性审计；② benchmark 指标审计（E1/E2）；③ 模型化标注器方案与校准（E3/E4/E5）
- **[Fig 1] L1/L2/L3 框架图**（待画：三层 + 各层失效模式 + 本文方法箭头）

## 2. Related Work

- Full-duplex SDS 评测：FD-Bench（指标=Silero VAD 区间规则；调研结论：无声道分离、无混音重叠检测、阈值硬编码 0.5s/2.5s）
- 话轮检测/标注模型：X2-Turn（80ms 6 类含 backchannel）、SoulX-Duplug（160ms 5 类，级联 ASR）
- 数据标注噪声：ASR 派生标签的噪声与捷径（引用 backchannel 声学文献 Levitan/Ward 作为特征依据）

## 3. Label Taxonomy（标签层级）

- L1 生成端 GT：完美但捷径化（B：100% 能量比承载；A1/A2：重叠不写入韵律；E：韵律=词条模板）
- L2 声学派生：噪声化（CANDOR ARI 0.058 上限 ~0.07，D；标签空间坍塌 A1；阈值敏感 A2）
- L3 模型化标注器：本工作的校准与应用（§6）
- 方法论：ARI 口径（平衡抽样、会话级 bootstrap B=500、K=3、permutation 基线）、事件定义、5 声道级特征

## 4. Audit Study：主实验与捷径审计

- **[Fig 2]** ARI 对比（0.058 vs 0.309 + CI + Δ 标注）
- **[Fig 3]** ΔARI bootstrap 分布（[−0.281, −0.223]）
- **[Fig 4]** 特征消融（energy_only 0.314 ≈ 全量；去能量比 0.050；其余 ≈ 随机）
- 辅证链（正文或补充材料）：A1/A2（重叠状态不写入 token 韵律，AUC 0.58/0.53，f0_slope d=0.002；实质重叠才有印记 0.64/0.73）、C（迁移折扣 0.72→0.45，去能量比归零）、D（人类基线清洗后仍 0.074）、E（韵律=文本模板，mhm/really 升调 +74/+87）

## 5. Benchmark Audit：FD-Bench 指标审计

- **[Fig 5]** E1 标签一致性矩阵（κ 0.37/0.06/0.24 + AWS×realized 冲突热图 + BC 检出率：VAD 只见 37%、44.5% 在宿主说话期间）
- **[Fig 6]** E2 指标漂移（EIR 25 倍；10% 串扰 n_round→1、SIR/NIR 归零；GT 打断 99.6% 真重叠作为数据侧对照）
- 结论句：声学派生+区间规则的评测值由工具链决定，不由模型行为决定；"独立干净流"假设是承重墙。

## 6. Model Annotator（模型化标注器）

- 校准（E3）：合成 GT 校准暴露**标签空间错配**（64% 触发为话轮内 token；上下文过滤反向验证渲染缺口 0.03）；协议修正：帧级概率+覆盖率口径
- 应用（E4）：真实数据上帧级 AUC 0.636、窗口 AUROC 0.667（BC 信号追踪真实重叠，优于 VAD κ 0.24 与 AWS κ 0.06）、错配率 5.6%
- **[Fig 7]** E3 校准曲线（P/R/F1 vs τ + 过滤版对照 + 帧级 AUC 条）
- **[Fig 8]** E4 结果（AUC 条 + 覆盖率曲线 + 窗口 AUROC 标注）
- 融合（E5）：κ=0.21、交集共识 0.638 不优于单标注器 0.730 → 并集+概率融合方向

## 7. Discussion（含 Table 1 协议建议）

- **Table 1："无捷径对照协议"**：每个 turn-taking 指标须附带 ① 无能量比消融下界；② 片段来源敏感性（换 VAD 参数/工具链的漂移量）；③ 串扰压力测试（10%/30% 声道泄漏退化曲线）；④ 标签源一致性矩阵（κ）。
- 局限：标签语义不可比（L1 跨说话人 vs L3 语言学 any-token）、20 会话 CANDOR 子集、KMeans 设定、64kbps 编码、ASR 依赖链（预留 Oracle ASR 对照位）
- 未来工作：probe 微调（Behavior-SD GT 校准标注器分类头）、概率融合策略、更大真实数据验证

## 8. Conclusion

三层标签审计 + 捷径量化 + 指标协议 + 模型化标注器可行性的四句总结。

---

## 附：图表清单与状态

| # | 图 | 文件 | 状态 |
|---|---|---|---|
| 1 | L1/L2/L3 框架图 | `figures/fig1_label_taxonomy.png` | ✅ |
| 2 | ARI 对比 | `figures/ari_comparison.png` | ✅ |
| 3 | ΔARI 分布 | `figures/delta_hist.png` | ✅ |
| 4 | 特征消融 | `figures/b_ablation.png` | ✅ |
| 5 | 标签一致性矩阵 | `figures/e1_label_matrix.png` | ✅ |
| 6 | FD-Bench 指标漂移 | `figures/e2_fdbench_audit.png` | ✅ |
| 7 | 校准曲线 | `figures/e3_calibration.png` | ✅ |
| 8 | 真实数据结果 | `figures/e4_candor.png` | ✅ |
| T1 | 无捷径对照协议 | 正文表格 | ⏳ 写 §7 时生成 |
