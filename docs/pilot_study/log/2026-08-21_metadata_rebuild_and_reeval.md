# 2026-08-21 按官方原始格式重建 metadata 并重评测 val+test

> 触发: 用户检查官方 JSON 后发现我们的 `metadata_{split}.json` "面目全非"。
> 根因: `prepare_real_data.py` 预处理只保留顶层 utterances, **丢弃了嵌套的 backchannels
> 数组 (每条 bc 带自己的 start_time/end_time, 位于宿主 utterance 窗口内部)**。
> 此前所有基于该 metadata 的 GT 结论作废, 本轮重做。全部操作在 kimi 窗口 (gpu02) 完成。

## 1. 修复

- `scripts/rebuild_metadata_full.py`: 从 tar 重新解出官方 JSON, **原样保留全部字段**
  (narrative/speakers/behaviors/utterances[含嵌套 backchannels]/tts_speaker_ids/
  statistics...), 仅附加 `file_id` 用于与 pipeline 结果 join。
  旧 reduced 版备份为 `metadata_{split}_reduced.bak.json`。
- `scripts/analyze_real_results.py`: `gt_overlap_events` 改为两个来源 —
  (1) 顶层 utterances 不同说话人 pairwise 交叠; (2) **嵌套 bc 与宿主 utterance 的交叠**。
- `scripts/verify_backchannel.py` 同步适配官方格式。

## 2. 重评测结果 (val+test, n=1,857, 零 error)

### 2.1 标签 vs GT(区间时间线) 一致率 **78.8%** (旧结论 27.0% — 伪象)

| label \ GT | Interruption | Backchannel | None |
|------------|-------------|------------|------|
| Interruption (1609) | 1417 | 192 | 0 |
| Backchannel (215) | 202 | 13 | 0 |
| None (33) | 0 | 0 | 33 |

### 2.2 标签 → 元数据实现率 (按对应事件类型存在性): **100%**

- Interruption 标签 1609/1609 有顶层交叠事件; Backchannel 标签 215/215 有嵌套 bc 事件;
  None 0 事件。**Generation Condition 在元数据层完全实现。**

### 2.3 检测器有效性 (修正后 GT): **r = 0.855** (旧 r=0.281 — 伪象)

- GT 重叠总量均值 4.42s vs pyannote 检出 2.21s → pyannote 相对区间 GT 是**低估**
  (旧结论"过分割高估"方向性错误); pyannote 是合格的相关性检测器。

## 3. 三层对照: 标签 → 元数据区间 → 声学渲染 (Backchannel 样本, 各 60 条)

| 层 | 量 | val | test |
|----|----|-----|------|
| 元数据 | 标称 bc 总时长/条 | 3.77s | 5.17s |
| 声学 | 宿主窗口内对方声道实际发声 | 1.73s (标称 46%) | 2.30s (44%) |
| 声学 | **其中真正同时发声** | **0.58s (标称 15%)** | **0.64s (12%)** |

- 立体声实测 vs 修正 GT: r = 0.812/0.810 (高相关), 但总量仅 GT 的 ~15%。
- 解读: bc 音频确在宿主 utterance 窗口内渲染 (存在性 91-100%), 但**大部分落在宿主
  语音的声学停顿中**; 标称的"时间重叠"只有 ~15% 实现为真实同时发声。

## 4. 结论修正 (对先前结论的诚实勘误)

| 旧结论 (已撤销) | 修正后 |
|----------------|--------|
| "Backchannel 渲染为间隙插入, 零重叠" | 伪象: 我们丢掉了嵌套 bc 字段。官方元数据中 bc **区间级**与宿主重叠 100% |
| "元数据标注错误 (Backchannel 91% 音频有重叠但 GT 标 0)" | 错怪数据集: 标签→元数据 100% 一致。**真正的不一致在元数据区间 vs 声学渲染** (同时发声只实现 ~15%) |
| "pyannote 过分割高估重叠 (r=0.281)" | 对正确 GT, r=0.855, pyannote 是低估 (2.21s vs 4.42s) |

**核心发现 (对标研究假设 "标签-实现一致性")**:
1. Generation Condition (behaviors 标签) → 元数据 utterance/backchannel 时间线: **完全一致**
2. 元数据区间时间线 → 声学渲染: **Backchannel 的系统性落差** — 标称重叠 ~4-5s/条,
   声学同时发声仅 ~0.6s/条 (15%), 其余渲染进宿主停顿
3. 0.5s 事件阈值规则对 BC 语义不适配 (bc 事件常 >0.5s, 导致 202/215 BC 标签判为
   GT-Interruption) — 建议按事件类型 (int 事件/bc 事件) 分别判定

## 5. 产物

- `real_data/metadata_{validation,test}.json` — 官方原始格式 (全部字段 + file_id)
- `real_data/results/analysis/` — 更新的 confusion matrix/statistics.csv/analysis_report.md
- `real_data/verification/` — 立体声验证 (stats/summary/plot/excerpt, 本轮结论)
- 新脚本: `scripts/rebuild_metadata_full.py`

## 6. 后续

- train 30% pipeline 运行中 (其 per-file 结果不受影响; 最终分析前需同样重建
  `metadata_train.json` 官方格式)
- 立体声验证可对 Interruption 样本复跑 (宿主窗口 footprint), 确认 int 事件的
  区间-声学实现率
