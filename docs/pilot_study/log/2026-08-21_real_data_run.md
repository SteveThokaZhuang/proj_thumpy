# 2026-08-21 Behavior-SD 真实数据全量验证日志 (val + test)

> ⚠️ **勘误**: 本文基于预处理产生的 reduced metadata (丢弃了嵌套 backchannels 字段),
> 其 GT 结论已作废。修正后的结论见
> `2026-08-21_metadata_rebuild_and_reeval.md` (标签→元数据 100% 一致; 真正的不一致
> 在元数据区间 vs 声学渲染, BC 同时发声仅实现标称 ~15%; pyannote r=0.855)。

> 承接 8/20 GPU 合成数据 pilot。本轮以 `/share/workspace3/shared_dataset/behavior-sd` 真实数据
> 为基准,validation + test 全量 (1,857 条, 37.2h 音频) 跑通真实 pyannote + whisper large-v3,
> 产出标签一致性与检测器有效性结论。结果全部放在 proj-thumpy 内。

## 1. 数据与预处理

- 来源: HF datasets 仓库本地 clone (git lfs), 每个 split 若干 tar 分片 (500 对话/tar)。
- validation: 932 条 (18.7h) | test: 925 条 (18.5h) | train: 106,317 条未处理 (见 §6)。
- 预处理 (`scripts/prepare_real_data.py`): 解包 → flac(22050 立体声) 转 16k 单声道 wav
  (ffmpeg) → 派生 Generation Condition 标签:
  `total_interruptions>0 → Interruption; elif total_backchannels>0 → Backchannel; else None`
  → 保留 utterance 时间线 (speaker_idx/start/end) 作为 GT realized behavior。
- 标签分布: Interruption 1609 / Backchannel 215 / None 33。

## 2. 运行方式 (4× RTX 4090 D 并行)

- GPU: kimi tmux 窗口, `srun --gres=gpu:4 --mem=32G -c 8 -p gpu2node -w gpu02`, 48h 时限。
- `scripts/run_real_pipeline.py`: SHARD_ID/NUM_SHARDS 分片 + CUDA_VISIBLE_DEVICES 绑卡 +
  断点续跑 (per-file JSON 已存在则跳过) + per-file 原子写 (tmp+rename)。
- 两批: validation (4 workers) → test (4 workers)。全部 1,857 条零 error, 每批 < 1h
  (4090 上 pyannote 3.1 实测显著快于实时)。
- 输出: `real_data/results/{split}/{diarization,transcription}/*.json` + `pilot_results_shard*.json`。

## 3. 核心结果 (事件级重叠规则: >0.5s=Interruption, 0.1–0.5s=Backchannel, 否则 None)

### 3.1 标签 vs GT 实现 (utterance 时间线) — 不一致率 73.0%

| label \ GT | Interruption | Backchannel | None |
|------------|-------------|------------|------|
| Interruption (1609) | 468 | 1141 | 0 |
| Backchannel (215) | 0 | 0 | 215 |
| None (33) | 0 | 0 | 33 |

### 3.2 标签 vs 声学检测 (pyannote) — 不一致率 35.2%

| label \ det | Interruption | Backchannel | None |
|------------|-------------|------------|------|
| Interruption (1609) | 1127 | 459 | 23 |
| Backchannel (215) | 149 | 47 | 19 |
| None (33) | 0 | 4 | 29 |

### 3.3 检测器有效性 (GT vs 检测)

- 重叠总量 Pearson r = 0.281 (n=1857) — 弱相关
- 均值: GT 0.81s / 1.79 事件 vs 检测 2.21s / 4.84 事件 → **pyannote 系统性高估** (过分割产生假重叠)

## 4. 关键发现 (按标签分组的连续量)

| 标签 | n | GT 有重叠占比 | GT 重叠均值 | 检测重叠均值 |
|------|---|--------------|------------|-------------|
| Interruption | 1609 | **100%** | 0.93s / 2.07 事件 | 2.32s |
| Backchannel | 215 | **0%** | 0.00s | 1.74s (91% 有假重叠) |
| None | 33 | 0% | 0.00s | 0.03s |

1. **Interruption 标签全部有声学实现**: 100% 的 Interruption 对话在渲染音频中存在真实时间重叠
   (71% 有 >0.5s 事件); 标签 vs GT 的 73% 不一致主要由"重叠事件幅度" (71% 只有 0.1–0.5s 短事件)
   与判定阈值 (来自合成 pilot 的 0.5s) 的错配造成。
2. **Backchannel 标签零声学重叠 — 本轮最干净的发现**: 215 条 Backchannel 标签对话的 utterance
   时间线重叠全部为 0。Behavior-SD 的 backchannel 在渲染时被放进主说话人语流的间隙 (顺序拼接),
   而非同时说话。即"backchannel 行为"在声学层面**没有**实现为时间重叠。
3. **pyannote 不能独立当"realized behavior"的检测器**: 在 GT 零重叠的 Backchannel 对话上
   平均检出 1.74s 重叠 (91% 样本有假阳性), 过分割 (均值 4.84 事件 vs GT 1.79) 是主因。
   后续研究须以数据集 utterance 时间线为 realized 基准, diarization 仅作辅助。

## 5. 结论 (对标 doc §5.2 假设)

- 假设"Generation Condition 与声学实现存在系统性不一致"在真实数据上**成立**,
  但结构比 pilot 预期更精细: Interruption 标签→实现为真重叠 (幅度偏短);
  Backchannel 标签→实现为零重叠 (行为以间隙插入方式渲染)。
- 若把"任意重叠事件"作为 Interruption 的 realized 判据, 标签-vs-GT 一致率升至 100% (Int 部分);
  Backchannel 的 0% 不变。建议正式分析采用连续量 (重叠总时长/事件数) + 分行为阈值,
  避免单一 0.5s 阈值。

## 6. 后续

- train 全量 (106K 条 / 2,164h) 未跑: 单卡 4090 需 ~90 GPU 天; 已实现的分片+续跑脚本可直接
  复用, 建议多卡/多节点或抽样子集 (如 5–10K 条) 验证结论稳定性。
- whisper 转写与词级时间戳已产出 (`transcription/*.json`), 可用于后续语义级 Backchannel
  判定 (yeah/uh-huh 词检测)。
- GPU 作业 (srun 42610) 用毕可释放。

## 附: 产物清单 (均在 proj-thumpy 内)

```
pilot_study/real_data/
├── audio/{validation,test}/            # 1,857 个 16k mono wav
├── metadata_{validation,test}.json     # 标签 + utterance 时间线
├── results/{validation,test}/
│   ├── diarization/*.json              # 932 + 925
│   ├── transcription/*.json            # 932 + 925
│   └── pilot_results_shard*.json       # 每 split 4 片
└── results/analysis/
    ├── confusion_matrix_label_vs_gt.png
    ├── confusion_matrix_label_vs_detected.png
    ├── analysis_report.md
    └── statistics.csv
```
