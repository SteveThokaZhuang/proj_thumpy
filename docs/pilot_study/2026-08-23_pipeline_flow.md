# Behavior-SD 评测运行流程 (pilot_study)

> 记录当前 (2026-08-23) 整个评测的完整运行方式:worker 如何启动、音频/元数据如何导入、
> 用什么模型评测、如何分析。所有脚本位于 `pilot_study/scripts/`, 负载任务一律在
> **kimi 窗口 (gpu02, srun 作业)** 内执行, **login01 窗口禁止跑大负载任务**。

## 0. 总览

```
shared_dataset/behavior-sd/{validation,test,train}/*.tar
        │  每个 tar = 500 对话: JSON 官方元数据 + flac 音频
        ▼
① 数据导入 (rebuild_metadata_full.py / check_tars.py)
   - 元数据: 从 tar 解出官方完整 JSON (嵌套 backchannels), 仅附加 file_id
   - 音频: 不再批量解压重命名, 需要时从 tar 流式读取原始 flac
        ▼
② Worker 启动 (run_real_workers.sh)  —— gpu02, 4×RTX 4090 D
   - source env.sh (conda 库路径 + HF_TOKEN)
   - 4 个 worker 各绑 1 卡 (CUDA_VISIBLE_DEVICES=0..3), 轮转分片 SHARD_ID/4
        ▼
③ 评测 pipeline (run_real_pipeline.py)
   - pyannote/speaker-diarization-3.1 → 说话人分离 + 重叠事件
   - whisper large-v3 → 转写 + 词级时间戳
   - per-file JSON 断点续跑; shard 结果原子追加
        ▼
④ 分析 (analyze_real_results.py) → 三层对比 + 混淆矩阵 + r
   - 报告输出 docs/reports/ (pilot_study 内不存报告), 数据 csv 留 real_data/results/analysis/
⑤ 补充验证 (verify_backchannel.py / check_channel_mapping.py) → 声道级客观裁决
```

## 1. 环境准备 (env.sh)

运行前必须 `source env.sh`, 它做两件事:

1. **修复 conda 库路径**: `conda run` 默认不设 `LD_LIBRARY_PATH`, pyannote 4.x 依赖链中的
   ICU 库需要 `CXXABI_1.3.15`, 而系统 /lib64 只有 1.3.11 → 报
   `ImportError: version 'CXXABI_1.3.15' not found`。解决办法是把它指到 conda 环境自己的 lib:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX_ENV/lib:$LD_LIBRARY_PATH"
```

2. **设置 Hugging Face 凭据** (pyannote 3.1 是 gated 模型, 需先在 HF 接受协议 + 创建
   Read token) 和国内镜像加速:

```bash
export HF_TOKEN="hf_..."                # gated 模型访问
export HF_ENDPOINT="https://hf-mirror.com"   # 国内加速 (gated 仓库 403 时改回官方站)
```

## 2. 数据导入

### 2.1 源数据结构

`/share/workspace3/shared_dataset/behavior-sd/{split}/*.tar`, 每 tar 内含:

- **官方 JSON 元数据**: 完整结构 = 顶层 `utterances[]` (各带 `speaker_idx/start_time/end_time`),
  嵌套 `backchannels[]` (Backchannel 是**嵌套在宿主 utterance 内部**的事件, 各带自身时间戳,
  位于宿主窗口内); 顶层 utterances 交叠 = Interruption。
- **flac 音频**: 22050 Hz **真立体声**, 每声道一个说话人 (实测 corr(L,R)≈2.7e-5;
  经 GT 时间线裁决: spk0 时段 L 能量 96.6%, spk1 时段 L 能量 3.4%, 932/932 可分离)。

### 2.2 元数据: 按官方原始格式重建 (rebuild_metadata_full.py)

> 背景: 旧版 `prepare_real_data.py` 预处理时丢弃了嵌套 backchannels 等字段,
> 导致 GT 重叠计算错误。官方 JSON 重建后"标签 → 元数据时间线"100% 一致。

```python
# scripts/rebuild_metadata_full.py (核心逻辑)
for tp in tars:                              # 遍历该 split 的 tar
    with tarfile.open(tp) as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.endswith(".json"):
                tar.extract(m, staging)      # 只解出 JSON
# ...
rec["file_id"] = f"{split}_{p.stem}"         # 唯一附加字段, 用于与结果 join
```

用法: `SPLITS=validation,test python scripts/rebuild_metadata_full.py`
train 子集: `TAR_LIMIT=64 python scripts/rebuild_metadata_full.py` (env `SPLITS=train`)。

输出: `real_data/metadata_{split}.json` = **官方原始格式, 原样保留全部字段, 仅附加 file_id**。

### 2.3 音频: 从 tar 流式读取, 不再解压重命名

旧版做法 (已弃用) 是解压 → flac 转 16k mono wav → 重命名, 造成两个问题:
**① 音频质量下降; ② 立体声 (每声道一个说话人) 被合为单声道, 声道分离信息丢失**。

现在需要原始音频时直接流式读 tar (不需解压):

```python
# scripts/check_tars.py / verify_backchannel.py 的 load_flac()
for tar_path in sorted((SRC / SPLIT).glob("*.tar")):
    with tarfile.open(tar_path) as tar:
        if f"{soda_id}.flac" not in tar.getnames():
            continue
        data = tar.extractfile(f"{soda_id}.flac").read()     # 流式读取
        y, sr = librosa.load(io.BytesIO(data), sr=None, mono=False)  # 保留双声道
        return y, sr
```

**模型输入口径**: pyannote 与 whisper 内部统一重采样到 16k mono, 与 16k mono wav 输入等价;
已用转换 wav 与原始 flac 对比验证保真度 (Pearson r ≈ 0.9995, 无削波) →
**pipeline 不需要重跑**, 立体声验证直接在 tar 内原始 flac 上进行。

> 2026-08-23 清理后 `real_data/audio/` 已删除 (73G); 需重跑 pipeline 时用
> `scripts/extract_audio.py` 按需重新提取 (flac -> 16k mono wav, 与 pipeline 命名一致):

## 3. Worker 启动

### 3.1 在 gpu02 申请独占计算资源 (kimi 窗口执行)

```bash
# kimi 窗口 (gpu02), 一次性申请 4×RTX 4090 D + 32G 内存 + 8 核, 48h 时限
srun --gres=gpu:4 --mem=32G -c 8 -p gpu2node -w gpu02 --pty bash
```

> 注意: srun 作业有 TIME LIMIT (实测原作业 42610 跑 58.6h 后被限时杀死, worker 死在第
> 7,960-7,980/8,000 条)。断点续跑机制保证被杀后重开作业即可继续。

### 3.2 启动 4 个 worker (run_real_workers.sh)

```bash
#!/usr/bin/env bash
# 用法: bash run_real_workers.sh validation   (或 test / train)
cd "$(dirname "$0")"
source ./env.sh

SPLIT="${1:-validation}"
for gpu in 0 1 2 3; do
  LOG="real_${SPLIT}_w${gpu}.log"
  CUDA_VISIBLE_DEVICES=$gpu SPLIT=$SPLIT SHARD_ID=$gpu NUM_SHARDS=4 \
    nohup python scripts/run_real_pipeline.py > "$LOG" 2>&1 &
  echo "worker gpu$gpu: split=$SPLIT shard=$gpu/4 log=$LOG pid=$!"
done
```

要点:
- **每张卡 1 个 worker**, 通过 `CUDA_VISIBLE_DEVICES` 绑卡 → 天然互不干扰
- `SHARD_ID/NUM_SHARDS=4` 轮转分片 → 各卡样本时长均衡
- `nohup + 日志文件` → 断线/窗口关闭不中断, 事后查 `real_{split}_w{gpu}.log`
- 状态检查: `srun --jobid=<作业号> --overlap bash -c "ps aux | grep run_real_pipeline"` 或 `nvidia-smi`

**收尾模式**: 重开作业续跑时, 大部分 per-file 结果已存在, 用
`ONLY_MISSING=1` 只处理缺失样本 (几十分钟 → 几分钟):

```bash
ONLY_MISSING=1 CUDA_VISIBLE_DEVICES=$gpu SPLIT=$SPLIT SHARD_ID=$gpu NUM_SHARDS=4 \
  nohup python scripts/run_real_pipeline.py > real_${SPLIT}_finish_w${gpu}.log 2>&1 &
```

## 4. 评测 pipeline (run_real_pipeline.py)

### 4.1 模型加载

```python
# 说话人分离: pyannote 3.1 (含分割+聚类+重叠检测)
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", token=HF_TOKEN)
diarization_pipeline.to(torch.device("cuda"))

# 转写: whisper large-v3 (词级时间戳)
asr_model = whisper.load_model("large-v3").cuda()
```

### 4.2 每样本处理 (process_sample)

对每个 `file_id` 依次做两件事, **各自写独立的 per-file JSON** (断点续跑的单位):

1. **说话人分离** → 从 pyannote Annotation 提取**不同说话人话轮的两两重叠** (>100ms):

```python
diarization = diarization_pipeline(str(audio_path))
if hasattr(diarization, "speaker_diarization"):   # pyannote 4.x 返回 DiarizeOutput
    diarization = diarization.speaker_diarization
overlaps, tracks = extract_overlaps(diarization)  # 两两话轮 & 交叠, inter.duration > 0.1
result["diarization"] = {"num_speakers": ..., "total_duration": ...,
                         "overlaps": overlaps, "total_overlap_duration": ...}
```

2. **Whisper 转写** → 词级时间戳:

```python
transcription = asr_model.transcribe(str(audio_path), word_timestamps=True, verbose=False)
words = [{"text": w["word"], "start": w["start"], "end": w["end"]}
         for seg in transcription.get("segments", []) for w in seg.get("words", [])]
```

### 4.3 标签派生 (兼容官方格式)

官方 metadata 无 `original_label` 字段, 从 `behaviors[]` 派生 Generation Condition:

```python
total_int = sum(b.get("interruptions", 0) for b in record.get("behaviors", []))
total_bc  = sum(b.get("backchannels", 0) for b in record.get("behaviors", []))
# total_int>0 → "Interruption"; elif total_bc>0 → "Backchannel"; else "None"
```

### 4.4 断点续跑 + 原子写 + 去重

- **per-file 断点**: diarization/transcription 两个 JSON 都存在 → 跳过, 直接读入
- **原子写**: 先写 `*.tmp` 再 `rename` → 随时可中断不损坏
- **shard 结果追加**: 临时文件 + rename 整体重写; `seen_ids` 去重, 已存在的 file_id 不再追加
  (修复旧版重启后重复追加的缺陷)

```python
if os.environ.get("ONLY_MISSING"):      # 收尾模式
    shard_samples = [s for s in shard_samples if _missing(s)]
seen_ids = {r.get("file_id") for r in json.load(SHARD_FILE)} if SHARD_FILE.exists() else set()
for sample in shard_samples:
    result = process_sample(sample)
    if result.get("file_id") not in seen_ids:
        append_shard_result(result)     # tmp + rename 原子追加
        seen_ids.add(result.get("file_id"))
```

### 4.5 产物

```
real_data/results/{split}/
├── diarization/{file_id}.json         # 分离结果 + 重叠事件列表
├── transcription/{file_id}.json       # 转写 + 词级时间戳
└── pilot_results_shard{0..3}.json     # 汇总结果 (4 片, 每样本含标签+diar+asr)
```

## 5. 结果分析 (analyze_real_results.py)

三层对比, 回答"**生成标签 (Generation Condition) vs 实际声学行为 (Realized Behavior)**":

| 层 | 对比 | 说明 |
|----|------|------|
| ① | Generation 标签 vs **GT 实现** (utterance 时间线重叠) | 核心假设: 标签与声学实现是否一致 |
| ② | Generation 标签 vs **检测实现** (pyannote 检出重叠) | 用声学检测器独立验证 |
| ③ | GT vs 检测 重叠总量/事件数 | **检测器有效性** (Pearson r) |

判定规则 (事件级, 与 pilot 同口径):

```python
if any(e - s > 0.5 for s, e in events):           # 任一事件 > 0.5s
    return "Interruption"
if any(0.1 < e - s <= 0.5 for s, e in events):    # 0.1–0.5s
    return "Backchannel"
return "None"
```

GT 重叠事件两来源 (官方格式):

```python
def gt_overlap_events(record):
    # 来源1: 顶层 utterances 不同 speaker_idx 的 pairwise 交叠 (>0.1s)
    # 来源2: 嵌套 backchannels 与其宿主 utterance 的交叠
```

用法:

```bash
SPLITS=validation,test python scripts/analyze_real_results.py
SPLITS=validation,test,train python scripts/analyze_real_results.py   # 合并 train 30%
```

输出: 报告 (`analysis_report.md` + 混淆矩阵 PNG ×2) → **`docs/reports/`**
(pilot_study 内不保存分析报告); 数据 `statistics.csv` → `real_data/results/analysis/`。

## 6. 补充验证脚本

| 脚本 | 目的 |
|------|------|
| `extract_audio.py` | 从 tar 提取音频: flac -> 16k mono wav (并行 ffmpeg, `TAR_LIMIT` 控制子集; audio/ 已清理, 重跑前用它重建) |
| `check_tars.py` | tar 完整性 (2+2+213 全好) + 演示流式读 flac |
| `check_channel_mapping.py` | 裁决"不同说话人分属不同声道": 按 GT 时间线统计 L/R 能量占比 (96.6%/3.4%, 932/932) |
| `verify_backchannel.py` | 声道级重叠裁决: 每声道短时 RMS (25ms 窗/10ms 步进), 活跃阈值 = 0.1×声道最大 RMS, 双声道同时活跃 = 声学重叠; 验证 Backchannel 是否有真实重叠 (结论: 91%/90% 有, None 0%) |
| `dedup_shards.py` | 按 file_id 去重 shard 结果文件 (`python scripts/dedup_shards.py train`), 修复旧版续跑重复追加 |

## 7. 常用命令速查

```bash
# 0. (可选) 重新提取音频 — real_data/audio/ 已清理
TAR_LIMIT=64 SPLITS=train python scripts/extract_audio.py

# 1. 申请计算资源 (kimi 窗口, 一次性)
srun --gres=gpu:4 --mem=32G -c 8 -p gpu2node -w gpu02 --pty bash

# 2. 启动 4 worker (validation / test / train 子集)
bash run_real_workers.sh validation

# 3. 收尾 (断点续跑后只补缺失)
ONLY_MISSING=1 CUDA_VISIBLE_DEVICES=$gpu SPLIT=$SPLIT SHARD_ID=$gpu NUM_SHARDS=4 \
  nohup python scripts/run_real_pipeline.py > real_${SPLIT}_finish_w${gpu}.log 2>&1 &

# 4. 重建官方元数据
SPLITS=validation,test python scripts/rebuild_metadata_full.py
TAR_LIMIT=64 SPLITS=train python scripts/rebuild_metadata_full.py

# 5. 去重 + 分析
python scripts/dedup_shards.py train
SPLITS=validation,test,train python scripts/analyze_real_results.py

# 6. 状态检查 (srun --overlap 是可靠通道, tmux send-keys 不可靠)
srun --jobid=<作业号> --overlap bash -c "grep -c '✅' real_train_finish_w*.log; nvidia-smi"
```

## 8. 关键结论速览 (截至 2026-08-23)

1. **标签 → 元数据时间线: 100% 一致** (官方 JSON 重建后); 真正的不一致在
   **元数据区间 vs 声学渲染**: Backchannel 标称 ~4-5s/条, 真正同时发声仅 ~15% (≈0.6s/条),
   其余渲染进宿主停顿。
2. **pyannote 是合格检测器**: GT vs 检测重叠总量 Pearson r = 0.855; 相对区间 GT 是
   低估 (2.21s vs 4.42s, 过分割导致)。
3. 详细结论见 `docs/log/2026-08-21_metadata_rebuild_and_reeval.md` 与
   `docs/log/2026-08-23_train30_final.md`。
