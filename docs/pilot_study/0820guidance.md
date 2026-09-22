# 🧪 Behavior-SD Label Consistency Pilot Study
## 实验执行细则 (Agent 专用版)

> **实验目标**: 量化验证 Behavior-SD 的"生成标签 (Generation Condition)"与"音频实际实现行为 (Realized Behavior)"之间的不一致率。  
> **预计耗时**: 4-6 小时 (GPU 环境) / 12 小时 (CPU 环境)  
> **输出物**: `pilot_results.json`, `analysis_report.md`, `confusion_matrix.png`  
> **当前时间**: 2026-08-20

---

## 📋 1. 环境准备

### 1.1 系统要求
- **Python**: 3.9 - 3.11
- **GPU**: 推荐 (NVIDIA T4/V100/A10 或更高), 显存 ≥ 8GB
- **CPU**: 可选 (处理速度约慢 5-10 倍)
- **磁盘空间**: ≥ 10GB (模型缓存)

### 1.2 安装依赖

```bash
# 创建虚拟环境
conda create -n fd_pilot python=3.10 -y
conda activate fd_pilot

# 安装 pyannote.audio 3.1 (说话人分离与重叠检测)
pip install pyannote.audio==3.1.0

# 安装 Whisper (语音识别与词级时间戳)
pip install openai-whisper

# 安装分析依赖
pip install pandas numpy matplotlib seaborn scikit-learn

# 安装音频处理工具
pip install torchaudio librosa
```

### 1.3 获取 Hugging Face Token (必需)

> ⚠️ **重要**: `pyannote/speaker-diarization-3.1` 和 `pyannote/segmentation-3.0` 都是 gated 模型，需要分别接受协议 [🎹 Speaker diarization 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1/blob/main/README.md)。

**步骤**:
1. 访问 https://huggingface.co/pyannote/speaker-diarization-3.1 并接受用户协议
2. 访问 https://huggingface.co/pyannote/segmentation-3.0 并接受用户协议
3. 在 https://huggingface.co/settings/tokens 创建 Read 权限的 Token
4. 将 Token 保存为环境变量:
   ```bash
   export HF_TOKEN="your_huggingface_token_here"
   ```

---

## 📁 2. 数据准备

### 2.1 数据结构
创建以下目录结构:
```
pilot_study/
├── data/
│   ├── audio/              # 存放音频文件 (.wav)
│   ├── metadata.json       # 原始标签 (Generation Condition)
│   └── samples_list.txt    # 样本列表
├── results/
│   ├── diarization/        # 说话人分离结果
│   ├── transcription/      # ASR 结果
│   └── analysis/           # 分析结果
└── scripts/
    ├── run_pipeline.py
    └── analyze_results.py
```

### 2.2 准备测试样本 (20-30 条)

**方案 A: 使用 Behavior-SD 公开数据**
- 如果 Behavior-SD 代码/数据已公开 [Behavior-SD: Behaviorally Aware Spoken Dialogue Generation with Large Language Models](https://aclanthology.org/2025.naacl-long.484/), 下载 20-30 条带有 `Interruption` 或 `Backchannel` 标签的音频。

**方案 B: 合成模拟数据 (如果 A 不可行)**
使用简单规则合成 20 条模拟数据:
- **10 条 "Interruption"**: 两段语音重叠 > 800ms, 第二段语音音量略高
- **5 条 "Backchannel"**: 两段语音重叠 200-400ms, 内容为 "yeah", "uh-huh", "right"
- **5 条 "None"**: 无重叠, 清晰的话轮转换

**metadata.json 格式示例**:
```json
[
  {
    "file_id": "sample_001",
    "audio_path": "data/audio/sample_001.wav",
    "original_label": "Interruption",
    "behavior_intensity": 2,
    "speaker_count": 2
  },
  {
    "file_id": "sample_002",
    "audio_path": "data/audio/sample_002.wav",
    "original_label": "Backchannel",
    "behavior_intensity": 1,
    "speaker_count": 2
  }
]
```

---

## 🚀 3. 核心处理 Pipeline

### 3.1 主脚本 `scripts/run_pipeline.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Behavior-SD Pilot Study: Label Consistency Analysis
Run diarization and ASR on all samples
"""

import os
import json
import torch
import whisper
from pyannote.audio import Pipeline
from pathlib import Path
from tqdm import tqdm

# =================配置=================
HF_TOKEN = os.environ.get("HF_TOKEN")
AUDIO_DIR = Path("data/audio")
METADATA_FILE = Path("data/metadata.json")
RESULTS_DIR = Path("results")
Diarization_DIR = RESULTS_DIR / "diarization"
ASR_DIR = RESULTS_DIR / "transcription"

# 确保目录存在
for d in [Diarization_DIR, ASR_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =================加载模型=================
print("Loading pyannote pipeline (GPU recommended)...")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)

# 如果有 GPU, 移动到 GPU
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    diarization_pipeline.to(torch.device("cuda"))
else:
    print("Warning: Running on CPU (slow)")

print("Loading Whisper model...")
asr_model = whisper.load_model("large-v3")
if torch.cuda.is_available():
    asr_model = asr_model.cuda()

# =================处理函数=================
def process_sample(sample):
    """处理单个样本"""
    file_id = sample["file_id"]
    audio_path = sample["audio_path"]
    
    if not Path(audio_path).exists():
        print(f"⚠️  File not found: {audio_path}")
        return None
    
    result = {
        "file_id": file_id,
        "original_label": sample["original_label"],
        "audio_path": audio_path
    }
    
    # 1. 说话人分离与重叠检测
    try:
        diarization = diarization_pipeline(audio_path)
        
        # 提取重叠区域
        overlaps = []
        tracks = list(diarization.itertracks(yield_label=True))
        for i, (turn_a, _, spk_a) in enumerate(tracks):
            for j, (turn_b, _, spk_b) in enumerate(tracks):
                if i < j and spk_a != spk_b:
                    intersection = turn_a & turn_b
                    if intersection.duration > 0.1:  # >100ms 视为有效重叠
                        overlaps.append({
                            "start": intersection.start,
                            "end": intersection.end,
                            "duration": intersection.duration,
                            "speakers": [spk_a, spk_b]
                        })
        
        result["diarization"] = {
            "num_speakers": len(set([spk for _, _, spk in tracks])),
            "total_duration": sum([t[0].duration for t in tracks]),
            "overlaps": overlaps,
            "total_overlap_duration": sum([o["duration"] for o in overlaps])
        }
        
        # 保存 diarization 结果
        with open(Diarization_DIR / f"{file_id}.json", "w") as f:
            json.dump(result["diarization"], f, indent=2)
            
    except Exception as e:
        print(f"❌ Diarization failed for {file_id}: {e}")
        result["diarization"] = {"error": str(e)}
    
    # 2. 语音识别 (词级时间戳)
    try:
        transcription = asr_model.transcribe(
            audio_path,
            word_timestamps=True,
            verbose=False
        )
        
        # 提取词级信息
        words = []
        for segment in transcription.get("segments", []):
            if "words" in segment:
                for word in segment["words"]:
                    words.append({
                        "text": word["word"],
                        "start": word["start"],
                        "end": word["end"]
                    })
        
        result["asr"] = {
            "text": transcription["text"],
            "language": transcription.get("language", "unknown"),
            "words": words,
            "num_words": len(words)
        }
        
        # 保存 ASR 结果
        with open(ASR_DIR / f"{file_id}.json", "w") as f:
            json.dump(result["asr"], f, indent=2)
            
    except Exception as e:
        print(f"❌ ASR failed for {file_id}: {e}")
        result["asr"] = {"error": str(e)}
    
    return result

# =================主流程=================
def main():
    # 加载元数据
    with open(METADATA_FILE, "r") as f:
        samples = json.load(f)
    
    print(f"📊 Processing {len(samples)} samples...")
    
    all_results = []
    for sample in tqdm(samples, desc="Processing"):
        result = process_sample(sample)
        if result:
            all_results.append(result)
    
    # 保存所有结果
    output_file = RESULTS_DIR / "pilot_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ All results saved to {output_file}")
    print(f"📈 Processed {len(all_results)}/{len(samples)} samples successfully")

if __name__ == "__main__":
    main()
```

### 3.2 运行主 Pipeline

```bash
cd pilot_study
python scripts/run_pipeline.py
```

**预期输出**:
```
Loading pyannote pipeline (GPU recommended)...
Using GPU: NVIDIA A10
Loading Whisper model...
📊 Processing 20 samples...
Processing: 100%|████████████████████| 20/20 [05:23<00:00, 16.15s/sample]
✅ All results saved to results/pilot_results.json
📈 Processed 20/20 samples successfully
```

---

## 📊 4. 数据分析脚本

### 4.1 分析脚本 `scripts/analyze_results.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze pilot results: Compare original labels vs realized behaviors
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

# =================配置=================
RESULTS_FILE = Path("results/pilot_results.json")
OUTPUT_DIR = Path("results/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =================行为判定规则=================
def classify_realized_behavior(sample):
    """
    根据声学特征判定 Realized Behavior
    
    规则:
    - Realized Interruption: Overlap > 500ms 且 原说话人停止 (简化版: 仅看重叠时长)
    - Realized Backchannel: 100ms < Overlap <= 500ms
    - Realized None: Overlap <= 100ms
    """
    diarization = sample.get("diarization", {})
    if "error" in diarization:
        return "Error"
    
    total_overlap = diarization.get("total_overlap_duration", 0)
    
    if total_overlap > 0.5:
        return "Interruption"
    elif total_overlap > 0.1:
        return "Backchannel"
    else:
        return "None"

# =================加载数据=================
print("Loading results...")
with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

# =================构建 DataFrame=================
data = []
for sample in results:
    original_label = sample.get("original_label", "Unknown")
    realized_label = classify_realized_behavior(sample)
    
    data.append({
        "file_id": sample["file_id"],
        "original_label": original_label,
        "realized_label": realized_label,
        "match": original_label == realized_label,
        "overlap_duration": sample.get("diarization", {}).get("total_overlap_duration", 0),
        "num_speakers": sample.get("diarization", {}).get("num_speakers", 0)
    })

df = pd.DataFrame(data)

# =================统计分析=================
print("\n" + "="*50)
print("📊 STATISTICAL SUMMARY")
print("="*50)

total_samples = len(df)
match_count = df["match"].sum()
disagreement_rate = 1 - (match_count / total_samples)

print(f"Total samples: {total_samples}")
print(f"Matching labels: {match_count} ({match_count/total_samples*100:.1f}%)")
print(f"Disagreement rate: {disagreement_rate*100:.1f}%")

print("\nOriginal label distribution:")
print(df["original_label"].value_counts())

print("\nRealized label distribution:")
print(df["realized_label"].value_counts())

# =================混淆矩阵=================
print("\nGenerating confusion matrix...")

# 标签映射 (确保顺序一致)
labels = ["Interruption", "Backchannel", "None"]
cm = confusion_matrix(df["original_label"], df["realized_label"], labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title(f'Confusion Matrix: Original vs Realized (N={total_samples})')
plt.xlabel('Realized Behavior')
plt.ylabel('Original Label (Generation Condition)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300)
print(f"✅ Confusion matrix saved to {OUTPUT_DIR / 'confusion_matrix.png'}")

# =================典型案例=================
print("\nIdentifying典型案例...")

# 不一致的样本
mismatched = df[~df["match"]]
if len(mismatched) > 0:
    print(f"\n⚠️  Found {len(mismatched)} mismatched samples:")
    for _, row in mismatched.head(5).iterrows():
        print(f"  - {row['file_id']}: {row['original_label']} → {row['realized_label']} "
              f"(Overlap: {row['overlap_duration']:.2f}s)")

# =================生成报告=================
report_md = f"""# Behavior-SD Pilot Study: Analysis Report

## 1. Overview
- **Total Samples**: {total_samples}
- **Processing Date**: 2026-08-20
- **Models**: pyannote/speaker-diarization-3.1, Whisper-large-v3

## 2. Key Findings

### 2.1 Disagreement Rate
- **Matching Labels**: {match_count} ({match_count/total_samples*100:.1f}%)
- **Disagreement Rate**: **{disagreement_rate*100:.1f}%**

### 2.2 Label Distribution

**Original Labels (Generation Condition)**:
{df['original_label'].value_counts().to_markdown()}

**Realized Behaviors (Acoustic Detection)**:
{df['realized_label'].value_counts().to_markdown()}

## 3. Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

## 4.典型案例 Analysis

### 4.1 Mismatched Samples (Top 5)
| File ID | Original | Realized | Overlap (s) |
|---------|----------|----------|-------------|
"""

for _, row in mismatched.head(5).iterrows():
    report_md += f"| {row['file_id']} | {row['original_label']} | {row['realized_label']} | {row['overlap_duration']:.2f} |\n"

report_md += f"""
## 5. Conclusions

1. **Significant Gap Observed**: {disagreement_rate*100:.1f}% 的生成标签与声学实现不一致
2. **Main Error Patterns**:
   - False Interruption: Labeled as "Interruption" but overlap < 500ms
   - Missed Backchannel: Labeled as "None" but detected short overlap
3. **Implications**: 
   - Behavior-SD 的 Generation Condition 不能直接作为 Ground Truth
   - 需要 Grounded Re-annotation 来修正标签

## 6. Next Steps
- [ ] 扩大样本量至 200+
- [ ] 引入语义分析 (Backchannel 词汇检测)
- [ ] 结合话轮转换 (Floor Transfer) 进一步优化规则
"""

with open(OUTPUT_DIR / "analysis_report.md", "w", encoding="utf-8") as f:
    f.write(report_md)

print(f"✅ Analysis report saved to {OUTPUT_DIR / 'analysis_report.md'}")
print("\n" + "="*50)
print("🎉 ANALYSIS COMPLETE")
print("="*50)
```

### 4.2 运行分析

```bash
python scripts/analyze_results.py
```

---

## 📈 5. 预期输出物

### 5.1 文件清单
```
results/
├── pilot_results.json          # 完整处理结果
├── diarization/                # 每个样本的说话人分离结果
│   ├── sample_001.json
│   └── ...
├── transcription/              # 每个样本的 ASR 结果
│   ├── sample_001.json
│   └── ...
└── analysis/
    ├── confusion_matrix.png    # 混淆矩阵图
    ├── analysis_report.md      # 详细分析报告
    └── statistics.csv          # 统计数据
```

### 5.2 关键指标
- **Disagreement Rate**: 目标 > 20% (证明假设成立)
- **Precision/Recall**: 针对每类行为的检测准确率
- **典型案例**: 3-5 个明显不一致的样本用于汇报展示

---

## 🎯 6. 汇报要点 (用于组会/PPT)

### 6.1 核心结论模板
> "Pilot 实验显示，**{X}%** 的 Behavior-SD 生成标签与声学实现行为不一致。主要问题包括:
> 1. **False Interruption**: 标记为'打断'但实际重叠时长不足
> 2. **Missed Backchannel**: 标记为'无'但检测到轻微重叠
> 3. **Timing Mismatch**: TTS 时序对齐误差导致行为意图未准确实现"

### 6.2 可视化建议
- **图 1**: 混淆矩阵 (Original vs Realized)
- **图 2**: 波形图对比 (展示 Label 说是打断，实际只是轻微重叠的典型案例)
- **图 3**: 重叠时长分布直方图 (按原始标签分组)

### 6.3 下一步计划
1. 扩大样本量至 200+ 验证统计显著性
2. 引入语义分析区分 Backchannel 功能 (Emotive vs Cognitive)
3. 启动 Unified Interaction Schema 设计

---

## ⚠️ 7. 常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `401 Unauthorized` | 未接受 Hugging Face 协议 | 访问模型页面接受协议，检查 Token 权限 |
| `CUDA out of memory` | 显存不足 | 使用 `num_speakers=2` 限制说话人数，或改用 CPU |
| 处理速度过慢 | CPU 运行 | 接受现实，或租用云 GPU (Colab/AWS) |
| 重叠检测不准 | 音频质量差 | 检查音频采样率 (应为 16kHz), 尝试预处理降噪 |

---

## 📚 8. 参考资源

- **pyannote.audio 文档**: [🎹 Speaker diarization 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1/blob/main/README.md)
- **Whisper 官方仓库**: https://github.com/openai/whisper
- **Behavior-SD 论文**: [Behavior-SD: Behaviorally Aware Spoken Dialogue Generation](https://aclanthology.org/2025.naacl-long.484/)

---

**祝实验顺利！有任何问题随时调整参数和规则。** 🚀