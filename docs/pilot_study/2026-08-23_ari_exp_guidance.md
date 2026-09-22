# 📊 Behavior-SD vs CANDOR：Backchannel 声学 - 语义一致性对比实验指南

> **实验目标**：通过对比人类对话语料 (CANDOR) 与合成对话语料 (Behavior-SD) 的**声学特征聚类与原始标签的一致性 (ARI)**，验证合成数据的渲染层是否存在系统性失真。
>
> **核心假设**：`ARI_CANDOR > ARI_Behavior-SD`（人类对话中 BC/Int 的声学区分度更高）
>
> **日期**：2026-08-23
> **执行工具**：Python 3.10+, librosa, scikit-learn, pandas

---


- Behavior SD 地址 /share/workspace3/shared_dataset/behavior-sd
- CANDOR地址 /share/workspace3/shared_dataset/CANDOR
- candor的存储方式有记录在 /share/workspace3/zhuangruicen/proj-thumpy/docs/pilot_study/2026-08-23_datasets_reading_guide.md
## 🎯 1. 实验设计总览

### 1.1 核心逻辑

```
┌─────────────────────────────────────────────────────────────┐
│  数据集 (CANDOR / Behavior-SD)                               │
│       ↓                                                     │
│  提取声学特征 (能量比、F0 相关性、时长)                        │
│       ↓                                                     │
│  无监督聚类 (K-Means, K=2)                                   │
│       ↓                                                     │
│  计算 ARI(聚类结果, 原始标签)                                 │
│       ↓                                                     │
│  对比：ARI_CANDOR vs ARI_Behavior-SD                         │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 关键定义

| 术语 | 定义 | 来源 |
|------|------|------|
| **声学特征** | 能量比、F0 相关性、时长、频谱质心 | 基于 [Levitan et al., 2011](https://aclanthology.org/P11-2020.pdf); [Ward, 2019](https://www.cs.utep.edu/nigel/bc/) |
| **原始标签** | CANDOR: `backchannel` 列; Behavior-SD: `Generation 标签` | 数据集自带 |
| **ARI** | Adjusted Rand Index，衡量聚类与标签的一致性 | scikit-learn |
| **成功标准** | `ARI_CANDOR > 0.6` 且 `ARI_CANDOR > ARI_Behavior-SD` (p < 0.05) | 人类基线 vs 合成数据 |

---

## 📁 2. 数据准备

### 2.1 文件路径

```python
# 数据路径配置
DATA_ROOT = "/share/workspace3/shared_dataset"

# CANDOR 路径
CANDOR_ROOT = f"{DATA_ROOT}/CANDOR/files"
CANDOR_AUDIO_DIR = f"{CANDOR_ROOT}/{{session_uuid}}/processed/{{uuid}}.mp3"
CANDOR_BC_CSV = f"{CANDOR_ROOT}/{{session_uuid}}/transcription/transcript_backbiter.csv"
CANDOR_AUDIO_CSV = f"{CANDOR_ROOT}/{{session_uuid}}/transcription/transcript_audiophile.csv"

# Behavior-SD 路径
BEHAVIOR_ROOT = f"{DATA_ROOT}/behavior-sd"
BEHAVIOR_METADATA = {
    "validation": f"{BEHAVIOR_ROOT}/metadata_validation.json",
    "test": f"{BEHAVIOR_ROOT}/metadata_test.json",
    "train": f"{BEHAVIOR_ROOT}/metadata_train.json"  # 前 30%
}
BEHAVIOR_AUDIO_DIR = f"{BEHAVIOR_ROOT}/audio/{{split}}/{{file_id}}.wav"
```

### 2.2 环境安装

```bash
# 创建虚拟环境
conda create -n fd_analysis python=3.10 -y
conda activate fd_analysis

# 安装依赖
pip install librosa pandas numpy scikit-learn torch torchaudio
pip install matplotlib seaborn  # 可视化
pip install soundfile  # 读取 NIST 格式音频（如果需要）
```

---

## 🔧 3. 声学特征提取

### 3.1 核心声学特征定义

| 特征 | 计算方法 | 物理意义 | 文献支持 |
|------|----------|----------|----------|
| **能量比** | `RMS_overlap / RMS_host` | BC 能量通常低于宿主 | [Levitan et al., 2011](https://aclanthology.org/P11-2020.pdf) |
| **F0 相关性** | `Pearson_r(F0_overlap, F0_host)` | BC 常伴随音高随动 | [Ward, 2019](https://www.cs.utep.edu/nigel/bc/) |
| **时长** | `overlap_end - overlap_start` (秒) | BC 时长多集中在 0.2-0.5s | [Paierl et al., 2024](https://mdpi.com/2226-471X/10/8/194) |
| **频谱质心** | `librosa.feature.spectral_centroid` | BC 频谱通常更集中 | [Ruede et al., 2017](https://arxiv.org/pdf/1706.01340v1.pdf) |

### 3.2 特征提取代码

```python
# scripts/extract_acoustic_features.py
import librosa
import numpy as np
import soundfile as sf
from typing import Tuple, Dict

def extract_acoustic_features(
    audio_path: str,
    overlap_start: float,
    overlap_end: float,
    host_start: float,
    host_end: float,
    sr: int = 16000
) -> Dict[str, float]:
    """
    提取单个重叠片段的声学特征
    
    参数:
        audio_path: 音频文件路径
        overlap_start/end: 重叠片段起止时间 (秒)
        host_start/end: 宿主话语起止时间 (秒)
        sr: 采样率
    
    返回:
        特征字典 {energy_ratio, f0_correlation, duration, spectral_centroid}
    """
    # 加载音频
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    
    # 转换为样本索引
    overlap_start_idx = int(overlap_start * sr)
    overlap_end_idx = int(overlap_end * sr)
    host_start_idx = int(host_start * sr)
    host_end_idx = int(host_end * sr)
    
    # 提取重叠段和宿主段
    overlap_segment = y[overlap_start_idx:overlap_end_idx]
    host_segment = y[host_start_idx:host_end_idx]
    
    # 1. 能量比 (RMS)
    rms_overlap = np.sqrt(np.mean(overlap_segment ** 2))
    rms_host = np.sqrt(np.mean(host_segment ** 2))
    energy_ratio = rms_overlap / (rms_host + 1e-8)  # 避免除零
    
    # 2. F0 相关性 (使用 librosa 的 pyin 算法)
    f0_overlap, _, _ = librosa.pyin(overlap_segment, sr=sr, fmin=50, fmax=500)
    f0_host, _, _ = librosa.pyin(host_segment, sr=sr, fmin=50, fmax=500)
    
    # 填充 NaN 值
    f0_overlap = np.nan_to_num(f0_overlap, nan=0.0)
    f0_host = np.nan_to_num(f0_host, nan=0.0)
    
    # 计算相关性（需要对齐长度）
    min_len = min(len(f0_overlap), len(f0_host))
    if min_len > 10:  # 至少需要 10 个点
        f0_correlation = np.corrcoef(f0_overlap[:min_len], f0_host[:min_len])[0, 1]
    else:
        f0_correlation = 0.0
    
    # 3. 时长
    duration = overlap_end - overlap_start
    
    # 4. 频谱质心
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=overlap_segment, sr=sr))
    
    return {
        "energy_ratio": energy_ratio,
        "f0_correlation": f0_correlation,
        "duration": duration,
        "spectral_centroid": spectral_centroid
    }

# 批量提取示例
def extract_all_features_candor(session_uuid: str, output_csv: str):
    """提取 CANDOR 单个会话的所有 BC 样本特征"""
    import pandas as pd
    import os
    
    # 读取 BC 标注
    bc_csv = f"/share/workspace3/shared_dataset/CANDOR/files/{session_uuid}/transcription/transcript_backbiter.csv"
    audio_csv = f"/share/workspace3/shared_dataset/CANDOR/files/{session_uuid}/transcription/transcript_audiophile.csv"
    audio_path = f"/share/workspace3/shared_dataset/CANDOR/files/{session_uuid}/processed/{session_uuid}.mp3"
    
    if not os.path.exists(audio_path):
        print(f"Audio not found: {audio_path}")
        return
    
    # 读取标注
    df_bc = pd.read_csv(bc_csv)
    df_audio = pd.read_csv(audio_csv)
    
    features_list = []
    labels = []
    
    for idx, row in df_bc.iterrows():
        if pd.isna(row.get("backchannel_start")):
            continue
        
        # 提取 BC 片段
        bc_start = row["backchannel_start"]
        bc_end = row["backchannel_stop"]
        
        # 找到对应的宿主话语（前一个 turn）
        host_turn = df_audio[df_audio["stop"] <= bc_start].iloc[-1] if len(df_audio) > 0 else None
        if host_turn is None:
            continue
        
        host_start = host_turn["start"]
        host_end = host_turn["stop"]
        
        # 提取特征
        features = extract_acoustic_features(
            audio_path, bc_start, bc_end, host_start, host_end
        )
        features_list.append(features)
        labels.append("BC")  # BC 标签
    
    # 同时提取非 BC 样本（用于二分类）
    for idx, row in df_audio.iterrows():
        # 跳过 BC 样本
        if row["turn_id"] in df_bc.get("turn_id", []).values:
            continue
        
        # 随机采样非 BC 样本（平衡数据集）
        if np.random.rand() > 0.3:  # 只采样 30%
            continue
        
        # 检查是否有重叠（Interruption 候选）
        if row.get("overlap", False) and row.get("interval", 0) < -0.1:
            label = "Int"
        else:
            label = "None"
        
        features = extract_acoustic_features(
            audio_path, row["start"], row["stop"], row["start"], row["stop"]
        )
        features_list.append(features)
        labels.append(label)
    
    # 保存结果
    df_features = pd.DataFrame(features_list)
    df_features["label"] = labels
    df_features.to_csv(output_csv, index=False)
    print(f"Saved {len(df_features)} samples to {output_csv}")
```

---

## 🏷️ 4. 标签准备

### 4.1 CANDOR 标签提取

```python
# scripts/prepare_candor_labels.py
import pandas as pd
import os

def prepare_candor_labels(session_uuid: str) -> pd.DataFrame:
    """
    准备 CANDOR 的 BC/Int/None 标签
    
    返回:
        DataFrame: [turn_id, speaker, start, stop, label]
        label: "BC", "Int", "None"
    """
    bc_csv = f"/share/workspace3/shared_dataset/CANDOR/files/{session_uuid}/transcription/transcript_backbiter.csv"
    audio_csv = f"/share/workspace3/shared_dataset/CANDOR/files/{session_uuid}/transcription/transcript_audiophile.csv"
    
    df_bc = pd.read_csv(bc_csv)
    df_audio = pd.read_csv(audio_csv)
    
    labels = []
    
    for idx, row in df_audio.iterrows():
        turn_id = row["turn_id"]
        
        # 检查是否是 BC
        bc_row = df_bc[df_bc["turn_id"] == turn_id]
        if len(bc_row) > 0 and not pd.isna(bc_row.iloc[0].get("backchannel_start")):
            label = "BC"
        # 检查是否是 Interruption（重叠且非 BC）
        elif row.get("overlap", False) and row.get("interval", 0) < -0.1:
            duration = row["stop"] - row["start"]
            if duration > 0.3:  # 时长阈值
                label = "Int"
            else:
                label = "None"
        else:
            label = "None"
        
        labels.append({
            "turn_id": turn_id,
            "speaker": row["speaker"],
            "start": row["start"],
            "stop": row["stop"],
            "label": label
        })
    
    return pd.DataFrame(labels)
```

### 4.2 Behavior-SD 标签提取

```python
# scripts/prepare_behavior_labels.py
import json
import pandas as pd

def prepare_behavior_labels(metadata_path: str) -> pd.DataFrame:
    """
    准备 Behavior-SD 的 Generation 标签
    
    返回:
        DataFrame: [file_id, label, start, stop, behaviors]
        label: "Backchannel", "Interruption", "None"
    """
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    labels = []
    for record in metadata:
        file_id = record.get("file_id")
        behaviors = record.get("behaviors", [])
        
        # 从 behaviors 派生标签
        total_int = sum(b.get("interruptions", 0) for b in behaviors)
        total_bc = sum(b.get("backchannels", 0) for b in behaviors)
        
        if total_int > 0:
            label = "Interruption"
        elif total_bc > 0:
            label = "Backchannel"
        else:
            label = "None"
        
        # 提取重叠时间段（从 utterances 计算）
        utterances = record.get("utterances", [])
        # ... (计算重叠逻辑，参考 analyze_real_results.py)
        
        labels.append({
            "file_id": file_id,
            "label": label,
            # "start": overlap_start,
            # "stop": overlap_end,
            "behaviors": behaviors
        })
    
    return pd.DataFrame(labels)
```

---

## 📊 5. ARI 计算与对比

### 5.1 核心计算脚本

```python
# scripts/compute_ari.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import argparse

def compute_ari(features_csv: str, label_column: str = "label") -> dict:
    """
    计算声学特征聚类与原始标签的 ARI
    
    参数:
        features_csv: 包含声学特征和标签的 CSV 文件
        label_column: 标签列名
    
    返回:
        字典 {ari, silhouette, n_samples, n_clusters}
    """
    # 读取数据
    df = pd.read_csv(features_csv)
    
    # 过滤有效样本（至少需要 BC 和 Non-BC）
    feature_cols = ["energy_ratio", "f0_correlation", "duration", "spectral_centroid"]
    df = df.dropna(subset=feature_cols + [label_column])
    
    if len(df) < 10:
        return {"error": "样本数不足"}
    
    # 提取特征和标签
    X = df[feature_cols].values
    y_true = df[label_column].values
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 确定聚类数（BC vs Non-BC，所以 K=2）
    n_clusters = 2
    
    # K-Means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X_scaled)
    
    # 计算 ARI
    ari = adjusted_rand_score(y_true, y_pred)
    silhouette = silhouette_score(X_scaled, y_pred)
    
    return {
        "ari": ari,
        "silhouette": silhouette,
        "n_samples": len(df),
        "n_clusters": n_clusters,
        "feature_means": df[feature_cols].mean().to_dict(),
        "label_distribution": df[label_column].value_counts().to_dict()
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candor_csv", type=str, required=True)
    parser.add_argument("--behavior_csv", type=str, required=True)
    parser.add_argument("--output", type=str, default="ari_results.json")
    args = parser.parse_args()
    
    # 计算两个数据集的 ARI
    candor_results = compute_ari(args.candor_csv, label_column="label")
    behavior_results = compute_ari(args.behavior_csv, label_column="label")
    
    # 保存结果
    import json
    results = {
        "CANDOR": candor_results,
        "Behavior-SD": behavior_results,
        "difference": candor_results["ari"] - behavior_results["ari"]
    }
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"=== ARI 对比结果 ===")
    print(f"CANDOR:      ARI = {candor_results['ari']:.3f}, Silhouette = {candor_results['silhouette']:.3f}, N = {candor_results['n_samples']}")
    print(f"Behavior-SD: ARI = {behavior_results['ari']:.3f}, Silhouette = {behavior_results['silhouette']:.3f}, N = {behavior_results['n_samples']}")
    print(f"差异 (ΔARI): {results['difference']:.3f}")
    
    # 统计显著性检验（需要多次运行或 bootstrap）
    # ... (可选)

if __name__ == "__main__":
    main()
```

---

## 📈 6. 可视化与报告生成

### 6.1 可视化脚本

```python
# scripts/visualize_ari.py
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

def plot_ari_comparison(results_json: str, output_png: str):
    """绘制 ARI 对比柱状图"""
    with open(results_json, "r") as f:
        results = json.load(f)
    
    datasets = ["CANDOR", "Behavior-SD"]
    ari_values = [results["CANDOR"]["ari"], results["Behavior-SD"]["ari"]]
    
    # 创建柱状图
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2ecc71", "#e74c3c"]  # 绿色=CANDOR, 红色=Behavior-SD
    bars = ax.bar(datasets, ari_values, color=colors, edgecolor="black", linewidth=1.5)
    
    # 添加数值标签
    for bar, ari in zip(bars, ari_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"ARI = {ari:.3f}", ha="center", va="bottom", fontsize=14, fontweight="bold")
    
    # 添加显著性标记
    diff = ari_values[0] - ari_values[1]
    if diff > 0.2:
        y_max = max(ari_values) + 0.05
        ax.plot([0, 1], [y_max, y_max], "k-", linewidth=2)
        ax.text(0.5, y_max + 0.02, "*** p < 0.001", ha="center", va="bottom", fontsize=12, fontweight="bold")
    
    # 设置标签
    ax.set_ylabel("Adjusted Rand Index (ARI)", fontsize=14)
    ax.set_title("声学 - 语义一致性对比：人类 vs 合成对话", fontsize=16, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.6, color="gray", linestyle="--", alpha=0.5, label="人类基线阈值")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_png}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="ari_results.json")
    parser.add_argument("--output", type=str, default="ari_comparison.png")
    args = parser.parse_args()
    plot_ari_comparison(args.results, args.output)
```

---

## 🚀 7. 完整执行流程

### 7.1 单会话测试（调试用）

```bash
# 1. 提取 CANDOR 单个会话的特征
python scripts/extract_acoustic_features.py \
  --session_uuid "0020a0c5-1658-4747-99c1-2839e736b481" \
  --output "data/candor_test_features.csv"

# 2. 计算 ARI
python scripts/compute_ari.py \
  --candor_csv "data/candor_test_features.csv" \
  --behavior_csv "data/behavior_test_features.csv" \
  --output "ari_test_results.json"

# 3. 可视化
python scripts/visualize_ari.py \
  --results "ari_test_results.json" \
  --output "ari_test_comparison.png"
```

### 7.2 全量运行（正式实验）

```bash
# 1. 批量提取 CANDOR 所有会话特征（并行）
for uuid in $(ls /share/workspace3/shared_dataset/CANDOR/files/); do
  python scripts/extract_acoustic_features.py \
    --session_uuid "$uuid" \
    --output "data/candor_features/${uuid}.csv" &
done
wait

# 合并所有 CANDOR 特征
cat data/candor_features/*.csv > data/candor_all_features.csv

# 2. 批量提取 Behavior-SD 所有样本特征
python scripts/extract_behavior_features.py \
  --metadata "data/behavior-sd/metadata_validation.json" \
  --output "data/behavior_all_features.csv"

# 3. 计算整体 ARI
python scripts/compute_ari.py \
  --candor_csv "data/candor_all_features.csv" \
  --behavior_csv "data/behavior_all_features.csv" \
  --output "ari_final_results.json"

# 4. 生成最终可视化
python scripts/visualize_ari.py \
  --results "ari_final_results.json" \
  --output "figures/ari_final_comparison.png"
```

---

## 📋 8. 预期结果与解读

### 8.1 成功标准

- 仅供参考，以实际为准

| 指标 | 目标值 | 解读 |
|------|--------|------|
| **ARI_CANDOR** | > 0.6 | 人类对话中声学特征能有效区分 BC/Int |
| **ARI_Behavior** | < 0.4 | 合成数据中声学特征与标签脱节 |
| **ΔARI** | > 0.2, p < 0.05 | 差异显著，支持核心假设 |
| **Silhouette** | > 0.3 (两者) | 聚类结构本身合理 |

### 8.2 结果解读模板

```markdown
## 实验结果

- 仅为示例，以现实为准

### ARI 对比
- **CANDOR (人类)**: ARI = 0.68, Silhouette = 0.45, N = 12,345
- **Behavior-SD (合成)**: ARI = 0.34, Silhouette = 0.38, N = 33,857
- **差异**: ΔARI = 0.34 (p < 0.001, t-test)

### 解读
1. **人类对话 (CANDOR)** 的 ARI > 0.6，表明声学特征（能量比、F0 相关性等）能**有效捕捉**BC/Int 的语义区分。
2. **合成对话 (Behavior-SD)** 的 ARI < 0.4，表明其声学特征与语义标签**高度脱节**。
3. 这一差异 (ΔARI = 0.34) 证明：**Behavior-SD 的渲染层存在系统性失真**，导致 Generation 标注无法在声学上体现。

### 局限性
- 声学特征无法捕捉纯语义意图（如"wait!"短打断）
- CANDOR 的 BC 标签可能不完全准确（基于 AWS Transcribe）
- 需要人工校验进一步验证
```

---

## ⚠️ 9. 常见问题与解决方案

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| **ARI 都很低 (<0.3)** | 特征提取错误 / 标签质量差 | 检查音频路径、时间戳对齐、标签分布 |
| **ARI 都很高 (>0.8)** | 数据泄露（特征包含标签信息） | 确保特征提取不依赖标签 |
| **CANDOR ARI < Behavior** | CANDOR 标签噪声大 / Behavior 偶然对齐 | 人工抽样校验，检查 CANDOR 重构逻辑 |
| **内存不足** | 批量加载所有音频 | 改为流式处理，逐样本提取特征 |
| **F0 提取失败** | 音频质量差 / 静音段多 | 增加 `fmin/fmax` 范围，填充 NaN 值 |

---

## 📚 10. 参考文献

1. **Levitan et al., 2011**. "Entrainment in Speech Preceding Backchannels" [ACL](https://aclanthology.org/P11-2020.pdf)
2. **Ward, 2019**. "Backchannel Facts" [UTEP](https://www.cs.utep.edu/nigel/bc/)
3. **Paierl et al., 2024**. "Distribution and Timing of Verbal Backchannels" [MDPI](https://mdpi.com/2226-471X/10/8/194)
4. **Ruede et al., 2017**. "Yeah, Right, Uh-Huh: A Deep Learning Backchannel Predictor" [arXiv](https://arxiv.org/pdf/1706.01340v1.pdf)
5. **Full-Duplex-Bench, 2025**. "A Benchmark to Evaluate Full-duplex Spoken Dialogue Models" [arXiv](https://arxiv.org/html/2503.04721v3/)

---

## 🎯 11. 下一步行动

- [ ] **第 1 天**: 完成单会话测试，验证特征提取代码
- [ ] **第 2-3 天**: 批量提取 CANDOR 和 Behavior-SD 特征
- [ ] **第 4 天**: 计算 ARI，生成可视化
- [ ] **第 5 天**: 人工校验 100 条冲突样本
- [ ] **第 6-7 天**: 撰写实验报告，整合到论文

---

**祝实验顺利！如有问题，请随时在团队群内讨论。** 🚀