#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Behavior-SD 真实数据结果分析

三层对比:
1. Generation Condition (behaviors 标签) vs GT Realized Behavior (utterance 时间线重叠)
   —— 本研究的核心假设: 生成标签与声学实现是否一致
2. Generation Condition vs Detected Realized Behavior (pyannote diarization 检出重叠)
   —— 用声学检测器独立验证 (与 pilot 的 pipeline 判定同口径)
3. GT vs Detected 重叠总量/事件数 —— 检测器有效性验证

行为判定规则 (事件级, 与 pilot 阈值一致):
- 任一重叠事件 > 0.5s           -> Interruption
- 否则任一事件 0.1s < d <= 0.5s -> Backchannel
- 否则                            -> None

输出:
  报告 (md + 混淆矩阵 PNG) -> ../docs/reports/  (pilot_study 内不保存报告)
  数据 -> real_data/results/analysis/statistics.csv
"""
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

BASE_DIR = Path(__file__).resolve().parents[1]
REAL_DIR = BASE_DIR / "real_data"
RESULTS_DIR = REAL_DIR / "results"
OUT_DIR = RESULTS_DIR / "analysis"                  # 数据 (csv) 存放处
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = BASE_DIR.parent / "docs" / "reports"   # 报告 (md/png) 统一放 docs
REPORT_DIR.mkdir(parents=True, exist_ok=True)
SPLITS = os.environ.get("SPLITS", "validation,test").split(",")
LABELS = ["Interruption", "Backchannel", "None"]

# =================重叠事件工具=================
def merge_intervals(intervals):
    """合并有交叠的区间 [(s,e),...] -> 不相交区间列表"""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def derive_label(record):
    """官方 behaviors 数组 -> Generation Condition 标签。"""
    total_int = sum(b.get("interruptions", 0)
                    for b in record.get("behaviors", []))
    total_bc = sum(b.get("backchannels", 0)
                   for b in record.get("behaviors", []))
    if total_int > 0:
        return "Interruption"
    if total_bc > 0:
        return "Backchannel"
    return "None"


def gt_overlap_events(record):
    """官方元数据 (完整格式) -> GT 重叠事件 (s,e)。

    来源1: 顶层 utterances 不同 speaker_idx 的 pairwise 交叠;
    来源2: 嵌套 backchannels (宿主 utterance 内, 由另一说话人发出) 与宿主的交叠。
    """
    utts = [(u["start_time"], u["end_time"], u["speaker_idx"])
            for u in record.get("utterances", [])]
    events = []
    for i in range(len(utts)):
        for j in range(i + 1, len(utts)):
            s1, e1, p1 = utts[i]
            s2, e2, p2 = utts[j]
            if p1 != p2:
                s, e = max(s1, s2), min(e1, e2)
                if e - s > 0.1:
                    events.append((s, e))
    for u in record.get("utterances", []):
        host_s, host_e = u["start_time"], u["end_time"]
        for bc in u.get("backchannels", []):
            s = max(bc["start_time"], host_s)
            e = min(bc["end_time"], host_e)
            if e - s > 0.1:
                events.append((s, e))
    return merge_intervals(events)


def classify_events(events):
    """事件列表 -> 3 类标签 (与 pilot 阈值同口径)。"""
    if any(e - s > 0.5 for s, e in events):
        return "Interruption"
    if any(0.1 < e - s <= 0.5 for s, e in events):
        return "Backchannel"
    return "None"


# =================数据组装=================
def load_all():
    """合并所有 shard 结果 + GT 元数据 -> DataFrame 行。"""
    rows = []
    for split in SPLITS:
        meta_file = REAL_DIR / f"metadata_{split}.json"
        if not meta_file.exists():
            print(f"⚠️ 缺 metadata: {meta_file}", flush=True)
            continue
        meta = {m["file_id"]: m for m in json.load(open(meta_file))}
        shards = sorted((RESULTS_DIR / split).glob("pilot_results_shard*.json"))
        if not shards:
            print(f"⚠️ {split} 无 shard 结果", flush=True)
            continue
        for shard_file in shards:
            for r in json.load(open(shard_file)):
                m = meta.get(r["file_id"], {})
                gt_events = gt_overlap_events(m)
                det = r.get("diarization", {})
                det_events = det.get("overlaps") or []
                det_events = [(o["start"], o["end"]) for o in det_events
                              if o.get("duration", 0) > 0.1]
                rows.append({
                    "split": split,
                    "file_id": r["file_id"],
                    "original_label": r.get("original_label") or derive_label(m),
                    "total_interruptions": r.get("total_interruptions", 0),
                    "total_backchannels": r.get("total_backchannels", 0),
                    "duration_s": max(
                        (u["end_time"] for u in m.get("utterances", [])),
                        default=0.0),
                    "gt_overlap_events": len(gt_events),
                    "gt_overlap_total": sum(e - s for s, e in gt_events),
                    "det_overlap_events": len(det_events),
                    "det_overlap_total": sum(e - s for s, e in det_events),
                    "gt_realized": classify_events(gt_events),
                    "det_realized": classify_events(det_events),
                    "diar_error": "error" in det,
                    "asr_error": "error" in r.get("asr", {}),
                    "num_speakers_det": det.get("num_speakers"),
                })
    return pd.DataFrame(rows)


# =================报告=================
def save_cm(df, col, title, path):
    cm = confusion_matrix(df["original_label"], df[col], labels=LABELS)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS)
    plt.title(f"{title}\n(N={len(df)})")
    plt.xlabel("Realized Behavior")
    plt.ylabel("Original Label (Generation Condition)")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return cm


def main():
    print("加载结果...", flush=True)
    df = load_all()
    if df.empty:
        print("❌ 无数据", flush=True)
        return 1
    print(f"共 {len(df)} 条: {df['split'].value_counts().to_dict()}", flush=True)
    print(f"diarization error: {df['diar_error'].sum()}, "
          f"asr error: {df['asr_error'].sum()}", flush=True)

    # 标签分布
    print("\nOriginal label 分布:\n", df["original_label"].value_counts(), flush=True)
    print("\nGT realized 分布:\n", df["gt_realized"].value_counts(), flush=True)
    print("\nDetected realized 分布:\n", df["det_realized"].value_counts(), flush=True)

    # 混淆矩阵
    cm_gt = save_cm(df, "gt_realized",
                    "Original vs GT Realized (utterance timelines)",
                    REPORT_DIR / "confusion_matrix_label_vs_gt.png")
    cm_det = save_cm(df, "det_realized",
                     "Original vs Detected Realized (pyannote)",
                     REPORT_DIR / "confusion_matrix_label_vs_detected.png")

    def agreement(col):
        return (df["original_label"] == df[col]).mean() * 100

    gt_agree = agreement("gt_realized")
    det_agree = agreement("det_realized")
    print(f"\n标签 vs GT 实现 一致率: {gt_agree:.1f}% (不一致 {100-gt_agree:.1f}%)", flush=True)
    print(f"标签 vs 检测实现 一致率: {det_agree:.1f}% (不一致 {100-det_agree:.1f}%)", flush=True)

    # 检测器有效性: GT vs 检测 重叠总量相关性 (仅看成功样本)
    ok = df[~df["diar_error"] & (df["duration_s"] > 0)]
    r = np.corrcoef(ok["gt_overlap_total"], ok["det_overlap_total"])[0, 1]
    print(f"\n检测器有效性 (n={len(ok)}): GT vs 检测重叠总量 Pearson r = {r:.3f}", flush=True)
    print(f"  GT 重叠总量: 均值 {ok['gt_overlap_total'].mean():.2f}s | "
          f"检测: 均值 {ok['det_overlap_total'].mean():.2f}s", flush=True)
    print(f"  GT 事件数: 均值 {ok['gt_overlap_events'].mean():.2f} | "
          f"检测: 均值 {ok['det_overlap_events'].mean():.2f}", flush=True)

    # 统计 CSV
    df.to_csv(OUT_DIR / "statistics.csv", index=False)

    # 报告
    cm_gt_md = pd.DataFrame(
        cm_gt, index=[f"label={x}" for x in LABELS],
        columns=[f"real={x}" for x in LABELS]).to_markdown()
    cm_det_md = pd.DataFrame(
        cm_det, index=[f"label={x}" for x in LABELS],
        columns=[f"real={x}" for x in LABELS]).to_markdown()

    splits_desc = ", ".join(f"{s} {len(df[df.split == s])}" for s in SPLITS)
    report = f"""# Behavior-SD 真实数据验证报告 ({" + ".join(SPLITS)})

- 数据: /share/workspace3/shared_dataset/behavior-sd ({splits_desc})
- 模型: pyannote/speaker-diarization-3.1 + whisper-large-v3 (4× RTX 4090 D)
- 判定规则: 事件级重叠 (>0.5s -> Interruption; 0.1-0.5s -> Backchannel; 否则 None)

## 1. 标签分布

| 类别 | Generation 标签 | GT 实现 | 检测实现 |
|------|----------------|---------|----------|
{chr(10).join(f"| {l} | {df['original_label'].eq(l).sum()} | {df['gt_realized'].eq(l).sum()} | {df['det_realized'].eq(l).sum()} |" for l in LABELS)}

## 2. 核心结果

- **标签 vs GT 实现一致率: {gt_agree:.1f}%** (不一致率 {100-gt_agree:.1f}%)
- **标签 vs 检测实现一致率: {det_agree:.1f}%** (不一致率 {100-det_agree:.1f}%)
- 检测器有效性: GT vs 检测重叠总量 Pearson r = {r:.3f} (n={len(ok)})

### 2.1 混淆矩阵: 标签 vs GT (utterance 时间线)

{cm_gt_md}

![label vs gt](confusion_matrix_label_vs_gt.png)

### 2.2 混淆矩阵: 标签 vs 声学检测 (pyannote)

{cm_det_md}

![label vs detected](confusion_matrix_label_vs_detected.png)

## 3. 解读

1. GT (utterance 时间线) 是 TTS 实际渲染时序, 代表"声学可实现的重叠";
   检测结果代表独立声学测量。两者与生成标签的差距即为标签-实现不一致。
2. Backchannel 为轻声短插入, 声学检测天然更难; 若标签-vs-GT 一致率高而
   标签-vs-检测一致率低, 说明 gap 主要来自检测难度而非标签错误。

## 4. 结论与后续

- 见 docs/log/2026-08-21_real_data_run.md
"""
    with open(REPORT_DIR / "analysis_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n✅ 报告 -> {REPORT_DIR / 'analysis_report.md'}", flush=True)
    print("ANALYSIS DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
