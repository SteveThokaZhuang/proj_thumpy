"""F4: ASR 归因 — X2-Turn 转写 WER 与状态预测质量的相关性.

设计 (金标准侧, Behavior-SD):
  对 E3 的 30 个文件 × 2 声道:
    WER = X2-Turn transcript vs GT tts_text 拼接 (词级 Levenshtein, 文本归一化)
    状态质量 = 帧级 p_bc AUC vs GT BC 窗口隶属 (与 E3 同口径)
  分析: WER 与 AUC 的 Pearson 相关 (每声道一点) + WER 直方图.
  裁决: 强负相关 -> ASR 是状态预测瓶颈; 弱相关 -> 状态误差主要来自状态头本身.

输出: f4_results.json + docs/pilot_study/figures/f4_asr_attribution.png
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FIG_DIR = "/share/workspace3/zhuangruicen/proj-thumpy/docs/pilot_study/figures"
ANNOT = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator"
META_DIR = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data"
C_BLUE, C_ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, BASE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower())


def edit_dist(r, h, free_prefix_del=False):
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    for i in range(len(r) + 1):
        d[i, 0] = 0 if free_prefix_del else i   # 子串匹配: 参考前缀删除免费
    for j in range(len(h) + 1):
        d[0, j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d


def wer(ref, hyp, float_align=True):
    """WER; float_align=True 时转写沿参考做子串匹配取最小错率
    (X2-Turn 流式转写缺头部 tokens, 固定对齐会虚高 WER)."""
    r = [w for w in norm(ref).split() if w]
    h = [w for w in norm(hyp).split() if w]
    if not r or not h:
        return float("nan")
    if not float_align:
        return edit_dist(r, h)[-1, -1] / len(r)
    d = edit_dist(r, h, free_prefix_del=True)
    best = min(d[i, len(h)] for i in range(len(r) + 1))
    return best / len(h)


def load_gt_texts():
    """file_name -> {spk_idx: gt_text}."""
    out = {}
    for split in ["validation"]:
        meta = json.load(open(f"{META_DIR}/metadata_{split}.json"))
        for rec in meta:
            stem = rec["file_name"][:-5]
            txt = {0: [], 1: []}
            for u in rec.get("utterances", []):
                txt[u["speaker_idx"]].append(u.get("tts_text", ""))
            out[f"{split}_{stem}"] = {k: " ".join(v) for k, v in txt.items()}
    return out


def main():
    gt_texts = load_gt_texts()
    rows = []
    for fp in sorted(glob.glob(f"{ANNOT}/e3_raw/*.json")):
        fid = os.path.basename(fp)[:-5]
        if fid not in gt_texts:
            continue
        d = json.load(open(fp))
        for ch in ("0", "1"):
            pred = d["pred"][ch]
            frames = pred["frames"]
            if not frames:
                continue
            p_bc = np.array([f["p_bc"] for f in frames])
            t0 = np.array([f["t0"] for f in frames])
            t1 = np.array([f["t1"] for f in frames])
            gt_bc = [(w[0], w[1]) for w in d["gt"][ch]["bc"]]
            ins = np.zeros(len(t0), bool)
            for (a, b) in gt_bc:
                ins |= (t0 < b) & (t1 > a)
            auc = roc_auc_score(ins, p_bc) if ins.any() and (~ins).any() else None
            w = wer(gt_texts[fid][int(ch)], pred.get("transcript", ""))
            if auc is not None and w == w:  # 非 NaN
                rows.append({"file": fid, "ch": ch, "wer": w, "auc": auc})

    df = pd.DataFrame(rows)
    r, p = pearsonr(df["wer"], df["auc"])
    res = {
        "n_channels": len(df),
        "wer_mean": round(float(df["wer"].mean()), 4),
        "wer_median": round(float(df["wer"].median()), 4),
        "auc_mean": round(float(df["auc"].mean()), 4),
        "pearson_r": round(float(r), 4),
        "pearson_p": float(p),
    }
    print(json.dumps(res, indent=1), flush=True)
    with open(f"{ANNOT}/f4_results.json", "w") as f:
        json.dump(res, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor=SURFACE)
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
    for s_ in ["left", "bottom"]:
        ax.spines[s_].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=9.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.scatter(df["wer"], df["auc"], s=22, color=C_BLUE, alpha=0.7,
               linewidths=0)
    ax.set_xlabel("X2-Turn ASR word error rate", fontsize=10.5, color=INK)
    ax.set_ylabel("frame-level p_bc AUC", fontsize=10.5, color=INK)
    ax.set_title(f"WER vs state quality (per channel, r={r:.3f}, p={p:.2f})",
                 fontsize=11, color=INK, fontweight="bold")
    ax2 = axes[1]
    ax2.set_facecolor(SURFACE)
    for s_ in ["top", "right"]:
        ax2.spines[s_].set_visible(False)
    for s_ in ["left", "bottom"]:
        ax2.spines[s_].set_color(BASE)
    ax2.tick_params(colors=INK2, labelsize=9.5)
    ax2.grid(axis="y", color=GRID, linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.hist(df["wer"], bins=25, color=C_ORANGE, alpha=0.85,
             edgecolor="white", linewidth=0.3)
    ax2.axvline(df["wer"].median(), color=INK, linestyle="--", linewidth=1.2)
    ax2.set_xlabel("WER", fontsize=10.5, color=INK)
    ax2.set_ylabel("channels", fontsize=10.5, color=INK)
    ax2.set_title(f"ASR health on synthetic TTS (median WER "
                  f"{df['wer'].median():.2f})", fontsize=11, color=INK,
                  fontweight="bold")
    fig.suptitle("F4: is ASR the bottleneck for annotator state quality?",
                 fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/f4_asr_attribution.png", dpi=300,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved f4_results.json + f4_asr_attribution.png")


if __name__ == "__main__":
    main()
