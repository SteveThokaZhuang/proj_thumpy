"""E2 分析: FD-Bench 区间规则的标签源敏感性.

问题: 同样的区间规则, 喂 GT 时间戳 vs 喂 VAD 片段 (干净/10%串扰/30%串扰),
SIR/EIR/NIR 漂移多大? GT "打断"事件有多少真实重叠 (声道级 realized)?

输出: e2_results.json + docs/pilot_study/figures/e2_fdbench_audit.png

用法 (gpu02 持久步骤内):
  python scripts/ari_e2_analyze.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_analyze import DEFAULT_OUT  # noqa: E402

FIG_DIR = "/share/workspace3/zhuangruicen/proj-thumpy/docs/pilot_study/figures"
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, BASE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
METRICS = ["sir", "eir", "nir"]


def main():
    dfs = [pd.read_csv(f, keep_default_na=False)
           for f in sorted(glob.glob(f"{DEFAULT_OUT}/e2_rows_w*.csv"))]
    df = pd.concat(dfs, ignore_index=True)
    for c in METRICS + ["n_round", "n_interrupt", "n_succ", "n_wrong", "n_noise",
                        "gt_overlap_frac"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"rows: {len(df)}, files: {df.file_id.nunique()}", flush=True)

    # 1. 各来源指标分布
    res = {}
    sources = ["GT", "VAD", "VAD_xt10", "VAD_xt30"]
    for src in sources:
        sub = df[df["source"] == src]
        res[src] = {
            "n_files": int(sub["file_id"].nunique()),
            "n_round_mean": round(float(sub["n_round"].mean()), 1),
            "sir_mean": round(float(sub["sir"].mean()), 4),
            "eir_mean": round(float(sub["eir"].mean()), 4),
            "nir_mean": round(float(sub["nir"].mean()), 4),
        }
        print(src, res[src], flush=True)

    # 2. per-file 指标漂移 (VAD 系 vs GT)
    pivot = df.pivot(index="file_id", columns="source", values=METRICS)
    drifts = {}
    for src in ["VAD", "VAD_xt10", "VAD_xt30"]:
        for m in METRICS:
            delta = (pivot[(m, src)] - pivot[(m, "GT")]).dropna()
            drifts[f"{m}_delta_{src}"] = {
                "mae": round(float(delta.abs().mean()), 4),
                "mean": round(float(delta.mean()), 4),
                "corr": round(float(pivot[(m, src)].corr(pivot[(m, "GT")])), 4),
            }
    print("drifts:", json.dumps(drifts, indent=1), flush=True)
    res["drifts"] = drifts

    # 3. GT 打断的真实重叠率
    gt = df[df["source"] == "GT"]["gt_overlap_frac"].dropna()
    res["gt_overlap_frac"] = {
        "n_files": int(len(gt)),
        "mean": round(float(gt.mean()), 4),
        "median": round(float(gt.median()), 4),
    }
    print("gt_overlap_frac:", res["gt_overlap_frac"], flush=True)

    with open(f"{DEFAULT_OUT}/e2_results.json", "w") as f:
        json.dump(res, f, indent=2)

    # 图
    fig = plt.figure(figsize=(11.5, 5.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.3)

    # 左上: 各来源指标均值
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    x = np.arange(3)
    w = 0.2
    for i, src in enumerate(sources):
        vals = [res[src][f"{m}_mean"] for m in METRICS]
        ax.bar(x + (i - 1.5) * w, vals, w, label=src,
               color=[C_BLUE, C_ORANGE, C_AQUA, "#eda100"][i])
    ax.set_xticks(x, ["SIR", "EIR", "NIR"], fontsize=10, color=INK2)
    ax.set_ylabel("mean over files", fontsize=10, color=INK)
    ax.set_title("FD-Bench metrics under four segment sources",
                 fontsize=11, color=INK, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5)

    # 右上: per-file SIR 漂移分布 (VAD vs GT)
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    data = []
    labels = []
    for src in ["VAD", "VAD_xt10", "VAD_xt30"]:
        d = (pivot[("sir", src)] - pivot[("sir", "GT")]).dropna()
        data.append(d.values)
        labels.append(src)
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops={"color": INK},
                    boxprops={"facecolor": "#cde2fb", "color": BASE},
                    whiskerprops={"color": BASE}, capprops={"color": BASE})
    ax.axhline(0, color=BASE, linewidth=1)
    ax.set_ylabel("per-file ΔSIR (source − GT)", fontsize=10, color=INK)
    ax.set_title("Per-file SIR drift vs GT timestamps",
                 fontsize=11, color=INK, fontweight="bold")

    # 下左: GT 打断真实重叠率直方图
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.hist(gt.values, bins=20, color=C_BLUE, alpha=0.85, edgecolor="white",
            linewidth=0.3)
    ax.axvline(gt.median(), color=INK, linestyle="--", linewidth=1.2)
    ax.set_xlabel("fraction of GT interruptions with real overlap",
                  fontsize=10, color=INK)
    ax.set_ylabel("files", fontsize=10, color=INK)
    ax.set_title("Do GT 'interruptions' actually overlap? (stereo RMS)",
                 fontsize=11, color=INK, fontweight="bold")

    # 右下: 文本小结
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    txt = (
        f"MAE of ΔSIR (VAD vs GT): {drifts['sir_delta_VAD']['mae']:.3f}\n"
        f"MAE of ΔSIR (VAD_xt30 vs GT): {drifts['sir_delta_VAD_xt30']['mae']:.3f}\n"
        f"corr(SIR_VAD, SIR_GT): {drifts['sir_delta_VAD']['corr']:.3f}\n"
        f"GT interruptions w/ real overlap: mean {res['gt_overlap_frac']['mean']:.2%}"
    )
    ax.text(0.05, 0.6, txt, fontsize=10.5, color=INK, va="top",
            family="monospace")

    fig.suptitle("E2: FD-Bench interval-rule audit (Behavior-SD val+test)",
                 fontsize=13, color=INK, fontweight="bold")
    fig.savefig(f"{FIG_DIR}/e2_fdbench_audit.png", dpi=300,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved e2_results.json + e2_fdbench_audit.png")


if __name__ == "__main__":
    main()
