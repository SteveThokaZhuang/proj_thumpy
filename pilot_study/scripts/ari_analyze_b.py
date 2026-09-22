"""B: 特征归因与消融 — 主实验三分类 ARI 由哪些特征支撑.

对每个特征组在 CANDOR 与 Behavior-SD 上重跑平衡 bootstrap (B=200, N=1000/类):
  all              : 5 特征全量 (主实验口径)
  no_energy_ratio  : 去 log10 能量比
  energy_only      : 仅能量比
  event_only       : 仅事件声道韵律 (f0_slope, spectral_centroid, voiced_ratio)
  f0_only          : 仅 F0 组 (f0_correlation, f0_slope)
  no_f0            : 去 F0 组
  5 个单特征

预期: Behavior-SD 的 ARI 去 energy_ratio 后大幅坍缩 (静音捷径), CANDOR 变化不大.

输出: b_results.json + docs/pilot_study/figures/b_ablation.png

用法 (gpu02 持久步骤内):
  python scripts/ari_analyze_b.py [--b 200] [--n3 1000]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_features import FEATURE_COLS  # noqa: E402
from ari_analyze import (load_dataset, make_sampler, draw_balanced,  # noqa: E402
                         cluster_ari, DEFAULT_OUT)
from sklearn.preprocessing import StandardScaler

FIG_DIR = "/share/workspace3/zhuangruicen/proj-thumpy/docs/pilot_study/figures"
C_BLUE, C_ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, BASE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"

GROUPS = {
    "all": FEATURE_COLS,
    "no_energy_ratio": [f for f in FEATURE_COLS if f != "energy_ratio"],
    "energy_only": ["energy_ratio"],
    "event_only": ["f0_slope", "spectral_centroid", "voiced_ratio"],
    "f0_only": ["f0_correlation", "f0_slope"],
    "no_f0": [f for f in FEATURE_COLS if f not in ("f0_correlation", "f0_slope")],
    "energy_ratio": ["energy_ratio"],
    "f0_correlation": ["f0_correlation"],
    "f0_slope": ["f0_slope"],
    "spectral_centroid": ["spectral_centroid"],
    "voiced_ratio": ["voiced_ratio"],
}
ORDER = ["all", "no_energy_ratio", "energy_only", "event_only", "f0_only",
         "no_f0", "energy_ratio", "f0_correlation", "f0_slope",
         "spectral_centroid", "voiced_ratio"]


def run_group(df, sampler, cols, units, B, N3, seed):
    rng = np.random.default_rng(seed)
    ari = []
    for b in range(B):
        # draw_balanced 内部完成会话/文件级重抽样 + 平衡抽取
        sel = draw_balanced(sampler, {"BC": N3, "Int": N3, "None": N3}, rng)
        if sel is None:
            continue
        X = df.iloc[sel][cols].values.astype(np.float64)
        y = df.iloc[sel]["cls"].values
        Xs = StandardScaler().fit_transform(X)
        a, _ = cluster_ari(Xs, y, 3, seed + b)
        ari.append(a)
    ari = np.array(ari)
    return {"mean": float(ari.mean()),
            "ci_low": float(np.percentile(ari, 2.5)),
            "ci_high": float(np.percentile(ari, 97.5))}


def plot(res, fig_path):
    fig, ax = plt.subplots(figsize=(11.5, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    x = np.arange(len(ORDER))
    w = 0.36
    for i, (ds, color) in enumerate([("candor", C_BLUE), ("behavior", C_ORANGE)]):
        means = [res[g][ds]["mean"] for g in ORDER]
        lows = [res[g][ds]["ci_low"] for g in ORDER]
        highs = [res[g][ds]["ci_high"] for g in ORDER]
        err = [[m - l for m, l in zip(means, lows)],
               [h - m for h, m in zip(highs, means)]]
        ax.bar(x + (i - 0.5) * w, means, w, color=color,
               yerr=err, capsize=2.5,
               error_kw={"elinewidth": 1.2, "ecolor": INK, "capthick": 1.2},
               label=("CANDOR (human)" if ds == "candor" else "Behavior-SD (synthetic)"))
    ax.axhline(0, color=BASE, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, rotation=30, ha="right", fontsize=8.5, color=INK2)
    ax.set_ylabel("3-class ARI (mean ± 95% CI)", fontsize=10.5, color=INK)
    ax.set_title("Feature ablation: what carries the acoustic–label agreement",
                 fontsize=12, color=INK, fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=200)
    ap.add_argument("--n3", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default=DEFAULT_OUT)
    args = ap.parse_args()

    candor = load_dataset(f"{args.out_dir}/candor_events_w*.csv", "session")
    behavior = load_dataset(f"{args.out_dir}/behavior_events_w*.csv", "file_id")
    sam_c, sam_b = make_sampler(candor), make_sampler(behavior)
    c_units, b_units = candor["unit"].unique(), behavior["unit"].unique()
    print(f"loaded candor={len(candor)} behavior={len(behavior)}", flush=True)

    res = {}
    t0 = time.time()
    for g in ORDER:
        cols = GROUPS[g]
        rc = run_group(candor, sam_c, cols, c_units, args.b, args.n3, args.seed)
        rb = run_group(behavior, sam_b, cols, b_units, args.b, args.n3, args.seed)
        res[g] = {"candor": rc, "behavior": rb,
                  "delta": round(rc["mean"] - rb["mean"], 4)}
        print(f"{g}: candor={rc['mean']:.3f} [{rc['ci_low']:.3f},{rc['ci_high']:.3f}] "
              f"behavior={rb['mean']:.3f} [{rb['ci_low']:.3f},{rb['ci_high']:.3f}] "
              f"Δ={res[g]['delta']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    with open(f"{args.out_dir}/b_results.json", "w") as f:
        json.dump(res, f, indent=2)
    plot(res, f"{FIG_DIR}/b_ablation.png")
    print("saved b_results.json + b_ablation.png")


if __name__ == "__main__":
    main()
