"""E1 分析: CANDOR 三套标签源的一致性矩阵.

同一批事件的三套标签:
  AWS     : backbiter/audiophile 派生 (cls: Int/None; BC 块为 BC 定义本身)
  VAD     : Silero VAD 区间规则 (vad_int: 起始 0.2s 内对方有语音; FD-Bench 同款工具)
  realized: 声道级 RMS 双活跃段 (Int/Backchannel/None)

分析:
  turn 块 (Int/None 事件):
    - AWS vs VAD vs realized 的两两 Cohen's κ (二分类 Int/None)
      realized 映射: Int->Int, Backchannel->排除(严格)/None(宽松), None->None
    - AWS x realized 完整交叉表 (揭示标签空间坍塌)
  BC 块 (BC 事件):
    - VAD 检出率 (vad_seen), 宿主活跃率 (host_active)
    - realized 分布 (真重叠/停顿内) x vad_seen / host_active 交叉

输出: e1_results.json + docs/pilot_study/figures/e1_label_matrix.png

用法 (gpu02 持久步骤内):
  python scripts/ari_e1_analyze.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_analyze import DEFAULT_OUT  # noqa: E402

FIG_DIR = "/share/workspace3/zhuangruicen/proj-thumpy/docs/pilot_study/figures"
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, BASE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"


def kappa_pair(a, b, mask):
    a = a[mask]
    b = b[mask]
    return cohen_kappa_score(a, b)


def analyze_turns(df):
    sub = df[df["cls"].isin(["Int", "None"])].copy()
    aws = (sub["cls"] == "Int").astype(int).values
    vad = sub["vad_int"].values.astype(int)
    # realized 二值: Int->1; None->0; Backchannel->严格排除 / 宽松归 0
    rz = sub["realized"].values
    strict = rz != "Backchannel"
    rz_strict = np.where(rz == "Int", 1, 0)
    rz_lenient = np.where(rz == "Int", 1, 0)

    out = {
        "n_turns": len(sub),
        "aws_int_frac": float(aws.mean()),
        "vad_int_frac": float(vad.mean()),
        "realized_int_frac": float((rz == "Int").mean()),
        "realized_bc_frac": float((rz == "Backchannel").mean()),
        "kappa": {
            "aws_vs_vad": round(kappa_pair(aws, vad, np.ones(len(aws), bool)), 4),
            "aws_vs_realized_strict": round(kappa_pair(aws, rz_strict, strict), 4),
            "aws_vs_realized_lenient": round(kappa_pair(aws, rz_lenient, np.ones(len(aws), bool)), 4),
            "vad_vs_realized_strict": round(kappa_pair(vad, rz_strict, strict), 4),
            "vad_vs_realized_lenient": round(kappa_pair(vad, rz_lenient, np.ones(len(aws), bool)), 4),
        },
        "crosstab_aws_x_realized": pd.crosstab(sub["cls"], sub["realized"],
                                               dropna=False).to_dict(),
    }
    print(json.dumps(out, indent=1, default=str), flush=True)
    return sub, out


def analyze_bc(df):
    sub = df[df["cls"] == "BC"].copy()
    out = {
        "n_bc": len(sub),
        "vad_seen_frac": round(float(sub["vad_seen"].mean()), 4),
        "host_active_frac": round(float(sub["host_active"].mean()), 4),
        "realized_dist": sub["realized"].value_counts().to_dict(),
        "crosstab_vadseen_x_realized": pd.crosstab(sub["vad_seen"], sub["realized"]).to_dict(),
        "crosstab_host_x_realized": pd.crosstab(sub["host_active"], sub["realized"]).to_dict(),
    }
    print(json.dumps(out, indent=1, default=str), flush=True)
    return sub, out


def plot(res_turns, res_bc, fig_path):
    fig = plt.figure(figsize=(11.5, 6.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.45, wspace=0.28)

    # 左: turn 块 κ 柱状
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    k = res_turns["kappa"]
    names = list(k.keys())
    vals = [k[n] for n in names]
    colors = [C_BLUE] * 2 + [C_ORANGE] * 2 + [C_AQUA]
    bars = ax.bar(range(len(names)), vals, 0.6, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", fontsize=9, color=INK)
    ax.axhline(0, color=BASE, linewidth=1)
    ax.set_xticks(range(len(names)),
                  ["AWS\nvs VAD", "AWS vs RZ\n(strict)", "AWS vs RZ\n(lenient)",
                   "VAD vs RZ\n(strict)", "VAD vs RZ\n(lenient)"],
                  fontsize=8, color=INK2)
    ax.set_ylabel("Cohen's κ (Int/None)", fontsize=9.5, color=INK)
    ax.set_ylim(-0.05, max(vals) + 0.12)
    ax.set_title("Turn block: pairwise agreement of three label sources",
                 fontsize=10.5, color=INK, fontweight="bold")

    # 右上: AWS x realized 交叉表热图
    ax = fig.add_subplot(gs[0, 1])
    ct = pd.DataFrame(res_turns["crosstab_aws_x_realized"])
    if "Int" in ct.index and "None" in ct.index:
        ct = ct.reindex(index=["Int", "None"],
                        columns=[c for c in ["Int", "Backchannel", "None"]
                                 if c in ct.columns], fill_value=0)
        ct = ct / ct.sum(axis=1).values[:, None]
        im = ax.imshow(ct.values, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(ct.shape[1]):
                ax.text(j, i, f"{ct.values[i, j]:.2f}", ha="center",
                        va="center", fontsize=10,
                        color="white" if ct.values[i, j] > 0.5 else INK2,
                        fontweight="bold" if ct.values[i, j] > 0.5 else "normal")
        ax.set_xticks(range(ct.shape[1]), ct.columns, fontsize=9, color=INK2)
        ax.set_yticks(range(2), ct.index, fontsize=9, color=INK2)
        ax.set_xlabel("realized (stereo RMS)", fontsize=9.5, color=INK)
        ax.set_ylabel("AWS label", fontsize=9.5, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_facecolor(SURFACE)
    ax.set_title("AWS × realized on turns (row-normalized)",
                 fontsize=10.5, color=INK, fontweight="bold")

    # 下左: BC 块 realized 分布
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    d = res_bc["realized_dist"]
    order = ["Backchannel", "None", "Int"]
    vals = [d.get(k, 0) / res_bc["n_bc"] for k in order]
    bars = ax.bar(range(3), vals, 0.55,
                  color=[C_AQUA, C_ORANGE, C_BLUE])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}",
                ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(3), ["true overlap\n(0.1-0.5s)", "no overlap\n(in gap)",
                             "long overlap\n(>0.5s)"], fontsize=8.5, color=INK2)
    ax.set_ylabel("fraction of AWS-BC events", fontsize=9.5, color=INK)
    ax.set_ylim(0, max(vals) + 0.08)
    ax.set_title("BC block: what stereo RMS says about AWS backchannels",
                 fontsize=10.5, color=INK, fontweight="bold")

    # 下右: BC 块 VAD 检出 x realized
    ax = fig.add_subplot(gs[1, 1])
    ct = pd.DataFrame(res_bc["crosstab_vadseen_x_realized"])
    ct = ct.reindex(index=[1, 0], columns=order, fill_value=0)
    ct = ct / ct.sum(axis=1).values[:, None]
    im = ax.imshow(ct.values, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(3):
            ax.text(j, i, f"{ct.values[i, j]:.2f}", ha="center",
                    va="center", fontsize=10,
                    color="white" if ct.values[i, j] > 0.5 else INK2,
                    fontweight="bold" if ct.values[i, j] > 0.5 else "normal")
    ax.set_xticks(range(3), ["overlap", "gap", "long"], fontsize=9, color=INK2)
    ax.set_yticks(range(2), ["VAD seen", "VAD missed"], fontsize=9, color=INK2)
    ax.set_xlabel("realized (stereo RMS)", fontsize=9.5, color=INK)
    ax.set_ylabel("Silero VAD", fontsize=9.5, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor(SURFACE)
    ax.set_title("BC block: VAD detection × realized (row-normalized)",
                 fontsize=10.5, color=INK, fontweight="bold")

    fig.suptitle("Label-source consistency matrix on CANDOR (E1)",
                 fontsize=13, color=INK, fontweight="bold")
    fig.savefig(fig_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    dfs = [pd.read_csv(f, keep_default_na=False)
           for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_e1_w*.csv"))]
    df = pd.concat(dfs, ignore_index=True)
    for c in ["vad_int", "vad_any", "vad_seen", "host_active"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"e1 rows: {len(df)}", flush=True)
    _, res_turns = analyze_turns(df)
    _, res_bc = analyze_bc(df)
    with open(f"{DEFAULT_OUT}/e1_results.json", "w") as f:
        json.dump({"turns": res_turns, "bc": res_bc}, f, indent=2, default=str)
    plot(res_turns, res_bc, f"{FIG_DIR}/e1_label_matrix.png")
    print("saved e1_results.json + e1_label_matrix.png")


if __name__ == "__main__":
    main()
