"""ARI 实验: 可视化 (论文/报告风格 matplotlib PNG).

调色板 (dataviz skill 已验证默认 palette, light 模式):
  分类 slot1 蓝 #2a78d6 / slot2 橙 #eb6834 / slot3 aqua #1baf7a
  墨色: primary #0b0b0b, secondary #52514e, muted #898781,
        gridline #e1e0d9, baseline #c3c2b7, 表面 #fcfcfb
  顺序色: 蓝 100-700

图:
  1. ari_comparison.png   : 3 指标 (3-class / BC-vs-rest / BC-vs-Int) x 2 数据集
                           柱状 + 95% CI + 基线 + p 标注
  2. delta_hist.png       : ΔARI bootstrap 分布 + CI 阴影
  3. pca_scatter.png      : 平衡抽样 PCA 散点 (CANDOR / Behavior-SD 两面板)
  4. confusion.png        : 两数据集 K=3 混淆矩阵 (行归一)
  5. realized_control.png : Behavior-SD generation vs realized 标签 ARI

用法:
  python ari_visualize.py [--results ari_bootstrap_results.json]
      [--out-dir real_data/results/analysis/ari] [--fig-dir docs/pilot_study/figures]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_features import FEATURE_COLS  # noqa: E402

# ---- palette (validated, light) ----
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE_RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
CLS_COLORS = {"BC": C_BLUE, "Int": C_ORANGE, "None": C_AQUA}
CLS_LABELS = {"BC": "BC", "Int": "Int", "None": "None"}
DS_LABELS = {"candor": "CANDOR (human)", "behavior": "Behavior-SD (synthetic)"}
DS_COLORS = {"candor": C_BLUE, "behavior": C_ORANGE}


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig1_ari_comparison(res, fig_dir):
    r = res
    groups = ["ari3", "ari_bcvrest", "ari_bcvint"]
    gnames = ["3-class\n(BC/Int/None)", "BC vs rest", "BC vs Int"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    style_ax(ax)
    x = np.arange(len(groups))
    w = 0.34
    for i, (ds, color) in enumerate(DS_COLORS.items()):
        means = [r[ds][g]["mean"] for g in groups]
        lows = [r[ds][g]["ci_low"] for g in groups]
        highs = [r[ds][g]["ci_high"] for g in groups]
        err = [[m - l for m, l in zip(means, lows)],
               [h - m for h, m in zip(highs, means)]]
        bars = ax.bar(x + (i - 0.5) * w, means, w, color=color,
                      yerr=err, capsize=3, error_kw={"elinewidth": 1.5,
                                                     "ecolor": INK, "capthick": 1.5})
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m + 0.012, f"{m:.2f}",
                    ha="center", va="bottom", fontsize=10,
                    color=INK, fontweight="bold")
    # 3-class 组标注 Δ 与 p (主指标; 二分类指标仅展示均值+CI)
    d3 = r["delta_ari3"]
    ymax = max(r["candor"]["ari3"]["ci_high"], r["behavior"]["ari3"]["ci_high"]) + 0.08
    ax.plot([x[0] - w, x[0] + w], [ymax, ymax], color=INK, linewidth=1.2)
    p = d3["p_two_sided"]
    ps = f"p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.text(x[0], ymax + 0.012, f"ΔARI = {d3['mean']:.2f} [{d3['ci_low']:.2f}, {d3['ci_high']:.2f}], {ps}",
            ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.axhline(0, color=BASE, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(gnames, fontsize=10, color=INK2)
    ax.set_ylabel("ARI (mean ± 95% CI)", fontsize=11, color=INK)
    ax.set_ylim(0, ymax + 0.10)
    ax.set_title("Acoustic–label agreement: human vs synthetic dialogue",
                 fontsize=12, color=INK, fontweight="bold", pad=10)
    bars_c = [Rectangle((0, 0), 1, 1, color=c) for c in DS_COLORS.values()]
    ax.legend(bars_c, [DS_LABELS[k] for k in DS_COLORS],
              loc="upper right", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/ari_comparison.png", dpi=300,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def fig2_delta_hist(res, fig_dir):
    delta = np.array(res["_reps"]["delta_ari3"])
    d = res["delta_ari3"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8), facecolor=SURFACE)
    style_ax(ax)
    ax.hist(delta, bins=40, color=C_BLUE, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvspan(d["ci_low"], d["ci_high"], color=C_AQUA, alpha=0.18)
    ax.axvline(0, color=BASE, linewidth=1)
    ax.axvline(d["mean"], color=INK, linewidth=1.5, linestyle="--")
    ax.text(0.02, 0.96,
            f"ΔARI = {d['mean']:.2f}  95% CI [{d['ci_low']:.2f}, {d['ci_high']:.2f}]",
            transform=ax.transAxes, va="top", fontsize=10, color=INK)
    ax.set_xlabel("ΔARI (CANDOR − Behavior-SD), 500 session-level bootstrap replicates",
                  fontsize=10.5, color=INK)
    ax.set_ylabel("Replicates", fontsize=10.5, color=INK)
    ax.set_title("Bootstrap distribution of the consistency gap",
                 fontsize=12, color=INK, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/delta_hist.png", dpi=300,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def fig3_pca_scatter(out_dir, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    for ax, name in zip(axes, ["candor", "behavior"]):
        style_ax(ax)
        df = pd.read_csv(f"{out_dir}/{name}_draw.csv", keep_default_na=False)
        X = df[FEATURE_COLS].values
        Xs = StandardScaler().fit_transform(X)
        Z = PCA(n_components=2, random_state=42).fit_transform(Xs)
        for cls, color in CLS_COLORS.items():
            m = df["cls"] == cls
            ax.scatter(Z[m, 0], Z[m, 1], s=6, color=color, alpha=0.45,
                       label=CLS_LABELS[cls], linewidths=0)
        ax.set_xlabel("PC1", fontsize=10.5, color=INK)
        if name == "candor":
            ax.set_ylabel("PC2", fontsize=10.5, color=INK)
        ax.set_title(DS_LABELS[name], fontsize=11.5, color=INK, fontweight="bold")
        if name == "behavior":
            ax.legend(markerscale=3, frameon=False, fontsize=9.5,
                      handletextpad=0.2, labelspacing=0.6)
    fig.suptitle("Balanced draws (N=1000/class) in feature space (PCA)",
                 fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/pca_scatter.png", dpi=300,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def optimal_confusion(y_true, y_pred, n_classes=3):
    """最优簇-类指派后的行归一混淆矩阵 (输入均为 0..n-1 整数)."""
    import sklearn.metrics as M
    cm = M.confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    ri, ci = linear_sum_assignment(-cm)
    cm = cm[:, ci]  # 列重排: 簇 -> 指派类
    return cm.astype(float) / cm.sum(axis=1, keepdims=True)


CLS_ORDER = ["BC", "Int", "None"]


def fig4_confusion(out_dir, fig_dir, seed=42):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=SURFACE)
    for ax, name in zip(axes, ["candor", "behavior"]):
        df = pd.read_csv(f"{out_dir}/{name}_draw.csv", keep_default_na=False)
        X = StandardScaler().fit_transform(df[FEATURE_COLS].values)
        y = df["cls"].map({c: i for i, c in enumerate(CLS_ORDER)}).values
        pred = KMeans(n_clusters=3, n_init=10, random_state=seed).fit_predict(X)
        cm = optimal_confusion(y, pred)
        im = ax.imshow(cm, cmap=matplotlib.colors.ListedColormap(BLUE_RAMP),
                       vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                v = cm[i, j]
                color = "white" if v > 0.5 else INK2
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=11, color=color,
                        fontweight="bold" if v > 0.5 else "normal")
        ax.set_xticks(range(3), CLS_ORDER, fontsize=10, color=INK2)
        ax.set_yticks(range(3), CLS_ORDER, fontsize=10, color=INK2)
        ax.set_xlabel("Acoustic cluster", fontsize=10.5, color=INK)
        if name == "candor":
            ax.set_ylabel("Annotation label", fontsize=10.5, color=INK)
        ax.set_title(DS_LABELS[name], fontsize=11.5, color=INK, fontweight="bold")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_facecolor(SURFACE)
    fig.suptitle("K=3 clustering vs annotation labels (row-normalized)",
                 fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/confusion.png", dpi=300,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def fig5_realized_control(res, fig_dir):
    gen = res["behavior"]["ari3"]
    rz = res["behavior_realized_ari3"]
    fig, ax = plt.subplots(figsize=(5.4, 4.0), facecolor=SURFACE)
    style_ax(ax)
    labels = ["Generation labels", "Realized labels\n(stereo overlap)"]
    means = [gen["mean"], rz["mean"]]
    err = [[gen["mean"] - gen["ci_low"], rz["mean"] - rz["ci_low"]],
           [gen["ci_high"] - gen["mean"], rz["ci_high"] - rz["mean"]]]
    bars = ax.bar([0, 1], means, 0.5, color=[C_ORANGE, C_BLUE],
                  yerr=err, capsize=3,
                  error_kw={"elinewidth": 1.5, "ecolor": INK, "capthick": 1.5})
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.012, f"{m:.2f}",
                ha="center", va="bottom", fontsize=10.5,
                color=INK, fontweight="bold")
    ax.set_xticks([0, 1], labels, fontsize=10, color=INK2)
    ax.set_ylabel("3-class ARI (mean ± 95% CI)", fontsize=10.5, color=INK)
    ax.set_ylim(0, max(gen["ci_high"], rz["ci_high"]) + 0.1)
    ax.set_title("Behavior-SD: same acoustic features,\ntwo label sources",
                 fontsize=11.5, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/realized_control.png", dpi=300,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str,
                    default="/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
                            "real_data/results/analysis/ari/ari_bootstrap_results.json")
    ap.add_argument("--out-dir", type=str,
                    default="/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
                            "real_data/results/analysis/ari")
    ap.add_argument("--fig-dir", type=str,
                    default="/share/workspace3/zhuangruicen/proj-thumpy/"
                            "docs/pilot_study/figures")
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    res = json.load(open(args.results))
    fig1_ari_comparison(res, args.fig_dir)
    fig2_delta_hist(res, args.fig_dir)
    fig3_pca_scatter(args.out_dir, args.fig_dir)
    fig4_confusion(args.out_dir, args.fig_dir)
    fig5_realized_control(res, args.fig_dir)
    print("figures saved to", args.fig_dir)


if __name__ == "__main__":
    main()
