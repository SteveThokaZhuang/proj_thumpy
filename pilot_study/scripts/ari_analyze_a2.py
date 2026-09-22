"""A2: 连续化敏感性检验 — A1 结论的稳健性.

A1 (类内判别) 依赖特定 realized 裁决阈值 (0.1x 声道最大 RMS, 0.1-0.5s 跨度).
A2 回答两个问题:
  1. 连续量: 各特征 (含 dur) 与 both_active_frac 的相关性 (Spearman + bootstrap CI)?
     预期 energy_ratio 强负相关 (构造性), 事件声道韵律特征 ≈ 0.
  2. 阈值敏感性: y = (frac > τ), τ ∈ {0, 0.05, 0.1, 0.2, 0.3}, 三组特征
     (全特征 / 无 energy_ratio / 仅事件声道韵律) 的 AUC 曲线是否随 τ 变化?
     若韵律特征在任何 τ 下都 ≈ 随机 → A1 结论不依赖阈值选择.

输出: a2_results.json + docs/pilot_study/figures/a2_threshold_auc.png

用法 (gpu02 持久步骤内):
  python scripts/ari_analyze_a2.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_features import FEATURE_COLS  # noqa: E402

OUT_DIR = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/analysis/ari"
FIG_DIR = "/share/workspace3/zhuangruicen/proj-thumpy/docs/pilot_study/figures"
FEAT = FEATURE_COLS
EVENT_ONLY = ["f0_slope", "spectral_centroid", "voiced_ratio"]
TAUS = [0.0, 0.05, 0.1, 0.2, 0.3]

C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, BASE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"


def bootstrap_spearman(a, b, B=500, seed=42):
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(B):
        i = rng.integers(0, len(a), len(a))
        rs.append(spearmanr(a[i], b[i]).statistic)
    rs = np.array(rs)
    return float(np.mean(rs)), float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def cv_auc(X, y, n_seeds=20, seed=42):
    aucs = []
    for s in range(seed, seed + n_seeds):
        skf = StratifiedKFold(5, shuffle=True, random_state=s)
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=1000, random_state=s)
            clf.fit(sc.transform(X[tr]), y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def analyze_class(df, cls):
    sub = df[df["cls"] == cls].copy()
    frac = sub["both_active_frac"].values
    print(f"\n=== {cls}: n={len(sub)}, frac mean={frac.mean():.3f}, "
          f"frac>0: {(frac > 0).mean():.2%} ===")

    out = {"n": len(sub), "frac_mean": float(frac.mean()),
           "frac_positive": float((frac > 0).mean())}

    # 1. Spearman 相关 (特征与 frac)
    corr = {}
    for f in FEAT + ["dur"]:
        m, lo, hi = bootstrap_spearman(sub[f].values, frac)
        corr[f] = {"r": round(m, 3), "ci": [round(lo, 3), round(hi, 3)]}
    print("Spearman r(feature, frac):",
          {k: v["r"] for k, v in corr.items()})
    out["spearman"] = corr

    # 2. 阈值敏感性
    groups = {
        "all": FEAT,
        "no_energy_ratio": [f for f in FEAT if f != "energy_ratio"],
        "event_only": EVENT_ONLY,
    }
    curves = {g: [] for g in groups}
    for tau in TAUS:
        y = (frac > tau).astype(int)
        n_pos = int(y.sum())
        if n_pos < 50 or len(y) - n_pos < 50:
            curves_tau = {g: None for g in groups}
        else:
            curves_tau = {}
            for g, cols in groups.items():
                X = sub[cols].values
                m, s = cv_auc(X, y)
                curves_tau[g] = [round(m, 4), round(s, 4)]
        for g in groups:
            curves[g].append(curves_tau[g])
    for g in groups:
        print(f"AUC({g}) vs tau:", [(t, c[0] if c else None)
                                   for t, c in zip(TAUS, curves[g])])
    out["threshold_curves"] = {"taus": TAUS, "curves": curves}
    return out


def plot_curves(res, fig_path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor=SURFACE)
    labels = {"all": "all 5 features", "no_energy_ratio": "w/o energy_ratio",
              "event_only": "event-channel prosody only"}
    colors = {"all": C_BLUE, "no_energy_ratio": C_ORANGE, "event_only": C_AQUA}
    for ax, cls in zip(axes, ["BC", "Int"]):
        ax.set_facecolor(SURFACE)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(BASE)
        ax.tick_params(colors=INK2, labelsize=9.5)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for g, label in labels.items():
            cs = res[cls]["threshold_curves"]["curves"][g]
            means = [c[0] if c else np.nan for c in cs]
            ax.plot(TAUS, means, "o-", color=colors[g], label=label,
                    markersize=4.5, linewidth=1.6)
        ax.axhline(0.5, color=BASE, linewidth=1, linestyle="--")
        ax.set_xlabel("overlap-fraction threshold τ (y = frac > τ)",
                      fontsize=10, color=INK)
        ax.set_ylabel("AUC (5-fold CV × 20)", fontsize=10, color=INK)
        ax.set_ylim(0.45, 0.95)
        ax.set_title(f"{cls} events: true-overlap vs in-gap",
                     fontsize=11.5, color=INK, fontweight="bold")
        if cls == "BC":
            ax.legend(frameon=False, fontsize=9)
    fig.suptitle("A1 discriminability is stable across realized-label thresholds",
                 fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    dfs = [pd.read_csv(f, keep_default_na=False)
           for f in sorted(glob.glob(f"{OUT_DIR}/behavior_events_w*.csv"))]
    df = pd.concat(dfs, ignore_index=True)
    for c in FEAT:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEAT)
    print(f"behavior events: {len(df)}")

    res = {}
    for cls in ["BC", "Int"]:
        res[cls] = analyze_class(df, cls)
    with open(f"{OUT_DIR}/a2_results.json", "w") as f:
        json.dump(res, f, indent=2)
    plot_curves(res, f"{FIG_DIR}/a2_threshold_auc.png")
    print("\nsaved a2_results.json + a2_threshold_auc.png")


if __name__ == "__main__":
    main()
