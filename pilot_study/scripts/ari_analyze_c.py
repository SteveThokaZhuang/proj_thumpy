"""C: 跨数据集迁移 — "静音捷径"在真实录音上是否失效.

设计:
  - 分类器: LogisticRegression (StandardScaler 在训练集上拟合后应用于测试集)
  - 任务: 三分类 (BC/Int/None) 与 BC-vs-rest; 特征: 全 5 特征 与 无 energy_ratio
  - 方向: Behavior-SD -> CANDOR (主), CANDOR -> Behavior-SD (对照),
    以及域内留出 CV (同数据集 80/20) 作为参照上界
  - 每种子: 训练集 = 源域平衡抽样 N=1000/类, 测试集 = 目标域平衡抽样 N=1000/类
  - 20 个种子, 报告 macro-F1 均值±std 与每类 F1

预期: Behavior-SD -> CANDOR 大幅退化 (静音捷径不迁移); 反向略好但仍差.

输出: c_results.json + docs/pilot_study/figures/c_transfer.png

用法 (gpu02 持久步骤内):
  python scripts/ari_analyze_c.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_features import FEATURE_COLS  # noqa: E402
from ari_analyze import load_dataset, make_sampler, draw_balanced, DEFAULT_OUT  # noqa: E402

FIG_DIR = "/share/workspace3/zhuangruicen/proj-thumpy/docs/pilot_study/figures"
NO_ER = [f for f in FEATURE_COLS if f != "energy_ratio"]
CLS = ["BC", "Int", "None"]
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, BASE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"


def draw(df, sampler, counts, rng):
    sel = draw_balanced(sampler, counts, rng)
    X = df.iloc[sel][FEATURE_COLS].values.astype(np.float64)
    y = df.iloc[sel]["cls"].values
    return X, y


def classify(Xtr, ytr, Xte, yte, cols, average="macro"):
    ci = [FEATURE_COLS.index(c) for c in cols]
    sc = StandardScaler().fit(Xtr[:, ci])
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(sc.transform(Xtr[:, ci]), ytr)
    p = clf.predict(sc.transform(Xte[:, ci]))
    return f1_score(yte, p, average=average)


def transfer(src, src_sampler, dst, dst_sampler, cols, n_seeds=20, seed=42,
             n_train=1000, n_test=1000, binary=False):
    """src -> dst 零样本迁移, macro-F1 均值±std."""
    rng = np.random.default_rng(seed)
    fs = []
    for s in range(seed, seed + n_seeds):
        r = np.random.default_rng(s)
        if binary:
            counts = {"BC": n_train, "Int": n_train // 2, "None": n_train // 2}
        else:
            counts = {c: n_train for c in CLS}
        Xtr, ytr = draw(src, src_sampler, counts, r)
        Xte, yte = draw(dst, dst_sampler, {c: n_test for c in CLS}, r)
        if binary:
            ytr_b = np.where(ytr == "BC", "BC", "rest")
            yte_b = np.where(yte == "BC", "BC", "rest")
            fs.append(classify(Xtr, ytr_b, Xte, yte_b, cols, average="binary"))
        else:
            fs.append(classify(Xtr, ytr, Xte, yte, cols))
    return float(np.mean(fs)), float(np.std(fs))


def in_domain_cv(df, cols, n_seeds=20, seed=42, n=3000, binary=False):
    """同域留出 CV (平衡 80/20), 作为参照上界."""
    rng = np.random.default_rng(seed)
    fs = []
    for s in range(seed, seed + n_seeds):
        r = np.random.default_rng(s)
        sel = draw_balanced(make_sampler(df), {c: n // 3 for c in CLS}, r)
        X = df.iloc[sel][FEATURE_COLS].values.astype(np.float64)
        y = df.iloc[sel]["cls"].values
        if binary:
            y = np.where(y == "BC", "BC", "rest")
        skf = StratifiedKFold(5, shuffle=True, random_state=s)
        ci = [FEATURE_COLS.index(c) for c in cols]
        f_ = []
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr][:, ci])
            clf = LogisticRegression(max_iter=1000)
            clf.fit(sc.transform(X[tr][:, ci]), y[tr])
            p = clf.predict(sc.transform(X[te][:, ci]))
            f_.append(f1_score(y[te], p, average="binary" if binary else "macro"))
        fs.append(float(np.mean(f_)))
    return float(np.mean(fs)), float(np.std(fs))


def plot(res, fig_path):
    fig, ax = plt.subplots(figsize=(9.5, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=9.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    names = ["b2c", "c2b", "b_in", "c_in", "b2c_noer", "b_in_noer"]
    labels = ["Behavior-SD → CANDOR", "CANDOR → Behavior-SD",
              "Behavior-SD in-domain CV", "CANDOR in-domain CV",
              "B→C (no energy_ratio)", "Behavior-SD in-domain (no ER)"]
    colors = [C_BLUE, C_ORANGE, C_BLUE, C_ORANGE, C_AQUA, C_AQUA]
    means = [res[n][0] for n in names]
    stds = [res[n][1] for n in names]
    bars = ax.bar(range(len(names)), means, 0.6, color=colors,
                  yerr=stds, capsize=3,
                  error_kw={"elinewidth": 1.2, "ecolor": INK, "capthick": 1.2})
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.02, f"{m:.2f}",
                ha="center", va="bottom", fontsize=10,
                color=INK, fontweight="bold")
    ax.axhline(1 / 3, color=BASE, linewidth=1, linestyle="--")
    ax.text(-0.4, 1 / 3 + 0.01, "chance (3-class macro-F1 = 1/3)",
            fontsize=8.5, color=INK2, va="bottom")
    ax.set_xticks(range(len(names)), labels, rotation=18, ha="right",
                  fontsize=9, color=INK2)
    ax.set_ylabel("macro-F1 (20 seeds, mean ± std)", fontsize=10.5, color=INK)
    ax.set_ylim(0, max(means) + 0.15)
    ax.set_title("Cross-dataset transfer of BC/Int/None classification",
                 fontsize=12, color=INK, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    candor = load_dataset(f"{DEFAULT_OUT}/candor_events_w*.csv", "session")
    behavior = load_dataset(f"{DEFAULT_OUT}/behavior_events_w*.csv", "file_id")
    sam_c, sam_b = make_sampler(candor), make_sampler(behavior)
    print(f"loaded candor={len(candor)} behavior={len(behavior)}", flush=True)

    res = {}
    res["b2c"] = transfer(behavior, sam_b, candor, sam_c, FEATURE_COLS)
    print(f"B->C 3-class: {res['b2c'][0]:.3f} ± {res['b2c'][1]:.3f}", flush=True)
    res["c2b"] = transfer(candor, sam_c, behavior, sam_b, FEATURE_COLS)
    print(f"C->B 3-class: {res['c2b'][0]:.3f} ± {res['c2b'][1]:.3f}", flush=True)
    res["b_in"] = in_domain_cv(behavior, FEATURE_COLS)
    print(f"B in-domain: {res['b_in'][0]:.3f} ± {res['b_in'][1]:.3f}", flush=True)
    res["c_in"] = in_domain_cv(candor, FEATURE_COLS)
    print(f"C in-domain: {res['c_in'][0]:.3f} ± {res['c_in'][1]:.3f}", flush=True)
    res["b2c_noer"] = transfer(behavior, sam_b, candor, sam_c, NO_ER)
    print(f"B->C no-ER: {res['b2c_noer'][0]:.3f} ± {res['b2c_noer'][1]:.3f}", flush=True)
    res["b_in_noer"] = in_domain_cv(behavior, NO_ER)
    print(f"B in-domain no-ER: {res['b_in_noer'][0]:.3f} ± {res['b_in_noer'][1]:.3f}", flush=True)

    with open(f"{DEFAULT_OUT}/c_results.json", "w") as f:
        json.dump({k: [round(v[0], 4), round(v[1], 4)] for k, v in res.items()},
                  f, indent=2)
    plot(res, f"{FIG_DIR}/c_transfer.png")
    print("saved c_results.json + c_transfer.png")


if __name__ == "__main__":
    main()
