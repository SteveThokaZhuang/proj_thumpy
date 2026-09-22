"""D: 强化人类基线 — CANDOR 的 ARI 是被串扰/标签噪声限制, 还是特征本身不足.

C 实验发现 C→B 迁移 (0.636) 高于 CANDOR 域内 CV (0.524), 提示 CANDOR 的
低 ARI 很大程度受标签噪声限制. D 用两种方式强化人类基线:
  1. 串扰分层: 按会话级"None 事件 log10 能量比的中位数" (对方声道在无重叠
     turn 期间的能量 ≈ 串扰/背景指标, 越高 = 声道越干净) 把会话分三档,
     分别在干净/中等/嘈杂档上重算三分类 ARI;
  2. per-session z-score: 特征按会话内标准化 (去除说话人/响度/设备差异) 后重算;
  3. 组合: 干净档 + z-score.

同时给 BC-vs-Int 二分类 ARI (人类 BC 与 Int 的可分性上限).

输出: d_results.json + docs/pilot_study/figures/d_human_baseline.png

用法 (gpu02 持久步骤内):
  python scripts/ari_analyze_d.py [--b 200]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_features import FEATURE_COLS  # noqa: E402
from ari_analyze import load_dataset, make_sampler, draw_balanced, cluster_ari, DEFAULT_OUT  # noqa: E402
from sklearn.preprocessing import StandardScaler

FIG_DIR = "/share/workspace3/zhuangruicen/proj-thumpy/docs/pilot_study/figures"
C_BLUE = "#2a78d6"
INK, INK2, GRID, BASE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"


def ari_bootstrap(df, sampler, units, B, N3, seed, zscore_session=False):
    """三分类 ARI bootstrap; zscore_session=True 时按会话内标准化."""
    rng = np.random.default_rng(seed)
    ari = []
    for b in range(B):
        sel = draw_balanced(sampler, {"BC": N3, "Int": N3, "None": N3}, rng)
        if sel is None:
            continue
        sub = df.iloc[sel]
        X = sub[FEATURE_COLS].values.astype(np.float64)
        if zscore_session:
            # 按会话内标准化: 每个会话自己的均值/标准差 (仅特征列;
            # ddof=0 防单事件会话 std=NaN, std=0 -> 置 1 即仅中心化)
            def _zs(g):
                s = g.std(ddof=0)
                return (g - g.mean()) / (1.0 if s == 0 else s)
            X = (sub[FEATURE_COLS].groupby(sub["unit"])
                 .transform(_zs)).values
        Xs = StandardScaler().fit_transform(X)
        y = sub["cls"].values
        a, _ = cluster_ari(Xs, y, 3, seed + b)
        ari.append(a)
    ari = np.array(ari)
    return {"mean": float(ari.mean()),
            "ci_low": float(np.percentile(ari, 2.5)),
            "ci_high": float(np.percentile(ari, 97.5))}


def bcvint_ari(df, sampler, units, B, N, seed):
    """BC-vs-Int 二分类 ARI (K=2)."""
    rng = np.random.default_rng(seed)
    ari = []
    for b in range(B):
        sel = draw_balanced(sampler, {"BC": N, "Int": N}, rng)
        if sel is None:
            continue
        sub = df.iloc[sel]
        X = sub[FEATURE_COLS].values.astype(np.float64)
        Xs = StandardScaler().fit_transform(X)
        yb = np.where(sub["cls"].values == "BC", 0, 1)
        a, _ = cluster_ari(Xs, yb, 2, seed + b)
        ari.append(a)
    ari = np.array(ari)
    return {"mean": float(ari.mean()),
            "ci_low": float(np.percentile(ari, 2.5)),
            "ci_high": float(np.percentile(ari, 97.5))}


def plot(res, fig_path):
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=9.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    names = ["full", "clean", "mid", "dirty", "full_z", "clean_z"]
    labels = ["full 1656", "clean tercile", "mid tercile", "dirty tercile",
              "full + z-score", "clean + z-score"]
    means = [res[n]["mean"] for n in names]
    err = [[res[n]["mean"] - res[n]["ci_low"] for n in names],
           [res[n]["ci_high"] - res[n]["mean"] for n in names]]
    ax.bar(range(len(names)), means, 0.6, color=[C_BLUE] * 4 + ["#86b6ef"] * 2,
           yerr=err, capsize=3,
           error_kw={"elinewidth": 1.2, "ecolor": INK, "capthick": 1.2})
    for i, m in enumerate(means):
        ax.text(i, m + 0.008, f"{m:.3f}", ha="center", va="bottom",
                fontsize=10, color=INK, fontweight="bold")
    ax.axhline(0, color=BASE, linewidth=1)
    ax.set_xticks(range(len(names)), labels, rotation=12, ha="right",
                  fontsize=9.5, color=INK2)
    ax.set_ylabel("3-class ARI (mean ± 95% CI)", fontsize=10.5, color=INK)
    ax.set_ylim(0, max(means) + 0.06)
    ax.set_title("Strengthening the human baseline (CANDOR)",
                 fontsize=12, color=INK, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=200)
    ap.add_argument("--n3", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    candor = load_dataset(f"{DEFAULT_OUT}/candor_events_w*.csv", "session")
    print(f"candor events: {len(candor)}", flush=True)

    # 会话级串扰指标: None 事件的 energy_ratio 中位数
    none = candor[candor["cls"] == "None"]
    sess_clean = none.groupby("unit")["energy_ratio"].median()
    print("session cleanliness (median None energy_ratio): "
          f"min={sess_clean.min():.2f} q1={sess_clean.quantile(0.25):.2f} "
          f"med={sess_clean.median():.2f} q3={sess_clean.quantile(0.75):.2f} "
          f"max={sess_clean.max():.2f}", flush=True)
    q1, q3 = sess_clean.quantile(0.33), sess_clean.quantile(0.67)
    groups = {
        "clean": sess_clean[sess_clean >= q3].index,
        "mid": sess_clean[(sess_clean > q1) & (sess_clean < q3)].index,
        "dirty": sess_clean[sess_clean <= q1].index,
    }
    for g, units in groups.items():
        print(f"{g}: {len(units)} sessions", flush=True)

    res = {}
    for g in ["full", "clean", "mid", "dirty"]:
        sub = candor if g == "full" else candor[
            candor["unit"].isin(groups[g])].reset_index(drop=True)
        sam = make_sampler(sub)
        units = sub["unit"].unique()
        res[g] = ari_bootstrap(sub, sam, units, args.b, args.n3, args.seed)
        print(f"{g}: {res[g]}", flush=True)

    # z-score 变体
    sam = make_sampler(candor)
    res["full_z"] = ari_bootstrap(candor, sam, candor["unit"].unique(),
                                  args.b, args.n3, args.seed, zscore_session=True)
    print(f"full_z: {res['full_z']}", flush=True)
    sub_c = candor[candor["unit"].isin(groups["clean"])].reset_index(drop=True)
    sam_c = make_sampler(sub_c)
    res["clean_z"] = ari_bootstrap(sub_c, sam_c, sub_c["unit"].unique(),
                                   args.b, args.n3, args.seed, zscore_session=True)
    print(f"clean_z: {res['clean_z']}", flush=True)

    # BC-vs-Int 上限 (全量 + 干净档)
    res["bcvint_full"] = bcvint_ari(candor, sam, candor["unit"].unique(),
                                    args.b, 1000, args.seed)
    res["bcvint_clean"] = bcvint_ari(sub_c, sam_c, sub_c["unit"].unique(),
                                     args.b, 1000, args.seed)
    print(f"bcvint_full: {res['bcvint_full']} "
          f"bcvint_clean: {res['bcvint_clean']}", flush=True)

    with open(f"{DEFAULT_OUT}/d_results.json", "w") as f:
        json.dump({k: {kk: round(vv, 4) for kk, vv in v.items()}
                   for k, v in res.items()}, f, indent=2)
    plot(res, f"{FIG_DIR}/d_human_baseline.png")
    print("saved d_results.json + d_human_baseline.png")


if __name__ == "__main__":
    main()
