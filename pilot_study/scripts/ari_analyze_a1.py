"""A1: Behavior-SD 类内判别 — 声学特征能否区分"真重叠"与"渲染进停顿".

对同一事件类型 (BC 或 Int), 用 realized 标签 (声道级双活跃裁决) 做二分类:
  y=1: realized != "None" (事件窗口内确有双声道同时活跃段, 即真重叠)
  y=0: realized == "None" (无真实重叠, 即渲染进宿主停顿)

关键设计 — 分离"同义反复"与"真信号":
  energy_ratio 与 realized 标签同源于"双方声道在窗口内的能量", 二者相关近乎构造性.
  真正非循环的问题是: 只用事件声道的韵律特征 (f0_slope, spectral_centroid,
  voiced_ratio) 能否区分? 即"渲染在重叠里 vs 停顿里的 BC, 自身音色/韵律是否不同".

指标:
  - 全特征 / 无 energy_ratio / 仅事件声道特征 三组的 AUC (重复分层 5-fold CV, 20 种子)
  - 单特征 AUC (uni-variate)
  - 特征组间差异 (Cohen's d + bootstrap 95% CI)
  - K=2 KMeans ARI (平衡抽样)
  - 时长 (dur) 作为描述性对照 (非分类特征)

用法 (gpu02 持久步骤内):
  python scripts/ari_analyze_a1.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_features import FEATURE_COLS  # noqa: E402

OUT_DIR = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/analysis/ari"
FEAT = FEATURE_COLS
EVENT_ONLY_FEAT = ["f0_slope", "spectral_centroid", "voiced_ratio"]


def load_behavior():
    dfs = [pd.read_csv(f, keep_default_na=False)
           for f in sorted(glob.glob(f"{OUT_DIR}/behavior_events_w*.csv"))]
    df = pd.concat(dfs, ignore_index=True)
    for c in FEAT:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=FEAT)


def cv_auc(X, y, n_splits=5, n_seeds=20, seed=42):
    """重复分层 K-fold CV 的 AUC 均值±std."""
    aucs = []
    for s in range(seed, seed + n_seeds):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=s)
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=1000, random_state=s)
            clf.fit(sc.transform(X[tr]), y[tr])
            p = clf.predict_proba(sc.transform(X[te]))[:, 1]
            aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs))


def univariate_auc(df, y, col):
    aucs = []
    X = df[[col]].values
    for s in range(42, 62):
        skf = StratifiedKFold(5, shuffle=True, random_state=s)
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=1000, random_state=s)
            clf.fit(sc.transform(X[tr]), y[tr])
            p = clf.predict_proba(sc.transform(X[te]))[:, 1]
            aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs))


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.std() ** 2 + (nb - 1) * b.std() ** 2) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp


def bootstrap_d(g0, g1, B=1000, seed=42):
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(B):
        a = g0[rng.integers(0, len(g0), len(g0))]
        b = g1[rng.integers(0, len(g1), len(g1))]
        ds.append(cohens_d(a, b))
    ds = np.array(ds)
    return float(ds.mean()), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def analyze_class(df, cls):
    sub = df[df["cls"] == cls].copy()
    sub["y"] = (sub["realized"] != "None").astype(int)
    n0, n1 = int((sub["y"] == 0).sum()), int((sub["y"] == 1).sum())
    print(f"\n=== {cls}: realized=None (停顿内) n={n0}, "
          f"realized=overlap (真重叠) n={n1} ===")
    if min(n0, n1) < 200:
        print("样本不足, 跳过")
        return None

    X = sub[FEAT].values
    y = sub["y"].values

    out = {"n_gap": n0, "n_overlap": n1, "overlap_frac": n1 / (n0 + n1)}

    # 1. 全特征 / 无 energy_ratio / 仅事件声道特征
    auc_all = cv_auc(X, y)
    print(f"AUC all 5 feats: {auc_all[0]:.3f} ± {auc_all[1]:.3f}")
    no_er = [f for f in FEAT if f != "energy_ratio"]
    auc_noer = cv_auc(X[:, [FEAT.index(f) for f in no_er]], y)
    print(f"AUC w/o energy_ratio: {auc_noer[0]:.3f} ± {auc_noer[1]:.3f}")
    auc_eo = cv_auc(X[:, [FEAT.index(f) for f in EVENT_ONLY_FEAT]], y)
    print(f"AUC event-channel prosody only: {auc_eo[0]:.3f} ± {auc_eo[1]:.3f}")

    # 2. 单特征 AUC
    uni = {f: round(univariate_auc(sub, y, f), 3) for f in FEAT}
    print("univariate AUC:", uni)

    # 3. 组间差异
    g0, g1 = sub[sub["y"] == 0], sub[sub["y"] == 1]
    diffs = {}
    for f in FEAT + ["dur"]:
        d, lo, hi = bootstrap_d(g1[f].values, g0[f].values)
        diffs[f] = {"cohens_d": round(d, 3), "ci": [round(lo, 3), round(hi, 3)],
                    "mean_overlap": round(float(g1[f].mean()), 3),
                    "mean_gap": round(float(g0[f].mean()), 3)}
    print("Cohen's d (overlap - gap):",
          {k: v["cohens_d"] for k, v in diffs.items()})

    # 4. K=2 KMeans ARI (平衡)
    rng = np.random.default_rng(42)
    n_min = min(n0, n1)
    idx = np.concatenate([
        rng.choice(np.where(y == 0)[0], n_min, replace=False),
        rng.choice(np.where(y == 1)[0], n_min, replace=False)])
    Xs = StandardScaler().fit_transform(X[idx])
    km = KMeans(2, n_init=10, random_state=42).fit_predict(Xs)
    ari = adjusted_rand_score(y[idx], km)
    print(f"K=2 ARI (balanced, N={n_min}): {ari:.3f}")

    return {"auc_all": auc_all, "auc_no_energy_ratio": auc_noer,
            "auc_event_only": auc_eo, "univariate_auc": uni,
            "cohens_d": diffs, "ari_k2": float(ari), **out}


def main():
    df = load_behavior()
    print(f"behavior events: {len(df)}")
    res = {}
    for cls in ["BC", "Int"]:
        r = analyze_class(df, cls)
        if r is not None:
            # numpy 类型转 float 以便 json
            r["auc_all"] = [round(r["auc_all"][0], 4), round(r["auc_all"][1], 4)]
            r["auc_no_energy_ratio"] = [round(r["auc_no_energy_ratio"][0], 4),
                                        round(r["auc_no_energy_ratio"][1], 4)]
            r["auc_event_only"] = [round(r["auc_event_only"][0], 4),
                                   round(r["auc_event_only"][1], 4)]
            res[cls] = r
    with open(f"{OUT_DIR}/a1_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nsaved to a1_results.json")


if __name__ == "__main__":
    main()
