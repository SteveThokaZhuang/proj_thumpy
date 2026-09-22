"""F: 报告补强数据 — 簇纯度 + 跨数据集分布偏移 (Cohen's d).

  1. 簇纯度: 从平衡抽样 ({candor,behavior}_draw.csv) 算 K=3 KMeans 的
     行归一混淆矩阵 (最优簇-类指派, 与 fig4 一致) + 各类召回率 + 平均簇纯度.
  2. 跨数据集每类每特征 Cohen's d (Behavior-SD - CANDOR, 事件级):
     量化"合成 vs 人类"的分布偏移, 直接支撑 C 迁移实验的解释.

输出: f_results.json

用法 (gpu02 持久步骤内):
  python scripts/ari_analyze_f.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_features import FEATURE_COLS  # noqa: E402
from ari_analyze import DEFAULT_OUT  # noqa: E402

CLS = ["BC", "Int", "None"]


def purities(draw_csv):
    df = pd.read_csv(draw_csv, keep_default_na=False)
    X = StandardScaler().fit_transform(df[FEATURE_COLS].values)
    y = df["cls"].map({c: i for i, c in enumerate(CLS)}).values
    pred = KMeans(3, n_init=10, random_state=42).fit_predict(X)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    ri, ci = linear_sum_assignment(-cm)
    cm = cm[:, ci]
    recall = cm.diagonal() / cm.sum(axis=1)
    # 各簇纯度 = 指派后各列的最大占比
    col_purity = cm.max(axis=0) / cm.sum(axis=0)
    return {"recall_per_class": {CLS[i]: round(float(recall[i]), 3)
                                 for i in range(3)},
            "cluster_purity_mean": round(float(col_purity.mean()), 3),
            "confusion_rownorm": [[round(float(cm[i, j] / cm[i].sum()), 3)
                                   for j in range(3)] for i in range(3)]}


def cross_dataset_d():
    """每类每特征的 Cohen's d (Behavior-SD - CANDOR)."""
    c = pd.concat([pd.read_csv(f, keep_default_na=False)
                   for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_events_w*.csv"))])
    b = pd.concat([pd.read_csv(f, keep_default_na=False)
                   for f in sorted(glob.glob(f"{DEFAULT_OUT}/behavior_events_w*.csv"))])
    out = {}
    for cls in CLS:
        bc, cc = b[b["cls"] == cls][FEATURE_COLS], c[c["cls"] == cls][FEATURE_COLS]
        row = {}
        for f in FEATURE_COLS:
            nb, nc = len(bc), len(cc)
            sp = np.sqrt(((nb - 1) * bc[f].std() ** 2 + (nc - 1) * cc[f].std() ** 2)
                         / (nb + nc - 2))
            row[f] = round(float((bc[f].mean() - cc[f].mean()) / sp), 3)
        out[cls] = row
    return out


def main():
    res = {}
    for name in ["candor", "behavior"]:
        res[f"purity_{name}"] = purities(f"{DEFAULT_OUT}/{name}_draw.csv")
        print(f"{name}: {res[f'purity_{name}']}", flush=True)
    res["cross_dataset_cohens_d"] = cross_dataset_d()
    print("cross-dataset Cohen's d (behavior - candor):",
          json.dumps(res["cross_dataset_cohens_d"], indent=1), flush=True)
    with open(f"{DEFAULT_OUT}/f_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("saved f_results.json")


if __name__ == "__main__":
    main()
