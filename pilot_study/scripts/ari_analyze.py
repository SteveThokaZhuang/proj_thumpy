"""ARI 实验: bootstrap 分析 (numpy 直抽版).

对比 CANDOR(人类) vs Behavior-SD(合成) 的声学-语义一致性 (ARI), 定位为
"可视化人类 vs 合成数据的声学-语义一致性差距", 而非验证渲染失真.

设计 (修复版, 2026-08-24):
- 类别平衡抽样 (消除类别分布对 ARI 天花板的影响)
- 以 会话(CANDOR)/文件(Behavior-SD) 为抽样单位 bootstrap (B=500)
- 主指标: K=3 三分类 ARI (每类 N3); 次指标: BC-vs-rest (N2:N2) 与
  BC-vs-Int (N2:N2) 二分类 ARI
- permutation 基线 (打乱标签, 同 X)
- Behavior-SD 内部对照: 与 generation 标签**同一次抽样**上计算 realized
  标签 (声道级 RMS 重叠裁决, 与 verify_backchannel.py 同口径) 的 ARI
- ΔARI 的 95% CI 与 p 值

性能: (unit, cls) -> 行位置索引预建一次; 每轮 bootstrap 只做 numpy 操作
(旧版每轮对 20 万行做 isin/sample, B=500 需 30+ CPU 分钟; 本版 ~30s).

用法:
  python ari_analyze.py [--b 500] [--n3 1000] [--n2 1000] [--out-dir ...]
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_features import FEATURE_COLS  # noqa: E402

DEFAULT_OUT = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/analysis/ari"


def load_dataset(glob_pattern, unit_col):
    """合并分片 CSV; 注意 keep_default_na=False (cls='None' 会被 pandas 误判为 NaN)."""
    dfs = [pd.read_csv(f, keep_default_na=False)
           for f in sorted(glob.glob(glob_pattern))]
    df = pd.concat(dfs, ignore_index=True)
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS)
    df["unit"] = df[unit_col]
    return df


def make_sampler(df):
    """预建 (unit, cls) -> 行位置数组 的索引, 及 unit 列表."""
    groups = {}
    for (u, c), idx in df.groupby(["unit", "cls"]).groups.items():
        groups[(u, c)] = idx.values
    return groups, df["unit"].unique()


def draw_balanced(sampler, counts, rng):
    """bootstrap 重抽样 units 后按类别计数抽取事件; 返回 (X, y, positions).

    速度关键: 每轮只做 numpy 操作 (~20ms).
    """
    groups, unit_list = sampler
    units, mult = np.unique(
        rng.choice(unit_list, size=len(unit_list), replace=True),
        return_counts=True)
    parts = []
    for cls, n in counts.items():
        pools = []
        for u, m in zip(units, mult):
            g = groups.get((u, cls))
            if g is not None and len(g):
                pools.append(np.repeat(g, m))
        if not pools:
            return None, None, None
        pos = np.concatenate(pools)
        n = min(n, len(pos))
        parts.append(rng.choice(pos, n, replace=False))
    sel = np.concatenate(parts)
    return sel


def draw_many(sampler, df, counts, rng):
    sel = draw_balanced(sampler, counts, rng)
    if sel is None:
        return None, None, None
    X = df.iloc[sel][FEATURE_COLS].values.astype(np.float64)
    y = df.iloc[sel]["cls"].values
    return X, y, sel


def cluster_ari(X, y, k, seed):
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    return adjusted_rand_score(y, km.labels_), km


def run_bootstrap(candor, behavior, B, N3, N2, seed, out_dir):
    sam_c, sam_b = make_sampler(candor), make_sampler(behavior)
    rng = np.random.default_rng(seed)

    keys = ["ari3", "ari3_perm", "ari3_sil", "ari_bcvrest", "ari_bcvint"]
    res = {
        "candor": {k: [] for k in keys},
        "behavior": {k: [] for k in keys},
        "behavior_realized_ari3": [],
    }
    counts3 = {"BC": N3, "Int": N3, "None": N3}
    t0 = time.time()

    for b in range(B):
        for name, (sam, df) in [("candor", (sam_c, candor)),
                                ("behavior", (sam_b, behavior))]:
            # 3-class (每类 N3)
            X, y, sel = draw_many(sam, df, counts3, rng)
            if X is None:
                continue
            Xs = StandardScaler().fit_transform(X)
            ari3, km = cluster_ari(Xs, y, 3, seed + b)
            res[name]["ari3"].append(ari3)
            res[name]["ari3_sil"].append(
                silhouette_score(Xs, km.labels_, sample_size=1000))
            yp = rng.permutation(y)
            res[name]["ari3_perm"].append(adjusted_rand_score(yp, km.labels_))

            # Behavior-SD 内部对照: 同一次抽样, y=realized
            if name == "behavior":
                y_r = df.iloc[sel]["realized"].values
                res["behavior_realized_ari3"].append(
                    adjusted_rand_score(y_r, km.labels_))

            # BC-vs-rest (BC N2 : Int N2/2 + None N2/2)
            X2, y2, _ = draw_many(
                sam, df, {"BC": N2, "Int": N2 // 2, "None": N2 // 2}, rng)
            if X2 is not None:
                Xs2 = StandardScaler().fit_transform(X2)
                a2, _ = cluster_ari(Xs2, np.where(y2 == "BC", 0, 1), 2, seed + b)
                res[name]["ari_bcvrest"].append(a2)

            # BC-vs-Int (BC N2 : Int N2)
            X3_, y3_, _ = draw_many(sam, df, {"BC": N2, "Int": N2}, rng)
            if X3_ is not None:
                Xs3 = StandardScaler().fit_transform(X3_)
                a3, _ = cluster_ari(Xs3, np.where(y3_ == "BC", 0, 1), 2, seed + b)
                res[name]["ari_bcvint"].append(a3)
        if (b + 1) % 50 == 0:
            print(f"replicate {b+1}/{B}, {time.time()-t0:.0f}s", flush=True)

    out = {"config": {"B": B, "N3": N3, "N2": N2, "seed": seed,
                      "candor_units": int(len(candor["unit"].unique())),
                      "behavior_units": int(len(behavior["unit"].unique()))}}
    for name in ["candor", "behavior"]:
        out[name] = {}
        for k in keys:
            arr = np.array(res[name][k])
            if len(arr):
                out[name][k] = {
                    "mean": float(arr.mean()),
                    "ci_low": float(np.percentile(arr, 2.5)),
                    "ci_high": float(np.percentile(arr, 97.5)),
                    "n": len(arr),
                }
    arr_r = np.array(res["behavior_realized_ari3"])
    out["behavior_realized_ari3"] = {
        "mean": float(arr_r.mean()),
        "ci_low": float(np.percentile(arr_r, 2.5)),
        "ci_high": float(np.percentile(arr_r, 97.5)),
    }

    a_c = np.array(res["candor"]["ari3"])
    a_b = np.array(res["behavior"]["ari3"])
    delta = a_c - a_b
    out["delta_ari3"] = {
        "mean": float(delta.mean()),
        "ci_low": float(np.percentile(delta, 2.5)),
        "ci_high": float(np.percentile(delta, 97.5)),
        "p_two_sided": float(2 * min(np.mean(delta <= 0), np.mean(delta >= 0))),
    }
    # 供直方图
    out["_reps"] = {
        "candor_ari3": [float(x) for x in a_c],
        "behavior_ari3": [float(x) for x in a_b],
        "delta_ari3": [float(x) for x in delta],
        "candor_ari3_perm": [float(x) for x in res["candor"]["ari3_perm"]],
        "behavior_ari3_perm": [float(x) for x in res["behavior"]["ari3_perm"]],
        "behavior_realized_ari3": [float(x) for x in arr_r],
    }
    with open(f"{out_dir}/ari_bootstrap_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "_reps"}, indent=2))
    return out


def save_deterministic_draws(candor, behavior, N3, seed, out_dir):
    """固定种子平衡抽样 (供混淆矩阵 / PCA 可视化), 保存 CSV."""
    rng = np.random.default_rng(seed)
    counts3 = {"BC": N3, "Int": N3, "None": N3}
    for name, df in [("candor", candor), ("behavior", behavior)]:
        sam = make_sampler(df)
        sel = draw_balanced(sam, counts3, rng)
        draw = df.iloc[sel].copy()
        draw.to_csv(f"{out_dir}/{name}_draw.csv", index=False)
        print(f"{name} draw: {len(draw)} events "
              f"{draw['cls'].value_counts().to_dict()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=500)
    ap.add_argument("--n3", type=int, default=1000)
    ap.add_argument("--n2", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default=DEFAULT_OUT)
    args = ap.parse_args()

    candor = load_dataset(f"{args.out_dir}/candor_events_w*.csv", "session")
    behavior = load_dataset(f"{args.out_dir}/behavior_events_w*.csv", "file_id")
    print(f"CANDOR: {len(candor)} events, {candor['unit'].nunique()} sessions",
          flush=True)
    print("CANDOR classes:", candor["cls"].value_counts().to_dict(), flush=True)
    print(f"Behavior-SD: {len(behavior)} events, "
          f"{behavior['unit'].nunique()} files", flush=True)
    print("Behavior classes:", behavior["cls"].value_counts().to_dict(),
          flush=True)
    print("Behavior realized:", behavior["realized"].value_counts().to_dict(),
          flush=True)

    run_bootstrap(candor, behavior, args.b, args.n3, args.n2,
                  args.seed, args.out_dir)
    save_deterministic_draws(candor, behavior, args.n3, args.seed, args.out_dir)


if __name__ == "__main__":
    main()
