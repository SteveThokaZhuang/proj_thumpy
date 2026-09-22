"""Q3 分析: 把跨种子方差拆成「初始化」与「shuffle」两份 —— 纯读盘, 0 GPU。

## 设计

两组 run, 都在两臂各 7 个:

  A) 既有「种子」组: `g1_{arm}[_s{seed}]_sft`      —— seed 变, data_seed 随 seed 走
  B) 新「data_seed」组: `g1_{arm}_ds{ds}_sft`      —— **seed 固定 42**, data_seed 变

  A 的离散 = Var(初始化) + Var(shuffle)   (seed 同时决定 LoRA 初值与数据顺序)
  B 的离散 =               Var(shuffle)   (初值逐位相同, 差异只能来自数据顺序)

  ⇒ Var(初始化) = Var(A) − Var(B)

## 一个必须带上的诚实性处理

`SD` 是从 **7 个点**估的, 本身有可观的抽样误差。本项目已经栽过一次
「用 2–3 个点估 SD, 把噪声包装成 7.6 SD 的异常」(见 memory:
sd-from-few-points-manufactures-anomalies), 所以这里:

  - 报 `SD` 的同时报它的 **χ² 置信区间** (正态假设下 (n−1)s²/σ² ~ χ²(n−1))
  - 差值 `Var(init) = Var(A) − Var(B)` 若为负, **不裁剪成一个好看的小正数** ——
    那样会把「估不出来」伪装成「占比 3%」。如实报「两项在现有精度下不可区分」。
  - 两条区间若大幅重叠, 结论只能是「分不开」, 不能挑一个点估计讲故事。

## 自检

`--group seed` 走 A 组, **必须复现 §7.2 的 SD = 0.0309 (ΔF1 = +0.0340)**。
对不上说明读错了文件或算错了, B 组的结果也不能采信。

用法:
  python scripts/g1_q3_analyze.py --group seed   # 自检
  python scripts/g1_q3_analyze.py --group ds     # Q3 主结果
  python scripts/g1_q3_analyze.py --group both   # 拆解
"""
import argparse
import json
import os

import numpy as np

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
SEEDS = [42, 1234, 7, 2024, 3407, 31337, 55555]      # A 组: seed 变化
DS_LIST = [42, 1234, 7, 2024, 3407, 31337, 55555]    # B 组: seed 固定 42, data_seed 变化
ARMS = ["own10", "mixnorm"]
BASELINE_SD_SEED = 0.0309      # §7.2, 用于自检
BASELINE_DELTA = 0.0340


def sfx(seed):
    return "" if seed == 42 else f"_s{seed}"


def load_seed_group(arm, seed):        # A 组
    p = f"{ANNOT}/g1_eval_{arm}{sfx(seed)}_filt.json"
    return next(iter(json.load(open(p)).values())) if os.path.exists(p) else None


def load_ds_group(arm, ds):            # B 组
    p = f"{ANNOT}/g1_eval_{arm}_ds{ds}_filt.json"
    return next(iter(json.load(open(p)).values())) if os.path.exists(p) else None


def f1_of(pc, idx):
    tp = fp = gt = 0
    for i in idx:
        e = pc[i]
        tp += e["tp"]; fp += e["fp"]; gt += e["n_gt"]
    p = tp / max(1, tp + fp)
    r = tp / max(1, gt)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def deltas(loader, keys):
    """返回 (可用 keys, 逐 key 的 ΔF1, 公共块下标集合)。"""
    P = {}
    for k in keys:
        a, b = loader(ARMS[0], k), loader(ARMS[1], k)
        if a is not None and b is not None:
            P[k] = (a, b)
    if not P:
        return [], np.array([]), []
    cids = sorted(set.intersection(*(set(v[0]["per_chunk"]) for v in P.values()),
                                   *(set(v[1]["per_chunk"]) for v in P.values())))
    ks, ds = [], []
    for k, (a, b) in sorted(P.items()):
        ks.append(k)
        ds.append(f1_of([a["per_chunk"][c] for c in cids], range(len(cids)))
                  - f1_of([b["per_chunk"][c] for c in cids], range(len(cids))))
    return ks, np.array(ds), cids


def sd_ci(s, n, alpha=0.05):
    """正态假设下 SD 的 χ² 置信区间: (n−1)s²/σ² ~ χ²(n−1)。"""
    try:
        from scipy.stats import chi2
    except ImportError:
        return None
    lo = s * np.sqrt((n - 1) / chi2.ppf(1 - alpha / 2, n - 1))
    hi = s * np.sqrt((n - 1) / chi2.ppf(alpha / 2, n - 1))
    return float(lo), float(hi)


def report(name, ks, d):
    n = len(d)
    m, s = float(d.mean()), float(d.std(ddof=1))
    ci = sd_ci(s, n)
    print(f"\n--- {name} (n={n}) ---")
    print(f"  {'key':>8}{'ΔF1':>10}")
    for k, x in zip(ks, d):
        print(f"  {k:>8}{x:>+10.4f}")
    print(f"  mean ΔF1 = {m:+.4f}")
    print(f"  SD       = {s:.4f}" + (f"   95% CI [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""))
    print(f"  Var      = {s**2:.6f}")
    return m, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="both", choices=["seed", "ds", "both"])
    args = ap.parse_args()

    res = {}
    if args.group in ("seed", "both"):
        ks, d, cids = deltas(load_seed_group, SEEDS)
        res["seed"] = (ks, d) + report("A 组: seed 变化 (seed 同时决定初值与顺序)",
                                       ks, d)
        e = load_seed_group(ARMS[0], SEEDS[0])
        n_gt = sum(e["per_chunk"][c]["n_gt"] for c in cids)
        print(f"  公共块 {len(cids)}, 真值事件 {n_gt}")

    if args.group in ("ds", "both"):
        ks, d, cids = deltas(load_ds_group, DS_LIST)
        if len(d) == 0:
            print("\n--- B 组: data_seed 变化 ---\n  ⚠️ 还没有产物 "
                  "(g1_eval_<arm>_ds<ds>_filt.json)。先跑 scripts/g1_q3_eval.sh q3")
        else:
            res["ds"] = (ks, d) + report("B 组: 固定 seed=42, data_seed 变化", ks, d)

    # ---------- 自检 ----------
    if args.group in ("seed", "both") and "seed" in res:
        m, s = res["seed"][2], res["seed"][3]
        ok = abs(s - BASELINE_SD_SEED) < 0.002 and abs(m - BASELINE_DELTA) < 0.003
        print("\n" + "=" * 74)
        print(f"  自检 @ A 组: SD 得到 {s:.4f} 期望 {BASELINE_SD_SEED} | "
              f"ΔF1 得到 {m:+.4f} 期望 {BASELINE_DELTA:+.4f}  "
              f"{'✅' if ok else '❌ 先修代码, 别看 B 组'}")
        if not ok:
            return

    # ---------- 拆解 ----------
    if "seed" in res and "ds" in res:
        va, vb = res["seed"][3] ** 2, res["ds"][3] ** 2
        print("\n" + "=" * 74)
        print("  方差拆解:  Var(seed) = Var(init) + Var(shuffle)")
        print(f"    Var(A: seed)      = {va:.6f}   (SD {res['seed'][3]:.4f})")
        print(f"    Var(B: data_seed) = {vb:.6f}   (SD {res['ds'][3]:.4f})")
        diff = va - vb
        print(f"    ⇒ Var(init)      = {diff:.6f}")
        if diff > 0:
            print(f"    占比: shuffle {vb/va:.1%} / init {diff/va:.1%}")
        else:
            print("    ⚠️ 为**负** —— 两项在现有精度下**分不开**, 别裁剪成小正数。")
            print("       7 个点估的 SD 本身就有 χ² 区间, 差值为负说明 B 组的")
            print("       离散已经和 A 组同量级 ⇒ 「初值运气」不是主要来源, 或者")
            print("       两个实验的精度都不足以把它们分开。如实这么报。")
        ci_a, ci_b = sd_ci(res["seed"][3], len(res["seed"][1])), sd_ci(res["ds"][3], len(res["ds"][1]))
        if ci_a and ci_b:
            print(f"\n    A 组 SD 95% CI [{ci_a[0]:.4f}, {ci_a[1]:.4f}]")
            print(f"    B 组 SD 95% CI [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")
            if ci_a[0] < ci_b[1] and ci_b[0] < ci_a[1]:
                print("    两条区间重叠 ⇒ 单看 SD 不能断言两者有差。")


if __name__ == "__main__":
    main()
