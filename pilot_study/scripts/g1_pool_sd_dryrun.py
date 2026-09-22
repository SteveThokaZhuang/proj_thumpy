"""池间 SD 管线的**干跑** —— 把**单个已有评估集**按轮切成"假批", 走一遍正式脚本的算法。

## 为什么

正式实验 (2026-09-18_pool_sd.md) 要花 13 h GPU, 判读又挂在一条**比值**上。
按项目纪律「先让工具在已知答案上跑通」, 先用 e4x 走一遍:
把它的 1600 块 (200 key × 8 轮) 切成 k=200/400/800/1600 的组, 算同一个 ρ = SD/SE_null。

**预期**: 这些"假批"是**同一份抽取的划分**, 块在 key 内可交换 ⇒ ρ 应当 ≈ 1。
若这里 ρ 明显偏离 1, 说明**管线本身有 bug**(分层零模型算错、轮切错、口径不一致),
那么正式实验的 ρ 也不能信 —— 先修这里。

⚠️ 它**不是**池间 SD 的估计: 同一份抽取内部的划分**结构上看不见池子之外的变异**,
正是 §5.13 说 bootstrap 看不见的那一项。它只用来验管线。

用法 (srun 内, fd_analysis): python scripts/g1_pool_sd_dryrun.py --set e4x
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from g1_e4_analyze import paired, d_of  # noqa: E402
from g1_pool_null_probe import key_of, strat_boot_se  # noqa: E402

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="tag", default="e4x")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    D = paired(args.tag)
    assert D is not None, f"{args.tag} 配对不足"
    vec, cids, seeds = D["vec"], D["cids"], D["seeds"]
    n = len(cids)
    by_key = collections.defaultdict(list)
    for i, c in enumerate(cids):
        by_key[key_of(c)].append(i)
    K = len(by_key)
    # key 出现顺序决定"轮": 同一个 key 的第 j 次出现就是第 j 轮
    order = {k: 0 for k in by_key}
    rnd_of = np.empty(n, dtype=int)
    for i, c in enumerate(cids):
        k = key_of(c)
        rnd_of[i] = order[k]
        order[k] += 1
    n_rounds = int(rnd_of.max()) + 1
    per_round = collections.Counter(rnd_of.tolist())
    print(f"=== 干跑 {args.tag}: {n} 块 / {K} key / {n_rounds} 轮, "
          f"每轮 {min(per_round.values())}–{max(per_round.values())} 块 ===")
    dv = np.array([D["d"][s] for s in seeds])
    print(f"  ΔF1(全集) = {dv.mean():+.4f}")

    def df1(idx):
        return float(np.mean(list(d_of(vec, seeds, list(idx)).values())))

    print(f"\n{'k':>6}{'m/key':>7}{'组数':>6}{'df':>4}{'SD_组':>10}"
          f"{'SE_null分层':>13}{'ρ=SD/SE':>10}{'SE_null_SRS':>13}{'ρ_SRS':>9}")
    out = {}
    for k, m in ((200, 1), (400, 2), (800, 4), (1600, 8)):
        gsize = k // K
        if gsize < 1 or n_rounds % gsize:
            continue
        groups = [np.where((rnd_of >= r) & (rnd_of < r + gsize))[0]
                  for r in range(0, n_rounds, gsize)]
        if len(groups) < 2:
            continue
        vals = [df1(g) for g in groups]
        sd = float(np.std(vals, ddof=1))
        se_str = strat_boot_se(vec, cids, seeds, m, args.boot)
        # SRS 对照 (项目历史口径)
        rng = np.random.default_rng(0)
        db = np.empty(args.boot)
        for b in range(args.boot):
            db[b] = df1(rng.integers(0, n, k))
        se_srs = float(db.std(ddof=1))
        print(f"{k:>6}{m:>7}{len(groups):>6}{len(groups)-1:>4}{sd:>10.5f}"
              f"{se_str:>13.5f}{sd/se_str:>10.3f}{se_srs:>13.5f}{sd/se_srs:>9.3f}")
        out[k] = {"sd": sd, "se_strat": se_str, "se_srs": se_srs,
                  "rho": sd / se_str, "n_groups": len(groups), "vals": vals}

    rs = [v["rho"] for v in out.values()]
    print(f"\n  ρ 范围 [{min(rs):.3f}, {max(rs):.3f}] —— 这是**同一份抽取的划分**, "
          f"预期 ≈ 1。")
    if all(0.6 <= r <= 1.6 for r in rs):
        print("  ✅ 管线自检通过: 分层零模型与划分口径一致 (memory: 先让工具在已知答案上跑通)")
    else:
        print("  🔴 管线自检**未过** —— 分层零模型或轮切有 bug, 正式实验的 ρ 不能信, 先修这里")
    json.dump({"set": args.tag, "scan": {str(k): v for k, v in out.items()}},
              open(f"{ANNOT}/g1_pool_sd_dryrun_{args.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
