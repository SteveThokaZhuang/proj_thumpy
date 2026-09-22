"""Q1 收尾: 「各种子平均 margin 的 SD, mixnorm 是 own10 的 2.9 倍」这句话本身站得住吗?

P1 的 Q5 报了 SD 0.525 vs 0.183 (2.9x)。但那是**从 7 个点估的 SD** ——
正是 [[sd-from-few-points-manufactures-anomalies]] 警告的那类操作。
在被 F=18.64 那次教训教过之后, 这里必须先对**这个 SD 比**做同样的检验:

  - F(6,6) 的方差比 + 精确置换 (把 14 个值随机分成 7/7)
  - 留一: 去掉一个种子后 F 变成多少

注意 margin 是**模型 logits 的性质, 与评估集标签无关** —— 所以它不吃
流行率那一套伪影。若这里也不显著, 那 §5 的 Q5 数字就只是小样本噪声,
和 §3 推翻掉的那个 F=18.64 是同一类错误。

用法: /share/home/zhuangruicen/miniconda3/envs/fd_analysis/bin/python \
        scripts/ari_g1_margin_sd_test.py
"""
import json
import os

import numpy as np

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
SEEDS = [42, 1234, 7, 2024, 3407, 31337, 55555]
ARMS = ["own10", "mixnorm"]


def mean_margins(arm):
    out = {}
    for s in SEEDS:
        p = f"{ANNOT}/g1_margin_{arm}_s{s}.json"
        if not os.path.exists(p):
            continue
        m = json.load(open(p))
        out[s] = float(np.mean([x["margin_max"] for x in m.values()]))
    return out


def var_ratio_perm(o, m, n_iter=200000, seed=0):
    """把 o+m 的 14 个值随机分成 7/7, 看方差比多大。"""
    rng = np.random.default_rng(seed)
    pool = np.concatenate([o, m])
    n = len(o)
    obs = m.var(ddof=1) / o.var(ddof=1)
    cnt = 0
    for _ in range(n_iter):
        idx = rng.permutation(len(pool))
        a, b = pool[idx[:n]], pool[idx[n:]]
        f = b.var(ddof=1) / max(a.var(ddof=1), 1e-12)
        if f >= obs:
            cnt += 1
    return obs, (cnt + 1) / (n_iter + 1)


def main():
    o, m = mean_margins("own10"), mean_margins("mixnorm")
    seeds = [s for s in SEEDS if s in o and s in m]
    ov = np.array([o[s] for s in seeds])
    mv = np.array([m[s] for s in seeds])

    print("=" * 80)
    print("  各种子【平均 margin】—— 逐种子值")
    print("=" * 80)
    print(f"  {'seed':>7}{'own10':>11}{'mixnorm':>11}{'差':>11}")
    for s, a, b in zip(seeds, ov, mv):
        print(f"  {s:>7}{a:>11.3f}{b:>11.3f}{b-a:>11.3f}")
    print(f"  {'SD':>7}{ov.std(ddof=1):>11.3f}{mv.std(ddof=1):>11.3f}")
    print(f"  {'均值':>7}{ov.mean():>11.3f}{mv.mean():>11.3f}")

    print("\n" + "=" * 80)
    print("  SD 比 / 方差比 —— 显著性检验")
    print("=" * 80)
    f_obs, p_perm = var_ratio_perm(ov, mv)
    print(f"  观测方差比 F(6,6) = {f_obs:.3f}   (SD 比 = {mv.std(ddof=1)/ov.std(ddof=1):.2f}x)")
    print(f"  精确置换 p (200k 次, 14 值随机分 7/7) = {p_perm:.4f}")

    # 参数检验的临界值作参照
    from scipy import stats as st
    crit = st.f.ppf(0.975, 6, 6)
    print(f"  F 分布 95% 临界值 F(6,6) = {crit:.3f}  -> "
          f"{'超过' if f_obs > crit else '未超过'}")

    print("\n  留一 (去掉一个种子):")
    for i, s in enumerate(seeds):
        oo = np.delete(ov, i)
        mm = np.delete(mv, i)
        f = mm.var(ddof=1) / oo.var(ddof=1)
        _, p = var_ratio_perm(oo, mm, n_iter=50000)
        flag = "  <-- 掉出 0.05" if p > 0.05 else ""
        print(f"    去掉 {s:<6} F = {f:>7.3f}   p = {p:.4f}{flag}")

    print("\n" + "=" * 80)
    print("  结论")
    print("=" * 80)
    if p_perm > 0.05:
        print("  ❌ 这个 SD 比**不显著**。§5 Q5 的「2.9 倍」是小样本产物,")
        print("     和 §3 推翻掉的 F=18.64 属于同一类错误 —— 只不过那次的伪影")
        print("     来自流行率, 这次的来自 n=7。")
        print("     -> 「mixnorm 的种子把边界画在更分散的位置」这句话,")
        print("        以目前的数据**没有证据支持**。")
    else:
        print("  ✅ 方差比显著 —— 种子间阈值离散确实是 mixnorm 更严重。")
        print("     这是一个不依赖评估集流行率的结论 (margin 与标签无关)。")


if __name__ == "__main__":
    main()
