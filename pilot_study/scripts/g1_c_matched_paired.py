"""同仪器读数**是不是真的把 ΔF1 拉下来了**？——配对检验。

## 为什么必须做这一步

C 主读数 +0.0426, 同仪器读数 +0.0254, 而 e4c 是 +0.0254。三个数摆在一起
"显然"讲了一个故事: **C 比 e4c 高出来的那截, 全部由流行率解释**。

但 memory `prediction-hit-is-not-mechanism` 记的就是这个陷阱的上一犯:
「预测 ΔF1 从 0.0340 掉到 0.0211, 实测差 0.0001『命中』, 但**收缩本身**
t=−0.55/p=0.60 没被确立」。**两个数长得像 ≠ 收缩成立。**

而且不配对地比是有毒的: 0.0426±0.0138 与 0.0254±0.0086 相减, 差 −0.0172,
naive SE = sqrt(0.0138²+0.0086²) = 0.0163 ⇒ t≈−1.05。**不显著。**
可这两个读数共用同一批会话与同一批种子, 不配对是在把共同的那部分噪声
重复计两遍 (memory `paired-design-floors-at-the-interaction`)。

## 配对怎么做

在**同一批** (会话 × 块 × 种子) 重抽上同时算 full 与 thin 的 ΔF1, 取逐次之差:

- 会话 j 有 48 个块, 其中 T_j 个属于抽稀保留集。
- 对每个重抽, 从**全 48 个**里抽 (multinomial) —— 抽中的块**总是**进 full 统计;
  **当且仅当它属于 T_j** 时才进 thin 统计。
- 于是两者共享同一份"抽中了哪些块"的随机性, 差里只剩"少掉的那些块"的影响。

这正是要估的量: **把流行率降到 e4c 的水平, ΔF1 会掉多少。**

## 判据 (跑之前写死)

- `ΔF1_thin − ΔF1_full` 的配对均值若 **t ≤ −2** ⇒ 收缩成立, 「流行率解释 C 与 e4c 之差」
  这句话才可以写。
- 若 **|t| < 2** ⇒ 只能写「两个读数数值上接近」, **不许**写「流行率解释了差异」。
  ⚠️ 特别注意: 就算 thin ≈ e4c 且 full > e4c, 也**不构成**机制成立的证据 ——
  那是三个点估计的巧合, 不是一次检验。

⚠️ 这条配对 SE **不含抽稀本身的随机** (换了 draw 就换一批 T_j)。抽稀间 SD 由
`g1_c_analyze.py` 单独报, 引用时要合并考虑。

用法 (纯读盘, 0 GPU):
  python scripts/g1_c_matched_paired.py --set c --target 0.039375 --boot 4000
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1_c_analyze as G  # noqa: E402


def paired_delta(data, cids_full, cids_thin, B=4000, seed=0):
    """返回 (配对差均值, 配对差 SE, 逐次差的数组)。

    差 = ΔF1(thin) − ΔF1(full), 逐次重抽各算一次, 同一份会话/块/种子重抽。
    """
    thin_set = set(cids_thin)
    bysess = {}
    for c in cids_full:
        bysess.setdefault(c.split("_")[0], []).append(c)
    sess = sorted(bysess)
    S = len(sess)
    keys = [s for s in G.SEEDS
            if ("own10", s) in data and ("mixnorm", s) in data]
    rng = np.random.default_rng(seed)

    si = rng.integers(0, S, size=(B, S))
    flat = (si + np.arange(B)[:, None] * S).ravel()
    m = np.bincount(flat, minlength=B * S).reshape(B, S).astype(np.int64)

    totF = {k: np.zeros((B, 3)) for k in data}
    totT = {k: np.zeros((B, 3)) for k in data}
    for j, s in enumerate(sess):
        cl = sorted(bysess[s])
        n_j = len(cl)
        inthin = np.array([c in thin_set for c in cl])
        if not inthin.any():
            print(f"    ⚠️ 会话 {s} 在抽稀集里一个块都不剩 —— 它的 thin 贡献恒为 0")
        Kj = int(m[:, j].max())
        if Kj == 0:
            continue
        cnt = rng.multinomial(n_j, np.full(n_j, 1.0 / n_j), size=(B, Kj))
        keep = np.arange(Kj)[None, :] < m[:, j][:, None]
        for k in data:
            v = np.array([[data[k][c]["tp"], data[k][c]["fp"],
                           data[k][c]["n_gt"]] for c in cl], float)
            totF[k] += ((cnt @ v) * keep[..., None]).sum(1)
            # 同一份 cnt, 只把落在抽稀集里的那些块计入
            vT = v[inthin]
            totT[k] += ((cnt[:, :, inthin] @ vT) * keep[..., None]).sum(1)

    # ⚠️ full 与 thin **必须共用同一个 gi** —— 否则种子那一路的噪声没被配对掉,
    #    差里会混进两个独立种子重抽的方差, 把配对的意义抹掉一大半。
    rng_seed = np.random.default_rng(seed)
    gi = rng_seed.integers(0, len(keys), size=(B, len(keys)))

    def dF1(tot):
        f1 = {k: G._f1_from_counts(t[:, 0], t[:, 1], t[:, 2])
              for k, t in tot.items()}
        ds = np.stack([f1[("own10", s)] - f1[("mixnorm", s)] for s in keys])
        return ds[gi, np.arange(B)[:, None]].mean(axis=1)

    dfull = dF1(totF)
    dthin = dF1(totT)
    d = dthin - dfull
    return float(d.mean()), float(d.std()), d, float(dfull.mean()), float(dthin.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="c")
    ap.add_argument("--target", type=float, default=0.039375)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--draws", type=int, default=5,
                    help="做几个抽稀次 (每个各跑一次配对 bootstrap, 再合并)")
    args = ap.parse_args()

    data = G.load_set(args.set)
    G.assert_complete(data, args.set)
    _, _, cids, _ = G.delta_by_seed(data)
    gt = sum(data[("own10", "42")][c]["n_gt"] for c in cids)
    print("full  %d 块 / %d 会话 / 事件 %d (%.4f%%)"
          % (len(cids), len({c.split('_')[0] for c in cids}), gt,
             100 * gt / len(cids)))

    # ⚠️ 单次抽稀的配对 SE **不含抽稀本身的随机** (换一个 draw 就换一批 T_j)。
    #    只报单次 = 只覆盖一个随机源 —— memory
    #    `permutation-test-covers-one-variance-source` 里那条反复犯的错。
    #    所以: 对 R 个抽稀各做一次配对 bootstrap, 再按全方差律合并
    #        总 SE² = 抽稀内 SE² 的均值  +  各次均值之间的方差
    R = args.draws
    means, ses, mfs, mts = [], [], [], []
    for r in range(R):
        keep, got, n = G.thin_to_prevalence(data, cids, args.target, seed=r)
        me, sd, _, mf, mt = paired_delta(data, cids, keep, B=args.boot, seed=r)
        means.append(me); ses.append(sd); mfs.append(mf); mts.append(mt)
        print("  draw %d: thin %d 块 / 事件 %d (流行率 %.4f%%, 目标 %.4f%%)  "
              "ΔF1_full %+.4f  ΔF1_thin %+.4f  配对差 %+.4f ± %.4f"
              % (r, n, got * n, 100 * got, 100 * args.target, mf, mt, me, sd))
    means = np.array(means); ses = np.array(ses)
    within = float(np.mean(ses ** 2))
    between = float(np.var(means, ddof=1)) if R > 1 else 0.0
    tot = (within + between) ** 0.5
    grand = float(means.mean())
    print("\n配对 (同一批会话/块/种子重抽, %d 个抽稀次合并):" % R)
    print("  ΔF1_full 均值 = %+.4f   ΔF1_thin 均值 = %+.4f"
          % (np.mean(mfs), np.mean(mts)))
    print("  配对差 (thin − full) = %+.4f" % grand)
    print("    ├ 抽稀内 SE      = %.4f  (paired bootstrap 均值)"
          % within ** 0.5)
    print("    ├ 抽稀间 SD      = %.4f  (R=%d 个 draw 的均值散布)" % (between ** 0.5, R))
    print("    └ 合并 SE        = %.4f   ⇒  t = %+.2f"
          % (tot, grand / tot if tot else float("nan")))
    print("  余量/MC: t=%.2f 对判据线 2.00, 差 %.2f; 而 SE 本身的 MC ≈ %.1f%% (B=%d)"
          % (abs(grand / tot), abs(grand / tot) - 2.0,
             100 / (2 * args.boot) ** 0.5, args.boot))
    print("\n  判据: t ≤ −2 ⇒ 收缩成立, 才可写「流行率解释了 C 与 e4c 之差」;")
    print("        |t| < 2 ⇒ 只能写「两个读数数值上接近」。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
