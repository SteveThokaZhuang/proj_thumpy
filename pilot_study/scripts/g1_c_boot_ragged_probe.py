"""**勘察**：`boot()` 的方形假设在抽稀集上会炸，能不能推广成"每会话各自重抽"？

## 问题（2026-09-23 实测）

`g1_c_analyze.boot()` 把账 reshape 成 `(S, C, 3)`，要求**每个会话块数相同**。
e4c/C 原集都是 100×48 ⇒ 没问题。但同仪器读数要把 C 抽稀到 e4c 的流行率，
抽稀是**按块**丢的 ⇒ 会话变得**参差**（实测 39–48，均值 45.7）⇒
`assert S * C == len(cids)` 抛异常，整个 `--set c` 挂掉。

## 这个探针要回答的唯一问题

把 `boot()` 推广成"**每个会话从自己的块里重抽自己的块数**"（ragged 版），
**在方形数据上是否复现原版**？

- 原版（`("sess","block")`）：抽 S 个会话槽 × 抽 C 个**列**槽（列对所有会话共用），
  取 `a[si[b,i], ci[b,j]]` ⇒ 因为 si、ci 独立均匀，等价于**从池化的 S×C 网格里 iid 抽 S·C 个块**。
- ragged 版：抽会话槽，**每个被抽中的会话各自**从自己的块里重抽 n_j 个 ⇒ **聚类**重抽。

⚠️ 两者**分布上可能不同**（池化 vs 聚类），这不是实现细节，是**估计量不同**。
所以必须先量出来差多少，再决定用哪个 —— 不许"反正都是 bootstrap"就换掉。

判据（写在这里）：
- **若 ragged 版在 e4c 上复现 0.0054 | 0.0052 | 0.0039 ⇒ 合并 0.0105**（及各分量 ±10%），
  说明两者在方形数据上等价 ⇒ 可以安全推广。
- **若不复现**，则如实报出两套数，**并优先用"截断到每会话同样块数"那条路**
  （它用的是原封不动的预登记估计量，代价是丢约 15% 的块）。

用法（纯读盘）：
  python scripts/g1_c_boot_ragged_probe.py --set e4c
"""
import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1_c_analyze as G  # noqa: E402


def boot_ragged(data, cids, B=4000, seed=0, lv=("sess", "block", "seed")):
    """ragged 版三级 bootstrap：每个被抽中的会话从**自己的块**里重抽 n_j 个。

    与 `G.boot` 的唯一差别：块重抽是**按会话**做的（聚类），而原版是共用列槽（池化）。
    会话块数不等时原版无定义，本版有定义。

    实现：会话 j 被抽中 m_j(b) 次 ⇒ 它对第 b 次重抽的贡献 = m_j(b) 个**独立**的
    重抽和，每个是 n_j 次有放回抽样 ⇒ 等价于从会话 j 的块里抽 m_j(b)·n_j 次。
    用 multinomial 一次算出一个"重抽和"的分布，再按 m_j(b) 掩码求和。
    """
    bysess = {}
    for c in cids:
        bysess.setdefault(c.split("_")[0], []).append(c)
    sess = sorted(bysess)
    S = len(sess)
    keys = [s for s in G.SEEDS
            if ("own10", s) in data and ("mixnorm", s) in data]
    rng = np.random.default_rng(seed)

    if "sess" in lv:
        si = rng.integers(0, S, size=(B, S))
        flat = (si + np.arange(B)[:, None] * S).ravel()
        m = np.bincount(flat, minlength=B * S).reshape(B, S).astype(np.int64)
    else:
        m = np.ones((B, S), dtype=np.int64)

    # 每会话的 (n_j, 3) 账，按 key 存
    per_key_tot = {k: np.zeros((B, 3)) for k in data}
    for j, s in enumerate(sess):
        cl = bysess[s]
        n_j = len(cl)
        vals = {k: np.array([[data[k][c]["tp"], data[k][c]["fp"],
                              data[k][c]["n_gt"]] for c in cl], float)
                for k in data}
        if "block" in lv:
            Kj = int(m[:, j].max())
            if Kj == 0:
                continue
            cnt = rng.multinomial(n_j, np.full(n_j, 1.0 / n_j),
                                  size=(B, Kj))          # (B,Kj,n_j)
            mask = (np.arange(Kj)[None, :] < m[:, j][:, None]).astype(float)
            for k in data:
                U = cnt @ vals[k]                         # (B,Kj,3) BLAS
                per_key_tot[k] += (U * mask[..., None]).sum(1)
        else:
            for k in data:
                per_key_tot[k] += m[:, j][:, None] * vals[k].sum(0)[None, :]

    f1 = {k: G._f1_from_counts(t[:, 0], t[:, 1], t[:, 2])
          for k, t in per_key_tot.items()}
    ds = np.stack([f1[("own10", s)] - f1[("mixnorm", s)] for s in keys])
    gi = (rng.integers(0, len(keys), size=(B, len(keys)))
          if "seed" in lv else None)
    d = (ds.mean(axis=0) if gi is None
         else ds[gi, np.arange(B)[:, None]].mean(axis=1))
    return float(d.mean()), float(d.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="e4c")
    ap.add_argument("--boot", type=int, default=4000)
    args = ap.parse_args()

    data = G.load_set(args.set)
    G.assert_complete(data, args.set)
    d, f1, cids, seeds_used = G.delta_by_seed(data)
    bysess = {}
    for c in cids:
        bysess.setdefault(c.split("_")[0], []).append(c)
    sizes = sorted(len(v) for v in bysess.values())
    print("集=%s  块=%d  会话=%d  每会话块数 %d..%d %s"
          % (args.set, len(cids), len(sizes), sizes[0], sizes[-1],
             "（方形）" if sizes[0] == sizes[-1] else "（参差！）"))
    print("\n%-12s %10s %10s" % ("层级", "原版 SE", "ragged SE"))
    for name, lv in (("sess", ("sess",)), ("block", ("block",)),
                     ("seed", ("seed",)),
                     ("all", ("sess", "block", "seed"))):
        try:
            _, s_old, _, _ = G.boot(data, cids, B=args.boot, lv=lv)
            o, ov = "%.5f" % s_old, s_old
        except AssertionError:
            o, ov = "无定义(参差)", None
        _, s_new = boot_ragged(data, cids, B=args.boot, lv=lv)
        rel = "" if ov is None else "  %+.1f%%" % (100 * (s_new / ov - 1))
        print("%-12s %10s %10.5f%s" % (name, o, s_new, rel))
    m_old, sd_old, _, _ = G.boot(data, cids, B=args.boot)
    m_new, sd_new = boot_ragged(data, cids, B=args.boot)
    print("\n三级 原版:   %.5f ± %.5f" % (m_old, sd_old))
    print("三级 ragged: %.5f ± %.5f   (%+.1f%%)"
          % (m_new, sd_new, 100 * (sd_new / sd_old - 1)))
    print("合并 SE 的 MC 误差 ≈ %.1f%% (B=%d) —— 小于这个数的差别不用解释"
          % (100 / (2 * args.boot) ** 0.5, args.boot))
    return 0


if __name__ == "__main__":
    sys.exit(main())
