"""新加的 400 块是"同一批块的又一次抽样", 还是"另一批性质不同的块"?

## 为什么要问这个

k=4b 的结果是: 加块到 1600 后 `SE_chunk` **几乎没动** (0.009725 -> 0.009772),
而 §5.8d 按集内指数 −0.488 外推预测它该降到 0.0084。
同时 e4x 池在**每个 k 上**的 `SE_chunk` 都比 e4 池高 16~23% (见两份 frontier 输出)。

⇒ 最省事的解释是「**新加的 400 块逐块方差更高**」, 那样整个池子的方差水平被抬高,
恰好抵消掉 k 从 1200→1600 的 `1/√k` 收益 (×0.866)。

本脚本直接检验这个解释: 把 e4b 的实测值, 放到「**从 e4 里随机抽 400 块**」的
分布里去看位置。

- 若 e4b 落在分布内 ⇒ 它就是**又一次抽样**, 加块不降 SE 是**抽样的性质**,
  不是这一批块特别倒霉。
- 若 e4b 落在很远的尾部 ⇒ 这一批 400 块**与旧池不同分布**, 那
  「同一自然流行率总体」这个前提本身要重新审。

## ⚠️ 这个参照分布**低估**了新抽样应有的散布

从 1200 里抽 400 (**无放回**) 时, 池子是**固定**的; 而 e4b 是从总体里**新抽**的,
不受旧池约束。所以无放回子采样天然比独立重抽更"稳"。
⇒ 本检验是**保守**的: e4b 若已落在子采样分布的尾部, 用真分布看只会更极端。

用法 (纯读盘, 0 GPU):
  python scripts/g1_e4b_pool_diag.py [--draws 60] [--boot 500]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_e4_analyze import boot_se, f1_of, paired  # noqa: E402

K_SUB = 400          # 与新增块数一致, 才可比


def delta_of(vec, cids, seeds, idx):
    """配对 ΔF1 = 多种子均值 (与全篇同口径, 走 f1_of)。"""
    return float(np.mean([f1_of(vec[("own10", s)], idx)
                          - f1_of(vec[("mixnorm", s)], idx) for s in seeds]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=60)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--match-prevalence", action="store_true",
                    help="抽样的 400 块与 e4b 的**正例块数**保持一致。"
                         "不加这个, 「新块更噪」与「新块正例更多」分不开 —— "
                         "而后者本项目已经栽过一次 (流行率能翻转方差结论)。")
    args = ap.parse_args()

    D = paired("e4")
    if D is None:
        print("🔴 e4 产物不齐"); return 1
    vec, cids, seeds = D["vec"], D["cids"], D["seeds"]
    n = len(cids)
    d_e4 = delta_of(vec, cids, seeds, range(n))

    Db = paired("e4b")
    if Db is None:
        print("🔴 e4b 产物不齐"); return 1
    d_e4b = delta_of(Db["vec"], Db["cids"], Db["seeds"], range(len(Db["cids"])))
    se_e4b = boot_se(Db["vec"], Db["cids"], Db["seeds"], 2000)

    print("=" * 84)
    print(f"  新 400 块 vs 「从旧 1200 里抽 400」的分布  ({args.draws} 次, "
          f"每次 bootstrap {args.boot})")
    print("=" * 84)
    print(f"  e4  (1200 块)  ΔF1 = {d_e4:+.4f}")

    rng = np.random.default_rng(args.seed)
    pos = {c: i for i, c in enumerate(cids)}

    # 自检: 用**全集**走一遍「建子 vec 再 boot_se」的机器, 必须与直接 boot_se
    # 逐位相同。没有这条, 子集机器写错了(比如子 vec 没对齐)会被静默吞掉 ——
    # 本脚本第一版的 SE 列就是这么错的。
    _ref = boot_se(vec, cids, seeds, 200)
    _got = boot_se({k: [vec[k][pos[c]] for c in cids] for k in vec},
                   list(cids), seeds, 200)
    if abs(_ref - _got) > 1e-12:
        print(f"🔴 子集机器对不上: 全集 {_ref:.6f} vs 重建 {_got:.6f} —— 停")
        return 1
    print(f"  ✅ 自检: 子集机器在全集上逐位复现 boot_se ({_ref:.6f})")

    # 正例块 (以 own10/首种子的 n_gt>0 为准; 各臂真值相同, 取哪个臂都一样)
    e0 = vec[("own10", seeds[0])]
    is_pos = np.array([e0[i]["n_gt"] > 0 for i in range(n)])
    n_pos_e4b = sum(1 for c in Db["cids"]
                    if Db["P"][("own10", Db["seeds"][0])]["per_chunk"][c]["n_gt"] > 0)
    pos_idx = np.flatnonzero(is_pos)
    neg_idx = np.flatnonzero(~is_pos)
    if args.match_prevalence:
        print(f"  🔒 流行率匹配: e4b 有 {n_pos_e4b} 个正例块 / {len(Db['cids'])} "
              f"= {n_pos_e4b/len(Db['cids']):.2%};")
        print(f"     e4 池有 {len(pos_idx)} 正 / {len(neg_idx)} 负 ⇒ 每次抽 "
              f"{n_pos_e4b} 正 + {K_SUB-n_pos_e4b} 负")
        print(f"     (不匹配的话 e4 池抽 400 块期望只有 "
              f"{K_SUB*len(pos_idx)/n:.1f} 个正例块, 比 e4b 少)")

    ds, ses = [], []
    for i in range(args.draws):
        if args.match_prevalence:
            ii = np.concatenate([rng.choice(pos_idx, n_pos_e4b, replace=False),
                                 rng.choice(neg_idx, K_SUB - n_pos_e4b,
                                            replace=False)])
        else:
            ii = rng.choice(n, K_SUB, replace=False)
        ds.append(delta_of(vec, cids, seeds, ii))
        # 🔴 `boot_se` 重抽的是 **len(cids)** 个块 —— 直接传全集 cids 会把
        #    「400 块的 SE」静默算成「1200 块的 SE」(本脚本第一版就踩了,
        #    那一列 60 次抽样的 SD 只有 0.0003, 因为它每次都在算同一个数)。
        #    要量子集的 SE, 就必须**把子集本身**作为 cids 传进去。
        sub = [cids[j] for j in ii]
        vsub = {k: [vec[k][pos[c]] for c in sub] for k in vec}
        ses.append(boot_se(vsub, sub, seeds, args.boot, rng_seed=i))
    ds, ses = np.array(ds), np.array(ses)

    # 🔴 **参照分布必须做有限总体校正 (FPC)** —— 这是本脚本最容易骗人的地方。
    #    「从固定的 1200 里无放回抽 400」的散布, 比「从总体里独立抽 400」**小**:
    #        Var(无放回, 从 N 抽 n) = Var(独立抽 n) × (N−n)/(N−1)
    #    ⇒ SD_独立 = SD_无放回 × sqrt((N−1)/(N−n))
    #    不校正就会把"e4b 超出这 60 次抽样的范围"读成"e4b 不是同一分布",
    #    而实际上只是参照分布被人为收窄了 (memory: 阈值必须先对零模型)。
    fpc = np.sqrt((n - 1) / (n - K_SUB))
    print(f"\n  ── 有限总体校正 FPC = sqrt(({n}−1)/({n}−{K_SUB})) = {fpc:.3f} ──")
    print(f"     (无放回子抽样的散布比独立重抽小, 乘这个系数才可比)")

    res = {}
    for nm, arr, obs in (("ΔF1", ds, d_e4b), ("SE_chunk", ses, se_e4b)):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        sd_sub = float(arr.std(ddof=1))
        sd_fresh = sd_sub * fpc                 # 独立重抽的 SD 估计
        z = (obs - arr.mean()) / sd_fresh if sd_fresh else float("nan")
        pct = float((arr < obs).mean() * 100)
        res[nm] = (z, sd_fresh)
        print(f"\n  ── {nm}: 抽 400 块的分布 ──")
        print(f"     无放回子抽样的 SD = {sd_sub:.4f}  ⇒ 独立重抽的 SD ≈ "
              f"{sd_fresh:.4f} (× FPC)")
        print(f"     均值 {arr.mean():+.4f}   95% 区间 [{lo:+.4f}, {hi:+.4f}]   "
              f"范围 [{arr.min():+.4f}, {arr.max():+.4f}]")
        print(f"     **e4b 实测 {obs:+.4f}** —— 第 {pct:.1f} 百分位, "
              f"z = {z:+.2f} (按独立重抽的 SD 算)")
        out = obs < arr.min() or obs > arr.max()
        print(f"     {'(超出 60 次抽样的范围)' if out else '(在 60 次抽样的范围内)'}"
              f"  ⚠️ 但**范围**本身也会随抽样次数变宽, 判据用 z 不用范围")

    # ---------- 结论判读 ----------
    print(f"\n  ── 判读 ──")
    zd, zs = res["ΔF1"][0], res["SE_chunk"][0]
    print(f"    ΔF1      偏离 {zd:+.2f} SD")
    print(f"    SE_chunk 偏离 {zs:+.2f} SD")
    print(f"    {'⚠️ 两者都在 2 SD 量级 —— **提示**新块不同质, 但**不足以确立**。' if max(abs(zd), abs(zs)) < 3 else '两者都远超 3 SD —— 新块不同质有相当证据。'}")
    print(f"    🔒 **但样本量是 1**: 我们只抓了**一批**新的 400 块。"
          f"要断言「新块系统性更噪」,")
    print(f"       需要**再抓几批**各自独立算 SE_chunk —— 一批落在 2 SD 外")
    print(f"       可能只是运气 (n=1, 本来就该有 ~5% 的批落在这个位置)。")
    print(f"    ⇒ 目前能确立的只有一件事: **加块没有按 1/√k 降 SE**, "
          f"而这不是 MC 噪声 (SE 的 MC 只有 1.4%)。")
    print(f"\n  ⚠️ e4(1200) 与 e4b(400) 的**正例率不同** (3.58% vs 5.50%);")
    print(f"     流行率是会翻转方差结论的量 (memory)。上面的 `--match-prevalence`")
    print(f"     就是为这个加的: 匹配后 ΔF1 的参照分布**整体抬高**(+0.0229→+0.0294),")
    print(f"     说明**高流行率本该给高 ΔF1** —— 而 e4b 反而最低, 与流行率的方向相反。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
