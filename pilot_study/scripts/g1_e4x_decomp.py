"""k=4b (e4x, 1600 块) 的**会话级分解** —— 纯读盘, 0 GPU。

## 为什么要做, 以及**先要拆掉一个假发现**

早先一次内联探针打出「100 个会话里 **65 个 ΔF1 ≡ 0**」, 看着像「效应集中在
少数会话」。**那不是发现, 是定义**:

    `f1_of` 在一个块集合上 tp=fp=n_gt=0 时, p=r=0 ⇒ 直接 `return 0.0`。
    于是**任何不含真值事件的会话, 两臂都返回 0.0 ⇒ ΔF1 恒等于 0**。

实测: e4x 的 65 个真值事件落在 **64 个块 / 46 个会话**里 ⇒ **54 个会话
（100−46）按构造就是 0**。⇒ 报分解前必须先把这些会话摘出去, 否则
「恒零会话」会被读成「这些会话没有效应」——一句信息量为零的话
(memory: hardcoded-conclusions-escape-reproduction 里的「恒真自检」同族)。

## 这个脚本真正要回答的

1. **效应挂在哪个计数上** —— ΔF1 是 tp/fp 的比值, 拆成**整数计数** Δtp/Δfp
   才知道它来自「多召回了事件」还是「少报了假阳性」。
2. **F1 这个端点扔掉了多少信息** —— 1536/1600 个块没有真值事件, 这些块上
   模型报了多少假阳性, F1 **完全看不见** (两臂都是 0.0)。
3. **块到底独不独立** —— 会话整簇 bootstrap vs 块级 bootstrap, 出设计效应。
   （簇内相关会让块级 SE 偏小; 这是本项目已知要查的一条。）
4. **别被单个会话带走** —— 留一会话 (LODO) 看 ΔF1 的摆幅, 以及符号翻不翻。

## 判据纪律

- 簇 bootstrap 也是**抽出来的**, 报 SE 必须连 MC 误差一起报
  (memory: stochastic-estimate-as-threshold-needs-mc)。
- 任何「按会话切」的 ΔF1 一律走 `g1_e4_analyze.d_of`, **不重写算式** ——
  两份定义 = 迟早只改一份。

用法: python scripts/g1_e4x_decomp.py [--set e4x]
"""
import argparse
import collections
import sys

import numpy as np

import g1_e4_analyze as A


def sess_of(cid):
    """`0020a0c5_ch0_t1400` -> `0020a0c5`。"""
    return cid.rsplit("_", 2)[0]


def counts(vec, idx, arm, seed):
    """在块下标集合上数 tp / fp / n_gt —— **整数**, 没有比值噪声。"""
    tp = fp = gt = 0
    for i in idx:
        e = vec[(arm, seed)][i]
        tp += e["tp"]; fp += e["fp"]; gt += e["n_gt"]
    return tp, fp, gt


def boot_cluster(stat_fn, by_sess, boot=A.B, rng_seed=0):
    """**簇级** bootstrap: 抽会话, 把该会话的块**整批**带进来, 再算 `stat_fn`。

    块若在同一会话内相关, 块级 bootstrap 会低估 SE; 簇级是这时该用的口径。
    注意每轮进来的块**总数是变的** (抽到的会话大小不同) —— 这正是簇抽样的
    正确行为, 不是 bug。

    `stat_fn(idx) -> float`: 传进来的下标集合上要算的标量。做成参数是为了让
    ΔF1 和 Δfp 共用同一套重抽样机器, 而不是各写一份。
    """
    names = sorted(by_sess)
    groups = [np.asarray(by_sess[n]) for n in names]
    n = len(groups)
    rng = np.random.default_rng(rng_seed)
    db = np.empty(boot)
    for b in range(boot):
        pick = rng.integers(0, n, n)
        ii = np.concatenate([groups[j] for j in pick])
        db[b] = stat_fn(ii)
    return float(db.std(ddof=1))


def stat_df1(vec, seeds):
    return lambda idx: float(np.mean(list(A.d_of(vec, seeds, idx).values())))


def dcount_per_seed(vec, idx, which, arm_a="own10", arm_b="mixnorm", seeds=None):
    """逐种子的 Δ计数 (Δtp 或 Δfp) —— **先把逐种子的值都留着**, 别急着取均值。

    🔴 为什么必须逐种子看: fp 计数**跨种子摆得极大**。e4x 无事件 864 块上
    逐种子 Δfp = [+112, −82, +34, +18, −19, −189, +52] —— **符号一个个翻**,
    跨种子 SD ≈ 99。只看 7 种子均值 (−10.6) 会以为「两臂差不多」,
    只看 seeds[0] 会得出「own10 多报了 112 个」—— **两个都是错的**。
    """
    idxk = {"tp": 0, "fp": 1}[which]      # counts() 返回 (tp, fp, n_gt)
    return np.array([counts(vec, idx, arm_a, s)[idxk] - counts(vec, idx, arm_b, s)[idxk]
                     for s in (seeds or [])])


def se_of(v):
    """跨种子 SE (SD/√n) —— bootstrap **看不见**这一项 (它只重抽块)。"""
    v = np.asarray(v, dtype=float)
    return float(v.std(ddof=1) / np.sqrt(len(v)))


def stat_dcount(vec, seeds, which, arm_a="own10", arm_b="mixnorm"):
    """Δ计数 (Δtp / Δfp), 逐种子**先算差再跨种子平均** (与 `d_of` 同构)。

    计数是**整数**, 所以这个量没有 F1 那种「分子是小整数 ⇒ 按定义恒零」的毛病。
    ⚠️ `which` 必须参数化: 早先只写了 fp 版, 于是 Δtp 那一行**只报了 SE_种子**
    —— 正是本文件顶部警告的「新端点悄悄只继承一个方差源」。
    """
    idxk = {"tp": 0, "fp": 1}[which]
    def f(idx):
        return float(np.mean([counts(vec, idx, arm_a, s)[idxk]
                              - counts(vec, idx, arm_b, s)[idxk] for s in seeds]))
    return f


def boot_mc(fn, reps=5):
    """把一个 bootstrap 估计量重复 `reps` 次, 返回 (点估计, MC 标准差)。

    点估计用 `rng_seed=0` 那一项, 与别处报的数**逐位一致**。
    """
    est = np.array([fn(r) for r in range(reps)])
    return float(est[0]), float(est.std(ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="tag_set", default="e4x")
    args = ap.parse_args()

    D = A.paired(args.tag_set)
    if D is None:
        print("配对数不足 2, 退出。")
        return 1
    vec, cids, seeds = D["vec"], D["cids"], D["seeds"]
    n = len(cids)
    print("=" * 78)
    print(f"  {args.tag_set} 会话级分解   块 {n}   配对种子 {len(seeds)}")
    print("=" * 78)

    # ---------- 结构 ----------
    by_sess = collections.defaultdict(list)
    for i, c in enumerate(cids):
        by_sess[sess_of(c)].append(i)
    n_gt_blk, n_gt_ev = 0, 0
    ev_of = {}
    for name, ii in by_sess.items():
        g = sum(vec[("own10", seeds[0])][i]["n_gt"] for i in ii)
        ev_of[name] = g
        n_gt_ev += g
        n_gt_blk += sum(1 for i in ii if vec[("own10", seeds[0])][i]["n_gt"] > 0)
    with_ev = sorted(k for k, v in ev_of.items() if v > 0)
    without = sorted(k for k, v in ev_of.items() if v == 0)
    print(f"\n── 结构 ──")
    print(f"  会话 {len(by_sess)}   每会话块数 {sorted(set(len(v) for v in by_sess.values()))}")
    print(f"  真值事件 {n_gt_ev} 个, 落在 {n_gt_blk} 个块 / **{len(with_ev)} 个会话**")
    print(f"  ⇒ **{len(without)} 个会话一个事件都没有** ⇒ 这些会话的 ΔF1 **按定义为 0**")
    print(f"     (f1_of 在 tp=fp=n_gt=0 时 return 0.0, 两臂一样) ⇒ 不是发现, 是定义")

    # ---------- 计数分解 ----------
    print(f"\n── 效应挂在哪个计数上 (整数, 全集 {n} 块) ──")
    print(f"  {'arm':<9}{'tp':>6}{'fp':>7}{'n_gt':>7}{'P':>8}{'R':>8}{'F1':>8}")
    tot = {}
    for arm in A.ARMS:
        tp = np.mean([counts(vec, range(n), arm, s)[0] for s in seeds])
        fp = np.mean([counts(vec, range(n), arm, s)[1] for s in seeds])
        gt = counts(vec, range(n), arm, seeds[0])[2]
        p = tp / max(1e-9, tp + fp); r = tp / max(1e-9, gt)
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        tot[arm] = (tp, fp, gt)
        print(f"  {arm:<9}{tp:>6.1f}{fp:>7.1f}{gt:>7}{p:>8.4f}{r:>8.4f}{f1:>8.4f}")
    # 逐种子的计数差 + 两个方差源 —— 计数本身跨种子就很吵, 不报 SE 会过度解读
    print(f"\n  Δ(own10−mixnorm) 逐种子, 带两个方差源:")
    allidx = list(range(n))
    for which in ("tp", "fp"):
        v = dcount_per_seed(vec, allidx, which, seeds=seeds)
        se_s = se_of(v)
        se_c = boot_cluster(stat_dcount(vec, seeds, which), by_sess, rng_seed=0)
        se_all = float(np.hypot(se_s, se_c))
        flag = "✅ 有信号" if abs(v.mean()) > 2 * se_all else "🔴 与 0 无异"
        print(f"    Δ{which} = {v.mean():+.1f}  SE_种子 {se_s:.1f}  "
              f"SE_簇 {se_c:.1f}  SE_合并 {se_all:.1f}   {flag}")
        print(f"         逐种子 {' '.join(f'{x:+.0f}' for x in v)}")
    print(f"  ⚠️ 逐种子的计数**摆幅极大** (fp 跨种子 SD ~100) —— 均值那一列看着"
          f"平稳, 是**平均**把它压平的;")
    print(f"     谁只看某一次 run 的计数, 就会把运行噪声读成臂间差异。")

    # ---------- F1 扔掉了多少信息 ----------
    # 把块按「有没有真值事件」切成两半 —— F1 只在前一半上有定义。
    idx_noev = [i for name in without for i in by_sess[name]]
    idx_ev = [i for name in with_ev for i in by_sess[name]]
    print(f"\n── F1 这个端点扔掉了什么 ──")
    # ⚠️ 这一格有两个方差源, **必须都报**: bootstrap 重抽的是块, 它**看不见**
    #    跨种子的运行噪声。第一版只印了 SE_簇 (=19.9), 而跨种子 SE 是 37 ⇒
    #    表看上去像「差 −10.6 ± 19.9, 差不多两倍」, 实际合并 SE ≈ 42, 什么都没有。
    print(f"  {'子集':<20}{'块':>6}{'Δfp':>8}{'SE_种子':>9}{'SE_簇':>8}"
          f"{'SE_合并':>9}{'逐种子 Δfp':>34}")
    # 🔴 子集的簇 bootstrap **必须在子集的会话里重抽**。第一版把全集的 `by_sess`
    #    传进去 ⇒ 三行的 SE_簇 印出同一个 19.9 —— 因为 `stat_fn` 拿到的是
    #    「全场重新拼出来的块」, 与 `idx` 无关。看到三行同值才发现。
    se_c_of = {}
    for label, idx, names in (("无事件 (F1 恒 0)", idx_noev, without),
                              ("有事件", idx_ev, with_ev),
                              ("全体", list(range(n)), sorted(by_sess))):
        v = dcount_per_seed(vec, idx, "fp", seeds=seeds)
        se_s = se_of(v)
        sub = {k: by_sess[k] for k in names}
        se_c = boot_cluster(stat_dcount(vec, seeds, "fp"), sub, rng_seed=0)
        se_c_of[label] = se_c
        se_all = float(np.hypot(se_s, se_c))
        print(f"  {label:<20}{len(idx):>6}{v.mean():>+8.1f}{se_s:>9.1f}{se_c:>8.1f}"
              f"{se_all:>9.1f}   {' '.join(f'{x:+.0f}' for x in v)}")
    v0 = dcount_per_seed(vec, idx_noev, "fp", seeds=seeds)
    se0 = float(np.hypot(se_of(v0), se_c_of["无事件 (F1 恒 0)"]))
    print(f"  ⇒ 无事件的 {len(idx_noev)} 个块 ({len(idx_noev)/n:.0%}) 上, "
          f"Δfp = {v0.mean():+.1f} ± {se0:.1f} (合并 SE) ⇒ "
          f"{'有信号' if abs(v0.mean()) > 2*se0 else '**与 0 无异**'}。")
    print(f"     ⚠️ **逐种子那一列会翻号** (跨种子 SD {v0.std(ddof=1):.0f}) —— 谁只看 "
          f"seeds[0] 就会得到 {v0[0]:+.0f} 并当成发现。")
    print(f"     但要点仍成立: 这些块的 ΔF1 **按定义恒为 0**, 这个量 F1 根本看不见。")

    # ---------- 逐会话 ΔF1 (只对有事件的会话) ----------
    print(f"\n── 逐会话 ΔF1 (只列有事件的 {len(with_ev)} 个会话, 按 ΔF1 降序) ──")
    rows = []
    for name in with_ev:
        ii = by_sess[name]
        dd = A.d_of(vec, seeds, ii)
        m = float(np.mean(list(dd.values())))
        sgn = sum(1 for v in dd.values() if v > 0) - sum(1 for v in dd.values() if v < 0)
        rows.append((m, name, ev_of[name], len(ii), sgn))
    rows.sort(reverse=True)
    print(f"  {'会话':<12}{'事件':>5}{'块':>4}{'ΔF1':>9}{'种子正负':>9}")
    for i, (m, name, g, nb, sgn) in enumerate(rows):
        if i == 6:
            print(f"  {'…':<12}")
        if i < 6 or i >= len(rows) - 6:
            print(f"  {name:<12}{g:>5}{nb:>4}{m:>+9.4f}{sgn:>+9d}")
    vals = np.array([r[0] for r in rows])
    print(f"  有事件会话的 ΔF1: 均值 {vals.mean():+.4f}  SD {vals.std(ddof=1):.4f}  "
          f"正号 {int((vals>0).sum())}/{len(vals)}")
    print(f"  ⚠️ 每个会话只有 {min(ev_of[k] for k in with_ev)}~"
          f"{max(ev_of[k] for k in with_ev)} 个事件 ⇒ F1 的分子是小整数, "
          f"**这个逐会话值没有分辨率**, 别按它排序做结论。")

    # ---------- 块独立吗: 簇 vs 块 ----------
    print(f"\n── 块到底独不独立: 簇 bootstrap vs 块级 bootstrap ──")
    fn1 = stat_df1(vec, seeds)
    se_chunk, mc_chunk = boot_mc(lambda r: A.boot_se(vec, cids, seeds, boot=A.B, rng_seed=r))
    se_clust, mc_clust = boot_mc(lambda r: boot_cluster(fn1, by_sess, rng_seed=r))
    print(f"  块级 SE_chunk   = {se_chunk:.6f}   (MC ±{mc_chunk:.6f}, {mc_chunk/se_chunk:.1%})")
    print(f"  会话簇 SE_clust = {se_clust:.6f}   (MC ±{mc_clust:.6f}, {mc_clust/se_clust:.1%})")
    ratio = se_clust / se_chunk
    deff = ratio ** 2
    print(f"  比值 = **{ratio:.3f}×**   设计效应 deff = {deff:.3f}")
    print(f"  ⇒ {'✅ 基本可交换 (簇内相关可忽略), 块级 SE 站得住' if abs(ratio-1) < 0.15 else '🔴 簇内相关不可忽略, 块级 SE 偏小'}")

    # ---------- 留一会话 ----------
    print(f"\n── 留一会话 (LODO): 去掉任一**有事件的**会话, ΔF1 摆到哪 ──")
    base = float(np.mean(list(A.d_of(vec, seeds, range(n)).values())))
    loo = []
    for name in with_ev:
        keep = np.array([i for i in range(n) if i not in set(by_sess[name])])
        loo.append(float(np.mean(list(A.d_of(vec, seeds, keep).values()))))
    loo = np.array(loo)
    print(f"  全集 ΔF1 = {base:+.4f}")
    print(f"  LODO 范围 [{loo.min():+.4f}, {loo.max():+.4f}]  摆幅 {loo.max()-loo.min():.4f}")
    print(f"  去掉后翻号的会话数: {int((np.sign(loo) != np.sign(base)).sum())} / {len(loo)}")
    worst = rows[0][1]
    keep = np.array([i for i in range(n) if i not in set(by_sess[worst])])
    w = float(np.mean(list(A.d_of(vec, seeds, keep).values())))
    print(f"  最大影响来自 {worst} (ΔF1 {rows[0][0]:+.4f}, {ev_of[worst]} 事件) "
          f"⇒ 去掉后 {base:+.4f} → {w:+.4f}")

    print(f"\n── 判读 ──")
    print(f"  ✅ 能确立: 效应在这一集上是 {'线上' if base > 2*se_chunk else '线下'} "
          f"(均值 {base:+.4f} vs 线 {2*se_chunk:.4f})")
    print(f"  ❌ 不能确立: 「效应挂在 tp 还是 fp 上」—— 计数差跨种子 SD ~100, "
          f"而臂间差只有个位数 ⇒ **计数分解没有分辨率**。")
    print(f"  ❌ 不能确立: 「效应集中在哪几个会话」—— {len(with_ev)} 个有事件的会话里多数只有 1 个事件,")
    print(f"     F1 的最小非零变化都是 0.5 量级, 逐会话排序纯属噪声排序。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
