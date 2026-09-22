"""跨评估集相关性: 每个 run 的 ΔF1 是**run 的性质**还是**评估集的噪声**?

## 为什么这是全篇的命门

Q3(§7.4d) 刚确认: 种子间那 0.0309 的抖动**由数据顺序主导, 不是初值**。
沿着这个逻辑往下推一步, 会得到一个**可检验**的推论:

  若每个训练 run 有一个「run 级别的效应量」,
  那么**同一个 run 在两套不同的评估集上测, ΔF1 应该正相关**。
  反之, 若 ΔF1 主要是「这次抽到哪些块」的噪声,
  换一套块就该**重抽一次**, 两次测量应该**不相关**。

这个问题**决定整篇怎么收尾**:
  · 相关 ⇒ mixnorm-vs-own10 的差是**模型的真实性质**, 只是被噪声盖住;
  · 不相关 ⇒ 我们一直在测**评估集的抽样噪声**, 「效应」没有模型层面的对应物。

## 🔴 本分析的效力**极低**, 先写在前面

临界 |r| 由 n 现算 (`r_crit`): n=5 → 0.878, n=7 → 0.754。
**n 越小这个阈值越苛刻** —— 「两套都有的种子」这个 n 会随评估进度变,
所以脚本里**不留常数**, 每次跑都重报。

即便 n=7, **本分析的结果仍不足以单独改判任何东西**, 只能当**线索**。

数据是**已经看过的**(旧集来自 §7.2), 所以它不是预登记。
**真正预登记的部分是脚本末尾对未到种子的外推与定性判据** ——
`PENDING` 由「有 filt 无 e4」**现取**, 跑满 7 个后外推窗口自动关闭。

## 口径

**直接 import `g1_e4_analyze`**, 不重写 F1 计算 ——
两套评估集必须走**同一条代码路径**, 否则比的是两套口径
(memory: validate-surrogate-per-component-not-on-total, 同一族错误)。

用法:
  python scripts/g1_cross_evalset_corr.py
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_e4_analyze import ANNOT, SEEDS, ARMS, load, f1_of, suffix  # noqa: E402
# `r_crit` **定义在共享统计模块里, 这里 import** —— 不是本地再写一份。
# 理由: 它已经被复制漏修过一次 (`g1_margin_ds_analyze.py` 里另有一份写死的
# 0.878, 而那是 **n=5** 的值被用在 n=7 上)。两份定义 = 迟早只改一份。
from g1_boundary_dispersion import r_crit  # noqa: E402

# 🔴 `R_CRIT` 与 `PENDING` **都不写死** —— 与 `SE_CHUNK` 同族的第三个陈旧风险:
#    第一版是 `R_CRIT_N5 = 0.878` / `PENDING = [31337, 55555]`, 那是 **5 个种子**时
#    的快照。e4 一跑满 7 个种子, 临界 |r| 从 0.878 掉到 0.754, PENDING 变空 ——
#    两个数都会**静默过期**, 而且是**判据本身**过期 (memory:
#    hardcoded-conclusions-escape-reproduction 的第五面)。
#    所以: 临界 r 由 n 现算, PENDING 由「有 filt 无 e4」现取。
T_TARGET = 2.0                # 判据 C 用的 t 目标; 临界线 = SE_chunk × T_TARGET



# 🔴 临界线**现算**, 不写死 —— 今晚第二次踩「硬编码结论」的坑:
#    第一版把 0.0162 直接写在 f-string 里, 那是 §5.5 **预测**的 SE_chunk(0.0081)×2;
#    e4 上 SE_chunk 实测 0.0098 ⇒ 线应为 0.0196。写死的线会让判据永远用旧口径。
#    (memory: hardcoded-conclusions-escape-reproduction)
#
# 第二版虽然把 0.0098 提成了模块常数, 但它**仍是 5 种子时的临时值** —— 种子数一变
# (7 个齐了) SE_chunk 就会变, 线跟着变。所以现在改成**每次从数据现算**(见
# `se_chunk_of`), 模块里不再留任何可以过时的数。这条线是**判据本身**, 不能有陈旧默认值。


def per_seed(tag_set):
    """{seed: (ΔF1, own10_F1, mixnorm_F1)} —— 与 g1_e4_analyze 同一算法。"""
    P = {}
    for arm in ARMS:
        for s in SEEDS:
            e = load(tag_set, arm, s)
            if e is not None:
                P[(arm, s)] = e
    seeds = [s for s in SEEDS if ("own10", s) in P and ("mixnorm", s) in P]
    if not seeds:
        return {}, seeds, None
    cids = sorted(set.intersection(*(set(P[k]["per_chunk"]) for k in P)))
    e0 = P[("own10", seeds[0])]
    n_gt = sum(e0["per_chunk"][c]["n_gt"] for c in cids)
    n_pos = sum(1 for c in cids if e0["per_chunk"][c]["n_gt"] > 0)
    out = {}
    for s in seeds:
        pc_o = [P[("own10", s)]["per_chunk"][c] for c in cids]
        pc_m = [P[("mixnorm", s)]["per_chunk"][c] for c in cids]
        fo = f1_of(pc_o, range(len(cids)))
        fm = f1_of(pc_m, range(len(cids)))
        out[s] = (fo - fm, fo, fm)
    meta = {"n_chunks": len(cids), "n_gt": n_gt,
            "prevalence": n_pos / len(cids)}
    return out, seeds, meta, P, cids


def se_chunk_of(P, seeds, cids, boot=2000):
    """块级 bootstrap 的 SE_chunk —— **与 `g1_e4_analyze.py` 同一个做法**:
    7 个种子共用同一份重抽块, 所以它测的是「换一批块」带来的抖动。

    现算而不是写常数: 它随**种子数**变(5 种子时 0.0098), 写死就会拿旧线判新数据。
    """
    n = len(cids)
    rng = np.random.default_rng(0)
    vec = {k: [P[k]["per_chunk"][c] for c in cids] for k in P}
    db = np.empty(boot)
    for b in range(boot):
        idx = rng.integers(0, n, n)
        db[b] = np.mean([f1_of(vec[("own10", s)], idx)
                         - f1_of(vec[("mixnorm", s)], idx) for s in seeds])
    return float(db.std(ddof=1))


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xm, ym = x - x.mean(), y - y.mean()
    den = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    return float((xm * ym).sum() / den) if den else float("nan")


def spearman(x, y):
    def rank(a):
        a = np.asarray(a, float)
        o = a.argsort()
        r = np.empty(len(a), float)
        r[o] = np.arange(len(a))
        return r
    return pearson(rank(x), rank(y))


def fit(x, y):
    """一元最小二乘 y = a + b x。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    b = pearson(x, y) * y.std(ddof=1) / x.std(ddof=1)
    return y.mean() - b * x.mean(), b


def main():
    print("=" * 74)
    print("  跨评估集相关性: ΔF1 是 run 的性质, 还是评估集的噪声?")
    print("=" * 74)

    A, sa, ma, PA, cidsA = per_seed("filt")
    B, sb, mb, PB, cidsB = per_seed("e4")
    print(f"\n  旧 300 (filt): {len(sa)} 个种子 {sa}"
          + (f"，{ma['prevalence']:.2%} 正例率" if ma else ""))
    print(f"  k=4   (e4)  : {len(sb)} 个种子 {sb}"
          + (f"，{mb['prevalence']:.2%} 正例率" if mb else ""))

    # ---------- 自检: filt 必须复现 §7.2 ----------
    dv = np.array([A[s][0] for s in sa])
    print(f"\n  ── 自检 (filt 应复现 §7.2: ΔF1=+0.0340, SD=0.0309) ──")
    print(f"    ΔF1 = {dv.mean():+.4f}   SD = {dv.std(ddof=1):.4f}   n={len(dv)}")
    ok = abs(dv.mean() - 0.0340) < 0.002 and abs(dv.std(ddof=1) - 0.0309) < 0.002
    print(f"    {'✅ 复现' if ok else '❌ 未复现 —— 口径对不上, 下面全部作废'}")
    if not ok:
        return

    # ---------- 临界线: **从当前数据现算**, 不是常数 ----------
    SE_CHUNK = se_chunk_of(PB, sb, cidsB) if sb else float("nan")
    CRIT_LINE = SE_CHUNK * T_TARGET
    print(f"\n  ── 临界线 (现算, 不写死) ──")
    print(f"    SE_chunk (e4 @{len(sb)} 种子, 块级 bootstrap) = {SE_CHUNK:.4f}")
    print(f"    **临界线 = SE_chunk × {T_TARGET} = {CRIT_LINE:.4f}**")
    print(f"    (原预测口径 0.0081×2 = 0.0162 已作废; 5 种子时实测 0.0098×2 = 0.0196)")
    print(f"    ⚠️ 种子数一变这条线就变 —— 所以它**必须**现算。")

    both = sorted(set(sa) & set(sb))
    print(f"\n  ── 两套都有的种子: {both} (n={len(both)}) ──")
    print(f"    {'seed':>7}{'filt ΔF1':>12}{'e4 ΔF1':>12}")
    for s in both:
        print(f"    {s:>7}{A[s][0]:>+12.4f}{B[s][0]:>+12.4f}")

    x = [A[s][0] for s in both]
    y = [B[s][0] for s in both]

    # ---------- 主结果 ----------
    r = pearson(x, y)
    rho = spearman(x, y)
    RC = r_crit(len(both))
    print(f"\n  ── 相关性 (n={len(both)}, 临界 |r| = {RC:.3f} 现算) ──")
    print(f"    Pearson  r   = {r:+.3f}")
    print(f"    Spearman ρ   = {rho:+.3f}")
    print(f"    ⇒ **{'达到显著' if abs(r) > RC else '未达显著'}**"
          f" (|r| {'>' if abs(r) > RC else '<'} {RC:.3f})")
    print(f"    ⚠️ 临界 r 随 n 走: n=5 → {r_crit(5):.3f}, n=7 → {r_crit(7):.3f} "
          f"(**现算**, 上一版这里也印了一对写死的数)")

    # ---------- 逐点留一 ----------
    print(f"\n  ── 留一稳定性 (n={len(both)} 时单点极可能主导 r) ──")
    loos = []
    for i, s in enumerate(both):
        xx = [x[j] for j in range(len(x)) if j != i]
        yy = [y[j] for j in range(len(y)) if j != i]
        rr = pearson(xx, yy)
        loos.append(rr)
        print(f"    去掉 seed {s:>6}: r = {rr:+.3f}")
    print(f"    留一范围 [{min(loos):+.3f}, {max(loos):+.3f}]"
          f"  ⇒ {'⚠️ 摆动很大, r 不稳' if max(loos)-min(loos) > 0.5 else '尚算稳定'}")

    # ---------- 逐臂: 差值相关可能只是两臂一起动 ----------
    print(f"\n  ── 逐臂 (memory: 别只在差值上验证) ──")
    for idx, nm in ((1, "own10 F1"), (2, "mixnorm F1")):
        xa = [A[s][idx] for s in both]
        xb = [B[s][idx] for s in both]
        print(f"    {nm:<12} r = {pearson(xa, xb):+.3f}   "
              f"ρ = {spearman(xa, xb):+.3f}")
    print(f"    {'ΔF1 (差)':<12} r = {r:+.3f}")

    # ---------- 每臂自己的跨种子 SD: 边界离散有没有落到 F1 上 ----------
    # §7.4f 证明 mixnorm 的**边界**在 run 间散得多 (2.9×)。自然要问:
    # 这个边界抖动有没有**落到 F1 上**? 若有, 「跑一次就报 F1」对 mixnorm 是误导性的。
    # ⚠️ §3 里那个「mixnorm F1 种子间 SD = 0.1020 vs own10 0.0236」是
    #    **100% 正例的旧评估集**上的数, 属于已被推翻的流行率伪影 —— 这里重算自然流行率的。
    from scipy import stats as _st
    print(f"\n  ── 每臂自己的跨种子 SD (评估集不变, 只换种子) ──")
    for nm, A_ in (("filt (旧 300, 自然流行率)", A), ("e4 (k=4, 自然流行率)", B)):
        if len(A_) < 3:
            continue
        o = np.array([A_[s][1] for s in A_])
        m = np.array([A_[s][2] for s in A_])
        n = len(o)
        F = o.var(ddof=1) / m.var(ddof=1)
        pv = 2 * min(_st.f.cdf(F, n - 1, n - 1), 1 - _st.f.cdf(F, n - 1, n - 1))
        print(f"    {nm}  (n={n})")
        print(f"      own10   F1 均值 {o.mean():.4f}  跨种子 SD {o.std(ddof=1):.4f}")
        print(f"      mixnorm F1 均值 {m.mean():.4f}  跨种子 SD {m.std(ddof=1):.4f}")
        print(f"      **SD 比 mixnorm/own10 = {m.std(ddof=1)/o.std(ddof=1):.2f}×**"
              f"   (F = {F:.3f}, 双侧 p = {pv:.4f})")
    print(f"    ⚠️ 若两臂各自都高度相关而差值不相关 ⇒ 是**共同漂移**互相抵消,")
    print(f"       不是「run 效应」—— 两者的物理含义完全不同。")

    # ---------- 结论 ----------
    print(f"\n  ── 判读 ──")
    if abs(r) > RC:
        print(f"    r={r:+.3f} 过线 ⇒ 支持「run 级别效应」")
    elif r > 0.5:
        print(f"    r={r:+.3f} 为正但**未达显著** ⇒ 方向与「run 级别效应」一致,")
        print(f"       但 n={len(both)} 下这个证据**不能改判任何结论**, 只能作线索。")
    else:
        print(f"    r={r:+.3f} 接近 0 或为负 ⇒ **不提示** run 级别效应;")
        print(f"       与「ΔF1 主要是评估集抽样噪声」相容。")

    # ---------- 🔒 尚未到货种子的外推 (真正的样本外) ----------
    # PENDING **现取**, 不写死: 有 filt 无 e4 的就是还没到的。
    miss = [s for s in A if s not in B]
    print(f"\n  ── 🔒 预登记: 对 {len(miss)} 个**尚未到货**种子的外推 ──")
    if not miss:
        print("    (已无未到种子 —— 外推窗口关闭)")
    elif abs(r) < 0.3:
        print(f"    r={r:+.3f} 太弱, **不做外推** —— 外推只会把一个弱相关放大成假预测。")
        print(f"    改登记一条**定性的**判据 (见下), 不登记点预测。")
    a0 = b0 = float("nan")
    if both:
        a0, b0 = fit(x, y)
        print(f"    回归 e4 = {a0:+.4f} + {b0:+.4f} × filt")
    for s in miss:
        pred = a0 + b0 * A[s][0]
        print(f"      seed {s:>6}: filt {A[s][0]:+.4f}  ->  预测 e4 {pred:+.4f}")

    # ---------- 定性判据 (无论 r 大小都能登记) ----------
    print(f"\n  ── 🔒 预登记: 定性判据 (不依赖 r 的大小) ──")
    neg_filt = [s for s in A if A[s][0] < 0]
    print(f"    旧集上的负号种子: {neg_filt}")
    if 55555 in miss:
        print(f"    判据 A: **55555 在 e4 上是否仍是组内最低 (或为负)?**")
        print(f"      若是 ⇒ 「55555 特别差」是 run 的性质, **跨评估集稳**,")
        print(f"             事后线索(§7.4d)升级为可检验假设。")
        print(f"      若否 ⇒ 那个线索**只是评估集抽样**, 应撤回。")
    lo_filt = min(A, key=lambda s: A[s][0])
    print(f"    判据 B: e4 上 {len(sb)} 种子的**最小值**落在 filt 的最小值 "
          f"({lo_filt}) 或次小值上?")
    order_f = sorted(A, key=lambda s: A[s][0])
    print(f"      filt 升序: {order_f}")
    print(f"      e4   升序: {sorted(B, key=lambda s: B[s][0])}  ({len(sb)} 个)")
    print(f"    判据 C: e4 的 {len(sb)} 种子**均值**在哪一侧 of 临界线 {CRIT_LINE:.4f}?")
    print(f"            (= SE_chunk {SE_CHUNK:.4f} × 2; ⚠️ 线随实测走, 别写死 ——")
    print(f"             原预测口径 0.0081 给出 0.0162, k=4 文档 §5.8b)")
    if sb:
        dv4 = np.array([B[s][0] for s in sb])
        print(f"      实测 e4 均值 = {dv4.mean():+.4f}  "
              f"⇒ {'**线上** (超临界线)' if dv4.mean() > CRIT_LINE else '线下 (未超临界线)'}"
              f"   跨种子 SD = {dv4.std(ddof=1):.4f}")

    # ---------- 🔴 选择偏差: 已完成的种子是否仍是"高选子样本" ----------
    print(f"\n  ── 🔴 选择偏差检查 (memory: 按结果挑点) ──")
    allf = np.array([A[s][0] for s in sa])
    have = np.array([A[s][0] for s in both])
    print(f"    filt 全 {len(sa)} 种子均值 = {allf.mean():+.4f}")
    print(f"    filt 已完成的 {len(both)} 个均值 = {have.mean():+.4f}"
          f"   (差 {have.mean()-allf.mean():+.4f})")
    if have.mean() - allf.mean() > 0.005:
        print(f"    ⚠️ **已完成的这批在旧集上系统性偏高** ⇒ 它们是**高选子样本**。")
        print(f"       按回归均值, 它们在新评估集上**本来就会往下掉一截**,")
        print(f"       这个回落里**没有信息**。")
        if abs(r) > 0.2:
            bias = b0 * (have.mean() - allf.mean())
            print(f"       用回归斜率 {b0:+.3f} 估这个偏差: "
                  f"{b0:+.3f} × {have.mean()-allf.mean():+.4f} = {bias:+.4f}")
            print(f"       ⇒ 当前 e4 均值若是无偏 {len(both)} 种子样本, "
                  f"应低约 {abs(bias):.4f}")
            print(f"       ⚠️ 这个更正**本身建立在弱相关(r={r:+.2f})上**, 只报方向不报数。")
    else:
        print(f"    ✅ 已完成这批在旧集上**不偏高** ⇒ 无明显选择偏差。")

    # ---------- 对临界线的影响: 缺的种子正是极端值 ----------
    if miss:
        print(f"\n  ── ⚠️ 缺的种子会怎样移动均值 ──")
        cur = np.array([B[s][0] for s in sb])
        # 🔴 这两行的种子数原来**写死**成「6 种子」「5 种子」: 前者在 sb 有 6 个时
        #    实际会变成 7 个, 后者印 5 而实际是 6 —— 与 `n=7, 临界 |r| = 0.878`
        #    同一族: 写死的数挨着现算的均值/SD 印在同一行, 借走了它的可信度
        #    (memory: hardcoded-conclusions-escape-reproduction 第六面)。
        #    下面一律现算; 唯一的例外是「+1」——每次只补**一个**缺失种子。
        print(f"    当前 {len(sb)} 种子均值 = {cur.mean():+.4f}, SD = {cur.std(ddof=1):.4f}")
        for s in miss:
            # 两种极端情形: 该种子在 e4 上与 filt 同号同量级 / 归零
            for label, v in ((f"沿用 filt 的 {A[s][0]:+.4f}", A[s][0]),
                             ("归零 (纯评估集噪声)", 0.0)):
                n_new = len(sb) + 1          # 补这一个之后的总数, 现算
                newmean = (cur.sum() + v) / n_new
                print(f"      若 {s} = {label:<26} ⇒ {n_new} 种子均值 {newmean:+.4f}"
                      f"  ({'线上' if newmean > CRIT_LINE else '线下'})")
        print(f"    📌 注意 §5.7 预登记的 e4 跨种子 SD ∈ [0.021, 0.026],")
        print(f"       而当前 {len(sb)} 种子 SD = {cur.std(ddof=1):.4f} **低于该区间** ——")
        print(f"       但 55555 这个极端值不在样本里, 加上它 SD 会**升**。")
        print(f"       所以 §5.7 那条**此刻还不能判定命中与否**, 等齐了再说。")


if __name__ == "__main__":
    main()
