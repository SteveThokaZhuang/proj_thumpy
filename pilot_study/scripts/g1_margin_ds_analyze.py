"""§7.4f 对账: margin 探针的 B 组 (定 seed=42、变 data_seed) —— 边界离散是**初值**还是**数据顺序**?

## 预登记在哪

判据写在 `docs/pilot_study/2026-09-17_status.md` **§7.4f**,
封存时刻 **2026-09-18 01:14:11**（当时 `g1_margin_*_ds*.json` 只有 1 个）。
本脚本**只做对账, 不挑说法**: 判据命中就写命中, 落空就写落空。

预登记判据: B 组 SD / A 组 SD ∈ [0.7, 1.4] ⇒ 结论保持; < 0.5 ⇒ 代理失真, 以探针为准。

## 口径

**必须与 §5.1 完全一致**, 否则对不上那个 2.88×:
  每适配器 → 该适配器在 400 块上的 **块平均 margin_max** → 跨适配器求 **SD**。
A 组变 seed (data_seed≡42), B 组变 data_seed (seed≡42)。

**共同块**: 28 个文件取交集。A 组与 B 组用的是**同一份** `g1_margin_ds_ids.txt`,
但探针有可能漏块 (argmax 落在两类之外会 return None), 所以仍要取交集而不是假设 400。

用法: python scripts/g1_margin_ds_analyze.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 置换统计量**从代理脚本 import**, 不重写 —— 两边必须是同一个函数,
# 否则「探针 2.4× 对代理 2.78×」这种比较就变成比两套口径了
# (memory: validate-surrogate-per-component-not-on-total, 同一族错误)。
from g1_boundary_dispersion import var_ratio_perm, r_crit  # noqa: E402

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
SEEDS = [42, 1234, 2024, 31337, 3407, 55555, 7]
ARMS = ["own10", "mixnorm"]

# §5.1 / §7.1 的探针口径值, 用作自检锚点
ANCHOR_RATIO = 2.88
ANCHOR_F = 8.276


def margin_path(arm, key, grp):
    """grp='A' -> `_s{seed}`; grp='B' -> `_ds{ds}`。

    ⚠️ A 组**即使 seed=42 也带 `_s42`** —— 与评估文件不同!
    评估是 `g1_eval_{arm}.json`(42 无后缀) + `g1_eval_{arm}_s{seed}.json`,
    但 margin 产物一律是 `g1_margin_{arm}_s{seed}.json`。
    第一版照着评估的命名写, 于是 seed=42 两个都读不到, 静默少 2 个文件。
    """
    if grp == "A":
        return f"{ANNOT}/g1_margin_{arm}_s{key}.json"
    return f"{ANNOT}/g1_margin_{arm}_ds{key}.json"


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def sd_ci(v):
    from scipy import stats as st
    n, s = len(v), np.std(v, ddof=1)
    return (s, s * np.sqrt((n - 1) / st.chi2.ppf(0.975, n - 1)),
            s * np.sqrt((n - 1) / st.chi2.ppf(0.025, n - 1)))


def f_test(va, vb):
    from scipy import stats as st
    n = len(va)
    F = np.var(va, ddof=1) / np.var(vb, ddof=1)
    return F, min(2 * min(st.f.cdf(F, n - 1, n - 1),
                          1 - st.f.cdf(F, n - 1, n - 1)), 1.0)


def main():
    print("=" * 78)
    print("  §7.4f 对账: 边界离散 —— 初值 vs 数据顺序 (margin 探针 B 组)")
    print("=" * 78)

    M, missing = {}, []
    for grp in ("A", "B"):
        for arm in ARMS:
            for k in SEEDS:
                p = margin_path(arm, k, grp)
                d = load(p)
                if d is None:
                    missing.append(os.path.basename(p))
                else:
                    M[(grp, arm, k)] = d
    complete = not missing
    print(f"\n  产物 {len(M)}/28" + (f"   ❌ 缺: {missing}" if missing else "   ✅"))
    if not complete:
        print(f"  🔴 **数据不完整 —— 下面的 A/B 判据一律不作数**, 只当管线自检看。")
        print(f"     (memory: 跑批没结束就不算最终统计量; 部分样本上的显著性最容易骗人)")

    # ---------- 自检 0: 块与符号 ----------
    cids = sorted(set.intersection(*(set(d) for d in M.values())))
    sizes = sorted({len(d) for d in M.values()})
    print(f"\n  ── 自检 0: 共同块 ──")
    print(f"    各文件块数 {sizes}   交集 {len(cids)}"
          f"   {'✅' if sizes == [len(cids)] else '⚠️ 有文件块数不一致, 已取交集'}")

    # 正确的不变式是 **单边** 的: argmax 落在 ev 类 ⇒ ev_max >= no_max ⇒ margin_max >= 0。
    # 反向**不成立**, 两个原因:
    #   (a) ev/no 两类**不构成全词表** —— 全局 argmax 可以落在第三类 token 上,
    #       此时 argmax_is_event=False 而 margin_max 仍可 > 0;
    #   (b) 并列 (margin_max 恰为 0) 时探针记的是 argmax_is_event=True。
    # 第一版写成 `(margin_max > 0) != argmax_is_event` 的**双向**断言, 于是把 (b)
    # 全判成矛盾 (401 处, 实测**全部**是 0.0 并列, 真矛盾 0 处)。
    # 这属于 memory 里「永远会失败的自检 = 假警报」那一面: 断言先写错了, 报警就没意义。
    contradict = tied = 0
    for d in M.values():
        for r in d.values():
            if r["argmax_is_event"]:
                if r["margin_max"] < 0:
                    contradict += 1
                elif r["margin_max"] == 0.0:
                    tied += 1
    print(f"  ── 自检 1: argmax_is_event=True ⇒ margin_max ≥ 0 (单边) ──")
    print(f"    真矛盾 {contradict}/{len(M)*len(cids)}"
          f"   {'✅' if contradict == 0 else '❌ 口径有问题, 停'}"
          f"   (另有 {tied} 处恰好并列在 0.0, 属正常)")

    # ---------- 主量: 每适配器块平均 margin ----------
    mean_m = {(g, a, k): float(np.mean([M[(g, a, k)][c]["margin_max"]
                                        for c in cids]))
              for (g, a, k) in M}

    # ---------- 自检 2: A 组跨臂应复现 §5.1 的 2.88× / F=8.276 ----------
    print(f"\n  ── 自检 2: A 组跨臂应复现 §5.1 (SD 比 {ANCHOR_RATIO}×, F={ANCHOR_F}) ──")
    for arm in ARMS:
        v = [mean_m[("A", arm, s)] for s in SEEDS]
        print(f"    {arm:<8} " + "  ".join(f"{x:+.3f}" for x in v) + f"   SD {np.std(v, ddof=1):.3f}")
    oa = [mean_m[("A", "own10", s)] for s in SEEDS]
    ma = [mean_m[("A", "mixnorm", s)] for s in SEEDS]
    F, p = f_test(ma, oa)
    print(f"    A 组 SD 比 = {np.std(ma,ddof=1)/np.std(oa,ddof=1):.2f}×  "
          f"F = {F:.3f}  p = {p:.4f}"
          f"   {'✅ 复现' if abs(F-ANCHOR_F) < 1.5 else '⚠️ 与 §5.1 对不上, 下面谨慎'}")

    # ---------- 🔒 预登记判据 ----------
    print(f"\n  ── 🔒 预登记判据: B 组 SD / A 组 SD ∈ [0.7, 1.4] ? ──")
    print(f"    {'arm':<8}{'A组 SD':>10}{'B组 SD':>10}{'B/A':>8}"
          f"{'95%CI(A)':>20}{'95%CI(B)':>20}")
    verdicts = {}
    for arm in ARMS:
        if not all(("A", arm, s) in mean_m and ("B", arm, s) in mean_m
                   for s in SEEDS):
            print(f"    {arm:<8} (该臂 A/B 未齐, 跳过)")
            continue
        va = np.array([mean_m[("A", arm, s)] for s in SEEDS])
        vb = np.array([mean_m[("B", arm, d)] for d in SEEDS])
        sa, la, ha = sd_ci(va)
        sb, lb, hb = sd_ci(vb)
        r = sb / sa
        verdicts[arm] = r
        print(f"    {arm:<8}{sa:>10.3f}{sb:>10.3f}{r:>8.2f}"
              f"{f'[{la:.3f}, {ha:.3f}]':>20}{f'[{lb:.3f}, {hb:.3f}]':>20}")
    for arm, r in verdicts.items():
        tag = ("✅ 在 [0.7,1.4] 内 ⇒ §7.4f 结论保持" if 0.7 <= r <= 1.4
               else ("❌ < 0.5 ⇒ 代理失真, 以探针为准" if r < 0.5
                     else "⚠️ 落在灰区, 不能改判"))
        if not complete:
            tag = "🔴 数据不完整, 此判据不作数"
        print(f"    {arm:<8} B/A = {r:.2f}   {tag}")

    # ---------- 每适配器的位置 + 留一 ----------
    print(f"\n  ── 逐适配器块平均 margin (A 变 seed / B 变 data_seed) ──")
    for arm in ARMS:
        if not all(("A", arm, s) in mean_m and ("B", arm, s) in mean_m
                   for s in SEEDS):
            print(f"    {arm}  (A/B 未齐, 跳过)")
            continue
        print(f"    {arm}")
        print(f"      {'key':>8}{'A(变seed)':>12}{'B(变ds)':>12}{'差':>10}")
        for s in SEEDS:
            a, b = mean_m[("A", arm, s)], mean_m[("B", arm, s)]
            print(f"      {s:>8}{a:>+12.3f}{b:>+12.3f}{b-a:>+10.3f}")
        va = np.array([mean_m[("A", arm, s)] for s in SEEDS])
        vb = np.array([mean_m[("B", arm, s)] for s in SEEDS])
        rr = [np.std(np.delete(vb, i), ddof=1) / np.std(np.delete(va, i), ddof=1)
              for i in range(len(SEEDS))]
        print(f"      留一 B/A: " + "  ".join(f"{x:.2f}" for x in rr)
              + f"   范围 [{min(rr):.2f}, {max(rr):.2f}]")

    # ---------- 跨臂: 直接用探针量出 B 组的两臂比 ----------
    # 代理在 B 组给的是 2.07×(置换 p=0.0761), 比 A 组的 2.78× 弱一档。
    # 现在有了 B 组的真实 margin, 可以**直接量**, 不必再靠 fire rate 代理。
    # 统计量用与 `g1_boundary_dispersion.var_ratio_perm` **完全相同**的那个
    # (14 个值随机分 7/7, 看 var(mixnorm)/var(own10) 多大), 两条口径才能直接比。
    print(f"\n  ── 跨臂 SD 比 (探针直接量; 代理口径 A=2.78× p=0.0355 / "
          f"B=2.07× p=0.0761) ──")
    for grp, gname in (("A", "变seed"), ("B", "变ds")):
        if not all((grp, arm, k) in mean_m for arm in ARMS for k in SEEDS):
            print(f"    {gname:<8} (未齐, 跳过)")
            continue
        vo = np.array([mean_m[(grp, "own10", k)] for k in SEEDS])
        vm = np.array([mean_m[(grp, "mixnorm", k)] for k in SEEDS])
        obs, pk = var_ratio_perm(vo, vm)
        F, pn = f_test(vm, vo)
        print(f"    {gname:<8} own10 SD {np.std(vo,ddof=1):.3f}   "
              f"mixnorm SD {np.std(vm,ddof=1):.3f}   "
              f"比 {np.std(vm,ddof=1)/np.std(vo,ddof=1):.2f}×   "
              f"置换 p = {pk:.4f}   (正态 F={F:.3f} p={pn:.3f}, 仅参照)")
    pAB = []
    for grp in ("A", "B"):
        if not all((grp, arm, k) in mean_m for arm in ARMS for k in SEEDS):
            continue
        vo = np.array([mean_m[(grp, "own10", k)] for k in SEEDS])
        vm = np.array([mean_m[(grp, "mixnorm", k)] for k in SEEDS])
        pAB.append(var_ratio_perm(vo, vm)[1])
    if len(pAB) == 2:
        from scipy import stats as st
        chi = -2 * sum(np.log(pAB))
        print(f"    Fisher 合并两组的跨臂证据: χ² = {chi:.3f} (df=4)  ⇒  "
              f"**p = {float(st.chi2.sf(chi, 4)):.4f}**")
        print(f"      ⚠️ A/B 两套适配器共享 seed42/ds42 那一个有效配置 ⇒ 合并略偏乐观,")
        print(f"         报的时候说「~0.0x」, 不说小数点后三位。")

    # ---------- 桥接: margin 与 fire rate 代理对不对得上 ----------
    print(f"\n  ── 桥接: 块平均 margin(400 块 E2) vs fire rate(300 块 filt) ──")
    print(f"    (两者块集不同, 只在**适配器层面**比)")
    for grp, lbl in (("A", "变seed"), ("B", "变ds")):
        print(f"    {lbl}")
        for arm in ARMS:
            # ⚠️ 两个条件都要查: _filt 评估在, **并且**该 (组,臂,种子) 有 margin 产物。
            # 少了后者就是 KeyError —— 第一版只查了前者, 于是在 B 组 own10
            # 还没跑完时崩在 `mean_m[(grp, arm, s)]`。本脚本的设计前提是
            # **任何时候都能跑**(数据不全时只当管线自检), 就不能有会崩的路径。
            if not all(os.path.exists(
                    f"{ANNOT}/g1_eval_"
                    + (f"{arm}{'' if k == 42 else f'_s{k}'}" if grp == "A"
                       else f"{arm}_ds{k}") + "_filt.json") for k in SEEDS) \
               or not all((grp, arm, k) in mean_m for k in SEEDS):
                print(f"      {arm:<8} (缺 margin 产物或缺 _filt 评估, 跳过)")
                continue
            fr = []
            for k in SEEDS:
                tag = f"{arm}{'' if k == 42 else f'_s{k}'}" if grp == "A" \
                    else f"{arm}_ds{k}"
                p = f"{ANNOT}/g1_eval_{tag}_filt.json"
                pc = next(iter(json.load(open(p)).values()))["per_chunk"]
                fr.append(np.mean([1.0 if pc[c]["n_pred"] > 0 else 0.0
                                   for c in sorted(pc)]))
            mm = [mean_m[(grp, arm, s)] for s in SEEDS]
            r = float(np.corrcoef(mm, fr)[0, 1])
            # 🔴 这里原来印的是写死的 `n=7, 临界 |r| = 0.878` —— **0.878 是 n=5 的**,
            #    n=7 的临界是 **0.7545**。同一个常数在一次 n 变更里被抄错,
            #    且方向是**变严**(0.878 > 0.754): 真有 r=0.80 的适配器会被误印成「未达显著」。
            #    与 `g1_cross_evalset_corr.py` 的 `R_CRIT_N5` 是同一族错误的两个实例
            #    (memory: hardcoded-conclusions-escape-reproduction)。改为现算。
            rc = r_crit(len(SEEDS))
            print(f"      {arm:<8} r(margin, fire) = {r:+.3f}   "
                  f"(n={len(SEEDS)}, 临界 |r| = {rc:.3f} 现算"
                  f" ⇒ {'显著' if abs(r) > rc else '**未达显著**'})")


if __name__ == "__main__":
    main()
