"""「mixnorm 的种子间决策边界离散更大」—— 那是**初值**还是**数据顺序**?

## 这一问从哪来 (主旨)

本项目绕到最后, 只剩一条还没被推翻的观察:

    mixnorm 各 run 把决策边界画在了**不同的地方** —— 种子间「平均 margin」的 SD
    比 own10 大 2.88 倍 (ari_g1_margin_analyze.py 的 Q5)。

Q3 (§7.4d) 刚证明 **ΔF1 的种子间方差由数据顺序主导, 不是初值**。
那么同一条逻辑必须问下去: **边界离散是不是也只是数据顺序?**

这个问题**决定那条观察还能不能留**:
  · 若边界离散同样由 shuffle 主导 ⇒ 它不是 mixnorm 的性质, 是「这批 run 恰好
    抽到了不同的数据顺序」, 与 ΔF1 同归于尽。
  · 若边界离散**主要来自初值** ⇒ 它是**唯一**一个 shuffle 解释不了的效应,
    整篇的收尾就得围绕它写。

## 为什么本脚本不需要 GPU

`ari_g1_margin_analyze.py` 的 Q5 用的是 **margin 探针** (需要 GPU 前向), 而探针
产物只按 `_s{seed}` 命名 —— **Q3 的 B 组 (定 seed=42、变 data_seed) 根本没有探针**。

但边界位置有一个**零成本的可观测量**: **报/不报的比例 (fire rate)**。
它已经躺在 28 个 `_filt` 评估里。margin 分布若形状大致不变、只是整体平移,
则 fire rate 是那个平移的单调函数 ⇒ **fire rate 的跨适配器 SD 就是边界离散的代理**。

本脚本先跑这个代理。**它决定要不要花 2.5h GPU 去跑真探针**:
  · 代理显示 A ≈ B ⇒ 大概率 shuffle 主导, 探针不值得跑 (但可作确认);
  · 代理显示 B ≪ A ⇒ 探针值得跑, 因为那意味着初值真的在动边界。

## 🔴 代理的失效模式 (先写下来, 免得看到数字才找说法)

fire rate 有**地板/天花板效应**: 若两臂的 fire rate 都挤在 0 或 1 附近,
SD 会被边界压扁, 代理失灵。所以本脚本**必须先报 fire rate 的取值范围**,
再报 SD —— 顺序不能反 (memory: prevalence-flips-variance-conclusions)。

## 显著性用**置换**, 不用 F

n=7 时正态理论 F 检验的假设站不住。显著性一律走
`var_ratio_perm`（14 个值随机分 7/7 看方差比）—— **与 §7.1 `ari_g1_margin_sd_test.py`
定的那个 p=0.0160 是同一个统计量**, 这样两条口径才能直接比。F 只作参照打印。

## 口径

A 组: seed ∈ {42,1234,2024,31337,3407,55555,7}, data_seed ≡ 42   (`g1_eval_{arm}[_s{s}]_filt`)
B 组: seed ≡ 42,                     data_seed ∈ 同一组值    (`g1_eval_{arm}_ds{ds}_filt`)

**共同块**: 28 个文件取交集, 全部统计量都在交集上算 —— 不混集。

用法: python scripts/g1_boundary_dispersion.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
SEEDS = [42, 1234, 2024, 31337, 3407, 55555, 7]
ARMS = ["own10", "mixnorm"]

# ari_g1_margin_analyze.py Q5 报的那个数 (SD 比 2.88×), 用来对齐两条口径
Q5_SD_RATIO = 2.88


def a_tag(arm, s):
    """A 组: 变 seed, data_seed 固定 42。seed=42 时无后缀。"""
    return f"{arm}{'' if s == 42 else f'_s{s}'}"


def b_tag(arm, ds):
    """B 组: seed 固定 42, 变 data_seed。"""
    return f"{arm}_ds{ds}"


def load_eval(tag):
    p = f"{ANNOT}/g1_eval_{tag}_filt.json"
    if not os.path.exists(p):
        return None
    r = next(iter(json.load(open(p)).values()))
    return r["per_chunk"]


def common_chunks(tags):
    """所有 tag 的 per_chunk 键交集 —— 少一个文件就整体作废, 不静默取子集。"""
    pcs = {}
    for t in tags:
        pc = load_eval(t)
        if pc is None:
            print(f"    ❌ 缺文件: {t}")
            return None
        pcs[t] = pc
    return pcs, sorted(set.intersection(*(set(pc) for pc in pcs.values())))


def stats(pcs, cids):
    """{tag: (fire_rate, mean_n_pred, n_swing)} —— 全部限定在 cids 上。"""
    out = {}
    for t, pc in pcs.items():
        pred = np.array([pc[c]["n_pred"] for c in cids], float)
        out[t] = {"fire": float((pred > 0).mean()),
                  "mean_npred": float(pred.mean()),
                  "n_pred": int(pred.sum())}
    return out


def sd_ci(v):
    """SD 的 95% 卡方区间 (n<10 时它自己的不确定度就有 30%+)。"""
    from scipy import stats as st
    n = len(v)
    s = np.std(v, ddof=1)
    lo = s * np.sqrt((n - 1) / st.chi2.ppf(0.975, n - 1))
    hi = s * np.sqrt((n - 1) / st.chi2.ppf(0.025, n - 1))
    return s, lo, hi


def f_test(va, vb):
    """F = Var(A)/Var(B), 双侧 p (正态理论, 仅作参照)。"""
    from scipy import stats as st
    sa, sb = np.var(va, ddof=1), np.var(vb, ddof=1)
    n = len(va)
    F = sa / sb
    p = 2 * min(st.f.cdf(F, n - 1, n - 1), 1 - st.f.cdf(F, n - 1, n - 1))
    return F, min(p, 1.0), sa, sb


def var_ratio_perm(a, b, n_iter=200000, seed=0):
    """精确置换: 把 a+b 的 14 个值随机分 7/7, 看方差比 b/a 多大。

    **与 §7.1 的 `ari_g1_margin_sd_test.py` 同一个统计量**。n=7 时正态理论
    F 检验的假设站不住, 置换不假设任何分布, 所以**以置换 p 为准, F 只作参照**。

    ⚠️ **两个实现会给出末位不同的 p, 那不是矛盾**: 这里固定 `seed=0`,
    那个脚本另有自己的流。同一份数据实测 **0.0161 (本函数) vs 0.0160 (旧脚本)** ——
    20 万次置换下 MC 标准误约 ±0.0003, 差 0.0001 完全在噪声内。
    **报数时以本函数(可复跑、固定 seed)的输出为准**, 别把两个值当两个结果。
    """
    rng = np.random.default_rng(seed)
    pool = np.concatenate([a, b])
    n = len(a)
    obs = np.var(b, ddof=1) / np.var(a, ddof=1)
    idx = np.argsort(rng.random((n_iter, len(pool))), axis=1)
    lo = pool[idx[:, :n]]
    hi = pool[idx[:, n:]]
    f = (np.var(hi, axis=1, ddof=1)
         / np.maximum(np.var(lo, axis=1, ddof=1), 1e-12))
    return obs, (int((f >= obs).sum()) + 1) / (n_iter + 1)


def r_crit(n, alpha=0.05):
    """Pearson 双侧 p<alpha 的临界 |r| (df = n-2)。

    ⚠️ **这个函数存在的唯一理由是那个数会随 n 变**, 而它被复制错过两次:
    **n=5 → 0.8783, n=7 → 0.7545**。谁把 0.878 抄到 n=7 的位置上, 判据就悄悄变严
    (0.878 > 0.755), 一个 r=0.80 的真结果会被印成「未达显著」。
    所以**它是判据, 必须现算, 不能写成常数** —— 与 `var_ratio_perm` 同理。
    (memory: hardcoded-conclusions-escape-reproduction / threshold-must-match-null-model)
    """
    from scipy import stats as st
    df = n - 2
    if df < 1:
        return float("nan")
    t = st.t.ppf(1 - alpha / 2, df)
    return float(t / np.sqrt(t * t + df))


def main():
    print("=" * 78)
    print("  边界离散: 初值还是数据顺序? (fire rate 代理, 0 GPU)")
    print("=" * 78)

    tags_a = [a_tag(a, s) for a in ARMS for s in SEEDS]
    tags_b = [b_tag(a, d) for a in ARMS for d in SEEDS]
    got = common_chunks(tags_a + tags_b)
    if got is None:
        return
    pcs, cids = got
    n_gt = sum(pcs[tags_a[0]][c]["n_gt"] for c in cids)
    n_pos = sum(1 for c in cids if pcs[tags_a[0]][c]["n_gt"] > 0)
    print(f"\n  共同块 {len(cids)}   GT 事件 {n_gt}   正例块 {n_pos} "
          f"({n_pos/len(cids):.1%})")

    S = stats(pcs, cids)

    # ---------- 🔴 先报范围, 再报 SD (地板/天花板效应会让代理失灵) ----------
    allf = np.array([S[t]["fire"] for t in tags_a + tags_b])
    print(f"\n  🔴 代理有效性前置检查 (顺序不能反)")
    print(f"    fire rate 范围 [{allf.min():.3f}, {allf.max():.3f}]   "
          f"全样本均值 {allf.mean():.3f}")
    span_ok = allf.min() > 0.02 and allf.max() < 0.98
    print(f"    {'✅ 没有贴地板/天花板, 代理可用' if span_ok else '⚠️ 贴边 —— SD 会被压扁, 代理不可信'}")

    for arm in ARMS:
        print(f"\n  ── {arm} ──")
        print(f"    {'seed(A组)':>10}{'fire':>9}{'mean_npred':>12}"
              f"   |{'ds(B组)':>10}{'fire':>9}{'mean_npred':>12}")
        fa, fb = [], []
        for s in SEEDS:
            A, B = S[a_tag(arm, s)], S[b_tag(arm, s)]
            fa.append(A["fire"])
            fb.append(B["fire"])
            print(f"    {s:>10}{A['fire']:>9.3f}{A['mean_npred']:>12.3f}"
                  f"   |{s:>10}{B['fire']:>9.3f}{B['mean_npred']:>12.3f}")
        fa, fb = np.array(fa), np.array(fb)
        sdA, loA, hiA = sd_ci(fa)
        sdB, loB, hiB = sd_ci(fb)
        F, p_norm, va, vb = f_test(fa, fb)
        _, p = var_ratio_perm(fa, fb)
        print(f"    {'SD':>10}{sdA:>9.4f}{'':>12}   |{'SD':>10}{sdB:>9.4f}")
        print(f"    SD 95%CI  [{loA:.4f}, {hiA:.4f}]            "
              f"[{loB:.4f}, {hiB:.4f}]")
        print(f"    **SD 比 A/B = {sdA/sdB:.2f}×**   置换 p = {p:.4f}"
              f"   (正态 F = {F:.3f} 双侧 p = {p_norm:.3f}, 仅参照)")
        print(f"    {'⚠️ A、B 两种随机源的离散**不同**' if p < 0.05 else '❌ A/B 差判不住 ⇒ 与「两种随机源贡献相当」相容'}")

        # 留一: n=7 时一个适配器就能撑起整个 SD 比 (memory: 第六种形态)
        ratios = []
        for i in range(len(SEEDS)):
            xa = np.delete(fa, i)
            xb = np.delete(fb, i)
            ratios.append(np.std(xa, ddof=1) / np.std(xb, ddof=1))
        print(f"    留一 SD 比: " + "  ".join(f"{r:.2f}" for r in ratios))
        print(f"      范围 [{min(ratios):.2f}, {max(ratios):.2f}]"
              f"  ⇒ {'⚠️ 摆动大, 比值不稳' if max(ratios)/max(min(ratios),1e-9) > 1.6 else '尚算稳定'}")
        if sdB < 1e-9:
            print(f"    ⚠️ B 组 SD ≈ 0, 比值无意义")

    # ---------- 跨臂: 两臂的边界离散谁大 (Q5 的原始问法) ----------
    print(f"\n  ── Q5 原问法: 两臂的 A 组 SD 谁大 (探针口径给的是 {Q5_SD_RATIO}×) ──")
    for grp, tags in (("A 变seed", tags_a), ("B 变data_seed", tags_b)):
        fo = np.array([S[a_tag("own10", s)]["fire"] for s in SEEDS]) \
            if grp.startswith("A") else \
            np.array([S[b_tag("own10", d)]["fire"] for d in SEEDS])
        fm = np.array([S[a_tag("mixnorm", s)]["fire"] for s in SEEDS]) \
            if grp.startswith("A") else \
            np.array([S[b_tag("mixnorm", d)]["fire"] for d in SEEDS])
        sdo, sdm = np.std(fo, ddof=1), np.std(fm, ddof=1)
        F, p_norm, _, _ = f_test(fm, fo)
        _, p = var_ratio_perm(fo, fm)
        print(f"    {grp:<14} own10 SD {sdo:.4f}   mixnorm SD {sdm:.4f}   "
              f"比 {sdm/sdo:.2f}×   置换 p = {p:.4f}   "
              f"(正态 F={F:.3f} p={p_norm:.3f})")

    # ---------- 合并两个跨臂检验 (Fisher) ----------
    # A 组用 `_s{seed}` 适配器、B 组用 `_ds{ds}` 适配器 —— **两套不重叠**
    # (只有 seed=42/ds=42 那一个有效配置两者都有), 所以近似独立, 可以合并。
    print(f"\n  ── 合并 A 组与 B 组的跨臂证据 (Fisher) ──")
    ps = []
    for grp in ("A", "B"):
        fo = np.array([S[(a_tag if grp == "A" else b_tag)("own10", k)]["fire"]
                       for k in SEEDS])
        fm = np.array([S[(a_tag if grp == "A" else b_tag)("mixnorm", k)]["fire"]
                       for k in SEEDS])
        _, pk = var_ratio_perm(fo, fm)
        ps.append(pk)
        print(f"    {grp} 组 比值 {np.std(fm,ddof=1)/np.std(fo,ddof=1):.2f}×   p = {pk:.4f}")
    from scipy import stats as st
    chi = -2 * np.sum(np.log(ps))
    p_fish = float(st.chi2.sf(chi, 2 * len(ps)))
    print(f"    Fisher 合并: χ² = {chi:.3f} (df={2*len(ps)})  ⇒  **p = {p_fish:.4f}**")
    print(f"    ⚠️ 两套适配器只共享 1 个有效配置 (seed42/ds42), 所以近似独立、")
    print(f"       合并 p 略偏乐观 —— 报的时候说「~0.02」, 不要说小数点后三位。")


    # fire rate 是在 300 块上估的, 每块自己的 p_c 不同 ⇒ 每个适配器的 fire rate
    # 都带一份抽样噪声。它**同时**加进 A 和 B 的方差里, 于是把观测到的 A/B 比
    # **压向 1** —— 所以「观测比值 ≈ 1」意味着真实比值也 ≈ 1, 但真实比值与 1 的
    # 偏离会被低估。这里把它估出来, 免得把「压向 1」当成「就是 1」。
    print(f"\n  ── 分块抽样噪声 (共同项, 会把 A/B 比压向 1) ──")
    print(f"    {'arm':<8}{'组':<6}{'观测方差':>12}{'抽样方差':>12}"
          f"{'扣掉后的 SD':>14}{'扣掉后的 B/A':>14}")
    ded = {}
    for arm in ARMS:
        X = np.array([[1.0 if pcs[t][c]["n_pred"] > 0 else 0.0 for c in cids]
                      for t in tags_a + tags_b if t.startswith(arm)])
        pc = X.mean(0)
        var_samp = float((pc * (1 - pc)).sum() / len(cids) ** 2)
        for grp, f in (("A", [S[a_tag(arm, s)]["fire"] for s in SEEDS]),
                       ("B", [S[b_tag(arm, d)]["fire"] for d in SEEDS])):
            vo = float(np.var(f, ddof=1))
            vd = max(vo - var_samp, 0.0)
            ded[(arm, grp)] = vd
            print(f"    {arm:<8}{grp:<6}{vo:>12.6f}{var_samp:>12.6f}"
                  f"{vd**0.5:>14.4f}"
                  + (f"{vd/ded[(arm,'A')]:>13.2f}×" if grp == "B" else ""))
        if var_samp > 0.5 * min(ded[(arm, "A")], ded[(arm, "B")]):
            print(f"      ⚠️ 抽样项吃掉了大半方差 —— 扣完剩下的东西不可信, 只看方向")

    print(f"\n  ── 判读 ──")
    print(f"    若「A/B 比」在两臂上都接近 1 ⇒ 边界位置由**数据顺序**定,")
    print(f"       与 ΔF1 同源; 那条「mixnorm 边界更散」的观察应**撤回**。")
    print(f"    若 own10 的 A/B 比 ≈ 1 而 mixnorm 的 ≫ 1 ⇒ **交互**: 只有 mixnorm")
    print(f"       的边界对初值敏感 —— 那才是本项目的收尾结论。")


if __name__ == "__main__":
    main()
