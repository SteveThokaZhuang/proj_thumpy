"""Q5: 自然流行率集上的 margin 探针 —— 闭环 §3.4, 并直接检验 §7 候选 (c)。

两个问题, 一份数据:

## A. 跨 0 率会不会在自然流行率集上回落 (§3.4 的预言)

§5.1 在 E2 (400 块, 100% 正例) 上测到 mixnorm 跨 0 率 60% vs own10 34%。
§3.4 的机制说: E2 上「多报只加 TP」, 于是阈值离散被原样放大成 F1 方差。
那么把**同一批 adapter** 拿到自然流行率集上, 机制预言:
    mixnorm 的跨 0 率应当**回落**, 而不是继续停在 60%。
这是一个跑之前就能证伪的预言 —— 若跨 0 率在自然集上依然是 60%,
那 §3.4 说的「放大通道」就只是 F1 那一步的事, 不是 margin 本身的事。

## B. 候选 (c): 归一化增益的不确定性 (§7 第 1 问)

`g1_mix_check.json` 给出每块的 gain = min(p95(own)/p95(part), 30)。
这份 gain 是重尾的: 中位 1.1, 但 22% 的块 >10, 5% 顶到 30 的上限,
另有约 20% 因为本人声道静音而 gain≈0。

**内对照**: gain≈0 时对方被压成静音, 该块的 mix == own ——
也就是**两臂看到的是完全相同的音频**。于是:
    低 gain 块上, mixnorm 与 own10 的种子间离散应当**没有差别**;
    高 gain 块上, mixnorm 应当明显更离散。
若这个梯度成立, 候选 (c) 就是被点名的那个机制; 若两臂的差距在
各 gain 箱里都一样, 那 gain 就不是原因。

## 与 §5 的口径

跨 0 率的定义逐字照搬 ari_g1_margin_analyze.py:
    straddle = ((M > 0).any(0)) & ((M < 0).any(0))     # M 是 (n_seed, n_chunk)
注意它对**恰好为 0** 的 margin 两边都不算, 而探针把 margin 四舍五入到 4 位,
所以极小的 margin 会变成 0.000。这里额外报一下 0 的个数。

用法: /share/home/zhuangruicen/miniconda3/envs/fd_analysis/bin/python \
        scripts/ari_g1_margin_natural.py
"""
import json
import os

import numpy as np

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
SEEDS = [42, 1234, 7, 2024, 3407, 31337, 55555]
ARMS = ["own10", "mixnorm"]

# (标签, 文件名前缀, 块数)
SETS = [("E2 事件密集集(400)", "g1_margin", 400),
        ("自然流行率集(300)", "g1_margin_nat", 300)]


def load_matrix(prefix, arm):
    """返回 (cids, M) —— M 是 (n_seed, n_chunk), 只保留 7 个种子都有的块。"""
    per_seed = {}
    for s in SEEDS:
        p = f"{ANNOT}/{prefix}_{arm}_s{s}.json"
        if not os.path.exists(p):
            return None, None
        per_seed[s] = json.load(open(p))
    if len(per_seed) < 3:
        return None, None
    cids = sorted(set.intersection(*(set(v) for v in per_seed.values())))
    M = np.array([[per_seed[s][c]["margin_max"] for c in cids] for s in SEEDS])
    return cids, M


def stats(M):
    med = np.mean([np.median(np.abs(row)) for row in M])
    means = M.mean(1)
    straddle = ((M > 0).any(0)) & ((M < 0).any(0))
    return {"med_abs": med, "sd_of_means": means.std(ddof=1),
            "straddle": straddle.mean(), "spread": M.std(0, ddof=1),
            "n_zero": int((M == 0).sum()), "means": means}


def main():
    Ms, cids_by = {}, {}
    print("=" * 88)
    print("  A. 跨 0 率: E2 vs 自然流行率集")
    print("=" * 88)
    print(f"  {'arm':<9}{'评估集':<22}{'块数':>6}{'跨0率':>9}"
          f"{'种子内|m|中位':>15}{'种子间平均margin的SD':>22}")
    for arm in ARMS:
        for label, prefix, _ in SETS:
            cids, M = load_matrix(prefix, arm)
            if M is None:
                print(f"  {arm:<9}{label:<22}  (缺文件)")
                continue
            st = stats(M)
            Ms[(arm, prefix)] = M
            cids_by[(arm, prefix)] = cids
            print(f"  {arm:<9}{label:<22}{M.shape[1]:>6}"
                  f"{st['straddle']*100:>8.0f}%{st['med_abs']:>15.3f}"
                  f"{st['sd_of_means']:>22.3f}")

    print("\n  ── 逐种子平均 margin (自然流行率集) ──")
    for arm in ARMS:
        cids, M = load_matrix("g1_margin_nat", arm)
        if M is not None:
            print(f"    {arm:<9}" + "  ".join(f"{m:+.3f}" for m in M.mean(1)))

    # ---------------------------------------------------------------- B
    print("\n" + "=" * 88)
    print("  B. §3.4 的预言: 自然流行率集上 mixnorm 的跨 0 率应当回落")
    print("=" * 88)
    for arm in ARMS:
        a = stats(Ms[(arm, "g1_margin")])["straddle"]
        b = stats(Ms[(arm, "g1_margin_nat")])["straddle"]
        arrow = "↓ 回落" if b < a else "↑ 未回落"
        print(f"  {arm:<9} E2 {a*100:.0f}%  ->  自然集 {b*100:.0f}%   {arrow}"
              f"   (差 {(b-a)*100:+.0f} 个百分点)")
    a_m = stats(Ms[("mixnorm", "g1_margin")])["straddle"]
    a_o = stats(Ms[("own10", "g1_margin")])["straddle"]
    b_m = stats(Ms[("mixnorm", "g1_margin_nat")])["straddle"]
    b_o = stats(Ms[("own10", "g1_margin_nat")])["straddle"]
    print(f"\n  E2 上 mixnorm − own10 的跨 0 率差 = {(a_m-a_o)*100:+.0f} pt "
          f"({a_m*100:.0f}% vs {a_o*100:.0f}%)")
    print(f"  自然集上 mixnorm − own10 的跨 0 率差 = {(b_m-b_o)*100:+.0f} pt "
          f"({b_m*100:.0f}% vs {b_o*100:.0f}%)")

    print(f"\n  注: 自然集上 margin 恰好为 0.000 的 (块,种子) 对 = "
          f"{stats(Ms[('mixnorm','g1_margin_nat')])['n_zero']}")

    # ---------------------------------------------------------------- C
    print("\n" + "=" * 88)
    print("  C. 候选 (c): gain 与种子间离散 (仅自然集, 同块号 join)")
    print("=" * 88)
    gain = {r["id"]: r["gain"] for r in
            json.load(open(f"{ANNOT}/g1_mix_check.json"))["rows"]}

    St = {arm: stats(Ms[(arm, "g1_margin_nat")]) for arm in ARMS}
    cids_m, cids_o = cids_by[("mixnorm", "g1_margin_nat")], cids_by[("own10", "g1_margin_nat")]
    if set(cids_m) != set(cids_o):
        print("  ⚠️ 两臂块号不一致, 取交集")
    cids = sorted(set(cids_m) & set(cids_o) & set(gain))
    print(f"  可 join 的块: {len(cids)}")

    gm = np.array([gain[c] for c in cids])
    idx_m = {c: i for i, c in enumerate(cids_m)}
    idx_o = {c: i for i, c in enumerate(cids_o)}
    sp_m = np.array([St["mixnorm"]["spread"][idx_m[c]] for c in cids])
    sp_o = np.array([St["own10"]["spread"][idx_o[c]] for c in cids])
    st_m = ((Ms[("mixnorm", "g1_margin_nat")] > 0).any(0) &
            (Ms[("mixnorm", "g1_margin_nat")] < 0).any(0))[[idx_m[c] for c in cids]]
    st_o = ((Ms[("own10", "g1_margin_nat")] > 0).any(0) &
            (Ms[("own10", "g1_margin_nat")] < 0).any(0))[[idx_o[c] for c in cids]]

    BINS = [(-0.01, 0.01, "≈0 对方被静音"),
            (0.01, 1.0, "0–1 本人更响"),
            (1.0, 3.0, "1–3 相当"),
            (3.0, 10.0, "3–10 对方更响"),
            (10.0, 29.99, "10–30 对方压倒"),
            (29.99, 100.0, "=30 顶到上限")]
    print(f"\n  {'gain 箱':<16}{'块数':>6}{'mixnorm跨0率':>14}{'own10跨0率':>13}"
          f"{'mixnorm SD':>13}{'own10 SD':>11}{'SD 差':>9}")
    for lo, hi, name in BINS:
        m = (gm > lo) & (gm <= hi)
        if m.sum() == 0:
            continue
        print(f"  {name:<16}{int(m.sum()):>6}{st_m[m].mean()*100:>13.0f}%"
              f"{st_o[m].mean()*100:>12.0f}%{sp_m[m].mean():>13.3f}"
              f"{sp_o[m].mean():>11.3f}{sp_m[m].mean()-sp_o[m].mean():>+9.3f}")

    print(f"\n  全体相关 (逐块, n={len(cids)}):")
    print(f"    corr(gain, mixnorm 种子间 SD) = {np.corrcoef(gm, sp_m)[0,1]:+.3f}")
    print(f"    corr(gain, own10   种子间 SD) = {np.corrcoef(gm, sp_o)[0,1]:+.3f}")
    print(f"    corr(gain, mixnorm 跨0)       = {np.corrcoef(gm, st_m.astype(float))[0,1]:+.3f}")

    # ---------------------------------------------------------------- D
    print("\n" + "=" * 88)
    print("  D. 内对照: gain≈0 的块上两臂的音频完全相同")
    print("=" * 88)
    dead = gm <= 0.01
    live = gm >= 3.0
    for name, m in (("gain≈0 (mix == own)", dead), ("gain≥3 (对方明显介入)", live)):
        if m.sum() == 0:
            continue
        print(f"  {name:<28} 块数 {int(m.sum()):>4}   "
              f"mixnorm SD {sp_m[m].mean():.3f}   own10 SD {sp_o[m].mean():.3f}   "
              f"差 {sp_m[m].mean()-sp_o[m].mean():+.3f}")
    if dead.sum() and live.sum():
        d_dead = sp_m[dead].mean() - sp_o[dead].mean()
        d_live = sp_m[live].mean() - sp_o[live].mean()
        print(f"\n  预言: 低 gain 块差值 ≈ 0, 高 gain 块差值 > 0")
        print(f"  实测: 低 {d_dead:+.3f}   高 {d_live:+.3f}   "
              f"-> {'符合预言' if (d_live > d_dead) else '**不符合预言**'}")


if __name__ == "__main__":
    main()
