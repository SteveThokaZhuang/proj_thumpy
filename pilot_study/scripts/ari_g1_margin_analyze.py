"""#56 P1 分析: 决策边界间隔能否解释 mixnorm 的不稳.

输入:
  g1_margin_<arm>_s<seed>.json   —— 探针输出 (同样的 400 块, margin_max 为主口径)
  g1_eval_<arm>[_s<seed>]_e2.json —— 已有的 1600 块评估, 取 per_chunk 的 n_pred
                                     (= 该 adapter 在该块上实际报了几个事件)

要回答的三件事:
  Q1 **间隔真的决定行为吗** —— margin_max 与"实际是否报"的一致率 / AUC。
     这是探针本身的有效性检验; 若 AUC 不高, 后面所有解释都不成立。
  Q2 **mixnorm 的决策是否更靠边界** —— 两臂的 margin 分布对比 (近 0 质量、中位数)。
  Q3 **间隔能否解释种子间差异** —— 每个种子的平均 margin vs 它的 F1 / n_pred。

用法: python scripts/ari_g1_margin_analyze.py
"""
import json
import os
import sys

import numpy as np

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
SEEDS = [42, 1234, 7, 2024, 3407, 31337, 55555]
ARMS = ["own10", "mixnorm"]


def suffix(seed):
    return "" if seed == 42 else f"_s{seed}"


def load_margin(arm, seed):
    p = f"{ANNOT}/g1_margin_{arm}_s{seed}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def load_fired(arm, seed, cids):
    """该 adapter 在 E2 上对这些块实际报没报 (n_pred>0)。"""
    p = f"{ANNOT}/g1_eval_{arm}{suffix(seed)}_e2.json"
    r = next(iter(json.load(open(p)).values()))["per_chunk"]
    return np.array([1 if r[c]["n_pred"] > 0 else 0 for c in cids], float)


def load_f1(arm, seed):
    p = f"{ANNOT}/g1_eval_{arm}{suffix(seed)}_e2.json"
    r = next(iter(json.load(open(p)).values()))
    return r["f1"], r["n_pred"]


def auc(score, label):
    """label=1 的 score 大于 label=0 的概率 (= ROC AUC), 用秩和算。"""
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    print("=" * 80)
    print("  Q1: margin_max 是否真的决定「报/不报」")
    print("=" * 80)
    print(f"  {'arm':<8}{'seed':>7}{'块数':>7}{'报的比例':>9}{'一致率':>9}{'AUC':>8}")
    perf = {}
    pooled = {}
    for arm in ARMS:
        for s in SEEDS:
            m = load_margin(arm, s)
            if m is None:
                continue
            cids = sorted(m)
            mgn = np.array([m[c]["margin_max"] for c in cids])
            fire = load_fired(arm, s, cids)
            agree = ((mgn > 0) == (fire == 1)).mean()
            a = auc(mgn, fire)
            f1, npred = load_f1(arm, s)
            perf[(arm, s)] = {"mean_margin": mgn.mean(), "med": np.median(mgn),
                              "agree": agree, "auc": a, "fire_rate": fire.mean(),
                              "f1": f1, "n_pred": npred, "n": len(cids)}
            pooled.setdefault(arm, []).append((mgn, fire))
            print(f"  {arm:<8}{s:>7}{len(cids):>7}{fire.mean():>9.2f}"
                  f"{agree:>9.3f}{a:>8.3f}")

    print("\n  ── 汇总 ──")
    for arm in ARMS:
        if arm not in pooled:
            continue
        mg = np.concatenate([p[0] for p in pooled[arm]])
        fr = np.concatenate([p[1] for p in pooled[arm]])
        print(f"    {arm:<8} 一致率 {((mg>0)==(fr==1)).mean():.3f}   "
              f"AUC {auc(mg, fr):.3f}   n={len(mg)}")

    print("\n" + "=" * 80)
    print("  Q2: 两臂的 margin 分布 —— mixnorm 是否更靠边界")
    print("=" * 80)
    print(f"  {'arm':<8}{'seed':>7}{'均值':>9}{'中位':>9}{'|m|<1':>8}{'|m|<0.5':>9}{'|m|<2':>8}")
    for arm in ARMS:
        for s in SEEDS:
            m = load_margin(arm, s)
            if m is None:
                continue
            v = np.array([x["margin_max"] for x in m.values()])
            print(f"  {arm:<8}{s:>7}{v.mean():>9.3f}{np.median(v):>9.3f}"
                  f"{np.mean(np.abs(v)<1)*100:>7.0f}%{np.mean(np.abs(v)<0.5)*100:>8.0f}%"
                  f"{np.mean(np.abs(v)<2)*100:>7.0f}%")

    print("\n" + "=" * 80)
    print("  Q3: 种子间差异 —— 平均间隔 vs 实际表现")
    print("=" * 80)
    for arm in ARMS:
        rows = [(s, perf[(arm, s)]) for s in SEEDS if (arm, s) in perf]
        if len(rows) < 3:
            continue
        mm = np.array([r[1]["mean_margin"] for r in rows])
        f1 = np.array([r[1]["f1"] for r in rows])
        fr = np.array([r[1]["fire_rate"] for r in rows])
        print(f"    {arm:<8} corr(平均margin, F1) = {np.corrcoef(mm,f1)[0,1]:+.3f}   "
              f"corr(平均margin, 报的比例) = {np.corrcoef(mm,fr)[0,1]:+.3f}")

    print("\n" + "=" * 80)
    print("  Q4: 逐块的跨种子间隔 —— 摇摆是不是「间隔跨过 0」")
    print("=" * 80)
    # 对每一块, 看 7 个种子各自的 margin_max。机制预言:
    #   7 个都同号且远离 0 -> fire-rate 0 或 7 (果断)
    #   margin 跨 0          -> fire-rate 居中 (抛硬币)
    # 于是「mixnorm 摇摆块更多」应该等价于「mixnorm 更多块的 margin 跨 0」。
    for arm in ARMS:
        Ms, Fs = [], []
        for s in SEEDS:
            m = load_margin(arm, s)
            if m is None:
                continue
            cids = sorted(m)
            Ms.append(np.array([m[c]["margin_max"] for c in cids]))
            Fs.append(load_fired(arm, s, cids))
        if len(Ms) < 3:
            continue
        M = np.array(Ms)            # (n_seed, n_chunk)
        F = np.array(Fs)            # (n_seed, n_chunk)
        fire_rate = F.mean(0)
        # 每块: 种子间 margin 的均值 / 跨零程度
        frac_pos = (M > 0).mean(0)          # 种子中 margin>0 的比例
        spread = M.std(0)
        # 1) 预测的报的比例 vs 实际 fire-rate
        r_pred = np.corrcoef(frac_pos, fire_rate)[0, 1]
        # 2) 果断 vs 摇摆块的 |平均margin|
        decisive = (fire_rate == 0) | (fire_rate == 1)
        swing = ~decisive
        print(f"  {arm:<8} 块数 {M.shape[1]}  "
              f"果断 {decisive.sum()} ({decisive.mean()*100:.0f}%)  "
              f"摇摆 {swing.sum()} ({swing.mean()*100:.0f}%)")
        print(f"           corr(种子中 margin>0 的比例, 实际 fire-rate) = {r_pred:+.3f}")
        print(f"           果断块 |平均 margin| = {np.abs(M.mean(0))[decisive].mean():.3f}"
              f"   摇摆块 |平均 margin| = {np.abs(M.mean(0))[swing].mean():.3f}")
        # 3) margin 跨 0 的块占比 (块内既有 margin>0 也有 <0 的种子)
        straddle = ((M > 0).any(0)) & ((M < 0).any(0))
        print(f"           margin 跨 0 的块: {straddle.sum()} "
              f"({straddle.mean()*100:.0f}%)")

    print("\n" + "=" * 80)
    print("  Q5: 关键判别 —— 「每个种子内部更靠边界」 还是 「种子之间边界位置差很多」")
    print("=" * 80)
    # 原假设是「mixnorm 把决策整体压到阈值附近」= 种子**内部**更靠边。
    # 但其实可能是另一种机制: 每个种子内部一样果断, 只是**各种子把边界画在了
    # 不同的地方**, 于是同样一批块在不同种子间翻来覆去。
    # 前者看 |margin| 的分布, 后者看「各种子平均 margin」的离散度。两者要分开报。
    print(f"  {'arm':<8}{'种子内|m|中位(平均)':>20}{'各种子平均margin的SD':>22}{'跨0块占比':>12}")
    for arm in ARMS:
        meds, means, st = [], [], None
        for s in SEEDS:
            v = np.array([x["margin_max"] for x in load_margin(arm, s).values()])
            meds.append(np.median(np.abs(v)))
            means.append(v.mean())
        # 跨 0 块占比
        Ms = [np.array([x["margin_max"] for x in load_margin(arm, s).values()])
              for s in SEEDS]
        M = np.array(Ms)
        st = (((M > 0).any(0)) & ((M < 0).any(0))).mean()
        print(f"  {arm:<8}{np.mean(meds):>20.3f}{np.std(means, ddof=1):>22.3f}"
              f"{st*100:>11.0f}%")
        print(f"           各种子平均 margin: "
              + "  ".join(f"{m:+.3f}" for m in means))

    print("\n" + "=" * 80)
    print("  Q6: 判别力 vs 操作点 —— 这是真正非平凡的一问")
    print("=" * 80)
    # Q1 的 AUC 拿「模型自己的决定」当标签, 那是恒等式, 不算证据。
    # 这里换成**真值**: 该块内是否存在完整落在块内的 BC 窗口 (与 E2 口径一致,
    # 用 require_inside=True 排掉横跨边界的不可答窗口)。
    #   - 若各种子 AUC(margin, 真值) 差不多、而平均 margin 差很多
    #     -> 同一根 ROC 曲线上换了操作点 (阈值问题)
    #   - 若 AUC 本身就在抖
    #     -> 表征的判别力不稳 (另一回事)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ari_g1_common import bc_windows, load_manifest  # noqa: E402

    npzdir = {"own10": f"{ANNOT}/g1_e2/own", "mixnorm": f"{ANNOT}/g1_e2/mixnorm"}
    print(f"  {'arm':<8}{'seed':>7}{'AUC(margin,真值)':>18}{'平均margin':>12}{'报的比例':>10}")
    for arm in ARMS:
        ml = load_manifest(npzdir[arm])
        y_cache = {}
        for s in SEEDS:
            m = load_margin(arm, s)
            if m is None:
                continue
            cids = sorted(m)
            if not y_cache:
                y_cache.update({c: int(bool(bc_windows(c, ml, require_inside=True)))
                                for c in cids})
            pos = np.mean(list(y_cache.values()))
            y = np.array([y_cache[c] for c in cids], float)
            v = np.array([m[c]["margin_max"] for c in cids])
            a = auc(v, y)
            print(f"  {arm:<8}{s:>7}{a:>18.3f}{v.mean():>12.3f}"
                  f"{(v > 0).mean():>10.2f}")
        if y_cache:
            print(f"    -> {arm:<8} 真值正例率 = {pos:.3f}")
            if pos in (0.0, 1.0):
                print(f"       ⚠️ E2 的块**几乎全是正例** —— 没有负例, 块级 AUC 无从定义。")
                print(f"          这不是代码错, 是评估集本身的性质: E2 是「事件密集」集")
                print(f"          (1640 事件 / 1600 块), 每块都有 BC。于是块级判别力"
                      f"在本集上不可测。")

    print("\n" + "=" * 80)
    print("  Q7: 事件密集集上的 F1 是不是「报得越多分越高」")
    print("=" * 80)
    # 如果 E2 几乎没有负例, 那么多报只增 TP 不增 FP -> F1 应该随报的比例单调上升。
    # 若真如此, 那么 mixnorm 的「不稳」有很大一部分是**指标在这套评估集上的性质**,
    # 而不是模型性质 —— 换回自然流行率的评估集未必复现。
    for arm in ARMS:
        rows = [(s, perf[(arm, s)]) for s in SEEDS if (arm, s) in perf]
        if len(rows) < 3:
            continue
        fr = np.array([r[1]["fire_rate"] for r in rows])
        f1 = np.array([r[1]["f1"] for r in rows])
        rc = np.array([r[1]["n_pred"] for r in rows], float)
        print(f"  {arm:<8} corr(报的比例, F1) = {np.corrcoef(fr, f1)[0,1]:+.3f}   "
              f"corr(n_pred, F1) = {np.corrcoef(rc, f1)[0,1]:+.3f}")
        for s, p in sorted(rows, key=lambda r: r[1]["fire_rate"]):
            print(f"            seed {s:<6} 报 {p['fire_rate']:.2f}  "
                  f"n_pred {int(p['n_pred']):<5} F1 {p['f1']:.4f}")

    json.dump({f"{k[0]}_{k[1]}": v for k, v in perf.items()},
              open(f"{ANNOT}/g1_margin_analyze.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"\n-> {ANNOT}/g1_margin_analyze.json")


if __name__ == "__main__":
    main()
