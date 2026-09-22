"""#56/#57 收尾: 评估集的「事件流行率」是不是 mixnorm 不稳的一部分原因?

## 为什么问这个

P1 的 Q7 发现: 在 E2 上, mixnorm 的 F1 与「报的比例」几乎单调挂钩
(corr = +0.925, n_pred 也是 +0.935), 而 own10 只有 +0.327。
Q5 又发现 mixnorm 的各种子**边界画在不同地方** (各种子平均 margin 的 SD
0.525 vs own10 0.183), 而种子**内部**的果断程度两臂差不多 (0.911 vs 0.777)。

合起来是一个很具体的机制猜想:

    mixnorm 的种子各自落在不同的「开火阈值」上
      + E2 是**事件密集集**(1640 事件/1600 块, 几乎每块都有真值, 没有负例)
      -> 多报几乎只增 TP 不增 FP, F1 随开火率单调上升
      -> 阈值方差被 F1 原样放大成种子间方差

若这个猜想对, 那么在**自然流行率**的评估集上:
    (a) corr(报的比例, F1) 应该显著变弱 (多报会被 FP 惩罚)
    (b) mixnorm 的 F1 种子间 SD 应该缩小, F 比也应该缩小
反之若两套集上结论一样, 那这个猜想就是错的, 不稳是模型本身的性质。

**这是纯读盘分析, 不需要 GPU** —— 14 个种子的两套评估 json 都已经在了。

用法: python scripts/ari_g1_prevalence_check.py
"""
import json
import os

import numpy as np

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
SEEDS = [42, 1234, 7, 2024, 3407, 31337, 55555]
ARMS = ["own10", "mixnorm"]


def suffix(seed):
    return "" if seed == 42 else f"_s{seed}"


def load(arm, seed, kind):
    """kind: '' = 旧 300 块集 (原始), '_filt' = 同集过滤版, '_e2' = E2。"""
    p = f"{ANNOT}/g1_eval_{arm}{suffix(seed)}{kind}.json"
    if not os.path.exists(p):
        return None
    return next(iter(json.load(open(p)).values()))


def summarise(label, kind):
    print("=" * 78)
    print(f"  {label}")
    print("=" * 78)
    print(f"  {'arm':<8}{'seed':>7}{'n_gt':>6}{'n_pred':>8}{'报的比例':>9}{'F1':>8}")
    stats = {}
    for arm in ARMS:
        f1s, frs, rcs = [], [], []
        for s in SEEDS:
            d = load(arm, s, kind)
            if d is None:
                continue
            pc = d["per_chunk"]
            cids = sorted(pc)
            fire = np.array([1 if pc[c]["n_pred"] > 0 else 0 for c in cids], float)
            gt = np.array([1 if pc[c]["n_gt"] > 0 else 0 for c in cids], float)
            f1s.append(d["f1"]); frs.append(fire.mean()); rcs.append(d["n_pred"])
            print(f"  {arm:<8}{s:>7}{int(d['n_gt']):>6}{int(d['n_pred']):>8}"
                  f"{fire.mean():>9.2f}{d['f1']:>8.4f}")
        if len(f1s) >= 3:
            f1s = np.array(f1s); frs = np.array(frs)
            stats[arm] = {"f1": f1s, "fire": frs, "gt_rate": gt.mean(),
                          "corr": np.corrcoef(frs, f1s)[0, 1]}
            print(f"    -> {arm:<8} 真值正例率 {gt.mean():.2f}   "
                  f"F1 {f1s.mean():.4f} ± {f1s.std(ddof=1):.4f}   "
                  f"corr(报的比例, F1) = {stats[arm]['corr']:+.3f}")
    if len(stats) == 2:
        va = stats["mixnorm"]["f1"].var(ddof=1)
        vb = stats["own10"]["f1"].var(ddof=1)
        print(f"\n    方差比 F = {va/vb:.2f} (mixnorm/own10, 各 7 种子, df 6,6)")
        n, d = stats["mixnorm"], stats["own10"]
        # 同种子配对: 两臂跑的是同样的 7 个种子, 配对能扣掉"种子难度"这一项
        dd = d["f1"] - n["f1"]
        t = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))
        # 精确置换: 每对内随机交换符号, 2^7 = 128 种, 全枚举
        signs = np.array([[1 if (m >> i) & 1 else -1 for i in range(len(dd))]
                          for m in range(2 ** len(dd))])
        perm = (signs * dd).mean(1)
        p_perm = np.mean(np.abs(perm) >= abs(dd.mean()) - 1e-12)
        print(f"    配对 ΔF1(own10-mixnorm) = {dd.mean():+.4f} ± "
              f"{dd.std(ddof=1):.4f}   t({len(dd)-1}) = {t:+.2f}   "
              f"精确置换 p = {p_perm:.4f}")
        print(f"    逐种子 Δ: " + "  ".join(f"{x:+.4f}" for x in dd))
    print()
    return stats


def main():
    print()
    old = summarise("旧 300 块集 (自然流行率) —— 原始", "")
    filt = summarise("旧 300 块集 —— require_inside 过滤版 (与报告口径一致)", "_filt")
    e2 = summarise("E2 (事件密集, 1640 事件/1600 块)", "_e2")

    print("=" * 78)
    print("  汇总: 机制猜想成立吗")
    print("=" * 78)
    print("  猜想: mixnorm 不稳 = (各种子阈值不同) × (评估集无负例 -> 多报免费)")
    print("  预言: 自然流行率集上  corr(报的比例,F1) 更弱, 且 mixnorm 的 SD 更小\n")
    print(f"  {'评估集':<26}{'真值正例率':>11}{'corr(own10)':>13}"
          f"{'corr(mixnorm)':>15}{'SD(mixnorm)':>13}{'SD(own10)':>11}{'F':>7}")
    for name, st in (("旧300 (自然流行率)", filt), ("E2 (事件密集)", e2)):
        if len(st) < 2:
            continue
        o, n = st["own10"], st["mixnorm"]
        f = n["f1"].var(ddof=1) / o["f1"].var(ddof=1)
        print(f"  {name:<26}{n['gt_rate']:>11.2f}{o['corr']:>13.3f}"
              f"{n['corr']:>15.3f}{n['f1'].std(ddof=1):>13.4f}"
              f"{o['f1'].std(ddof=1):>11.4f}{f:>7.2f}")


if __name__ == "__main__":
    main()
