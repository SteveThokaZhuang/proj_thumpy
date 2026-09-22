"""G1 配对分析: 两臂在同一批种子下的 ΔF1.

**为什么是配对**: LLaMA-Factory 的 seed 同时决定权重初始化与数据打乱顺序。
两臂在同一 seed 下跑, 共享除观测空间外的一切 —— 于是同一 seed 的 own10 与
mixnorm 构成一对, 差值 d_s = mixnorm_s − own10_s 消掉了"这个种子运气好不好"
这一项。独立样本的误差棒 √(SD_own² + SD_mix²) 在**两臂正相关**时会高估。

**估计对象**: ΔF1 的**均值**。CI 用 t 区间 (小样本, 假设 d 近似正态),
另给一个**精确置换检验** (符号翻转, 2^n 全枚举, 不假设分布) 作为稳健性对照。

注意与 chunk 级配对 bootstrap 的区别: 那个重采样 chunk、把训练好的权重当固定,
只捕捉评估集抽样方差, **是反保守的** (报告 caveat c)。这里重采样的是**种子**
(每次重跑 = 一次独立训练), 才是 run-to-run 方差。

用法: python ari_g1_paired.py
"""
import itertools
import json
import os
import sys

import numpy as np

# 自足: 只读 JSON, 不 import ari_g1_common —— 那个模块顶层 import soundfile,
# 会让本脚本在没装音频库的环境里跑不起来 (而它一个采样点都不需要)。
ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")

SEEDS = [42, 1234, 7, 2024, 3407, 31337, 55555]


def f1_of(path):
    """读一个 eval json, 返回 (f1, precision, recall, n_pred, per_chunk)."""
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    v = next(iter(d.values()))
    return (v["f1"], v["precision"], v["recall"], v["n_pred"], v["per_chunk"])


def t_crit_975(df):
    """t_{0.975}(df), 避免依赖 scipy (fd_analysis 环境没有)。"""
    # 常见自由度的小表; df>=30 用 1.96
    tbl = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
           7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179}
    return tbl.get(df, 1.96) if df < 30 else 1.96


def main():
    own, mix = {}, {}
    for s in SEEDS:
        op = (f"{ANNOT}/g1_eval_own10.json" if s == 42
              else f"{ANNOT}/g1_eval_own10_s{s}.json")
        mp = (f"{ANNOT}/g1_eval_mixnorm.json" if s == 42
              else f"{ANNOT}/g1_eval_mixnorm_s{s}.json")
        o, m = f1_of(op), f1_of(mp)
        if o:
            own[s] = o
        if m:
            mix[s] = m

    paired = [s for s in SEEDS if s in own and s in mix]
    print(f"两臂都有的种子: {paired}  (n={len(paired)})")
    missing = [s for s in SEEDS if s not in paired]
    if missing:
        print(f"⚠ 缺失: {missing}")

    print("\n" + "=" * 76)
    print("  逐种子 F1")
    print("=" * 76)
    print(f"  {'seed':>7} {'own10':>9} {'mixnorm':>9} {'d=m-o':>9}   "
          f"{'P_o':>7} {'P_m':>7} {'R_o':>7} {'R_m':>7}")
    ds = []
    for s in paired:
        o, m = own[s], mix[s]
        d = m[0] - o[0]
        ds.append(d)
        print(f"  {s:>7} {o[0]:9.4f} {m[0]:9.4f} {d:+9.4f}   "
              f"{o[1]:7.4f} {m[1]:7.4f} {o[2]:7.4f} {m[2]:7.4f}")

    ds = np.array(ds)
    n = len(ds)
    print(f"\n  均值 own10 = {np.mean([own[s][0] for s in paired]):.4f}   "
          f"mixnorm = {np.mean([mix[s][0] for s in paired]):.4f}")

    print("\n" + "=" * 76)
    print(f"  配对 ΔF1  (n={n} 对)")
    print("=" * 76)
    dbar = ds.mean()
    sd = ds.std(ddof=1) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    tc = t_crit_975(n - 1)
    print(f"  ΔF1 均值        = {dbar:+.4f}")
    print(f"  SD(d)           = {sd:.4f}")
    print(f"  SE = SD/√n      = {se:.4f}")
    print(f"  95% CI (t)      = [{dbar - tc*se:+.4f}, {dbar + tc*se:+.4f}]"
          f"   (t_{ {n-1} }={tc})")

    # 独立样本误差棒 (之前的做法) —— 用于对比
    so = np.std([own[s][0] for s in paired], ddof=1)
    sm = np.std([mix[s][0] for s in paired], ddof=1)
    unpaired = np.sqrt(so**2 / n + sm**2 / n)
    print(f"\n  对照 —— 当独立样本处理:")
    print(f"    SD(own10)={so:.4f}  SD(mixnorm)={sm:.4f}")
    print(f"    独立样本 SE = √(SD_o²/n + SD_m²/n) = {unpaired:.4f}")
    print(f"    95% CI = [{dbar - tc*unpaired:+.4f}, {dbar + tc*unpaired:+.4f}]")
    if unpaired > 0:
        print(f"    → 配对把标准误压到 {se/unpaired*100:.0f}%"
              f"  (比值 {se/unpaired:.3f})")
    if sd > 0 and so > 0 and sm > 0:
        r = np.corrcoef([own[s][0] for s in paired],
                        [mix[s][0] for s in paired])[0, 1]
        print(f"    两臂逐种子相关 r = {r:+.3f}")

    # ── 精确符号翻转置换检验 ────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("  稳健性: 精确置换检验 (符号翻转, 不假设分布)")
    print("=" * 76)
    obs = abs(dbar)
    cnt = tot = 0
    for signs in itertools.product([1, -1], repeat=n):
        tot += 1
        if abs(np.mean(ds * np.array(signs))) >= obs - 1e-12:
            cnt += 1
    print(f"  零假设: d 的分布关于 0 对称")
    print(f"  {cnt}/{tot} 种符号翻转下 |均值| ≥ 观测值 → p = {cnt/tot:.4f}")

    neg = int((ds < 0).sum())
    from math import comb
    p_sign = sum(comb(n, k) for k in range(neg, n + 1)) / 2**n
    print(f"  符号检验: {neg}/{n} 个种子 ΔF1 < 0,  单边 p = {p_sign:.4f}")

    # ── 方向一致性 (P/R) ────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("  方向一致性 (精度 / 召回)")
    print("=" * 76)
    pl = sum(1 for s in paired if mix[s][1] < own[s][1])
    rl = sum(1 for s in paired if mix[s][2] < own[s][2])
    print(f"  mixnorm 精度更低的种子: {pl}/{n}")
    print(f"  mixnorm 召回更低的种子: {rl}/{n}")

    # ── 机制层: tp / fp 计数 (比 F1 更稳的信号) ─────────────────────────
    print("\n" + "=" * 76)
    print("  机制层: 每对种子的 tp / fp / n_pred")
    print("=" * 76)
    cids = sorted(set.intersection(*[set(own[s][4]) for s in paired]))

    def counts(rec, s):
        tp = sum(rec[s][4][c]["tp"] for c in cids)
        fp = sum(rec[s][4][c]["fp"] for c in cids)
        return tp, fp

    print(f"  {'seed':>7} | {'tp_o':>5} {'tp_m':>5} {'Δtp':>5} | "
          f"{'fp_o':>5} {'fp_m':>5} {'Δfp':>5} | {'np_o':>5} {'np_m':>5}")
    dtp, dfp = [], []
    for s in paired:
        to, fo = counts(own, s)
        tm, fm = counts(mix, s)
        dtp.append(tm - to)
        dfp.append(fm - fo)
        print(f"  {s:>7} | {to:>5} {tm:>5} {tm-to:>+5} | "
              f"{fo:>5} {fm:>5} {fm-fo:>+5} | {to+fo:>5} {tm+fm:>5}")
    dtp, dfp = np.array(dtp), np.array(dfp)
    print(f"\n  Δtp 均值 = {dtp.mean():+.2f}  (SD {dtp.std(ddof=1):.2f})")
    print(f"  Δfp 均值 = {dfp.mean():+.2f}  (SD {dfp.std(ddof=1):.2f})")
    for name, d in [("Δtp", dtp), ("Δfp", dfp)]:
        m = d.mean()
        sdd = d.std(ddof=1)
        sed = sdd / np.sqrt(n)
        ci = (m - tc * sed, m + tc * sed)
        nz = int((d != 0).sum())
        neg = int((d < 0).sum())
        p = sum(comb(nz, k) for k in range(min(neg, nz - neg), nz + 1)) / 2**nz \
            if nz else 1.0
        print(f"    {name} 95% CI = [{ci[0]:+.2f}, {ci[1]:+.2f}]  "
              f"非零 {nz}/{n}, 其中为负 {neg}  双边符号 p = {min(1.0, 2*p):.4f}")

    d_tp = int(dtp.sum())
    d_fp = int(dfp.sum())
    print(f"\n  跨 {n} 对累计: Δtp = {d_tp:+d}   Δfp = {d_fp:+d}")
    print("  (每个 run 平均漏检 %+.1f 个、误报 %+.1f 个)"
          % (dtp.mean(), dfp.mean()))


if __name__ == "__main__":
    main()
