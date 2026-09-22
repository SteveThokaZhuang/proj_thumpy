"""#53 扩大评估集后的配对分析: 两套评估集 × 7 对同种子.

回答两个问题:
  1. **G1 的 ΔF1 到底是多少?** —— 在 1640 个真值事件 (旧评估集只有 26 个) 上重算,
     此时 recall 的分辨率是 1/1640 而不是 1/26。
  2. **run-to-run 方差里, 有多少是"评估集抽样"、有多少是"训练随机性"?**
     —— 这决定了"再加种子"和"再扩评估集"哪个才是对的下一步。

第 2 点是本脚本的主要新贡献。做法是**方差分解**:
  SD_across_seeds  = 换种子重训带来的 F1 波动 (含评估集抽样 + 训练随机性)
  SE_eval          = 固定权重、只重采样 chunk 的 bootstrap 标准误 (只含评估集抽样)
若 SD_across_seeds 远大于 SE_eval, 说明训练随机性才是主项, 扩评估集已到边际收益;
若两者相当, 说明评估集还不够大。**注意 SE_eval 单独看是反保守的** (它把训练好的
权重当固定, 见报告 caveat c) —— 这里只用它做分解, 不用它当显著性标尺。

用法: python scripts/ari_g1_e2_analyze.py
"""
import itertools
import json
import os
import sys

import numpy as np

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
SEEDS = [42, 1234, 7, 2024, 3407, 31337, 55555]
SETS = [("E2", "e2", "扩集 1600 块"), ("OLD", "filt", "旧 300 块+边界过滤")]
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179}


def tcrit(df):
    return T975.get(df, 1.96) if df < 30 else 1.96


def suffix(seed):
    return "" if seed == 42 else f"_s{seed}"


def load(arm, seed, st):
    p = f"{ANNOT}/g1_eval_{arm}{suffix(seed)}_{st}.json"
    if not os.path.exists(p):
        return None
    return next(iter(json.load(open(p)).values()))


def f1_from(rec):
    tp = sum(c["tp"] for c in rec["per_chunk"].values())
    fp = sum(c["fp"] for c in rec["per_chunk"].values())
    gt = sum(c["n_gt"] for c in rec["per_chunk"].values())
    p = tp / max(1, tp + fp)
    r = tp / max(1, gt)
    return (2 * p * r / (p + r) if (p + r) else 0.0), p, r, tp, fp, gt


def chunk_bootstrap_sd(recs, n_boot=2000, seed=0):
    """固定权重、只重采样 chunk 的 F1 标准误 (只反映评估集抽样方差).

    对每一对 (arm, seed) 用**同一批** chunk 重采样, 再算配对 ΔF1 —— 与报告
    §4 的配对 bootstrap 同口径。
    """
    cids = sorted(set.intersection(*[set(r["per_chunk"]) for r in recs]))
    tp = np.array([[recs[k]["per_chunk"][c]["tp"] for c in cids]
                   for k in range(len(recs))], dtype=float)
    fp = np.array([[recs[k]["per_chunk"][c]["fp"] for c in cids]
                   for k in range(len(recs))], dtype=float)
    gt = np.array([[recs[k]["per_chunk"][c]["n_gt"] for c in cids]
                   for k in range(len(recs))], dtype=float)

    def f1_of(idx):
        t, f, g = tp[:, idx].sum(1), fp[:, idx].sum(1), gt[:, idx].sum(1)
        p = t / np.maximum(1, t + f)
        r = t / np.maximum(1, g)
        return np.where((p + r) > 0, 2 * p * r / np.maximum(1e-12, p + r), 0.0)

    rng = np.random.default_rng(seed)
    n = len(cids)
    d = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        v = f1_of(idx)
        d[b] = v[1::2].mean() - v[0::2].mean()   # 偶=own10, 奇=mixnorm
    return d.mean(), d.std(ddof=1)


def report(st, label, verbose=True):
    own = {s: load("own10", s, st) for s in SEEDS}
    mix = {s: load("mixnorm", s, st) for s in SEEDS}
    paired = [s for s in SEEDS if own[s] and mix[s]]
    if len(paired) < 2:
        print(f"\n### {label}: 数据不足 (n={len(paired)}), 跳过")
        return None

    print("\n" + "=" * 78)
    print(f"  {label}   (stage={st}, n={len(paired)} 对)")
    print("=" * 78)
    n_gt = sum(c["n_gt"] for c in own[paired[0]]["per_chunk"].values())
    n_ch = len(own[paired[0]]["per_chunk"])
    print(f"  评估集: {n_ch} 块 / {n_gt} 个真值事件")

    print(f"\n  {'seed':>7} {'own10':>9} {'mixnorm':>9} {'d':>9} | "
          f"{'tp_o':>5} {'tp_m':>5} {'gt':>5}")
    ds, d_tp = [], []
    fo, fm = [], []
    for s in paired:
        o, m = own[s], mix[s]
        f_o = f1_from(o)
        f_m = f1_from(m)
        fo.append(f_o[0]); fm.append(f_m[0])
        d = f_m[0] - f_o[0]
        ds.append(d)
        d_tp.append(f_m[3] - f_o[3])
        print(f"  {s:>7} {f_o[0]:9.4f} {f_m[0]:9.4f} {d:+9.4f} | "
              f"{f_o[3]:>5.0f} {f_m[3]:>5.0f} {f_o[5]:>5.0f}")

    ds = np.array(ds)
    n = len(ds)
    dbar, sd = ds.mean(), ds.std(ddof=1)
    se = sd / np.sqrt(n)
    tc = tcrit(n - 1)
    lo, hi = dbar - tc * se, dbar + tc * se
    so, sm = np.std(fo, ddof=1), np.std(fm, ddof=1)
    r = np.corrcoef(fo, fm)[0, 1] if so > 0 and sm > 0 else float("nan")

    print(f"\n  均值 own10 = {np.mean(fo):.4f}   mixnorm = {np.mean(fm):.4f}")
    print(f"  ΔF1  = {dbar:+.4f}   SD(d) = {sd:.4f}   SE = {se:.4f}")
    print(f"  95% CI (t_{n-1}={tc}) = [{lo:+.4f}, {hi:+.4f}]"
          f"   {'**含 0**' if lo <= 0 <= hi else '**排除 0**'}")
    print(f"  每个 arm 的 SD(跨种子): own10 {so:.4f}  mixnorm {sm:.4f}"
          f"   (旧评估集 was 0.031 / 0.028)")
    print(f"  两臂相关 r = {r:+.3f}")

    cnt = sum(1 for signs in itertools.product([1, -1], repeat=n)
              if abs(np.mean(ds * np.array(signs))) >= abs(dbar) - 1e-12)
    neg = int((ds < 0).sum())
    from math import comb
    p_sign = sum(comb(n, k) for k in range(neg, n + 1)) / 2 ** n
    print(f"  置换检验 p = {cnt/2**n:.4f}   符号检验 {neg}/{n} 为负, 单边 p = {p_sign:.4f}")

    # ── 方差分解 ────────────────────────────────────────────────────────
    # **必须交错**成 [own_s1, mix_s1, own_s2, mix_s2, ...], 因为
    # chunk_bootstrap_sd 用 v[0::2]/v[1::2] 分臂。写成
    # `[own...] + [mix...]` 会让偶数位变成"前 4 个 own + 前 3 个 mix",
    # 分解出来的 ΔF1 均值就完全不是配对量了 —— 而且它照样出数, 不会报错。
    recs = [r for s in paired for r in (own[s], mix[s])]
    bmean, bse = chunk_bootstrap_sd(recs)
    print(f"\n  ── 方差分解 ──")
    print(f"  SD(跨种子)  = {sd:.4f}   ← 总波动 (评估集抽样 + 训练随机性)")
    print(f"  SE_eval     = {bse:.4f}   ← 只重采样 chunk, 权重固定 (只含评估集抽样)")
    # SD(d) 极小时说明**扩集成功了** (评估集抽样不再是主项), 这时除法会爆 —— 别让
    # 一个 RuntimeWarning 盖住真正的好消息。sd==0 在合成自检里出现过 (case A)。
    frac = (bse ** 2) / (sd ** 2) if sd > 1e-9 else float("nan")
    print(f"  → 评估集抽样解释总方差的 {frac*100:.0f}%; "
          f"训练随机性解释 {(1-frac)*100:.0f}%")
    print(f"     (bootstrap ΔF1 均值 {bmean:+.4f}, 与实测 {dbar:+.4f} 对照)")

    # ── 功效 ────────────────────────────────────────────────────────────
    mde = (tcrit(n - 1) + 0.84) * sd / np.sqrt(n)     # 80% power, 双侧 0.05
    print(f"\n  ── 功效 ──")
    print(f"  n={n} 的最小可检出效应 (80% power) = {mde:.4f}")
    if mde > 1e-9:
        print(f"  观测 |ΔF1| = {abs(dbar):.4f} = MDE 的 {abs(dbar)/mde*100:.0f}%")
    else:
        print(f"  观测 |ΔF1| = {abs(dbar):.4f}  (SD(d)≈0, 任何非零效应都远超 MDE)")
    for target in (0.01, 0.02, 0.03):
        need = ((tcrit(n - 1) + 0.84) * sd / target) ** 2
        print(f"    要在 80% power 下检出 {target:.3f} 需要 n ≈ {need:.1f} 对")
    return {"dbar": dbar, "sd": sd, "se": se, "ci": (lo, hi), "r": r,
            "mde": mde, "se_eval": bse, "n_gt": n_gt, "n_chunks": n_ch,
            "mean_own": float(np.mean(fo)), "mean_mix": float(np.mean(fm)),
            "ds": [float(x) for x in ds],
            "frac_eval_var": float(frac)}


def main():
    out = {}
    for label, st, _ in SETS:
        out[st] = report(st, f"{label} ({_})")

    print("\n" + "=" * 78)
    print("  对照: 扩集把什么改变了")
    print("=" * 78)
    a, b = out.get("e2"), out.get("filt")
    if a and b:
        print(f"  {'':16} {'旧 300+过滤':>14} {'扩集 1600':>14}")
        for k, lab in [("n_chunks", "块数"), ("n_gt", "真值事件"),
                       ("mean_own", "own10 均值"), ("mean_mix", "mixnorm 均值"),
                       ("dbar", "ΔF1"), ("sd", "SD(d)"), ("se_eval", "SE_eval"),
                       ("mde", "MDE(n=7)")]:
            va, vb = b[k], a[k]
            print(f"  {lab:16} {va:>14.4f} {vb:>14.4f}")
        print(f"  {'ΔF1 95% CI':16} [{b['ci'][0]:+.4f},{b['ci'][1]:+.4f}]"
              f" [{a['ci'][0]:+.4f},{a['ci'][1]:+.4f}]")
        print(f"\n  SD(d) 从 {b['sd']:.4f} 降到 {a['sd']:.4f} "
              f"({a['sd']/b['sd']*100:.0f}%)")
        print(f"  评估集抽样占方差的比: 旧 {b['frac_eval_var']*100:.0f}% → "
              f"扩集 {a['frac_eval_var']*100:.0f}%")
    json.dump(out, open(f"{ANNOT}/g1_e2_analyze.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"\n-> {ANNOT}/g1_e2_analyze.json")


if __name__ == "__main__":
    main()
