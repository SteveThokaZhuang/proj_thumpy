"""Q3 结果的补充分析: 负号在**哪** — 以及一个必须标注为「事后」的观察。

## 背景

`g1_q3_analyze.py --group both` 给出主结果 (方差拆解)。
本脚本补三件它没做的事, 且**逐件标注是不是预登记的**:

  ① (预登记) B 组负号**个数** → 对照 `g1_q3_prereg.py` 的判读表。
  ② (**事后**) 负号在**哪个 ds** → 两组的负号是否指向同一个 data_seed?
  ③ (**事后**) A、B 两组的**均值**之差。

## ② 为什么值得单独看 (以及为什么必须标「事后」)

A 组里 `seed=55555` 那个 run, 其 `data_seed` 默认取
`torch.random.initial_seed()` —— `set_seed(55555)` 之后就是 **55555**。
也就是说 **A 的 seed=55555 同时是 data_seed=55555**。

而 B 组 `ds=55555` 是 seed=42、data_seed=55555。

若两组唯一的负号**都是 55555**, 那共同变量是**数据顺序**而非初值 ——
这恰好回答 §4.3 提出的问题 (H-初值 vs H-顺序), 而且答案指向 **H-顺序**。

**但这是事后看到的, 不在预登记里。** 事后挑出的模式必须：
  · 明确标注为探索性, 不作为确证;
  · 报出「若纯属巧合, 撞上同一个 ds 的概率」;
  · 若要坐实, 得**另起一次预登记** (例如换一组新 data_seed 再测 55555)。

用法:
  python scripts/g1_q3_results_extra.py
"""
import json
import os
from math import sqrt

import numpy as np

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
SEEDS = [42, 1234, 7, 2024, 3407, 31337, 55555]
ARMS = ["own10", "mixnorm"]


def sfx(s):
    return "" if s == 42 else f"_s{s}"


def load(tag):
    p = f"{ANNOT}/g1_eval_{tag}.json"
    return next(iter(json.load(open(p)).values())) if os.path.exists(p) else None


def f1_of(pc, idx):
    tp = fp = gt = 0
    for i in idx:
        e = pc[i]
        tp += e["tp"]; fp += e["fp"]; gt += e["n_gt"]
    p = tp / max(1, tp + fp); r = tp / max(1, gt)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def deltas(tags):
    """{key: ΔF1}, 全部在**公共块**上算 —— 缺任何一对就整组丢掉, 不混集。"""
    P = {}
    for k, (ta, tb) in tags.items():
        a, b = load(ta), load(tb)
        if a and b:
            P[k] = (a, b)
    if not P:
        return {}
    cids = sorted(set.intersection(
        *(set(v[0]["per_chunk"]) for v in P.values()),
        *(set(v[1]["per_chunk"]) for v in P.values())))
    out = {}
    for k, (a, b) in sorted(P.items()):
        out[k] = (f1_of([a["per_chunk"][c] for c in cids], range(len(cids)))
                  - f1_of([b["per_chunk"][c] for c in cids], range(len(cids))))
    return out


def main():
    A = deltas({s: (f"{ARMS[0]}{sfx(s)}_filt",
                    f"{ARMS[1]}{sfx(s)}_filt") for s in SEEDS})
    B = deltas({ds: (f"{ARMS[0]}_ds{ds}_filt",
                     f"{ARMS[1]}_ds{ds}_filt") for ds in SEEDS})

    print("=" * 74)
    print("  Q3 补充: 负号在**哪** (逐项标注是否预登记)")
    print("=" * 74)
    print(f"\n  A 组 (seed 变): {[(k, round(v,4)) for k, v in A.items()]}")
    print(f"  B 组 (data_seed 变, seed 固定 42): "
          f"{[(k, round(v,4)) for k, v in B.items()]}")

    # ---------- ① 预登记: 负号个数 ----------
    na = sum(1 for v in A.values() if v < 0)
    nb = sum(1 for v in B.values() if v < 0)
    print(f"\n  ── ① (预登记) 负号个数 ──")
    print(f"    A 组 {na}/7,  B 组 {nb}/7")
    print(f"    预登记判读表: 0~1 个 ⇒ 与零模型相符, 不加不减")
    print(f"    结论: {'✅ 落在预登记区间内' if nb <= 1 else '⚠️ 超出预登记区间'}")

    # ---------- ② 事后: 负号在哪 ----------
    print(f"\n  ── ② (🔴 **事后**, 非预登记) 负号落在哪个 key ──")
    neg_a = [k for k, v in A.items() if v < 0]
    neg_b = [k for k, v in B.items() if v < 0]
    print(f"    A 组负号: {neg_a}")
    print(f"    B 组负号: {neg_b}")
    if len(neg_a) == 1 and len(neg_b) == 1 and neg_a[0] == neg_b[0]:
        k = neg_a[0]
        print(f"    ⚠️ **两组唯一的负号都落在 {k}。**")
        print(f"       A 的 seed={k} 那个 run, data_seed 默认 = {k}")
        print(f"       (SeedableRandomSampler: initial_seed 取 torch.random.initial_seed(),")
        print(f"        而 set_seed({k}) 之后就是 {k}) ⇒ **A 的 seed={k} 同时是 data_seed={k}**。")
        print(f"       两组唯一共同变量 = **数据顺序 {k}** ⇒ 指向 H-顺序, 不是 H-初值。")
        print(f"\n       🔴 但这是**事后看到的**, 不在预登记内。若纯属巧合,")
        print(f"          负号在 B 组 7 个位置里撞上同一个的概率 = **1/7 ≈ "
              f"{1/7:.3f}**。")
        print(f"          这**不足以**单独坐实, 只能作为**探索性**线索。")
        print(f"\n       注意两者**量级差很多**: A 的 {k} 是 {A[k]:+.4f},")
        print(f"       B 的 {k} 只有 {B[k]:+.4f}。若差异全来自 shuffle,")
        print(f"       两者该接近 ⇒ 提示 **init({k}) 也贡献了一部分**,")
        print(f"       但那是**单点相减**, 没有任何误差棒, 别当分解读。")
    else:
        print(f"    两组的负号不重合 ⇒ 没有「同一个 data_seed 特别差」的证据。")

    # ---------- ③ 事后: 两组均值之差 ----------
    va, vb = np.array(list(A.values())), np.array(list(B.values()))
    diff = vb.mean() - va.mean()
    se = sqrt(va.var(ddof=1) / len(va) + vb.var(ddof=1) / len(vb))
    print(f"\n  ── ③ (🔴 **事后**) 两组均值之差 ──")
    print(f"    A 组均值 = {va.mean():+.4f} (n={len(va)})")
    print(f"    B 组均值 = {vb.mean():+.4f} (n={len(vb)})")
    print(f"    差       = {diff:+.4f}   合并 SE = {se:.4f}   t = {diff/se:+.2f}")
    print(f"    ⇒ {'两均值不能断言有差 (|t| < 2)' if abs(diff/se) < 2 else '两均值有差'}")
    print(f"\n    ⚠️ 两组**不是**同一总体的两个随机样本:")
    print(f"       A 是 (seed=s, data_seed=s) 的 7 对, B 是 (seed=42, data_seed=ds) 的 7 对。")
    print(f"       均值差里混着「固定 seed=42 这个初值好不好」的效应 ——")
    print(f"       A 组里 seed=42 本来就偏高 (+{A.get(42, float('nan')):.4f} vs 均值 "
          f"{va.mean():+.4f})。")

    # ---------- 方差拆解 ----------
    print(f"\n  ── 方差拆解 (主结果, 由 g1_q3_analyze.py 给) ──")
    sA, sB = va.var(ddof=1), vb.var(ddof=1)
    n = len(va)
    print(f"    Var(A) = {sA:.6f}  (SD {va.std(ddof=1):.4f})  n={n}")
    print(f"    Var(B) = {sB:.6f}  (SD {vb.std(ddof=1):.4f})  n={len(vb)}")
    d = sA - sB
    print(f"    Var(init) = {d:.6f}  (占比 {d/sA:.1%})")

    # 🔴 上面那个点估计**没有误差棒**, 必须补。这是本夜反复踩的坑:
    #    「两个噪声量相减得到的数」不能当结论报, 先看它能不能和 0 区分。
    print(f"\n    🔴 先给这个差值配误差棒 —— 拿 F 检验, 别报裸点估计:")
    try:
        from scipy import stats
        F = sA / sB
        p = 2 * min(stats.f.cdf(F, n - 1, len(vb) - 1),
                    1 - stats.f.cdf(F, n - 1, len(vb) - 1))
        print(f"      F = Var(A)/Var(B) = {F:.3f}   (df={n-1},{len(vb)-1})   "
              f"双侧 p = {p:.2f}")
        print(f"      ⇒ {'**不能拒绝 Var(A)=Var(B)**' if p > 0.05 else '两者有差'}"
              f" —— 即 **Var(init) 与 0 不可区分**。")

        # 各自方差的 95% CI (卡方), 再逐端相减取最宽 —— 保守但诚实
        chi_lo = stats.chi2.ppf(0.975, n - 1)
        chi_hi = stats.chi2.ppf(0.025, n - 1)
        ciA = ((n - 1) * sA / chi_lo, (n - 1) * sA / chi_hi)
        ciB = ((n - 1) * sB / chi_lo, (n - 1) * sB / chi_hi)
        print(f"\n      Var(A) 95% CI = [{ciA[0]:.6f}, {ciA[1]:.6f}]")
        print(f"      Var(B) 95% CI = [{ciB[0]:.6f}, {ciB[1]:.6f}]")
        print(f"      ⇒ Var(init) 的保守区间 ≈ "
              f"[{ciA[0]-ciB[1]:+.6f}, {ciA[1]-ciB[0]:+.6f}]")
        print(f"        **横跨 0**。" if (ciA[0]-ciB[1]) < 0 < (ciA[1]-ciB[0])
              else "        **不跨 0**。")
        print(f"        上界 {ciA[1]-ciB[0]:.6f} 换算成 SD = "
              f"{(ciA[1]-ciB[0])**0.5:.4f} —— 比总 SD "
              f"{va.std(ddof=1):.4f} 还大 ⇒ **约束力几乎为零**。")
    except ImportError:
        print(f"      (scipy 不可用, 跳过)")

    print(f"\n    ✅ 正确的说法:")
    print(f"       **init 分量在 n=7 下检不出**; shuffle 占**已分辨部分**的几乎全部。")
    print(f"       ❌ 不要写「shuffle 95.2% / init 4.8%」—— 那个 4.8% 是 "
          f"{d:.6f},\n          它的区间横跨 0, 当点估计报就是把噪声包装成发现。")
    print(f"       (memory: sd-from-few-points-manufactures-anomalies)")
    print(f"\n    📌 对本篇主旨的含义 (关键, 别漏):")
    print(f"       拆出「方差来自 shuffle 还是 init」**不改判任何东西** ——")
    print(f"       功效计算吃的是 SD_across={va.std(ddof=1):.4f}, 与它由谁构成无关。")
    print(f"       而且两组 SD 几乎相等 ⇒ **「固定 data_seed」也不降方差**。")
    print(f"       ⇒ Q3 买到的是**诊断**, 不是**功效**; 要 t>2 仍得那 15 个种子。")
    print(f"       (memory: expand-eval-set-buys-diagnosis-not-power)")


if __name__ == "__main__":
    main()
