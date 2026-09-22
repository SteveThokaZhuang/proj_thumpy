"""预登记: k=4 会不会把 **SE_seed 也缩小**? —— 纯计算, 0 GPU, 0 读盘。

## 为什么这件事直击主旨

§5 的 k=4 功效预测、以及 §5.5 的流行率修正, **两者都假设**:

    SE_seed = 0.0117  「跨种子分量; **不随评估集增大而缩**」

于是「合并 SE = hypot(SE_seed, SE_chunk)」在 k=4 上只能靠 SE_chunk
那一半往下走, 天花板被钉死在

    t_max = ΔF1 / SE_seed = 0.0340 / 0.0117 = 2.91

**但那条假设本身是建模假设, 不是事实。** 跨种子 SD 0.0309 里混了两样东西:

    Var(观测跨种子) = Var(真实种子差异) + Var(每种子自身的块抽样测量误差)

后一项**是随块数缩的**。k=4 把块数从 300 提到 1200 (×4),
每种的测量误差 SD 应缩到 1/2。所以只要测量误差占了可观份额,
**跨种子 SD 本身就会从 0.0309 往下掉**, SE_seed 跟着掉, 天花板就**不是** 2.91。

## 模型 (先写死, 免得事后编)

记每种的块抽样测量误差 SD 为 σ_m (在 filt/300 块尺度上), 并记 7 个种子的
测量误差之间的相关系数为 ρ。则:

    观测跨种子 SD_filt² = Var(true) + σ_m²                    ... (1)
    SE_chunk(filt)      = σ_m · sqrt((1 + 6ρ) / 7)             ... (2)

(2) 的来历: 7 个量取均值, 每个方差 σ_m²、两两相关 ρ,
均值的方差 = σ_m²(1 + 6ρ)/7。

由 (2) 反解 σ_m:

    ρ = 1  ⇒ SE_chunk = σ_m          ⇒ σ_m = 0.0198
    ρ = 0  ⇒ SE_chunk = σ_m/√7       ⇒ σ_m = 0.0524

**ρ=0 那一端会推出 Var(true) < 0** —— 见下面脚本输出, 这本身就是信息:
它说明 7 个种子的块抽样误差**必然高度相关**(同一批难块对所有种子都难),
ρ=0 的模型与数据不相容。

## 预登记的预测 (2026-09-18 00:2x, k=4 结果到货**之前**)

见脚本输出的表。核心几条:

  1. **跨种子 SD 在 e4 上应落在 0.021 ~ 0.031 之间** (ρ=1 那端给 0.0257)。
     若实测 **< 0.021** ⇒ 说明 filt 上的跨种子离散主要来自测量噪声,
     **「SE_seed = 0.0117 不缩」这条假设是错的**, 天花板要重算。
  2. 相应地 SE_seed(e4) 应落在 **0.008 ~ 0.012**。
  3. **若 SE_seed 真缩到 0.008 附近, 而 ΔF1 只缩到 +0.0211**,
     则 t = 0.0211/hypot(0.008, 0.0081) = 1.86 —— **比原预测的 1.48 高**,
     但仍 < 2。所以即使这条假设错了, **结论方向不变**, 只是没那么绝望。

## 判读表

| e4 实测跨种子 SD | 读法 |
|---|---|
| < 0.021 | 假设错: filt 的跨种子离散主要是测量噪声 ⇒ 天花板高于 2.91, 重算 |
| 0.021 ~ 0.031 | 与 ρ≈1 的模型相符 ⇒ 假设基本成立, 天花板 ≈ 2.91 |
| > 0.031 | 种子差异在更大的块集上**暴露得更多** ⇒ 也要重算, 方向相反 |

用法:
  python scripts/g1_e4_seseed_prereg.py
"""
import os
import json

import numpy as np
from math import sqrt

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
SEEDS = [42, 1234, 7, 2024, 3407, 31337, 55555]

SD_FILT = 0.0309        # §7.2 实测: A 组跨种子 SD (300 块)
SE_CHUNK_FILT = 0.0198  # §7.2 实测: 块抽样 SE (300 块)
N = len(SEEDS)
K_RATIO = 4.0           # 1200 块 / 300 块


def sd_e4(v_true, sigma_m_filt):
    """e4 上的跨种子 SD: 真实方差不变, 测量方差缩 K_RATIO 倍。"""
    return sqrt(v_true + (sigma_m_filt / sqrt(K_RATIO)) ** 2)


def main():
    print("=" * 74)
    print("  预登记: k=4 会不会把 SE_seed 也缩小?")
    print("=" * 74)
    print(f"\n  实测 (filt/300 块): 跨种子 SD = {SD_FILT:.4f}, "
          f"SE_chunk = {SE_CHUNK_FILT:.4f}, n = {N}")
    print(f"  假设 (待检验): SE_seed = SD/√n = {SD_FILT/sqrt(N):.4f} **不缩**")

    print(f"\n  由 SE_chunk = σ_m·sqrt((1+6ρ)/7) 反解 σ_m:")
    print(f"    {'ρ':>6}{'σ_m':>10}{'Var(true)':>14}{'Var<0?':>10}")
    rows = []
    for rho in [1.0, 0.8, 0.5, 0.2, 0.0]:
        sigma_m = SE_CHUNK_FILT / sqrt((1 + 6 * rho) / 7)
        v_true = SD_FILT ** 2 - sigma_m ** 2
        bad = "❌ 是" if v_true < 0 else "否"
        rows.append((rho, sigma_m, v_true))
        print(f"    {rho:>6.1f}{sigma_m:>10.4f}{v_true:>+14.6f}{bad:>10}")

    print(f"\n  ⚠️ ρ ≲ 0.5 时 Var(true) 变负 ⇒ **模型与数据不相容**。")
    print(f"     物理含义: 同一批难块对**所有**种子都难, 7 个种子的块抽样")
    print(f"     误差必然高度相关 (ρ 接近 1)。这本身就是一条有用的结论 ——")
    print(f"     它意味着 σ_m 只能取到 0.0198 那一端, 不能再大。")

    print(f"\n  ── 预登记的预测 (只保留 Var(true) > 0 的相容区间) ──")
    print(f"    {'ρ':>6}{'σ_m':>10}{'SD_true':>12}{'预测 SD(e4)':>14}"
          f"{'预测 SE_seed(e4)':>18}")
    for rho, sigma_m, v_true in rows:
        if v_true <= 0:
            continue
        sd_t = sqrt(v_true)
        sde4 = sd_e4(v_true, sigma_m)
        print(f"    {rho:>6.1f}{sigma_m:>10.4f}{sd_t:>12.4f}{sde4:>14.4f}"
              f"{sde4/sqrt(N):>18.4f}")

    # 相容区间的端点
    valid = [(r, s, v) for r, s, v in rows if v > 0]
    lo = min(sd_e4(v, s) for _, s, v in valid)
    hi = max(sd_e4(v, s) for _, s, v in valid)
    print(f"\n    ⇒ 预测 **跨种子 SD(e4) ∈ [{lo:.4f}, {hi:.4f}]**")
    print(f"      预测 **SE_seed(e4) ∈ [{lo/sqrt(N):.4f}, {hi/sqrt(N):.4f}]**")

    print(f"\n  ── 对天花板的影响 (最有利的 SE 假设 vs 两种 ΔF1) ──")
    BEST_SESEED, BEST_SEC = 0.0080, 0.0081   # 上表最有利端 (ρ=0.5)
    for label, seseed, se_chunk, delta in [
        ("原假设不动 (SE_seed=0.0117, 无 SE_chunk)", 0.0117, 0.0, 0.0340),
        ("原假设 + 流行率修正 ΔF1", 0.0117, 0.0, 0.0211),
        ("SE_seed 缩到 0.0080 + 修正 ΔF1", BEST_SESEED, 0.0, 0.0211),
        ("SE 全缩到最有利 + 修正 ΔF1 + SE_chunk", BEST_SESEED, BEST_SEC, 0.0211),
        ("  ↑ 但 ΔF1 **不缩** (流行率修正是错的)", BEST_SESEED, BEST_SEC, 0.0340),
    ]:
        se_all = sqrt(seseed ** 2 + se_chunk ** 2)
        t = delta / se_all
        mark = "  ← 唯一可能过 2 的一行" if t > 2 else ""
        print(f"    {label:<42} t = {t:>5.2f}{mark}")

    print(f"\n  🔑 **真正的结论: 判据的胜负手是 ΔF1, 不是 SE_seed。**")
    print(f"     即使把两个 SE 都放到最有利端 (0.0080 / 0.0081),")
    print(f"       · ΔF1 若缩到 +0.0211 (流行率修正成立) ⇒ t = "
          f"{0.0211/sqrt(BEST_SESEED**2+BEST_SEC**2):.2f} < 2, **仍判不住**")
    print(f"       · ΔF1 若守在 +0.0340 (修正不成立)     ⇒ t = "
          f"{0.0340/sqrt(BEST_SESEED**2+BEST_SEC**2):.2f} > 2, **能过**")
    print(f"     所以「扩集能不能改判」**完全取决于 ΔF1 缩不缩**,")
    print(f"     而 ΔF1 缩不缩正是 k=4 (自然流行率) 要回答的事。")
    print(f"     这条**预先堵死**了一个后门: 事后不能拿「SE_seed 其实会缩」")
    print(f"     去把不显著的结果说成显著 —— 那要看 ΔF1, 不是看 SE_seed。")

    print(f"\n  ── 判读表 (预登记) ──")
    print(f"    e4 跨种子 SD < 0.021 : 假设错 ⇒ filt 的跨种子离散主要是测量")
    print(f"                           噪声 ⇒ 天花板高于 2.91, **必须重算**")
    print(f"    0.021 ~ 0.031        : 与 ρ≈1 相容 ⇒ 假设基本成立, 天花板 ≈ 2.91")
    print(f"    > 0.031              : 种子差异在更大的块集上暴露**更多** ⇒")
    print(f"                           也要重算, 方向相反")
    print(f"\n  ⚠️ 无论落在哪一档, 都**不改**「扩集能不能改判」的判据 (§7.4b):")
    print(f"     那要由 e4 上实测的 t 说话, 不由本条预登记说话。")


if __name__ == "__main__":
    main()
