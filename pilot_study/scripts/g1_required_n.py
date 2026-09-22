"""要多少个种子才判得住? —— 0 GPU, 但**读 e4 产物** (不再是纯计算)。

## 为什么这是「主旨」问题

本项目绕了一大圈, 结论始终是同一句: **ΔF1 ≈ +0.021 判不住 (t = 1.90)**。
k=4 扩集能压的是 SE_chunk (块抽样), 而 **SE_seed (跨种子) 压不动** ——
它由「有几种种子」决定, 不由「评多少块」决定。

所以真正的问题是: **既然扩评估集有天花板, 那要多少个种子?**

本脚本把这笔账算清楚。它是**前瞻性**的: 不改变已有结论,
只回答「下一步该跑什么」。

## 模型

    ΔF1 的合并标准误   SE(n_s) = hypot( SD_across/√n_s , SE_chunk )
    判得住的条件       ΔF1 / SE(n_s) >= t_target

其中 `SD_across` 是**真实的跨种子 SD** (排掉测量噪声的那一份),
不随评估集大小而变 —— 这正是它压不动的原因。

## ⚠️ 输入从哪来 (2026-09-18 03:xx 大改: 三个输入全部改为**从产物现算**)

上一版的三个输入都是**写死的常数**(delta 0.0211 / sd_across 0.0257 /
se_chunk 0.0098), 其中 se_chunk 还是 **5 种子**时的临时口径,
而 e4 已经跑满 **7 种子**。这类「陈旧默认值」是 memory
(hardcoded-conclusions-escape-reproduction) 里那一族的同一个坑:
**结论不会随口径变, 但错的默认值会让结论悄悄错下去。**

现在默认值由 `g1_e4_analyze.paired()/boot_se()` **现算** ——
那也正是一处**唯一定义** (不重写, 只 import)。命令行参数仍可覆盖,
但一旦覆盖, 脚本会明确印出「你用的是手填值, 不是实测」。

## 🔴 本脚本现在会告诉你一件不舒服的事

`t_∞ = ΔF1 / SE_chunk ≈ 2.16`, 只比 2 高 8%。
而所需种子数 `n ∝ SD_across² / ((ΔF1/2)² − SE_chunk²)` ——
**分母是两个大数的小差**, 所以 n 对 ΔF1 和 SE_chunk **极度敏感**:
ΔF1 掉 5% ⇒ n 翻两三倍。**别把这里的 n 当成一个"要跑多少"的计划数字**,
它是一张「这个效应量级本来就处在分辨率边缘」的图。

用法:
  python scripts/g1_required_n.py
  python scripts/g1_required_n.py --delta 0.0211 --sd-across 0.0257
"""
import argparse
import os
import sys
from math import ceil, hypot, sqrt

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ⚠️ 一律 **import**, 不重写 (本项目约定: 判据/口径类只此一处定义)
from g1_e4_analyze import paired, boot_se, boot_se_mc, SEEDS  # noqa: E402

# §5.5 流行率修正的**预测**值。只用来对照, 不作默认 ——
# 它已经被实测取代 (两者差 0.0001, 但"差多少"不该由人记着)。
PRED_DELTA = 0.0211
PRED_SD = 0.0257          # §5.7 预测区间上端
PRED_SE_CHUNK = 0.0081    # §5.5 预测 (注意不是 5 种子实测的 0.0098)


def ceiling_t(delta, se_chunk):
    """n -> ∞ 时的极限 t (SE_seed -> 0, 只剩 SE_chunk)。"""
    return delta / se_chunk


def required_n(delta, sd_across, se_chunk, t_target=2.0):
    """达到 t_target 所需的种子数; 天花板本身就够不着时返回 None。"""
    if ceiling_t(delta, se_chunk) < t_target:
        return None
    # 解 delta / hypot(sd/√n, se_chunk) = t_target
    need = (delta / t_target) ** 2 - se_chunk ** 2
    if need <= 0:
        return None
    return ceil((sd_across ** 2) / need)


def sd_ci(v):
    """跨种子 SD 的 95% 卡方区间 —— n 小的时候它宽得吓人, 必须一起报。"""
    from scipy import stats as st
    n, s = len(v), float(np.std(v, ddof=1))
    return (s, s * sqrt((n - 1) / st.chi2.ppf(0.975, n - 1)),
            s * sqrt((n - 1) / st.chi2.ppf(0.025, n - 1)))


def measured(tag_set="e4", boot=2000, mc_reps=8):
    """从产物现算三个输入。产物不齐则返回 None (不许拿残缺样本算计划)。

    `se_mc` = `SE_chunk` 自己的蒙特卡洛误差。**必须一起报**, 因为
    `SE_chunk` 同时是**判据线**的输入 —— 报四位小数等于给它一个它没有的精度。
    """
    D = paired(tag_set)
    if D is None or len(D["seeds"]) != len(SEEDS):
        return None
    dv = np.array([D["d"][s] for s in D["seeds"]])
    return {"delta": float(dv.mean()),
            "sd_across": float(dv.std(ddof=1)),
            "se_chunk": boot_se(D["vec"], D["cids"], D["seeds"], boot),
            "se_mc": boot_se_mc(D["vec"], D["cids"], D["seeds"], boot, mc_reps),
            "boot": boot,
            "mc_reps": mc_reps,
            "n_seeds": len(D["seeds"]),
            "n_chunks": len(D["cids"]),
            "sd_ci": sd_ci(dv)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=None,
                    help="待检验的 ΔF1 (默认: e4 实测)")
    ap.add_argument("--sd-across", type=float, default=None,
                    help="真实跨种子 SD (默认: e4 实测)")
    ap.add_argument("--se-chunk", type=float, default=None,
                    help="块抽样 SE (默认: e4 实测)")
    ap.add_argument("--t-target", type=float, default=2.0)
    ap.add_argument("--min-per-run", type=float, default=12.0,
                    help="单个训练 run 的分钟数 (§3.1 实测约 13 min, Q4 约 12 min)")
    args = ap.parse_args()

    M = measured()
    print("=" * 74)
    print("  要多少个种子才判得住? (前瞻性功效账)")
    print("=" * 74)

    if M is None:
        print("\n  🔴 e4 产物不齐, **算不了实测输入**。")
        print("     本脚本拒绝退回到写死的常数 —— 那正是它被改掉的原因。")
        print("     要用手填值, 显式给 --delta/--sd-across/--se-chunk。")
        if args.delta is None or args.sd_across is None or args.se_chunk is None:
            return
        M = None

    # ---- 定输入: 实测优先, 手填覆盖 ----
    if M is not None:
        delta = args.delta if args.delta is not None else M["delta"]
        sd = args.sd_across if args.sd_across is not None else M["sd_across"]
        se = args.se_chunk if args.se_chunk is not None else M["se_chunk"]
        src = ("实测" if args.delta is None else "**手填**",
               "实测" if args.sd_across is None else "**手填**",
               "实测" if args.se_chunk is None else "**手填**")
    else:
        delta, sd, se = args.delta, args.sd_across, args.se_chunk
        src = ("**手填**",) * 3
        print("\n  ⚠️ 三个输入**全部手填** —— 下面的数不是从产物来的。")

    print(f"\n  输入: ΔF1 = {delta:+.4f} ({src[0]})   SD_across = {sd:.4f} ({src[1]})"
          f"   SE_chunk = {se:.4f} ({src[2]})")
    if M is not None and M.get("se_mc") is not None:
        _p, mc, lo_se, hi_se = M["se_mc"]
        print(f"\n  🔴 **SE_chunk 本身是抽出来的** (bootstrap {M['boot']} 次 × {M['mc_reps']} 个独立 rng):")
        print(f"     SE_chunk = {_p:.6f} ± {mc:.6f} (MC)   范围 [{lo_se:.6f}, {hi_se:.6f}]"
              f"   ({mc/_p:.1%})")
        print(f"     ⇒ **临界线在 [{2*lo_se:.4f}, {2*hi_se:.4f}] 之间漂**, "
              f"ΔF1 稳定值 {delta:+.4f}")
        print(f"     ⇒ MC 误差按 1/√boot 缩 (B 要 ×4 才减半); 判决稳不稳看下面的余量检查。")
    if M is not None:
        lo, hi = M["sd_ci"][1], M["sd_ci"][2]
        print(f"    现算自 e4: {M['n_seeds']} 种子 × {M['n_chunks']} 共同块"
              f"   (bootstrap 2000 次, 与 g1_e4_analyze 同一函数)")
        print(f"    SD_across 的 95% CI = [{lo:.4f}, {hi:.4f}]  ← **宽得吓人, 见下**")
        print(f"    对照 §5.5 预测: ΔF1 {PRED_DELTA:+.4f} / SD {PRED_SD:.4f} / "
              f"SE_chunk {PRED_SE_CHUNK:.4f}")
    print(f"    临界线 = SE_chunk×t_target = {se*args.t_target:.4f}"
          f"   (预测口径曾是 {PRED_SE_CHUNK*args.t_target:.4f})")
    ceil_here = ceiling_t(delta, se)
    print(f"    n→∞ 天花板 t = ΔF1/SE_chunk = {ceil_here:.3f}"
          f"   (要 > {args.t_target} 才有得谈; 余量 {ceil_here/args.t_target-1:+.1%})")

    print(f"\n  ── 种子数 vs 可达到的 t ──")
    print(f"    {'n_seeds':>8}{'SE_seed':>10}{'SE_all':>10}{'t':>8}   判得住?")
    for n in [3, 5, 7, 10, 12, 15, 20, 30, 50]:
        se_seed = sd / sqrt(n)
        se_all = hypot(se_seed, se)
        t = delta / se_all
        ok = "✅" if t >= args.t_target else ""
        cur = "   <- 现在这里" if n == (M["n_seeds"] if M else 7) else ""
        print(f"    {n:>8}{se_seed:>10.4f}{se_all:>10.4f}{t:>8.2f}   {ok}{cur}")

    # ---- 结论: 给区间, 不给单点 ----
    n_req = required_n(delta, sd, se, args.t_target)
    print(f"\n  ── 结论 ──")
    if n_req is None:
        print(f"    ❌ **无论多少种子都判不住**: n→∞ 时 SE → SE_chunk = {se:.4f},")
        print(f"       天花板 t = {ceil_here:.2f} < {args.t_target}")
        print(f"       要改判, 只能**把 ΔF1 做大**(改实验设计/加数据), 或者接受这个效应量级")
    else:
        runs = 2 * n_req
        hours = runs * args.min_per_run / 60.0
        print(f"    点估计: 需要约 **{n_req} 个种子** (现有 {M['n_seeds'] if M else 7} 个)"
              f" 才能到 t = {args.t_target}")
        print(f"    代价: 每臂 {n_req} 个训练 run = {runs} 个 "
              f"≈ {hours:.1f}h @ {args.min_per_run:.0f} min/run")
        if M is not None:
            print(f"\n    🔴 **但这是一个极不稳的点估计, 别当计划用。** 两个来源:")
            lo, hi = M["sd_ci"][1], M["sd_ci"][2]
            n_lo = required_n(delta, lo, se, args.t_target)
            n_hi = required_n(delta, hi, se, args.t_target)
            print(f"      (a) n ∝ SD², 而 SD 自己只有 {M['n_seeds']} 个点:"
                  f" 用 CI 两端算 ⇒ **{n_lo} ~ {n_hi}**")
            for d in (delta, delta * 0.95, 0.0180):
                nr = required_n(d, sd, se, args.t_target)
                print(f"      (b) ΔF1 = {d:+.4f} ⇒ " + (
                    f"{nr} 个种子" if nr else
                    f"**判不住** (恰好低于临界线 {se*args.t_target:.4f})"))
            print(f"      ⇒ 两个方向都能把答案挪好几倍。**它说明的是"
                  f"「效应量就在分辨率边缘」, 不是一个待办数字。**")
            # (c) 第三个来源: SE_chunk 自己的 MC 噪声 —— 刚发现的, 见 boot_se_mc
            _p, mc, lo_se, hi_se = M["se_mc"]
            n_mc_lo = required_n(delta, sd, lo_se, args.t_target)
            n_mc_hi = required_n(delta, sd, hi_se, args.t_target)
            print(f"      (c) 🔴 n ∝ 1/((ΔF1/2)²−SE_chunk²), 而 SE_chunk **本身是抽出来的**:"
                  f" ±{mc:.6f} (MC) ⇒ **{n_mc_lo} ~ {n_mc_hi}**")
            print(f"          (这一项与 (a)(b) 独立 —— 光把 bootstrap 做到 {M['boot']*4} 次"
                  f"就能把它减半, 但它**永远不为零**)")

            # ---- 判决的 MC 稳健性: 余量 / MC ≫ 1 才算稳 ----
            print(f"\n  ── 判决在 MC 误差下稳不稳? ──")
            for name, se_x in (("rng_seed=0 (点估计)", _p),
                               ("最极端的那次重抽", hi_se if delta > 2 * _p else lo_se)):
                line = se_x * args.t_target
                margin = delta - line
                print(f"    {name:<20} 线 = {line:.4f}   ΔF1 − 线 = {margin:+.4f}"
                      f"   {'线上' if margin > 0 else '线下'}")
            line_lo, line_hi = 2 * lo_se, 2 * hi_se
            worst = delta - line_hi
            ratio = abs(delta - 2 * _p) / (2 * mc)
            print(f"    ⇒ 线随 MC 在 [{line_lo:.4f}, {line_hi:.4f}] 漂;"
                  f" ΔF1 = {delta:+.4f}")
            print(f"    ⇒ **余量 / MC = {ratio:.1f}×** —— "
                  f"{'✅ 判决扛得住 MC' if ratio > 3 else '🔴 判决会被 MC 掀翻, 必须加大 boot'}")
            if worst <= 0:
                print(f"    🔴 最坏情形下 ΔF1 落到线下 —— 这个判决**不能报**。")
        else:
            print(f"\n  ⚠️ 三个输入是**手填**的, 没有 MC 误差可报 ——"
                  f" 手填值没有「抽出来」这一层, 但也就不享受「现算」的可复现性。")

    print(f"\n  ── 敏感度: ΔF1 缩水对所需种子的影响 (n ∝ 1/ΔF1²) ──")
    print(f"    {'ΔF1':>8}{'所需种子':>12}{'n→∞ 天花板 t':>16}")
    for d in [0.0340, 0.0250, PRED_DELTA, 0.0210, 0.0180, 0.0150, 0.0100]:
        n_r = required_n(d, sd, se, args.t_target)
        n_s = f"{n_r}" if n_r else "**判不住**"
        print(f"    {d:>+8.4f}{n_s:>12}{ceiling_t(d, se):>16.2f}")

    print(f"\n  ⚠️ 上表最右列说明: 只要 ΔF1 掉到 {se*args.t_target:.4f} 以下,")
    print(f"     就**连无穷多个种子也救不回来** —— 那时问题不在功效, 在效应量本身。")
    print(f"\n  ⚠️ 本脚本**不**为已有结论背书: 它只回答「下一步要跑多少」,")
    print(f"     不回答「这个效应存在吗」。后者由 e4 上实测的 t 说话 (§7.4b)。")


if __name__ == "__main__":
    main()
