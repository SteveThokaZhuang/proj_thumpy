"""评估集块数 k 与所需种子数 n 的**前沿** —— 纯读盘, 0 GPU。

## 要回答的问题

§5.8b 算出「线上（+0.0210 > 线 0.0195）」但**要多少种子没有稳定答案**（13, 区间 6~62）。
那个不定性来自 `n ∝ SD²/((ΔF1/2)² − SE_chunk²)`：**分母是两个大数的小差**。

分母小 = `SE_chunk` 逼近 `ΔF1/2` = 天花板 `t→∞` 逼近 2。
所以真正的问题是：**把 `SE_chunk` 压下去多少，这笔账才不病态？**

而 `SE_chunk` **只认总块数**（§5.5b 定案）。**加块比加种子便宜得多** ——
块只要跑评估（0 训练），种子要重训。

⇒ 本脚本算的就是：**块数 k → 天花板 t 与所需种子 n**，给「下一步跑什么」一个可判的账。

## 三条纪律（本项目踩过的坑，逐条对应）

1. **不假设指数，先量**。§5 当初拟合 `SE_chunk ∝ k^−0.672` 外推 32 倍，
   实测是 **−0.515**（§5.5b）。所以这里**在 e4 内部**直接把 k 扫一遍，
   量出 `SE_chunk(k)`，并**报每个 k 的指数**，不外推。
2. **自检必须能失败**：k = 1200（= 全集）时必须**逐位复现** `boot_se` 的
   `SE_chunk = 0.009745`。复现不了就停 —— 说明这个"推广版"和口径定义不一致。
3. **口径只有一处**：`SE_chunk` 的定义在 `g1_e4_analyze.boot_se`。
   本脚本的 `se_at_k` 是它的**推广**（块数可变），并靠上面那条断言绑死。

⚠️ **本脚本量的是「评估精度」，不是「换一批新块的批间方差」**。
把 k 从 1200 加到 2400，如果新块来自**另一批录音时刻**，`ΔF1` 本身也会动
（§7.5 的落空正是这个：子采样拟合测的是"从这批块里取更少"，不是"再抓一批"）。
所以下面的账回答的是「**在这批块的总体里**把精度做到多少」，
**不含**「换一批块 ΔF1 会漂多少」那一项 —— 那一项只能靠真跑一批新的来量。

用法: python scripts/g1_e4_block_frontier.py [--set e4] [--boot 2000]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ⚠️ 一律 import, 不重写 (判据/口径类只此一处定义)
from g1_e4_analyze import paired, boot_se, boot_se_mc, f1_of, SEEDS  # noqa: E402

# 天花板 t 的目标值 —— 不是"希望 t 等于这个"，而是"到这个量级，账才不病态"
CEIL_TARGETS = [2.5, 3.0, 4.0]


def se_at_k(vec, cids, seeds, k, boot, rng_seed=0):
    """块数取 k 时的 `SE_chunk` —— `boot_se` 的推广。

    k == len(cids) 时**必须**与 `boot_se` 逐位相同 (main 里有断言)。
    做法: 每次重抽 **k** 个块 (有放回), 而非全集大小。
    """
    n = len(cids)
    rng = np.random.default_rng(rng_seed)
    db = np.empty(boot)
    for b in range(boot):
        ii = rng.integers(0, n, k)
        db[b] = np.mean([f1_of(vec[("own10", s)], ii)
                         - f1_of(vec[("mixnorm", s)], ii) for s in seeds])
    return float(db.std(ddof=1))


def required_n(delta, sd, se_chunk, t_target=2.0):
    """t = delta / hypot(sd/sqrt(n), se_chunk) = t_target 解出 n。

    返回 None 表示**判不住** (分母 ≤ 0, 即连无穷多个种子也够不到 t_target)。
    """
    need = (delta / t_target) ** 2 - se_chunk ** 2
    if need <= 0:
        return None
    return sd ** 2 / need


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="tag_set", default="e4")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    print("=" * 78)
    print("  块数 k → 天花板 t 与所需种子数 n  (评估精度前沿)")
    print("=" * 78)

    D = paired(args.tag_set)
    if D is None or len(D["seeds"]) != len(SEEDS):
        print(f"  🔴 {args.tag_set} 产物不齐, 退出 —— 不拿残缺样本做计划。")
        return
    vec, cids, seeds = D["vec"], D["cids"], D["seeds"]
    dv = np.array([D["d"][s] for s in seeds])
    delta, sd = float(dv.mean()), float(dv.std(ddof=1))
    n_full = len(cids)

    print(f"  集: {args.tag_set}   配对种子 {len(seeds)}/{len(SEEDS)}   "
          f"共同块 {n_full}")
    print(f"  ΔF1 = {delta:+.4f}   SD_across = {sd:.4f}   (现算)\n")

    # ---------- 自检: k = 全集时必须是 boot_se 本身 ----------
    print("  ── 自检: k = 全集时须与 g1_e4_analyze.boot_se 逐位相同 ──")
    ref = boot_se(vec, cids, seeds, args.boot)
    got = se_at_k(vec, cids, seeds, n_full, args.boot)
    same = abs(ref - got) < 1e-12
    print(f"    boot_se               = {ref:.6f}")
    print(f"    se_at_k(k={n_full})  = {got:.6f}   "
          f"{'✅ 逐位相同' if same else '🔴 不同 —— 停, 推广版口径歪了'}")
    if not same:
        return

    # ---------- 扫 k ----------
    ks = [75, 150, 300, 600, 900, n_full]
    ks = sorted(set(k for k in ks if k <= n_full))
    print(f"\n  ── SE_chunk(k) 实测 (每点 bootstrap {args.boot} 次) ──")
    print(f"    {'k':>6}{'SE_chunk':>12}{'天花板 t':>12}{'所需种子':>10}"
          f"{'  vs 全集':>12}{'局部指数':>12}")
    rows = []
    prev = None
    for k in ks:
        se = se_at_k(vec, cids, seeds, k, args.boot)
        ceil_t = delta / se
        n_req = required_n(delta, sd, se)
        ratio = se / ref
        # 局部指数: SE ∝ k^a ⇒ a = ln(se/prev_se)/ln(k/prev_k)
        if prev is None:
            expo = float("nan")
        else:
            expo = float(np.log(se / prev[1]) / np.log(k / prev[0]))
        prev = (k, se)
        rows.append((k, se, ceil_t, n_req, expo))
        print(f"    {k:>6}{se:>12.6f}{ceil_t:>12.3f}"
              f"{(f'{n_req:.1f}' if n_req else '判不住'):>10}"
              f"{ratio:>11.2f}×"
              f"{(f'{expo:+.3f}' if expo == expo else '—'):>12}")

    print(f"\n  ⚠️ 最后一列是**局部**指数（相邻两点之间量的），不是全局拟合 ——")
    print(f"     全集 vs k=75 的**平均**指数 = "
          f"{np.log(ref / rows[0][1]) / np.log(n_full / rows[0][0]):+.3f}"
          f"   (1/√k 对应 −0.500)")

    # ---------- 反过来问: 要到给定的天花板, 需要多少块 ----------
    print(f"\n  ── 反解: 天花板 t 要到某值, 需要多少块? ──")
    print(f"     (按实测的**平均**指数外推, 并**标出外推倍数**)")
    a = float(np.log(ref / rows[0][1]) / np.log(n_full / rows[0][0]))
    # 外推该用的其实是**靠近全集**的局部指数, 不是全程平均 —— 两个都算, 摆出来
    a_loc = rows[-1][4] if rows[-1][4] == rows[-1][4] else a
    print(f"    {'天花板 t':>10}{'需 SE_chunk':>14}{'需块数 k':>12}"
          f"{'是当前':>10}{'该 k 下所需种子':>16}")
    for ct in CEIL_TARGETS:
        se_need = delta / ct
        k_need = n_full * (se_need / ref) ** (1.0 / a)
        n_at = required_n(delta, sd, se_need)
        print(f"    {ct:>10.1f}{se_need:>14.6f}{k_need:>12.0f}"
              f"{k_need/n_full:>9.2f}×"
              f"{(f'{n_at:.1f}' if n_at else '判不住'):>16}")
    print(f"    ⚠️ 上面的 k 用**全程平均**指数 {a:+.3f}; "
          f"换成靠近全集的局部指数 {a_loc:+.3f} 或理论的 −0.500:")
    for ct in CEIL_TARGETS:
        se_need = delta / ct
        ks = [n_full * (se_need / ref) ** (1.0 / ax)
              for ax in (a, -0.500, a_loc)]
        print(f"      天花板 {ct}: k = {ks[0]:.0f} / {ks[1]:.0f} / {ks[2]:.0f}"
              f"   (差 {max(ks)/min(ks)-1:.1%})")
    print(f"    ⇒ 若三者一致, 说明结论**不是指数挑出来的**; 不一致就要报区间。")

    # ---------- 现状那一行, 摆在一起看 ----------
    n_now = required_n(delta, sd, ref)
    print(f"\n  ── 现状 ──")
    print(f"    k = {n_full}, SE_chunk = {ref:.6f} ⇒ 天花板 t = {delta/ref:.3f}"
          f" (比 2 高 {(delta/ref-2)/2:+.1%})")
    # ⚠️ required_n 返回 None = **判不住** (分母 ≤ 0: 天花板 t < 2, 无穷多种子
    #    也够不到)。e4 上它恰好是 12.7, 于是这里写成 `{n_now:.1f}` 也一直没炸 ——
    #    直到 e4x 的天花板掉到 1.67 才 TypeError。宁可在这里判空, 不要靠运气。
    if n_now is None:
        print(f"    所需种子 n = **判不住** —— 天花板 t = {delta/ref:.3f} < 2,"
              f" 无穷多个种子也够不到 2。")
    else:
        print(f"    所需种子 n = {n_now:.1f}  ← §5.8b 的 13")
    den = (delta / 2) ** 2 - ref ** 2
    print(f"    分母 = (ΔF1/2)² − SE_chunk² = {(delta/2)**2:.3e} − {ref**2:.3e}"
          f" = {den:.3e}")
    if den <= 0:
        print(f"    ⇒ **分母 ≤ 0** ⇒ 天花板 t < 2: 这条账在这里是**死路**, "
              f"不是'不稳'。")
    else:
        print(f"    ⇒ 分母只有 ΔF1/2 那一项的 "
              f"{den/(delta/2)**2:.1%} —— 这就是 n 不稳的根源。")

    # ---------- MC: 这条曲线的每个点都是抽出来的 ----------
    _p, mc, lo_se, hi_se = boot_se_mc(vec, cids, seeds, args.boot, reps=6)
    print(f"\n  ── 蒙特卡洛误差 (每个 SE 都是 bootstrap 抽的, 不是算的) ──")
    print(f"    k={n_full} 处: SE_chunk = {_p:.6f} ± {mc:.6f} (MC, {mc/_p:.1%})"
          f"   范围 [{lo_se:.6f}, {hi_se:.6f}]")
    print(f"    ⇒ 天花板 t 在 [{delta/hi_se:.3f}, {delta/lo_se:.3f}] 之间漂,")
    print(f"      反解出的「需块数 k」**同比例**漂 —— 上表的 k 只能读到**两位有效数字**。")
    print(f"    ⇒ 但**结论方向不受影响**: 加块降 SE 是 `1/√k` 这个结构, 不是 MC 的产物。")

    print(f"\n  🔑 读法: **加块治的是分母, 加种子治的是分子。**")
    print(f"     `n ∝ SD²/分母`, 分母随 `1/k` 走 ⇒ 块数翻倍, 分母大致翻倍,")
    print(f"     所需种子**大致减半** —— 而块只要评估、不要训练。")
    print(f"     ⚠️ 但外推倍数见上表: 只在**这批块的总体**内成立,")
    print(f"        不含「换一批块 ΔF1 会漂多少」。")


if __name__ == "__main__":
    main()
