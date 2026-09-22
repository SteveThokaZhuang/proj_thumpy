"""k=4 (e4) 的全部预登记 vs 实测 —— 一张表对完, 不靠手抄。

## 为什么要单独一个脚本

e4 牵动的预登记散在四处（§5.5 / §5.7 / §5.8b / §5.9 与 §7.4b 的表），
而且**有两套互相竞争的预测**（§7.5 原始的 t≈2.42~2.66 与 §5.5 修正后的 t≈1.48）。
靠人眼在长文档里逐个对, 迟早抄错一个数 —— 或者更糟: 挑一个对结论有利的报。

所以这里把**冻结值写成常量**(它们的全部意义就是不能再动),
把**实测值从产物现算**, 并排打印, 每行自判。

## 🔴 本脚本第一版栽过的坑（2026-09-18 02:4x, 值得留档）

第一版把**旧 filt 集的逐种子 ΔF1 也写死**在 `FROZEN["filt"]` 里。抄串了一列:

    seed   跨集脚本现算    我写死的
    1234   +0.0497         +0.0327   ❌
    2024   +0.0327         +0.0354   ❌
    3407   +0.0442         +0.0470   ❌

后果: 判据 B 的排序错了, 且 #71 的 r 算成 **+0.561** 而非 **+0.548**。
**差得小到不会被眼睛抓住 —— 但那就是一个伪造的数。**
⇒ 修法不是「改对数字」, 而是**取消写死**: filt 也从产物现算,
并加一条**会炸的自检**（复现 §7.2 的 ΔF1=+0.0340 / SD=0.0309）。
(memory: hardcoded-conclusions-escape-reproduction)

## 本脚本**不做**的事

不重算任何统计量的定义 —— `load` / `f1_of` / `SEEDS` / `ARMS` 从 `g1_e4_analyze`
import, `r_crit` 从 `g1_boundary_dispersion` import。
本脚本只负责**并排与判读**, 不定义口径。

> ⚠️ **2026-09-18 04:0x 补一句**：上面这句原先**只说对了一半** ——
> `deltas()` / `boot_se()` 当时是**本文件里的一份拷贝**（自己算共同块、自己重抽），
> 那正是「两份定义 = 迟早只改一份」的形状。现在已经改成直接调
> `g1_e4_analyze.paired()` / `.boot_se()`，并与改前**逐位一致**（已 `diff` 验过）。

用法: python scripts/g1_e4_reconcile.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ⚠️ 一律 **import**, 不重写 (本项目约定: 判据/口径类只此一处定义)
from g1_e4_analyze import (SEEDS, ARMS, load, f1_of,   # noqa: E402
                           paired, boot_se as boot_se_of)
from g1_boundary_dispersion import r_crit            # noqa: E402

# ---------- 冻结值: 写死在预登记页上的, 不得再动 ----------
#   ⚠️ 只放**预测**。任何**观测量**都必须现算 —— 见文件头那个坑。
FROZEN = {
    "sd7_band": (0.0210, 0.0257),      # §5.7  e4 跨种子 SD 该落的区间
    "sd7_reset": (0.021, 0.031),       # §5.7  判读表: 打破这两条边就要重算
    "delta_ub": 0.0211,                # §5.5/§7.4b  ΔF1 预测, 且声明为**上界**
    "t_corrected": 1.48,               # §5.5  修正后预测 t, 明确说「不跨 2」
    "se_chunk_pred": 0.0081,           # §5.5  修正后的 SE_chunk 预测
    "se_chunk_orig": (0.0052, 0.0078),  # §7.5  原始预测, 已被 §5.5 取代
    "t_orig": (2.42, 2.66),            # §7.5  原始 t 预测, 已被 §5.5 取代
    "extrap_55555_frozen": -0.0458,    # §5.9  封存时拟合 e4=-0.0161+0.8593*filt
    "extrap_55555_now": -0.0511,       # §5.9  6 种子重拟合 e4=-0.0187+0.9397*filt
    "s31337_pred": 0.0255,             # §5.9  唯一已完成的样本外点
    "s31337_actual": 0.0323,           # §5.9  实测(已记录在 §5.9)
    # filt 自检锚点 (§7.2 封存值) —— **只用来断言**
    "filt_anchor_delta": 0.0340,
    "filt_anchor_sd": 0.0309,
    # 历史临界线 (只作参照, 不作判据 —— 线必须现算)
    "crit_hist": {5: 0.0196, 6: 0.0185},
}
B = 2000


def deltas(tag_set):
    """现算某评估集的逐种子 ΔF1 + 共同块 + 完整产物表。

    **口径不在这里** —— 直接调 `g1_e4_analyze.paired()` (唯一一处定义)。
    本脚本只做并排与判读, 所以这里必须是**一次调用**, 不是一份拷贝。
    每个集合用**自己的**共同块 (filt 300 vs e4 1200), 不混。
    """
    D = paired(tag_set)
    if D is None:
        return {}, [], {}
    return D["d"], D["cids"], D["vec"]


def boot_se(d, V, cids, seeds=None):
    """块级 bootstrap SE —— 计算本身在 `g1_e4_analyze.boot_se` (唯一一处定义)。

    这里只补一个「种子表的默认值」, 不改算法。
    对**现有种子求平均后**再重抽块 ⇒ 种子数一变这个值就变(真实循环性)。
    """
    seeds = list(d) if seeds is None else [s for s in d if s in seeds]
    return boot_se_of(V, cids, seeds, B)


def main():
    print("=" * 78)
    print("  k=4 (e4) 端到端对账: 预登记 (冻结) vs 实测 (现算)")
    print("=" * 78)

    # ---------- 自检 0: filt 必须复现 §7.2 (这条会炸, 就是它防住了抄错) ----------
    df, cids_f, Vf = deltas("filt")
    af = np.array([df[s] for s in df])
    ok_f = (abs(af.mean() - FROZEN["filt_anchor_delta"]) < 5e-4
            and abs(af.std(ddof=1) - FROZEN["filt_anchor_sd"]) < 5e-4)
    print(f"\n  ── 自检 0: filt 现算应复现 §7.2 (ΔF1 +0.0340 / SD 0.0309) ──")
    print(f"    n={len(af)}  现算 ΔF1 {af.mean():+.4f}  SD {af.std(ddof=1):.4f}"
          f"   (封存 {FROZEN['filt_anchor_delta']:+.4f} / {FROZEN['filt_anchor_sd']:.4f})"
          f"   {'✅' if ok_f else '🔴 对不上 —— 口径或产物被改过, 停'}")
    if not ok_f:
        print("    🔴 自检不过 ⇒ 下面的对账一律不作数。")
        return

    # ---------- 载入 e4 ----------
    d, cids, V = deltas("e4")
    n_all = sum(1 for s in SEEDS for a in ARMS if load("e4", a, s) is not None)
    print(f"\n  产物 {n_all}/14" + ("" if n_all == 14 else "   🔴 **不齐 —— 判据不作数**"))
    n = len(d)
    print(f"  配对种子 {n}/7   共同块 {len(cids)}")
    if n < 2:
        print("  配对不足, 停。")
        return

    dv = np.array([d[s] for s in d])
    mean, sd = dv.mean(), dv.std(ddof=1)
    SE_seed = sd / np.sqrt(n)
    SE_chunk = boot_se(d, V, cids)
    CRIT = SE_chunk * 2.0
    t_merged = mean / float(np.hypot(SE_seed, SE_chunk))

    print(f"\n  ── 实测总览 (n={n} 种子) ──")
    for s in SEEDS:
        if s in d:
            print(f"    seed {s:>6}: ΔF1 {d[s]:+.4f}")
    print(f"    ΔF1 = {mean:+.4f}  SD = {sd:.4f}  SE_seed = {SE_seed:.4f}")
    print(f"    SE_chunk = {SE_chunk:.4f}   合并 SE = {np.hypot(SE_seed,SE_chunk):.4f}"
          f"   t = {t_merged:.3f}")
    print(f"    临界线 (SE_chunk×2) = {CRIT:.4f}")
    print(f"    (历史线: " + ", ".join(f"{k} 种子 {v:.4f}"
                                      for k, v in FROZEN["crit_hist"].items()) + ")")

    # ---------- 逐条对账 ----------
    rows = []
    lo, hi = FROZEN["sd7_band"]
    rows.append(("§5.7  e4 跨种子 SD ∈ [%.3f, %.3f]" % (lo, hi),
                 f"{sd:.4f}",
                 "✅ 命中" if lo <= sd <= hi else "❌ 区间外 ⇒ 按 §5.7 判读表要重算"))

    ub = FROZEN["delta_ub"]
    rows.append((f"§5.5  ΔF1 预测 ≤ {ub:+.4f} (声明为上界)",
                 f"{mean:+.4f}",
                 "✅ 在上界内" if mean <= ub else
                 "🔴 **上界被突破 —— 预登记的干净 miss, 方向与预期相反**"))

    tp = FROZEN["t_corrected"]
    rows.append((f"§5.5  修正预测 t ≈ {tp:.2f} (「不跨 2」)",
                 f"{t_merged:.3f}",
                 ("✅ 不跨 2, 与预测同向" if t_merged <= 2 else
                  "🔴 **跨了 2 —— 预测 miss(但这是参照量, 不是预登记主判据)**")))

    sep = FROZEN["se_chunk_pred"]
    rows.append((f"§5.5  SE_chunk 预测 {sep:.4f}",
                 f"{SE_chunk:.4f}",
                 "✅ 同量级" if abs(SE_chunk - sep) < 0.002 else
                 f"⚠️ 差 {SE_chunk-sep:+.4f}"))

    rows.append((f"§5.8b **主判据**: {n} 种子均值 vs 临界线 SE_chunk×2",
                 f"{mean:+.4f} vs {CRIT:.4f}",
                 "**线上** (加种子能救)" if mean > CRIT else
                 "**线下** (连无穷多种子也救不回来)"))

    # ---- §5.9 封存程序第 2 条: 附注 CRIT₆ (不含 55555) ----
    CRIT6 = None
    if 55555 in d and n == len(SEEDS):
        CRIT6 = boot_se(d, V, cids, seeds=[s for s in d if s != 55555]) * 2.0
        rows.append(("§5.9  C 的两口径同侧?", f"CRIT₇ {CRIT:.4f} / CRIT₆ {CRIT6:.4f}",
                     "✅ 同侧, C 可判" if (mean > CRIT) == (mean > CRIT6) else
                     "🔴 **不同侧 ⇒ C 不可判** (口径造成, 照实写)"))

    for s in SEEDS:
        if s == 55555 and s not in d:
            rows.append((f"§5.9  55555 封存/当前外推 "
                         f"{FROZEN['extrap_55555_frozen']:+.4f} / "
                         f"{FROZEN['extrap_55555_now']:+.4f}",
                         "见下", "与实测比序, 不比精度(外推本身会漂)"))

    o, m = FROZEN["t_orig"]
    rows.append((f"§7.5  原始 t 预测 {o}~{m} (已被 §5.5 取代, 留档)",
                 f"{t_merged:.3f}",
                 "（只记录: 两套预测互不相同, 报的时候要说清以哪套为准）"))

    w = max(len(r[0]) for r in rows)
    print(f"\n  ── 逐条对账 ──")
    for a, b, c in rows:
        print(f"    {a:<{w}} | {b:<22} {c}")

    # ---------- §5.9 定性判据 (序关系, 不受线的循环性影响) ----------
    print(f"\n  ── §5.9 定性判据 (不依赖 r 的大小) ──")
    filt = {s: df[s] for s in df}
    if 55555 not in d:
        print(f"    ⏳ 55555 未入 (需两臂) ⇒ 判据 A/B 无法判")
    else:
        order_e4 = sorted(d, key=lambda s: d[s])
        order_filt = sorted(filt, key=lambda s: filt[s])
        print(f"    filt 升序: {[(s, round(filt[s],4)) for s in order_filt]}")
        print(f"    e4   升序: {[(s, round(d[s],4)) for s in order_e4]}")
        print(f"    判据 A  55555 在 e4 上是组内最低? "
              f"{'✅ 是' if order_e4[0]==55555 else '❌ 否'}   "
              f"为负? {'✅ 是' if d[55555]<0 else '❌ 否'}")
        print(f"    判据 B  e4 最小值落在 filt 最小值或次小值上? "
              f"{'✅' if order_e4[0] in order_filt[:2] else '❌'}"
              f"   (e4 最小 = {order_e4[0]}; filt 前二 = {order_filt[:2]})")
        print(f"    判据 C  均值 {mean:+.4f} vs 线 {CRIT:.4f} ⇒ "
              f"{'**线上**' if mean > CRIT else '**线下**'}")

    if CRIT6 is not None:
        side7, side6 = mean > CRIT, mean > CRIT6
        print(f"\n    ── 判据 C 的循环性 (封存程序第 2/3 条) ──")
        print(f"    主判 CRIT₇ = {CRIT:.4f}  (完整 {n} 种子, SE_chunk={CRIT/2:.4f})"
              f"   均值 {mean:+.4f} ⇒ {'线上' if side7 else '线下'}")
        print(f"    附注 CRIT₆ = {CRIT6:.4f}  (不含 55555, SE_chunk={CRIT6/2:.4f})"
              f"   均值 {mean:+.4f} ⇒ {'线上' if side6 else '线下'}")
        print(f"    (历史口径 5 种子线 0.0196、6 种子线 0.0185 —— "
              f"线的漂移与余量同量级, 所以这条附注不是形式)")
        if side7 != side6:
            print(f"    🔴 **两口径指向不同侧 ⇒ 照实写「C 不可判」**, 说明是线的口径造成的;")
            print(f"       **不得**挑对结论有利的那条报。")
        else:
            print(f"    ✅ 两口径同侧 ⇒ C 可判 ({'线上' if side7 else '线下'})。")

    if 55555 in d:
        print(f"    55555 实测 ΔF1 = {d[55555]:+.4f}   "
              f"(封存外推 {FROZEN['extrap_55555_frozen']:+.4f} / "
              f"当前外推 {FROZEN['extrap_55555_now']:+.4f})")

    # ---------- 落盘前钉死的两条阈值, 换成可直接核对的 m ----------
    #   两条判据都只通过 m = mixnorm_55555 的 F1 起作用, 所以按 m 判读最透明。
    print(f"\n  ── 🔒 落盘前钉死的两条阈值 (02:48, 见 §7.4f) ──")
    if 55555 in d:
        idx = range(len(cids))
        o5 = f1_of(V[("own10", 55555)], idx)
        m5 = f1_of(V[("mixnorm", 55555)], idx)
        # 两条阈值都由「Σ其他种子 + own10_55555_F1 − 7×线」现算 —— 不抄常数
        s6 = dv.sum() - d[55555]          # 其余 6 个种子之和
        base = s6 + o5                    # = Σ全部 7 个 ΔF1 + m  (即 m=0 时的 7 种子和)
        band = base - 7 * FROZEN["delta_ub"]
        crit_lo = base - 7 * CRIT
        print(f"    m = mixnorm_55555 的 F1 = {m5:.4f}   "
              f"(own10_55555 = {o5:.4f}, 硬上界 ΔF1 ≤ {o5:.4f})")
        print(f"    ① §5.5 上界: 未突破 ⟺ m ≥ base − 7×{FROZEN['delta_ub']:.4f}"
              f" = {band:.4f}")
        print(f"       m = {m5:.4f} ⇒ " + (
              "✅ 未被突破 —— 但**只是回到预测, 不是验证**"
              if m5 >= band else
              "🔴 **上界被突破 —— 预登记的干净 miss, 方向与预期相反**"))
        print(f"    ② 判据 C: 线上 ⟺ m ≤ base − 7×CRIT = {crit_lo:.4f}")
        print(f"       m = {m5:.4f} ⇒ {'**线上**' if m5 <= crit_lo else '**线下**'}")
        print(f"    (base = Σ其他6种子ΔF1 + own10_55555_F1 = {base:.4f}, **现算**;"
              f" 换种子集这个和就变, 所以不抄常数)")

    # ---------- #71: 边界散 ≠ F1 散 (另一个判据, 别和 §5.9 混) ----------
    print(f"\n  ── #71 旧集 ΔF1 vs e4 ΔF1 的跨种子相关 (n={n}) ──")
    both = [s for s in d if s in filt]
    if len(both) >= 3:
        r = float(np.corrcoef([filt[s] for s in both], [d[s] for s in both])[0, 1])
        rc = r_crit(len(both))
        print(f"    n={len(both)}  r = {r:+.3f}   临界 |r| = {rc:.4f} (现算, 非抄写)")
        print(f"    ⇒ {'达到显著 —— 旧集上的排序能外推到 e4' if abs(r) > rc else '未达显著 —— 旧集排序**不能**外推'}")
        print(f"    ⚠️ 临界值随 n 变 (n=5→0.878, n=7→0.7545); 抄错方向 = 判据悄悄变严。")
        print(f"    ⚠️ 这里用的是**跨集脚本的同一批产物**, 两处 r 必须一致; 不一致就是有一处口径歪了。")
    else:
        print(f"    (可比种子不足 3 个, 跳过)")


if __name__ == "__main__":
    main()
