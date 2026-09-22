"""ΔF1 从 filt 掉到 e4 —— 这个**收缩本身**显著吗? (0 GPU)

## 为什么需要这个脚本

§5.5 的整套叙事是「自然流行率把 ΔF1 从 +0.0340 压到 +0.0211」。
`§5.5b` 只对了「预测 vs 实测」这一件事: 预测 +0.0211、实测 +0.0210 ⇒ 命中。
**但"命中"回答的是"预测准不准", 不是"收缩存不存在"。**

两者是不同的问句, 而只有后者是机制主张。一个预测可以在**什么效应都没有**的
情况下命中 (只要基线够稳)。所以收缩必须**另外检验一次**。

## 检验设计

两套评估集评的是**同一批 7 个 adapter** ⇒ 天然配对, 用**每种子自己与自己比**,
消掉 run 层面的选择偏差 (§5.6 那条 🔴 警告要求的正是这个)。

    Δ_s = ΔF1_e4(s) − ΔF1_filt(s)

两个误差源都要算, 且**必须分开报**:
  · 种子侧: SD(Δ)/√7 —— 以种子为抽样单位
  · 块侧: 两套块集不同 ⇒ 块抽样误差**近似独立**, 按 hypot 相加。
    ⚠️ 这个"独立"要用实测的块重叠率来支持, 不能假设 —— 脚本会先打出来。

## 与 §5.5b 的关系

§5.5b 说「ΔF1 命中」; 本脚本说「**收缩**在什么置信水平上站得住」。
如果收缩不显著, §5.5b 的"命中"就要改写成
「实测落在预测上, 但**这个预测所描述的那个收缩本身并未被这套数据确立**」。

用法: python scripts/g1_shrink_test.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ⚠️ 一律 **import**, 不重写 (本项目约定: 判据/口径类只此一处定义)
from g1_e4_analyze import paired, boot_se, SEEDS  # noqa: E402

SETS = ["filt", "e4"]


def main():
    print("=" * 76)
    print("  ΔF1 的收缩 (filt → e4) 本身显著吗?")
    print("=" * 76)

    D = {}
    for t in SETS:
        D[t] = paired(t)
        if D[t] is None:
            print(f"  🔴 {t} 产物不齐, 退出。")
            return
        n = len(D[t]["seeds"])
        print(f"  {t:<5} 配对 {n}/7   共同块 {len(D[t]['cids'])}")
        if n != len(SEEDS):
            print(f"  🔴 {t} 只有 {n} 个配对种子 —— 下面的数不是最终统计量, 只当自检。")

    seeds = [s for s in SEEDS if s in D["filt"]["d"] and s in D["e4"]["d"]]

    # ---------- 自检 0: 两套必须各自复现已知值 ----------
    print("\n  ── 自检 0: 两套 ΔF1 应复现各自文档里的值 ──")
    EXP = {"filt": 0.0340, "e4": 0.0210}
    ok = True
    for t in SETS:
        v = np.array([D[t]["d"][s] for s in seeds])
        good = abs(v.mean() - EXP[t]) < 5e-4
        ok &= good
        print(f"    {t:<5} ΔF1 = {v.mean():+.4f}  (文档 {EXP[t]:+.4f})  "
              f"{'✅' if good else '🔴 对不上 —— 停, 先查口径'}")
    if not ok:
        return

    # ---------- 块重叠: 「块侧独立」这个假设的实测支持 ----------
    cf, ce = set(D["filt"]["cids"]), set(D["e4"]["cids"])
    ov = len(cf & ce)
    print(f"\n  ── 块集关系 (决定块侧误差能不能按独立相加) ──")
    print(f"    filt {len(cf)} 块 / e4 {len(ce)} 块 / **重叠 {ov}**"
          f"   ({ov/len(cf):.1%} of filt, {ov/len(ce):.1%} of e4)")
    indep = ov == 0
    print(f"    ⇒ 块侧独立性: {'✅ 成立' if indep else '⚠️ **不成立** —— 重叠部分有共同抽样误差, hypot 相加会**高估**总误差'}")
    if not indep:
        print(f"       (重叠率不高时偏差有限, 但结论要按「块侧独立」是近似来措辞)")

    # ---------- 配对差 ----------
    print(f"\n  ── 逐种子配对差 Δ = e4 − filt ──")
    print(f"    {'seed':>7}{'filt':>10}{'e4':>10}{'Δ':>10}")
    dl = []
    for s in seeds:
        a, b = D["filt"]["d"][s], D["e4"]["d"][s]
        dl.append(b - a)
        print(f"    {s:>7}{a:>+10.4f}{b:>+10.4f}{b-a:>+10.4f}")
    dl = np.array(dl)
    n = len(dl)
    mean = dl.mean()
    sd = dl.std(ddof=1)
    se_seed = sd / np.sqrt(n)
    n_neg = int((dl < 0).sum())

    # 块侧: 两套各自的 SE_chunk, 按独立相加
    se_c = {t: boot_se(D[t]["vec"], D[t]["cids"], D[t]["seeds"]) for t in SETS}
    se_chunk_diff = float(np.hypot(se_c["filt"], se_c["e4"]))

    print(f"\n  ── 收缩的量级与误差 ──")
    print(f"    收缩 = {mean:+.4f}   (filt {D['filt']['cids'] and np.mean([D['filt']['d'][s] for s in seeds]):+.4f}"
          f" → e4 {np.mean([D['e4']['d'][s] for s in seeds]):+.4f})")
    print(f"    SD(Δ) = {sd:.4f}   SE_seed = {se_seed:.4f}   (n={n})")
    print(f"    SE_chunk(filt) = {se_c['filt']:.4f}   SE_chunk(e4) = {se_c['e4']:.4f}"
          f"   ⇒ 差值块侧 {se_chunk_diff:.4f}")
    print(f"    同向个数: {n_neg}/{n} 为负")

    from scipy import stats as st
    print(f"\n  ── 检验 (两侧) ──")
    t_seed = mean / se_seed
    p_seed = 2 * st.t.sf(abs(t_seed), n - 1)
    print(f"    (a) **只看种子** (忽略块噪声, 乐观):  t = {t_seed:+.2f}  p = {p_seed:.3f}")
    print(f"        95% CI = [{mean - st.t.ppf(0.975, n-1)*se_seed:+.4f}, "
          f"{mean + st.t.ppf(0.975, n-1)*se_seed:+.4f}]")
    se_all = float(np.hypot(se_seed, se_chunk_diff))
    t_all = mean / se_all
    p_all = 2 * st.t.sf(abs(t_all), n - 1)
    print(f"    (b) **种子 + 块**(块侧按{'' if indep else '近似'}独立):  t = {t_all:+.2f}  p = {p_all:.3f}"
          f"   合并 SE = {se_all:.4f}")
    print(f"        95% CI = [{mean - st.t.ppf(0.975, n-1)*se_all:+.4f}, "
          f"{mean + st.t.ppf(0.975, n-1)*se_all:+.4f}]")
    p_sign = 2 * st.binom.sf(n_neg - 1, n, 0.5) if n_neg >= n / 2 else 2 * st.binom.cdf(n_neg, n, 0.5)
    print(f"    (c) 符号检验 (丢幅度, 只看方向):  {n_neg}/{n} ⇒ p = {p_sign:.3f}")

    print(f"\n  ── 判读 ──")
    sig = p_all < 0.05
    print(f"    {'✅ **收缩成立**' if sig else '🔴 **收缩未被确立**'} "
          f"(主口径 = (b), 因为 (a) 漏掉了块这个随机源)")
    if not sig:
        print(f"    ⇒ 措辞必须是: 「实测**落在 §5.5 的预测上**, 但「流行率让 ΔF1 变小」")
        print(f"       这个**收缩本身**, 在现有 7 个种子下**并未被这套数据确立**」。")
        print(f"       两者是不同的话 —— 前者关于预测, 后者关于机制。")
    print(f"\n    ⚠️ n={n} 时 t 的分辨力本来就差; 这里报的是**能否确立**, 不是「有没有」。")
    print(f"       CI 的上端 {mean + st.t.ppf(0.975, n-1)*se_all:+.4f} "
          f"{'覆盖 0 ⇒ 不能排除「没有收缩」' if mean + st.t.ppf(0.975, n-1)*se_all > 0 else ''}")


if __name__ == "__main__":
    main()
