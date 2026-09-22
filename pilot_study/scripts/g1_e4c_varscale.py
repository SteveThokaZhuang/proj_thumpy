"""方差分量的**标度**: 那 49.4% 的「种子×批」交互是**真交互**还是 400 块的**测量噪声**?

> 🔴 **2026-09-19 后记: 这个脚本的 p̂ 判读已被 `g1_e4c_se_scaling.py` 取代。**
> 它的结论有一半是**估计噪声的产物**:
> ① 三档 k 的 df_b 是 23/11/5, 而 `var_a = (MS_b − MS_r)/ns` 是两个 χ² 之差,
>    相对 SD 在 df_b=5 时约 60% ⇒ p̂ 的误差棒宽到判读不了 (本文件那版 SE 公式
>    更差, 印出 ±3.19 —— **那个数不可用**)。
> ② 真正干净的测法是**不放回抽轮次直接量 SE(N)** (见 se_scaling), 它给:
>    σ̂ = 0.0201, 三档独立反解一致到 6.5% ⇒ **轮可交换、无成簇**,
>    「var_a 缩不动」只是 df_b=5 的噪声。
> **留此文件是因为它给出了方差分量本身 (三项占比), 那部分仍然有效**;
> 但**不要**引用它的 p̂。


## 为什么这个问句决定下一步

`g1_e4c_pool_sd_ext.py` 的两向分解给出 (k=400): 批间 34.6% / 种子间 16.0% /
残差 **49.4%**。两种解释, 后果完全相反:

- **(甲) 残差 = 测量噪声** ⇒ 它该按 `1/k` 缩。那么 k 一大就只剩种子那一项,
  地板 = `√(σ_c²/ns)` 成立, **「加种子 + 加块」这条路走得通**, 只是要花 GPU。
- **(乙) 残差 = 真实的种子×批交互** ⇒ 它**不缩**。那么「这个 run 的 ΔF1 是多少」
  **不是一个稳定的性质** —— 换个种子、换批块就换个答案, 而两条路都缩不掉它。
  ⇒ 该换的是**指标和问题**, 不是样本量。

判别法很便宜: **同一个集上把 k 翻倍, 看 `var_e` 掉不掉一半。**
只在 **e4c 内部**做 (4800 块 / 24 轮)—— e4c 的外层是 `(arm,seed)`、内层 batch,
批身份不与评估时刻混杂, 是四段里唯一干净的; e4/e4b 并进来会引入段间时刻差。

⚠️ 只用 4800 / 2400 两档: 再往上 (1600) 只剩 3 组, df_b=2, 分解没有意义。

用法 (srun 内, fd_analysis): python scripts/g1_e4c_varscale.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from g1_e4_analyze import paired, d_of  # noqa: E402
from g1_pool_null_probe import key_of  # noqa: E402

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
K = 200
B_BATCH = 400


def twoway(PS):
    """两向随机效应分解。PS[i, s] = 第 i 组的第 s 个种子的 ΔF1。

    返回 (var_batch, var_seed, var_resid, grand, se_full, n_groups, n_seeds)。
    自检: σ_a² + σ_e²/ns 必须 == 实测的 Var(组均值), 否则模型写错。
    """
    J, ns = PS.shape
    grand = PS.mean()
    ss_s = J * ((PS.mean(axis=0) - grand) ** 2).sum(); df_s = ns - 1
    ss_b = ns * ((PS.mean(axis=1) - grand) ** 2).sum(); df_b = J - 1
    ss_t = ((PS - grand) ** 2).sum(); df_r = (J - 1) * (ns - 1)
    ms_s, ms_b, ms_r = ss_s / df_s, ss_b / df_b, (ss_t - ss_s - ss_b) / df_r
    var_a = max(ms_b - ms_r, 0.0) / ns
    var_c = max(ms_s - ms_r, 0.0) / J
    var_e = ms_r
    # 自检: 组均值的实测方差 vs 模型预测
    obs = float(PS.mean(axis=1).var(ddof=1))
    pred = var_a + var_e / ns
    assert abs(obs - pred) < 0.05 * max(obs, pred), \
        (f"分解自检失败: 实测 Var(组均值)={obs:.3e} vs 预测 {pred:.3e} "
         f"⇒ 模型不对, 停止判读")
    se_full = float(np.sqrt(var_a / J + var_c / ns + var_e / (J * ns)))
    return var_a, var_c, var_e, float(grand), se_full, J, ns


def main():
    # ---- 合并池 (只用 e4c 的 12 批 = 4800 块, 24 轮) ----
    Ds = []
    for j in range(12):
        D = paired(f"e4c{j:02d}")
        assert D is not None and not D["missing"], f"e4c 批 {j} 不完整"
        Ds.append((j, D))
    seeds = list(Ds[0][1]["seeds"])
    cids_p, vec_p = [], None
    for j, D in Ds:
        order = [l.strip() for l in
                 open(f"{ANNOT}/g1_eval4c_ids_b{j:02d}_ids.txt") if l.strip()]
        pos = {c: i for i, c in enumerate(D["cids"])}
        if vec_p is None:
            vec_p = {k: [] for k in D["vec"]}
        cids_p.extend(order)
        for k in vec_p:
            vec_p[k].extend(D["vec"][k][pos[c]] for c in order)
    n_pool = len(cids_p)
    assert n_pool == 4800, f"合并池 {n_pool} 块, 应为 4800"
    print(f"e4c 合并池: {n_pool} 块 / {len({key_of(c) for c in cids_p})} 个 key")

    pos_p = {c: i for i, c in enumerate(cids_p)}
    rounds = []
    for j, D in Ds:
        order = [l.strip() for l in
                 open(f"{ANNOT}/g1_eval4c_ids_b{j:02d}_ids.txt") if l.strip()]
        for r in range(len(order) // K):
            chunk = order[r * K:(r + 1) * K]
            assert len({key_of(c) for c in chunk}) == K, f"批 {j} 轮 {r} 序错"
            rounds.append([pos_p[c] for c in chunk])
    # k=200 (24 组, df_b=23) 是关键的一档: var_a 在 k=800 上 df_b 只有 5, 测不准。
    # 加上它才能把「var_a 不缩」到底是**估计噪声**还是**真成簇**分开。
    gsize_of = {200: 1, 400: 2, 800: 4}
    print(f"轮结构: {len(rounds)} 轮 × {K} 块\n")

    print(f"{'k':>6}{'组数':>6}{'df_b':>6}"
          f"{'var_a':>12}{'var_c':>12}{'var_e':>12}")
    print(f"{'':>6}{'':>6}{'':>6}{'σ_a':>12}{'σ_c':>12}{'σ_e':>12}")
    out = {}
    PSmap = {}
    for k, gsize in gsize_of.items():
        groups = [rounds[i:i + gsize] for i in range(0, len(rounds), gsize)]
        # PS[i, s] = 第 i 组第 s 个种子的 ΔF1 (在**并集**上算一次, 见 d_of 文档)
        PS = np.array([[d_of(vec_p, [s], [i for r in g for i in r])[s]
                        for s in seeds] for g in groups])
        PSmap[k] = PS
        var_a, var_c, var_e, grand, se_full, J, ns = twoway(PS)
        print(f"{k:>6}{J:>6}{J-1:>6}{var_a:>12.3e}{var_c:>12.3e}{var_e:>12.3e}")
        print(f"{'':>6}{'':>6}{'':>6}{np.sqrt(var_a):>12.5f}"
              f"{np.sqrt(var_c):>12.5f}{np.sqrt(var_e):>12.5f}")
        print(f"{'':>18}ΔF1={grand:+.4f}  SE={se_full:.5f}  "
              f"t={grand/se_full:.2f}  占比 批{var_a/(var_a+var_c+var_e):.1%}"
              f"/种{var_c/(var_a+var_c+var_e):.1%}"
              f"/残{var_e/(var_a+var_c+var_e):.1%}")
        out[k] = {"var_a": var_a, "var_c": var_c, "var_e": var_e,
                  "grand": grand, "se": se_full, "J": J, "df_b": J - 1,
                  "ns": ns, "df_r": (J - 1) * (ns - 1)}

    # ---- 标度指数: var(k) = A·k^-p。纯抽样 ⇒ p=1; 真簇/真交互 ⇒ p=0 ----
    ks = sorted(out)
    print(f"\n=== 标度指数 p (var ∝ k^-p; 纯抽样噪声 p=1, 真效应 p=0) ===")
    print(f"{'分量':<8}" + "".join(f"{'A(k='+str(k)+')':>14}" for k in ks)
          + f"{'p̂':>9}{'SE(p̂)':>9}")
    phat = {}
    for nm in ("var_a", "var_c", "var_e"):
        Ak = [out[k][nm] * k for k in ks]
        x = np.log(ks); y = np.log([out[k][nm] for k in ks])
        p = -np.polyfit(x, y, 1)[0]
        # p 的 SE: 由各点 df 粗估 (MS 的相对 SD = √(2/df))
        dfs = [out[k]["df_b"] if nm == "var_a" else
               (out[k]["J"] - 1) * (out[k]["ns"] - 1) if nm == "var_e" else
               out[k]["ns"] - 1 for k in ks]
        w = np.array([1.0 / np.sqrt(2.0 / d) for d in dfs])   # 权 = 1/相对SD
        p_se = float(np.sqrt((w ** 2).sum()) / abs(x[-1] - x[0]))  # 粗估
        phat[nm] = (p, p_se, Ak)
        print(f"{nm:<8}" + "".join(f"{a:>14.4f}" for a in Ak)
              + f"{p:>9.2f}{p_se:>9.2f}")
    print(f"  ⚠️ `A(k)` 那一列**应当恒定** (那是 p=1 的定义); 它单调涨 ⇒ p<1 ⇒ 有真成分。")
    print(f"  ⚠️ SE(p̂) 是粗估 (各点 df 差得远), 只用来看「p̂ 离 1 有几个 SE」。")

    pa, pa_se = phat["var_a"][0], phat["var_a"][1]
    pc, pc_se = phat["var_c"][0], phat["var_c"][1]
    pe_, pe_se = phat["var_e"][0], phat["var_e"][1]

    print(f"\n=== 结论 ===")
    print(f"  ① **var_c 的 p̂ = {pc:.2f} ± {pc_se:.2f}** ⇒ 与 p=0 (真效应、缩不掉) 相容"
          f" ⇒ 种子那一路是真的。")
    ve_ok = abs(pe_ - 1) < 2 * pe_se
    va_ok = abs(pa - 1) < 2 * pa_se
    print(f"  ② **var_e 的 p̂ = {pe_:.2f} ± {pe_se:.2f}** ⇒ "
          f"{'与 p=1 (纯测量噪声) 相容 ✅ ⇒ 那 49.4% 的「种子×批」大部分是 400 块的噪声, 不是真交互' if ve_ok else '⚠️ 偏离 p=1, 有真交互成分'}")
    print(f"     ⇒ 「ΔF1 是 run 的稳定性质」比 §2.7 的占比看起来**更站得住**。")
    print(f"  ③ **var_a 的 p̂ = {pa:.2f} ± {pa_se:.2f}** ⇒ "
          f"{'与 p=1 (纯抽样噪声) 相容 ✅ ⇒ **不是成簇**, 之前 0.87 那个比值是 df_b=5 的噪声' if va_ok else '🔴 **显著小于 1 ⇒ 块成簇** (相邻轮不是独立信息)'}")
    print(f"     ⚠️ 这一档是靠 **k=200 的 df_b=23** 撑起来的 —— 只看 400→800 (df_b 11→5)"
          f" 会误判。")
    print(f"\n  ⇒ 设计含义 (等 p̂ 定了才敢写):")
    if va_ok:
        print(f"     var_a 与 var_e **都按 1/k 缩** ⇒ `SE² = C/N + var_c/ns` 成立,")
        print(f"     加块**确实**降 SE, 直到撞上地板 √(var_c/ns) = "
              f"{np.sqrt(out[800]['var_c']/out[800]['ns']):.5f}。")
    else:
        print(f"     var_a **不缩** ⇒ 加块降 SE 的效果有额外上限,")
        print(f"     §2.8 的 `SE² = C/N + var_c/ns` 这个式子**不成立**, 得重推。")
        print(f"     ⇒ 开跑前先查清「块为什么成簇」(会话/说话人/时间), 别急着加块。")

    json.dump({str(k): v for k, v in out.items()},
              open(f"{ANNOT}/g1_e4c_varscale.json", "w"), indent=1)
    print(f"\n-> {ANNOT}/g1_e4c_varscale.json")


if __name__ == "__main__":
    main()
