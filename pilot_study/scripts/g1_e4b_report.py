"""k=4b 加块 (1200 -> 1600) 的报告: 直接检验 §5.8d 的外推, 并量批间漂移。

## 这个脚本要回答的两件事

**1. §5.8d 的外推准不准。**
§5.8d 在 e4 **集内**量出 `SE_chunk ∝ k^-0.488`, 反解出「≈1627 块 ⇒ 天花板
t = 2.5、只需 4.9 个种子」。那是**外推** (1200 → 1627 超出了量过的范围),
三条限制里排第一。现在真有 1600 块了, 可以直接对账:
`SE_chunk(1600)` 是落在预测附近, 还是又一次落空 (§7.5 就是落空的)。

**2. 「加块是不是免费的」—— 批间漂移 (§5.8d 限制 ②)。**
§5.8d 只算了「在**这批块的总体里**把精度做到多少」, 明确标注**不含**
「换一批块 ΔF1 会漂多少」。现在新增的 400 块是同一总体、同一程序的**下一次
轮转抽样**, 于是可以直接量:
    漂移 = (新 400 块上的 ΔF1) − (旧 1200 块上的 ΔF1)
两者**不是独立样本** (同一批 7 个模型、同一批 100 个会话), 所以:
- 逐种子算, 得到 `drift[s]` —— 跨种子的散布 (`SE_seed`) 只含**模型**那一层;
- 块抽样那一层用 `boot_se` 分别算两个子集再合成 (块集不相交 ⇒ 方差可加)。
**读法**: 若 `mean(drift)` 大而 `SE_seed(drift)` 小 ⇒ 这是**所有种子一致位移**,
即块组成效应 (真漂移), 不是模型噪声。这正是本项目反复区分的那件事:
**块级散布会被平均掉, 一致位移不会。**

## 自检

`paired("e4x", only=<旧 1200 块的 id>)` 必须**逐位复现** §5.8b:
ΔF1 = +0.0210, SE_chunk = 0.009725, SE_seed = 0.0053。
复现不了说明 merge 动过逐块结果 —— 那么 1600 块的数一个都不能信。

用法 (纯读盘, 0 GPU):
  python scripts/g1_e4b_report.py [--boot 2000]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ⚠️ 一律 import, 不重写 (ΔF1 / SE_chunk / 临界线 只此一处定义)
from g1_e4_analyze import (ANNOT, ARMS, SEEDS, boot_se, boot_se_mc, f1_of,  # noqa: E402
                           paired)

# §5.8b 的既有结论 (e4 / 1200 块), 用于自检与对照
E4 = {"delta": 0.0210, "se_seed": 0.0053, "se_chunk": 0.009725}
# §5.8d 对 k=1600 的**外推预测** —— 注意当时报的是区间 1627/1615/1588 块
PRED = {"k": 1600, "ceil_t": 2.5, "n_req": 4.9}


def ids_of(name):
    return [l.strip() for l in open(f"{ANNOT}/{name}") if l.strip()]


def stats(D, label, boot):
    """一个评估集的核心量, 全部走唯一口径。"""
    vec, cids, seeds = D["vec"], D["cids"], D["seeds"]
    dv = np.array([D["d"][s] for s in seeds])
    delta, sd = float(dv.mean()), float(dv.std(ddof=1))
    se_seed = sd / np.sqrt(len(dv))
    se_chunk = boot_se(vec, cids, seeds, boot)
    return {"label": label, "delta": delta, "sd": sd, "se_seed": se_seed,
            "se_chunk": se_chunk, "hyp": float(np.hypot(se_seed, se_chunk)),
            "k": len(cids), "d": D["d"], "seeds": seeds}


def row(r):
    ceil_t = r["delta"] / r["se_chunk"]
    crit = 2 * r["se_chunk"]
    need = (r["delta"] / 2) ** 2 - r["se_chunk"] ** 2
    n_req = (r["sd"] ** 2 / need) if need > 0 else float("nan")
    return (f"  {r['label']:<10}{r['k']:>6}{r['delta']:>+10.4f}{r['sd']:>10.4f}"
            f"{r['se_seed']:>10.4f}{r['se_chunk']:>11.6f}{r['hyp']:>10.4f}"
            f"{ceil_t:>9.2f}{n_req:>9.1f}  {'线上' if r['delta'] > crit else '线下'}")


HEAD = (f"  {'集':<10}{'块数':>6}{'ΔF1':>10}{'SD':>10}{'SE_seed':>10}"
        f"{'SE_chunk':>11}{'合并SE':>10}{'天花板t':>9}{'所需n':>9}  判据")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    boot = args.boot

    ids_e4 = ids_of("g1_eval4_ids.txt")
    ids_e4b = ids_of("g1_eval4b_ids.txt")
    ids_all = ids_of("g1_eval4x_ids.txt")

    print("=" * 96)
    print("  k=4b 加块报告: 1200 -> 1600 块, 0 重训")
    print("=" * 96)

    # ---------- 自检: 1200 子集必须逐位复现 §5.8b ----------
    print("\n== 自检: e4x 里只取旧 1200 块, 必须逐位复现 §5.8b ==")
    D_e4 = paired("e4x", only=ids_e4)
    if D_e4 is None:
        print("  🔴 产物不齐, 退出"); return 1
    S_e4 = stats(D_e4, "e4(旧1200)", boot)
    ok = True
    for nm, got, want, tol in (
            ("ΔF1", S_e4["delta"], E4["delta"], 0.001),
            ("SE_seed", S_e4["se_seed"], E4["se_seed"], 0.001),
            ("SE_chunk", S_e4["se_chunk"], E4["se_chunk"], 0.0005)):
        good = abs(got - want) <= tol
        ok &= good
        print(f"    {nm:<9} 得到 {got:+.6f}  期望 {want:+.6f}  "
              f"{'✅' if good else '❌ 差 ' + format(got-want, '+.6f')}")
    if not ok:
        print("  🔴 自检失败 —— merge 动过逐块结果, 下面的数一个都不能信。")
        return 1
    print("  ✅ 1200 子集逐位复现 §5.8b ⇒ merge 是保值的, 1600 块可信")

    # ---------- 三个集摆在一起 ----------
    print(f"\n== 1200 / 1600 / 新增 400 三集对照 (bootstrap {boot} 次) ==")
    D_x = paired("e4x")
    D_b = paired("e4x", only=ids_e4b)
    if D_x is None or D_b is None:
        print("  🔴 产物不齐, 退出"); return 1
    S_x = stats(D_x, "e4x(1600)", boot)
    S_b = stats(D_b, "e4b(新400)", boot)
    print(HEAD)
    print(row(S_e4)); print(row(S_x)); print(row(S_b))
    print(f"\n  ⚠️ 「所需n」列在 SE_chunk 逼近 ΔF1/2 时病态 (分母是两个大数的小差),")
    print(f"     只有**块数**那一列是稳的 —— 见 §5.8b/§5.8c。")

    # ---------- 对账: §5.8d 的外推 ----------
    print(f"\n== 对账 §5.8d 的外推 (k=1600) ==")
    pred_se = E4["delta"] / PRED["ceil_t"]
    print(f"  预测 (§5.8d, 按集内指数 −0.488 外推): "
          f"SE_chunk ≈ {E4['se_chunk']:.6f} × (1200/1600)^0.488 = {pred_se:.6f}")
    print(f"         ⇒ 天花板 t ≈ {PRED['ceil_t']}, 所需种子 ≈ {PRED['n_req']}")
    print(f"  实测: SE_chunk = {S_x['se_chunk']:.6f}, "
          f"天花板 t = {S_x['delta']/S_x['se_chunk']:.3f}")
    ratio = S_x["se_chunk"] / S_e4["se_chunk"]
    pred_ratio = (1200 / 1600) ** 0.488
    print(f"  收缩比: 实测 ×{ratio:.4f} vs 预测 ×{pred_ratio:.4f} "
          f"({'✅ 命中' if abs(ratio/pred_ratio - 1) < 0.10 else '❌ 差 ' + format(ratio/pred_ratio-1, '+.1%')})")
    # 实测的集内指数: SE ∝ k^a
    a_meas = float(np.log(ratio) / np.log(1200 / 1600))
    print(f"  从 1200→1600 实测的指数 = {a_meas:+.3f}   "
          f"(§5.8d 集内量的是 −0.488; 1/√k 是 −0.500)")
    print(f"  ⚠️ 只有**两个点** (1200, 1600), 两点定不出指数 —— 上面这个 a_meas")
    print(f"     是把它当幂律时的隐含值, 精度由 SE_chunk 的 MC 决定 (见末尾)。")

    # ---------- 批间漂移: §5.8d 限制 ② ----------
    print(f"\n== 批间漂移: 「换一批块, ΔF1 会漂多少」(§5.8d 限制 ②) ==")
    print(f"  {'seed':>7}{'旧1200 ΔF1':>14}{'新400 ΔF1':>14}{'漂移':>12}")
    dr = []
    for s in S_e4["seeds"]:
        a, b = S_e4["d"][s], S_b["d"][s]
        dr.append(b - a)
        print(f"  {s:>7}{a:>+14.4f}{b:>+14.4f}{b-a:>+12.4f}")
    dr = np.array(dr)
    m_dr, sd_dr = float(dr.mean()), float(dr.std(ddof=1))
    se_seed_dr = sd_dr / np.sqrt(len(dr))
    # 块抽样那一层: 两个块集不相交 ⇒ 方差可加 (各自用唯一口径的 boot_se)
    se_chunk_dr = float(np.hypot(S_e4["se_chunk"], S_b["se_chunk"]))
    se_dr = float(np.hypot(se_seed_dr, se_chunk_dr))
    print(f"\n  漂移 = {m_dr:+.4f} ± {se_dr:.4f} (合并 SE)")
    print(f"    SE_seed  = {se_seed_dr:.4f}  (跨种子; 只含**模型**那一层)")
    print(f"    SE_chunk = {se_chunk_dr:.4f}  (= hypot(两子集的 SE_chunk); 块集不相交)")
    neg = sum(1 for x in dr if x < 0)
    print(f"    逐种子符号: {len(dr)-neg} 正 / {neg} 负")
    _t = m_dr / se_dr if se_dr else float("nan")
    print(f"    t = {m_dr:+.4f} / {se_dr:.4f} = {_t:+.2f}"
          f"   {'(漂移不显著)' if abs(_t) < 2 else '(漂移显著!)'}")
    print(f"\n  🔑 读法: 若 |均值| 相对 SE_seed **很大**而逐种子符号一致 ⇒ 这是")
    print(f"     **所有种子共同的位移**, 即块组成效应 (真漂移), 不是模型噪声。")
    print(f"     「块级散布会被平均掉, 一致位移不会」—— 这正是加块的隐含代价:")
    print(f"     它买到的是**精度**(SE 更小), 但同时也换了一个**可能不同的估计值**。")
    print(f"  ⚠️ 两个子集覆盖的是**同一批 100 个会话** (旧 12 块/会话, 新 4 块/会话),")
    print(f"     若存在会话级效应, 两者的块噪声并不独立, 上面 hypot 会**低估**漂移的 SE。")

    # ---------- 跨子集相关: ΔF1 是 run 的性质, 还是评估集的噪声? ----------
    # 这正是 §5.9 的问题, 而这里有一个**比 §5.9 更干净**的版本: 两个块集
    # **完全不相交** (旧 1200 vs 新 400), 所以"相关"不是靠共享块堆出来的。
    print(f"\n== 跨子集相关 (ΔF1 是 run 的性质, 还是评估集的噪声?) ==")
    print(f"  {'seed':>7}{'旧1200':>12}{'新400':>12}")
    xs = np.array([S_e4["d"][s] for s in S_e4["seeds"]])
    ys = np.array([S_b["d"][s] for s in S_b["seeds"]])
    for i, s in enumerate(S_e4["seeds"]):
        print(f"  {s:>7}{xs[i]:>+12.4f}{ys[i]:>+12.4f}")
    n = len(xs)
    r = float(np.corrcoef(xs, ys)[0, 1])
    # 🔴 n=7 的临界 r —— 本项目踩过「n=7 配 n=5 的临界」「相关系数靠单点」
    R_CRIT = 0.7545          # n=7, p=0.05 双尾
    # 留一: 一个点撑起来的 r 会在这里塌掉
    loo = []
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        loo.append(float(np.corrcoef(xs[m], ys[m])[0, 1]))
    loo = np.array(loo)
    print(f"\n  r = {r:+.3f}   (n={n}, 临界 {R_CRIT} ⇒ "
          f"{'✅ 显著' if abs(r) > R_CRIT else '❌ 未达显著'})")
    print(f"  留一: [{loo.min():+.3f}, {loo.max():+.3f}], 极差 {loo.max()-loo.min():.3f}"
          f"   {'✅ 不靠单点' if loo.min() > 0 and (loo.max()-loo.min()) < 0.35 else '🔴 靠单点/会翻号'}")
    print(f"  ⚠️ n=7 本身是硬约束: 临界 0.7545 很高, 且留一极差大说明这个 r 的")
    print(f"     分辨率有限 —— **别把 r 的大小读成效应强度**, 只读方向。")

    # ---------- 流行率 (本项目踩过: 报方差前先报正例率) ----------
    print(f"\n== 流行率 (报方差前先报正例率 —— 它是能翻转结论的量) ==")
    for nm, D in (("e4(旧1200)", D_e4), ("e4b(新400)", D_b), ("e4x(1600)", D_x)):
        e0 = D["P"][("own10", D["seeds"][0])]
        gt = sum(e0["per_chunk"][c]["n_gt"] for c in D["cids"])
        pos = sum(1 for c in D["cids"] if e0["per_chunk"][c]["n_gt"] > 0)
        print(f"  {nm:<12} 块 {len(D['cids']):>5}  事件 {gt:>4}  "
              f"含事件块 {pos:>4} ({pos/len(D['cids']):>6.2%})  "
              f"事件/块 {gt/len(D['cids']):>6.3%}")

    # ---------- MC: 上面每个 SE 都是抽出来的 ----------
    print(f"\n== 蒙特卡洛误差 (§5.8c: 别给随机估计量一个它没有的精度) ==")
    for nm, D, S in (("e4(旧1200)", D_e4, S_e4), ("e4x(1600)", D_x, S_x)):
        _p, mc, lo, hi = boot_se_mc(D["vec"], D["cids"], D["seeds"], boot, reps=6)
        crit = 2 * S["se_chunk"]
        # 余量相对 MC: 判决 (ΔF1 vs 2×SE_chunk) 稳不稳
        _r = abs(S["delta"] - crit) / (2 * mc) if mc else float("inf")
        print(f"  {nm:<12} SE_chunk = {_p:.6f} ± {mc:.6f} (MC {mc/_p:.1%})  "
              f"范围 [{lo:.6f}, {hi:.6f}]")
        print(f"  {'':<12} ⇒ 线在 [{2*lo:.4f}, {2*hi:.4f}] 漂; "
              f"余量 {abs(S['delta']-crit):.4f} / MC {2*mc:.4f} = **{_r:.1f}×**"
              f" {'✅ 扛得住' if _r > 3 else '🔴 会被 MC 掀翻'}")
    print(f"\n  ⚠️ 加块同时也把 MC 压小了 (SE_chunk 的 MC 按 1/√boot, 但**点估计**")
    print(f"     本身更小); 读判决时要连这一层一起看, 不能只看点估计的第四位小数。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
