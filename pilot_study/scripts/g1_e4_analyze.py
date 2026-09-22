"""k=4 扩集的结果分析 —— 纯读盘, 0 GPU。

## 为什么先写

评估要跑 7.8 小时 (03:45 才完)。分析逻辑现在就写好并**用旧集自检**,
结果一落地就能出报告, 而不是那时候再写代码 —— 半夜写代码正是
本项目已经踩过坑的地方 (见 memory: 改完码重启)。

## 自检 (关键)

同一个脚本用 `--set filt` 跑, **必须复现 §7.2 的已知数字**:

    ΔF1 (own10−mixnorm) = +0.0340
    SE_seed             = 0.0117
    SE_chunk            = 0.0198
    t                   = 1.48

对不上说明分析代码本身有 bug, 那么 k=4 跑出来的数也不可信。
这是本项目反复用的一条纪律: **先让工具在已知答案上跑通**。

## 判据

- `SE_seed` 不随评估集增大而缩小 (它是模型的性质, 7 个种子评的是同一批块,
  块抽样在种子间被约掉了)
- `SE_chunk` 按 1/sqrt(k) 缩 —— 这才是扩集能压下去的那一项
- `t = ΔF1 / hypot(SE_seed, SE_chunk)`
- 预测 (§5): k=4 后 t = 2.42 ~ 2.66, 天花板 2.91

用法:
  python scripts/g1_e4_analyze.py --set filt     # 自检, 应复现 §7.2
  python scripts/g1_e4_analyze.py --set e4       # k=4 结果
"""
import argparse
import json
import os

import numpy as np

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
SEEDS = [42, 1234, 7, 2024, 3407, 31337, 55555]
ARMS = ["own10", "mixnorm"]
B = 2000

# 已知的 §7.2 基准, 用于自检
BASELINE = {"delta": 0.0340, "se_seed": 0.0117, "se_chunk": 0.0198, "t": 1.48}


def suffix(seed):
    return "" if seed == 42 else f"_s{seed}"


def load(tag_set, arm, seed):
    p = f"{ANNOT}/g1_eval_{arm}{suffix(seed)}_{tag_set}.json"
    if not os.path.exists(p):
        return None
    return next(iter(json.load(open(p)).values()))


def f1_of(pc, idx):
    tp = fp = gt = 0
    for i in idx:
        e = pc[i]
        tp += e["tp"]; fp += e["fp"]; gt += e["n_gt"]
    p = tp / max(1, tp + fp)
    r = tp / max(1, gt)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


# ============================================================================
# 以下两个函数是**唯一一处**定义「ΔF1 / 块共同集 / SE_chunk 怎么算」的地方。
# 其它脚本 (g1_required_n.py, g1_e4_reconcile.py) 一律 import 它们, **不得重写** ——
# 两份定义 = 迟早只改一份 (memory: hardcoded-conclusions-escape-reproduction,
# 同 §7.4f 里 r_crit 只留一处定义的做法)。
# main() 自己也走这两个函数, 这样「脚本印的数」与「别处 import 的数」
# **按构造**逐位一致, 不需要靠人眼核对。
# ============================================================================

def d_of(vec, seeds, idx):
    """逐种子 ΔF1 (own10 − mixnorm) 在块下标集合 `idx` 上 —— **唯一一处**定义它。

    `idx` 可以是 `range(n)` (全集)、bootstrap 的抽中下标、或某个会话的块下标。
    把定义收在这里, 是为了让「全集 ΔF1」「bootstrap 里的 ΔF1」「按会话切的 ΔF1」
    **按构造**同一口径 —— 复制一遍算式就是两份定义, 迟早只改一份
    (memory: hardcoded-conclusions-escape-reproduction)。

    ⚠️ 一个会误导人的**恒真**性质: 若 `idx` 里的块**一个真值事件都没有**,
    则 tp=fp=n_gt=0 ⇒ `f1_of` 对**两臂都返回 0.0** ⇒ ΔF1 **按定义为 0**。
    所以「某些会话 ΔF1 恰好是 0」**不是发现**, 是定义 —— 报分解时必须先
    把「无事件的会话」摘出去, 否则会把 54/100 个恒零当成「效应集中在少数会话」。
    """
    return {s: f1_of(vec[("own10", s)], idx) - f1_of(vec[("mixnorm", s)], idx)
            for s in seeds}


def paired(tag_set, only=None):
    """载入一个评估集, 返回逐种子 ΔF1 与配套结构。

    返回 dict: P(原始记录) / seeds / cids(共同块) / vec(对齐后的块向量表)
              / d(逐种子 ΔF1) / missing(缺哪些)
    配对种子不足 2 个时返回 None (此时 d 无意义, 不许拿去算统计量)。

    `only`: 可选的块 id 集合 —— 只保留落在里面的块。默认 None = 全部 (行为与
    加这个参数之前**逐位相同**, 所以既有结论不受影响)。加它是为了在同一份
    产物上问「只取新增那 400 块时 ΔF1 是多少」而不必复制一份口径 —— 复制
    口径就是两份定义, 迟早只改一份 (见本文件顶部的注释)。
    """
    P, missing = {}, []
    for arm in ARMS:
        for s in SEEDS:
            e = load(tag_set, arm, s)
            if e is None:
                missing.append(f"{arm}{suffix(s)}")
            else:
                P[(arm, s)] = e
    seeds = [s for s in SEEDS if ("own10", s) in P and ("mixnorm", s) in P]
    if len(seeds) < 2:
        return None
    cids = sorted(set.intersection(*(set(P[k]["per_chunk"]) for k in P)))
    if only is not None:
        keep = set(only)
        cids = [c for c in cids if c in keep]
    vec = {k: [P[k]["per_chunk"][c] for c in cids] for k in P}
    idx = range(len(cids))
    d = d_of(vec, seeds, idx)
    return {"P": P, "seeds": seeds, "cids": cids, "vec": vec,
            "d": d, "missing": missing}


def boot_se(vec, cids, seeds, boot=B, rng_seed=0):
    """块级 bootstrap 的 SE: **所有种子共用同一份重抽块** (配对结构不能拆)。

    这就是别处报的 `SE_chunk`。`rng_seed=0` 与历史产物逐位一致, 别改。
    """
    n = len(cids)
    rng = np.random.default_rng(rng_seed)
    db = np.empty(boot)
    for b in range(boot):
        ii = rng.integers(0, n, n)
        db[b] = np.mean(list(d_of(vec, seeds, ii).values()))
    return float(db.std(ddof=1))


def boot_se_mc(vec, cids, seeds, boot=B, reps=8):
    """`SE_chunk` 的**蒙特卡洛误差** —— 它是抽出来的, 不是算出来的。

    返回 `(点估计, MC 标准差, reps 次估计的 min, max)`。
    点估计 = `boot_se(..., rng_seed=0)`, **逐位一致** (reps 的第一项就是它)。

    ## 为什么必须有这个函数

    `SE_chunk` 同时是**判据线**的输入 (`线 = 2×SE_chunk`)。B=2000 时它的 MC
    相对误差实测 **2.1%** (0.009392~0.009967), 也就是说线在
    **0.0188~0.0199** 之间随 `rng_seed` 漂 —— **第四位小数根本不是数据的性质**。
    再往下 `n ∝ 1/((ΔF1/2)²−SE_chunk²)` 是个小差, 同一批数据能给出 **9~18.5**。

    ⇒ **凡是拿 `SE_chunk` 当阈值用的地方, 必须连 MC 误差一起报**,
    并**检查判决是否在 MC 误差下稳**(余量 / MC ≫ 1)。只报"现算"不够:
    现算的量若本身是随机估计, 报四位小数等于给它一个它没有的精度。

    MC 误差按 `1/√boot` 缩: B=8000 时实测 0.84% (半个数量级 ≈ 2×)。
    """
    est = np.array([boot_se(vec, cids, seeds, boot, rng_seed=r)
                    for r in range(reps)])
    return (float(est[0]), float(est.std(ddof=1)),
            float(est.min()), float(est.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="tag_set", default="e4",
                    choices=["filt", "e2", "e4", "e4x"])
    ap.add_argument("--boot", type=int, default=B)
    args = ap.parse_args()

    D = paired(args.tag_set)
    print(f"=== 评估集 {args.tag_set} ===")
    if D is None:
        print("  配对数不足 2, 退出。")
        return
    P, seeds, cids, vec, d, missing = (D["P"], D["seeds"], D["cids"],
                                       D["vec"], D["d"], D["missing"])
    if missing:
        print(f"  ⚠️ 缺失 {len(missing)} 个: {missing}")
    print(f"  可用配对: {len(seeds)}/7  ->  {seeds}")
    ok = (len(seeds) == len(SEEDS))

    e0 = P[("own10", seeds[0])]
    n_gt = sum(e0["per_chunk"][c]["n_gt"] for c in cids)
    n_pos = sum(1 for c in cids if e0["per_chunk"][c]["n_gt"] > 0)
    print(f"  公共块 {len(cids)}, 真值事件 {n_gt}, 含事件块 {n_pos} "
          f"({n_pos/len(cids):.2%} 正例率)")
    if not ok:
        # 🔴 产物不齐时算出来的统计量不是最终值 —— 必须显眼, 不能只印一行「缺失」。
        print(f"  🔴 **产物只有 {len(seeds)}/{len(SEEDS)} 对 —— 下面的数不是最终统计量**, "
              f"只当管线自检看。")

    # ---------- 逐种子 ΔF1 ----------
    print("\n=== 逐种子 F1 与配对差 ===")
    print(f"  {'seed':>7}{'own10':>10}{'mixnorm':>10}{'ΔF1':>10}")
    for s in seeds:
        f_o = f1_of(vec[("own10", s)], range(len(cids)))
        f_m = f1_of(vec[("mixnorm", s)], range(len(cids)))
        print(f"  {s:>7}{f_o:>10.4f}{f_m:>10.4f}{d[s]:>+10.4f}")
    dv = np.array([d[s] for s in seeds])
    mean, sd = dv.mean(), dv.std(ddof=1)
    SE_seed = sd / np.sqrt(len(dv))
    print(f"\n  ΔF1 = {mean:+.4f} ± {sd:.4f}  (n={len(dv)}, SE_seed = SD/√n = {SE_seed:.4f})")

    # ---------- 块级 bootstrap: 7 个种子共用同一份重抽块 ----------
    SE_chunk = boot_se(vec, cids, seeds, args.boot)
    SE_all = float(np.hypot(SE_seed, SE_chunk))

    print(f"\n=== 两个方差源 ({args.boot} 次块级 bootstrap) ===")
    print(f"  SE_seed  = {SE_seed:.4f}   (跨种子; **不随评估集增大而缩**)")
    print(f"  SE_chunk = {SE_chunk:.4f}   (块抽样; 按 1/√k 缩)")
    print(f"  合并 SE  = {SE_all:.4f}   (= hypot)")
    print(f"\n  t(只看种子) = {mean/SE_seed:.3f}")
    print(f"  t(只看块)   = {mean/SE_chunk:.3f}")
    print(f"  t(合并)     = {mean/SE_all:.3f}   <- 应报这个")

    # ---------- 55555 翻不翻 ----------
    if 55555 in d:
        x = d[55555]
        print(f"\n=== 关键词: 种子 55555 的 ΔF1 = {x:+.4f} ===")
        if x < 0:
            # ⚠️ 这里原来写死「既有值 −0.0345」。那个数来自 filt 那一跑, 本分支
            #    并没有读 filt —— 写死就等于把它变成永远不被复跑纠正的常数。
            #    改成不报数、只报方向, 具体值以 `--set filt` 的输出为准。
            print(f"  仍为负（与旧集同向; 旧集具体值以 `--set filt` 的输出为准, 此处不写死）。")
            print(f"  它独树一帜地反向, 是 n=7 的分辨率下限被打掉的主因 ——")
            print(f"  两尾最小 p = 2/2^7 = 0.0156 要求 7/7 同向。")
        else:
            print(f"  **翻正** ⇒ 7/7 同向 ⇒ 两尾最小 p = 0.0156（n=7 的分辨率下限）")
        n_neg = sum(1 for s in seeds if d[s] < 0)
        print(f"  负号个数: {n_neg}/{len(seeds)}")

    # ---------- 自检 ----------
    print("\n" + "=" * 74)
    if args.tag_set == "filt":
        print("  自检 @ filt: 应复现 §7.2 的已知数字")
        ok = True
        for name, got, want, tol in (
                ("ΔF1", mean, BASELINE["delta"], 0.003),
                ("SE_seed", SE_seed, BASELINE["se_seed"], 0.002),
                ("SE_chunk", SE_chunk, BASELINE["se_chunk"], 0.004),
                ("t", mean / SE_all, BASELINE["t"], 0.15)):
            good = abs(got - want) <= tol
            ok &= good
            print(f"    {name:<9} 得到 {got:+.4f}  期望 {want:+.4f}  "
                  f"{'✅' if good else '❌ 差 ' + format(got-want, '+.4f')}")
        print(f"\n  {'✅ 自检通过 —— 分析代码可信, e4 的结果可直接采信' if ok else '❌ 自检失败 —— 先修代码, 别看 e4 结果'}")
    else:
        # 🔴 这里原来印的是「预测 (§5): k=4 后 t = 2.42~2.66, 天花板 2.91」——
        #    **已作废**, 两处都错 (2026-09-18 01:3x 改):
        #    (a) 那句「天花板 2.91」建立在「SE_seed 不缩」的假设上, 而 §5.7 证明
        #        SE_seed 里混着会缩的 SE_chunk, 天花板其实更高 —— §5.7 自己写了
        #        「这与我此前的说法相反」;
        #    (b) 更要紧的是: **判据不是 t>2, 而是「七种子均值 vs 临界线 SE_chunk×2」**
        #        (§5.8/§5.8b)。写成 `mean/SE_all > 2` 是把两条不同的判据混成一条 ——
        #        SE_all > SE_chunk, 所以它**比预登记的那条更严**, 会静默改判。
        #        (memory: hardcoded-conclusions-escape-reproduction / threshold-must-match-null-model)
        CRIT = SE_chunk * 2.0
        print(f"  逐种子 SD (跨种子) = {sd:.4f}   ← §5.7 预登记该落 [0.0210, 0.0257]")
        _in = 0.0210 <= sd <= 0.0257
        print(f"     {'✅ 命中 §5.7 预登记区间' if _in else '❌ 落在区间外 —— §5.7 判读表: <0.021 或 >0.031 都要重算'}")
        print(f"\n  🔒 **主判据 (§5.8/§5.8b): 七种子均值 vs 临界线 SE_chunk×2**")
        # ⚠️ 这里必须打 5 位: 4 位下 "0.0097 × 2 = 0.0195" 看着自相矛盾 (实则 0.01949)
        print(f"     临界线 = {SE_chunk:.5f} × 2 = {CRIT:.5f}"
              f"   (印 4 位是 {CRIT:.4f}; §5.8b 那条「两位小数不够用」的坑)")
        print(f"     实测均值 = {mean:+.4f}   ⇒  {'**线上**' if mean > CRIT else '**线下**'}"
              f" ({'能靠加种子救' if mean > CRIT else '连无穷多种子也救不回来'})")
        # §5.8c: 这条线是**抽出来的**, 判决稳不稳要一起看
        _p, _mc, _lo, _hi = boot_se_mc(vec, cids, seeds, B)
        _ratio = abs(mean - CRIT) / (2 * _mc)
        print(f"     🔴 但 SE_chunk 是 bootstrap 抽的: ±{_mc:.6f} (MC) ⇒ "
              f"线在 [{2*_lo:.4f}, {2*_hi:.4f}] 漂 (§5.8c)")
        print(f"        余量 {abs(mean-CRIT):.4f} / MC {2*_mc:.4f} = **{_ratio:.1f}×**"
              f" ⇒ {'✅ 判决扛得住 MC' if _ratio > 3 else '🔴 判决会被 MC 掀翻, 先加 boot'}")
        print(f"\n  参照 t = {mean/SE_all:.2f} (合并 SE; ⚠️ 这不是预登记判据, 比上面那条更严)")


if __name__ == "__main__":
    main()
