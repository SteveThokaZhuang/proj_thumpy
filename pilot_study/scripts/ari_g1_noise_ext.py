"""把 own10 臂的 run-to-run 分布描厚, 回答: f8c_v2 的 0.1203 是重尾吗?

背景 (报告 §7.3): f8c_v2 (0.1203) 比三条 own10 臂 (0.1727 / 0.1887 / 0.1805) 低
7.6 个标准差, 而那 0.052 的成因排查后仍未找到。**但三个点估 SD 太薄** —— 如果分布
本来就是重尾的, 那么"0.1203 是异常值"这个判断本身就不成立, 而且报告里"噪声下限
0.016–0.024"这个结论也得跟着修。本脚本用更多种子来检验这一点。

做四件事:
  ① 列出所有 own10 重跑, 给出经验分布 (均值 / SD / 极差 / 分位);
  ② **给 SD 自己一个 CI** —— 这是本分析的关键。SD 是估出来的, 不是已知的,
     有限样本下它本身有很宽的不确定性; 只看点估计会把结论说满;
  ③ 在最不利的假设下 (取 SD 的置信上界) 重算 v2 偏离几个 SD —— 如果连那时
     0.1203 都远远落在分布外, "异常值"的判断才对重尾稳健;
  ④ 对每对种子做 chunk 级配对比较, 看种子间差异是"整体平移"还是"换个种子就
     换一批块做对" —— 后者会让 F1 的自然波动更大, 也是重尾的机制来源。

注: 本脚本**不重采样权重**。报告 §8 已经说明 chunk 级配对 bootstrap 是反保守的
(它固定训练权重, 只重采样 chunk), 所以这里的标尺一律是 **run/seed 级变异**。
"""
import glob
import json
import os

import numpy as np

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
V2_F1 = 0.1203                      # f8c_v2 在匹配 prompt 下的 F1 (报告 §7.1)
N_BOOT = 20000
BOOT_SEED = 0


def load_own10():
    """收集所有 own10 重跑的 eval json, 返回 [(标签, F1, per_chunk), ...] 按 F1 排序."""
    runs = []
    # 主实验 (seed 42) 的 json 文件名没有种子后缀
    p = f"{ANNOT}/g1_eval_own10.json"
    if os.path.exists(p):
        runs.append(("seed 42 (主)", json.load(open(p))["own10"]))
    # 其余各种子: g1_eval_own10_s<seed>.json
    for f in sorted(glob.glob(f"{ANNOT}/g1_eval_own10_s*.json")):
        seed = os.path.basename(f)[len("g1_eval_own10_s"):-len(".json")]
        tag = f"own10_s{seed}"
        d = json.load(open(f))
        if tag in d:
            runs.append((f"seed {seed}", d[tag]))
    # c1024 臂: 已证 cutoff_len 是空操作, 可当第四个等价重跑; 单独标注不混入统计
    p = f"{ANNOT}/g1_eval_own10_c1024.json"
    c1024 = json.load(open(p))["own10_c1024"] if os.path.exists(p) else None
    return sorted(runs, key=lambda t: t[1]["f1"]), c1024


def mean_ci(x, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(x, size=len(x), replace=True).mean()
                   for _ in range(n_boot)])
    return np.percentile(bs, [2.5, 97.5])


# t 分布 97.5% 分位 (双尾 95%), 按自由度; 避免为几个数引入 scipy 依赖
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def t975(df):
    return T975.get(df, 1.96)


def sd_ci(x):
    """SD 的 95% CI —— 卡方法 (正态理论), 小样本下才是对的.

    **不要用 bootstrap 估 SD 的 CI**: SD 在重抽样上的分布**上界就是原样本的 SD**
    (只有"恰好重现原样本全部取值"的那种重抽样才能取到), 所以 bootstrap 给出的
    区间系统性偏窄、且永远够不到真值上方 —— 用它来论证"SD 很不确定"会自相矛盾。
    """
    n = len(x)
    if n < 2:
        return 0.0, 0.0
    s = x.std(ddof=1)
    df = n - 1
    # 卡方分位: 用 Wilson–Hilferty 近似, 免去 scipy
    def chi2(p, k):
        z = np.sqrt(2 * k) - 1.0
        # 标准正态分位
        from math import erf, sqrt

        def norm_ppf(q):
            lo, hi = -10.0, 10.0
            for _ in range(200):
                mid = (lo + hi) / 2
                if 0.5 * (1 + erf(mid / sqrt(2))) < q:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2
        z = norm_ppf(p)
        return k * (1 - 2 / (9 * k) + z * np.sqrt(2 / (9 * k))) ** 3
    lo = np.sqrt(df * s * s / chi2(0.975, df))
    hi = np.sqrt(df * s * s / chi2(0.025, df))
    return lo, hi


def pred_interval(x):
    """新的一次重跑, 95% 预测区间 —— 回答"0.1203 是不是这一过程的正常抽样".

    这是比 z 分数正确的工具: z 把均值/SD 当成已知, 而它们都是估出来的。
    样本越少这个区间越宽, 那正是我们要传达的不确定性。
    """
    n = len(x)
    m, s = x.mean(), x.std(ddof=1)
    half = t975(n - 1) * s * np.sqrt(1 + 1 / n)
    return m - half, m + half


def f1_of(per, cids):
    tp = sum(per[c]["tp"] for c in cids)
    fp = sum(per[c]["fp"] for c in cids)
    gt = sum(per[c]["n_gt"] for c in cids)
    p = tp / max(1, tp + fp)
    r = tp / max(1, gt)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def main():
    runs, c1024 = load_own10()
    f1s = np.array([r["f1"] for _, r in runs])
    n = len(f1s)

    print("=" * 74)
    print("① own10 臂的 run-to-run 分布 (同数据、同 config, 只换随机种子)")
    print("=" * 74)
    print(f"\n{'run':16}{'F1':>8}{'P':>8}{'R':>8}{'n_pred':>8}")
    for tag, r in runs:
        print(f"{tag:16}{r['f1']:8.4f}{r['precision']:8.4f}"
              f"{r['recall']:8.4f}{r['n_pred']:8}")
    if c1024 is not None:
        print(f"{'c1024 (参考)':16}{c1024['f1']:8.4f}{c1024['precision']:8.4f}"
              f"{c1024['recall']:8.4f}{c1024['n_pred']:8}   ← cutoff 空操作, 等价重跑")

    mu, sd = float(f1s.mean()), float(f1s.std(ddof=1)) if n > 1 else 0.0
    print(f"\n  n = {n}  均值 = {mu:.4f}  SD = {sd:.4f}  "
          f"极差 = {f1s.max() - f1s.min():.4f}  [{f1s.min():.4f}, {f1s.max():.4f}]")
    if n > 1:
        lo_m, hi_m = mean_ci(f1s)
        print(f"  均值的 95% CI      : [{lo_m:.4f}, {hi_m:.4f}]")

    print("\n" + "=" * 74)
    print("② SD 自己有多不确定 —— 只看点估计会把结论说满")
    print("=" * 74)
    if n > 2:
        lo_s, hi_s = sd_ci(f1s)
        print(f"\n  SD = {sd:.4f}   95% CI = [{lo_s:.4f}, {hi_s:.4f}]"
              f"   (卡方法; 相对不确定度 ≈ 1/√(2(n-1)) = "
              f"{1 / np.sqrt(2 * (n - 1)):.0%})")
        print(f"  n 只有 {n}, 所以 SD 本身还差着 ±{(hi_s - lo_s) / 2:.4f} —— "
              f"任何'噪声下限 = {sd:.3f}'的说法都该带上这个区间")
    else:
        lo_s = hi_s = sd
        print("\n  样本太少, SD 无法估计")
    # 把 c1024 并进来再看一次 —— 它是只差一个空操作的第四个等价重跑
    if c1024 is not None:
        both = np.append(f1s, c1024["f1"])
        print(f"\n  并入 c1024 (n={len(both)}): 均值 {both.mean():.4f} "
              f"SD {both.std(ddof=1):.4f} 极差 {both.max() - both.min():.4f}")

    print("\n" + "=" * 74)
    print("③ f8c_v2 的 0.1203 是这一过程的正常抽样吗")
    print("=" * 74)
    print(f"\n  v2 F1 = {V2_F1:.4f}")
    print(f"  比 own10 观测到的最低一次还低 {f1s.min() - V2_F1:.4f}")
    if sd > 0:
        print(f"  偏离均值 {(mu - V2_F1) / sd:.1f} 个 SD (点估计; 但 SD 本身"
              f"有 ±{(hi_s - lo_s) / 2:.4f} 的不确定度)")
    if n > 1:
        plo, phi = pred_interval(f1s)
        inside = plo <= V2_F1 <= phi
        print(f"\n  **新一次重跑的 95% 预测区间** = [{plo:.4f}, {phi:.4f}]"
              f"   (t 分布, 已计入均值与 SD 都是估出来的)")
        print(f"  v2 落在区间{'内' if inside else '外'} —— "
              f"{'无法排除它是同一过程的普通抽样' if inside else '落在区间之外, 才谈得上异常值'}")
        print(f"\n  这个区间有 {phi - plo:.4f} 宽, 而我们要解释的差距是 0.052 ——")
        print(f"  样本只有 {n} 次重跑时, 「分不清」恰恰是数据能支持的唯一结论。")

    print("\n" + "=" * 74)
    print("④ 种子间差异是「整体平移」还是「换一批块做对」")
    print("=" * 74)
    if n > 1:
        cids = sorted(runs[0][1]["per_chunk"].keys())
        # **只在含 GT 的块上统计**。300 块里绝大多数根本没有真值事件, "没命中"
        # 对它们是废话 —— 拿全部 300 块当分母会把"从未命中 284 块"说成一个发现,
        # 其实那 284 里 275 块压根没有 GT 可命中。
        gtc = [c for c in cids if runs[0][1]["per_chunk"][c]["n_gt"] > 0]
        n_gt_ev = sum(runs[0][1]["per_chunk"][c]["n_gt"] for c in gtc)
        print(f"\n  chunk 总数 {len(cids)}, 含 GT 的 chunk {len(gtc)} "
              f"(共 {n_gt_ev} 个真值事件); 其余 {len(cids) - len(gtc)} 块无 GT, 不参与下述统计")
        # 每次重跑做对的块集合
        hit = {}
        for tag, r in runs:
            hit[tag] = {c for c in gtc if r["per_chunk"][c]["tp"] > 0}
        print(f"\n{'run':16}{'命中的 GT 块数':>14}")
        for tag, _ in runs:
            print(f"{tag:16}{len(hit[tag]):>14}")
        # 两两 Jaccard: 低 = 各次重跑命中不同的块 (→ 自然波动更大)
        print("\n  命中块集合的两两 Jaccard 相似度 (1=完全相同, 0=毫不重叠):")
        tags = [t for t, _ in runs]
        vals = []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = hit[tags[i]], hit[tags[j]]
                jac = len(a & b) / max(1, len(a | b))
                vals.append(jac)
        if vals:
            print(f"    中位 {np.median(vals):.3f}  范围 "
                  f"[{min(vals):.3f}, {max(vals):.3f}]  (共 {len(vals)} 对)")
        # 把含 GT 的块按"稳定性"分三层 —— 这比一个 SD 数字更能说明噪声来自哪里
        union = set().union(*hit.values())
        inter = set.intersection(*hit.values()) if hit else set()
        never = set(gtc) - union
        print(f"\n  含 GT 的 {len(gtc)} 块分三层:")
        print(f"    总是命中 (全部 {n} 次重跑都做对): {len(inter)} 块  ← 学稳了的")
        print(f"    时灵时不灵 (部分重跑命中)      : {len(union - inter)} 块"
              f"  ← **噪声就是这 {len(union - inter)} 块贡献的**")
        print(f"    从未命中 (7 次重跑全做不对)    : {len(never)} 块"
              f"  ← 与种子运气无关, 更像标签/音频问题, 值得单独查")
        print(f"\n    即: 26 个真值事件里, {len(inter)} 个稳定可做, "
              f"{len(union - inter)} 个看运气, {len(never)} 个谁做不出来。")

    out = {
        "n": n,
        "runs": [{"tag": t, "f1": r["f1"], "precision": r["precision"],
                  "recall": r["recall"], "n_pred": r["n_pred"]} for t, r in runs],
        "mean": round(mu, 4),
        "sd": round(sd, 4),
        "range": round(float(f1s.max() - f1s.min()), 4),
        "min": round(float(f1s.min()), 4),
        "max": round(float(f1s.max()), 4),
        "sd_ci95": [round(float(lo_s), 4), round(float(hi_s), 4)] if n > 2 else None,
        "v2_f1": V2_F1,
        "v2_z_vs_sd": round((mu - V2_F1) / sd, 2) if sd > 0 else None,
        "v2_pred_interval95": ([round(float(plo), 4), round(float(phi), 4)]
                               if n > 1 else None),
        "v2_inside_pred_interval": bool(plo <= V2_F1 <= phi) if n > 1 else None,
    }
    p = f"{ANNOT}/g1_noise_ext.json"
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
    print(f"\n已写出 {p}")


if __name__ == "__main__":
    main()
