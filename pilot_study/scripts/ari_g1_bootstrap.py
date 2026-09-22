"""G1 消融: ΔF1 的配对 bootstrap 置信区间.

为什么必须做: 评估集只有 26 个真值 BC 事件, 两臂的 tp 是 12 vs 7。这个量级下
"ΔF1 = -0.03" 既可能是真实效应, 也可能纯属噪声 —— 不给出区间就没法在论文里
下任何结论。

配对: 两臂评的是**同一批 300 个 chunk**, 所以按 chunk 有放回重采样 (同一组
下标同时作用于两臂), 消掉了 chunk 难度带来的方差, 比独立重采样更有效力。

用法: python scripts/ari_g1_bootstrap.py
"""
import json
import sys

import numpy as np

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
N_BOOT = 10000
SEED = 0


def load(tag):
    r = json.load(open(f"{ANNOT}/g1_eval_{tag}.json"))[tag]
    return r["per_chunk"], r


def f1_of(per, idx):
    tp = sum(per[c]["tp"] for c in idx)
    fp = sum(per[c]["fp"] for c in idx)
    gt = sum(per[c]["n_gt"] for c in idx)
    p = tp / max(1, tp + fp)
    r = tp / max(1, gt)
    return 2 * p * r / (p + r) if (p + r) else 0.0, p, r, tp, fp, gt


def main():
    o, ro = load("own10")
    m, rm = load("mixnorm")
    cids = sorted(set(o) & set(m))
    if len(cids) != len(o) or len(cids) != len(m):
        print(f"**警告**: chunk 集合不一致 own={len(o)} mix={len(m)} 交集={len(cids)}")
    if "per_chunk" not in ro or not ro["per_chunk"]:
        sys.exit("没有 per_chunk —— 请用改过的 ari_g1_eval.py 重跑评估")

    fo, po, rco, tpo, fpo, gto = f1_of(o, cids)
    fm, pm, rcm, tpm, fpm, gtm = f1_of(m, cids)
    print(f"评估集 {len(cids)} chunk, 真值事件 {gto}")
    print(f"{'arm':8} {'F1':>7} {'P':>7} {'R':>7} {'tp':>4} {'fp':>5} {'gt':>4}")
    for nm, f, p, r, tp, fp, gt in (("own10", fo, po, rco, tpo, fpo, gto),
                                    ("mixnorm", fm, pm, rcm, tpm, fpm, gtm)):
        print(f"{nm:8} {f:7.4f} {p:7.4f} {r:7.4f} {tp:4d} {fp:5d} {gt:4d}")
    obs = fm - fo
    print(f"\n观测 ΔF1 (mixnorm - own10) = {obs:+.4f}")

    # 配对 bootstrap: 同一组下标同时作用于两臂
    rng = np.random.default_rng(SEED)
    n = len(cids)
    ci = np.arange(n)
    ds = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = [cids[i] for i in rng.integers(0, n, n)]
        ds[b] = f1_of(m, idx)[0] - f1_of(o, idx)[0]
    lo, hi = np.percentile(ds, [2.5, 97.5])
    print(f"配对 bootstrap ({N_BOOT} 次) 95% CI = [{lo:+.4f}, {hi:+.4f}]")
    print(f"P(ΔF1 > 0) = {(ds > 0).mean():.3f}   (0.5 附近 = 与噪声无异)")
    if lo <= 0 <= hi:
        print("结论: **区间跨 0, 两臂差异不显著** —— 观测空间不是瓶颈")
    elif obs > 0:
        print("结论: 区间不跨 0, 观测对方**显著有益**")
    else:
        print("结论: 区间不跨 0, 观测对方**显著有害** (需解释为何)")

    json.dump({"n_chunks": n, "n_gt": gto,
               "own10": ro, "mixnorm": rm,
               "delta_f1": round(obs, 4),
               "ci95": [round(float(lo), 4), round(float(hi), 4)],
               "p_delta_gt_0": round(float((ds > 0).mean()), 4)},
              open(f"{ANNOT}/g1_bootstrap.json", "w"), indent=2)
    print("saved g1_bootstrap.json")


if __name__ == "__main__":
    main()
