"""G1 多种子重复分析: 量化事件级 F1 的 run-to-run 噪声下限.

为什么必须做: 报告 §7 发现 f8c_v2 与 g1_own10 训练数据几乎同源 (2,095/2,100 块
逐字节相同) 却差 0.052 F1, 说明噪声很可能比待检效应 (0.033) 还大。但那一对还混着
5 个块的数据差异和 micro-batch 切分, 只是噪声的**上界估计**。这里用**同臂同数据、
只换种子**给出干净测量。

三个问题:
  ① 同臂换种子, F1 动多少?          -> 噪声下限的直接测量
  ② 换种子后, 两臂的差还是那个方向吗? -> 消融结论稳不稳
  ③ 把两臂各自的种子间波动当噪声, 观测到的 ΔF1=-0.033 还显著吗?

沿用主分析的配对 bootstrap (同一组 chunk 下标同时作用于两边), 因为两臂/两种子
评的都是**同一批 300 个 chunk**, 配对能消掉 chunk 难度带来的方差。
"""
import json
import os
import sys

import numpy as np

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
SEED = int(os.environ.get("G1_SEED", "1234"))
N_BOOT = 10000
BOOT_SEED = 0


def load(tag):
    p = f"{ANNOT}/g1_eval_{tag}.json"
    if not os.path.exists(p):
        return None, None
    r = json.load(open(p))[tag]
    return r.get("per_chunk"), r


def f1_of(per, idx):
    tp = sum(per[c]["tp"] for c in idx)
    fp = sum(per[c]["fp"] for c in idx)
    gt = sum(per[c]["n_gt"] for c in idx)
    p = tp / max(1, tp + fp)
    r = tp / max(1, gt)
    return (2 * p * r / (p + r) if (p + r) else 0.0), p, r, tp, fp, gt


def paired_boot(a, b, cids, n_boot=N_BOOT, seed=BOOT_SEED):
    """返回 (观测 ΔF1, 95% CI, P(Δ>0)); a、b 为 per_chunk, 同一组下标同作用于两者."""
    obs = f1_of(a, cids)[0] - f1_of(b, cids)[0]
    rng = np.random.default_rng(seed)
    n = len(cids)
    ds = np.empty(n_boot)
    for i in range(n_boot):
        idx = [cids[j] for j in rng.integers(0, n, n)]
        ds[i] = f1_of(a, idx)[0] - f1_of(b, idx)[0]
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return obs, lo, hi, float((ds > 0).mean())


def main():
    tags = ["own10", f"own10_s{SEED}", "mixnorm", f"mixnorm_s{SEED}"]
    P, meta = {}, {}
    for t in tags:
        per, r = load(t)
        if per is None:
            sys.exit(f"缺少 g1_eval_{t}.json —— 先跑 g1_eval_seeds.sh")
        P[t], meta[t] = per, r

    cids = sorted(set.intersection(*(set(P[t]) for t in tags)))
    n_gt = sum(P["own10"][c]["n_gt"] for c in cids)
    print(f"评估集 {len(cids)} chunk, 真值事件 {n_gt}")
    print(f"{'run':22} {'F1':>7} {'P':>7} {'R':>7} {'tp':>4} {'fp':>5} {'n_pred':>7}")
    for t in tags:
        f, p, r, tp, fp, gt = f1_of(P[t], cids)
        print(f"{t:22} {f:7.4f} {p:7.4f} {r:7.4f} {tp:4d} {fp:5d} "
              f"{meta[t]['n_pred']:7d}")

    print("\n=== ① 同臂换种子的波动 = 噪声下限 ===")
    noise = []
    for base in ("own10", "mixnorm"):
        s = f"{base}_s{SEED}"
        d = f1_of(P[s], cids)[0] - f1_of(P[base], cids)[0]
        noise.append(abs(d))
        print(f"  {base:8} 换种子后 ΔF1 = {d:+.4f}  (|Δ| = {abs(d):.4f})")
    noise_hi = max(noise)
    print(f"  -> 观测到的种子间波动: {min(noise):.4f} ~ {noise_hi:.4f}")

    print("\n=== ② 换种子后消融结论是否复现 ===")
    obs0 = f1_of(P["mixnorm"], cids)[0] - f1_of(P["own10"], cids)[0]
    obs1 = f1_of(P[f"mixnorm_s{SEED}"], cids)[0] - f1_of(P[f"own10_s{SEED}"], cids)[0]
    o, lo, hi, pgt = paired_boot(P[f"mixnorm_s{SEED}"], P[f"own10_s{SEED}"], cids)
    print(f"  种子 默认(42): ΔF1 (mixnorm - own10) = {obs0:+.4f}   [主实验]")
    print(f"  种子 {SEED}: ΔF1 (mixnorm - own10) = {obs1:+.4f}"
          f"   95% CI [{lo:+.4f}, {hi:+.4f}]  P(Δ>0)={pgt:.3f}")
    same_sign = (obs0 < 0) == (obs1 < 0)
    print(f"  -> 两次符号{'一致' if same_sign else '**不一致**'}")

    print("\n=== ③ 把种子间波动当噪声, -0.033 还站得住吗 ===")
    # 变量名必须与 ② 区分开: 早先这里复用了 lo/hi, 循环最后一轮是 mixnorm,
    # 把 ② 的跨臂 CI 覆盖成了 mixnorm 的同臂噪声 CI, 结果**存进 json 的 CI 是错的**
    # (控制台印的对, 落盘的错)。已改用 n_lo/n_hi。
    noise_ci = {}
    for base in ("own10", "mixnorm"):
        o3, n_lo, n_hi, p3 = paired_boot(P[f"{base}_s{SEED}"], P[base], cids)
        noise_ci[base] = [round(float(n_lo), 4), round(float(n_hi), 4)]
        print(f"  {base:8} 同臂两种子间: ΔF1 = {o3:+.4f}  95% CI "
              f"[{n_lo:+.4f}, {n_hi:+.4f}]  P(Δ>0)={p3:.3f}")
    print("  (同臂两种子的 CI 就是'纯噪声能造出多大差异'的标尺;")
    print("   主实验 ΔF1=-0.0327 若落在它之内, 就不能主张观测空间有真实效应)")

    out = {"seed": SEED, "n_chunks": len(cids), "n_gt": n_gt,
           "f1": {t: round(f1_of(P[t], cids)[0], 4) for t in tags},
           "seed_noise": {b: round(f1_of(P[f"{b}_s{SEED}"], cids)[0]
                                  - f1_of(P[b], cids)[0], 4)
                          for b in ("own10", "mixnorm")},
           "delta_default_seed": round(obs0, 4),
           "delta_new_seed": round(obs1, 4),
           # ② 的跨臂 CI (注意别被 ③ 的同臂 CI 覆盖 —— 见上面的注释)
           "delta_new_seed_ci95": [round(float(lo), 4), round(float(hi), 4)],
           # ③ 同臂两种子的"纯噪声"CI, 作为 ΔF1 的对照标尺
           "same_arm_noise_ci95": noise_ci,
           "same_sign": bool(same_sign)}
    json.dump(out, open(f"{ANNOT}/g1_noise.json", "w"), indent=2)
    print(f"\nsaved {ANNOT}/g1_noise.json")


if __name__ == "__main__":
    main()
