"""直接量 `SE(N)`: 加块到底还降不降 SE? —— 不用任何标度假设。

## 为什么不用 `var ∝ k^-p` 拟合

`g1_e4c_varscale.py` 拟出 var_a 的 p̂=0.43、var_e 的 p̂=1.20, 但只有三档 k、
df 又悬殊 ⇒ p̂ 的误差棒宽到没法判读。**与其拟合指数再外推, 不如直接量想要的量。**

## 做法

e4c 有 24 轮 × 200 块, 每轮覆盖全部 200 个 key。随机抽 N 轮 → 取并集 → 算一次
ΔF1 → 重复 B 次 → 取 SD。这就是 `ΔF1` 在「200N 块」时的**抽样 SD**, 不经 ANOVA。

⚠️ **两个第一版踩到的坑, 都修在这里**:
1. `N=24` **是退化的** —— 从 24 轮里抽 24 轮只有一种子集, SD 恒为 0, 不是「SE 归零」。
   参考点不能用它。(第一版就打印出 `SE(24)=0.00000` 并据此报「已贴地板」。)
2. 「连续取 N 轮」的窗口**互相重叠** ⇒ 窗口间 SD 被人为压小, 看着像「成簇」。
   改成正交的次序检验: 轮级 ΔF1 的 **lag-1 自相关 + 置换零模型**。

判读用**有限总体**公式 (从 R 轮里无放回抽 N 轮):
    SD(N) = σ · √( (R−N) / (N(R−1)) )
⇒ 由三档 N 各自反解 σ, **三者一致**才说明「轮可交换、无成簇」。

用法 (srun 内, fd_analysis): python scripts/g1_e4c_se_scaling.py
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
B = 600
NROUNDS = (3, 6, 12)          # ⚠️ 不含 24 (退化)


def main():
    Ds = []
    for j in range(12):
        D = paired(f"e4c{j:02d}")
        assert D is not None and not D["missing"], f"e4c 批 {j} 不完整"
        Ds.append((j, D))
    seeds = list(Ds[0][1]["seeds"])
    ns = len(seeds)

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
    assert len(cids_p) == 4800, f"合并池 {len(cids_p)} 块, 应为 4800"
    pos_p = {c: i for i, c in enumerate(cids_p)}

    rounds = []
    for j, D in Ds:
        order = [l.strip() for l in
                 open(f"{ANNOT}/g1_eval4c_ids_b{j:02d}_ids.txt") if l.strip()]
        for r in range(len(order) // K):
            chunk = order[r * K:(r + 1) * K]
            assert len({key_of(c) for c in chunk}) == K, f"批 {j} 轮 {r} 序错"
            rounds.append([pos_p[c] for c in chunk])
    R = len(rounds)
    assert R == 24, f"应有 24 轮, 实得 {R}"

    def df1(idx):
        if len(seeds) == 1:
            return d_of(vec_p, seeds, idx)[seeds[0]]
        return float(np.mean(list(d_of(vec_p, seeds, idx).values())))

    full = df1([i for r in rounds for i in r])
    round_vals = np.array([df1(r) for r in rounds])
    print(f"e4c: {len(cids_p)} 块 / {len({key_of(c) for c in cids_p})} 个 key / "
          f"{R} 轮 × {K} 块")
    print(f"全 24 轮 ΔF1 = {full:+.4f}")
    print(f"轮级 ΔF1: 均值 {round_vals.mean():+.4f}, "
          f"SD {round_vals.std(ddof=1):.5f}, "
          f"范围 [{round_vals.min():+.4f}, {round_vals.max():+.4f}]\n")

    # ---------- ① 直接量 SE(N) ----------
    out = {}
    print(f"{'N(轮)':>6}{'块数':>7}{'SD(随机抽)':>13}{'σ̂ 反解':>11}")
    for n in NROUNDS:
        rng = np.random.default_rng(0)
        vals = np.empty(B)
        allr = np.arange(R)
        for b in range(B):
            pick = rng.choice(allr, n, replace=False)
            vals[b] = df1([i for r in pick for i in rounds[r]])
        sd = float(vals.std(ddof=1))
        # 有限总体: SD = σ·√((R−N)/(N(R−1)))  ⇒ 反解 σ
        sigma = sd / np.sqrt((R - n) / (n * (R - 1)))
        print(f"{n:>6}{n*K:>7}{sd:>13.5f}{sigma:>11.5f}")
        out[n] = {"n_rounds": n, "n_blocks": n * K, "sd": sd, "sigma_hat": sigma,
                  "mean": float(vals.mean())}

    sig = np.array([out[n]["sigma_hat"] for n in NROUNDS])
    print(f"\n  σ̂ 三档 = {', '.join(f'{s:.5f}' for s in sig)}")
    spread = (sig.max() - sig.min()) / sig.mean()
    print(f"  相对散布 {spread:.1%} ⇒ "
          f"{'✅ 三档一致 ⇒ **轮可交换、无成簇**, 就是一个 σ' if spread < 0.15 else '🔴 不一致 ⇒ 轮不可交换, 有成簇'}")
    sigma = float(sig.mean())

    # ---------- ② 次序检验 (正交, 不重叠) ----------
    js = np.arange(R, dtype=float)
    r1 = float(np.corrcoef(round_vals[:-1], round_vals[1:])[0, 1])
    rng = np.random.default_rng(0)
    null = np.array([abs(np.corrcoef(round_vals[:-1],
                                     rng.permutation(round_vals)[1:])[0, 1])
                     for _ in range(20000)])
    p1 = float((null >= abs(r1)).mean())
    print(f"\n=== 次序检验: 轮级 ΔF1 的 lag-1 自相关 ===")
    print(f"  r = {r1:+.3f} (n={R}), 置换 p = {p1:.3f} ⇒ "
          f"{'✅ 与无序相容' if p1 > 0.05 else '🔴 有次序结构 ⇒ 成簇'}")
    print(f"  (另: 轮序 vs ΔF1 的 r = "
          f"{float(np.corrcoef(js, round_vals)[0,1]):+.3f})")

    # ---------- ③ 外推: 加块到哪里撞地板 ----------
    try:
        vs = json.load(open(f"{ANNOT}/g1_e4c_varscale.json"))
        var_c = vs["200"]["var_c"]
        floor = float(np.sqrt(var_c / ns))
        print(f"\n=== 加块撞地板的位置 ===")
        print(f"  种子地板 √(var_c/{ns}) = {floor:.5f}")
        print(f"  块抽样项 σ̂/√N, σ̂ = {sigma:.5f}")
        n_star = (sigma / floor) ** 2
        print(f"  两者相等 ⇒ N* = (σ̂/地板)² = {n_star:.1f} 轮 = "
              f"{n_star*K:,.0f} 块")
        print(f"  现在 24 轮 ({24*K} 块): 块项 {sigma/np.sqrt(24):.5f} vs "
              f"地板 {floor:.5f} ⇒ "
              f"{'块项仍占优, **加块还有用**' if sigma/np.sqrt(24) > floor else '**已到地板, 加块无用**'}")
        for n in (24, 48, 96):
            se = float(np.sqrt(sigma ** 2 / n + floor ** 2))
            print(f"    N={n:>3} 轮 ({n*K:>6,} 块) ⇒ SE = {se:.5f}, "
                  f"t = {full/se:.2f}")
    except Exception as e:                                    # noqa: BLE001
        print(f"  (没读到 varscale.json, 跳过: {e})")

    json.dump({str(k): v for k, v in out.items()} |
              {"sigma": sigma, "sigma_spread": float(spread),
               "lag1_r": r1, "lag1_p": p1, "full_df1": full},
              open(f"{ANNOT}/g1_e4c_se_scaling.json", "w"), indent=1)
    print(f"\n-> {ANNOT}/g1_e4c_se_scaling.json")


if __name__ == "__main__":
    main()
