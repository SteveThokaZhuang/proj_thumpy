"""**事后扩展**（不是预登记的一部分）: 把 e4 / e4b 并进来, 16 批 × 400 块 = 6400 块。

## 为什么另开一个脚本

预登记 (`docs/pilot_study/2026-09-18_pool_sd.md` §1) 写死的是 **J=12 批**,
那一份结果已经出了 (`g1_e4c_pool_sd.py`)。本脚本**不改它、也不改判据**,
只是把**同一个统计量**放到更多批上, 用来回答预登记结果留下的那个分辨率问题:

> ρ̂ = 0.906, 但 95% CI 上界 **1.537 越过了 1.3 的判读线** ⇒
> "不拒绝 H₀" 不等于 "ρ=1"。df=11 太粗。

## 结构基础（本脚本逐条断言, 不是听说的）

四个集不是四次独立抽取, 而是**同一次顺序轮转抽取的连续切片**
(同一个 `--shuffle-seed 0`、同一个候选池, 后一个把前一个整体排除):

    e4   = e4x 的前 1200 行 = 轮 1–6   (3 批)
    e4b  = e4x 的后  400 行 = 轮 7–8   (1 批)
    e4c  =                   轮 9–32   (12 批)
    互不相交, 合起来 32 轮 × 200 key = 6400 块。

⚠️ **代价 (必须在报告里写)**: e4(09-18 02:57) / e4b(09-18 13:20) / e4c(09-19 09:41)
是**三次不同时刻**的评估 ⇒ 批身份与评估时刻在**这三段之间**是混杂的。
e4c 内部靠循环顺序避开了这一条 (见 `g1_e4c_eval.sh` 头部), 并进来就丢了。
所以本脚本的结论只作**佐证**, 主判读仍以预登记的 J=12 为准。

用法 (srun 内, fd_analysis): python scripts/g1_e4c_pool_sd_ext.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from g1_e4_analyze import paired, d_of, SEEDS  # noqa: E402
from g1_pool_null_probe import key_of, strat_boot_se  # noqa: E402

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
K = 200          # pool 的 key 数 (每轮 200 块)
B_BATCH = 400    # 一批 = 2 轮


def ids_of(path):
    return [l.strip() for l in open(path) if l.strip()]


def main():
    # ---------- 0. 结构断言: 先证明"连续切片", 再用它 ----------
    e4, e4b, e4x = (ids_of(f"{ANNOT}/g1_eval4_ids.txt"),
                    ids_of(f"{ANNOT}/g1_eval4b_ids.txt"),
                    ids_of(f"{ANNOT}/g1_eval4x_ids.txt"))
    assert e4x[:1200] == e4, "e4 ≠ e4x 的前 1200 行 —— 结构假设不成立"
    assert e4x[1200:] == e4b, "e4b ≠ e4x 的后 400 行 —— 结构假设不成立"
    for nm, v in (("e4", e4), ("e4b", e4b)):
        assert len(v) % K == 0, f"{nm} {len(v)} 块不是 {K} 的整数倍"
        for r in range(len(v) // K):
            ks = {key_of(c) for c in v[r * K:(r + 1) * K]}
            assert len(ks) == K, f"{nm} 的第 {r} 轮只覆盖 {len(ks)}/{K} 个 key"
    print(f"✅ 结构验证: e4 = e4x[:1200] (轮 1–6), e4b = e4x[1200:] (轮 7–8)")

    e4c_b = [ids_of(f"{ANNOT}/g1_eval4c_ids_b{j:02d}_ids.txt") for j in range(12)]
    seen = set(e4) | set(e4b)
    for j, b in enumerate(e4c_b):
        assert len(b) == B_BATCH, f"e4c 批 {j} 有 {len(b)} 块"
        assert not (set(b) & seen), f"e4c 批 {j} 与 e4/e4b 重叠 ⇒ 不是连续切片"
    print(f"✅ e4c 12 批与 e4/e4b 零重叠 ⇒ 四集互斥, 共 "
          f"{len(e4)+len(e4b)+sum(len(b) for b in e4c_b)} 块")

    # ---------- 1. 载入三个评估集, 拼成合并池 ----------
    sets = []                                    # [(name, tag, id_order)]
    for nm, tag, order in (("e4", "e4", e4), ("e4b", "e4b", e4b)):
        sets.append((nm, tag, order))
    D_all = {}
    for nm, tag, order in sets:
        D = paired(tag)
        assert D is not None, f"{tag} 配对不足"
        assert not D["missing"], f"{tag} 缺 {D['missing']}"
        assert len(D["seeds"]) == len(SEEDS), f"{tag} 只有 {len(D['seeds'])} 对种子"
        D_all[nm] = D
    Dc = paired("e4c00")                          # 只用来取 seeds/键集合
    seeds = list(Dc["seeds"])
    assert seeds == list(D_all["e4"]["seeds"]) == list(D_all["e4b"]["seeds"]), \
        "三段的配对种子集合不一致, 不能拼"

    # 合并池: cids_p + vec_p。**必须**先有合并的数组, 才谈得上跨段聚合
    cids_p, vec_p = [], {k: [] for k in Dc["vec"]}
    for nm, tag, order in sets:
        D = D_all[nm]
        pos = {c: i for i, c in enumerate(D["cids"])}
        for c in order:                           # 按 id 文件序 (= 轮序) 铺
            cids_p.append(c)
            for k in vec_p:
                vec_p[k].append(D["vec"][k][pos[c]])
    for j, b in enumerate(e4c_b):
        D = paired(f"e4c{j:02d}")
        assert D is not None and not D["missing"], f"e4c 批 {j} 不完整"
        assert set(D["seeds"]) == set(seeds), f"e4c 批 {j} 种子集合不同"
        pos = {c: i for i, c in enumerate(D["cids"])}
        for c in b:
            cids_p.append(c)
            for k in vec_p:
                vec_p[k].append(D["vec"][k][pos[c]])
    n_pool = len(cids_p)
    assert n_pool == 6400, f"合并池 {n_pool} 块, 应为 6400"
    assert len(set(cids_p)) == n_pool, "合并池里有重复块"
    print(f"合并池: {n_pool} 块 / {len({key_of(c) for c in cids_p})} 个 key")

    pos_p = {c: i for i, c in enumerate(cids_p)}

    def df1_of(idx):
        """一组的 ΔF1 —— 在合并池上按 `d_of` 算 (口径只有一处)。

        ⚠️ 对 idx 的**并集**算一次, 不能各组先算再平均:
        F1 是比值, ΔF1(A∪B) ≠ [ΔF1(A)+ΔF1(B)]/2。见 memory
        `delta-f1-does-not-decompose` 与 g1_e4c_pool_sd.py 的同名函数。
        """
        return float(np.mean(list(d_of(vec_p, seeds, idx).values())))

    # ---------- 2. 16 批 ----------
    # 每批记住**它自己那次评估的 D** —— n_gt 是预测的属性, 只存在各自的 P 里,
    # 合并池上没有 (合并池只为 d_of 服务)。
    batches = []                        # [(名字, 源 D, [cids])]
    for i in range(len(e4) // B_BATCH):
        batches.append((f"e4#{i}", D_all["e4"],
                        e4[i * B_BATCH:(i + 1) * B_BATCH]))
    batches.append(("e4b#0", D_all["e4b"], e4b))
    for j, b in enumerate(e4c_b):
        D = paired(f"e4c{j:02d}")
        batches.append((f"e4c{j:02d}", D, b))
    J = len(batches)
    assert J == 16, f"应有 16 批, 实得 {J}"

    print(f"\n{'批':>8}{'块数':>7}{'事件':>6}{'ΔF1':>10}")
    vals, ngts = [], []
    for nm, D, chunk in batches:
        idx = [pos_p[c] for c in chunk]
        v = df1_of(idx)
        ngt = sum(D["P"][("own10", D["seeds"][0])]["per_chunk"][c]["n_gt"]
                  for c in chunk)
        print(f"{nm:>8}{len(idx):>7}{ngt:>6}{v:>+10.4f}")
        vals.append(v); ngts.append(ngt)

    vals = np.array(vals)
    SD = float(vals.std(ddof=1))
    df = J - 1

    # ---------- 3. 零模型 (在 6400 块上分层) ----------
    print(f"\n=== 零模型 (分层 bootstrap, 每 key 重抽 2 块, 与一批同构) ===")
    se_str = strat_boot_se(vec_p, cids_p, seeds, 2, 2000)
    print(f"  SE_null_strat(400) = {se_str:.5f}")
    rho = SD / se_str
    chi2_stat = df * rho ** 2
    crit = float(chi2.ppf(0.95, df))
    print(f"\n=== 主统计量 (J={J}, df={df}) ===")
    print(f"  SD_batch(400) = {SD:.5f}")
    print(f"  **ρ = {rho:.3f}**   (J−1)ρ² = {chi2_stat:.2f} vs 临界 {crit:.2f} "
          f"⇒ {'🔴 拒绝 H0' if chi2_stat > crit else '✅ 不拒绝 H0'}")
    lo = rho * np.sqrt(df / float(chi2.ppf(0.975, df)))
    hi = rho * np.sqrt(df / float(chi2.ppf(0.025, df)))
    print(f"  95% CI: ρ ∈ [{lo:.3f}, {hi:.3f}]")
    print(f"  (预登记 J=12 时是 0.906, CI [0.641, 1.537]; 加 4 批后上界"
          f" {'已落到 1.3 以下 ✅' if hi <= 1.3 else '仍越过 1.3 ⚠️'})")

    # ---------- 4. 池化 ΔF1 与「ΔF1>0 是否被确立」 ----------
    mean_b = float(vals.mean())
    se_mean = SD / np.sqrt(J)
    pooled = df1_of(list(range(n_pool)))
    print(f"\n=== 池化 ΔF1 (6400 块) ===")
    print(f"  16 批均值的均值 = {mean_b:+.4f} ± {se_mean:.4f} (SE = SD/√{J}) "
          f"⇒ t = {mean_b/se_mean:.2f}")
    print(f"  按 d_of 直接算的并集 ΔF1 = {pooled:+.4f}  (两者应很接近; "
          f"F1 是比值, 严格说不相等)")
    print(f"  95% CI(批均值口径): [{mean_b-1.96*se_mean:+.4f}, "
          f"{mean_b+1.96*se_mean:+.4f}]")

    # ---------- 4b. 方差分解: 上面那个 SE 只算了「块」这一路 ----------
    # ⚠️ `SD/√J` 把 16 个批当唯一随机源, 但 7 个种子在**所有 16 批上是同一批种子**
    # ⇒ 加批**一点也**缩不了种子那一路的 SE。这正是 memory
    # `permutation-test-covers-one-variance-source` 记的「只重抽块漏掉种子 SE:
    # 19.9→合并 42.4」。两路都算, 才敢把 t 报出去。
    PS = np.array([list(d_of(vec_p, seeds, [pos_p[c] for c in chunk]).values())
                   for _nm, _D, chunk in batches])          # [J, n_seeds]
    ns = PS.shape[1]
    grand = float(PS.mean())
    m_s = PS.mean(axis=0)                                   # 每个种子跨 16 批
    m_b = PS.mean(axis=1)                                   # 每个批跨 7 种子
    ss_seed = float(J * ((m_s - grand) ** 2).sum())
    ss_batch = float(ns * ((m_b - grand) ** 2).sum())
    ss_tot = float(((PS - grand) ** 2).sum())
    ss_res = ss_tot - ss_seed - ss_batch
    df_s, df_b, df_r = ns - 1, J - 1, (J - 1) * (ns - 1)
    ms_s, ms_b, ms_r = ss_seed / df_s, ss_batch / df_b, ss_res / df_r
    # 两向随机效应: d_bs = μ + a_b + c_s + e_bs
    var_a = max(ms_b - ms_r, 0.0) / ns        # 批间
    var_c = max(ms_s - ms_r, 0.0) / J         # 种子间
    var_e = ms_r                              # 残差
    se_full = float(np.sqrt(var_a / J + var_c / ns + var_e / (J * ns)))
    print(f"\n=== 方差分解 (两向随机效应: 批 × 种子) ===")
    print(f"  {'源':<10}{'SD':>10}{'占比':>9}")
    tot = var_a + var_c + var_e
    for nm, v in (("批间 a", var_a), ("种子间 c", var_c), ("残差 e", var_e)):
        print(f"  {nm:<10}{np.sqrt(v):>10.5f}{v/tot:>9.1%}")
    print(f"  只算块路 SE = {se_mean:.4f} ⇒ t = {mean_b/se_mean:.2f}  "
          f"⚠️ **偏乐观**")
    print(f"  两路都算 SE = {se_full:.4f} ⇒ t = {grand/se_full:.2f}  ← 报这个")
    print(f"  (± {1.96*se_full:.4f} ⇒ 95% CI "
          f"[{grand-1.96*se_full:+.4f}, {grand+1.96*se_full:+.4f}])")
    print(f"  ⚠️ 这是**事前没登记**的推断。§5.5c 说「未确立」用的是 e4x 一侧的"
          f"块 bootstrap (t=1.667); 这里把 32 轮都用上了, 但**批数买不到种子"
          f"的精度** —— 两句话的差别在「用了几轮」, 不在「种子更可信」。")

    # ---------- 4c. 加块的**地板**: SE 不随 k→∞ 归零 ----------
    # 预登记 §1.6 把下一步说成「还要多少块」的算术题, 那句算术隐含 SE(k)→0。
    # 但 7 个种子在每一批上都是同 7 个 ⇒ var_c 是一个**不随 k 缩小**的项:
    #   SE(k) = sqrt(var_a·400/k + var_c/ns + var_e·400/(ns·k))
    #                               ^^^^^^^^^^^ 这一项是地板。
    floor_se = float(np.sqrt(var_c / ns))
    print(f"\n=== 加块的天花板: 种子地板 ===")
    print(f"  SE(k→∞) → √(var_c/{ns}) = {floor_se:.5f}")
    print(f"  ⇒ 「均值 > 2×SE」这条判据**最多**只能做到 2×{floor_se:.5f} = "
          f"{2*floor_se:.5f}")
    print(f"  实测 ΔF1 = {grand:+.4f} ⇒ "
          f"{'✅ 在地板之上, 加块仍然有用' if grand > 2*floor_se else '🔴 在地板之下, 加块**永远**达不到'}")
    k_min = float("inf")
    if grand > 2 * floor_se:
        # 反解 SE(k) = |ΔF1|/2
        rem = (abs(grand) / 2) ** 2 - var_c / ns
        per_k = B_BATCH * (var_a + var_e / ns)      # SE(k)² = per_k/k + var_c/ns
        k_min = per_k / rem if rem > 0 else float("inf")
        k_naive = B_BATCH * (2 * SD / abs(grand)) ** 2
        print(f"  带地板的 k_min ≈ {k_min:,.0f} 块 (无地板的老算法给 "
              f"{k_naive:,.0f} 块; 后者把地板当 0, 系统性乐观)")
        print(f"  ⚠️ 两者都用**同一批数据估出来的 ΔF1**, 所以仍是事前算术, "
              f"不是对已观测效应的检验 (memory: 先填表后核数)。")

    # ---------- 5. 批序 / 事件数 (沿用预登记的诊断) ----------
    js = np.arange(J, dtype=float)
    r = float(np.corrcoef(js, vals)[0, 1])
    rng = np.random.default_rng(0)
    null_r = np.array([abs(np.corrcoef(js, rng.permutation(vals))[0, 1])
                       for _ in range(20000)])
    print(f"\n=== 批序 vs ΔF1 ===\n  r = {r:+.3f} (n={J}), 置换 p = "
          f"{(null_r >= abs(r)).mean():.3f}")
    r_ev = float(np.corrcoef(np.array(ngts, dtype=float), vals)[0, 1])
    print(f"  事件数 vs ΔF1: r = {r_ev:+.3f}")

    json.dump({"n_batches": J, "n_pool": n_pool, "df": df,
               "batch_names": [nm for nm, _D, _c in batches],
               "delta_f1": vals.tolist(), "n_gt": ngts,
               "sd_batch_400": SD, "se_null_strat_400": se_str,
               "rho": rho, "chi2_stat": chi2_stat, "crit": crit,
               "rho_ci": [float(lo), float(hi)],
               "mean_of_batches": mean_b, "se_mean": se_mean,
               "pooled_delta_f1": pooled, "batch_order_r": r,
               "n_gt_r": r_ev,
               "var_decomp": {"var_batch": var_a, "var_seed": var_c,
                              "var_resid": var_e, "n_seeds": ns,
                              "se_block_only": se_mean, "se_full": se_full,
                              "grand_mean": grand, "df": [df_s, df_b, df_r],
                              "ms": [ms_s, ms_b, ms_r]},
               "se_floor": floor_se,
               "k_min_with_floor": (None if k_min == float("inf") else k_min),
               "k_min_naive": float(B_BATCH * (2 * SD / abs(grand)) ** 2),
               "caveat": ("事后扩展, 非预登记。e4/e4b/e4c 是三段不同时刻的评估, "
                          "批身份与评估时刻在段间混杂。")},
              open(f"{ANNOT}/g1_e4c_pool_sd_ext.json", "w"), indent=1)
    print(f"\n-> {ANNOT}/g1_e4c_pool_sd_ext.json")


if __name__ == "__main__":
    main()
