"""Q2 决策: 「扩大自然流行率评估集」到底能不能救 ΔF1 ≈ +0.034?

## 为什么要先算这个

Q2 的结论是「弱显著」(配对 ΔF1 = +0.0340 ± 0.0309, n=7, p=0.0469)。
用户提的补救是「更多种子或更大的自然流行率集」。前者要重训 (未授权),
后者不要 —— 但**很贵**: 生成式评估实测 1.666 s/块, 14 个 adapter 跑 3000 块
要 ~19 GPU 小时。

在花这 19 小时之前, 必须先问一个零成本的问题:
**那 0.0309 的离散, 到底是「评估集抽样噪声」还是「种子间真实差异」?**

  - 若是**抽样噪声**占大头 -> 换更大的自然集真的有用, 按 1/sqrt(k) 缩。
  - 若是**种子间真实差异**占大头 -> 换多大的评估集都没用 (只是把同一个
    效应量测得更准), 唯一出路是更多种子, 而那是重训。

判据: 配对检验关心的是 d̄ = mean_s d_s 的标准误 SE = SD/√7 = 0.0117。
用**块级 bootstrap** (7 个种子共用同一份重抽的块 —— 因为它们本来评的就是
同一批块) 估 Var(d̄) 里来自评估集抽样的那一份,
再和 SE 比: 接近 -> 抽样噪声主导; 远小于 -> 种子差异主导。

## 顺带

bootstrap 还能给出「事件数扩到 E 倍时 SD 会变成多少」的外推,
于是可以反过来算: 要多少事件才够, 对应多少块、多少 GPU 小时。

用法: /share/home/zhuangruicen/miniconda3/envs/fd_analysis/bin/python \
        scripts/ari_g1_eval_power.py
"""
import json
import os

import numpy as np

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
SEEDS = [42, 1234, 7, 2024, 3407, 31337, 55555]
ARMS = ["own10", "mixnorm"]
B = 2000
SEC_PER_CHUNK = 1.666      # 实测生成式评估斜率


def suffix(seed):
    return "" if seed == 42 else f"_s{seed}"


def load_filt(arm, seed):
    p = f"{ANNOT}/g1_eval_{arm}{suffix(seed)}_filt.json"
    if not os.path.exists(p):
        return None
    return next(iter(json.load(open(p)).values()))


def f1_of(pc, idx):
    """给定块下标(可重复), 用存下来的 tp/fp/n_gt 重算 F1。"""
    tp = fp = gt = 0
    for i in idx:
        e = pc[i]
        tp += e["tp"]
        fp += e["fp"]
        gt += e["n_gt"]
    p = tp / max(1, tp + fp)
    r = tp / max(1, gt)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def main():
    P = {}
    for arm in ARMS:
        for s in SEEDS:
            e = load_filt(arm, s)
            if e is not None:
                P[(arm, s)] = e
    seeds = [s for s in SEEDS if ("own10", s) in P and ("mixnorm", s) in P]
    print(f"可用配对: {len(seeds)}  ->  {seeds}")

    # 块号必须一致 (同一批块)
    cids = sorted(set.intersection(*(set(P[k]["per_chunk"]) for k in P)))
    print(f"公共块: {len(cids)}")
    for k in P:
        P[k]["_vec"] = [P[k]["per_chunk"][c] for c in cids]

    n_gt = sum(P[("own10", seeds[0])]["_vec"][i]["n_gt"] for i in range(len(cids)))
    print(f"真值事件数 (own10 s{seeds[0]}): {n_gt}")

    # ---------- 观测配对差 ----------
    d_obs = np.array([f1_of(P[("own10", s)]["_vec"], range(len(cids)))
                      - f1_of(P[("mixnorm", s)]["_vec"], range(len(cids)))
                      for s in seeds])
    print(f"\n配对 ΔF1 (own10−mixnorm) = {d_obs.mean():+.4f} ± {d_obs.std(ddof=1):.4f}")
    print(f"  逐种子: " + "  ".join(f"{x:+.4f}" for x in d_obs))
    SE = d_obs.std(ddof=1) / np.sqrt(len(d_obs))
    print(f"  SE = SD/√{len(d_obs)} = {SE:.4f}")

    # ---------- 块级 bootstrap: 7 个种子共用同一份重抽块 ----------
    n = len(cids)
    rng = np.random.default_rng(0)
    db = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        db[b] = np.mean([f1_of(P[("own10", s)]["_vec"], idx)
                         - f1_of(P[("mixnorm", s)]["_vec"], idx) for s in seeds])
    sd_chunk = db.std(ddof=1)

    # 两个**不同**的随机来源, 不能混为一谈:
    #   SE_seed  = 7 个种子之间 d_s 的离散 / √7  —— 置换检验用的就是它
    #   SE_chunk = 重抽块时 d̄ 的离散          —— 置换检验**完全没算**它
    # 7 个种子评的是同一批块, 所以「换一批块会怎样」这件事在种子间离散里
    # 是看不见的: 它是所有种子**共有**的偏移, 在 across-seed SD 里被约掉了。
    # 也就是说, 只按种子做置换检验, 系统性地**低估**了真实不确定度。
    SE_seed, SE_chunk = SE, sd_chunk
    SE_all = float(np.hypot(SE_seed, SE_chunk))
    print(f"\n块级 bootstrap ({B} 次, 7 种子共用同一份重抽块):")
    print(f"  SE_seed  = {SE_seed:.4f}   (置换检验实际用的那个)")
    print(f"  SE_chunk = {SE_chunk:.4f}   (置换检验**没有**算的那个)")
    print(f"  合成 SE  = {SE_all:.4f}   (= hypot)")

    print("\n" + "=" * 78)
    print("  判据: 同一个 ΔF1, 用不同的不确定度去比")
    print("=" * 78)
    print(f"  只看种子 (现在的 p=0.0469 口径): t = {d_obs.mean()/SE_seed:.2f}")
    print(f"  只看块抽样                    : t = {d_obs.mean()/SE_chunk:.2f}")
    print(f"  两个都算                      : t = {d_obs.mean()/SE_all:.2f}")
    if SE_chunk > SE_seed:
        print(f"\n  ⚠️ 块抽样那一项 ({SE_chunk:.4f}) 比种子那一项 ({SE_seed:.4f}) **还大**。")
        print("     所以 p=0.0469 是**乐观的** —— 它把 20 个事件当成固定的, 只让")
        print("     种子随机。而 Q2 想问的「站得住吗」隐含的是能否推广到别的块,")
        print("     那就必须算上块抽样。算上之后这个效应量判不住。")

    # ---------- 外推: 事件数扩到 k 倍 ----------
    print("\n" + "=" * 78)
    print("  外推: 事件数扩到 k 倍后")
    print("=" * 78)
    var_chunk = SE_chunk ** 2
    var_seed = SE_seed ** 2
    print(f"  Var(种子) = {var_seed:.3e}  —— **不随事件数缩小** (它是模型的性质)")
    print(f"  Var(块抽样) = {var_chunk:.3e}  —— 按 1/k 缩")
    print(f"  块抽样要缩到和种子项一样大, 需要 k = {var_chunk/var_seed:.1f} 倍")
    print(f"\n  {'事件数':>7}{'块数':>8}{'GPU小时':>10}{'SE_chunk':>10}{'SE':>9}"
          f"{'t':>7}")
    for k in (1, 2, 4, 8, 16, 32, 64, 128):
        ev = n_gt * k
        blocks = int(round(len(cids) * k))
        hrs = blocks * 14 * SEC_PER_CHUNK / 3600
        sec = SE_chunk / np.sqrt(k)
        se = float(np.hypot(SE_seed, sec))
        print(f"  {ev:>7}{blocks:>8}{hrs:>10.1f}{sec:>10.4f}{se:>9.4f}"
              f"{d_obs.mean()/se:>7.2f}")

    t_inf = d_obs.mean() / SE_seed
    print(f"\n  **天花板**: k->inf 时 SE -> SE_seed = {SE_seed:.4f}, t -> {t_inf:.2f}")
    print(f"  也就是说, 把评估集扩到无穷大, 最好也只是回到**现在这个** t —— ")
    print(f"  因为种子那一项根本不随事件数缩。扩集不能把这个效应变显著,")
    print(f"  它只能让现在这个 p=0.0469 从「乐观」变成「名副其实」。")
    print(f"\n  真正能改判的只有一件事: 种子 55555 那个 −0.0345 到底是真反向")
    print(f"  还是噪声。若扩集后它翻正 -> 7/7 同向, 两尾最小 p = 0.0156 (触及")
    print(f"  n=7 的分辨率下限); 若它稳稳地负 -> 这个结论就该被放弃。")
    # 候选池规模**从产物读**, 不再硬编码。
    # 原先这里写死 "82544 块 / 4982 事件 (5.8%)" —— 那个数对应的是
    # `--used g1_ids.txt` 只排 2400 块、且没做时间排除的口径, 即 E2 (1600 块)
    # 还不算"已用"时算的。因为写死在 print 字符串里, 之后没有任何一次复跑
    # 能纠正它, 结果 §7.5 的 k=4 成本表按 ~80 事件外推, 实际只有 43 个。
    meta_p = f"{ANNOT}/g1_eval4_meta.json"
    if os.path.exists(meta_p):
        with open(meta_p) as f:
            m = json.load(f)
        print(f"\n  另: 自然流行率候选池 (来自 {os.path.basename(meta_p)}): "
              f"{m['candidates_available']} 块 / {m['candidate_events']} 事件 "
              f"({m['candidate_events']/m['candidates_available']:.2%} 正例),")
        print(f"      已排除 {m['excluded_pool_size']} 个已用块, "
              f"其中 {m['dropped_temporal_overlap']} 个因**时间重叠**再被弃。")
        print(f"      选定 k=4: {m['n_chunks']} 块 / {m['n_events']} 事件 "
              f"({m['prevalence']:.2%} 正例)。")
    else:
        print(f"\n  另: 候选池规模未知 —— 先跑 scripts/ari_g1_eval4_ids.py "
              f"生成 {os.path.basename(meta_p)}。")


if __name__ == "__main__":
    main()
