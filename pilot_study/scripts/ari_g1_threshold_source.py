"""Q1 + Q2: 把「各种子阈值离散」拆成 音频效应 vs 种子效应。

## 为什么能做这个拆分

`ari_g1_seed_recon.py` 测出:
  - 所有 run 的 `data_seed = None`  -> 样本顺序由 `seed` 决定
  - 同 seed 跨臂的 lora_A 相对差 0.111, 而跨 seed 同臂是 1.414 (= √2,
    两个独立随机向量的期望距离) -> **同 seed 的两臂共享 LoRA 初值**

所以对每个 seed, own10_s<seed> 与 mixnorm_s<seed> 构成一对
**同初值 + 同数据顺序 + 同超参, 唯一差异是音频内容** 的对照。

于是任一观测量 x 可以做配对分解:
    s_i = (m_i + o_i) / 2     配对均值 -> 种子层面的运气 (初值/顺序)
    d_i = m_i - o_i           配对差   -> 音频本身
    Var(m) ≈ Var(s) + Var(d)/4
两个分量谁大, 就说明阈值离散主要来自哪一边。

## 注意这个拆分的边界

它**不能**把「初值」和「数据顺序」分开 —— 两者都随 seed 一起变。
要分开需要一个 data_seed 实验 (Q3, 需重训, 未授权)。这里只回答
「音频 vs 种子」这一层, 而这一层恰好是 §5 留下的那个问题。

零成本: 纯读盘。用法:
  /share/home/zhuangruicen/miniconda3/envs/fd_analysis/bin/python \
      scripts/ari_g1_threshold_source.py
"""
import json
import math
import os

import numpy as np

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
SEEDS = [42, 1234, 7, 2024, 3407, 31337, 55555]
ARMS = ["own10", "mixnorm"]


def suffix(seed):
    return "" if seed == 42 else f"_s{seed}"


def load_margin(arm, seed):
    p = f"{ANNOT}/g1_margin_{arm}_s{seed}.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def load_eval(arm, seed, kind):
    p = f"{ANNOT}/g1_eval_{arm}{suffix(seed)}{kind}.json"
    if not os.path.exists(p):
        return None
    return next(iter(json.load(open(p)).values()))


def decompose(label, own, mix):
    """own/mix: {seed: float}。返回配对分解。"""
    seeds = [s for s in SEEDS if s in own and s in mix]
    o = np.array([own[s] for s in seeds])
    m = np.array([mix[s] for s in seeds])
    s_ = (o + m) / 2
    d_ = m - o
    v_m, v_o = m.var(ddof=1), o.var(ddof=1)
    v_seed = s_.var(ddof=1)          # 种子层 (初值 + 顺序)
    v_audio = d_.var(ddof=1) / 4     # 音频层 (配对差 / 4)
    print(f"\n  ── {label} ──")
    print(f"    own10  均值 {o.mean():+.4f}  SD {o.std(ddof=1):.4f}")
    print(f"    mixnorm均值 {m.mean():+.4f}  SD {m.std(ddof=1):.4f}")
    print(f"    配对差 d = mixnorm − own10 : {d_.mean():+.4f} ± {d_.std(ddof=1):.4f}")
    print(f"      逐种子 d: " + "  ".join(f"{x:+.3f}" for x in d_))
    tot = v_m if v_m > 0 else 1e-12
    print(f"    方差分解 (以 mixnorm 的 Var={v_m:.4f} 为分母):")
    print(f"      种子层 Var(s)   = {v_seed:.4f}  占 {v_seed/tot*100:5.1f}%")
    print(f"      音频层 Var(d)/4 = {v_audio:.4f}  占 {v_audio/tot*100:5.1f}%")
    tot_o = v_o if v_o > 0 else 1e-12
    print(f"    (以 own10 的 Var={v_o:.4f} 为分母: 种子层 {v_seed/tot_o*100:.1f}%"
          f"  音频层 {v_audio/tot_o*100:.1f}%)")
    # 配对检验: 音频效应本身显不显著
    if len(d_) >= 3:
        t = d_.mean() / (d_.std(ddof=1) / np.sqrt(len(d_)))
        signs = np.array([[1 if (x >> i) & 1 else -1 for i in range(len(d_))]
                          for x in range(2 ** len(d_))])
        perm = np.abs((signs * d_).mean(1))
        p = np.mean(perm >= abs(d_.mean()) - 1e-12)
        print(f"    配对检验 d: t({len(d_)-1}) = {t:+.2f}   精确置换 p = {p:.4f}")
    return {"seeds": seeds, "own": o, "mix": m, "d": d_,
            "v_seed": v_seed, "v_audio": v_audio}


def loo(d_):
    """留一: 去掉每个种子后配对 p 变成多少。"""
    n = len(d_)
    out = []
    for i in range(n):
        dd = np.delete(d_, i)
        signs = np.array([[1 if (x >> j) & 1 else -1 for j in range(len(dd))]
                          for x in range(2 ** len(dd))])
        perm = np.abs((signs * dd).mean(1))
        p = np.mean(perm >= abs(dd.mean()) - 1e-12)
        out.append((SEEDS[i], dd.mean(), p))
    return out


def main():
    print("=" * 84)
    print("  Q1: 各种子平均 margin 的离散 —— 来自音频, 还是来自种子?")
    print("=" * 84)

    res = {}
    own_mm, mix_mm = {}, {}
    for arm, store in (("own10", own_mm), ("mixnorm", mix_mm)):
        for s in SEEDS:
            m = load_margin(arm, s)
            if m is None:
                continue
            v = np.array([x["margin_max"] for x in m.values()])
            store[s] = float(v.mean())
    decompose("各种子【平均 margin】(E2 的 400 块)", own_mm, mix_mm)

    # 同样拆 F1
    for kind, name in (("_e2", "E2 事件密集集"), ("_filt", "旧 300 自然流行率集")):
        own_f1, mix_f1 = {}, {}
        for arm, store in (("own10", own_f1), ("mixnorm", mix_f1)):
            for s in SEEDS:
                e = load_eval(arm, s, kind)
                if e is not None:
                    store[s] = float(e["f1"])
        if own_f1 and mix_f1:
            decompose(f"{name} 的 F1", own_f1, mix_f1)
            res[kind] = (own_f1, mix_f1)

    # 报的比例
    for kind, name in (("_e2", "E2 事件密集集"), ("_filt", "旧 300 自然流行率集")):
        own_fr, mix_fr = {}, {}
        for arm, store in (("own10", own_fr), ("mixnorm", mix_fr)):
            for s in SEEDS:
                e = load_eval(arm, s, kind)
                if e is None:
                    continue
                pc = e["per_chunk"]
                store[s] = float(np.mean([1 if pc[c]["n_pred"] > 0 else 0
                                          for c in pc]))
        if own_fr and mix_fr:
            decompose(f"{name} 的【报的比例】", own_fr, mix_fr)

    print("\n" + "=" * 84)
    print("  Q2: ΔF1 ≈ +0.034 站得住吗 (旧 300 自然流行率集)")
    print("=" * 84)
    if "_filt" in res:
        own_f1, mix_f1 = res["_filt"]
        seeds = [s for s in SEEDS if s in own_f1 and s in mix_f1]
        d_ = np.array([own_f1[s] - mix_f1[s] for s in seeds])
        print(f"  配对 ΔF1(own10−mixnorm) = {d_.mean():+.4f} ± {d_.std(ddof=1):.4f}"
              f"   n={len(d_)}")
        print(f"  逐种子: " + "  ".join(f"{x:+.4f}" for x in d_))
        print(f"\n  留一稳健性 (去掉一个种子后):")
        for s, mu, p in loo(d_):
            flag = "  <-- 掉出 0.05" if p > 0.05 else ""
            print(f"    去掉 seed {s:<6} ΔF1 = {mu:+.4f}   p = {p:.4f}{flag}")
        # 这个检验的**分辨率下限**。配对符号翻转检验是离散的: n 对只有 2^n 个
        # 符号模式, 两尾最小可能 p = 2/2^n。所以「还需要多少对」不能用正态近似
        # 反推 —— 那会算出 n≈3, 而事实是 n=7 实测才 p≈0.047, 自相矛盾。
        # 观测 p 落在多少个模式上, 才是「这个结论有多脆」的直接度量。
        signs = np.array([[1 if (x >> i) & 1 else -1 for i in range(len(d_))]
                          for x in range(2 ** len(d_))])
        perm = np.abs((signs * d_).mean(1))
        n_ext = int(np.sum(perm >= abs(d_.mean()) - 1e-12))
        n_tot = 2 ** len(d_)
        print(f"\n  检验的分辨率 (这比「需要多少对」更能说明脆弱程度):")
        print(f"    n={len(d_)} 对的符号翻转检验只有 {n_tot} 个符号模式, "
              f"两尾最小可能 p = 2/{n_tot} = {2/n_tot:.4f}")
        print(f"    观测落在 {n_ext}/{n_tot} 个模式上 -> p = {n_ext/n_tot:.4f}"
              f"  (已在分布的尾部, 但尾部只有 {n_ext} 个原子)")
        n_conc = int((d_ > 0).sum())
        print(f"    符号检验 (只看方向不看大小): {n_conc}/{len(d_)} 对同向, "
              f"两尾 p = {2*sum(math.comb(len(d_), k) for k in range(n_conc, len(d_)+1))/n_tot:.4f}")
        print(f"    -> n 每加一对, 最小可能 p 减半: ", end="")
        print("  ".join(f"n={nn}:{2/2**nn:.3f}" for nn in (8, 9, 10, 12)))
        print(f"\n  同期 E2 上的配对 ΔF1:")
        if "_e2" in res:
            o2, m2 = res["_e2"]
            s2 = [s for s in SEEDS if s in o2 and s in m2]
            d2 = np.array([o2[s] - m2[s] for s in s2])
            print(f"    {d2.mean():+.4f} ± {d2.std(ddof=1):.4f}   n={len(d2)}")


if __name__ == "__main__":
    main()
