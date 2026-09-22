"""§5 的**流行率修正**预测 —— 纯读盘, 0 GPU, 结果落地前先算。

## 为什么要补这一步

`g1_eval4_predict.py` 的 t = 2.42~2.66 是在**旧 300 集**上拟合的:
那里正例率 6.33%、F1≈0.15。k=4 集是**自然流行率** 3.5%（42/1200 块）。

k=4 的首个产物已经暴露了量级差:

    own10 ds=42   旧300: F1=0.1504  P=0.088  R=0.500
                  k=4  : F1=0.0866  P=0.048  R=0.442

**recall 几乎没动 (0.500→0.442), precision 腰斩 (0.088→0.048)。**
也就是说这个 F1 差主要是**流行率效应**, 不是模型变差: 模型的正例率
基本恒定 (~35%), base rate 减半, precision 就跟着减半。

⇒ §5 那个拟合的 **prefactor 是在 F1≈0.15 的世界里定的**, 直接搬到
F1≈0.087 的世界里外推, 标度指数可能还对, 但**绝对标度会偏**。
与其等结果出来再找补, 不如现在把它算出来, 让它成为**预登记**。

## 方法

旧 300 的逐块 tp/fp 里, 天然就存着**负例块的 fp 分布**（281 块中
100 块有误报, 均值 0.356/块）—— 这正是自然流行率集里占 96% 的那部分。
所以可以直接拿旧集的两类块当**经验分布**, 按自然流行率**分层重采样**:

    每个槽位独立地:  以 p_pos 概率从「正例块池」抽, 否则从「负例块池」抽

- **K=300 / p_pos=旧集自身流行率** → 必须复现 SE_chunk = 0.0198（自检）
- **K=1200 / p_pos=3.5%**        → 这就是对 k=4 的修正预测

## 纪律

**同一份重抽块索引必须用在全部 7 个种子上。** 7 个种子评的是同一批块,
块抽样在种子间是**共有**的方差源; 若每个种子独立重抽, 这部分方差会被
当成噪声约掉, SE_chunk 会系统性偏小。

用法:
  python scripts/g1_e4_prevalence_predict.py            # 自检 + 修正预测
  python scripts/g1_e4_prevalence_predict.py --k 1200
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

# §7.2 已知量, 用于自检
BASE_SE_CHUNK = 0.0198
BASE_SE_SEED = 0.0117

# k=4 实测的构成 (g1_eval4_meta.json / 首个产物)
K4_BLOCKS = 1200
K4_POS_BLOCKS = 42


def suffix(seed):
    return "" if seed == 42 else f"_s{seed}"


def load_vecs(tag_set="filt"):
    """读出逐块 tp/fp/n_gt, 排成 [pos..., neg...] 的定序池。

    返回 (tp, fp, gt, n_pos, n_neg), 数组形状 (2 臂 * 7 种子, n_blocks)。
    """
    ev = {}
    for arm in ARMS:
        for s in SEEDS:
            p = f"{ANNOT}/g1_eval_{arm}{suffix(s)}_{tag_set}.json"
            if not os.path.exists(p):
                raise SystemExit(f"缺 {p} —— 自检需要完整 14 个旧集产物")
            ev[(arm, s)] = next(iter(json.load(open(p)).values()))

    keys = list(ev)
    cids = sorted(set.intersection(*(set(ev[k]["per_chunk"]) for k in keys)))
    # 按「是否含事件」定序, 让正例块集中在前面
    pos = [c for c in cids if ev[keys[0]]["per_chunk"][c]["n_gt"] > 0]
    neg = [c for c in cids if ev[keys[0]]["per_chunk"][c]["n_gt"] == 0]
    order = pos + neg

    tp = np.array([[ev[k]["per_chunk"][c]["tp"] for c in order] for k in keys],
                  dtype=np.float64)
    fp = np.array([[ev[k]["per_chunk"][c]["fp"] for c in order] for k in keys],
                  dtype=np.float64)
    gt = np.array([[ev[k]["per_chunk"][c]["n_gt"] for c in order] for k in keys],
                  dtype=np.float64)
    return tp, fp, gt, len(pos), len(neg), keys


def f1_vec(tp, fp, gt, idx):
    """向量化 F1: idx 是 (B, K) 的块索引, 返回 **(B, n_models)** 的 F1。

    注意 `tp[:, idx]` 的形状是 (n_models, B, K) —— 花式索引把 idx 的两维
    整体插在原有的第 1 维位置上, 所以 sum(axis=2) 得到 (n_models, B),
    **不是** (B, n_models)。这里必须转置: 下游 `se_chunk_of` 按
    (B, n_models) 切分臂, 若维度反了它会去切 bootstrap 那一维,
    把 2000 次重抽劈成两半相减 —— 重抽方差当场被平均掉,
    SE_chunk 假性地小 30 倍。
    """
    t = tp[:, idx].sum(axis=2)      # (n_models, B, K) -> (n_models, B)
    f = fp[:, idx].sum(axis=2)
    g = gt[:, idx].sum(axis=2)
    p = t / np.maximum(1.0, t + f)
    r = t / np.maximum(1.0, g)
    s = p + r
    out = np.where(s > 0, 2 * p * r / np.where(s > 0, s, 1), 0.0)
    assert out.shape == (tp.shape[0], idx.shape[0]), out.shape
    return out.T                    # -> (B, n_models)


def sim(tp, fp, gt, n_pos, n_neg, K, p_pos, B, rng):
    """分层重抽 K 块, 返回 (B, n_models) 的 F1。"""
    n_mod = tp.shape[0]
    n_blocks = n_pos + n_neg
    # 每个槽位: 先定是不是正例块, 再从对应池里抽（正例池 = [0,n_pos)）
    is_pos = rng.random((B, K)) < p_pos
    j_pos = rng.integers(0, n_pos, (B, K))
    j_neg = n_pos + rng.integers(0, n_neg, (B, K))
    idx = np.where(is_pos, j_pos, j_neg)          # (B, K)
    assert idx.min() >= 0 and idx.max() < n_blocks
    return f1_vec(tp, fp, gt, idx)                # (B, n_mod)


def se_chunk_of(f1):
    """f1: (B, n_models), 前 7 个是 own10, 后 7 个是 mixnorm(同序种子)。

    关键: 同一次重抽里两个臂用的是**同一份块**, 所以配对差已经约掉了
    块抽样在臂间的共有部分 —— 这正是配对比较的意义。
    """
    # 断言挡住「维度反了」: 第 1 维必须是模型维 (2 臂 × 7 种子 = 14),
    # 第 0 维必须比它大得多 (bootstrap 次数)。反了的话下面切的是重抽维,
    # 静默给出小 30 倍的 SE_chunk —— 第一版就是这么错的。
    assert f1.shape[1] == 2 * len(SEEDS), f"模型维应为 {2*len(SEEDS)}, 得到 {f1.shape}"
    assert f1.shape[0] > f1.shape[1], f"第 0 维应是 bootstrap 维, 得到 {f1.shape}"
    n = f1.shape[1] // 2
    d = f1[:, :n] - f1[:, n:]                     # (B, n_seeds)
    return d.mean(axis=1)                          # (B,) 每个重抽的 ΔF1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=K4_BLOCKS)
    ap.add_argument("--boot", type=int, default=B)
    args = ap.parse_args()

    tp, fp, gt, n_pos, n_neg, keys = load_vecs("filt")
    n_blocks = n_pos + n_neg
    p_old = n_pos / n_blocks
    print("=" * 74)
    print("  流行率修正预测 —— 用旧 300 的两类块作经验分布, 分层重采样")
    print("=" * 74)
    print(f"  旧集: {n_blocks} 块 = {n_pos} 正例块 + {n_neg} 负例块"
          f"  (正例率 {p_old:.2%})")

    rng = np.random.default_rng(0)

    # ---------- 自检: K=300, p_pos=旧集自身 ----------
    print(f"\n=== 自检: K={n_blocks}, p_pos={p_old:.4f} (旧集自身口径) ===")
    f1 = sim(tp, fp, gt, n_pos, n_neg, n_blocks, p_old, args.boot, rng)
    db = se_chunk_of(f1)
    se = db.std(ddof=1)
    got_d = db.mean()
    print(f"  复现 SE_chunk = {se:.4f}   期望 {BASE_SE_CHUNK:.4f}  "
          f"{'✅' if abs(se - BASE_SE_CHUNK) < 0.006 else '❌ 差 %.4f' % (se - BASE_SE_CHUNK)}")
    print(f"  (同一口径下 ΔF1 的均值 = {got_d:+.4f})")

    # ---------- 修正预测: K=1200, 自然流行率 ----------
    p_k4 = K4_POS_BLOCKS / K4_BLOCKS
    print(f"\n=== 修正预测: K={args.k}, p_pos={p_k4:.4f} "
          f"(k=4 实测 {K4_POS_BLOCKS}/{K4_BLOCKS}) ===")
    f1k = sim(tp, fp, gt, n_pos, n_neg, args.k, p_k4, args.boot, rng)
    dk = se_chunk_of(f1k)
    se_k = dk.std(ddof=1)
    se_all = float(np.hypot(BASE_SE_SEED, se_k))
    d_k = dk.mean()

    print(f"  SE_chunk(预测) = {se_k:.4f}")
    print(f"  合并 SE        = hypot({BASE_SE_SEED:.4f}, {se_k:.4f}) = {se_all:.4f}")
    print(f"\n  预测的 ΔF1     = {d_k:+.4f}   (旧集实测 +0.0340)")
    print(f"  t = ΔF1/SE     = {d_k/se_all:.2f}   "
          f"<- 对比: 原 §5 预测 2.42~2.66, 现状 1.48")

    # ---------- 迁移性检验: 拿 k=4 已到货的 adapter 验 ----------
    # 整套修正都建立在一个假设上: **k=4 的块与旧集的块, 逐块 tp/fp 行为可比**。
    # 这个假设不是自明的 —— 新集换了一批会话、且零时间重叠。
    # 好在 k=4 已经到货 1 个 adapter (own10 ds=42), 可以当场验:
    # 用旧集推出来的 F1 分布, 与实测的 F1 对不对得上。
    print(f"\n=== 迁移性检验: 模拟的 F1 能否命中 k=4 的实测值 ===")
    got_own = f1k[:, 0].mean()          # 模型 0 = own10 ds42
    got_mix = f1k[:, len(SEEDS)].mean()  # 对应 mixnorm ds42
    real_p = f"{ANNOT}/g1_eval_own10_e4.json"
    if os.path.exists(real_p):
        r = next(iter(json.load(open(real_p)).values()))
        rp = f"{ANNOT}/g1_eval_mixnorm_e4.json"
        rm = (next(iter(json.load(open(rp)).values()))
              if os.path.exists(rp) else None)
        print(f"  own10 ds=42   模拟 {got_own:.4f}   实测 {r['f1']:.4f}   "
              f"差 {got_own - r['f1']:+.4f}")
        if rm:
            print(f"  mixnorm ds=42 模拟 {got_mix:.4f}   实测 {rm['f1']:.4f}   "
                  f"差 {got_mix - rm['f1']:+.4f}")
        print(f"  (实测精度: n_gt={r['n_gt']}, n_pred={r['n_pred']}, "
              f"正例率 {r['n_gt']/r['n_chunks']:.2%})")
    else:
        print("  k=4 首个产物还没到货, 跳过")

    # ---------- 标度: 自然流行率下指数还是不是 −0.672 ----------
    print(f"\n=== 标度复核: 自然流行率下 SE_chunk 随 K 怎么走 ===")
    print(f"  {'K':>6}{'SE_chunk':>12}{'相对 K=300':>14}")
    base = None
    pts = []
    for K in (300, 600, 1200, 2400):
        f = sim(tp, fp, gt, n_pos, n_neg, K, p_k4, args.boot, rng)
        s = se_chunk_of(f).std(ddof=1)
        pts.append((K, s))
        if base is None:
            base = s
        print(f"  {K:>6}{s:>12.4f}{s/base:>14.3f}")
    ks = np.array([p[0] for p in pts], float)
    ss = np.array([p[1] for p in pts], float)
    sl = np.polyfit(np.log(ks), np.log(ss), 1)[0]
    print(f"\n  拟合指数 = {sl:+.3f}   (旧集上拟合的是 −0.672)")
    print(f"  {'✅ 指数稳定, 修正只在 prefactor' if abs(sl + 0.672) < 0.15 else '⚠️ 指数也变了 —— 说明分层重采样改变了方差的来源结构, 需要重新审视方法'}")

    print("\n" + "=" * 74)
    print("  预登记: 这是**结果落地前**算出的修正预测。")
    print("  k=4 跑完后先报实测 SE_chunk, 再和上面这个数对照 ——")
    print("  对不上就如实报对不上, 不要用这个预测去替代实测。")


if __name__ == "__main__":
    main()
