#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DuplexGen 标注: **slot 票数是否携带 slot 级信息** (0 GPU, 不需要 join)

—— 为什么需要这个脚本 (与 duplexgen_annot_kappa.py 的分工) ——

kappa 只说「没超出**边际**的一致」。DuplexGen 论文(Sec. 3)明确说他们
**不取多数票**, 而是把每个槽的 5 人票数当 **soft label**, 理由原话:
  "This reflects the subjectivity of turn-taking: multiple actions may be
   appropriate at a given slot, and a hard label discards information about
   human disagreement."
⇒ **低 kappa 不是对他们的反驳**, 他们从没声称人一致。

但上面那句话把整座楼架在一个**可检验的假设**上:
  「那个分歧分布携带关于**这个槽**的信息」。
kappa 没测这个。本脚本测。

—— 检验设计 ——

零模型 H0: 第 i 个槽的 backchannel 票数 n_i ~ Binomial(m_i, p),
            p = 该场景的总体 BC 基准率, 各槽独立。
            ⇒ Pearson 残差 z_i = (n_i - m_i p) / sqrt(m_i p (1-p))
            ⇒ 统计量 S = mean(z_i^2), 在 H0 下 = 1

⚠️ **聚类会推高 S**: 同一个对话内的槽共享对话级倾向, 各槽不独立。
   ⇒ 光比 S 与 1 会把聚类误读成 slot 信号。
   ⇒ 用**对话内置换**做零分布: 在每个对话内部打乱各槽的 BC 票数,
      保住 (a) 对话的总 BC 票数 → 对话级倾向不变, (b) 各槽的 m 不变,
      (c) 对话内槽数不变。**只破坏「哪一票落在哪个槽」**。
   ⇒ S 显著超出该零分布, 才是真的 slot 级信号。

⚠️ **功效对照 (必做)**: 本项目的老账 `power-analysis-before-ablation-runs`
   —— 不能只说「没检出」。脚本在**真实的对话/槽结构**上注入已知强度的
   合成 slot 信号, 报检出率 ⇒ 得到本设计的 MDE。否则「无信号」不可信。
   同时报 delta=0 的**尺寸 (size)** —— 零假设下拒绝率应 ≈5%,
   否则检验本身是反保守的, 「有信号」的结论也不可信。

口径 (沿用 duplexgen_annot_kappa.py):
  1. 先报流行率;
  2. 一切区间按**对话**重抽 (cluster bootstrap);
  3. 自检必须会炸。
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

CORPUS = "/share/workspace3/shared_dataset/duplexgen-corpus"
SCENARIOS = ["TEA", "PLN", "INT", "NEG", "PER", "SOC"]
CATS = ["silent", "backchannel", "take_floor"]
BC = 1  # CATS 里 backchannel 的下标
OUT_DIR = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
           "real_data/results/duplexgen_annot")


def load_slots(splits=("train", "test")):
    """返回 [(key, scenario, split, m, vec)]; 载入时逐槽断言 counts 闭合。"""
    rows = []
    for sc in SCENARIOS:
        for split in splits:
            path = f"{CORPUS}/annotations/{sc}/{split}.jsonl"
            if not os.path.exists(path):
                continue
            for line in open(path):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                key = f"{sc}/{split}/{d['example_id']}"
                for hi, turn in enumerate(d.get("history", [])):
                    for bi, sl in enumerate(turn.get("boundaries", []) or []):
                        m = int(sl["total_count"])
                        c = sl["counts"]
                        s = sum(int(v) for v in c.values())
                        if s != m:
                            raise AssertionError(
                                f"counts 之和 {s} != total_count {m} @ {key} "
                                f"turn{hi} slot{bi} -> {c}")
                        vec = np.array([int(c.get(k, 0)) for k in CATS], dtype=np.int64)
                        if int(vec.sum()) != m:
                            raise AssertionError(
                                f"类别分解不闭合 @ {key} turn{hi} slot{bi}")
                        rows.append((key, sc, split, m, vec))
    return rows


def group_indices(keys):
    """key -> 该对话所有 slot 的下标数组, 按 key 排序返回列表。"""
    d = defaultdict(list)
    for i, k in enumerate(keys):
        d[k].append(i)
    return [np.array(d[k], dtype=np.int64) for k in sorted(d)]


# ------------------------------------------------------------ 统计量

def S_stat(counts, m, p):
    """S = mean(z_i^2); counts/m/p 等长。H0 (纯二项) 下 = 1。"""
    z = (counts - m * p) / np.sqrt(m * p * (1.0 - p))
    return float(np.mean(z * z))


def perm_within_dialogue(groups, counts, m, rng):
    """对话内置换。

    🔴 **必须打乱「伯努利试验」而不是「票数」** —— 上一版直接对调各槽的
       counts, 会把 m=5 槽的 3 票挪到 m=2 的槽上 (2 个人里 3 个人说是),
       造出非法的 n>m, 把零分布抬到 1.40 而观测只有 1.04, **方向都反了**。
       功效对照 (delta=0 的 size 检验) 正是靠这个照出来的。

    正确做法: 把对话内 sum(m_i) 个 0/1 试验池化 → 洗牌 → 按各槽 m_i 切回。
    保住: 各槽 m、对话总票数。破坏: 「哪个试验落在这个槽」。
    """
    c = counts.copy()
    for g in groups:
        if len(g) > 1:
            mg = m[g].astype(np.int64)
            total = int(mg.sum())
            pool = np.zeros(total, dtype=np.int64)
            pos = 0
            for j, i in enumerate(g):
                k = int(counts[i])
                pool[pos:pos + k] = 1
                pos += int(mg[j])
            rng.shuffle(pool)
            pos = 0
            for j, i in enumerate(g):
                c[i] = pool[pos:pos + int(mg[j])].sum()
                pos += int(mg[j])
    return c


def perm_null(groups, counts, m, p, n_perm, rng):
    out = np.empty(n_perm, dtype=np.float64)
    for t in range(n_perm):
        out[t] = S_stat(perm_within_dialogue(groups, counts, m, rng), m, p)
    return out


def self_check_perm(keys, m, counts, rng):
    """置换的不变量自检 —— 必须会炸。"""
    groups = group_indices(keys)
    c1 = perm_within_dialogue(groups, counts, m, rng)
    if not np.allclose(c1, np.round(c1)):
        raise AssertionError("置换后票数不是整数")
    if (c1 < 0).any() or (c1 > m).any():
        bad = int(np.argmax(c1 > m))
        raise AssertionError(
            f"置换后出现非法票数 n>m: 槽{bad} n={c1[bad]:.0f} > m={m[bad]:.0f}")
    for g in groups:
        if int(np.round(c1[g].sum())) != int(np.round(counts[g].sum())):
            raise AssertionError("置换改变了对话总票数")
    if not (np.round(c1).sum() == np.round(counts).sum()):
        raise AssertionError("置换改变了总票数")
    return True


def cluster_boot_S(groups, counts, m, p, B, rng):
    """S 按**对话**重抽的 bootstrap 区间 (口径 2)。"""
    out = np.empty(B, dtype=np.float64)
    for b in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        sel = np.concatenate([groups[i] for i in pick])
        out[b] = S_stat(counts[sel], m[sel], p)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)),
            float(out.std(ddof=1)))


def analyse(tag, rows, cat, n_perm, boot, rng, lines, res_list):
    keys = [r[0] for r in rows]
    m = np.array([r[3] for r in rows], dtype=np.float64)
    counts = np.array([r[4][cat] for r in rows], dtype=np.float64)
    p = float(counts.sum() / m.sum())        # 🔴 先报流行率
    groups = group_indices(keys)
    obs = S_stat(counts, m, p)
    perm = perm_null(groups, counts, m, p, n_perm, rng)
    pval = float((np.sum(perm >= obs) + 1) / (n_perm + 1))
    lo, hi, sd = cluster_boot_S(groups, counts, m, p, boot, rng)

    lines.append(f"\n### {tag}   [{CATS[cat]}]")
    lines.append(f"  槽数 = {len(rows)}   对话数 = {len(groups)}   "
                 f"🔴 基准率 p = {p:.4f}  (总票 {int(counts.sum())}/{int(m.sum())})")
    lines.append(f"  观测 S = mean(z^2) = {obs:.4f}   (纯二项零模型 = 1.0)   "
                 f"bootstrap 95% CI [{lo:.4f}, {hi:.4f}] (SD={sd:.4f})")
    lines.append(f"  对话内置换零分布: 均值 {perm.mean():.4f}  "
                 f"[{np.percentile(perm, 2.5):.4f}, {np.percentile(perm, 97.5):.4f}]")
    lines.append(f"  **p = {pval:.4f}**  "
                 f"({'有' if pval < 0.05 else '无'}超出对话级倾向的 slot 级信号)")
    exc_c = float(perm.mean()) - 1.0
    lines.append(f"  分解: 总超额 (S-1) = {obs - 1:.4f}   聚类可解释 {exc_c:.4f} "
                 f"({exc_c / max(1e-9, obs - 1):.1%})   余下(slot 级) {obs - perm.mean():.4f}")
    res_list.append({
        "tag": tag, "cat": CATS[cat], "n_slots": len(rows),
        "n_dialogues": len(groups), "base_rate": p,
        "S_obs": obs, "S_ci_lo": lo, "S_ci_hi": hi, "S_boot_sd": sd,
        "S_null_mean": float(perm.mean()),
        "S_null_lo": float(np.percentile(perm, 2.5)),
        "S_null_hi": float(np.percentile(perm, 97.5)),
        "p_value": pval, "n_perm": n_perm,
        "excess_total": obs - 1.0, "excess_cluster": exc_c,
        "excess_slot": obs - float(perm.mean()),
    })


# ------------------------------------------------------------ 功效对照

def power_check(keys, m, base, n_rep, n_perm, rng, lines):
    """在**真实对话/槽结构**上注入已知强度的 slot 信号, 报检出率。

    "信号"定义为**重分配**而非加量: 让一部分槽的 BC 概率抬到 p_hi,
    其余槽按比例压低到 p_lo, 使总体均值**仍 = base**。
    (这与真实情形一致 —— 票数是固定的, 问题只在落在哪个槽。)

    frac=1.0 是退化情形 (所有槽一起抬高 ≡ 换了基准率), 故不列。
    delta=0 那一行是 **size 检验**: 零假设下拒绝率应 ≈ 5%。
    """
    groups = group_indices(keys)
    ms = np.array(m, dtype=np.float64)
    n_all = len(keys)
    lines.append(f"\n### 功效对照 (真实结构: {len(groups)} 对话 / {n_all} 槽, "
                 f"基准 p={base})")
    hdr = (f"  {'受影响槽':>8s} {'delta':>6s} {'抬升槽p':>8s} {'压低槽p':>8s} "
           f"{'观测S均值':>10s} {'置换零均值':>10s} {'拒绝率':>8s}")
    lines.append(hdr)
    out = []
    cells = [(0.0, 0.0)] + [(f, d) for f in (0.10, 0.25, 0.50)
                            for d in (0.10, 0.20, 0.35)]
    for frac, delta in cells:
        if frac == 0.0:
            probs = np.full(n_all, base)
            n_hi = 0
        else:
            n_hi = int(round(n_all * frac))
            p_hi = base + delta
            p_lo = (n_all * base - n_hi * p_hi) / (n_all - n_hi)
            if p_lo < 0.0 or p_hi > 1.0:
                lines.append(f"  {frac:>8.2f} {delta:>6.2f}   (p_lo={p_lo:.3f} 越界, 跳过)")
                continue
            probs = np.full(n_all, p_lo)
            # 随机散布受影响的槽 (避免集中在前 N 个而与对话结构共振)
            probs[rng.choice(n_all, size=n_hi, replace=False)] = p_hi
        rej, Ss, nulls = 0, [], []
        for _ in range(n_rep):
            c = rng.binomial(m.astype(np.int64), probs).astype(np.float64)
            S = S_stat(c, ms, base)
            null = perm_null(groups, c, ms, base, n_perm, rng)
            nulls.append(float(null.mean()))
            if (np.sum(null >= S) + 1) / (n_perm + 1) < 0.05:
                rej += 1
            Ss.append(S)
        rate = rej / n_rep
        # 🔴 上一版这里打印的是 base, 表头却写「零均值」—— 标签与实际值不符。
        #    修成真正的置换零分布均值 (合成数据无对话聚类, 应当 ≈1.0)。
        lines.append(f"  {frac:>8.2f} {delta:>6.2f} {probs.max():>8.3f} "
                     f"{probs.min():>8.3f} {np.mean(Ss):>10.3f} "
                     f"{np.mean(nulls):>10.3f} {rate:>7.1%}")
        out.append({"frac": frac, "delta": delta, "p_hi": float(probs.max()),
                    "p_lo": float(probs.min()), "S_mean": float(np.mean(Ss)),
                    "null_mean": float(np.mean(nulls)),
                    "power": rate, "n_rep": n_rep})
    lines.append("  ⚠️ 第一行 (delta=0) 是 size: 零假设下拒绝率应 ≈5%。")
    lines.append("  ⚠️ 拒绝率按 p<0.05 计, 未做多重比较校正。")
    lines.append(f"  ⚠️ 拒绝率的分辨率受限于 n_rep={n_rep}: "
                 f"SD = sqrt(p(1-p)/n_rep) ≈ {np.sqrt(0.05*0.95/n_rep):.1%} "
                 f"⇒ **本表只能粗判, size 请见下方专用校验**。")
    return out


def size_check(keys, m, base, n_rep, n_perm, rng, lines):
    """**专用 size 校验**: delta=0 下多重复, 把拒绝率估准。

    存在的理由: 上面那张功效表用 n_rep=30 报 size, 分辨率 ~4%,
    根本分不清「装置是反保守的」和「赶上运气了」。
    而 size 是这套对话内置换装置**唯一能自证没坏**的东西
    (本项目老账: 置换检验的结构不变量一旦保不住, 结论会整个反向,
     而照出它的正是 size 行 —— 见 claims_ledger §四.15)。

    报 95% CI 与「与 5% 是否相容」的判定。
    """
    from scipy.stats import binomtest  # 局部导入, 免得不必要的硬依赖
    groups = group_indices(keys)
    ms = np.array(m, dtype=np.float64)
    probs = np.full(len(keys), base)
    rej = 0
    for _ in range(n_rep):
        c = rng.binomial(ms.astype(np.int64), probs).astype(np.float64)
        S = S_stat(c, ms, base)
        null = perm_null(groups, c, ms, base, n_perm, rng)
        if (np.sum(null >= S) + 1) / (n_perm + 1) < 0.05:
            rej += 1
    rate = rej / n_rep
    lo, hi = binomtest(rej, n_rep, 0.05).proportion_ci(0.95)
    lines.append(f"\n### 专用 size 校验 (delta=0, n_rep={n_rep}, "
                 f"n_perm={n_perm})")
    lines.append(f"  拒绝率 = {rej}/{n_rep} = {rate:.2%}   "
                 f"95% CI [{lo:.2%}, {hi:.2%}]   (名义 5%)")
    lines.append(f"  ⚠️ 只报数, **不设通过阈值** —— 阈值是拍的常数就会误事"
                 f"(本项目老账 §四.16)。CI 是否盖住 5% 由读者判断。")
    return {"n_rep": n_rep, "n_perm": n_perm, "n_rej": rej,
            "rate": rate, "ci_lo": float(lo), "ci_hi": float(hi)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--power-rep", type=int, default=30)
    ap.add_argument("--power-perm", type=int, default=300)
    ap.add_argument("--size-rep", type=int, default=300,
                    help="专用 size 校验的重复数 (delta=0); 0 表示跳过")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "slot_signal.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = load_slots()
    print(f"载入 {len(rows)} 槽 / {len(set(r[0] for r in rows))} 对话")

    # ---- 装置自检: 置换的不变量 (会炸) ----
    _k = [r[0] for r in rows]
    _m = np.array([r[3] for r in rows], dtype=np.float64)
    _c = np.array([r[4][BC] for r in rows], dtype=np.float64)
    self_check_perm(_k, _m, _c, rng)
    print("装置自检: 对话内置换保持 各槽 m / 对话总票 / 总票, 且不产生 n>m ✅")

    res = {"n_slots": len(rows), "n_perm": args.n_perm, "boot": args.boot,
           "reads": [], "power": None, "size": None}
    lines = ["=" * 78, "1) slot 级信号检验 (超出对话级倾向的部分)", "=" * 78]
    for cat in (BC, 2):
        analyse("全量 (6 场景 × {train,test})", rows, cat,
                args.n_perm, args.boot, rng, lines, res["reads"])
        for sc in SCENARIOS:
            sub = [r for r in rows if r[1] == sc]
            if sub:
                analyse(f"场景 {sc}", sub, cat, args.n_perm, args.boot, rng,
                        lines, res["reads"])
    for ln in lines:
        print(ln)

    # 功效对照只在 **backchannel / 全量** 的真实结构上做 (最关心的那一类)
    m_all = np.array([r[3] for r in rows], dtype=np.float64)
    c_all = np.array([r[4][BC] for r in rows], dtype=np.float64)
    p_all = float(c_all.sum() / m_all.sum())
    lines2 = ["", "=" * 78,
              "2) 功效对照 (真实结构; 本设计能检出多大的 slot 信号)", "=" * 78]
    res["power"] = power_check([r[0] for r in rows], m_all, p_all,
                               args.power_rep, args.power_perm, rng, lines2)
    if args.size_rep > 0:
        res["size"] = size_check([r[0] for r in rows], m_all, p_all,
                                 args.size_rep, args.power_perm, rng, lines2)
    for ln in lines2:
        print(ln)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n→ 落盘: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
