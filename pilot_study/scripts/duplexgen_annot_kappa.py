"""DuplexGen 人类标注的**标注者间一致性** (Fleiss' kappa).

## 这个脚本回答的唯一问题

F3 人工核验 (`docs/pilot_study/2026-09-15_f3_human_verification_results.md`) 的硬限制是
**单人标注、无 kappa** ⇒ 当时只敢写"还不能说必须改成三级标注"。

DuplexGen 的 `annotations/` 是**本项目至今唯一一份"同一个 slot 多个真人投票"**的数据
(每槽 `total_count` 个评分者, 投票落进 {silent, backchannel, take_floor} 三类)
⇒ **这个洞能补上**, 且**零 GPU**。

⚠️ 本脚本**只用 `annotations/` 一个目录**, **不碰 `dialogues/`, 不做任何 join**
—— 因为已发布的两部分不共享文本 (`2026-09-23_duplexgen_humdial_recon.md` §3)。

## 四条口径 (本项目的教训, 缺一不可)

1. 🔴 **先报流行率再报 kappa**。kappa 对边际分布极敏感, 一个低的 kappa 可能只是
   某一类稀有造成的伪影 (memory `prevalence-flips-variance-conclusions`: 本项目
   最大的翻车就是"18.6 倍方差"其实是流行率伪影) ⇒ **先打 `p_j`, 再打 kappa**。
2. 🔴 **bootstrap 必须在"对话"这一层重抽**, 不能在 slot 层。同一个对话内的 slot
   显然不独立, slot 层重抽会把 CI 做窄、把精度吹高
   (memory `permutation-test-covers-one-variance-source`、`paired-design-floors-at-the-interaction`)。
3. **变评分者数的推广**: 每槽 `total_count` 在 1~5 之间变 (实测), 经典 Fleiss 公式
   要求固定 m。这里用逐 item 的 P_i 推广式, **并附一个只取 m=5 的敏感性分析**
   (那一档下经典公式精确成立) 来验证推广没有把数做大。
4. **对零模型自检** (memory `threshold-must-match-null-model`): 把每个 slot 内的投票
   随机置换后 kappa 必须 ≈ 0; 全一致的合成数据 kappa 必须 == 1。

用法 (fd_analysis, 纯读盘, 0 GPU):
  python scripts/duplexgen_annot_kappa.py                 # 全量 (train+test 共 420 条)
  python scripts/duplexgen_annot_kappa.py --bootstrap 5000
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

# 数据根 (只读)
CORPUS = "/share/workspace3/shared_dataset/duplexgen-corpus"
SCENARIOS = ["TEA", "PLN", "INT", "NEG", "PER", "SOC"]

# 类别固定顺序 (annotations 的术语; 注意 dialogues 用的是 silence/floor_taking,
# 两边只有 backchannel 同名 —— 见 recon §2.2 的术语陷阱)
CATS = ["silent", "backchannel", "take_floor"]

# 分组 (我方口径, 非论文原文): 合作类 vs 竞争类 vs 社交
GROUP = {
    "TEA": "cooperative", "PLN": "cooperative", "INT": "cooperative",
    "NEG": "competitive", "PER": "competitive",
    "SOC": "social",
}

OUT_DIR = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/duplexgen_annot"


# ---------------------------------------------------------------- 载入

def load_slots(splits):
    """返回 [(dialogue_key, scenario, split, m_i, counts ndarray[CATS])] —— 只含有边界的槽。

    逐槽做**结构自检**: sum(counts.values()) 必须 == total_count, 否则炸。
    (memory `delta-f1-does-not-decompose`: 错值只差第四位、肉眼必放过, 必须靠会炸的自检)
    """
    rows = []
    n_dlg = 0
    n_slots_all = 0
    for sc in SCENARIOS:
        for split in splits:
            path = os.path.join(CORPUS, "annotations", sc, f"{split}.jsonl")
            if not os.path.exists(path):
                continue
            for li, line in enumerate(open(path)):
                d = json.loads(line)
                n_dlg += 1
                key = f"{sc}/{split}/{d['example_id']}"
                for hi, h in enumerate(d["history"]):
                    for bi, b in enumerate(h.get("boundaries", [])):
                        n_slots_all += 1
                        m = int(b["total_count"])
                        c = b["counts"]
                        s = sum(int(v) for v in c.values())
                        if s != m:
                            raise AssertionError(
                                f"counts 之和 {s} != total_count {m} "
                                f"@ {key} turn{hi} slot{bi} -> {c}")
                        vec = np.array([int(c.get(k, 0)) for k in CATS], dtype=np.int64)
                        if int(vec.sum()) != m:
                            raise AssertionError(f"类别分解不闭合 @ {key} turn{hi} slot{bi}")
                        rows.append((key, sc, split, m, vec))
    return rows, n_dlg, n_slots_all


# ---------------------------------------------------------------- 统计

def fleiss_kappa(items):
    """items: [(m_i, counts ndarray[C])]。**m_i < 2 的 item 在本函数内直接滤掉**
    (P_i 的分母 m_i(m_i-1) 为 0, 单评分者对一致性无定义)。

    变 m 的推广 (Fleiss 1971):
      P_i      = (sum_j n_ij^2 - m_i) / (m_i (m_i - 1))     逐 item 观测一致度
      P_bar    = mean_i P_i
      p_j      = sum_i n_ij / sum_i m_i                     总体类别边际
      P_e      = sum_j p_j^2
      kappa    = (P_bar - P_e) / (1 - P_e)
    返回 (kappa, P_bar, P_e, p_j)。
    """
    items = [(it[0], it[1]) for it in items if it[0] >= 2]
    if not items:
        return float("nan"), float("nan"), float("nan"), np.full(len(CATS), np.nan)
    m = np.array([it[0] for it in items], dtype=np.float64)
    N = np.stack([it[1] for it in items]).astype(np.float64)   # (n, C)
    P_i = (np.sum(N * N, axis=1) - m) / (m * (m - 1.0))
    P_bar = float(P_i.mean())
    p_j = N.sum(axis=0) / m.sum()
    P_e = float(np.sum(p_j * p_j))
    kappa = (P_bar - P_e) / (1.0 - P_e) if P_e < 1.0 else float("nan")
    return kappa, P_bar, P_e, p_j


def remap(vec, groups):
    """把 CATS 的计数按 groups (每个超类 = 一组下标) 合并。groups=None 表示不动。"""
    if groups is None:
        return vec
    return np.array([int(vec[list(g)].sum()) for g in groups], dtype=np.int64)


def gwet_ac1(items):
    """Gwet's AC1 —— **对流行率不敏感**的一致度系数。

    为什么必须一起报: kappa 的 P_e = sum_j p_j^2 在边际高度倾斜时会被推高,
    把 kappa 压低 (流行率悖论)。本项目的教训正是"报方差前先报正例率"
    (memory `prevalence-flips-variance-conclusions`) ⇒ **只报 kappa 会高估"人根本不一致"**。

      p_a = P_bar (与 Fleiss 同)
      p_e = (1/(q-1)) * sum_j pi_j (1 - pi_j)     q = 超类个数
      AC1 = (p_a - p_e) / (1 - p_e)
    """
    items = [(it[0], it[1]) for it in items if it[0] >= 2]
    if not items:
        return float("nan"), float("nan"), float("nan")
    m = np.array([it[0] for it in items], dtype=np.float64)
    N = np.stack([it[1] for it in items]).astype(np.float64)
    q = N.shape[1]
    P_i = (np.sum(N * N, axis=1) - m) / (m * (m - 1.0))
    p_a = float(P_i.mean())
    pi = N.sum(axis=0) / m.sum()
    p_e = float(np.sum(pi * (1.0 - pi)) / (q - 1.0))
    ac1 = (p_a - p_e) / (1.0 - p_e) if p_e < 1.0 else float("nan")
    return ac1, p_a, p_e


def cluster_bootstrap(rows_by_dlg, B, seed, groups=None):
    """按**对话**重抽 (口径 2)。rows_by_dlg: {key: [(m, vec), ...]}

    同时给 kappa 与 AC1 的 CI —— 两个读数都要带误差棒 (memory
    `sd-from-few-points-manufactures-anomalies`: 差值/系数没误差棒是本项目的老毛病)。
    """
    keys = sorted(rows_by_dlg)
    rng = np.random.default_rng(seed)
    ks, acs = [], []
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        items = []
        for i in pick:
            for m, v in rows_by_dlg[keys[i]]:
                items.append((m, remap(v, groups)))
        k, _, _, _ = fleiss_kappa(items)
        a, _, _ = gwet_ac1(items)
        if not np.isnan(k):
            ks.append(k)
        if not np.isnan(a):
            acs.append(a)
    ks = np.array(ks)
    acs = np.array(acs)
    out = {}
    for nm, arr in (("kappa", ks), ("ac1", acs)):
        out[nm] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)),
                   float(arr.std(ddof=1)))
    return out


def self_checks():
    """口径 4: 对零模型 + 恒真情况各验一次。任何一条不过就炸。"""
    rng = np.random.default_rng(12345)
    n_items, m, C = 400, 5, len(CATS)

    # (a) 全一致 ⇒ kappa == 1 (精确相等, 不是 approx)
    perfect = [(m, np.eye(C, dtype=np.int64)[i % C] * m) for i in range(n_items)]
    k, _, _, _ = fleiss_kappa(perfect)
    assert k == 1.0, f"全一致应给 kappa==1, 实得 {k!r}"

    # (b) 独立均匀随机 ⇒ kappa ≈ 0
    null = []
    for _ in range(n_items):
        v = np.zeros(C, dtype=np.int64)
        for _ in range(m):
            v[rng.integers(0, C)] += 1
        null.append((m, v))
    k0, _, _, _ = fleiss_kappa(null)
    assert abs(k0) < 0.10, f"零模型应给 kappa≈0, 实得 {k0:.4f}"

    # (c) m 在 2~5 之间变化的随机 ⇒ 仍 ≈ 0 (验证变 m 推广式没有系统性偏移)
    nullv = []
    for _ in range(n_items):
        mm = int(rng.integers(2, 6))
        v = np.zeros(C, dtype=np.int64)
        for _ in range(mm):
            v[rng.integers(0, C)] += 1
        nullv.append((mm, v))
    kv, _, _, _ = fleiss_kappa(nullv)
    assert abs(kv) < 0.10, f"变 m 零模型应给 kappa≈0, 实得 {kv:.4f}"

    # (d) m=1 的槽**不得**进入 kappa (P_i 分母为 0)
    k1, _, _, _ = fleiss_kappa([(1, np.array([1, 0, 0]))])
    assert np.isnan(k1), "单评分者槽应被排除"

    # (e) AC1 的三条: 全一致==1 / 零模型≈0 / **倾斜边际下 AC1 必须显著高于 kappa**
    a_perf, _, _ = gwet_ac1(perfect)
    assert a_perf == 1.0, f"全一致应给 AC1==1, 实得 {a_perf!r}"
    a0, _, _ = gwet_ac1(null)
    assert abs(a0) < 0.10, f"零模型应给 AC1≈0, 实得 {a0:.4f}"
    # 9:1 倾斜 + 完全随机 ⇒ kappa 被压低, AC1 不应跟着被压到同一量级
    skew = []
    for _ in range(n_items):
        v = np.zeros(C, dtype=np.int64)
        for _ in range(m):
            v[0 if rng.random() < 0.9 else rng.integers(1, C)] += 1
        skew.append((m, v))
    ks_, _, _, _ = fleiss_kappa(skew)
    as_, _, _ = gwet_ac1(skew)
    assert as_ > ks_ + 0.10, (
        f"倾斜边际下 AC1({as_:.4f}) 应明显高于 kappa({ks_:.4f}) —— "
        f"这是 AC1 存在的理由, 若相等说明公式写错")
    a1, _, _ = gwet_ac1([(1, np.array([1, 0, 0]))])
    assert np.isnan(a1), "单评分者槽应被 AC1 排除"

    return {"perfect": k, "null_fixed_m": k0, "null_varying_m": kv,
            "ac1_perfect": a_perf, "ac1_null": a0,
            "ac1_skewed": as_, "kappa_skewed": ks_}


# ---------------------------------------------------------------- 报告

def describe(tag, rows, rows_by_dlg, B, seed, lines, groups=None, cat_names=None):
    """rows 是主表里的 5 元组 (key, scenario, split, m, vec)。

    groups 非 None 时先做超类合并 (如 backchannel vs 其余) —— 见 section 4。
    """
    names = cat_names if cat_names else CATS
    items = [(r[3], remap(r[4], groups)) for r in rows]
    k, P_bar, P_e, p_j = fleiss_kappa(items)
    a, _, P_e_ac1 = gwet_ac1(items)
    if items:
        bt = cluster_bootstrap(rows_by_dlg, B, seed, groups=groups)
        (lo, hi, sd) = bt["kappa"]
        (alo, ahi, asd) = bt["ac1"]
    else:
        lo = hi = sd = alo = ahi = asd = np.nan
    mm = [it[0] for it in items if it[0] >= 2]
    m_arr = np.array(mm) if mm else np.array([np.nan])
    lines.append(f"\n### {tag}")
    lines.append(f"  槽数 (m>=2) = {len(mm)}   评分者数 m: "
                 f"min={m_arr.min() if len(m_arr) else '-'} "
                 f"max={m_arr.max() if len(m_arr) else '-'} "
                 f"mean={m_arr.mean() if len(m_arr) else float('nan'):.3f}")
    lines.append(f"  🔴 类别边际 p_j (先看这个): " +
                 "  ".join(f"{c}={p:.4f}" for c, p in zip(names, p_j)))
    lines.append(f"  观测一致度 P_bar = {P_bar:.4f}   "
                 f"机遇一致度: P_e(kappa) = {P_e:.4f}   P_e(AC1) = {P_e_ac1:.4f}")
    lines.append(f"  **Fleiss' kappa = {k:.4f}**  95% CI [{lo:.4f}, {hi:.4f}]  (SD={sd:.4f})")
    lines.append(f"  **Gwet's AC1   = {a:.4f}**  95% CI [{alo:.4f}, {ahi:.4f}]  (SD={asd:.4f})"
                 f"   ← 对流行率不敏感")
    return {"tag": tag, "n_slots": len(mm), "kappa": k, "P_bar": P_bar, "P_e": P_e,
            "ci_lo": lo, "ci_hi": hi, "boot_sd": sd,
            "ac1": a, "P_e_ac1": P_e_ac1, "ac1_ci_lo": alo, "ac1_ci_hi": ahi,
            "ac1_boot_sd": asd, "cats": list(names),
            "p_j": {c: float(p) for c, p in zip(names, p_j)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "kappa.json"))
    args = ap.parse_args()

    rows, n_dlg, n_slots_all = load_slots(["train", "test"])

    print("=" * 78)
    print("0) 结构自检 (counts 之和 == total_count) —— 已在载入时逐槽断言, 全过")
    print("=" * 78)
    print(f"  对话 {n_dlg} 条 / 全部有边界的槽 {n_slots_all} 个")
    mdist = Counter(r[3] for r in rows)
    print(f"  评分者数 m 分布: {dict(sorted(mdist.items()))}")
    n_multi = sum(1 for r in rows if r[3] >= 2)
    print(f"  **可进 kappa 的槽 (m>=2) = {n_multi}**  "
          f"(排除 m=1 的 {n_slots_all - n_multi} 个, 占 "
          f"{(n_slots_all - n_multi) / max(1, n_slots_all):.1%})")

    print("\n" + "=" * 78)
    print("1) 判读口径自检 (零模型 + 恒真)")
    print("=" * 78)
    chk = self_checks()
    for kk, vv in chk.items():
        print(f"  {kk:18s} = {vv:.6f}")
    print("  ✅ 全过 (全一致==1 精确相等 / 定 m 零模型≈0 / 变 m 零模型≈0 / m=1 被排除 /")
    print("          AC1: 全一致==1 & 零模型≈0 & 倾斜边际下 AC1 高于 kappa)")

    multi = [r for r in rows if r[3] >= 2]
    by_dlg = {}
    for key, sc, split, m, v in multi:
        by_dlg.setdefault(key, []).append((m, v))

    lines = ["", "=" * 78, "2) 主结果", "=" * 78]
    res = {"self_checks": chk, "n_dialogues": n_dlg, "n_slots_all": n_slots_all,
           "n_slots_kappa": n_multi, "m_dist": dict(sorted(mdist.items())), "reads": []}

    # 2a 全量
    res["reads"].append(describe("全量 (6 场景 × {train,test})", multi, by_dlg,
                                 args.bootstrap, args.seed, lines))

    # 2b 只 test (干净的那半 —— train 那 120 条被用来校准生成器)
    for split in ["test", "train"]:
        sub = [r for r in rows if r[3] >= 2 and r[2] == split]
        bd = {}
        for key, sc, sp, m, v in sub:
            bd.setdefault(key, []).append((m, v))
        res["reads"].append(describe(f"只 {split}", sub, bd, args.bootstrap, args.seed, lines))

    # 2c 逐场景
    for sc in SCENARIOS:
        sub = [r for r in rows if r[3] >= 2 and r[1] == sc]
        bd = {}
        for key, s2, sp, m, v in sub:
            bd.setdefault(key, []).append((m, v))
        if sub:
            res["reads"].append(describe(f"场景 {sc} ({GROUP[sc]})", sub, bd,
                                         args.bootstrap, args.seed, lines))

    # 2d 按组
    for g in ["cooperative", "competitive", "social"]:
        sub = [r for r in rows if r[3] >= 2 and GROUP[r[1]] == g]
        bd = {}
        for key, s2, sp, m, v in sub:
            bd.setdefault(key, []).append((m, v))
        if sub:
            res["reads"].append(describe(f"组 {g}", sub, bd, args.bootstrap, args.seed, lines))

    # 2e 敏感性: 只用 m=5 (经典 Fleiss 公式在此精确成立)
    sub = [r for r in rows if r[3] == 5]
    bd = {}
    for key, sc, sp, m, v in sub:
        bd.setdefault(key, []).append((m, v))
    res["reads"].append(describe("敏感性: 只取 m=5 (定评分者数)", sub, bd,
                                 args.bootstrap, args.seed, lines))

    for ln in lines:
        print(ln)

    # 3) 投票极化 / 分歧结构
    print("\n" + "=" * 78)
    print("3) 分歧结构 (不含 m=1)")
    print("=" * 78)
    pol = Counter()
    for _, _, _, m, v in multi:
        mx = int(v.max())
        if mx == m:
            pol["全体一致"] += 1
        elif mx * 2 > m:
            pol["有严格多数但非全体"] += 1
        else:
            pol["无多数 (平票/相对多数)"] += 1
    tot = len(multi)
    for kk in ["全体一致", "有严格多数但非全体", "无多数 (平票/相对多数)"]:
        print(f"  {kk:24s} {pol[kk]:5d}  ({pol[kk] / tot:.2%})")
    res["polarization"] = {k: {"n": pol[k], "frac": pol[k] / tot} for k in pol}

    # 多数票的类别构成
    maj = Counter()
    for _, _, _, m, v in multi:
        if int(v.max()) * 2 > m:
            maj[CATS[int(np.argmax(v))]] += 1
    print("\n  多数票落在哪一类:")
    for c in CATS:
        print(f"    {c:12s} {maj[c]:5d}  ({maj[c] / max(1, sum(maj.values())):.2%})")
    res["majority_category"] = dict(maj)

    # backchannel 的"票share"分布 —— 本项目关心的那一类
    bc = np.array([v[1] / m for _, _, _, m, v in multi])
    # 显式区间 (不用 edges 数组 —— 上一次就是它差一个边界把最后一段漏了)。
    # 🔴 分箱必须**铺满且不重叠**: 下面用一条断言守住 (memory `threshold-must-match-null-model`
    #    的"值域要铺满"那一面: 没定义的区间会把值静默吞掉)。
    bins = [
        ("0 (无人投 BC)",   bc == 0.0),
        ("(0,0.25]",       (bc > 0.0) & (bc <= 0.25)),
        ("(0.25,0.5]",     (bc > 0.25) & (bc <= 0.5)),
        ("(0.5,0.75]",     (bc > 0.5) & (bc <= 0.75)),
        ("(0.75,1)",       (bc > 0.75) & (bc < 1.0)),
        ("1 (全票 BC)",     bc == 1.0),
    ]
    _tot = sum(int(mask.sum()) for _, mask in bins)
    assert _tot == len(bc), f"分箱没铺满: {_tot} != {len(bc)}"
    assert (0.0 <= bc).all() and (bc <= 1.0).all(), "得票率越界"
    print("\n  backchannel 得票率分布:")
    for name, mask in bins:
        c = int(mask.sum())
        print(f"    {name:16s} {c:5d}  ({c / len(bc):.2%})")
    res["bc_vote_share"] = {name: int(mask.sum()) for name, mask in bins}

    # 4) 二值塌缩: backchannel vs 其余
    #    ⚠️ 这一节是**看了三分类结果之后才加的**, 属事后稳健性检查, 必须标明。
    #    动机: 三分类的 kappa 低, 有两种解释 —— (i) 人真的不一致;
    #    (ii) silent vs take_floor 这条细分难, 但"是不是 BC"其实一致。
    #    本项目关心的是 BC ⇒ 必须把这两种解释分开。
    print("\n" + "=" * 78)
    print("4) 事后稳健性: 二值塌缩 backchannel vs 其余 (⚠️ 见脚本注释, 事后选定)")
    print("=" * 78)
    G_BC = [[1], [0, 2]]
    N_BC = ["backchannel", "其余(silent+take_floor)"]
    lines2 = ["", "=" * 78, "4) 事后稳健性: 二值塌缩 backchannel vs 其余", "=" * 78]
    res["posthoc_binary"] = []
    for tag, sub in [("二值·全量", multi),
                     ("二值·只 test", [r for r in multi if r[2] == "test"]),
                     ("二值·只 m=5", [r for r in multi if r[3] == 5])]:
        bd = {}
        for key, sc, sp, m, v in sub:
            bd.setdefault(key, []).append((m, v))
        res["posthoc_binary"].append(
            describe(tag, sub, bd, args.bootstrap, args.seed, lines2,
                     groups=G_BC, cat_names=N_BC))
    for sc in SCENARIOS:
        sub = [r for r in multi if r[1] == sc]
        bd = {}
        for key, s2, sp, m, v in sub:
            bd.setdefault(key, []).append((m, v))
        if sub:
            res["posthoc_binary"].append(
                describe(f"二值·场景 {sc}", sub, bd, args.bootstrap, args.seed, lines2,
                         groups=G_BC, cat_names=N_BC))
    for ln in lines2:
        print(ln)

    res["polarization_note"] = "m=1 的槽已排除; 分歧结构只在 m>=2 上定义"
    res["posthoc_note"] = ("section 4 的二值塌缩是看到三分类结果后才加的, 属事后稳健性检查; "
                           "section 2 的三分类读数是预先定下的主读数")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n→ 落盘: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
