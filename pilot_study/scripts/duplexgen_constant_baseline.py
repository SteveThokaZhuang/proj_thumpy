#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DuplexGen: **Table 6 的常数基线** —— 「不读上下文」能拿到多低的 KL? (0 GPU)

—— 为什么必须算这个 ——

DuplexGen 的核心量化主张是 Table 6: 在**评测集**上量
    D_KL(Human ‖ Model)
  PROMPT-ONLY 1.7B = 6.411  →  DUPLEXGEN-ONLY 1.7B = 0.465   (>10×)
并据此说「human calibration 才是场景化轮换行为的来源」。

但 duplexgen_slot_signal.py 实测: **backchannel 的逐槽票数与「该场景基准率
的二项抽样」不可区分** (6 场景里 5 个 p>0.75, 全量 p=1.0000)。
⇒ 若 Human 分布本身就在场景基准率附近散, 那么一个**输出场景常数的预测器**
  可能已经逼近 KL 的下界 ⇒ Table 6 测的可能是「有没有学会场景基准率」,
  而不是「有没有学会槽级人类偏好」。

本脚本算几个常数基线, 直接和 Table 6 并排。

—— 读数 ——
  A. uniform        : 三分类均匀, 退化的下界
  B. const_train    : 每个场景用**训练集**的边际 (公平的「无上下文」基线)
  C. const_test     : 每个场景用**测试集**的边际 (预言机常数 = 任何
                      无上下文预测器的**最好可能值**)
  D. const_global   : 六场景合并的单一常数
  E. noise_floor    : **不可约的噪声地板** —— 即使模型完全知道真实的逐槽分布,
                      人只有 m=5 票, 经验分布 p̂ 本身是抽出来的,
                      KL(p̂ ‖ 真值) 仍 > 0。用参数 bootstrap 估。

几何口径 (与 Table 6 对齐):
  * 只用 **test** split (= 论文的 evaluation split), 逐场景槽数须与
    Table 5 的 Evaluation 行一致 (TEA 383 / PLN 341 / INT 1016 /
    NEG 439 / PER 577 / SOC 497 = 3253) —— 这是**载入器的自检**;
  * 逐槽算 KL, 场景内取均值, 六场景**等权平均** (Table 6 的 AVG 就是等权平均,
    已用 PROMPT-ONLY 那一行验算过);
  * KL 方向 = D_KL(Human ‖ Model), 与论文一致;
  * p_k = 0 时该项记 0 (0 log 0 = 0)。
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

CORPUS = "/share/workspace3/shared_dataset/duplexgen-corpus"
SCENARIOS = ["TEA", "PLN", "INT", "NEG", "PER", "SOC"]
CATS = ["silent", "backchannel", "take_floor"]
# 论文 Table 5 的 Evaluation 行 —— 用来验证载入器 (差一个就炸)
TABLE5_EVAL = {"TEA": 383, "PLN": 341, "INT": 1016,
               "NEG": 439, "PER": 577, "SOC": 497}
# 论文 Table 6, 1.7B 那一档 (lower is better)
TABLE6_1_7B = {"PROMPT-ONLY": 6.411, "SWBD-ONLY": 1.260,
               "DUPLEXGEN-ONLY": 0.465, "SWBD+DUPLEXGEN": 0.466}
OUT_DIR = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
           "real_data/results/duplexgen_annot")


def raw_boundary_count(sc, split):
    """独立数一遍文件里 `boundaries` 的条目数 —— 用来验**载入器没有漏抽**。

    刻意不复用 load_split 的代码路径 (那样等于自己证明自己)。
    """
    path = f"{CORPUS}/annotations/{sc}/{split}.jsonl"
    n = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        for t in d.get("history", []) or []:
            n += len(t.get("boundaries", []) or [])
    return n


def load_split(split):
    """返回 {scenario: [(m, vec)]}。载入时逐槽断言 counts 闭合。"""
    out = {}
    for sc in SCENARIOS:
        path = f"{CORPUS}/annotations/{sc}/{split}.jsonl"
        rows = []
        if not os.path.exists(path):
            out[sc] = rows
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
                            f"turn{hi} slot{bi}")
                    vec = np.array([int(c.get(k, 0)) for k in CATS], dtype=np.int64)
                    if int(vec.sum()) != m:
                        raise AssertionError(f"类别分解不闭合 @ {key}")
                    rows.append((m, vec))
        out[sc] = rows
    return out


def load_split_utt(split):
    """逐 **utterance** 聚合: 把一个 turn 内所有槽的票池化。

    §4.2.2 原文: "the task is formulated as a turn-taking action prediction
    for individual user turns" ⇒ 论文的单位可能是 turn 而非 slot。
    已实测: 每个场景正好 **100 个带边界的 turn** (2/对话 × 50 对话),
    与 Table 4 的 "# Utterances (Evaluation) 600 (100 per task)" 吻合。
    """
    out = {}
    for sc in SCENARIOS:
        utt = []
        path = f"{CORPUS}/annotations/{sc}/{split}.jsonl"
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            for t in d.get("history", []) or []:
                b = t.get("boundaries") or []
                if not b:
                    continue
                M = sum(int(s["total_count"]) for s in b)
                v = np.zeros(len(CATS), dtype=np.int64)
                for s in b:
                    for i, k in enumerate(CATS):
                        v[i] += int(s["counts"].get(k, 0))
                if int(v.sum()) != M:
                    raise AssertionError(f"turn 级池化不闭合 @ {sc}/{split}")
                utt.append((M, v))
        out[sc] = utt
    return out


def marginal(rows):
    """池化边际分布。"""
    tot = np.zeros(len(CATS), dtype=np.float64)
    for m, v in rows:
        tot += v
    return tot / tot.sum()


def kl_matrix(H, q):
    """H: (n, K) 每槽的人类经验分布; q: (K,) 模型分布。返回 (n,) 的 KL。"""
    q = np.clip(q, 1e-12, None)
    q = q / q.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(H > 0, H * (np.log(np.where(H > 0, H, 1.0)) - np.log(q)), 0.0)
    return t.sum(axis=1)


def per_scenario_mean(rows, q):
    """场景内逐槽 KL 的均值。"""
    H = np.stack([v / m for m, v in rows])
    return float(kl_matrix(H, q).mean()), len(rows)


def noise_floor(rows, rng, n_rep=200):
    """不可约噪声地板。

    即使模型输出**该槽真实**的分布, 人只有 m 票 ⇒ 经验分布 p̂ 有抽样噪声。
    做法: 以 p̂ 当真值, 从它重抽 m 票得 p̂', 量 KL(p̂' ‖ p̂)。
    (这是对「模型完全正确时的期望 KL」的参数 bootstrap 估计。)
    """
    vals = []
    for _ in range(n_rep):
        tot = 0.0
        n = 0
        for m, v in rows:
            p = v / m
            draws = rng.multinomial(m, p)
            ph = draws / m
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(ph > 0,
                             ph * (np.log(np.where(ph > 0, ph, 1.0)) - np.log(np.clip(p, 1e-12, None))),
                             0.0)
            tot += float(t.sum())
            n += 1
        vals.append(tot / n)
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "constant_baseline.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    tr, te = load_split("train"), load_split("test")

    print("=" * 78)
    print("0) 载入器自检: test 逐场景槽数 vs 论文 Table 5 的 Evaluation 行")
    print("=" * 78)
    # 🔴 2026-09-23 实测: 已发布文件比 Table 5 的 Evaluation 行**少 9 个槽**
    #    (PLN 340 vs 341, PER 570 vs 577, SOC 496 vs 497; 其余 3 个精确吻合,
    #     含最大的 INT 1016)。已核查: `boundaries` 是槽的唯一来源, 无缺失/
    #     无 None/无空表 ⇒ **不是载入器漏抽, 是发布件与论文口径本身差 9 个**
    #     (最可能是论文数的是 GPT-4.1 定位出的候选槽, 而发布件只留了落在
    #     被抽样 utterance 内的那些 —— **未证实**)。
    #    ⇒ 纪律: **不掩盖, 但也不让 0.28% 的差假阻塞一个 0.4 vs 6.4 量级的比较**。
    #      逐场景报差值; 只有偏差 >1% 才炸 (那是真·口径对不上)。
    # ⚠️ 这里**故意不设阈值**。上一版拿一个拍脑袋的「1%」当判据, 结果 PER
    #    差 7/577=1.21% 就炸了 —— 那是**为了让脚本通过而调参**, 正是
    #    `hardcoded-conclusions-escape-reproduction` 警告的「判据常数」反模式。
    #
    #    退一步: 这道检查的**零模型**是「载入器读的是论文数的那个对象吗」。
    #    但载入器正确性**已由独立路径验过** —— 直接数 test.jsonl 里
    #    `boundaries` 的条目数, 与载入器给出一致 (见下面 raw 那一列)。
    #    所以:
    #      * **载入器正确性 = 硬断言** (raw 计数必须逐场景相等, 这会炸);
    #      * **与 Table 5 的差 = 报告项** (它是「发布件 vs 论文」的事实,
    #        不是载入器的错, 用阈值把它变成门禁只会掩盖事实)。
    total_want = sum(TABLE5_EVAL.values())
    raw_ok = True
    for sc in SCENARIOS:
        got, want = len(te[sc]), TABLE5_EVAL[sc]
        raw = raw_boundary_count(sc, "test")
        if raw != got:
            raw_ok = False
        d = got - want
        flag = "✅" if d == 0 else f"⚠️ {d:+d}"
        print(f"  {sc}: 实测 {got:>5d}  raw {raw:>5d}  Table5 {want:>5d}   {flag}")
    tot = sum(len(te[sc]) for sc in SCENARIOS)
    print(f"  合计: 实测 {tot}   Table5 {total_want}   "
          f"差 {tot - total_want:+d} ({(tot - total_want) / total_want:+.2%})")
    if not raw_ok:
        raise AssertionError("载入器漏抽: 逐场景实测 != 直接数 boundaries 的 raw 计数")
    print("  ✅ 载入器自检通过 (逐场景 = 直接数 boundaries 的 raw 计数, 无漏抽)")
    print(f"  ⚠️ 报告项: 已发布 test 件比论文 Table 5 **少 {total_want - tot} 个槽**, "
          f"集中在 PLN/PER/SOC。原因未知 (推测论文数的是 GPT-4.1 定位的候选槽, "
          f"而发布件只留了被抽样 utterance 内、且至少一人标注的 —— **未证实**)。"
          f" 对本题无实质影响 (0.28% vs 一个 0.4↔6.4 量级的比较), 但引用 Table 5 时必须写明。")

    gm = marginal([r for sc in SCENARIOS for r in tr[sc]])
    print(f"\n  全局边际 (train 池化): " +
          "  ".join(f"{c}={p:.4f}" for c, p in zip(CATS, gm)))

    models = {}
    models["A. uniform"] = {sc: np.full(len(CATS), 1.0 / len(CATS)) for sc in SCENARIOS}
    models["B. const_train"] = {sc: marginal(tr[sc]) for sc in SCENARIOS}
    models["C. const_test(预言机)"] = {sc: marginal(te[sc]) for sc in SCENARIOS}
    models["D. const_global"] = {sc: gm for sc in SCENARIOS}

    print("\n" + "=" * 78)
    print("1) 各常数模型在评测集上的 KL(Human ‖ Model) —— 与 Table 6 (1.7B) 并排")
    print("=" * 78)
    hdr = f"  {'模型':<24}" + "".join(f"{sc:>9}" for sc in SCENARIOS) + f"{'AVG':>10}"
    print(hdr)
    res = {"table5_eval": TABLE5_EVAL, "table6_1_7b": TABLE6_1_7B,
           "n_test_slots_total": tot, "models": {}}
    for name, per_sc in models.items():
        vals = []
        row = f"  {name:<24}"
        for sc in SCENARIOS:
            v, _ = per_scenario_mean(te[sc], per_sc[sc])
            vals.append(v)
            row += f"{v:>9.3f}"
        avg = float(np.mean(vals))
        row += f"{avg:>10.3f}"
        print(row)
        res["models"][name] = {"per_scenario": dict(zip(SCENARIOS, vals)), "avg": avg}

    # 论文自己的数字, 同格式并排
    print("\n  ---- 论文 Table 6 (1.7B) 原文数字, 供对照 ----")
    for name, v in TABLE6_1_7B.items():
        print(f"  {name:<24}{'':>9}" * 0 + f"  {name:<24}  AVG = {v:.3f}")

    print("\n" + "=" * 78)
    print("2) 不可约噪声地板 (人只有 m 票 ⇒ 经验分布本身有抽样噪声)")
    print("=" * 78)
    print(f"  {'场景':<8}{'槽数':>7}{'KL 地板':>12}{'SD':>10}")
    floors = []
    for sc in SCENARIOS:
        f, s = noise_floor(te[sc], rng)
        floors.append(f)
        print(f"  {sc:<8}{len(te[sc]):>7}{f:>12.4f}{s:>10.4f}")
    favg = float(np.mean(floors))
    print(f"  {'等权平均':<8}{'':>7}{favg:>12.4f}")
    res["noise_floor"] = {"per_scenario": dict(zip(SCENARIOS, floors)), "avg": favg}
    print("\n  📌 解读: 任何模型 (哪怕完全知道真实逐槽分布) 的 KL 都不可能低于这个地板。")

    print("\n" + "=" * 78)
    print("3) 判决")
    print("=" * 78)
    c = res["models"]["C. const_test(预言机)"]["avg"]
    b = res["models"]["B. const_train"]["avg"]
    dx = TABLE6_1_7B["DUPLEXGEN-ONLY"]
    print(f"  预言机常数 (C)      AVG = {c:.3f}")
    print(f"  公平常数 (B)        AVG = {b:.3f}")
    print(f"  噪声地板 (E)        AVG = {favg:.3f}")
    print(f"  DuplexGen 论文 1.7B AVG = {dx:.3f}  (DUPLEXGEN-ONLY)")
    if c < dx:
        print(f"  🔴 **预言机常数 {c:.3f} < 论文 {dx:.3f}** —— "
              f"一个不读上下文的常数预测器就打败了论文的完整方法。")
    else:
        print(f"  ⇒ 预言机常数 {c:.3f} > 论文 {dx:.3f} —— "
              f"论文确实比常数好 {dx / c:.2f}× (越小越好, 比值 <1 才算赢)。")
    print(f"  ⇒ 论文值 / 噪声地板 = {dx / favg:.2f}×")

    # ---------------------------------------------------------------- 4) 逐 turn
    print("\n" + "=" * 78)
    print("4) 敏感性: 逐 **turn** 聚合 (§4.2.2 说单位是 individual user turns)")
    print("=" * 78)
    tr_u, te_u = load_split_utt("train"), load_split_utt("test")
    for sc in SCENARIOS:
        if len(te_u[sc]) != 100:
            raise AssertionError(
                f"{sc} 带边界的 turn 数 {len(te_u[sc])} != 100 "
                f"(Table 4 说 100 utterances per task)")
    print("  ✅ 自检: 逐场景 100 个带边界的 turn, 与 Table 4 的 600 utterances 吻合")
    print(f"  {'场景':<8}{'turn':>6}{'票/turn':>9}{'const_train':>13}"
          f"{'const_test':>12}{'uniform':>10}{'地板':>9}")
    ru_b, ru_c, ru_u, ru_f = [], [], [], []
    for sc in SCENARIOS:
        H = np.stack([v / m for m, v in te_u[sc]])
        u = np.full(len(CATS), 1.0 / len(CATS))
        vb = float(kl_matrix(H, marginal(tr_u[sc])).mean())
        vc = float(kl_matrix(H, marginal(te_u[sc])).mean())
        vu = float(kl_matrix(H, u).mean())
        vf, _ = noise_floor(te_u[sc], rng)
        ru_b.append(vb); ru_c.append(vc); ru_u.append(vu); ru_f.append(vf)
        mv = float(np.mean([m for m, _ in te_u[sc]]))
        print(f"  {sc:<8}{len(te_u[sc]):>6}{mv:>9.1f}{vb:>13.3f}"
              f"{vc:>12.3f}{vu:>10.3f}{vf:>9.3f}")
    ub, uc, uu, uf = (float(np.mean(x)) for x in (ru_b, ru_c, ru_u, ru_f))
    print(f"  {'等权平均':<8}{'':>6}{'':>9}{ub:>13.3f}{uc:>12.3f}{uu:>10.3f}{uf:>9.3f}")
    res["utterance_level"] = {
        "const_train_avg": ub, "const_test_avg": uc,
        "uniform_avg": uu, "noise_floor_avg": uf,
        "const_train_per_scenario": dict(zip(SCENARIOS, ru_b)),
    }
    print(f"\n  📌 论文 Table 6 的 SEM 在 0.008–0.046 量级 —— 与「每场景 100 个 turn "
          f"求均值」的标准误一致，\n     与逐 slot (n≈500) 的不一致 ⇒ 侧面支持论文用的是逐 turn 口径。")
    print(f"  ⇒ 逐 turn 下: 常数 {ub:.3f} vs 论文 {dx:.3f}  "
          f"(论文是常数的 {dx / ub:.2f} 倍, 越低越好)")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n→ 落盘: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
