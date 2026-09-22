"""§7.4f 的效度折扣: 那 400 个 E2 块里, 有多少是**训练块的时间邻居**?

## 为什么要查这个

§7.4f 的结论是「换掉 data_seed 造成的边界离散 ≈ 换掉 seed 造成的」。
但 E2 评估集有一个已知缺陷 (`2026-09-17_k4_eval_set.md` §3):
块按 **5 秒步进**生成、chunk 长 **10 秒**, 所以相邻两块**共享 50% 的音频**。
实测 **E2 有 184/1600 = 11.5%** 的块与训练块时间重叠。

**这给出一个替代解释**: 与训练音频重叠的块可能被**记住**了, 而记住什么、
记多牢, 恰恰**与数据顺序有关** —— 那会把 B 组 (变 data_seed) 的方差**抬高**,
于是「B/A ≈ 1」就可能不是「两种随机源贡献相当」, 而是「B 被污染撑大了」。

## 怎么证伪

**只用「与训练块不共享音频」的那部分块, 重算 A/B 分解。**
若干净子集上 mixnorm 的 B/A 仍 ≈ 1 ⇒ 污染不是驱动因素, 结论保住。
若 B/A 塌向 0 ⇒ 之前那个 ≈1 是污染造成的, **结论要撤**。

判据是**定性的**(塌/不塌), 不预设阈值 —— 因为这个检查是事后加的,
没有预登记, 所以只报数、只报方向, **不能拿它改判任何东西**
(memory: threshold-must-match-null-model 的第三面: 事后定的线不可信)。

## 口径

「共享音频」= 同一 (session, channel), 且 |Δt| < CHUNK_S(10s)。
STRIDE=5 ⇒ |Δt|=5 的邻居共享 5s; |Δt|=10 只在端点相触, 不算。

用法: python scripts/g1_margin_adjacency_robust.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_boundary_dispersion import var_ratio_perm  # noqa: E402

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
SEEDS = [42, 1234, 2024, 31337, 3407, 55555, 7]
ARMS = ["own10", "mixnorm"]
CHUNK_S = 10.0


def parse(cid):
    s, ch, t = cid.rsplit("_", 2)
    return s, ch, float(t[1:])


def read_ids(name):
    p = f"{ANNOT}/{name}"
    if not os.path.exists(p):
        return None
    return [l.strip() for l in open(p) if l.strip()]


def train_spans(train_ids):
    """{(sess, ch): 排序好的 t 列表} —— 训练块的时间位置。"""
    d = {}
    for cid in train_ids:
        s, ch, t = parse(cid)
        d.setdefault((s, ch), []).append(t)
    for k in d:
        d[k] = np.sort(np.array(d[k]))
    return d


def adjacent(cid, spans):
    """该块是否与任一训练块**共享音频** (同 session/ch, |Δt| < CHUNK_S)。"""
    s, ch, t = parse(cid)
    ts = spans.get((s, ch))
    if ts is None or len(ts) == 0:
        return False
    i = np.searchsorted(ts, t)
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(ts):
            dt = abs(ts[j] - t)
            if 0 < dt < CHUNK_S:
                return True
    return False


def mean_margin(arm, key, grp, cids):
    """块平均 margin_max —— 与 §5.1 / g1_margin_ds_analyze 同口径。"""
    tag = f"_s{key}" if grp == "A" else f"_ds{key}"
    p = f"{ANNOT}/g1_margin_{arm}{tag}.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    miss = [c for c in cids if c not in d]
    if miss:
        return None
    return float(np.mean([d[c]["margin_max"] for c in cids]))


def decompose(cids, label):
    """在 cids 上重算 A/B 分解 + 跨臂比。"""
    print(f"\n{'='*78}\n  {label}   (n={len(cids)} 块)\n{'='*78}")
    out = {}
    for arm in ARMS:
        va, vb = [], []
        for s in SEEDS:
            a = mean_margin(arm, s, "A", cids)
            b = mean_margin(arm, s, "B", cids)
            if a is None or b is None:
                print(f"    {arm}: 产物缺失, 跳过")
                va = None
                break
            va.append(a)
            vb.append(b)
        if va is None:
            continue
        va, vb = np.array(va), np.array(vb)
        sa, sb = va.std(ddof=1), vb.std(ddof=1)
        rr = [np.std(np.delete(vb, i), ddof=1) / np.std(np.delete(va, i), ddof=1)
              for i in range(len(SEEDS))]
        out[arm] = (sa, sb)
        print(f"    {arm:<8} A组 SD {sa:.3f}   B组 SD {sb:.3f}   "
              f"**B/A = {sb/sa:.2f}**   留一 [{min(rr):.2f}, {max(rr):.2f}]")
    if len(out) == 2:
        for grp in ("A", "B"):
            vo = np.array([mean_margin("own10", s, grp, cids) for s in SEEDS])
            vm = np.array([mean_margin("mixnorm", s, grp, cids) for s in SEEDS])
            _, p = var_ratio_perm(vo, vm)
            print(f"    跨臂 {grp} 组: {np.std(vm,ddof=1)/np.std(vo,ddof=1):.2f}×   "
                  f"置换 p = {p:.4f}")
    return out


def main():
    print("=" * 78)
    print("  §7.4f 效度折扣: 训练块的**时间邻居**是不是在撑大 B 组方差?")
    print("=" * 78)

    ids = read_ids("g1_margin_ds_ids.txt")
    train = read_ids("g1_ids.txt")
    if ids is None or train is None:
        print("  ❌ 缺 id 文件")
        return
    # 训练块 = g1_ids.txt 减去用作评估的 300 —— 那些不参与训练
    seen = read_ids("g1_eval_ids.txt") or []
    eval300 = set(seen)
    train = [c for c in train if c not in eval300]
    print(f"\n  探针块 {len(ids)}   g1_ids {len(train)+len(eval300)}  "
          f"其中用作评估 {len(eval300)}  ⇒ 训练块 {len(train)}")

    spans = train_spans(train)
    flag = {c: adjacent(c, spans) for c in ids}
    n_adj = sum(flag.values())
    print(f"\n  ── 与训练块共享音频的探针块 ──")
    print(f"    {n_adj}/{len(ids)} = **{n_adj/len(ids):.1%}**"
          f"   (k=4 文档 §3 报 E2 整体为 11.5%, 这里看的是被抽中的这 400 块)")

    clean = [c for c in ids if not flag[c]]
    dirty = [c for c in ids if flag[c]]

    full = decompose(ids, "① 全部 400 块 (与 §7.4f 正文同一个集)")
    cleanout = decompose(clean, "② **干净子集** (与训练音频无重叠) —— 判据看这一个")
    if dirty:
        decompose(dirty, f"③ 受污染子集 (n={len(dirty)}, 只作参照, 样本太小)")

    print(f"\n{'='*78}\n  ── 判读 ──\n{'='*78}")
    for arm in ARMS:
        if arm not in full or arm not in cleanout:
            continue
        r_all = full[arm][1] / full[arm][0]
        r_cln = cleanout[arm][1] / cleanout[arm][0]
        print(f"    {arm:<8} 全部 400 块 B/A = {r_all:.2f}   "
              f"干净子集 B/A = {r_cln:.2f}   移动 {r_cln-r_all:+.2f}")
    print(f"\n    🔴 **本检查是事后加的, 没有预登记** —— 所以它**只能证伪、不能改判**。")
    print(f"       干净子集若仍 ≈ 1 ⇒ 污染不是驱动因素, §7.4f 结论**不受影响**;")
    print(f"       若塌向 0 ⇒ §7.4f 那个 ≈1 **要撤回** (污染撑大的)。")
    print(f"       两种情形都只报方向, 不拿这个事后阈值去宣布新的显著/不显著。")


if __name__ == "__main__":
    main()
