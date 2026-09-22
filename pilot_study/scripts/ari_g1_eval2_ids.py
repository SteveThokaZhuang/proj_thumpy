"""#53 扩大评估集: 选出一批**事件密集、且与现有 2400 池完全不重叠**的 chunk id.

## 为什么要扩

冻结评估集是 300 个 chunk, 其中**只有 25 个含真值事件 (26 个)**。而 G1 的
run-to-run SD 是 0.031、待检效应 0.034 —— 26 个事件里每命中/漏掉 1 个, recall
就跳 1/26 = 0.038, 于是 F1 的落点几乎全由"哪几个事件这次蒙对了"决定 (见 §7.4
的三层块结构: 11 个摇摆块贡献了几乎全部噪声)。**要回答 G1 的问题, 需要的是
事件数, 不是种子数。**

## 一个此前没注意到的分布错配

| | 块数 | 含 GT 的块 | 占比 |
|---|---|---|---|
| 训练集 (2100) | 2100 | 1817 | **86.5%** |
| 冻结评估集 (300) | 300 | 25 | **8.3%** |

训练集是 `select_train_chunks` **按事件富集**选出来的 (正例 3/4), 而评估集是
F8 任务留下的**随机**块样本。两者的事件流行率差了 10 倍 —— 评估集测的是一个
模型几乎没见过的分布, 这既压低了绝对 F1, 也放大了 run-to-run 方差。

## 本脚本选什么

`g1_ids.txt` 那 2400 块 (2100 训练 + 300 评估) **一块都不用** —— 于是新集合与
两臂的训练数据、与旧评估集都**完全不相交**, 因此**现有 14 个 adapter 可以直接
拿来做评估, 一行都不用重训**。

候选 = 100 个 e4_raw 会话按 5s 步进全部 10s 窗中, 满足:
  1. 不在 `g1_ids.txt` 里;
  2. 至少含 1 个 `cls=BC & realized != None` 且**完整落在块内**的窗口
     (横跨 chunk 边界的窗口不可答, 见 §7.5 / `bc_windows(require_inside=True)`);
  3. 该块内 X2-Turn 帧数 >= 20 (与建库口径一致, 保证 states 有意义)。

**抽样**: 按 (session, ch) 轮转取样 (round-robin), 而不是取前 N 个 ——
否则会偏向字典序靠前的会话, 让评估集聚集在少数人身上 (会话内相关会虚高精度)。

用法 (fd_analysis 环境即可, 不碰 GPU):
  python scripts/ari_g1_eval2_ids.py --n-eval 1600
  # -> results/annotator/g1_eval2_ids.txt  +  g1_eval2_meta.json
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_g1_common import ANNOT  # noqa: E402

CHUNK_S = 10.0
STRIDE = 5.0
MIN_FRAMES = 20


def load_bc_index():
    """(session, ch_event) -> [(start, end)] : 只要 realized != None 的 BC。

    只读 events/e1 两张表的交集, 不碰音频 —— 整个选块过程是纯表操作。
    """
    import glob

    import pandas as pd
    D = f"{ANNOT}/../analysis/ari"
    ev = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{D}/candor_events_w*.csv"))])
    e1 = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{D}/candor_e1_w*.csv"))])
    m = ev.merge(e1[["event_id", "realized"]], on="event_id",
                 suffixes=("", "_e1"), how="left")
    m = m[(m["cls"] == "BC") & (m["realized"] != "None")]
    by = collections.defaultdict(list)
    for s, ch, a, b in zip(m["session"], m["ch_event"], m["start"], m["end"]):
        by[(str(s), int(ch))].append((float(a), float(b)))
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=1600,
                    help="目标块数 (轮转取样, 实际可能略少)")
    ap.add_argument("--used", default=f"{ANNOT}/g1_ids.txt",
                    help="已用过的 chunk id 清单 (训练+旧评估), 新集必须与之不相交")
    ap.add_argument("--out-ids", default=f"{ANNOT}/g1_eval2_ids.txt")
    ap.add_argument("--out-meta", default=f"{ANNOT}/g1_eval2_meta.json")
    args = ap.parse_args()

    used = set(l.strip() for l in open(args.used) if l.strip())
    by = load_bc_index()
    x2 = sorted(f for f in os.listdir(f"{ANNOT}/e4_raw") if f.endswith(".json"))
    print(f"已用池 {len(used)} 块; e4_raw {len(x2)} 会话; "
          f"BC(realized!=None) 索引 {len(by)} 个 (session,ch)", flush=True)

    # ── 逐 (session, ch) 枚举, 收集候选 ────────────────────────────────
    per_key = collections.OrderedDict()
    n_scanned = n_offpool = 0
    for fn in x2:
        full = fn[:-5]
        s8 = full[:8]
        d = json.load(open(f"{ANNOT}/e4_raw/{fn}"))
        for ch in ("0", "1"):
            fr = d["channels"].get(ch, {}).get("frames") or []
            if not len(fr):
                continue
            t0a = np.array([f[0] for f in fr])
            tmax = float(t0a[-1])
            wins_all = by.get((full, int(ch)), [])
            cand = []
            t = 0.0
            while t + CHUNK_S <= tmax:
                cid = f"{s8}_ch{ch}_t{int(t)}"
                n_scanned += 1
                if cid in used:
                    t += STRIDE
                    continue
                n_offpool += 1
                fm = (t0a >= t) & (t0a < t + CHUNK_S)
                if int(fm.sum()) < MIN_FRAMES:
                    t += STRIDE
                    continue
                # 完整落在块内的窗口 (与 bc_windows(require_inside=True) 同口径)
                inside = [(a, b) for a, b in wins_all
                          if a >= t - 1e-9 and b <= t + CHUNK_S + 1e-9]
                if inside:
                    cand.append((cid, len(inside)))
                t += STRIDE
            if cand:
                per_key[(s8, ch)] = cand

    n_cand = sum(len(v) for v in per_key.values())
    n_ev = sum(n for _, n in (c for v in per_key.values() for c in v))
    print(f"扫描 {n_scanned} 个窗口位置 (池外 {n_offpool}); "
          f"候选 {n_cand} 块 / {n_ev} 个事件, 分布在 "
          f"{len(per_key)} 个 (session,ch)", flush=True)

    # ── 轮转取样: 每轮从每个 (session,ch) 各取 1 个, 直到够数 ───────────
    # 这样块数按会话均匀铺开, 不会让评估集挤在少数会话里。
    picked, idx = [], {k: 0 for k in per_key}
    while len(picked) < args.n_eval:
        progressed = False
        for k, v in per_key.items():
            if len(picked) >= args.n_eval:
                break
            if idx[k] < len(v):
                picked.append(v[idx[k]])
                idx[k] += 1
                progressed = True
        if not progressed:
            break

    ids = [c for c, _ in picked]
    n_ev_sel = sum(n for _, n in picked)
    sess = collections.Counter(c.split("_")[0] for c in ids)
    print(f"\n选定 {len(ids)} 块 / {n_ev_sel} 个事件")
    print(f"  覆盖 {len(sess)} 个会话, 每会话 {min(sess.values())}–"
          f"{max(sess.values())} 块 (中位 "
          f"{sorted(sess.values())[len(sess)//2]})")
    print(f"  每块事件数分布: "
          f"{dict(sorted(collections.Counter(n for _, n in picked).items()))}")
    print(f"  相对旧评估集: 块 ×{len(ids)/300:.1f}, 事件 ×{n_ev_sel/26:.1f}")

    overlap = set(ids) & used
    assert not overlap, f"与已用池重叠 {len(overlap)} 块!"
    assert len(set(ids)) == len(ids), "id 有重复"

    with open(args.out_ids, "w") as f:
        f.write("\n".join(ids) + "\n")
    json.dump({
        "n_chunks": len(ids), "n_events": n_ev_sel,
        "n_sessions": len(sess),
        "require_inside": True, "chunk_s": CHUNK_S, "stride": STRIDE,
        "excluded_pool": args.used, "excluded_pool_size": len(used),
        "candidates_available": n_cand, "candidate_events": n_ev,
        "events_per_chunk": dict(sorted(
            collections.Counter(n for _, n in picked).items())),
        "chunks_per_session": dict(sorted(sess.items())),
    }, open(args.out_meta, "w"), indent=2, ensure_ascii=False)
    print(f"\n-> {args.out_ids}\n-> {args.out_meta}")


if __name__ == "__main__":
    main()
