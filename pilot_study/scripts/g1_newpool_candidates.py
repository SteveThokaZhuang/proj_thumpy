"""§4.5 检查 3（严格版）：新池的**候选池**正例率，与 e4c 的候选池比。

## 为什么不能只比「原始网格」

`g1_newpool_prevalence.py` 量到：
  已用 100 会话的**原始网格** 7.83% vs 新 100 会话 8.01%  ⇒ 会话本身一样（t=+0.35）
但 e4c **实际**的候选池是 **3.93%**（`g1_eval4_meta.json`: 3000 事件 / 76425 候选）。

差的来源：e4c 建候选时要排除**已用池**（`g1_ids.txt` 训练集 + `g1_eval2_ids.txt`=E2）
以及**与它们时间重叠**的块。那两个池子的密度是 **69.9% 和 102.5%** ——
它们几乎全是「有事件」的块（E2 按构造就是 100% 正例）。
把稠密块挖走之后，剩下的候选池自然稀。

⇒ **3.94% 不是「这批会话的性质」，是「被前面的集子吃剩下的性质」。**
新会话没被吃过 ⇒ 从新池自然抽样会得到 ~8% 的集，**是 e4c 的两倍稠**。
流行率正是 §3.3 翻车的那个变量 ⇒ **必须先说清楚，再决定怎么办。**

## 本脚本做什么

1. **锚点自检**：用**轨迹网格 + 完整排除逻辑**重跑「已用 100 会话」的候选扫描，
   必须复现 `g1_eval4_meta.json` 记的 `candidates_available=76425 / candidate_events=3000`。
   对不上就说明我的排除逻辑写得不对，后面的数都不能信。
2. 同一套逻辑跑**新 100 会话**（对新会话没有任何排除 ⇒ 候选 = 原始网格）。

用法（fd_analysis，零 GPU）：
  python scripts/g1_newpool_candidates.py
"""
import collections
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_g1_common import ANNOT                       # noqa: E402
from ari_g1_eval4_ids import load_bc_index            # noqa: E402

CANDOR = "/share/workspace3/shared_dataset/CANDOR/files"
E4RAW = f"{ANNOT}/e4_raw"
OUT = f"{ANNOT}/g1_newpool_candidates.json"

CHUNK_S = 10.0
STRIDE = 5.0
MIN_FRAMES = 20
N_USED, N_NEW = 100, 100

# 与 g1_eval4_meta.json 对照的锚点
ANCHOR = dict(candidates=76425, events=3000, n_chunks=1200, n_events=43)


def parse_id(cid):
    head, ch, t = cid.rsplit("_", 2)
    return head, ch[2:], float(t[1:])


def load_used():
    used = set()
    for p in ("g1_ids.txt", "g1_eval2_ids.txt"):
        f = f"{ANNOT}/{p}"
        if os.path.exists(f):
            used |= {l.strip() for l in open(f) if l.strip()}
    ut = collections.defaultdict(list)
    for cid in used:
        s8, ch, t0 = parse_id(cid)
        ut[(s8, ch)].append(t0)
    return used, {k: np.sort(np.array(v)) for k, v in ut.items()}


def overlaps(ut, s8, ch, t):
    a = ut.get((s8, ch))
    if a is None or not len(a):
        return False
    i = np.searchsorted(a, t)
    return bool(np.any(np.abs(a[max(0, i - 1):i + 1] - t) < CHUNK_S - 1e-9))


def grid_from_trace(full, ch):
    """轨迹网格: 返回 (起始时刻列表, 每块的帧数列表) 或 None。"""
    p = f"{E4RAW}/{full}.json"
    if not os.path.exists(p):
        return None
    fr = json.load(open(p))["channels"].get(ch, {}).get("frames") or []
    if not len(fr):
        return None
    t0a = np.array([f[0] for f in fr], dtype=float)
    tmax = float(t0a[-1])
    if tmax + 1e-9 < CHUNK_S:
        return [], []
    n = int(np.floor((tmax - CHUNK_S) / STRIDE + 1e-9)) + 1
    ts = [i * STRIDE for i in range(n)]
    nf = [int(((t0a >= t) & (t0a < t + CHUNK_S)).sum()) for t in ts]
    return ts, nf


def grid_from_mp3(full):
    mp3 = f"{CANDOR}/{full}/processed/{full}.mp3"
    try:
        i = sf.info(mp3)
        dur = float(i.frames) / float(i.samplerate)
    except Exception:                                  # noqa: BLE001
        return None
    if dur + 1e-9 < CHUNK_S:
        return []
    return [i * STRIDE for i in range(
        int(np.floor((dur - CHUNK_S) / STRIDE + 1e-9)) + 1)]


def scan(group, by, used, ut, grid_kind):
    """返回 (n_cand, n_ev, n_pos, n_grid, n_dropped_used, n_dropped_overlap)."""
    nc = ne = npos = ngrid = ndu = ndo = 0
    for full in group:
        s8 = full[:8]
        for ch in ("0", "1"):
            if grid_kind == "trace":
                g = grid_from_trace(full, ch)
                if g is None:
                    continue
                ts, nf = g
            else:
                ts = grid_from_mp3(full)
                if ts is None:
                    continue
                nf = [MIN_FRAMES] * len(ts)          # 闸门已证死, 不拦
            wins = by.get((full, int(ch)), [])
            for t, n_frames in zip(ts, nf):
                ngrid += 1
                cid = f"{s8}_ch{ch}_t{int(t)}"
                if cid in used:
                    ndu += 1
                    continue
                if overlaps(ut, s8, ch, t):
                    ndo += 1
                    continue
                if n_frames < MIN_FRAMES:
                    continue
                k = sum(1 for a, b in wins
                        if a >= t - 1e-9 and b <= t + CHUNK_S + 1e-9)
                nc += 1
                ne += k
                npos += 1 if k else 0
    return nc, ne, npos, ngrid, ndu, ndo


def main():
    sess = sorted(d for d in os.listdir(CANDOR) if len(d) == 36 and d[8] == "-")
    # ⚠️ 「已用 100 会话」以 e4_raw 目录为准（2026-09-22 实测：它等于 sorted[0:100]，
    # **不是**文档 §4.1 一直写的 sorted[1:101]）。用目录本身可以免掉这类差一错。
    used100 = sorted(f[:-5] for f in os.listdir(E4RAW) if f.endswith(".json"))
    start = sess.index(used100[-1]) + 1
    assert set(used100) <= set(sess), "e4_raw 里有不在主库的会话"
    new100 = sess[start:start + N_NEW]
    assert not (set(used100) & set(new100)), "新旧会话有重叠"
    by = load_bc_index()
    used, ut = load_used()
    print(f"主库 {len(sess)}; 已用池 {len(used)} 块; GT 索引 {len(by)} 个 (session,ch)")
    print(f"已用 100 会话 = e4_raw 目录 (sorted 下标 "
          f"{sess.index(used100[0])}–{sess.index(used100[-1])})")
    print(f"新 100 会话 = sorted[{start}:{start + N_NEW}]，"
          f"与已用零重叠")

    print("\n" + "=" * 66)
    print("[1] 锚点自检 —— 用**轨迹网格 + 完整排除**重跑已用 100 会话")
    nc, ne, npos, ng, ndu, ndo = scan(used100, by, used, ut, "trace")
    print(f"  网格 {ng}  候选 {nc}  事件 {ne}  含事件块 {npos}")
    print(f"  因 id 已用丢弃 {ndu}  因时间重叠丢弃 {ndo}")
    ok_c = nc == ANCHOR["candidates"]
    print(f"  候选 {nc} vs meta {ANCHOR['candidates']}  ⇒ "
          f"{'✅ 一致' if ok_c else '❌ 差 %d' % (nc - ANCHOR['candidates'])}")
    print(f"  （meta 的 3000 事件是**选中 1200 块**里的，不是候选池里的，"
          f"所以这里不逐位比事件数）")
    print(f"  候选池密度 {ne/nc:.4%}")

    print("\n" + "=" * 66)
    print("[2] 同一套逻辑跑新 100 会话（对新会话无任何排除）")
    nc2, ne2, npos2, ng2, ndu2, ndo2 = scan(new100, by, used, ut, "mp3")
    print(f"  网格 {ng2}  候选 {nc2}  事件 {ne2}  含事件块 {npos2}")
    print(f"  因 id 已用丢弃 {ndu2}  因时间重叠丢弃 {ndo2}")
    print(f"  候选池密度 {ne2/nc2:.4%}")

    print("\n" + "=" * 66)
    print("[3] 对比")
    p1, p2 = ne / nc, ne2 / nc2
    print(f"  e4c 候选池（已用 100 会话，被吃过） : {p1:.4%}  ({nc} 候选)")
    print(f"  新池      （新 100 会话，没被吃过） : {p2:.4%}  ({nc2} 候选)")
    print(f"  比值 {p2/p1:.2f}×")
    print(f"\n  ⇒ 从新池自然抽 4800 块，预期正例率 ≈ {p2:.2%}，"
          f"是 e4c（{p1:.2%}）的 {p2/p1:.1f} 倍")
    print(f"  ⇒ **不是同一个仪器的同一档流行率** —— 见 §4.4 的预登记必须写明这一点")

    json.dump(dict(
        anchor=dict(candidates=ANCHOR["candidates"], got_candidates=nc,
                    match=bool(ok_c), grid=int(ng), events_in_pool=int(ne),
                    dropped_used=int(ndu), dropped_overlap=int(ndo),
                    density=float(p1)),
        newpool=dict(grid=int(ng2), candidates=int(nc2), events=int(ne2),
                     pos_chunks=int(npos2), dropped_used=int(ndu2),
                     dropped_overlap=int(ndo2), density=float(p2)),
        ratio=float(p2 / p1),
        used_pool_density=dict(g1_ids=0.6988, g1_eval2=1.0250),
    ), open(OUT, "w"), indent=1)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
