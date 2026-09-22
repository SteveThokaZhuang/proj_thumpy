"""#51 结论量化: "从未命中"到底是模型不行, 还是窗口本身不可答?

ari_g1_pred_times.py 显示 never 组是**双峰**的: 要么直接答 'no backchannel',
要么预测在离窗口 2.3–7.5s 的地方 —— 没有"差一点点"的簇。同时发现 3 个块的 GT
窗口**横跨 chunk 边界**(如 (-0.76, 0.16): 83% 在块外)。

本脚本在全 25 个含 GT 的块上量化两件事:
  A. 有多少 GT 窗口不是完整落在 chunk 内的 (声学证据大半在相邻块 -> 不可答)
  B. 有多少块里存在 model 能听到、但 realized=None 因而不算 GT 的事件
     (预测落在它上面 = 记成 FP, 而真正的 GT 是另一个) —— 即 F3 那个
     "停顿内其实人耳是 BC" 的问题在指标上的投影

用法: python ari_g1_boundary.py
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_g1_common import ANNOT, eval_chunk_ids, load_manifest  # noqa: E402

OWN = f"{ANNOT}/g1_own_eval/own"
D = f"{ANNOT}/../analysis/ari"
CHUNK_S = 10.0

SEEDS = {
    "s42": f"{ANNOT}/g1_eval_own10.json",
    "s7": f"{ANNOT}/g1_eval_own10_s7.json",
    "s1234": f"{ANNOT}/g1_eval_own10_s1234.json",
    "s2024": f"{ANNOT}/g1_eval_own10_s2024.json",
    "s3407": f"{ANNOT}/g1_eval_own10_s3407.json",
    "s31337": f"{ANNOT}/g1_eval_own10_s31337.json",
    "s55555": f"{ANNOT}/g1_eval_own10_s55555.json",
}
# pred_times_out.txt 里的实测预测 (臂,tag) -> [时刻...]
PRED_FILE = f"{os.path.dirname(os.path.abspath(__file__))}/pred_times_out.txt"


def frame_rms(audio, sr=16000, win=0.05):
    n = int(sr * win)
    k = len(audio) // n
    a = audio[:k * n].reshape(k, n)
    return (np.arange(k) + 0.5) * win, np.sqrt((a ** 2).mean(axis=1))


def parse_preds():
    """从 pred_times_out.txt 解析 {tag: [时刻]}."""
    if not os.path.exists(PRED_FILE):
        return {}
    out, cur = {}, None
    for line in open(PRED_FILE):
        m = re.search(r"▌ \[(NEVER|ALWAYS)\] (\S+)", line)
        if m:
            cur = m.group(2)
            out[cur] = []
            continue
        m = re.match(r"\s+预测\s+([\d.]+)s", line)
        if m and cur:
            out[cur].append(float(m.group(1)))
    return out


def main():
    per = {t: next(iter(json.load(open(p)).values()))["per_chunk"]
           for t, p in SEEDS.items()}
    mlook = load_manifest(OWN)
    ids = eval_chunk_ids(300)
    gt_chunks = [c for c in ids if per["s42"].get(c, {}).get("n_gt", 0) > 0]

    tier = {}
    for c in gt_chunks:
        n = sum(1 for t in per if per[t][c]["tp"] > 0)
        tier[c] = "always" if n == len(per) else ("never" if n == 0 else "swing")

    e1 = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{D}/candor_e1_w*.csv"))])
    ev = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{D}/candor_events_w*.csv"))])
    ev["session"] = ev["session"].astype(str)
    e1["session"] = e1["session"].astype(str)
    ev["ch_event"] = ev["ch_event"].astype(int)
    ev = ev.merge(e1[["event_id", "realized", "cls"]], on="event_id",
                  suffixes=("", "_e1"))
    bc = ev[(ev["cls"] == "BC")]

    print(f"含 GT 的块 {len(gt_chunks)}  "
          f"(always {sum(1 for v in tier.values() if v=='always')} / "
          f"swing {sum(1 for v in tier.values() if v=='swing')} / "
          f"never {sum(1 for v in tier.values() if v=='never')})")

    # ── A. 窗口是否完整落在 chunk 内 ─────────────────────────────────────
    rows = []
    for cid in gt_chunks:
        m = mlook[cid]
        t0 = m["t0"]
        sub = bc[(bc["session"] == str(m["session"]))
                 & (bc["ch_event"] == int(m["ch"]))]
        gtw, other = [], []
        for _, r in sub.iterrows():
            if r["end"] > t0 and r["start"] < t0 + CHUNK_S:
                rel = (r["start"] - t0, r["end"] - t0)
                (gtw if r["realized"] != "None" else other).append(rel)
        if not gtw:
            continue
        inside = [max(0.0, min(b, CHUNK_S) - max(a, 0.0)) / max(1e-9, b - a)
                  for a, b in gtw]
        rows.append({
            "cid": cid, "tier": tier[cid],
            "n_gt": len(gtw), "n_nonreal": len(other),
            "min_inside": round(min(inside), 3),
            "straddles": min(inside) < 0.999,
            "gtw": [(round(a, 2), round(b, 2)) for a, b in gtw],
            "other": [(round(a, 2), round(b, 2)) for a, b in other],
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("  A. GT 窗口是否完整落在 chunk 内")
    print("=" * 78)
    for t in ["always", "swing", "never"]:
        s = df[df.tier == t]
        if not len(s):
            continue
        n = int(s.straddles.sum())
        print(f"  {t:<8} n={len(s):2d}  横跨边界 {n:2d}  ({n/len(s)*100:.0f}%)"
              f"   窗口内占比中位数 {s.min_inside.median():.3f}")
    print()
    print(df[df.straddles][["cid", "tier", "min_inside", "gtw", "other"]]
          .to_string(index=False))

    # ── B. 有 realized=None 事件、且模型预测落在它上面的块 ───────────────
    preds = parse_preds()
    print("\n" + "=" * 78)
    print("  B. never 块的预测落在哪")
    print("=" * 78)
    tol = 0.5
    for cid in [r["cid"] for r in rows if r["tier"] == "never"]:
        r = next(x for x in rows if x["cid"] == cid)
        ts = preds.get(cid, [])
        note = []
        if not ts:
            note.append("答 'no backchannel'")
        for t in ts:
            hit_gt = any(a - tol <= t <= b + tol for a, b in r["gtw"])
            hit_no = any(a - tol <= t <= b + tol for a, b in r["other"])
            if hit_gt:
                note.append(f"{t}s 命中 GT")
            elif hit_no:
                note.append(f"**{t}s 落在 realized=None 事件上** "
                            f"(该事件不计 GT -> 记 FP)")
            else:
                note.append(f"{t}s 两边都不沾")
        print(f"  {cid}\n     GT {r['gtw']}  非GT事件 {r['other']}\n     "
              + "; ".join(note))


if __name__ == "__main__":
    main()
