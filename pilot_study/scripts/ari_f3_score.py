"""F3 人工验证打分: 收到填好的 judgments.csv 后计算三项核验.

  ① 人工 vs realized  : overlaps_host 与 truth.json 的 realized 一致率 + κ
                        -> 声道级"真重叠"证据链 (E1-E5) 的最终校准
  ② 人工 vs 标注器    : heard_bc 与 X2-Turn 窗口内 max p_bc (τ) 的一致
                        -> 标注器 BC 检出的 precision/recall
  ③ 分歧案例清单      : 人与系统不一致的片段, 供复查

用法 (fd_analysis): python scripts/ari_f3_score.py [--tau 0.2]
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

ANNOT = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator"
KIT = f"{ANNOT}/f3_kit"


def norm(v):
    v = (v or "").strip().lower()
    if v in ("y", "yes", "1", "true", "是"): return 1
    if v in ("n", "no", "0", "false", "否"): return 0
    return None


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = float((a == b).mean())
    pa1, pb1 = float(a.mean()), float(b.mean())
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def max_pbc(session, ch, t0, t1, cache):
    if session not in cache:
        p = f"{ANNOT}/e4_raw/{session}.json"
        cache[session] = json.load(open(p)) if os.path.exists(p) else None
    d = cache[session]
    if d is None:
        return None
    fr = d["channels"][str(ch)]["frames"]
    v = [f[3] for f in fr if f[1] > t0 and f[0] < t1]
    return float(max(v)) if v else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.2)
    args = ap.parse_args()

    truth = json.load(open(f"{KIT}/truth.json"))
    rows = list(csv.DictReader(open(f"{KIT}/judgments.csv")))
    cache = {}

    recs, unparsed = [], []
    for r in rows:
        cid = r["clip_id"]
        hb, ov = norm(r.get("heard_bc")), norm(r.get("overlaps_host"))
        if hb is None or ov is None:
            unparsed.append(cid)
            continue
        t = truth[cid]
        a, b = t["aws_window"]
        p = max_pbc(t["session"], t["listener_ch"], a, b, cache)
        recs.append({"cid": cid, "heard_bc": hb, "human_ov": ov,
                     "realized": 0 if t["realized"] == "None" else 1,
                     "x2": p, "x2_hit": int(p is not None and p >= args.tau)})

    if not recs:
        print("没有可解析的判定 (heard_bc / overlaps_host 都还是空的?)")
        print("未解析片段数:", len(unparsed))
        return

    n = len(recs)
    hum_ov = [r["human_ov"] for r in recs]
    rea = [r["realized"] for r in recs]
    hb = [r["heard_bc"] for r in recs]
    print(f"=== F3 人工验证 ({n}/{len(rows)} 条已填) ===\n")

    # ① 人工 vs realized
    agree = float(np.mean([a == b for a, b in zip(hum_ov, rea)]))
    print("① 人工 vs realized (声道级真重叠)")
    print(f"   一致率 {agree:.3f}   κ {kappa(hum_ov, rea):.3f}")
    for name, mask in [("realized=重叠", [i for i, x in enumerate(rea) if x]),
                       ("realized=停顿", [i for i, x in enumerate(rea) if not x])]:
        if mask:
            print(f"   {name}: n={len(mask)} 人工判重叠率 "
                  f"{np.mean([hum_ov[i] for i in mask]):.3f}")
    print()

    # ② 人工 vs 标注器
    tp = sum(1 for r in recs if r["x2_hit"] and r["heard_bc"])
    fp = sum(1 for r in recs if r["x2_hit"] and not r["heard_bc"])
    fn = sum(1 for r in recs if not r["x2_hit"] and r["heard_bc"])
    prec = tp / max(1, tp + fp)
    rec_ = tp / max(1, tp + fn)
    f1 = 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0
    print(f"② 人工 vs 标注器 (X2-Turn τ={args.tau})")
    print(f"   标注器 precision {prec:.3f}  recall {rec_:.3f}  F1 {f1:.3f}"
          f"   (tp {tp} / fp {fp} / fn {fn})")
    print(f"   一致率 {np.mean([a == b for a, b in zip(hb, [r['x2_hit'] for r in recs])]):.3f}"
          f"   κ {kappa(hb, [r['x2_hit'] for r in recs]):.3f}")
    print(f"   人工听到 BC 率 {np.mean(hb):.3f}  "
          f"(realized=重叠 {np.mean([hb[i] for i in range(n) if rea[i]] or [0]):.3f} / "
          f"停顿 {np.mean([hb[i] for i in range(n) if not rea[i]] or [0]):.3f})")
    print()

    # ③ 分歧清单
    dis = [r for r in recs if r["human_ov"] != r["realized"]
           or r["heard_bc"] != r["x2_hit"]]
    print(f"③ 分歧片段 {len(dis)} 个 (人工 vs realized 或 人工 vs 标注器)")
    for r in dis[:25]:
        print(f"   {r['cid']}: 人工[BC={r['heard_bc']} 重叠={r['human_ov']}] "
              f"| realized={r['realized']} | x2max={r['x2']:.3f}")
    if unparsed:
        print(f"\n未填片段 {len(unparsed)}: {unparsed[:10]}")

    json.dump({"n": n, "agree_human_realized": round(agree, 4),
               "kappa_human_realized": round(kappa(hum_ov, rea), 4),
               "annotator_precision": round(prec, 4),
               "annotator_recall": round(rec_, 4),
               "annotator_f1": round(f1, 4),
               "tau": args.tau},
              open(f"{ANNOT}/f3_score.json", "w"), indent=2)
    print("\nsaved f3_score.json")


if __name__ == "__main__":
    main()
