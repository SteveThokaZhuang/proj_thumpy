"""F4c: 四标签源一致性矩阵 (CANDOR-FD 100 会话, BC 窗口级).

在 E1 三源 (AWS/VAD/realized) 基础上加入 X2-Turn (L3) 成为四源:
  AWS      : backbiter BC 窗口 (常量, 所有窗口都是)
  VAD      : Silero 检出 (E1 的 vad_seen)
  realized : 声道级真重叠 (realized != None)
  X2-Turn  : 窗口内 max p_bc >= tau (L3 标注器)
指标: 三信息源的 pairwise Cohen's kappa + 各自覆盖率 + 共识子集分布.

用法 (fd_analysis): python scripts/ari_f4c_foursource.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_analyze import DEFAULT_OUT  # noqa: E402

ANNOT = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator"
TAU = 0.2


def main():
    ev = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_events_w*.csv"))])
    e1 = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_e1_w*.csv"))])
    e1 = e1.merge(ev[["session", "event_id", "cls", "ch_event", "start", "end"]],
                  on="event_id", suffixes=("", "_ev"))
    e1["vad_seen"] = pd.to_numeric(e1["vad_seen"], errors="coerce").fillna(0)
    # X2 帧 (会话->ch->(t0,t1,p_bc))
    x2 = {}
    for fp in glob.glob(f"{ANNOT}/e4_raw/*.json"):
        s = os.path.basename(fp)[:-5]
        d = json.load(open(fp))
        x2[s] = {}
        for ch in ("0", "1"):
            fr = d["channels"][ch]["frames"]
            x2[s][ch] = (np.array([f[0] for f in fr]),
                         np.array([f[1] for f in fr]),
                         np.array([f[3] for f in fr]))
    bc = e1[(e1["cls"] == "BC") & (e1["session"].isin(x2))]
    rows = []
    for _, r in bc.iterrows():
        t0a, t1a, p = x2[r["session"]][str(r["ch_event"])]
        fm = (t0a < r["end"]) & (t1a > r["start"])
        hit = bool(fm.any() and p[fm].max() >= TAU)
        rows.append({
            "vad": int(r["vad_seen"] == 1),
            "realized": int(r["realized"] != "None"),
            "x2": int(hit),
        })
    df = pd.DataFrame(rows)
    n = len(df)
    k = {
        "vad_vs_realized": round(cohen_kappa_score(df.vad, df.realized), 4),
        "x2_vs_realized": round(cohen_kappa_score(df.x2, df.realized), 4),
        "vad_vs_x2": round(cohen_kappa_score(df.vad, df.x2), 4),
    }
    cov = {src: round(float(df[src].mean()), 4)
           for src in ["vad", "realized", "x2"]}
    consensus = {
        "all3": int(((df.vad == 1) & (df.realized == 1) & (df.x2 == 1)).sum()),
        "none": int(((df.vad == 0) & (df.realized == 0) & (df.x2 == 0)).sum()),
    }
    res = {"n_windows": n, "kappa": k, "coverage": cov,
           "consensus": consensus}
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(f"{ANNOT}/f4c_results.json", "w"), indent=2)
    print("saved f4c_results.json")


if __name__ == "__main__":
    main()
