"""E5 分析: SoulX-Duplug 与 X2-Turn 双标注器交叉验证 (CANDOR BC 窗口邻域).

问题: 两个模型化标注器在真实音频上的一致性与互补性?
  1. 各自对 AWS BC 窗口的检出覆盖率
  2. 窗口级一致性 (both/one/none 分布)
  3. 共识子集假说: 双标注器同时触发的窗口, realized 真重叠率是否更高?
     (若成立 -> 共识子集是更高置信的标注)
  4. 窗口外的误触发率

输入: e5_regions_manifest.json + e5_regions/*_states.json (SoulX 160ms 状态)
      + e4_raw/*.json (X2-Turn 80ms 帧) + candor_events/e1 (AWS/realized)

输出: e5_results.json + docs/pilot_study/figures/e5_dual_annotator.png
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_analyze import DEFAULT_OUT  # noqa: E402

FIG_DIR = "/share/workspace3/zhuangruicen/proj-thumpy/docs/pilot_study/figures"
ANNOT = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator"
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, BASE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"


def load_soulx(mani):
    """soulx: session->ch->(t0,t1,state) 列表 (区域拼接, 会话时间轴)."""
    out = {}
    for key, meta in mani.items():
        s, ch = meta["session"], str(meta["ch"])
        p = f"{ANNOT}/e5_regions/{key}_states.json"
        if not os.path.exists(p):
            continue
        states = json.load(open(p))
        recs = [(meta["t0"] + st["timestamp"][0], meta["t0"] + st["timestamp"][1],
                 st["state"]) for st in states]
        out.setdefault(s, {}).setdefault(ch, []).extend(recs)
    return out


def load_x2():
    out = {}
    for fp in glob.glob(f"{ANNOT}/e4_raw/*.json"):
        s = os.path.basename(fp)[:-5]
        d = json.load(open(fp))
        out[s] = {}
        for ch in ("0", "1"):
            fr = d["channels"][ch]["frames"]
            out[s][ch] = {"t0": np.array([f[0] for f in fr]),
                          "t1": np.array([f[1] for f in fr]),
                          "p_bc": np.array([f[3] for f in fr])}
    return out


def window_hit_soulx(sess, ch, w, sx):
    recs = sx.get(sess, {}).get(ch, [])
    return any(max(a, w[0]) < min(b, w[1]) and st == "backchannel"
               for (a, b, st) in recs)


def window_hit_x2(sess, ch, w, x2, tau=0.5):
    fr = x2.get(sess, {}).get(ch)
    if fr is None:
        return False
    m = (fr["t0"] < w[1]) & (fr["t1"] > w[0]) & (fr["p_bc"] >= tau)
    return bool(m.any())


def main():
    mani = json.load(open(f"{ANNOT}/e5_regions_manifest.json"))
    sx = load_soulx(mani)
    x2 = load_x2()
    print(f"soulx sessions: {len(sx)}, x2 sessions: {len(x2)}", flush=True)

    ev = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_events_w*.csv"))])
    e1 = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_e1_w*.csv"))])
    e1 = e1.merge(ev[["session", "event_id", "cls", "ch_event", "start", "end"]],
                  on="event_id", suffixes=("", "_ev"))
    e1 = e1[e1["session"].isin(set(mani[v]["session"] for v in mani))]

    rows = []
    for sess in sorted(set(m["session"] for m in mani.values())):
        sub = e1[(e1["session"] == sess) & (e1["cls"] == "BC")]
        for _, r in sub.iterrows():
            w = (r["start"], r["end"])
            h_s = window_hit_soulx(sess, str(r["ch_event"]), w, sx)
            h_x = window_hit_x2(sess, str(r["ch_event"]), w, x2)
            rows.append({"session": sess, "hit_soulx": h_s, "hit_x2": h_x,
                         "realized": r["realized"]})
    df = pd.DataFrame(rows)
    n = len(df)
    res = {
        "n_bc_windows": n,
        "cov_soulx": round(float(df.hit_soulx.mean()), 4),
        "cov_x2": round(float(df.hit_x2.mean()), 4),
        "both_frac": round(float((df.hit_soulx & df.hit_x2).mean()), 4),
        "either_frac": round(float((df.hit_soulx | df.hit_x2).mean()), 4),
        "agreement": round(float((df.hit_soulx == df.hit_x2).mean()), 4),
        "kappa": None,
    }
    try:
        from sklearn.metrics import cohen_kappa_score
        res["kappa"] = round(cohen_kappa_score(df.hit_soulx, df.hit_x2), 4)
    except Exception:
        pass

    # 共识子集假说: 与 realized 真重叠率的关系
    df["ov"] = (df["realized"] != "None").astype(int)
    grp = {}
    for name, mask in [("both", df.hit_soulx & df.hit_x2),
                       ("soulx_only", df.hit_soulx & ~df.hit_x2),
                       ("x2_only", ~df.hit_soulx & df.hit_x2),
                       ("neither", ~df.hit_soulx & ~df.hit_x2)]:
        if mask.sum():
            grp[name] = {"n": int(mask.sum()),
                         "overlap_frac": round(float(df.loc[mask, "ov"].mean()), 4)}
    res["groups"] = grp
    print(json.dumps(res, indent=1), flush=True)
    with open(f"{ANNOT}/e5_results.json", "w") as f:
        json.dump(res, f, indent=2)

    # 图
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor=SURFACE)
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
    for s_ in ["left", "bottom"]:
        ax.spines[s_].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=9.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    names = ["SoulX", "X2-Turn", "both", "either"]
    vals = [res["cov_soulx"], res["cov_x2"], res["both_frac"], res["either_frac"]]
    bars = ax.bar(range(4), vals, 0.55, color=[C_BLUE, C_ORANGE, C_AQUA, "#86b6ef"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_xticks(range(4), names, fontsize=9.5, color=INK2)
    ax.set_ylabel("AWS BC window coverage", fontsize=10.5, color=INK)
    ax.set_ylim(0, max(vals) + 0.08)
    ax.set_title(f"Dual-annotator coverage ({n} windows, "
                 f"κ={res['kappa']})", fontsize=11, color=INK, fontweight="bold")

    ax = axes[1]
    ax.set_facecolor(SURFACE)
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
    for s_ in ["left", "bottom"]:
        ax.spines[s_].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=9.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    order = ["both", "soulx_only", "x2_only", "neither"]
    vals = [grp.get(k, {}).get("overlap_frac", 0) for k in order]
    ns = [grp.get(k, {}).get("n", 0) for k in order]
    bars = ax.bar(range(4), vals, 0.55, color=[C_AQUA, C_BLUE, C_ORANGE, "#c3c2b7"])
    for b, v, nn in zip(bars, vals, ns):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_xticks(range(4), [f"{k}\n(n={nn})" for k, nn in zip(order, ns)],
                  fontsize=9, color=INK2)
    ax.set_ylabel("fraction with real overlap (realized)", fontsize=10.5, color=INK)
    ax.set_ylim(0, max(vals + [0.5]) + 0.12)
    ax.set_title("Consensus hypothesis: overlap rate by annotator agreement",
                 fontsize=11, color=INK, fontweight="bold")
    fig.suptitle("E5: SoulX × X2-Turn dual annotator on CANDOR",
                 fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/e5_dual_annotator.png", dpi=300,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved e5_results.json + e5_dual_annotator.png")


if __name__ == "__main__":
    main()
