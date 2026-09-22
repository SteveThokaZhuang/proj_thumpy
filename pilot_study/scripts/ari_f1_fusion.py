"""F1: X2-Turn × SoulX 概率级融合 (零新推理, 复用 E4+E5 产物).

数据: CANDOR 5 会话, 328 个 AWS BC 窗口 (E5 区域 + manifest).
流程: 每区域一次对齐 — X2-Turn 80ms 帧为基准, SoulX 160ms 块升采样后最近邻
对齐到同一帧网格, 然后:
  x2_only  : p_bc 软概率
  sx_only  : backchannel 状态伪概率 (1.0/0.0)
  max      : max(x2, sx)
  mean     : (x2 + sx)/2
  w0.3/0.5/0.7 : w·x2 + (1-w)·sx
评价 (仅区域内帧):
  1. 帧级 AUC vs AWS BC 窗口隶属
  2. 窗口级 AUROC: 窗口内 max 融合分 vs realized 真重叠
  3. 事件级: 每个变体在 τ 网格上的窗口覆盖率 (报告最优)

输出: f1_fusion_results.json + docs/pilot_study/figures/f1_fusion.png
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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
TAUS = np.arange(0.1, 1.0, 0.1)


def load_x2(sessions):
    out = {}
    for fp in glob.glob(f"{ANNOT}/e4_raw/*.json"):
        s = os.path.basename(fp)[:-5]
        if s not in sessions:
            continue
        d = json.load(open(fp))
        out[s] = {}
        for ch in ("0", "1"):
            fr = d["channels"][ch]["frames"]
            out[s][ch] = (np.array([f[0] for f in fr]),
                          np.array([f[1] for f in fr]),
                          np.array([f[3] for f in fr]))
    return out


def load_soulx(mani):
    """soulx 帧: 160ms 块拆成 2 个 80ms 帧 (同分).

    注意: 同一 (session, ch) 的多个区域必须累加 (此前 setdefault 元组只
    保留首个区域, 导致每声道只有 r0 区域的帧).
    """
    out = {}
    for key, meta in mani.items():
        s, ch = meta["session"], str(meta["ch"])
        p = f"{ANNOT}/e5_regions/{key}_states.json"
        if not os.path.exists(p):
            continue
        states = json.load(open(p))
        t0s, t1s, ps = [], [], []
        for st in states:
            a = meta["t0"] + st["timestamp"][0]
            b = meta["t0"] + st["timestamp"][1]
            v = 1.0 if st["state"] == "backchannel" else 0.0
            t0s += [a, a + 0.08]
            t1s += [a + 0.08, b]
            ps += [v, v]
        buf = out.setdefault(s, {}).setdefault(ch, [[], [], []])
        buf[0] += t0s
        buf[1] += t1s
        buf[2] += ps
    for s in out:
        for ch in out[s]:
            out[s][ch] = tuple(np.array(x) for x in out[s][ch])
    return out


VARIANTS = {"x2_only": None, "sx_only": None, "max": None, "mean": None,
            "w0.3": 0.3, "w0.5": 0.5, "w0.7": 0.7}


def fused_scores(x2s, s_al, variant):
    if variant == "x2_only":
        return x2s
    if variant == "sx_only":
        return s_al
    if variant == "max":
        return np.maximum(x2s, s_al)
    if variant == "mean":
        return (x2s + s_al) / 2.0
    return VARIANTS[variant] * x2s + (1 - VARIANTS[variant]) * s_al


def main():
    mani = json.load(open(f"{ANNOT}/e5_regions_manifest.json"))
    sessions = set(m["session"] for m in mani.values())
    x2 = load_x2(sessions)
    sx = load_soulx(mani)

    ev = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_events_w*.csv"))])
    e1 = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_e1_w*.csv"))])
    e1 = e1.merge(ev[["session", "event_id", "cls", "ch_event", "start", "end"]],
                  on="event_id", suffixes=("", "_ev"))
    e1 = e1[(e1["session"].isin(sessions)) & (e1["cls"] == "BC")]

    frame_scores = {v: [] for v in VARIANTS}
    frame_ins = []
    win_scores = {v: [] for v in VARIANTS}
    win_lab = []
    win_cov = {v: {t: [] for t in TAUS} for v in VARIANTS}

    for s in sorted(sessions):
        if s not in x2 or s not in sx:
            continue
        for ch in ("0", "1"):
            if ch not in x2[s] or ch not in sx[s]:
                continue
            xt0, xt1, xp = x2[s][ch]
            st0, st1, sp = sx[s][ch]
            if len(xt0) == 0 or len(st0) == 0:
                continue
            spans = sorted(set((m["t0"], m["t1"]) for m in mani.values()
                               if m["session"] == s and str(m["ch"]) == ch))
            bc = e1[(e1["session"] == s) & (e1["ch_event"] == int(ch))]
            for (a, b) in spans:
                mx = (xt0 < b) & (xt1 > a)
                ms = (st0 < b) & (st1 > a)
                if mx.sum() < 50 or ms.sum() < 20:
                    continue
                xct = (xt0[mx] + xt1[mx]) / 2
                sct = (st0[ms] + st1[ms]) / 2
                s_al = sp[ms][np.abs(sct[:, None] - xct[None, :]).argmin(axis=0)]
                ins = np.zeros(mx.sum(), bool)
                for _, r in bc.iterrows():
                    ins |= (xt0[mx] < r["end"]) & (xt1[mx] > r["start"])
                frame_ins.append(ins)
                for v in VARIANTS:
                    sc = fused_scores(xp[mx], s_al, v)
                    frame_scores[v].append(sc)
                    # 窗口级: 窗口内 max
                    for _, r in bc.iterrows():
                        wm = ((xt0[mx] < r["end"]) & (xt1[mx] > r["start"]))
                        if wm.any():
                            win_scores[v].append(float(sc[wm].max()))
                            if v == "x2_only":
                                win_lab.append(
                                    1.0 if r["realized"] != "None" else 0.0)
                            for t in TAUS:
                                win_cov[v][t].append(int(sc[wm].max() >= t))

    win_lab = np.array(win_lab)
    res = {}
    for v in VARIANTS:
        fs = np.concatenate(frame_scores[v])
        ins = np.concatenate(frame_ins)
        auc = roc_auc_score(ins, fs) if ins.any() and (~ins).any() else None
        auroc = roc_auc_score(win_lab, np.array(win_scores[v])) \
            if len(set(win_lab)) > 1 else None
        best_cov = max((np.mean(win_cov[v][t]), t) for t in TAUS)
        res[v] = {"frame_auc": round(float(auc), 4) if auc is not None else None,
                  "win_auroc": round(float(auroc), 4) if auroc is not None else None,
                  "best_coverage": round(float(best_cov[0]), 4),
                  "best_coverage_tau": round(float(best_cov[1]), 2)}
        print(v, res[v], flush=True)

    with open(f"{ANNOT}/f1_fusion_results.json", "w") as f:
        json.dump(res, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor=SURFACE)
    for ax, key, title in [(axes[0], "frame_auc", "Frame AUC vs BC windows"),
                           (axes[1], "win_auroc", "Window AUROC vs realized overlap")]:
        ax.set_facecolor(SURFACE)
        for s_ in ["top", "right"]:
            ax.spines[s_].set_visible(False)
        for s_ in ["left", "bottom"]:
            ax.spines[s_].set_color(BASE)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        names = list(VARIANTS.keys())
        vals = [res[k][key] or 0 for k in names]
        colors = [C_ORANGE, C_BLUE, C_AQUA, "#86b6ef", "#86b6ef", "#86b6ef"]
        bars = ax.bar(range(len(names)), vals, 0.6, color=colors)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                    ha="center", fontsize=9.5, color=INK, fontweight="bold")
        ax.axhline(0.5, color=BASE, linewidth=1, linestyle="--")
        ax.set_xticks(range(len(names)), names, rotation=25, ha="right",
                      fontsize=8.5, color=INK2)
        ax.set_ylabel(key, fontsize=10, color=INK)
        ax.set_ylim(0.45, max(vals + [0.6]) + 0.08)
        ax.set_title(title, fontsize=11, color=INK, fontweight="bold")
    fig.suptitle("F1: annotator fusion (X2-Turn × SoulX) on CANDOR BC windows",
                 fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/f1_fusion.png", dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved f1_fusion_results.json + f1_fusion.png")


if __name__ == "__main__":
    main()
