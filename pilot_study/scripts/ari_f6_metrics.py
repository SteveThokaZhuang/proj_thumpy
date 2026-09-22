"""F6: L3 指标定义 + 标注噪声鲁棒性 (v0).

指标 (在融合标注器 w0.7 的帧级软分数上定义, 窗口级聚合):
  BCRR (Backchannel Response Rate) : 融合分 >= τ 的窗口占比 (BC 行为密度)
  BCA  (Backchannel Alignment)     : 窗口 max 融合分 vs realized 真重叠的 AUROC
                                      (BC 是否落在真实重叠上)
  TBR  (Turn-Boundary Responsiveness): 预留 (需模型输出流, F7 实现)

敏感性 (v0 = 标注噪声鲁棒性; 行为敏感性待 F7):
  P1 延迟: 帧分数时间轴前移 δ ∈ {0.5, 1, 2}s
  P2 漏检: 随机置零 30/60% 帧
  P3 误报: 随机抬高 30/60% 帧至 1.0
  每扰动水平重算 BCRR/BCA -> 响应曲线 (指标应单调退化).

输出: f6_results.json + docs/pilot_study/figures/f6_metric_sensitivity.png
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
W = 0.7  # F1 最优融合权重


def build_frames():
    """F1 同款对齐: 返回 (窗口融合分列表, realized 标签, 帧级分数+窗口索引)."""
    mani = json.load(open(f"{ANNOT}/e5_regions_manifest.json"))
    sessions = set(m["session"] for m in mani.values())
    x2 = {}
    for fp in glob.glob(f"{ANNOT}/e4_raw/*.json"):
        s = os.path.basename(fp)[:-5]
        if s not in sessions:
            continue
        d = json.load(open(fp))
        x2[s] = {}
        for ch in ("0", "1"):
            fr = d["channels"][ch]["frames"]
            x2[s][ch] = (np.array([f[0] for f in fr]),
                         np.array([f[1] for f in fr]),
                         np.array([f[3] for f in fr]))
    sx = {}
    for key, meta in mani.items():
        s, ch = meta["session"], str(meta["ch"])
        p = f"{ANNOT}/e5_regions/{key}_states.json"
        if not os.path.exists(p):
            continue
        t0s, t1s, ps = [], [], []
        for st in json.load(open(p)):
            a = meta["t0"] + st["timestamp"][0]
            b = meta["t0"] + st["timestamp"][1]
            v = 1.0 if st["state"] == "backchannel" else 0.0
            t0s += [a, a + 0.08]
            t1s += [a + 0.08, b]
            ps += [v, v]
        # 同 (session, ch) 的多个区域必须累加 (setdefault 元组只保留首区域)
        buf = sx.setdefault(s, {}).setdefault(ch, [[], [], []])
        buf[0] += t0s
        buf[1] += t1s
        buf[2] += ps
    for s in sx:
        for ch in sx[s]:
            sx[s][ch] = tuple(np.array(x) for x in sx[s][ch])
    ev = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_events_w*.csv"))])
    e1 = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_e1_w*.csv"))])
    e1 = e1.merge(ev[["session", "event_id", "cls", "ch_event", "start", "end"]],
                  on="event_id", suffixes=("", "_ev"))
    e1 = e1[(e1["session"].isin(sessions)) & (e1["cls"] == "BC")]

    win_scores, win_lab = [], []
    frame_parts = []  # (scores, win_ids)
    wid = 0
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
                fused = W * xp[mx] + (1 - W) * s_al
                frame_parts.append((fused, xt0[mx], xt1[mx], (s, ch)))
                for _, r in bc.iterrows():
                    wm = ((xt0[mx] < r["end"]) & (xt1[mx] > r["start"]))
                    if wm.any():
                        win_scores.append(float(fused[wm].max()))
                        win_lab.append(1.0 if r["realized"] != "None" else 0.0)
                        wid += 1
    return np.array(win_scores), np.array(win_lab), frame_parts


def metrics(win_scores, win_lab, tau=0.5):
    bcrr = float((win_scores >= tau).mean())
    bca = roc_auc_score(win_lab, win_scores) if len(set(win_lab)) > 1 else None
    return {"BCRR": round(bcrr, 4),
            "BCA": round(float(bca), 4) if bca is not None else None}


def main():
    win_scores, win_lab, frame_parts = build_frames()
    print(f"windows: {len(win_scores)}, overlap frac: {win_lab.mean():.3f}",
          flush=True)

    res = {"clean": metrics(win_scores, win_lab)}

    # P1 延迟: 帧时间前移 δ 后窗口 max 重算 (预测迟到 = 帧内容相对窗口后移)
    mani = json.load(open(f"{ANNOT}/e5_regions_manifest.json"))
    for d in (0.5, 1.0, 2.0):
        ev = pd.concat([pd.read_csv(f, keep_default_na=False)
                        for f in sorted(glob.glob(
                            f"{DEFAULT_OUT}/candor_events_w*.csv"))])
        e1 = pd.concat([pd.read_csv(f, keep_default_na=False)
                        for f in sorted(glob.glob(
                            f"{DEFAULT_OUT}/candor_e1_w*.csv"))])
        e1 = e1.merge(ev[["session", "event_id", "cls", "ch_event", "start",
                          "end"]], on="event_id", suffixes=("", "_ev"))
        e1 = e1[(e1["session"].isin(set(m["session"] for m in mani.values())))
                & (e1["cls"] == "BC")]
        wsc, wlb = [], []
        for (fused, t0, t1, (s, ch)) in frame_parts:
            t0d, t1d = t0 + d, t1 + d
            bc = e1[(e1["session"] == s) & (e1["ch_event"] == int(ch))]
            for _, r in bc.iterrows():
                wm = ((t0d < r["end"]) & (t1d > r["start"]))
                if wm.any():
                    wsc.append(float(fused[wm].max()))
                    wlb.append(1.0 if r["realized"] != "None" else 0.0)
        res[f"delay_{d:g}s"] = metrics(np.array(wsc), np.array(wlb))

    # P2 漏检 / P3 误报 (在窗口分数上随机扰动)
    rng = np.random.default_rng(42)
    for rate in (0.3, 0.6):
        sc_miss = win_scores.copy()
        miss_idx = rng.random(len(sc_miss)) < rate
        sc_miss[miss_idx] = 0.0
        res[f"miss_{int(rate*100)}%"] = metrics(sc_miss, win_lab)
        sc_false = win_scores.copy()
        false_idx = rng.random(len(sc_false)) < rate
        sc_false[false_idx] = 1.0
        res[f"false_{int(rate*100)}%"] = metrics(sc_false, win_lab)

    print(json.dumps(res, indent=1), flush=True)
    with open(f"{ANNOT}/f6_results.json", "w") as f:
        json.dump(res, f, indent=2)

    # 图: 响应曲线
    fig, ax = plt.subplots(figsize=(9.5, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
    for s_ in ["left", "bottom"]:
        ax.spines[s_].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=9.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    names = ["clean", "delay_0.5s", "delay_1s", "delay_2s",
             "miss_30%", "miss_60%", "false_30%", "false_60%"]
    labels = ["clean", "delay 0.5s", "delay 1s", "delay 2s",
              "miss 30%", "miss 60%", "false 30%", "false 60%"]
    bca = [res[n]["BCA"] or 0 for n in names]
    bcrr = [res[n]["BCRR"] for n in names]
    x = np.arange(len(names))
    ax.plot(x, bca, "o-", color=C_BLUE, label="BCA (alignment with real overlap)",
            markersize=5)
    ax.plot(x, bcrr, "s-", color=C_ORANGE, label="BCRR (response rate)",
            markersize=5)
    ax.axhline(0.5, color=BASE, linewidth=1, linestyle="--")
    ax.set_xticks(x, labels, rotation=15, ha="right", fontsize=9, color=INK2)
    ax.set_ylabel("metric value", fontsize=10.5, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_title("F6: metric response under annotation perturbations",
                 fontsize=12, color=INK, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/f6_metric_sensitivity.png", dpi=300,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved f6_results.json + f6_metric_sensitivity.png")


if __name__ == "__main__":
    main()
