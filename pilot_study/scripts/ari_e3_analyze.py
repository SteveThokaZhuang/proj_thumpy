"""E3 分析: X2-Turn backchannel 状态 vs Behavior-SD 金标准 GT.

帧级: p_bc 对"帧是否落在 GT BC 窗口内"的 AUC; p_speak 对 GT utterance 的 AUC (合理性).
事件级: p_bc >= τ 的连续帧段 -> 事件, 与 GT BC 窗口 IoU>=0.3 匹配 -> P/R/F1 vs τ;
  argmax label=backchannel 的段作参照.

输出: e3_results.json + docs/pilot_study/figures/e3_calibration.png
用法 (gpu02 持久步骤内, fd_analysis 环境):
  python scripts/ari_e3_analyze.py [--raw-dir .../e3_raw]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FIG_DIR = "/share/workspace3/zhuangruicen/proj-thumpy/docs/pilot_study/figures"
DEFAULT_RAW = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator/e3_raw"
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, BASE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
TAUS = np.arange(0.1, 1.0, 0.05)
IOU_TH = 0.3


def frame_inside(frames, windows):
    """帧 (t0+t1)/2 落在任一窗口内 -> bool 数组."""
    mid = np.array([(f["t0"] + f["t1"]) / 2 for f in frames])
    ins = np.zeros(len(mid), bool)
    for w in windows:
        ins |= (mid >= w[0]) & (mid <= w[1])
    return ins


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def runs_to_events(times, mask, merge_gap=2):
    """mask (帧级 bool) -> 连续段 (合并 <=merge_gap 帧的间隙) -> [(t0,t1)]."""
    evs, s = [], None
    gap = 0
    for i, m in enumerate(mask):
        if m and s is None:
            s = times[i][0]
            gap = 0
        elif m:
            gap = 0
        elif s is not None:
            gap += 1
            if gap > merge_gap:
                evs.append((s, times[i - gap][1]))
                s = None
    if s is not None:
        evs.append((s, times[-1][1]))
    return evs


def event_prf(pred_evs, gt_wins):
    """匹配 = 预测事件 ≥50% 时长落在某 GT 窗口内 (贪心, 每窗口一次).

    注: 不用 IoU — GT BC 窗口 (~0.6s) 远长于模型的短暂 backchannel 触发
    (1-3 帧), IoU 对短预测永远不达标.
    """
    matched = 0
    used = [False] * len(gt_wins)
    for pe in pred_evs:
        best, bi = -1.0, -1
        for i, gw in enumerate(gt_wins):
            if used[i]:
                continue
            inter = max(0.0, min(pe[1], gw[1]) - max(pe[0], gw[0]))
            frac = inter / (pe[1] - pe[0]) if pe[1] > pe[0] else 0.0
            if frac > best:
                best, bi = frac, i
        if best >= 0.5 and bi >= 0:
            matched += 1
            used[bi] = True
    p = matched / len(pred_evs) if pred_evs else 0.0
    r = matched / len(gt_wins) if gt_wins else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=str, default=DEFAULT_RAW)
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.raw_dir}/*.json"))
    print(f"raw files: {len(files)}", flush=True)

    all_pbc, all_ins_bc = [], []
    all_psp, all_ins_utt = [], []
    per_file_f1_best = []
    n_bc_windows = 0
    bc_covered = 0
    best_f1, best_tau = 0.0, 0.0
    P, R, F1, COV = [], [], [], [0.0] * len(TAUS)
    PF, RF, F1F = [], [], []  # 上下文过滤版

    # 先经验估计 turn 标签的滞后量: p_speak 与 GT utterance 的互相关
    lags = np.arange(-40, 41)
    lag_scores = np.zeros(len(lags))
    tmp_pairs = []
    for fp in files:
        d = json.load(open(fp))
        for ch in ("0", "1"):
            frames = d["pred"][ch]["frames"]
            p_sp = np.array([f["p_speak"] for f in frames])
            gt_utt = [(u[0], u[1]) for u in d["gt"][ch]["utterances"]]
            if len(frames) < 100:
                continue
            ins = frame_inside(frames, gt_utt).astype(float)
            for li, lag in enumerate(lags):
                a, b = (p_sp, ins) if lag >= 0 else (ins, p_sp)
                lag = abs(lag)
                n = min(len(a) - lag, len(b))
                if n < 50:
                    continue
                x = a[lag: lag + n] if lag else a[:n]
                y = b[:n]
                if x.std() > 0 and y.std() > 0:
                    lag_scores[li] += np.corrcoef(x, y)[0, 1]
    best_lag = int(lags[int(np.argmax(lag_scores))])
    print(f"estimated turn-label lag: {best_lag} frames "
          f"({best_lag*80}ms)", flush=True)

    for fp in files:
        d = json.load(open(fp))
        for ch in ("0", "1"):
            frames = d["pred"][ch]["frames"]
            times = [(f["t0"], f["t1"]) for f in frames]
            p_bc = np.array([f["p_bc"] for f in frames])
            p_sp = np.array([f["p_speak"] for f in frames])
            gt_bc = [(w[0], w[1]) for w in d["gt"][ch]["bc"]]
            gt_utt = [(u[0], u[1]) for u in d["gt"][ch]["utterances"]]
            if not frames:
                continue
            # 对齐: 帧标签滞后音频 best_lag 帧 -> 时间轴前移 best_lag 帧
            # (帧 i 的真实时间 = times[i][0] - best_lag*0.08)
            shift_s = best_lag * 0.08
            times = [(t0 - shift_s, t1 - shift_s) for t0, t1 in times]
            ins_bc = frame_inside(frames, gt_bc)
            ins_utt = frame_inside(frames, gt_utt)
            if ins_bc.any() and (~ins_bc).any():
                all_pbc.append(p_bc)
                all_ins_bc.append(ins_bc)
            if ins_utt.any() and (~ins_utt).any():
                all_psp.append(p_sp)
                all_ins_utt.append(ins_utt)
            # 事件级: 每个 τ 的 P/R/F1 (原始 + 上下文过滤)
            other = d["pred"]["0" if ch == "1" else "1"]["frames"]
            ot_times = [(f["t0"], f["t1"]) for f in other]
            ot_speak = np.array([f["p_speak"] >= 0.5 for f in other])
            for j, tau in enumerate(TAUS):
                evs = runs_to_events(times, p_bc >= tau)
                p, r, f1 = event_prf(evs, gt_bc)
                if j == len(P):
                    P.append(p)
                    R.append(r)
                    F1.append(f1)
                else:
                    P[j] += p
                    R[j] += r
                    F1[j] += f1
                # 过滤: 事件期间对方声道 >=50% 帧在 speaking (模型判定宿主在说)
                evs_f = []
                for (t0, t1) in evs:
                    mid = (t0 + t1) / 2
                    win = [i for i, (a, b) in enumerate(ot_times)
                           if max(a, t0) < min(b, t1)]
                    if win and np.mean([ot_speak[i] for i in win]) >= 0.5:
                        evs_f.append((t0, t1))
                pf, rf, f1f = event_prf(evs_f, gt_bc)
                if j == len(PF):
                    PF.append(pf)
                    RF.append(rf)
                    F1F.append(f1f)
                else:
                    PF[j] += pf
                    RF[j] += rf
                    F1F[j] += f1f
            # 窗口检出率: GT 窗口内含 >=1 帧 p_bc>=τ
            if gt_bc:
                inside_idx = []
                for w in gt_bc:
                    inside_idx.append([i for i, (t0, t1) in enumerate(times)
                                       if max(t0, w[0]) < min(t1, w[1])])
                for j, tau in enumerate(TAUS):
                    mask = p_bc >= tau
                    cov = sum(int(any(mask[i] for i in idx))
                              for idx in inside_idx)
                    COV[j] += cov / len(gt_bc)
            # argmax 参照
            evs_argmax = runs_to_events(
                times, [f["label"] == "backchannel" for f in frames])
            p, r, f1 = event_prf(evs_argmax, gt_bc)
            per_file_f1_best.append(f1)
            n_bc_windows += len(gt_bc)
            if gt_bc:
                bc_covered += sum(1 for w in gt_bc
                                  if any(iou(w, e) >= IOU_TH for e in evs_argmax))
    n_ch = len(files) * 2
    P = [p / max(1, n_ch) for p in P]
    R = [r / max(1, n_ch) for r in R]
    F1 = [f / max(1, n_ch) for f in F1]
    COV = [c / max(1, n_ch) for c in COV]
    PF = [p / max(1, n_ch) for p in PF]
    RF = [r / max(1, n_ch) for r in RF]
    F1F = [f / max(1, n_ch) for f in F1F]
    best_i = int(np.argmax(F1))
    best_tau, best_f1 = float(TAUS[best_i]), F1[best_i]
    bestf_i = int(np.argmax(F1F))
    best_tau_f, best_f1_f = float(TAUS[bestf_i]), F1F[bestf_i]

    auc_bc = roc_auc_score(np.concatenate(all_ins_bc), np.concatenate(all_pbc)) \
        if all_pbc else float("nan")
    auc_sp = roc_auc_score(np.concatenate(all_ins_utt), np.concatenate(all_psp)) \
        if all_psp else float("nan")

    res = {
        "n_files": len(files), "n_bc_windows": int(n_bc_windows),
        "best_lag_frames": int(best_lag),
        "auc_bc": round(float(auc_bc), 4), "auc_speak": round(float(auc_sp), 4),
        "best_tau": round(best_tau, 2), "best_f1": round(float(best_f1), 4),
        "P": [round(x, 4) for x in P], "R": [round(x, 4) for x in R],
        "F1": [round(x, 4) for x in F1],
        "P_filtered": [round(x, 4) for x in PF],
        "R_filtered": [round(x, 4) for x in RF],
        "F1_filtered": [round(x, 4) for x in F1F],
        "best_tau_filtered": round(best_tau_f, 2),
        "best_f1_filtered": round(float(best_f1_f), 4),
        "window_coverage": [round(x, 4) for x in COV],
        "argmax_covered_frac": round(float(bc_covered / max(1, n_bc_windows)), 4),
        "per_file_argmax_f1_mean": round(float(np.mean(per_file_f1_best)), 4),
        "per_file_argmax_f1_std": round(float(np.std(per_file_f1_best)), 4),
    }
    print(json.dumps(res, indent=1), flush=True)
    with open(f"{os.path.dirname(args.raw_dir)}/e3_results.json", "w") as f:
        json.dump(res, f, indent=2)

    # 图
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=9.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.plot(TAUS, P, "o-", color=C_BLUE, label="precision", markersize=4)
    ax.plot(TAUS, R, "s-", color=C_ORANGE, label="recall", markersize=4)
    ax.plot(TAUS, F1, "^-", color=C_AQUA, label="F1", markersize=4)
    ax.plot(TAUS, F1F, "v--", color=C_BLUE, label="F1 (context-filtered)",
            markersize=4)
    ax.axvline(best_tau, color=INK, linestyle="--", linewidth=1)
    ax.text(best_tau + 0.02, best_f1 + 0.02, f"best τ={best_tau:.2f}, F1={best_f1:.2f}",
            fontsize=9.5, color=INK)
    ax.text(0.05, 0.08, f"filtered: τ={best_tau_f:.2f}, F1={best_f1_f:.2f}",
            transform=ax.transAxes, fontsize=9.5, color=C_BLUE)
    ax.set_xlabel("p(backchannel) threshold τ", fontsize=10.5, color=INK)
    ax.set_ylabel("event-level (IoU≥0.3)", fontsize=10.5, color=INK)
    ax.set_ylim(0, 1.02)
    ax.set_title("X2-Turn backchannel events vs GT (per τ)",
                 fontsize=11, color=INK, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=9.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    bars = ax.bar([0, 1], [auc_bc, auc_sp], 0.5, color=[C_BLUE, C_AQUA])
    for b, v in zip(bars, [auc_bc, auc_sp]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.axhline(0.5, color=BASE, linewidth=1, linestyle="--")
    ax.set_xticks([0, 1], ["p(backchannel)\nvs GT BC frames",
                           "p(speaking)\nvs GT utterances"],
                  fontsize=9.5, color=INK2)
    ax.set_ylabel("frame-level AUC", fontsize=10.5, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_title("Frame-level discriminability (sanity + target)",
                 fontsize=11, color=INK, fontweight="bold")
    fig.suptitle(f"E3: X2-Turn gold calibration on Behavior-SD "
                 f"({len(files)} files, {n_bc_windows} GT backchannels)",
                 fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/e3_calibration.png", dpi=300,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved e3_results.json + e3_calibration.png")


if __name__ == "__main__":
    main()
