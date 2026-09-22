"""E4 分析: X2-Turn 帧级信号在 CANDOR 上的校准与对照 (语义一致口径).

指标 (与 E3 修正后的协议一致, 帧级为主):
  1. p_bc 帧级 AUC vs AWS BC 窗口隶属 (对照 E3 合成数据 0.78)
  2. p_speak 帧级 AUC vs 本声道 turn 窗口 (管线合理性)
  3. 窗口级 AUROC: BC 窗口的 max p_bc 是否能判别该窗口 realized 真重叠
     (E1 的声道 RMS 标签) — 新问题: 真实音频上模型 BC 信号追踪重叠吗?
  4. 标签空间错配率: 高 p_bc 帧落在自己话轮内的比例 (对照 E3 的 64%)
  5. 事件检出: best-τ 对 AWS BC 窗口的覆盖率 (对照 E3 的 41.5%)

输出: e4_results.json + docs/pilot_study/figures/e4_candor.png
用法 (fd_analysis 环境, gpu02):
  python scripts/ari_e4_analyze.py
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
RAW = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator/e4_raw"
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, BASE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"


def load_frames(session):
    d = json.load(open(f"{RAW}/{session}.json"))
    out = {}
    for ch in ("0", "1"):
        fr = d["channels"][ch]["frames"]
        out[ch] = {
            "t0": np.array([f[0] for f in fr]),
            "t1": np.array([f[1] for f in fr]),
            "p_bc": np.array([f[3] for f in fr]),
            "p_sp": np.array([f[4] for f in fr]),
        }
    return out


def inside(frames, windows):
    mid = (frames["t0"] + frames["t1"]) / 2
    m = np.zeros(len(mid), bool)
    for w in windows:
        m |= (mid >= w[0]) & (mid <= w[1])
    return m


def main():
    sessions = sorted(os.path.basename(f)[:-5]
                      for f in glob.glob(f"{RAW}/*.json"))
    print(f"sessions: {len(sessions)}", flush=True)

    # AWS 事件 + E1 realized
    ev = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_events_w*.csv"))])
    e1 = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_e1_w*.csv"))])
    e1 = e1.merge(ev[["session", "event_id", "cls", "ch_event", "start", "end"]],
                  on="event_id", suffixes=("", "_ev"))
    e1 = e1[e1["session"].isin(sessions)]
    print(f"events in sessions: {len(e1)}", flush=True)

    aucs_bc, aucs_sp = [], []
    win_feat, win_lab = [], []      # 窗口级: max p_bc vs realized overlap
    own_utt_total, own_utt_high = 0, 0   # 错配率
    cover = {t: [] for t in np.arange(0.1, 1.0, 0.1)}
    for s in sessions:
        try:
            fr = load_frames(s)
        except Exception as e:
            print(f"skip {s}: {e}", flush=True)
            continue
        sub = e1[e1["session"] == s]
        for ch in ("0", "1"):
            chf = fr[ch]
            if len(chf["p_bc"]) < 1000:
                continue
            bc = sub[(sub["cls"] == "BC") & (sub["ch_event"] == int(ch))]
            turns = sub[(sub["cls"].isin(["Int", "None"]))
                        & (sub["ch_event"] == int(ch))]
            bc_wins = list(zip(bc["start"], bc["end"]))
            turn_wins = list(zip(turns["start"], turns["end"]))
            # 1. p_bc vs BC 窗口
            ins_bc = inside(chf, bc_wins)
            if ins_bc.any() and (~ins_bc).any():
                # 负样本下采样至 5 万
                neg_idx = np.where(~ins_bc)[0]
                if len(neg_idx) > 50000:
                    neg_idx = np.random.default_rng(0).choice(
                        neg_idx, 50000, replace=False)
                idx = np.concatenate([np.where(ins_bc)[0], neg_idx])
                aucs_bc.append(roc_auc_score(ins_bc[idx], chf["p_bc"][idx]))
            # 2. p_sp vs turn 窗口
            ins_t = inside(chf, turn_wins)
            if ins_t.any() and (~ins_t).any():
                neg_idx = np.where(~ins_t)[0]
                if len(neg_idx) > 50000:
                    neg_idx = np.random.default_rng(0).choice(
                        neg_idx, 50000, replace=False)
                idx = np.concatenate([np.where(ins_t)[0], neg_idx])
                aucs_sp.append(roc_auc_score(ins_t[idx], chf["p_sp"][idx]))
            # 3. 窗口级: max p_bc vs realized overlap (BC 事件)
            bc_sorted = bc.sort_values("start")
            for _, row in bc_sorted.iterrows():
                m = (chf["t0"] < row["end"]) & (chf["t1"] > row["start"])
                if m.any():
                    win_feat.append(float(chf["p_bc"][m].max()))
                    win_lab.append(1.0 if row["realized"] != "None" else 0.0)
            # 4. 错配率: 高 p_bc 帧的位置
            high = chf["p_bc"] >= 0.5
            if high.any():
                own = inside(
                    {"t0": chf["t0"][high], "t1": chf["t1"][high]}, turn_wins)
                own_utt_total += int(high.sum())
                own_utt_high += int(own.sum())
            # 5. 覆盖率
            for tau in cover:
                m2 = chf["p_bc"] >= tau
                cov = 0
                for w in bc_wins:
                    hit = ((chf["t0"] < w[1]) & (chf["t1"] > w[0]) & m2).any()
                    cov += int(hit)
                cover[tau].append(cov / len(bc_wins) if bc_wins else 0.0)

    res = {
        "n_sessions": len(sessions),
        "auc_bc_mean": round(float(np.mean(aucs_bc)), 4) if aucs_bc else None,
        "auc_bc_std": round(float(np.std(aucs_bc)), 4) if aucs_bc else None,
        "auc_speak_mean": round(float(np.mean(aucs_sp)), 4) if aucs_sp else None,
        "win_auroc_overlap": round(float(roc_auc_score(win_lab, win_feat)), 4)
        if len(set(win_lab)) > 1 else None,
        "n_bc_windows": int(len(win_lab)),
        "own_turn_frac_high": round(float(own_utt_high / max(1, own_utt_total)), 4),
        "coverage": {f"tau{t:.1f}": round(float(np.mean(v)), 4)
                     for t, v in cover.items()},
    }
    print(json.dumps(res, indent=1), flush=True)
    with open(f"{os.path.dirname(RAW)}/e4_results.json", "w") as f:
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
    bars = ax.bar([0, 1], [res["auc_bc_mean"] or 0, res["auc_speak_mean"] or 0],
                  0.5, color=[C_BLUE, C_AQUA])
    for b, v in zip(bars, [res["auc_bc_mean"], res["auc_speak_mean"]]):
        ax.text(b.get_x() + b.get_width() / 2, (v or 0) + 0.02,
                f"{v}", ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.axhline(0.5, color=BASE, linewidth=1, linestyle="--")
    ax.set_xticks([0, 1], ["p_bc vs AWS BC\nwindows", "p_speak vs\nturn windows"],
                  fontsize=9.5, color=INK2)
    ax.set_ylabel("frame-level AUC", fontsize=10.5, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_title("X2-Turn on CANDOR (real audio)", fontsize=11,
                 color=INK, fontweight="bold")

    ax = axes[1]
    ax.set_facecolor(SURFACE)
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
    for s_ in ["left", "bottom"]:
        ax.spines[s_].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=9.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    covs = [res["coverage"][f"tau{t:.1f}"] for t in taus]
    ax.plot(taus, covs, "o-", color=C_ORANGE, markersize=4.5)
    ax.set_xlabel("p_bc threshold τ", fontsize=10.5, color=INK)
    ax.set_ylabel("AWS BC window coverage", fontsize=10.5, color=INK)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Window coverage by τ "
                 f"(win AUROC vs overlap: {res['win_auroc_overlap']})",
                 fontsize=11, color=INK, fontweight="bold")
    fig.suptitle(f"E4: X2-Turn as annotator on CANDOR ({len(sessions)} sessions, "
                 f"{res['n_bc_windows']} BC windows; "
                 f"own-turn high-p_bc frac {res['own_turn_frac_high']})",
                 fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/e4_candor.png", dpi=300,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved e4_results.json + e4_candor.png")


if __name__ == "__main__":
    main()
