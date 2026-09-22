"""Fig 1: 三层标签源框架图 (论文用).

L1 生成端 GT / L2 声学派生 / L3 模型化标注器 — 每层: 来源, 关键数字, 失效模式.
底部: 本文审计工具条. 调色板与项目其余图一致.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG = "/share/workspace3/zhuangruicen/proj-thumpy/docs/pilot_study/figures/fig1_label_taxonomy.png"
SURFACE = "#fcfcfb"
INK, INK2 = "#0b0b0b", "#52514e"
C_L1, C_L2, C_L3 = "#2a78d6", "#eb6834", "#1baf7a"
LIGHT = {"#2a78d6": "#e8f0fb", "#eb6834": "#fdeee7", "#1baf7a": "#e6f5ee"}


def band(ax, y, h, color, title, src, findings):
    ax.add_patch(FancyBboxPatch((0.02, y), 0.96, h,
                                boxstyle="round,pad=0.008,rounding_size=0.012",
                                facecolor=LIGHT[color], edgecolor=color,
                                linewidth=1.6))
    ax.text(0.05, y + h / 2, title, fontsize=15, fontweight="bold",
            color=color, va="center")
    ax.text(0.05, y + h - 0.055, src, fontsize=9.5, color=INK2, va="top")
    ax.text(0.05, y + 0.055, findings, fontsize=10, color=INK, va="bottom",
            linespacing=1.45)


def main():
    fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor=SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.965, "Where do turn-taking labels come from?\n"
            "Three annotation layers for full-duplex dialogue evaluation",
            ha="center", va="top", fontsize=15.5, fontweight="bold", color=INK)

    band(ax, 0.66, 0.24, C_L1, "L1 · Generation GT",
         "Source: dataset-generation metadata (Behavior-SD; FD-Bench GPT dialogues, manually revised)\n"
         "Claim: \"the behavior the generator intended\" — exact event timestamps, but only for synthetic data",
         "Audit: ARI 0.309 vs 0.058 on human data (Δ = −0.251, p < 0.001)\n"
         "Failure: 84% of the agreement is a silence shortcut (energy ratio, B)\n"
         "          overlap state is absent from token prosody (A1/A2); prosody = lexical templates (E)")

    band(ax, 0.38, 0.24, C_L2, "L2 · Acoustic-derived",
         "Source: VAD/ASR timestamps + hard threshold rules\n"
         "(FD-Bench: Silero VAD intervals, 0.5s/2.5s rules; CANDOR: AWS-Transcribe-derived labels)\n"
         "Claim: \"labels computed from the audio itself\" — works on real recordings",
         "Audit: human-data ARI 0.058, ceiling ≈0.07 after channel cleaning (D)\n"
         "Failure: pairwise κ between three toolchains only 0.06–0.37 (E1)\n"
         "          EIR drifts 25× across segment sources; 10% crosstalk collapses the metrics (E2)")

    band(ax, 0.10, 0.24, C_L3, "L3 · Model annotators",
         "Source: turn-detection models (X2-Turn: 80ms, 6 states; SoulX-Duplug: 160ms, 5 states)\n"
         "Claim: \"LLM-assisted labeling of real model outputs\" — calibrated on synthetic GT",
         "Audit: frame AUC 0.636 on real audio; window AUROC 0.667 tracks real overlap (E4)\n"
         "Caveat: label-space mapping is part of calibration (E3); fuse by union + probabilities (E5)")

    ax.add_patch(FancyBboxPatch((0.02, 0.015), 0.96, 0.055,
                                boxstyle="round,pad=0.004,rounding_size=0.008",
                                facecolor="#f0efec", edgecolor="#c3c2b7",
                                linewidth=1.2))
    ax.text(0.5, 0.042, "Audit toolkit: channel-level stereo features · ARI with balanced "
            "session-level bootstrap · shortcut ablation · cross-dataset transfer · "
            "crosstalk stress tests →  the no-shortcut evaluation protocol (Table 1)",
            ha="center", va="center", fontsize=9.5, color=INK2)

    fig.savefig(FIG, dpi=300, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("saved", FIG)


if __name__ == "__main__":
    main()
