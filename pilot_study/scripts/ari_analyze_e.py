"""E: 合成 BC 韵律缺陷验证 — "应降调的肯定词"实际有多平.

主实验特征表显示合成 BC 的音高下降弱于人类 (-14 vs -61 Hz/s). E 验证这是
TTS 韵律控制缺陷而非文本选择差异:
  - 把 BC 事件与其文本对齐 (Behavior-SD: 元数据嵌套 backchannels 的 tts_text;
    CANDOR: backbiter 的 backchannel 列)
  - 相同文本 (yeah/sure/right/okay/mhm...) 在人类 vs 合成中的 f0_slope 对比
  - 这些肯定词语义上应降调 (falling), 平坦 = 韵律缺陷

输出: e_results.json + docs/pilot_study/figures/e_prosody_text.png

用法 (gpu02 持久步骤内):
  python scripts/ari_analyze_e.py
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_analyze import DEFAULT_OUT  # noqa: E402
import ari_extract_candor as AC  # noqa: E402 (read_transcript)

FIG_DIR = "/share/workspace3/zhuangruicen/proj-thumpy/docs/pilot_study/figures"
C_BLUE, C_ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, BASE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
META_DIR = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data"


def norm_text(t):
    t = re.sub(r"[^a-z0-9 ]", "", str(t).lower()).strip()
    return " ".join(t.split())


def load_behavior_texts():
    """(split, stem, bc_start) -> text 的字典."""
    out = {}
    for split in ["validation", "test", "train"]:
        meta = json.load(open(f"{META_DIR}/metadata_{split}.json"))
        for rec in meta:
            stem = rec["file_name"][:-5]  # 去 .flac
            for u in rec.get("utterances", []):
                for bc in u.get("backchannels", []):
                    out[(split, stem, round(bc["start_time"], 2))] = bc.get(
                        "tts_text", "")
    return out


def load_candor_texts(sessions):
    """(session, round(bc_start,2)) -> text."""
    out = {}
    for s in sessions:
        try:
            bb = AC.read_transcript(
                f"{AC.DATA_ROOT}/{s}/transcription/transcript_backbiter.csv")
        except Exception:
            continue
        for row in bb:
            bs = AC._f(row.get("backchannel_start"))
            if bs != bs:
                continue
            out[(s, round(bs, 2))] = row.get("backchannel", "")
    return out


def main():
    # Behavior-SD 事件 + 文本
    be = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/behavior_events_w*.csv"))],
                   ignore_index=True)
    bbc = be[be["cls"] == "BC"].copy()
    print(f"behavior BC events: {len(bbc)}", flush=True)
    btext = load_behavior_texts()
    bbc["split"] = bbc["file_id"].str.rsplit("_", n=1).str[0]
    bbc["stem"] = bbc["file_id"].str.rsplit("_", n=1).str[1]
    bbc["text"] = [btext.get((r.split, r.stem, round(r.start, 2)), "")
                   for r in bbc.itertuples()]
    bbc["ntext"] = bbc["text"].map(norm_text)
    bbc = bbc[bbc["ntext"] != ""]
    print(f"behavior BC with text: {len(bbc)}", flush=True)

    # CANDOR BC 事件 + 文本
    ce = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_events_w*.csv"))],
                   ignore_index=True)
    cbc = ce[ce["cls"] == "BC"].copy()
    sessions = sorted(cbc["session"].unique())
    print(f"candor BC events: {len(cbc)}, loading texts for {len(sessions)} sessions...",
          flush=True)
    ctext = load_candor_texts(sessions)
    cbc["text"] = [ctext.get((r.session, round(r.start, 2)), "")
                   for r in cbc.itertuples()]
    cbc["ntext"] = cbc["text"].map(norm_text)
    cbc = cbc[cbc["ntext"] != ""]
    print(f"candor BC with text: {len(cbc)}", flush=True)

    # 总体统计
    def stats(df, name):
        s = df["f0_slope"]
        flat = (s.abs() < 20).mean()
        falling = (s < -20).mean()
        rising = (s > 20).mean()
        print(f"{name}: n={len(df)} median_slope={s.median():.1f} "
              f"flat%={flat:.1%} falling%={falling:.1%} rising%={rising:.1%}",
              flush=True)
        return {"n": len(df), "median_slope": float(s.median()),
                "flat_frac": float(flat), "falling_frac": float(falling),
                "rising_frac": float(rising)}

    res = {"behavior": stats(bbc, "behavior"), "candor": stats(cbc, "candor")}

    # 相同文本对比 (两数据集都有且 n>=30 的 top 文本)
    bcounts = bbc["ntext"].value_counts()
    ccounts = cbc["ntext"].value_counts()
    shared = [t for t in bcounts.index
              if t in ccounts.index and bcounts[t] >= 30 and ccounts[t] >= 30]
    shared = shared[:10]
    per_text = {}
    for t in shared:
        per_text[t] = {
            "behavior": {"n": int(bcounts[t]),
                         "median_slope": float(bbc[bbc.ntext == t]["f0_slope"].median())},
            "candor": {"n": int(ccounts[t]),
                       "median_slope": float(cbc[cbc.ntext == t]["f0_slope"].median())},
        }
        print(f"  '{t}': behavior n={per_text[t]['behavior']['n']} "
              f"slope={per_text[t]['behavior']['median_slope']:.1f} | "
              f"candor n={per_text[t]['candor']['n']} "
              f"slope={per_text[t]['candor']['median_slope']:.1f}", flush=True)
    res["per_text"] = per_text

    with open(f"{DEFAULT_OUT}/e_results.json", "w") as f:
        json.dump(res, f, indent=2)

    # 图: 相同文本的人类 vs 合成 F0 斜率
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=9.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    x = np.arange(len(shared))
    w = 0.36
    ax.bar(x - w / 2, [per_text[t]["candor"]["median_slope"] for t in shared],
           w, color=C_BLUE, label="CANDOR (human)")
    ax.bar(x + w / 2, [per_text[t]["behavior"]["median_slope"] for t in shared],
           w, color=C_ORANGE, label="Behavior-SD (synthetic)")
    ax.axhline(0, color=BASE, linewidth=1)
    ax.set_xticks(x, shared, fontsize=10, color=INK2)
    ax.set_ylabel("median F0 slope (Hz/s, negative = falling)",
                  fontsize=10, color=INK)
    ax.set_title("Backchannel prosody by text: human vs synthetic",
                 fontsize=12, color=INK, fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/e_prosody_text.png", dpi=300,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved e_results.json + e_prosody_text.png")


if __name__ == "__main__":
    main()
