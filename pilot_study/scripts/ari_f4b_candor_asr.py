"""F4b: CANDOR 侧 ASR 归因 — 转写 WER 与状态质量的相关性 (补齐真实域归因).

金标准侧 (Behavior-SD) 已证明 ASR 非瓶颈 (r=-0.27 n.s.); F4b 在真实域重复:
  对 80 个有 X2-Turn 转写的 CANDOR 会话:
    WER = X2-Turn transcript vs AWS audiophile 文本 (子串对齐, 参考本身带噪)
    状态质量 = 帧级 p_bc AUC vs AWS BC 窗口 (会话级)
  分析: Pearson r(WER, AUC) + WER 分布 + 会话数.

注意: CANDOR 参考文本是 AWS ASR (带噪), WER 实为"X2 vs AWS 的转写分歧",
下界估计真实 WER. 这正是"真实域归因不完整"缺口的一步.

用法 (fd_analysis): python scripts/ari_f4b_candor_asr.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_analyze import DEFAULT_OUT  # noqa: E402
from ari_wer_utils import wer  # noqa: E402
import ari_extract_candor as AC  # noqa: E402

ANNOT = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator"
CANDOR = "/share/workspace3/shared_dataset/CANDOR/files"


def main():
    ev = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_events_w*.csv"))])
    rows = []
    for fp in sorted(glob.glob(f"{ANNOT}/e4_raw/*.json")):
        s = os.path.basename(fp)[:-5]
        d = json.load(open(fp))
        has_tr = any(d["channels"][ch].get("transcript")
                     for ch in ("0", "1"))
        if not has_tr:
            continue
        # AWS 参考文本 (audiophile, 每声道)
        refs = {0: [], 1: []}
        try:
            for row in AC.read_transcript(
                    f"{CANDOR}/{s}/transcription/transcript_audiophile.csv"):
                ch = 0 if row.get("speaker") == json.load(open(
                    f"{CANDOR}/{s}/processed/channel_map.json")).get("L") else 1
                refs[ch].append(row.get("utterance", ""))
        except Exception:
            continue
        for ch in ("0", "1"):
            frames = d["channels"][ch]["frames"]
            if not frames:
                continue
            p_bc = np.array([f[3] for f in frames])
            t0 = np.array([f[0] for f in frames])
            t1 = np.array([f[1] for f in frames])
            sub = ev[(ev["session"] == s) & (ev["cls"] == "BC")
                     & (ev["ch_event"] == int(ch))]
            ins = np.zeros(len(t0), bool)
            for _, r in sub.iterrows():
                ins |= (t0 < r["end"]) & (t1 > r["start"])
            auc = roc_auc_score(ins, p_bc) if ins.any() and (~ins).any() else None
            w = wer(" ".join(refs[int(ch)]),
                    d["channels"][ch].get("transcript", ""))
            if auc is not None and w == w:
                rows.append({"session": s[:8], "ch": ch, "wer": w, "auc": auc})

    df = pd.DataFrame(rows)
    r, p = pearsonr(df["wer"], df["auc"])
    res = {
        "n_channels": len(df),
        "wer_median": round(float(df["wer"].median()), 4),
        "auc_mean": round(float(df["auc"].mean()), 4),
        "pearson_r": round(float(r), 4),
        "pearson_p": round(float(p), 4),
    }
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(f"{ANNOT}/f4b_results.json", "w"), indent=2)
    print("saved f4b_results.json")


if __name__ == "__main__":
    main()
