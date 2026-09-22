"""F3 人工验证工具包: 抽取 BC 窗口立体声片段 + 空白判定表.

抽样: 100 个窗口, 分层 (realized 真重叠 50 / 停顿内 50), 来自 CANDOR-FD 会话.
每片段: 窗口 ±3s 立体声 16k wav (L=声道0, R=声道1, 即一方说话人各一声道).
判定表: clip_id, session, 窗口起止, AWS BC 文本(backbiter), realized 标签
(隐藏列由脚本生成但标记 *_hidden, 防止先入为主), X2-Turn max p_bc (隐藏),
以及空白列: heard_bc / overlaps_host / notes.

用法 (fd_analysis): python scripts/ari_f3_prep_kit.py [--n 100]
"""
import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_analyze import DEFAULT_OUT  # noqa: E402
from ari_extract_candor import read_transcript, _f  # noqa: E402

ANNOT = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator"
CANDOR = "/share/workspace3/shared_dataset/CANDOR/files"
KIT = f"{ANNOT}/f3_kit"
MARGIN = 3.0


def bc_text(session, start):
    """从 backbiter 取 BC 文本 (会话级缓存)."""
    try:
        for row in read_transcript(
                f"{CANDOR}/{session}/transcription/transcript_backbiter.csv"):
            bs = _f(row.get("backchannel_start"))
            if bs == bs and abs(bs - start) < 0.01:
                return row.get("backchannel", "")
    except Exception:
        return ""
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(f"{KIT}/clips", exist_ok=True)

    ev = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_events_w*.csv"))])
    e1 = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_e1_w*.csv"))])
    e1 = e1.merge(ev[["session", "event_id", "cls", "ch_event", "start", "end"]],
                  on="event_id", suffixes=("", "_ev"))
    bc = e1[e1["cls"] == "BC"].copy()
    # 只取有 X2-Turn 标注的会话 (前 100 个)
    x2_sessions = {os.path.basename(f)[:-5]
                   for f in glob.glob(f"{ANNOT}/e4_raw/*.json")}
    bc = bc[bc["session"].isin(x2_sessions)]

    ov = bc[bc["realized"] != "None"]
    gap = bc[bc["realized"] == "None"]
    rng = np.random.RandomState(42)
    sel_ov = ov.sample(min(50, len(ov)), random_state=rng)
    sel_gap = gap.sample(min(50, len(gap)), random_state=rng)
    sel = pd.concat([sel_ov, sel_gap]).sample(frac=1, random_state=rng)

    rows = []
    for i, (_, r) in enumerate(sel.iterrows()):
        cid = f"clip_{i:03d}"
        mp3 = f"{CANDOR}/{r['session']}/processed/{r['session']}.mp3"
        y, sr = sf.read(mp3, dtype="float32", always_2d=True)
        a = max(0, int((r["start"] - MARGIN) * sr))
        b = min(y.shape[0], int((r["end"] + MARGIN) * sr))
        seg = resample_poly(y[a:b, :], 16000, sr).astype(np.float32)
        clip = f"{KIT}/clips/{cid}.wav"
        sf.write(clip, seg, 16000)
        rows.append({
            "clip_id": cid,
            "clip_path": clip,
            "session": r["session"][:8],
            "window_start_s": round(r["start"], 2),
            "window_end_s": round(r["end"], 2),
            "channel_listener": int(r["ch_event"]),
            "text_context": bc_text(r["session"], r["start"]),
            "realized_hidden": r["realized"],
            "heard_bc": "",
            "overlaps_host": "",
            "notes": "",
        })
    df = pd.DataFrame(rows)
    df.to_csv(f"{KIT}/judgments.csv", index=False)
    print(f"kit ready: {len(df)} clips, balanced ov/gap "
          f"({(df.realized_hidden != 'None').sum()}/{len(df)})")
    print("->", KIT)


if __name__ == "__main__":
    main()
