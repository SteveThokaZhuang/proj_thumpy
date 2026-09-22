"""F3b 人工验证工具包 v2: 加上下文文本, 真值外置, 边距 5s.

相对 v1 (ari_f3_prep_kit.py) 的修正:
  1. **上下文文本**: v1 的 text_context 只是 BC token 自己的 ASR 文本
     ("Oh yeah. Good.") —— 看起来毫无上下文。v2 给出**双方**在窗口
     ±MARGIN 内的完整话语 (带相对时间), 来自 transcript_audiophile.csv。
  2. **真值外置**: v1 把 realized_hidden 直接写在判定表里 (只是列名带 hidden),
     填表的人能看见 —— 是泄题。v2 移到 truth.json, 判定表保持干净。
  3. **边距 3s -> 5s**: 给"对方是否还在说"留足上下文; 片段 ~10-13s。
  4. 增加 clip_target_at_s 列: 目标窗口在片段内的位置 (秒), 便于定位。
  5. 音量归一: 每片段按峰值归一到 -3dBFS, 避免不同会话响度差异干扰判断。

片段为**立体声**: 左=L=声道0, 右=R=声道1 (由 channel_map.json 决定哪个
参与者占哪个声道)。

用法 (fd_analysis): python scripts/ari_f3b_prep_kit.py [--n 100] [--margin 5.0]
"""
import argparse
import glob
import json
import os
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
TRANSCRIPT_CACHE = {}


def channel_users(session):
    """channel_map.json -> {0: user_id, 1: user_id}."""
    try:
        cm = json.load(open(f"{CANDOR}/{session}/processed/channel_map.json"))
        return {0: cm.get("L"), 1: cm.get("R")}
    except Exception:
        return {}


def utterances(session):
    """会话的全部话语 (audiophile 细粒度流): [(start, stop, speaker, text)]."""
    if session not in TRANSCRIPT_CACHE:
        out = []
        try:
            for row in read_transcript(
                    f"{CANDOR}/{session}/transcription/transcript_audiophile.csv"):
                s, e = _f(row.get("start")), _f(row.get("stop"))
                t = (row.get("utterance") or "").strip()
                if s != s or e != e or not t:
                    continue
                out.append((s, e, row.get("speaker", ""), t))
        except Exception:
            pass
        TRANSCRIPT_CACHE[session] = out
    return TRANSCRIPT_CACHE[session]


def side_text(session, ch_users, want_ch, a, b, t_ref, max_chars=700):
    """某一方在 [a,b] 内的话语, 带相对 t_ref 的时间前缀."""
    uid = ch_users.get(want_ch)
    parts = []
    for (s, e, spk, t) in utterances(session):
        if spk != uid or e < a or s > b:
            continue
        rel = s - t_ref
        parts.append(f"[{rel:+.1f}s] {t}")
    txt = " ".join(parts)
    return txt[:max_chars] + ("…" if len(txt) > max_chars else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--margin", type=float, default=5.0)
    args = ap.parse_args()
    M = args.margin
    os.makedirs(f"{KIT}/clips", exist_ok=True)

    ev = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_events_w*.csv"))])
    e1 = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_e1_w*.csv"))])
    e1 = e1.merge(ev[["session", "event_id", "cls", "ch_event", "start", "end"]],
                  on="event_id", suffixes=("", "_ev"))
    bc = e1[e1["cls"] == "BC"].copy()
    x2_sessions = {os.path.basename(f)[:-5]
                   for f in glob.glob(f"{ANNOT}/e4_raw/*.json")}
    bc = bc[bc["session"].isin(x2_sessions)]

    ov = bc[bc["realized"] != "None"]
    gap = bc[bc["realized"] == "None"]
    rng = np.random.RandomState(42)
    sel = pd.concat([ov.sample(min(50, len(ov)), random_state=rng),
                     gap.sample(min(50, len(gap)), random_state=rng)])
    sel = sel.sample(frac=1, random_state=rng).head(args.n)

    rows, truth = [], {}
    for i, (_, r) in enumerate(sel.iterrows()):
        cid = f"clip_{i:03d}"
        sess = r["session"]
        mp3 = f"{CANDOR}/{sess}/processed/{sess}.mp3"
        y, sr = sf.read(mp3, dtype="float32", always_2d=True)
        a = max(0, int((r["start"] - M) * sr))
        b = min(y.shape[0], int((r["end"] + M) * sr))
        seg = resample_poly(y[a:b, :], 16000, sr).astype(np.float32)
        peak = float(np.max(np.abs(seg)))
        if peak > 1e-6:
            seg = seg / peak * 0.7079          # -3 dBFS
        clip = f"{KIT}/clips/{cid}.wav"
        sf.write(clip, seg, 16000)

        ch_l = int(r["ch_event"])
        ch_users = channel_users(sess)
        a_txt, b_txt = r["start"] - M, r["end"] + M
        rows.append({
            "clip_id": cid,
            "clip_path": clip,
            "session": sess[:8],
            "window_start_s": round(r["start"], 2),
            "window_end_s": round(r["end"], 2),
            "clip_target_at_s": round(r["start"] - a / sr, 2),
            "channel_listener": ch_l,
            "partner_text": side_text(sess, ch_users, 1 - ch_l, a_txt, b_txt,
                                      r["start"]),
            "own_text": side_text(sess, ch_users, ch_l, a_txt, b_txt, r["start"]),
            "heard_bc": "",
            "overlaps_host": "",
            "notes": "",
        })
        truth[cid] = {
            "session": sess, "realized": r["realized"],
            "aws_window": [float(r["start"]), float(r["end"])],
            "listener_ch": ch_l,
        }
    pd.DataFrame(rows).to_csv(f"{KIT}/judgments.csv", index=False)
    json.dump(truth, open(f"{KIT}/truth.json", "w"), indent=1)

    # 自检: 上下文文本不应为空
    n_empty = sum(1 for r in rows if not r["partner_text"].strip())
    print(f"kit v2 ready: {len(rows)} clips, margin ±{M}s, "
          f"balanced ov/gap ({(sel.realized != 'None').sum()}/{len(sel)})")
    print(f"partner_text 为空的片段: {n_empty}  <- 应为 0")
    print("->", KIT)


if __name__ == "__main__":
    main()
