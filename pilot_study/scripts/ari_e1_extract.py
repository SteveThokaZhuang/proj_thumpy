"""E1 提取: CANDOR 事件的三套标签 (同一事件窗口, 三种工具链).

  L2-ASR  (cls)     : 已有 backbiter/audiophile 标签 (AWS Transcribe 派生)
  L2-VAD  (vad_*)   : Silero VAD (与 FD-Bench 同款工具, 同阈值 0.5) 窗口级
                      区间规则复刻: turn 起始 0.2s 内对方声道有语音 -> Int
  L3-realized       : 声道级 RMS 双活跃段 (绝对阈值 0.01, 与有声门控一致)
                      >0.5s -> Int, 0.1-0.5s -> Backchannel, 无 -> None

所有标签都在事件窗口上计算 (seek-read, 无需全量解码).
turn 事件 (cls=Int/None): vad_int (窗口起始 0.2s 内对方声道 VAD 活跃),
  vad_any (窗口内任何对方语音), realized (双活跃段规则).
BC 事件 (cls=BC): vad_seen (事件声道 VAD 语音占比≥0.5), host_active
  (宿主声道语音占比≥0.3), realized 同上.

用法 (gpu02 持久步骤内):
  python scripts/ari_e1_extract.py --worker W --n-workers 8 \
      --out real_data/results/analysis/ari/candor_e1_wW.csv
"""
import argparse
import csv
import os
import re
import sys
import time

import numpy as np
import soundfile as sf_lib
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_features import load_event_window, SR  # noqa: E402
import ari_extract_candor as AC  # noqa: E402

OUT_COLS = ["session", "event_id", "cls", "realized", "both_active_frac",
            "max_span", "vad_int", "vad_any", "vad_seen", "host_active"]

VAD_THRESHOLD = 0.5        # 与 FD-Bench 同款阈值
VAD_MIN_SPEECH_MS = 100
RMS_ACTIVE = 0.01          # 双活跃绝对阈值 (有声门控同款)
INT_LEAD = 0.2             # turn 起始判定窗 (s)


def load_vad():
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad",
                                  trust_repo=True)
    return model, utils[0]


def vad_segments(vad_model, get_ts, y_ch, threshold=VAD_THRESHOLD):
    """声道窗口上的 VAD 语音段 -> [(start_s, end_s)] (get_ts 返回样本数)."""
    if len(y_ch) < SR // 2:
        return []
    try:
        ts = get_ts(y_ch, vad_model, sampling_rate=SR, threshold=threshold,
                    min_speech_duration_ms=VAD_MIN_SPEECH_MS)
        return [(t["start"] / SR, t["end"] / SR) for t in ts]
    except Exception:
        return []


def realized_from_window(win):
    """win: (2, n) 16k 立体声窗口 -> (realized, frac, max_span_s)."""
    n = win.shape[1]
    win_l, hop = int(0.025 * SR), int(0.01 * SR)
    frames = []
    for i in range(0, n - win_l + 1, hop):
        r0 = np.sqrt(np.mean(win[0, i:i + win_l] ** 2))
        r1 = np.sqrt(np.mean(win[1, i:i + win_l] ** 2))
        frames.append(r0 > RMS_ACTIVE and r1 > RMS_ACTIVE)
    if not frames:
        return "None", 0.0, 0.0
    frames = np.array(frames)
    frac = float(frames.mean())
    spans, s, in_span = [], None, False
    for i, act in enumerate(frames):
        if act and not in_span:
            s = i * hop / SR
            in_span = True
        elif not act and in_span:
            spans.append(i * hop / SR - s)
            in_span = False
    if in_span:
        spans.append(len(frames) * hop / SR - s)
    mx = max(spans) if spans else 0.0
    if mx > 0.5:
        return "Int", frac, mx
    if mx > 0.1:
        return "Backchannel", frac, mx
    return "None", frac, mx


def process_session(session, events, writer, vad_model, get_ts):
    mp3 = f"{AC.DATA_ROOT}/{session}/processed/{session}.mp3"
    try:
        sf = sf_lib.SoundFile(mp3)
    except Exception:
        return {"session": session, "error": "open"}
    n_ok = 0
    for ev in events:
        win = load_event_window(sf, ev["start"], ev["end"])
        if win is None:
            continue
        ch_e, ch_o = ev["ch_event"], ev["ch_other"]
        rz, frac, mx = realized_from_window(win)
        row = {"session": session, "event_id": ev["event_id"],
               "cls": ev["cls"], "realized": rz,
               "both_active_frac": round(frac, 4),
               "max_span": round(mx, 4),
               "vad_int": "", "vad_any": "", "vad_seen": "", "host_active": ""}
        if ev["cls"] in ("Int", "None"):
            segs = vad_segments(vad_model, get_ts, win[ch_o])
            row["vad_int"] = int(any(s < INT_LEAD for s, e in segs))
            row["vad_any"] = int(len(segs) > 0)
        else:  # BC
            segs_e = vad_segments(vad_model, get_ts, win[ch_e])
            dur = ev["end"] - ev["start"]
            sp = sum(e - s for s, e in segs_e)
            row["vad_seen"] = int(sp / max(dur, 1e-9) >= 0.5)
            segs_h = vad_segments(vad_model, get_ts, win[ch_o])
            sp_h = sum(e - s for s, e in segs_h)
            row["host_active"] = int(sp_h / max(dur, 1e-9) >= 0.3)
        writer.writerow(row)
        n_ok += 1
    sf.close()
    return {"session": session, "n": n_ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--n-workers", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    # 事件 (来自主实验提取)
    import pandas as pd
    import glob
    from ari_analyze import DEFAULT_OUT
    ev = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(
                        f"{DEFAULT_OUT}/candor_events_w*.csv"))],
                   ignore_index=True)
    sessions = sorted(ev["session"].unique())
    mine = [s for i, s in enumerate(sessions) if i % args.n_workers == args.worker]

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        try:
            with open(args.out, newline="") as f:
                done = {r["session"] for r in csv.DictReader(f)}
        except Exception:
            pass
    mine = [s for s in mine if s not in done]
    new_file = (not os.path.exists(args.out)) or (not done)
    f = open(args.out, "a" if not new_file else "w", newline="")
    writer = csv.DictWriter(f, fieldnames=OUT_COLS)
    if new_file:
        writer.writeheader()

    print(f"[w{args.worker}] loading silero vad...", flush=True)
    vad_model, get_ts = load_vad()
    print(f"[w{args.worker}] sessions: {len(mine)}", flush=True)
    t0 = time.time()
    for j, s in enumerate(mine):
        sub = ev[ev["session"] == s].to_dict("records")
        res = process_session(s, sub, writer, vad_model, get_ts)
        f.flush()
        if j % 10 == 0:
            print(f"[w{args.worker}] {j}/{len(mine)} {res} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
    f.close()
    print(f"[w{args.worker}] DONE {len(mine)}", flush=True)


if __name__ == "__main__":
    main()
