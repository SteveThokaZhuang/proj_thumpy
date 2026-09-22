"""F8c v2 数据: 软置信事件 + 硬负例.

相对 v1 (ari_f8c_build_event_data.py) 的三处升级:
  1. τ=0.1 (更多事件, 覆盖率优先)
  2. 答案带置信度: "backchannel at 3.4s (0.85), 7.1s (0.62)"
  3. 硬负例: 有 AWS BC 窗口但 realized=None 的块 (标注器拒判/假 BC)
     -> "no backchannel" (教模型区分真重叠与停顿内 BC)

用法 (fd_analysis): python scripts/ari_f8c_v2_build.py --n-samples 2400
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_analyze import DEFAULT_OUT  # noqa: E402

ANNOT = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator"
OUT = f"{ANNOT}/f8c_v2_sft"
INSTRUCTION = ("When does the listener produce backchannels in this segment? "
               "List the times in seconds with confidence, or answer "
               "'no backchannel'.")
TAU = 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=2400)
    args = ap.parse_args()
    wav_dir = f"{OUT}/wavs"
    os.makedirs(wav_dir, exist_ok=True)

    ev = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_events_w*.csv"))])
    e1 = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_e1_w*.csv"))])
    e1 = e1.merge(ev[["session", "event_id", "cls", "ch_event", "start", "end"]],
                  on="event_id", suffixes=("", "_ev"))
    with open(f"{ANNOT}/f8_training/manifest.jsonl") as f:
        manifest = [json.loads(l) for l in f]
    random.Random(42).shuffle(manifest)

    x2_cache = {}
    pos, hard_neg, neg = [], [], []
    for m in manifest:
        # X2 帧
        if m["session"] not in x2_cache:
            try:
                d = json.load(open(f"{ANNOT}/e4_raw/{m['session']}.json"))
            except Exception:
                continue
            fr = d["channels"][str(m["ch"])]["frames"]
            x2_cache[m["session"]] = (
                np.array([f[0] for f in fr]),
                np.array([f[1] for f in fr]),
                np.array([f[3] for f in fr]))
        t0a, t1a, p_bc = x2_cache[m["session"]]
        fm = (t0a >= m["t0"]) & (t1a < m["t0"] + 10)
        if fm.sum() < 50:
            continue
        t = (t0a[fm] + t1a[fm]) / 2
        p = p_bc[fm]
        sub = e1[(e1["session"] == m["session"]) & (e1["cls"] == "BC")
                 & (e1["ch_event"] == int(m["ch"]))]
        wins = [(r["start"], r["end"], r["realized"]) for _, r in sub.iterrows()
                if r["end"] > m["t0"] and r["start"] < m["t0"] + 10]
        # 事件: 连续帧 p>=TAU, 峰值; 只保留落在 realized!=None 窗口内的
        events = []
        run = []
        for i in range(len(t)):
            if p[i] >= TAU:
                run.append(i)
            elif run:
                pk = run[int(np.argmax([p[j] for j in run]))]
                if any(a <= t[pk] <= b and rz != "None"
                       for a, b, rz in wins):
                    events.append((t[pk], float(p[pk])))
                run = []
        if run:
            pk = run[int(np.argmax([p[j] for j in run]))]
            if any(a <= t[pk] <= b and rz != "None" for a, b, rz in wins):
                events.append((t[pk], float(p[pk])))
        has_fake = any(rz == "None" for a, b, rz in wins)
        if events:
            pos.append((m, events))
        elif has_fake:
            hard_neg.append(m)
        else:
            neg.append(m)
        if (len(pos) >= args.n_samples * 3 // 4
                and len(hard_neg) + len(neg) >= args.n_samples // 4):
            break

    chosen = (pos[: args.n_samples * 3 // 4]
              + hard_neg[: args.n_samples // 8]
              + neg[: max(0, args.n_samples // 4 - len(hard_neg))])
    random.Random(42).shuffle(chosen)
    rows = []
    for item in chosen:
        if isinstance(item, tuple):
            m, evts = item
            answer = ", ".join(f"{e - m['t0']:.1f}s ({c:.2f})"
                               for e, c in evts)
        else:
            m = item
            answer = "no backchannel"
        d = np.load(f"{ANNOT}/f8_training/{m['id']}.npz")
        wav = f"{wav_dir}/{m['id']}.wav"
        sf.write(wav, d["audio"], 16000)
        audio_field = json.dumps({
            "path": os.path.realpath(wav), "text": "",
            "token": "<|audio_pad|>" * 250,
            "ref_path": "", "ref_text": "",
        }, ensure_ascii=False, sort_keys=True)
        rows.append({
            "system": "",
            "messages": [
                {"role": "user",
                 "content": "<|audio_bos|><|AUDIO|><|audio_eos|>" + INSTRUCTION},
                {"role": "assistant", "content": answer},
            ],
            "audio": audio_field,
        })
    with open(f"{OUT}/train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_pos = sum(1 for r in rows if r["messages"][1]["content"] != "no backchannel")
    print(f"samples: {len(rows)} (pos {n_pos}, hard_neg "
          f"{min(len(hard_neg), args.n_samples//8)}) -> {OUT}")


if __name__ == "__main__":
    main()
