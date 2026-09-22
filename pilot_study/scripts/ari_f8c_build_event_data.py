"""F8c 事件式任务数据: 输出 BC 事件时间戳列表 (替代逐秒词序列).

F7 v0/v1 发现词序列任务"标签匹配≠行为匹配"; F8c 改为事件式输出:
  输入: 10s 对方声道音频 + 指令
  输出: "backchannel at 3.4s, 7.1s" 或 "no backchannel"
标签: X2-Turn p_bc 的事件 (连续帧 >= τ=0.2, 合并, 事件时间 = 峰值帧时间)
  只用 BC 邻域信号更纯: 事件须与 AWS BC 窗口相交 (标注器与 GT 共识).

用法 (fd_analysis 环境):
  python scripts/ari_f8c_build_event_data.py --n-samples 2400
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ANNOT = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator"
OUT = f"{ANNOT}/f8c_sft"
INSTRUCTION = ("When does the listener produce backchannels in this segment? "
               "List the times in seconds, or answer 'no backchannel'.")
TAU = 0.2


def chunk_bc_events(m, ev, x2_cache):
    """从 X2-Turn 帧取 chunk 内 BC 事件 (与 AWS BC 窗口共识)."""
    # X2 帧 (会话级缓存)
    if m["session"] not in x2_cache:
        d = json.load(open(f"{ANNOT}/e4_raw/{m['session']}.json"))
        fr = d["channels"][str(m["ch"])]["frames"]
        x2_cache[m["session"]] = {
            "t0": np.array([f[0] for f in fr]),
            "t1": np.array([f[1] for f in fr]),
            "p": np.array([f[3] for f in fr]),
        }
    fr = x2_cache[m["session"]]
    t0a, t1a, p_bc = fr["t0"], fr["t1"], fr["p"]
    fm = (t0a >= m["t0"]) & (t1a < m["t0"] + 10)
    if fm.sum() < 50:
        return []
    t = (t0a[fm] + t1a[fm]) / 2
    p = p_bc[fm]
    # AWS BC 窗口 (共识过滤, 全量只读一次)
    sub = ev[(ev["session"] == m["session"]) & (ev["cls"] == "BC")
             & (ev["ch_event"] == int(m["ch"]))]
    wins = [(r["start"], r["end"]) for _, r in sub.iterrows()
            if r["end"] > m["t0"] and r["start"] < m["t0"] + 10]
    # 事件: p_bc >= TAU 的连续帧, 峰值时间; 且峰值落在 AWS 窗口内
    events = []
    run = []
    for i in range(len(t)):
        if p[i] >= TAU:
            run.append(i)
        elif run:
            pk = run[int(np.argmax([p[j] for j in run]))]
            if any(a <= t[pk] <= b for a, b in wins):
                events.append(t[pk])
            run = []
    if run:
        pk = run[int(np.argmax([p[j] for j in run]))]
        if any(a <= t[pk] <= b for a, b in wins):
            events.append(t[pk])
    return [round(e - m["t0"], 2) for e in events]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=2400)
    ap.add_argument("--out", type=str, default=OUT)
    args = ap.parse_args()
    wav_dir = f"{args.out}/wavs"
    os.makedirs(wav_dir, exist_ok=True)

    import glob
    import pandas as pd
    from ari_analyze import DEFAULT_OUT
    ev = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_events_w*.csv"))])
    x2_cache = {}
    with open(f"{ANNOT}/f8_training/manifest.jsonl") as f:
        manifest = [json.loads(l) for l in f]
    random.Random(42).shuffle(manifest)

    pos, neg = [], []
    for m in manifest:
        evs = chunk_bc_events(m, ev, x2_cache)
        if evs:
            pos.append((m, evs))
        else:
            neg.append(m)
        if len(pos) >= args.n_samples * 3 // 4 and len(neg) >= args.n_samples // 4:
            break
    chosen = pos[: args.n_samples * 3 // 4] + neg[: args.n_samples // 4]
    random.Random(42).shuffle(chosen)

    rows = []
    for item in chosen:
        if isinstance(item, tuple):
            m, evs = item
            answer = ", ".join(f"{e:.1f}s" for e in evs)
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
    with open(f"{args.out}/train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_pos = sum(1 for r in rows if r["messages"][1]["content"] != "no backchannel")
    print(f"samples: {len(rows)} (pos {n_pos}, neg {len(rows)-n_pos}) -> {args.out}")


if __name__ == "__main__":
    main()
