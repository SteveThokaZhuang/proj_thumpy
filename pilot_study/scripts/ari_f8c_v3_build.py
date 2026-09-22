"""F8c v3 数据: 软分数回归 — 输出 200ms x 50 的 p_bc 概率曲线.

标签 = X2-Turn p_bc 软分数直接按 200ms 块均值池化 (取 0.1 精度),
不再离散成事件/词 — 保留"标注器认为这里是 BC 的置信度曲线".
统一格式: 正负例都用 50 值曲线 (负例即全零), 天然编码 listen 决策.

用法 (fd_analysis): python scripts/ari_f8c_v3_build.py --n-samples 2400
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
OUT = f"{ANNOT}/f8c_v3_sft"
N_BLOCKS = 50
BLOCK_S = 0.2
INSTRUCTION = ("Output the backchannel probability for every 200-millisecond "
               "block of this segment, as 50 numbers from 0.0 to 1.0.")


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
        # 200ms 块均值池化
        curve = np.zeros(N_BLOCKS)
        for k in range(N_BLOCKS):
            a, b = m["t0"] + k * BLOCK_S, m["t0"] + (k + 1) * BLOCK_S
            fm = (t0a < b) & (t1a > a)
            if fm.any():
                curve[k] = p_bc[fm].mean()
        curve = np.round(curve, 1)
        has_signal = float(curve.max()) >= 0.2
        sub = e1[(e1["session"] == m["session"]) & (e1["cls"] == "BC")
                 & (e1["ch_event"] == int(m["ch"]))]
        has_fake = any(r["realized"] == "None"
                       for _, r in sub.iterrows()
                       if r["end"] > m["t0"] and r["start"] < m["t0"] + 10)
        entry = (m, curve)
        if has_signal:
            pos.append(entry)
        elif has_fake:
            hard_neg.append(entry)
        else:
            neg.append(entry)
        if (len(pos) >= args.n_samples * 3 // 4
                and len(hard_neg) + len(neg) >= args.n_samples // 4):
            break

    chosen = (pos[: args.n_samples * 3 // 4]
              + hard_neg[: args.n_samples // 8]
              + neg[: max(0, args.n_samples // 4 - len(hard_neg))])
    random.Random(42).shuffle(chosen)
    rows = []
    for m, curve in chosen:
        d = np.load(f"{ANNOT}/f8_training/{m['id']}.npz")
        wav = f"{wav_dir}/{m['id']}.wav"
        sf.write(wav, d["audio"], 16000)
        audio_field = json.dumps({
            "path": os.path.realpath(wav), "text": "",
            "token": "<|audio_pad|>" * 250,
            "ref_path": "", "ref_text": "",
        }, ensure_ascii=False, sort_keys=True)
        answer = " ".join(f"{v:.1f}" for v in curve)
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
    n_pos = sum(1 for r in rows if float(r["messages"][1]["content"].split()[0]) > 0
                or any(float(x) >= 0.2 for x in r["messages"][1]["content"].split()))
    print(f"samples: {len(rows)} (signal {n_pos}) -> {OUT}")


if __name__ == "__main__":
    main()
