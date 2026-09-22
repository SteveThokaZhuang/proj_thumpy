"""F9 事件式对照数据: 合成 metadata BC 窗口中点事件 (同 F8c 格式).

合成臂的事件标签 = Behavior-SD metadata GT 的嵌套 backchannels 窗口中点
(生成端意图, 精确时间戳) — 与真实臂 (X2-Turn 峰值事件) 形成事件级对照,
检验"真实事件标注 > 合成事件标注"的预言是否延续到事件式任务.

用法 (fd_analysis): python scripts/ari_f9_event_build.py --n-samples 2400
"""
import argparse
import io
import json
import os
import random
import sys
import tarfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BEHAVIOR = "/share/workspace3/shared_dataset/behavior-sd"
ANNOT = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator"
OUT = f"{ANNOT}/f9_event_synthetic"
INSTRUCTION = ("When does the listener produce backchannels in this segment? "
               "List the times in seconds with confidence, or answer "
               "'no backchannel'.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=2400)
    args = ap.parse_args()
    wav_dir = f"{OUT}/wavs"
    os.makedirs(wav_dir, exist_ok=True)
    rng = random.Random(42)

    files = []
    for split in ["validation", "test"]:
        for tname in sorted(os.listdir(f"{BEHAVIOR}/{split}")):
            if not tname.endswith(".tar"):
                continue
            with tarfile.open(f"{BEHAVIOR}/{split}/{tname}") as tf:
                for n in sorted(tf.getnames()):
                    if n.endswith(".flac"):
                        files.append((split, tname, n[:-5]))
    rng.shuffle(files)

    pos, neg = [], []
    seen = set()
    for (split, tname, stem) in files:
        with tarfile.open(f"{BEHAVIOR}/{split}/{tname}") as tf:
            rec = json.loads(tf.extractfile(f"{stem}.json").read().decode())
            data = tf.extractfile(f"{stem}.flac").read()
        y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        if y.ndim != 2:
            continue
        gt_bc = {0: [], 1: []}
        for u in rec.get("utterances", []):
            for bc in u.get("backchannels", []):
                gt_bc[1 - u["speaker_idx"]].append(
                    (bc["start_time"], bc["end_time"]))
        ch16 = {ch: resample_poly(y[:, ch], 16000, sr).astype(np.float32)
                for ch in (0, 1)}
        dur = ch16[0].shape[0] / 16000
        t = 0.0
        while t + 10 <= dur:
            for ch in (0, 1):
                wid = f"{stem}_ch{ch}_t{int(t)}"
                if wid in seen:
                    continue
                seen.add(wid)
                evts = [((a + b) / 2 - t, 1.0) for a, b in gt_bc[ch]
                        if b > t and a < t + 10]
                i0, i1 = int(t * 16000), int((t + 10) * 16000)
                wav = f"{wav_dir}/{wid}.wav"
                sf.write(wav, ch16[ch][i0:i1], 16000)
                answer = (", ".join(f"{e:.1f}s ({c:.2f})" for e, c in evts)
                          if evts else "no backchannel")
                audio_field = json.dumps({
                    "path": os.path.realpath(wav), "text": "",
                    "token": "<|audio_pad|>" * 250,
                    "ref_path": "", "ref_text": "",
                }, ensure_ascii=False, sort_keys=True)
                entry = {
                    "system": "",
                    "messages": [
                        {"role": "user",
                         "content": "<|audio_bos|><|AUDIO|><|audio_eos|>"
                                    + INSTRUCTION},
                        {"role": "assistant", "content": answer},
                    ],
                    "audio": audio_field,
                    "_wid": wid,
                }
                (pos if evts else neg).append(entry)
            t += 5
            if len(pos) >= args.n_samples * 3 // 4 and len(neg) >= args.n_samples // 4:
                break
        if len(pos) >= args.n_samples * 3 // 4 and len(neg) >= args.n_samples // 4:
            break

    chosen = pos[: args.n_samples * 3 // 4] + neg[: args.n_samples // 4]
    rng.shuffle(chosen)
    with open(f"{OUT}/train.jsonl", "w") as f:
        for r in chosen:
            r.pop("_wid")
            f.write(json.dumps(r) + "\n")
    print(f"samples: {len(chosen)} (pos {len(pos[:args.n_samples*3//4])}) -> {OUT}")


if __name__ == "__main__":
    main()
