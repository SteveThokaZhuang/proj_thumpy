"""G1 消融: 用固定 (chunk id, 答案) 列表重建训练数据, 只换音频.

这是消融的**不变式保证**: 各观测空间臂共用同一份 id 列表与同一份答案,
唯一变量是 npz 里的音频。基线臂 = g1_own10_sft/train.jsonl (已逐条复现 v2,
并排除了评估集污染的 5 条)。

用法 (fd_analysis):
  python scripts/ari_g1_rebuild.py --src-jsonl .../g1_own10_sft/train.jsonl \
      --npz-dir .../g1_mixnorm10_sub --out .../g1_mixnorm10_sft
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

INSTRUCTION = ("When does the listener produce backchannels in this segment? "
               "List the times in seconds with confidence, or answer "
               "'no backchannel'.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-jsonl", required=True)
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-s", type=float, default=10.0)
    args = ap.parse_args()

    wav_dir = f"{args.out}/wavs"
    os.makedirs(wav_dir, exist_ok=True)
    rows, n_pos, peaks = [], 0, []
    for line in open(args.src_jsonl):
        r = json.loads(line)
        cid = os.path.basename(json.loads(r["audio"])["path"])[:-4]
        answer = r["messages"][1]["content"]
        if answer != "no backchannel":
            n_pos += 1
        d = np.load(f"{args.npz_dir}/{cid}.npz")
        audio = d["audio"]
        peaks.append(float(np.max(np.abs(audio))))
        wav = f"{wav_dir}/{cid}.wav"
        sf.write(wav, audio, 16000)
        audio_field = json.dumps({
            "path": os.path.realpath(wav), "text": "",
            "token": "<|audio_pad|>" * int(args.chunk_s * 25),
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
    pk = np.array(peaks)
    print(f"samples {len(rows)} (pos {n_pos}) -> {args.out}")
    print(f"峰值: max {pk.max():.4f}, 超 1 的比例 {(pk > 1).mean():.4f} "
          f"(应为 0.0000, 否则 PCM_16 会削波)")


if __name__ == "__main__":
    main()
