"""F9 合成臂数据集: Behavior-SD metadata GT -> 同格式话轮状态序列.

与真实臂 (ari_f8_build_dataset.py) 完全同构:
  输入: 10s 对方声道音频 (16k mono) + 指令
  输出: 每 1s 一个状态词 (listen/speak/backchannel)
标签来源: metadata GT (嵌套 backchannels + utterances) — 生成端意图,
  与真实臂的 X2-Turn 标签形成对照 (F9 核心预言检验).

状态定义 (1s 块):
  backchannel : 块与听者声道的任一 GT BC 窗口重叠 >= 50%
  speak       : 块与该声道说话人自身 utterance 重叠 >= 50% (且非 bc)
  listen      : 其余

用法 (fd_analysis 环境):
  python scripts/ari_f9_build_synthetic.py --n-samples 3600 --out .../f9_sft_synthetic
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
INSTRUCTION = ("Predict the listener's turn-taking states (listen / speak / "
               "backchannel) over this segment, one word per second.")


def block_label(ch, t, gt_bc, gt_utt):
    """ch 声道在 [t, t+1) 的状态词 (GT 语义)."""
    a, b = t, t + 1.0

    def ov(p, q):
        return min(p[1], q[1]) - max(p[0], q[0])

    for (s, e) in gt_bc[ch]:
        if ov((a, b), (s, e)) >= 0.5:
            return "backchannel"
    for (s, e) in gt_utt[ch]:
        if ov((a, b), (s, e)) >= 0.5:
            return "speak"
    return "listen"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=3600)
    ap.add_argument("--out", type=str, default=f"{ANNOT}/f9_sft_synthetic")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    wav_dir = f"{args.out}/wavs"
    os.makedirs(wav_dir, exist_ok=True)
    rng = random.Random(args.seed)

    # 收集文件清单 (validation + test)
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

    rows = []
    seen = set()
    for (split, tname, stem) in files:
        with tarfile.open(f"{BEHAVIOR}/{split}/{tname}") as tf:
            rec = json.loads(tf.extractfile(f"{stem}.json").read().decode())
            data = tf.extractfile(f"{stem}.flac").read()
        y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        if y.ndim != 2:
            continue
        # GT 状态窗口
        gt_bc = {0: [], 1: []}
        gt_utt = {0: [], 1: []}
        for u in rec.get("utterances", []):
            ch = u["speaker_idx"]
            gt_utt[ch].append((u["start_time"], u["end_time"]))
            for bc in u.get("backchannels", []):
                gt_bc[1 - ch].append((bc["start_time"], bc["end_time"]))
        # 每声道 16k
        ch16 = {}
        for ch in (0, 1):
            ch16[ch] = resample_poly(y[:, ch], 16000, sr).astype(np.float32)
        # 分块 (10s, 5s 步进)
        dur = ch16[0].shape[0] / 16000
        t = 0.0
        while t + 10 <= dur:
            for ch in (0, 1):
                i0, i1 = int(t * 16000), int((t + 10) * 16000)
                words = [block_label(ch, t + k, gt_bc, gt_utt)
                         for k in range(10)]
                wid = f"{stem}_ch{ch}_t{int(t)}"
                if wid in seen:
                    continue
                seen.add(wid)
                wav = f"{wav_dir}/{wid}.wav"
                sf.write(wav, ch16[ch][i0:i1], 16000)
                n_pad = 10 * 25
                audio_field = json.dumps({
                    "path": os.path.realpath(wav), "text": "",
                    "token": "<|audio_pad|>" * n_pad,
                    "ref_path": "", "ref_text": "",
                }, ensure_ascii=False, sort_keys=True)
                rows.append({
                    "system": "",
                    "messages": [
                        {"role": "user",
                         "content": "<|audio_bos|><|AUDIO|><|audio_eos|>"
                                    + INSTRUCTION},
                        {"role": "assistant", "content": " ".join(words)},
                    ],
                    "audio": audio_field,
                })
            t += 5
            if len(rows) >= args.n_samples * 3:  # 冗余后平衡筛选
                break
        if len(rows) >= args.n_samples * 3:
            break

    # 类别平衡 (与真实臂同口径: 全 bc 块 + speak 到上限 + listen 补足)
    rng.shuffle(rows)
    cat = {"bc": [], "speak": [], "listen": []}
    for r in rows:
        words = r["messages"][1]["content"].split()
        if "backchannel" in words:
            cat["bc"].append(r)
        elif "speak" in words:
            cat["speak"].append(r)
        else:
            cat["listen"].append(r)
    n_bc = min(len(cat["bc"]), max(1, args.n_samples // 6))
    n_speak = min(len(cat["speak"]), max(1, args.n_samples // 2))
    n_listen = max(1, args.n_samples - n_bc - n_speak)
    chosen = cat["bc"][: n_bc] + cat["speak"][: n_speak] + cat["listen"][: n_listen]
    rng.shuffle(chosen)
    with open(f"{args.out}/train.jsonl", "w") as f:
        for r in chosen:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    cnt = Counter()
    for r in chosen:
        cnt.update(r["messages"][1]["content"].split())
    print(f"samples: {len(chosen)} (bc chunks {n_bc}, speak {n_speak}, "
          f"listen {n_listen}), words: {dict(cnt)}")
    print("dataset ->", args.out)


if __name__ == "__main__":
    main()
