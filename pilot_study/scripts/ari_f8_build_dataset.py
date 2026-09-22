"""F8 数据集构建: f8_training chunks -> LLaMA-Factory sharegpt JSONL.

任务 (B 变体, 话轮状态 token SFT):
  输入: 10s 对方声道音频 (说话人的语音上下文) + 指令
  输出: 每 1s 一块的话轮状态词序列 (listen/speak/backchannel)
        状态来自 X2-Turn 80ms 软标签的 1s 块 argmax

sharegpt 格式 (与 Fun-Audio-Chat 官方数据集一致, audios 列 + AUDIO 占位符).

用法 (fd_analysis 环境):
  python scripts/ari_f8_build_dataset.py --n-samples 3600 [--out .../f8_sft]
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
OUT = f"{ANNOT}/f8_sft"
STATE_WORDS = ["listen", "speak", "backchannel"]
INSTRUCTION = ("Predict the listener's turn-taking states (listen / speak / "
               "backchannel) over this segment, one word per second.")


def states_to_words(states, n_sec, bc_tau=0.2, speak_tau=0.5):
    """states: (n_frames, 3) 软标签 [p_idle, p_speak, p_bc] -> 1s 块状态词.

    阈值离散化 (E3/E4 校准: p_bc 通常 < 0.5, 用低阈值; 优先级 bc > speak > listen).
    """
    words = []
    frames_per_sec = round(1.0 / 0.08)
    for i in range(n_sec):
        blk = states[i * frames_per_sec: (i + 1) * frames_per_sec]
        if len(blk) == 0:
            words.append("listen")
            continue
        m = blk.mean(axis=0)
        if m[2] >= bc_tau:
            words.append("backchannel")
        elif m[1] >= speak_tau:
            words.append("speak")
        else:
            words.append("listen")
    return words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=3600)
    ap.add_argument("--out", type=str, default=OUT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    wav_dir = f"{args.out}/wavs"
    os.makedirs(wav_dir, exist_ok=True)

    rng = random.Random(args.seed)
    with open(f"{ANNOT}/f8_training/manifest.jsonl") as f:
        manifest = [json.loads(l) for l in f]
    rng.shuffle(manifest)

    # 类别平衡: 按块的状态词构成分层选择
    # (v1 全随机导致 94% listen 主导, 训练坍缩到多数类)
    cat = {"bc": [], "speak": [], "listen": []}
    for m in manifest:
        npz = f"{ANNOT}/f8_training/{m['id']}.npz"
        d = np.load(npz)
        words = states_to_words(d["states"], int(len(d["audio"]) / 16000))
        if any(w == "backchannel" for w in words):
            cat["bc"].append(m)
        elif any(w == "speak" for w in words):
            cat["speak"].append(m)
        else:
            cat["listen"].append(m)
    n_bc = min(len(cat["bc"]), max(1, args.n_samples // 6))
    n_speak = min(len(cat["speak"]), max(1, args.n_samples // 2))
    n_listen = max(1, args.n_samples - n_bc - n_speak)
    chosen = (cat["bc"][: n_bc] + cat["speak"][: n_speak]
              + cat["listen"][: n_listen])
    rng.shuffle(chosen)

    rows = []
    for m in chosen:
        npz = f"{ANNOT}/f8_training/{m['id']}.npz"
        d = np.load(npz)
        audio = d["audio"]
        states = d["states"]
        n_sec = int(len(audio) / 16000)
        words = states_to_words(states, n_sec)
        wav = f"{wav_dir}/{m['id']}.wav"
        sf.write(wav, audio, 16000)
        # 输入音频: path=真实 wav, token=pad 占位 (TOKEN_FPS=25/s), 官方格式
        n_pad = int(len(audio) / 16000 * 25)
        audio_field = json.dumps({
            "path": os.path.realpath(wav), "text": "",
            "token": "<|audio_pad|>" * n_pad,
            "ref_path": "", "ref_text": "",
        }, ensure_ascii=False, sort_keys=True)
        rows.append({
            "system": "",
            "messages": [
                {"role": "user",
                 "content": "<|audio_bos|><|AUDIO|><|audio_eos|>" + INSTRUCTION},
                {"role": "assistant", "content": " ".join(words)},
            ],
            "audio": audio_field,
        })
    with open(f"{args.out}/train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # 状态词分布统计
    from collections import Counter
    cnt = Counter()
    for r in rows:
        cnt.update(r["messages"][1]["content"].split())
    print("samples:", len(rows), "state words:", dict(cnt))
    print("dataset ->", args.out)


if __name__ == "__main__":
    main()
