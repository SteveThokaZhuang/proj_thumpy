"""F5: X2-Turn 跨域验证 (Fisher, 人工转写的电话对话).

Fisher: 11,699 段电话对话, NIST SPHERE 8kHz 双声道 (每声道一说话人),
LDC 人工转写 (utterance 级 `start end A|B: text`).
跨域金标准: 短 utterance (<=1.5s 的 yeah/uh-huh 类) 作为 BC 候选窗口
(人工转写出的真实附和 token, 比 AWS 派生标签干净).
指标: p_speak 帧 AUC vs utterance 隶属; p_bc 帧 AUC vs BC 候选窗口;
子串对齐 WER vs 人工转写.

注意: Fisher 是 8kHz (电话带宽), 上采样到 16k 喂模型 (域外+带宽域外).

用法 (gpu01, x2-turn 环境):
  python scripts/ari_f5_fisher.py --n-conversations 50 --out-dir .../f5_fisher
"""
import argparse
import glob
import json
import os
import random
import re
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import AutoProcessor

from voxtral_realtime.transformers.inference import infer_asr_turn
from voxtral_realtime.transformers.modeling import load_mtp_checkpoint

MODEL_DIR = "/share/workspace3/shared_models/X2-Turn-4B-0812"
FISHER = "/share/workspace3/shared_dataset/Fisher"
TMP = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator/tmp"
FFMPEG = "/share/home/zhuangruicen/miniconda3/envs/x2-turn/bin/ffmpeg"
CHUNK_S, OVERLAP_S = 180.0, 30.0


def list_conversations(n):
    wavs = []
    for d in sorted(os.listdir(FISHER)):
        p = os.path.join(FISHER, d)
        if not os.path.isdir(p) or d == "Missing" or d == "transcription":
            continue
        wavs += [os.path.join(p, f) for f in os.listdir(p) if f.endswith(".wav")]
    random.Random(42).shuffle(wavs)
    return wavs[:n]


def parse_transcript(fid):
    """LDC 格式: 头两行 # 注释, 之后 `start end A|B: text` -> {0: [...], 1: [...]}."""
    pat = f"{FISHER}/transcription/LDC2004T19/fe_03_p1_tran/data/trans/"
    # 子目录 = ID 数字前三位 (fe_03_00005 -> 000/)
    p = os.path.join(pat, fid[-5:-2], f"{fid}.txt")
    if not os.path.exists(p):
        return None
    out = {0: [], 1: []}
    for line in open(p, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"(\S+)\s+(\S+)\s+([AB]):\s*(.*)", line)
        if m:
            ch = 0 if m.group(3) == "A" else 1
            out[ch].append((float(m.group(1)), float(m.group(2)), m.group(4)))
    return out if (out[0] or out[1]) else None


def infer_channel(model, processor, wav):
    frames = []
    transcripts = []
    for t_chunk in range(0, max(int(sf.info(wav).duration) - 1, 1),
                         int(CHUNK_S - OVERLAP_S)):
        cw = f"{TMP}/f5_c.wav"
        subprocess.run([FFMPEG, "-v", "error", "-ss", f"{t_chunk}", "-i", wav,
                        "-t", f"{CHUNK_S}", "-c", "copy", cw, "-y"], check=True)
        r = infer_asr_turn(model, processor, cw)
        for f in r.turn_frames:
            if f.start_ms / 1000.0 < CHUNK_S - OVERLAP_S:
                frames.append((t_chunk + f.start_ms / 1000.0,
                               t_chunk + f.end_ms / 1000.0,
                               float(f.probabilities.get("backchannel", 0.0)),
                               float(f.probabilities.get("speaking", 0.0))))
        transcripts.append(r.transcript)
        os.remove(cw)
    return frames, " ".join(transcripts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-conversations", type=int, default=50)
    ap.add_argument("--out-dir", type=str,
                    default="/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
                            "real_data/results/annotator/f5_fisher")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)

    convs = list_conversations(args.n_conversations)
    print(f"conversations: {len(convs)}", flush=True)
    model = load_mtp_checkpoint(MODEL_DIR, device="cuda",
                                dtype=torch.bfloat16).eval()
    processor = AutoProcessor.from_pretrained(MODEL_DIR)

    for i, wav_path in enumerate(convs):
        fid = os.path.basename(wav_path)[:-4]
        out = f"{args.out_dir}/{fid}.json"
        if os.path.exists(out):
            print(f"[{i}] {fid} cached", flush=True)
            continue
        t0 = time.time()
        y, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        gt = parse_transcript(fid)
        if gt is None:
            print(f"[{i}] {fid} no transcript, skip", flush=True)
            continue
        ch_frames = {}
        ch_tr = {}
        for ch in (0, 1):
            x = torch.from_numpy(y[:, ch]).float().view(1, 1, -1)
            up = F.interpolate(x, scale_factor=16000 / sr, mode="linear",
                               align_corners=False).view(-1).numpy()
            ch16 = up.astype(np.float32)
            wav16 = f"{TMP}/f5_{fid}_ch{ch}.wav"
            sf.write(wav16, ch16, 16000)
            frames, tr = infer_channel(model, processor, wav16)
            ch_frames[str(ch)] = [[round(a, 3), round(b, 3), round(p, 4),
                                   round(s, 4)] for (a, b, p, s) in frames]
            ch_tr[str(ch)] = tr
            os.remove(wav16)
        d = {"fid": fid, "gt": gt, "frames": ch_frames, "transcript": ch_tr}
        json.dump(d, open(out, "w"))
        print(f"[{i}] {fid} done {time.time()-t0:.0f}s", flush=True)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
