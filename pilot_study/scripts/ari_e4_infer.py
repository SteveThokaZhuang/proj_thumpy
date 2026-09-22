"""E4: X2-Turn 在 CANDOR 会话上的分块推理 (帧级概率).

每会话: ffmpeg pan 拆声道 (48k mp3 -> 16k mono wav) -> 3min 块 (30s 重叠)
-> infer_asr_turn 逐块 -> 去重叠拼接 -> 保存 per-session JSON.

注意: CANDOR 是 48k 立体声 mp3, 有串扰; 每声道 20-45 分钟音频.
显存安全: 3min 块 ~2250 帧 (E3 的 1000 帧用 12.3GB, 此规模 ~16GB).

用法 (gpu02, x2-turn 环境, 单卡顺序, 断点续跑):
  python scripts/ari_e4_infer.py --n-sessions 20
"""
import argparse
import glob
import json
import os
import subprocess
import time

import torch
from transformers import AutoProcessor

from voxtral_realtime.transformers.inference import infer_asr_turn
from voxtral_realtime.transformers.modeling import load_mtp_checkpoint

MODEL_DIR = "/share/workspace3/shared_models/X2-Turn-4B-0812"
CANDOR = "/share/workspace3/shared_dataset/CANDOR/files"
TMP = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator/tmp"
OUT = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator/e4_raw"
FFMPEG = "/share/home/zhuangruicen/miniconda3/envs/x2-turn/bin/ffmpeg"
CHUNK_S = 180.0
OVERLAP_S = 30.0


def run_ffmpeg(args):
    subprocess.run([FFMPEG, "-v", "error", *args], check=True)


def infer_audio(model, processor, wav):
    result = infer_asr_turn(model, processor, wav)
    frames = [(f.start_ms / 1000.0, f.end_ms / 1000.0, f.label,
               float(f.probabilities.get("backchannel", 0.0)),
               float(f.probabilities.get("speaking", 0.0)))
              for f in result.turn_frames]
    return frames, result.transcript


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sessions", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(TMP, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    sessions = sorted(d for d in os.listdir(CANDOR)
                      if len(d) == 36 and d[8] == "-")
    sessions = sessions[args.start: args.start + args.n_sessions]
    print(f"sessions: {len(sessions)}", flush=True)

    model, processor = load_mtp_checkpoint(
        MODEL_DIR, device="cuda", dtype=torch.bfloat16), AutoProcessor.from_pretrained(MODEL_DIR)
    model.eval()

    for i, s in enumerate(sessions):
        out_path = f"{OUT}/{s}.json"
        if os.path.exists(out_path):
            print(f"[{i}] {s[:8]} cached", flush=True)
            continue
        t0 = time.time()
        mp3 = f"{CANDOR}/{s}/processed/{s}.mp3"
        # 拆声道全量 wav
        ch_wavs = []
        for ch in (0, 1):
            w = f"{TMP}/e4_{s[:8]}_ch{ch}.wav"
            run_ffmpeg(["-i", mp3, "-af", f"pan=mono|c0=c{ch}",
                        "-ar", "16000", "-ac", "1", w, "-y"])
            ch_wavs.append(w)
        # 时长 (ffprobe 用 ffmpeg 查)
        import re
        dur_out = subprocess.run(
            [FFMPEG, "-i", ch_wavs[0]],
            capture_output=True, text=True).stderr
        m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", dur_out)
        total = (int(m.group(1)) * 3600 + int(m.group(2)) * 60
                 + float(m.group(3))) if m else 0.0
        # 分块推理
        ch_frames = {0: [], 1: []}
        ch_transcripts = {0: [], 1: []}
        t_chunk = 0.0
        while t_chunk < max(total - 1.0, 1.0):
            for ch in (0, 1):
                cw = f"{TMP}/e4_{s[:8]}_ch{ch}_c{int(t_chunk)}.wav"
                run_ffmpeg(["-ss", f"{t_chunk:.1f}", "-i", ch_wavs[ch],
                            "-t", f"{CHUNK_S}", "-c", "copy", cw, "-y"])
                frames, transcript = infer_audio(model, processor, cw)
                # 丢弃重叠区 (最后 OVERLAP_S 秒), 帧时间 = t_chunk + 局部时间
                keep = [(t0 + t_chunk, t1 + t_chunk, *rest)
                        for (t0, t1, *rest) in frames
                        if t0 < CHUNK_S - OVERLAP_S]
                ch_frames[ch].extend(keep)
                ch_transcripts[ch].append(transcript)
                os.remove(cw)
            t_chunk += CHUNK_S - OVERLAP_S
        out = {"session": s,
               "channels": {
                   "0": {"frames": [[round(t0, 3), round(t1, 3), lab,
                                     round(pb, 4), round(ps, 4)]
                                    for (t0, t1, lab, pb, ps) in ch_frames[0]],
                         "transcript": " ".join(ch_transcripts[0])},
                   "1": {"frames": [[round(t0, 3), round(t1, 3), lab,
                                     round(pb, 4), round(ps, 4)]
                                    for (t0, t1, lab, pb, ps) in ch_frames[1]],
                         "transcript": " ".join(ch_transcripts[1])},
               }}
        with open(out_path, "w") as f:
            json.dump(out, f)
        for w in ch_wavs:
            os.remove(w)
        print(f"[{i}] {s[:8]} done {time.time()-t0:.0f}s "
              f"frames {len(ch_frames[0])}/{len(ch_frames[1])} "
              f"dur={total/60:.1f}min", flush=True)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
