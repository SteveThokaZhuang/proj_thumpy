"""E3: X2-Turn 在 Behavior-SD 上的金标准校准 (帧级 + 事件级).

对每个文件: 从 tar 取 flac -> ffmpeg 拆声道 (16k mono) -> 每声道 infer_asr_turn
-> 与 metadata GT 对齐 (utterances + 嵌套 backchannels 窗口).
保存 per-file JSON 供分析.

用法 (gpu02, x2-turn 环境, 单卡顺序):
  python scripts/ari_e3_calibrate.py --n-files 30 [--out-dir real_data/results/annotator/e3_raw]
"""
import argparse
import json
import os
import subprocess
import sys
import tarfile
import time

import torch
from transformers import AutoProcessor

from voxtral_realtime.transformers.inference import infer_asr_turn
from voxtral_realtime.transformers.modeling import load_mtp_checkpoint

MODEL_DIR = "/share/workspace3/shared_models/X2-Turn-4B-0812"
BEHAVIOR_ROOT = "/share/workspace3/shared_dataset/behavior-sd"
TMP = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator/tmp"
FFMPEG = "/share/home/zhuangruicen/miniconda3/envs/x2-turn/bin/ffmpeg"
DEFAULT_OUT = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator/e3_raw"


def run_ffmpeg(args):
    subprocess.run([FFMPEG, "-v", "error", *args], check=True)


def load_model():
    t0 = time.time()
    model = load_mtp_checkpoint(MODEL_DIR, device="cuda", dtype=torch.bfloat16)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    print(f"model loaded {time.time()-t0:.0f}s", flush=True)
    return model, processor


def infer_channel(model, processor, wav_path):
    result = infer_asr_turn(model, processor, wav_path)
    return {
        "transcript": result.transcript,
        "frames": [
            {"t0": f.start_ms / 1000.0, "t1": f.end_ms / 1000.0,
             "label": f.label, "conf": float(f.confidence),
             "p_bc": float(f.probabilities.get("backchannel", 0.0)),
             "p_speak": float(f.probabilities.get("speaking", 0.0))}
            for f in result.turn_frames
        ],
    }


def gt_from_metadata(rec):
    """GT: 每声道 (speaker_idx 即声道) 的 utterances 与 BC 窗口."""
    gt = {0: {"utterances": [], "bc": []}, 1: {"utterances": [], "bc": []}}
    for u in rec.get("utterances", []):
        ch = u["speaker_idx"]
        gt[ch]["utterances"].append([u["start_time"], u["end_time"]])
        for bc in u.get("backchannels", []):
            # BC 由听者发出 (对方声道)
            gt[1 - ch]["bc"].append([bc["start_time"], bc["end_time"],
                                     bc.get("tts_text", "")])
    for ch in (0, 1):
        gt[ch]["utterances"].sort()
        gt[ch]["bc"].sort()
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=30)
    ap.add_argument("--out-dir", type=str, default=DEFAULT_OUT)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)

    # 收集 validation 文件清单 (按 file_id 排序)
    files = []
    for tname in sorted(os.listdir(f"{BEHAVIOR_ROOT}/validation")):
        if not tname.endswith(".tar"):
            continue
        with tarfile.open(f"{BEHAVIOR_ROOT}/validation/{tname}") as tf:
            for n in sorted(tf.getnames()):
                if n.endswith(".flac"):
                    files.append((tname, n[:-5]))
    files = files[args.start: args.start + args.n_files]
    print(f"files: {len(files)}", flush=True)

    model, processor = load_model()

    for i, (tname, stem) in enumerate(files):
        fid = f"validation_{stem}"
        out_path = f"{args.out_dir}/{fid}.json"
        if os.path.exists(out_path):
            print(f"[{i}] {fid} cached", flush=True)
            continue
        t0 = time.time()
        flac = f"{TMP}/{stem}.flac"
        with tarfile.open(f"{BEHAVIOR_ROOT}/validation/{tname}") as tf:
            rec = json.loads(tf.extractfile(f"{stem}.json").read().decode())
            data = tf.extractfile(f"{stem}.flac").read()
        with open(flac, "wb") as f:
            f.write(data)
        wavs = {}
        for ch in (0, 1):
            wav = f"{TMP}/{stem}_ch{ch}.wav"
            run_ffmpeg(["-i", flac, "-af", f"pan=mono|c0=c{ch}",
                        "-ar", "16000", "-ac", "1", wav, "-y"])
            wavs[ch] = wav
        pred = {}
        for ch in (0, 1):
            pred[ch] = infer_channel(model, processor, wavs[ch])
        gt = gt_from_metadata(rec)
        out = {"file_id": fid, "split": "validation", "pred": pred, "gt": gt}
        with open(out_path, "w") as f:
            json.dump(out, f)
        os.remove(flac)
        for w in wavs.values():
            os.remove(w)
        print(f"[{i}] {fid} done {time.time()-t0:.0f}s "
              f"(frames {len(pred[0]['frames'])}/{len(pred[1]['frames'])}, "
              f"bc {len(gt[0]['bc'])}/{len(gt[1]['bc'])})", flush=True)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
