"""E5 准备: 为 SoulX 生成 BC 窗口 ±5s 的声道区域 wav (16k mono).

SoulX 全会话推理太慢 (~1.9 块/s), E5 只需 BC 窗口邻域的双标注器对照.
用 soundfile + resample_poly (已验证对 CANDOR mp3 解码正确; ffmpeg 亦正确,
此前"频谱异常"系频率索引误算, 虚惊).

用法 (fd_analysis 环境): python scripts/ari_e5_prep.py --n-sessions 5
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

from ari_analyze import DEFAULT_OUT  # noqa: E402

CANDOR = "/share/workspace3/shared_dataset/CANDOR/files"
OUT = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator/e5_regions"
MARGIN = 5.0
MIN_GAP = 2.0   # 相邻窗口间隔小于此值则合并


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sessions", type=int, default=5)
    ap.add_argument("--out-dir", type=str, default=OUT)
    args = ap.parse_args()
    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    ev = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_events_w*.csv"))])
    sessions = sorted(ev["session"].unique())[: args.n_sessions]

    n_regions = 0
    for s in sessions:
        sub = ev[(ev["session"] == s) & (ev["cls"] == "BC")]
        mp3 = f"{CANDOR}/{s}/processed/{s}.mp3"
        y, sr = sf.read(mp3, dtype="float32", always_2d=True)  # (N,2) 48k
        total = y.shape[0] / sr
        for ch in (0, 1):
            wins = sorted(zip(sub[sub["ch_event"] == ch]["start"],
                              sub[sub["ch_event"] == ch]["end"]))
            if not wins:
                continue
            # 合并相邻窗口
            regions = []
            for (a, b) in wins:
                a0, b0 = max(0, a - MARGIN), min(total, b + MARGIN)
                if regions and a0 - regions[-1][1] < MIN_GAP:
                    regions[-1] = (regions[-1][0], b0)
                else:
                    regions.append((a0, b0))
            for ri, (a0, b0) in enumerate(regions):
                i0, i1 = int(a0 * sr), int(b0 * sr)
                seg = resample_poly(y[i0:i1, ch], 1, 3).astype(np.float32)
                wav = f"{out}/{s[:8]}_ch{ch}_r{ri}.wav"
                sf.write(wav, seg, 16000)
                n_regions += 1
        print(f"{s[:8]}: {len(sub)} BCs", flush=True)
    print(f"total regions: {n_regions}", flush=True)


if __name__ == "__main__":
    main()
