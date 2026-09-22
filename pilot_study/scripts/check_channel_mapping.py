#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
决定性检验: flac 双声道是否按说话人分离?

方法: 用 GT utterance 时间线把帧分成三类
  - spk0-only: GT 只有说话人 0 在说
  - spk1-only: GT 只有说话人 1 在说
  - both:      GT 两人同时 (理论为 0, 已证 Backchannel 无 GT 重叠)
对每类统计左/右声道 RMS 能量占比。若声道=说话人:
  spk0-only 能量几乎全在 L, spk1-only 几乎全在 R。

用法: python scripts/check_channel_mapping.py [validation|test]
"""
import io
import json
import sys
import tarfile
from pathlib import Path

import librosa
import numpy as np

SRC = Path("/share/workspace3/shared_dataset/behavior-sd")
REAL = Path(__file__).resolve().parents[1] / "real_data"
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "validation"
HOP = 0.01
FRAME = 0.025


def load(soda_id):
    for tar_path in sorted((SRC / SPLIT).glob("*.tar")):
        with tarfile.open(tar_path) as tar:
            if f"{soda_id}.flac" in tar.getnames():
                y, sr = librosa.load(
                    io.BytesIO(tar.extractfile(f"{soda_id}.flac").read()),
                    sr=None, mono=False)
                return y, sr
    return None, None


def main():
    meta_all = json.load(open(REAL / f"metadata_{SPLIT}.json"))
    print(f"[{SPLIT}] 检验 {len(meta_all)} 条", flush=True)

    # 聚合: E[k] = [spk0-only 时 L 能量, spk0-only 时 R 能量, spk1-only 时 L, spk1-only 时 R]
    agg = np.zeros(4)
    n_ok = 0
    rows = []
    for m in meta_all:
        y, sr = load(m["soda_id"])
        if y is None:
            continue
        L, R = y[0].astype(np.float32), y[1].astype(np.float32)
        rl = librosa.feature.rms(y=L, frame_length=int(FRAME * sr),
                                 hop_length=int(HOP * sr))[0]
        rr = librosa.feature.rms(y=R, frame_length=int(FRAME * sr),
                                 hop_length=int(HOP * sr))[0]
        n = len(rl)
        spk = [np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)]
        for u in m.get("utterances", []):
            i0, i1 = int(u["start"] / HOP), int(u["end"] / HOP)
            spk[u["speaker_idx"] % 2][i0:min(i1 + 1, n)] = True
        only0 = spk[0] & ~spk[1]
        only1 = spk[1] & ~spk[0]
        e0l, e0r = rl[only0].sum(), rr[only0].sum()
        e1l, e1r = rl[only1].sum(), rr[only1].sum()
        agg += [e0l, e0r, e1l, e1r]
        # 该样本判定: spk0 主导声道 / spk1 主导声道
        spk0_ch = "L" if e0l >= e0r else "R"
        spk1_ch = "L" if e1l >= e1r else "R"
        if spk0_ch != spk1_ch and e0l + e0r > 0 and e1l + e1r > 0:
            n_ok += 1
        rows.append((m["file_id"], round(float(e0l), 1), round(float(e0r), 1),
                     round(float(e1l), 1), round(float(e1r), 1), spk0_ch, spk1_ch))

    e0l, e0r, e1l, e1r = agg
    p0l = 100 * e0l / (e0l + e0r) if e0l + e0r else 0
    p1l = 100 * e1l / (e1l + e1r) if e1l + e1r else 0
    print(f"\n=== [{SPLIT}] 声道-说话人分离检验 (n={len(rows)}) ===")
    print(f"GT spk0-only 时段: L 能量占比 {p0l:.2f}%  (应≈100% 若 L=spk0)")
    print(f"GT spk1-only 时段: L 能量占比 {p1l:.2f}%  (应≈0%   若 L=spk0)")
    print(f"样本级: 两说话人主导声道不同 (可分离) 的样本: {n_ok}/{len(rows)}")
    print("\n前 15 条 (e0L, e0R, e1L, e1R, spk0主导, spk1主导):")
    for r in rows[:15]:
        print(f"  {r[0]}: {r[1]:>8} {r[2]:>8} | {r[3]:>8} {r[4]:>8} | {r[5]} {r[6]}")
    print("MAPPING DONE")


if __name__ == "__main__":
    main()
