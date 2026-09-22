"""F8 数据准备: CANDOR-FD -> 训练样本 (模型无关格式).

每个声道切成 10s 块 (步进 5s), 每块:
  audio  : 16k mono float32 (soundfile 直读 mp3 -> resample_poly 拆声道, 已验证正确)
  states : 每 80ms 帧的三态软标签 [p_idle, p_speak, p_bc] (来自 X2-Turn 概率)
  fused  : SoulX 融合分 (仅 30 会话 SoulX 层可用处, F1 mean 配方)
  text   : X2-Turn 转写 (80 会话有; 训练时可按需用)
输出: 分块 npz + manifest.jsonl (会话/声道/块索引/时间窗/是否有转写).

用法 (fd_analysis 环境):
  python scripts/ari_f8_prep_data.py [--n-sessions 100] [--chunk-s 10]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_analyze import DEFAULT_OUT  # noqa: E402

CANDOR = "/share/workspace3/shared_dataset/CANDOR/files"
ANNOT = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator"
OUT = f"{ANNOT}/f8_training"
FRAME_S = 0.08
W = 0.7


def load_soulx(mani):
    """F1 同款累加版 soulx 帧 (伪概率)."""
    out = {}
    for key, meta in mani.items():
        s, ch = meta["session"], str(meta["ch"])
        p = f"{ANNOT}/f2_soulx_regions/{key}_states.json"
        if not os.path.exists(p):
            continue
        t0s, t1s, ps = [], [], []
        for st in json.load(open(p)):
            a = meta["t0"] + st["timestamp"][0]
            b = meta["t0"] + st["timestamp"][1]
            v = 1.0 if st["state"] == "backchannel" else 0.0
            t0s += [a, a + 0.08]
            t1s += [a + 0.08, b]
            ps += [v, v]
        buf = out.setdefault(s, {}).setdefault(ch, [[], [], []])
        buf[0] += t0s
        buf[1] += t1s
        buf[2] += ps
    for s in out:
        for ch in out[s]:
            out[s][ch] = tuple(np.array(x) for x in out[s][ch])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sessions", type=int, default=100)
    ap.add_argument("--chunk-s", type=float, default=10.0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    mani = json.load(open(f"{ANNOT}/e5_regions_manifest.json"))
    soulx = load_soulx(mani)
    soulx_sessions = set(m["session"] for m in mani.values())

    x2_files = {os.path.basename(f)[:-5]: f
                for f in glob.glob(f"{ANNOT}/e4_raw/*.json")}
    manifest = []
    n_chunks = 0
    for s in sorted(x2_files)[: args.n_sessions]:
        d = json.load(open(x2_files[s]))
        mp3 = f"{CANDOR}/{s}/processed/{s}.mp3"
        y, sr = sf.read(mp3, dtype="float32", always_2d=True)
        total = y.shape[0] / sr
        for ch in ("0", "1"):
            ch16 = resample_poly(y[:, int(ch)], 1, 3).astype(np.float32)
            fr = d["channels"][ch]["frames"]
            t0a = np.array([f[0] for f in fr])
            p_bc = np.array([f[3] for f in fr])
            p_sp = np.array([f[4] for f in fr])
            has_fused = s in soulx_sessions and ch in soulx.get(s, {})
            fused = np.zeros(len(t0a), dtype=np.float32)
            if has_fused:
                st0, st1, sp = soulx[s][ch]
                sct = (st0 + st1) / 2
                xct = (t0a + t0a + FRAME_S) / 2
                idx = np.abs(sct[:, None] - xct[None, :]).argmin(axis=0)
                fused = W * p_bc + (1 - W) * sp[idx]
            has_tr = bool(d["channels"][ch].get("transcript"))
            # 分块
            step = args.chunk_s / 2
            t = 0.0
            while t + args.chunk_s <= total:
                i0, i1 = int(t * 16000), int((t + args.chunk_s) * 16000)
                fm = (t0a >= t) & (t0a < t + args.chunk_s)
                if fm.sum() >= 20:
                    states = np.stack([
                        1 - p_sp[fm] - p_bc[fm],   # p_idle (近似, 未归一)
                        p_sp[fm], p_bc[fm]], axis=1).astype(np.float32)
                    nid = f"{s[:8]}_ch{ch}_t{int(t)}"
                    np.savez_compressed(
                        f"{OUT}/{nid}.npz", audio=ch16[i0:i1],
                        states=states, fused=fused[fm])
                    manifest.append({
                        "id": nid, "session": s, "ch": ch, "t0": t,
                        "n_frames": int(fm.sum()), "has_fused": has_fused,
                        "has_transcript": has_tr})
                    n_chunks += 1
                t += step
        print(f"{s[:8]}: chunks so far {n_chunks}", flush=True)

    with open(f"{OUT}/manifest.jsonl", "w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    print(f"total chunks: {n_chunks} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
