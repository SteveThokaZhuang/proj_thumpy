"""G1 观测空间消融: 数据准备 (参数化 chunk 构建).

动机: F8/F8c 的训练输入是 10s **单声道、仅说话人自己那一路**。而 BC 是跨说话人
事件 ("对方还在说 + 我插一句短的")——单声道输入里对方的语音只能靠麦克风回声
泄漏。本脚本构建不同观测空间的 chunk 集, **chunk id 与 f8_training 完全一致**
(同 session/ch/t0), 标签 (states/fused) 也逐位相同——唯一变量是音频内容。

input-mode:
  own     : 自己声道 (与 f8_training 相同, 用于校验)
  mix     : 两声道直接相加 (真实单麦视角, 对方处于串扰电平)
  mixnorm : 两声道各自 RMS 对齐后相加 (对方与本人等响, 信息上界)

用法 (fd_analysis 环境):
  python scripts/ari_g1_prep.py --input-mode mixnorm --out .../g1_mixnorm10
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_f8_prep_data import load_soulx  # noqa: E402

CANDOR = "/share/workspace3/shared_dataset/CANDOR/files"
ANNOT = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator"
FRAME_S = 0.08
W = 0.7
EPS = 1e-8


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def p95(x):
    """95 分位幅度: 对孤立瞬态稳健的响度代理 (RMS 会被偶发尖峰骗)."""
    return float(np.percentile(np.abs(x), 95))


def make_audio(own, part, mode):
    """按观测空间模式合成送入模型的单通道音频 (已是 16k).

    mixnorm 的三道保护 (v1 只有 RMS 对齐, 实测 31% 的 chunk 峰值 >1,
    PCM_16 落盘会削波, 且低 RMS 声道里的瞬态会被放大上百倍):
      a. 用 p95 而非 RMS 估响度, 并按 +30dB 封顶 -> 对方真静音时不放大底噪;
      b. 对方贡献的峰值不超过本人声道峰值;
      c. 整体 RMS 对齐本人后做峰值保护, 保证 |m| <= 0.99.
    """
    if mode == "own":
        return own
    if mode == "mix":
        return own + part
    if mode == "mixnorm":
        lo, lp = p95(own), p95(part)
        gain = min(lo / (lp + EPS), 30.0)
        d = gain * part
        pk_o = float(np.max(np.abs(own))) + EPS
        pk_d = float(np.max(np.abs(d)))
        if pk_d > pk_o:
            d = d * (pk_o / pk_d)
        m = own + d
        ro, rm = rms(own), rms(m)
        if rm > EPS:
            m = m * (ro / rm)
        pk = float(np.max(np.abs(m)))
        if pk > 0.99:
            m = m * (0.99 / pk)
        return m.astype(np.float32)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sessions", type=int, default=100)
    ap.add_argument("--chunk-s", type=float, default=10.0)
    ap.add_argument("--input-mode", choices=["own", "mix", "mixnorm"],
                    default="mixnorm")
    # 默认值原先指向已废弃的 g1_mixnorm10 (46G, 2026-09-12 已删)。这里改成当前
    # 正式产物目录: 脚本是**平铺**写入 --out 的, 所以调用方给的是 mode 级目录
    # (mixnorm -> g1_sub/mixnorm, own -> g1_own_eval/own), 不带 mode 子层。
    ap.add_argument("--out", type=str, default=f"{ANNOT}/g1_sub/mixnorm")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    x2_files = {os.path.basename(f)[:-5]: f
                for f in glob.glob(f"{ANNOT}/e4_raw/*.json")}
    mani = json.load(open(f"{ANNOT}/e5_regions_manifest.json"))
    soulx = load_soulx(mani)
    soulx_sessions = set(m["session"] for m in mani.values())

    # 增量写 manifest: 支持中断续跑 (srun 步骤可能被清理)
    mani_path = f"{args.out}/manifest.jsonl"
    done = set()
    if os.path.exists(mani_path):
        for line in open(mani_path):
            done.add(json.loads(line)["session"])
        print(f"resume: {len(done)} sessions already done", flush=True)
    mf = open(mani_path, "a")

    manifest = []
    n_chunks = 0
    for s in sorted(x2_files)[: args.n_sessions]:
        if s in done:
            continue
        d = json.load(open(x2_files[s]))
        mp3 = f"{CANDOR}/{s}/processed/{s}.mp3"
        y, sr = sf.read(mp3, dtype="float32", always_2d=True)
        total = y.shape[0] / sr
        chans = {ch: resample_poly(y[:, int(ch)], 1, 3).astype(np.float32)
                 for ch in ("0", "1")}
        for ch in ("0", "1"):
            other = "1" if ch == "0" else "0"
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
            step = args.chunk_s / 2
            t = 0.0
            while t + args.chunk_s <= total:
                i0, i1 = int(t * 16000), int((t + args.chunk_s) * 16000)
                fm = (t0a >= t) & (t0a < t + args.chunk_s)
                if fm.sum() >= 20:
                    audio = make_audio(chans[ch][i0:i1], chans[other][i0:i1],
                                       args.input_mode)
                    states = np.stack([
                        1 - p_sp[fm] - p_bc[fm],
                        p_sp[fm], p_bc[fm]], axis=1).astype(np.float32)
                    nid = f"{s[:8]}_ch{ch}_t{int(t)}"
                    np.savez_compressed(f"{args.out}/{nid}.npz",
                                        audio=audio, states=states,
                                        fused=fused[fm])
                    manifest.append({
                        "id": nid, "session": s, "ch": ch, "t0": t,
                        "n_frames": int(fm.sum()), "has_fused": has_fused,
                        "has_transcript": has_tr})
                    n_chunks += 1
                t += step
        for m in manifest:
            mf.write(json.dumps(m) + "\n")
        mf.flush()
        manifest.clear()
        print(f"{s[:8]}: chunks so far {n_chunks}", flush=True)

    mf.close()
    print(f"mode={args.input_mode} total chunks: {n_chunks} -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
