"""duplexgen-spoken 的 `dialogue.wav` 里, 到底有多少时间是**双方同时出声**?

## 为什么问这个

数据集叫 **full-duplex**, 而全双工的核心恰恰是**重叠说话** (overlap)。
探针 (`duplexgen_bc_probe.py`) 已看到 `utterances/NN.wav` 的声道占用是**严格交替**的
(0=L, 1=R, 2=L, … 无例外) ⇒ **turn 层面根本没有重叠**。那么混音里的重叠只可能来自
backchannel (L 在 R 说话时插话)。

本脚本量就是这件事: 逐帧判每个声道"在不在出声", 统计四种状态各占多少时间,
再看"双方同时出声"的时段**是不是正好落在**该对话列出的 backchannel 上。

## 判据: 用 Otsu, 不用拍的常数

"在出声"要定阈值, 而拍一个常数正是本项目老账 `threshold-must-match-null-model`
拦的那件事。这里改用 **Otsu**: 在**该声道自己的**帧能量分布上找使类间方差最大的分割点,
**零个自由参数**。每个声道、每个对话各算各的 —— 因为不同 TTS 音色/录音的电平本来就不同,
跨对话共享一个常数会把"谁声音小"误判成"谁没说话"。

⚠️ 这是**探索性刻画**, 不是预登记实验: 只跑 6 个对话 (每场景第一个)。
   要当结论引用必须按功效配样本量, 见 §5.2-formal 设计草图。

用法 (fd_analysis, 0 GPU):
  python duplexgen_overlap_probe.py --tar <tar> --member <目录>
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile
import wave

import numpy as np

SR = 24000
FRAME_MS, HOP_MS = 25.0, 10.0


def read_wav(path):
    with wave.open(path) as w:
        n, ch = w.getnframes(), w.getnchannels()
        raw = w.readframes(n)
    d = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    return d.reshape(-1, ch)


def frame_rms(x, sr=SR, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    fl, hl = int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)
    if len(x) < fl:
        return np.zeros(0)
    n = 1 + (len(x) - fl) // hl
    idx = np.arange(fl)[None, :] + hl * np.arange(n)[:, None]
    return np.sqrt((x[idx] ** 2).mean(axis=1))


def otsu(v, nbins=256):
    """Otsu 阈值: 使分两类的**类间方差**最大的那个分割点。零自由参数。

    在 log 域上做 —— 能量是重尾的, 线性分箱会把阈值全挤到低端。
    """
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < 2:
        return 0.0
    lv = np.log(v)
    hist, edges = np.histogram(lv, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2
    w = hist.astype(np.float64)
    tot = w.sum()
    if tot <= 0:
        return float(np.exp(np.median(lv)))
    p = w / tot
    omega = np.cumsum(p)                      # 类 0 的累积权重
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 1e-12, (mu_t * omega - mu) ** 2 / denom, 0.0)
    k = int(np.argmax(sigma_b))
    return float(np.exp(centers[k]))


def probe(tar, member, workdir=None):
    tmp = workdir or tempfile.mkdtemp(prefix="dgov_")
    made = workdir is None
    try:
        subprocess.run(["tar", "-xf", tar, member], cwd=tmp, check=True,
                       stderr=subprocess.DEVNULL)
        root = os.path.join(tmp, member)
        dlg = read_wav(os.path.join(root, "dialogues", "dialogue.wav"))
        L, R = dlg[:, 0], dlg[:, 1]
        rl, rr = frame_rms(L), frame_rms(R)
        tl, tr = otsu(rl), otsu(rr)
        al, ar = rl > tl, rr > tr              # 逐帧: 该声道在不在出声
        n = min(len(al), len(ar))
        al, ar = al[:n], ar[:n]
        both = al & ar
        dur = n * HOP_MS / 1000
        print(f"dialogue {len(dlg)/SR:.1f}s   帧数 {n}")
        print(f"  Otsu 阈值  L={tl:.2e}  R={tr:.2e}")
        print(f"  各声道出声占比  L={al.mean():6.1%}  R={ar.mean():6.1%}")
        print(f"  ┌ 双方同时出声  {both.mean():6.2%}  ({both.sum()*HOP_MS/1000:.2f}s)")
        print(f"  ├ 只 L 出声      {(al & ~ar).mean():6.2%}")
        print(f"  ├ 只 R 出声      {(~al & ar).mean():6.2%}")
        print(f"  └ 都不出声      {(~al & ~ar).mean():6.2%}")

        # 双方同时出声的时段, 是不是正好是 backchannel?
        bcs = sorted(glob.glob(os.path.join(root, "backchannels", "*.wav")))
        spans = []
        i = 0
        while i < n:
            if both[i]:
                j = i
                while j < n and both[j]:
                    j += 1
                if (j - i) * HOP_MS / 1000 >= 0.06:      # 短于 60ms 的不算段
                    spans.append((i * HOP_MS / 1000, j * HOP_MS / 1000))
                i = j
            else:
                i += 1
        bc_dur = sum(len(read_wav(p)) for p in bcs) / SR
        ov_dur = sum(b - a for a, b in spans)
        print(f"\n  backchannel {len(bcs)} 个 / 合计 {bc_dur:.2f}s")
        print(f"  双方同时出声的段 {len(spans)} 个 / 合计 {ov_dur:.2f}s"
              f"   比值 {ov_dur/max(bc_dur,1e-9):.2f}")
        if spans:
            s = "  ".join(f"[{a:.1f},{b:.1f}]" for a, b in spans[:8])
            print(f"  前几段: {s}")
    finally:
        if made:
            subprocess.run(["rm", "-rf", tmp], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--member", required=True)
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()
    probe(a.tar, a.member, a.workdir)


if __name__ == "__main__":
    main()
