"""可行性探针: duplexgen-spoken 里列出的 backchannel, 在混音后**真的听得见**吗?

⚠️ 这是**仪器可行性检查**, 不是 §5.2 那个实验。只在**单个样本**上跑, 目的是回答
   「这套判据在这个数据上能不能用」, 不产出任何结论数字。§5.2 的正式设计见
   docs/pilot_study/2026-09-23_duplexgen_humdial_recon.md §5.2 及其预登记。

## 为什么用互相关而不是做减法

`dialogue.wav` 是双声道、两声道相关 ≈ 0 (实测 +0.0000) ⇒ 两个说话人各占一个声道。
但**不能**用 `dialogue − Σ utterances` 提取 backchannel: utterances 在对话里的
**时间位置**未知, 顺序累加 (从 t=0) 会让残差能量爆到 backchannel 总能量的 ~190 倍
(实测 78480 vs 408) —— 减法是**错位**的, 不是 backchannel 不存在。

互相关不需要知道对齐: 把每个 `backchannels/NN_K.wav` 拿去和 `dialogue.wav` 的
每个声道做**归一化**互相关, 混进去了就会有显著峰。附带好处: 峰在哪个声道,
就说明它 pan 到了谁那边 ⇒ 「谁的 backchannel」也是可测的, 不用信命名。

## 零模型 (跑正式实验前必须补)

本脚本只打印**观测**峰, 不设阈值。阈值必须先对零模型 —— 本项目老账
`threshold-must-match-null-model`: 拍一个数当判据会出假警报或假发现。
正式实验的零分布取法是: 把 backchannel 片段与**别的对话**的声道做同样的互相关
(或与同一声道的时间平移版本), 得到「没混进去时的峰分布」, 判据线从中取分位。

用法 (fd_analysis, 纯读盘, 0 GPU):
  python duplexgen_bc_probe.py --tar <tar 路径> --member <tar 内对话目录>
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np
from scipy.signal import fftconvolve

SR_EXPECT = 24000


def read_wav(path):
    """读 16-bit PCM wav -> (samples, n_channels)。用标准库, 不引入 soundfile。"""
    with wave.open(path) as w:
        n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
        raw = w.readframes(n)
    d = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    return d.reshape(-1, ch), sr


def ncc_peak(short, long_, floor_frac=1e-3):
    """短片段 short 在长信号 long_ 上的**归一化**互相关峰值与位置。

    归一化很要紧: 不归一化的话, 长信号里能量大的段落会无条件胜出,
    峰高就只是在读音量而不是读"像不像"。

    ⚠️ 两处必须防的数值坑 (第一版都踩了, 全是 nan):
      1. `fftconvolve(long_**2, ones)` 在**静音**段落真值是 0, 但 FFT 舍入会给出
         极小**负数** ⇒ sqrt 出 nan ⇒ 整个 ncc 被 nan 污染。
         ⇒ 先 `maximum(ev, 0)`。
      2. 就算夹住了, 分母近零的位置仍会把比值放大成无意义的巨值 (而且会抢走 argmax)。
         ⇒ 只保留"窗口能量不低于全段最大窗口能量的 floor_frac"的位置 ——
         在静音处谈"像不像"本来就没有意义。
    """
    c = fftconvolve(long_, short[::-1], mode="valid")
    ev = fftconvolve(long_ ** 2, np.ones(len(short)), mode="valid")
    ev = np.maximum(ev, 0.0)
    ey = np.sqrt(ev)
    ex = np.sqrt(float((short ** 2).sum()))
    if ex <= 0:
        return 0.0, 0, 0
    ok = ey > floor_frac * float(ey.max()) if ey.size and ey.max() > 0 \
        else np.zeros_like(ey, dtype=bool)
    ncc = np.zeros_like(c)
    ncc[ok] = np.abs(c[ok]) / (ey[ok] * ex)
    k = int(np.argmax(ncc))
    # 顺带给出**同一片段自己**的有效 ncc 分布。这不是判据线 —— 判据线必须对
    # 「没混进去」的零模型取 (老账 `threshold-must-match-null-model`)。它只是让
    # 峰值可读: 峰若与中位数同量级, 那就什么都不是。
    vals = ncc[ok]
    med = float(np.median(vals)) if vals.size else 0.0
    p999 = float(np.quantile(vals, 0.999)) if vals.size else 0.0
    return float(ncc[k]), k, int(ok.sum()), med, p999


def probe(tar, member, workdir=None):
    tmp = workdir or tempfile.mkdtemp(prefix="dgprobe_")
    made = workdir is None
    try:
        subprocess.run(["tar", "-xf", tar, member], cwd=tmp, check=True,
                       stderr=subprocess.DEVNULL)
        root = os.path.join(tmp, member)
        dlg, sr = read_wav(os.path.join(root, "dialogues", "dialogue.wav"))
        if sr != SR_EXPECT:
            print(f"  ⚠️ 采样率 {sr} ≠ 预期 {SR_EXPECT}, 下游按实际值处理")
        print(f"dialogue {dlg.shape[0] / sr:.2f}s @ {sr}Hz  声道数={dlg.shape[1]}")
        print(f"  L/R 相关 = {np.corrcoef(dlg[:, 0], dlg[:, 1])[0, 1]:+.4f}"
              f"   (≈0 ⇒ 两声道是不同说话人)")

        bcs = sorted(glob.glob(os.path.join(root, "backchannels", "*.wav")))
        utts = sorted(glob.glob(os.path.join(root, "utterances", "*.wav")))
        print(f"  utterances {len(utts)} 个 / backchannels {len(bcs)} 个\n")
        hdr = (f"{'片段':>10s} {'时长s':>7s} | {'L峰':>7s} {'L@t':>7s} | "
               f"{'R峰':>7s} {'R@t':>7s} | {'self中位':>8s} {'selfp99.9':>9s} | 判")
        print(hdr)
        print("-" * len(hdr))
        for p in bcs:
            x0, _ = read_wav(p)
            x = x0[:, 0]                      # backchannel 是单声道 (实测)
            out = [ncc_peak(x, dlg[:, ch]) for ch in range(dlg.shape[1])]
            best = max(range(len(out)), key=lambda i: out[i][0])
            side = "LR"[best]
            print(f"{os.path.basename(p):>10s} {len(x) / sr:7.3f} | "
                  f"{out[0][0]:7.4f} {out[0][1] / sr:7.2f} | "
                  f"{out[1][0]:7.4f} {out[1][1] / sr:7.2f} | "
                  f"{out[best][3]:8.4f} {out[best][4]:9.4f} | 峰在 {side}")
        print("\n⚠️ 上面只有**观测**峰, **没有判据线**。self中位/selfp99.9 是同一片段"
              "\n   在自己有效窗口上的分布, 只用来判断峰是不是「什么都不是」——"
              "\n   真正的判据线必须对**没混进去**的零模型取, 见 §5.2 预登记。")
    finally:
        if made:
            subprocess.run(["rm", "-rf", tmp], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--member", required=True,
                    help="tar 内的对话目录, 如 INT/work_0000/var00")
    ap.add_argument("--workdir", default=None,
                    help="解包去处 (默认建临时目录并在结束时删除)")
    a = ap.parse_args()
    probe(a.tar, a.member, a.workdir)


if __name__ == "__main__":
    main()
