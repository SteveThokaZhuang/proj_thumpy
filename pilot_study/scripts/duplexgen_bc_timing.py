"""backchannel 落在**哪儿**? 落在对方说话的**时候** (真重叠), 还是落在**空档**里?

## 为什么必须另写一个, 而不是信 `duplexgen_overlap_probe.py`

那个探针用**每个声道自己的 Otsu 阈值**判"在不在出声"。实测两声道阈值差 5 倍
(INT: L=3.99e-03 vs R=2.10e-02), 而 backchannel 是短的轻声「嗯哼」——
**很可能顶不过 L 自己的阈值** (L 的分布被长轮次主导)。
⇒ 那个探针量到的重叠是**下界**, 不能读成"数据集没有重叠"。
两个声道的阈值不对称这件事本身也说明: 全局单阈值不适合这个量。

## 本脚本的判据: 不设阈值

1. BC 的**时刻**由 NCC 峰给出 —— 峰=1.0000 已实测 (对齐是样本级的),
   所以时刻是**测出来的**, 不是从命名或元数据推的。
2. BC 在混音里的**增益** g = <窗口, bc> / <bc, bc>。NCC 是尺度不变的,
   所以它找得到位置**不代表音量正常** —— 一个混在 −30 dB 的 backchannel
   照样给出峰 1.0000。**增益必须单独测**, 这正是那个探针没做的。
3. 对方声道此刻活不活跃: 给出该窗口能量在**那个声道自己**的帧能量分布里的**分位数**。
   分位数不需要阈值就能读: 90 分位 = 在说话, 10 分位 = 安静。
   (老账 `threshold-must-match-null-model`: 拍常数会出假警报/假发现; 分位数不是常数。)

⚠️ 仍是**探索性**: 6 个对话 (每场景第一条)。要当结论引用须按功效配样本量。

用法 (fd_analysis, 0 GPU):
  python duplexgen_bc_timing.py --tar <tar> --member <目录>
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from duplexgen_bc_probe import read_wav, ncc_peak, SR_EXPECT  # noqa: E402

FRAME_MS, HOP_MS = 25.0, 10.0


def frame_rms(x, sr=SR_EXPECT, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    fl, hl = int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)
    if len(x) < fl:
        return np.zeros(0)
    n = 1 + (len(x) - fl) // hl
    idx = np.arange(fl)[None, :] + hl * np.arange(n)[:, None]
    return np.sqrt((x[idx] ** 2).mean(axis=1))


def pct_of_window(chan, sr, t0, t1):
    """[t0,t1] 窗口的帧能量中位数, 落在该声道**自己**帧能量分布的第几分位。

    🔴 **必须从样本下标切片, 不许把秒换算成帧下标。** 第一版写的是
        i0 = int(t0 * sr / 1000 / HOP_MS)
    左结合 ⇒ `(t0*sr)/1000/10` = `t0*sr/10000`, 而真正的 hop 是
    `sr*HOP_MS/1000 = 240` 个样本 —— **差了 24 倍**。于是 t0=21.78s 被算成帧 52
    (0.52s), 整整早了 21 秒, 整列「对方声道分位」量的是错误的时间窗。
    它错得不像错: 输出全是 0.0%/74.9%/98.7% 这类合法百分位, 肉眼无从察觉;
    而 `t1 <= 总时长` 之类的边界 assert **也拦不住** (22.4s < 188.4s)。
    ⇒ 换算法而不是换常数: 直接切样本, 结构上做不出下标错。
    """
    r = frame_rms(chan)
    if r.size == 0:
        return float("nan"), float("nan")
    a, b = int(round(t0 * sr)), int(round(t1 * sr))
    a = max(a, 0)
    b = min(max(b, a + 1), len(chan))
    w = frame_rms(chan[a:b])
    if w.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(w))
    return med, float((r < med).mean())


def probe(tar, member, workdir=None):
    tmp = workdir or tempfile.mkdtemp(prefix="dgbt_")
    made = workdir is None
    try:
        subprocess.run(["tar", "-xf", tar, member], cwd=tmp, check=True,
                       stderr=subprocess.DEVNULL)
        root = os.path.join(tmp, member)
        dlg, sr = read_wav(os.path.join(root, "dialogues", "dialogue.wav"))
        nch = dlg.shape[1]
        bcs = sorted(glob.glob(os.path.join(root, "backchannels", "*.wav")))
        print(f"=== {member}   {dlg.shape[0]/sr:.1f}s  {nch}ch  "
              f"backchannel {len(bcs)} 个")
        hdr = ("%10s %7s | %4s %7s %8s | %14s %14s"
               % ("BC", "时长s", "声道", "峰", "增益dB", "对方声道分位", "同声道分位"))
        print(hdr)
        print("-" * len(hdr))
        rows = []
        for p in bcs:
            x0, _ = read_wav(p)
            x = x0[:, 0]
            out = [ncc_peak(x, dlg[:, ch]) for ch in range(nch)]
            best = max(range(len(out)), key=lambda i: out[i][0])
            peak, k, _nok, _med, _p999 = out[best]
            t0 = k / sr
            t1 = t0 + len(x) / sr
            win = dlg[k:k + len(x), best]
            # 增益: 窗口里那一份 bc 的幅度 (最小二乘投影)。NCC 尺度不变,
            # 所以峰高不告诉你音量 —— 这个数才告诉。
            denom = float((x ** 2).sum())
            g = float((win * x).sum() / denom) if denom > 0 else 0.0
            g_db = 20 * np.log10(abs(g)) if abs(g) > 1e-12 else -999.0
            # 对方声道此刻的活跃度: 用分位数, 不拍阈值
            others = [c for c in range(nch) if c != best]
            o_med = o_pct = float("nan")
            if others:
                o_med, o_pct = pct_of_window(dlg[:, others[0]], sr, t0, t1)
            s_med, s_pct = pct_of_window(dlg[:, best], sr, t0, t1)
            # 窗口里 bc 自己占多少能量 (若对方也在说话, 这个比会明显 < 1)
            tot = float((win ** 2).sum())
            share = float(((g * x) ** 2).sum()) / tot if tot > 0 else 0.0
            rows.append((t0, t1, best, peak, g_db, o_pct, s_pct, share))
            print("%10.2f %7.3f | %4s %7.4f %8.1f | %11.1f%% %13.1f%%"
                  % (t0, len(x) / sr, "LR"[best], peak, g_db,
                     100 * o_pct, 100 * s_pct))
        if rows:
            a = np.array([[r[4], r[5], r[6], r[7]] for r in rows])
            print("\n  中位: 增益 %.1f dB | 对方声道 %.1f 分位 | "
                  "同声道 %.1f 分位 | bc 占窗口能量 %.0f%%"
                  % (np.median(a[:, 0]), 100 * np.median(a[:, 1]),
                     100 * np.median(a[:, 2]), 100 * np.median(a[:, 3])))
            print("  对方声道分位 >50%% 的 BC 数: %d/%d"
                  % (int((a[:, 1] > 0.5).sum()), len(rows)))
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
