"""用**精确数字零**判"这个声道此刻有没有人说话", 量 dialogue.wav 的重叠。

## 为什么扔掉 Otsu

`duplexgen_overlap_probe.py` 用每声道自己的 Otsu 阈值判活跃, 得到重叠 0.17%–1.74%。
**那个数是错的, 而且错法有两层** (2026-09-23 实测):

1. 波形显示: 说话人不在说话时, 该声道是**精确的 0.00000** (−240 dB) ——
   没有混响、没有串音、没有本底噪声。**根本不需要阈值**。
2. Otsu 把分割点放在「响亮语音 vs 静音」上, 而 backchannel 是短的轻声,
   落在自己的说话人分布的中低段 (实测 34–86 分位) ⇒ **恰好被阈值漏掉**。
   于是"重叠"只剩轮次交接的碎渣。

⇒ 本脚本的判据: 某声道"有人" ⟺ 该声道样本 **≠ 0**。零个自由参数, 零个刻度。

## 自带零模型

不需要另造零分布: **判据线是 0**。某个 (对话, 声道) 有没有重叠, 是可数的确定事实;
而"重叠时段是否等于 backchannel 时段"靠**时间区间是否重合**判, 也不需要阈值。

用法 (fd_analysis, 0 GPU):
  python duplexgen_overlap_exact.py --tar <tar> --member <目录>
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


def spans_of(mask, sr, min_s=0.06):
    """布尔掩码 -> [(起, 止)] 秒, 丢掉短于 min_s 的碎渣。"""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if (j - i) / sr >= min_s:
                out.append((i / sr, j / sr))
            i = j
        else:
            i += 1
    return out


def overlap_frac(a, b):
    """两个区间列表的重合总时长 (两边都需已排序)。"""
    tot, i, j = 0.0, 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            tot += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def probe(tar, member, workdir=None):
    tmp = workdir or tempfile.mkdtemp(prefix="dgoe_")
    made = workdir is None
    try:
        subprocess.run(["tar", "-xf", tar, member], cwd=tmp, check=True,
                       stderr=subprocess.DEVNULL)
        root = os.path.join(tmp, member)
        dlg, sr = read_wav(os.path.join(root, "dialogues", "dialogue.wav"))
        L, R = dlg[:, 0], dlg[:, 1]
        nl, nr = L != 0.0, R != 0.0
        both = nl & nr
        dur = len(dlg) / sr
        print("=== %s   %.1fs  声道 %d" % (member, dur, dlg.shape[1]))
        print("  非零样本占比  L=%.1f%%  R=%.1f%%" % (100 * nl.mean(), 100 * nr.mean()))
        print("  🔴 双方同时非零  %.3f%%  (%.2fs)"
              % (100 * both.mean(), both.sum() / sr))
        # 交叉核对: 精确零与 Otsu 差多少 (把老仪器的偏差量出来, 不留悬案)
        ov = spans_of(both, sr)
        print("  重叠段 %d 个, 最长 %.2fs" % (len(ov), max((b - a for a, b in ov), default=0)))

        bcs = sorted(glob.glob(os.path.join(root, "backchannels", "*.wav")))
        # BC 的时刻: 不靠命名, 靠互相关峰 (已实测峰=1.0000)。但 BC 落在哪个声道,
        # 现在可以**精确**判: 它在哪个声道上非零。
        # 🔴 **不要**用 spans_of 的结果去和 BC 求交。第一版这么做了, 得到
        #    「BC 被重叠覆盖 15%」, 而逐 BC 直接数是 98.9–100% —— 差 6 倍。
        #    原因: both 是**样本级**掩码, 而 BC 窗口里 L 有约 5% 的散点恰为零
        #    ⇒ 掩码碎成几十片 ⇒ spans_of 的「丢短于 60ms」把整个 BC 窗口丢掉,
        #    只剩 1.04s 碎渣去求交。**过滤是为了「打印好看的段」, 不能用来算总量。**
        #    ⇒ 覆盖率一律在样本级数, 不经过任何分段/过滤。
        in_bc = np.zeros(len(dlg), dtype=bool)
        bc_spans = []
        nbc = 0
        for p in bcs:
            x0, _ = read_wav(p)
            x = x0[:, 0]
            # NCC 峰已在 `duplexgen_bc_timing.py` 上验到 1.0000 (样本级对齐),
            # 所以峰位就是要用的时刻。别在这里另造一套定位逻辑 —— 两套定位
            # 一旦不一致, 排查成本远高于复用。
            out = [ncc_peak(x, dlg[:, ch]) for ch in range(dlg.shape[1])]
            k = out[int(np.argmax([o[0] for o in out]))][1]
            t0 = k / sr
            bc_spans.append((t0, t0 + len(x) / sr))
            in_bc[k:k + len(x)] = True
            nbc += len(x)
        bc_spans.sort()
        n_both = int(both.sum())
        n_both_in_bc = int((both & in_bc).sum())
        print("  backchannel %d 个 / 合计 %.2fs（样本 %d）" % (len(bcs), nbc / sr, nbc))
        print("  ┌ 重叠总样本        %d  (%.2fs)" % (n_both, n_both / sr))
        print("  ├ 其中落在 BC 窗口  %d  (%.2fs, 占重叠 %.0f%%)"
              % (n_both_in_bc, n_both_in_bc / sr,
                 100 * n_both_in_bc / max(n_both, 1)))
        print("  └ BC 窗口被重叠覆盖 %.0f%%  (若≈100%% ⇒ 重叠就是 BC 造成的)"
              % (100 * n_both_in_bc / max(nbc, 1)))
        print("     [参考] 含 ≥60ms 的连续重叠段 %d 个 / 合计 %.2fs"
              % (len(ov), sum(b - a for a, b in ov)))
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
