"""§5.2d 追加诊断：为什么有些 utterance **整文件**对不上，计数却严丝合缝？

`duplexgen_utterance_tiling.py` 的结果是分裂的：
  T3 计数恒等 24/24 **精确整数相等**（Σutt非零 == L非零）
  T4 命中区段两两不重叠 24/24
  T1 整文件逐样本相等 只有 12/24 —— 每个对话有 0~4 个「定位失败」

但 T3 成立意味着：定位失败的那些 utterance 的非零样本数**必须**恰好等于
L 未被命中区段覆盖的那部分。也就是说它们的**内容在 L 里**，只是**整文件**不等。
最可能的解释是静音填充不同（切片时掐头去尾），但这只是假设 —— 本脚本去量它。

⚠️ 本脚本的判据是**事后**追加的（看到 T1 失败之后才写的），
   因此它产出的任何"通过"都**不能**冒充预登记的 T1。文档里必须分开写。

用法:
  python duplexgen_utt_unlocated_probe.py --per-scenario 2
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from duplexgen_utterance_tiling import DS, SCEN, locate, read_wav  # noqa: E402


def probe(root):
    dlg, sr = read_wav(os.path.join(root, "dialogues", "dialogue.wav"))
    utts = sorted(glob.glob(os.path.join(root, "utterances", "*.wav")))
    key = root.split("/")[-2] + "/" + root.split("/")[-1]
    print("\n=== %s  (L=%.3fs, %d utt)" % (key, len(dlg) / sr, len(utts)))
    print("  %-9s %5s %8s %8s %8s %6s %8s %s"
          % ("utt", "ch", "非零", "头静音", "尾静音", "双声道", "内容命中", "说明"))
    for p in utts:
        x, _ = read_wav(p)
        dup = "同" if x.shape[1] == 2 and np.array_equal(x[:, 0], x[:, 1]) \
            else ("异" if x.shape[1] == 2 else "单")
        u = x[:, 0]
        nz = np.nonzero(u)[0]
        nm = os.path.basename(p)
        if nz.size == 0:
            # 全零 utterance: 另一声道呢?
            alt = int((x[:, 1] != 0).sum()) if x.shape[1] == 2 else -1
            print("  %-9s %5s %8d %8s %8s %6s %8s %s"
                  % (nm, "零", 0, "—", "—", dup, "—",
                     "**整条全零**; 另一声道非零=%d" % alt))
            continue
        i0, i1 = int(nz[0]), int(nz[-1]) + 1
        seg = u[i0:min(i0 + 512, len(u))]
        placed = None
        for ch in range(dlg.shape[1]):
            for c in locate(seg, dlg[:, ch]):
                off = c - i0
                if off < 0:
                    continue
                # 只看**非零跨度**是否逐样本相等（这正是事后的那一步）
                a = off + i0
                b = off + i1
                if b > len(dlg):
                    continue
                if np.array_equal(dlg[a:b, ch], u[i0:i1]):
                    placed = (ch, off)
                    break
            if placed:
                break
        ch_lbl = "L" if placed and placed[0] == 0 else \
                 ("R" if placed else "—")
        if placed:
            ch, off = placed
            whole = (off + len(u) <= len(dlg)) and \
                np.array_equal(dlg[off:off + len(u), ch], u)
            note = "整文件也相等" if whole else \
                ("仅非零跨度相等 (头差%d/尾差%d 样本)"
                 % (i0, (off + len(u)) - (off + i1)
                    if off + len(u) <= len(dlg) else -1))
        else:
            note = "🔴 连非零跨度都找不到"
        print("  %-9s %5s %8d %8d %8d %6s %8s %s"
              % (nm, ch_lbl, nz.size, i0, len(u) - i1, dup,
                 ("✅" if placed else "🔴"), note))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scenario", type=int, default=2)
    ap.add_argument("--scenarios", default="INT,PLN,TEA")
    a = ap.parse_args()
    for sc in [s for s in a.scenarios.split(",") if s]:
        tar = os.path.join(DS, "shards", sc, "%s-00000.tar" % sc)
        tmp = tempfile.mkdtemp(prefix="dgup_")
        try:
            subprocess.run(["tar", "-xf", tar, "-C", tmp, "--wildcards",
                            "*/utterances/*.wav", "*/dialogues/dialogue.wav"],
                           stderr=subprocess.DEVNULL, check=False)
            roots = []
            for dp, _d, fs in os.walk(tmp):
                if os.path.basename(dp) == "dialogues" and "dialogue.wav" in fs:
                    roots.append(os.path.dirname(dp))
            for root in sorted(roots)[:a.per_scenario]:
                probe(root)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
