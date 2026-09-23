"""把 `clean_` 钉死：**逐段**看能量，不再用「整条掩码」这个脏仪器。

## 前四遍为什么全歪了

前几遍一律用 `X.json` 的**全部** speech_segments 造一个掩码，然后拿它去量 clean。
而 group_dump 摊开一看：

    0001_0004  主版词级 text = "…remote work environment? How often should these
                              video check-ins ideally be scheduled?"   (24 chunks)
    clean_0001_0004 词级 text = "…remote work environment?"             (15 chunks)

⇒ **clean 版只有第一段（用户的第一句），第二段（要求重复那句）没了。**
而主版的掩码把第二段也算成「用户区间」，于是：

- clean 在第二段上是**静音** ⇒ 逐样本 maxΔ 大、逐帧能量相关低 ⇒ 我上一遍
  据此写了「clean 是另一路音频」—— **那条结论是被脏掩码造出来的假象**
  （memory `tiebreak-default-read-as-evidence`：「过滤器会制造缺口」，
  这次是「用错掩码制造出差异」）。

## 本遍的干净仪器

**逐段**算：对主版的第 i 段，分别量主版与 clean 在该段上的高能帧占比与
逐帧能量相关。判据（写死）：

- 若 clean 在**第 1 段**上与主版近乎逐样本相同（maxΔ ≈ 0），
  且在第 2 段上恒为静音 ⇒ **clean = 主版掐掉后续轮次**（同一段录音，只留首轮）。
- 若 clean 在第 1 段上就与主版不同 ⇒ 才是「另一路音频」。
- 两段都静音 / 都不静音 ⇒ 说明假设错，如实报。

用法（纯读盘，0 GPU）：
  python scripts/humdial_clean_resolve.py --zip <zip> --n 12
"""
import argparse
import json
import os
import struct
import sys
import zipfile

import numpy as np

FRAME, HOP = 0.020, 0.005


def read_wav(zf, name):
    with zf.open(name) as f:
        raw = f.read()
    if raw[:4] != b"RIFF":
        return None, None
    p, sr, off, sz = 12, None, None, None
    while p < len(raw) - 8:
        cid = raw[p:p + 4]
        n = struct.unpack("<I", raw[p + 4:p + 8])[0]
        if cid == b"fmt ":
            _, ch, sr, br, ba, bits = struct.unpack("<HHIIHH", raw[p + 8:p + 24])
        elif cid == b"data":
            off, sz = p + 8, n
            break
        p += 8 + n + (n & 1)
    if off is None:
        return None, None
    return np.frombuffer(raw[off:off + sz], dtype="<i2").astype(np.float32) / 32768.0, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--thr", type=float, default=0.003)
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.zip)
    names = set(i.filename for i in zf.infolist() if i.file_size > 0)
    jset = set(n for n in names if n.endswith(".json"))
    dirs = sorted({os.path.dirname(n) for n in names
                   if os.path.dirname(n).startswith("test/")})

    picks = []
    for d in dirs:
        for n in sorted(names):
            if os.path.dirname(n) != d or not n.endswith(".json"):
                continue
            b = os.path.basename(n)
            if b.endswith("_timestamp.json") or b.startswith("clean_"):
                continue
            st = b[:-5]
            if "%s/clean_%s.wav" % (d, st) in names:
                picks.append((d, st))
                break
    picks = picks[:args.n]

    print("=" * 90)
    print("逐段对比：主版 vs clean（帧 %.0f ms，阈值 %.3f）" % (FRAME * 1000, args.thr))
    print("=" * 90)
    print("%-32s %-4s %8s %8s %8s %8s %9s" %
          ("场景/id", "段", "主版高能", "clean高能", "逐样本maxΔ", "能量r", "段时长s"))
    verdict = []
    for d, st in picks:
        w1, w2 = "%s/%s.wav" % (d, st), "%s/clean_%s.wav" % (d, st)
        sp = "%s/%s.json" % (d, st)
        if sp not in jset:
            continue
        segs = json.loads(zf.read(sp))["speech_segments"]
        a1, sr = read_wav(zf, w1)
        a2, _ = read_wav(zf, w2)
        if a1 is None or a2 is None:
            continue
        n = min(len(a1), len(a2))
        a1, a2 = a1[:n], a2[:n]
        st_e = np.lib.stride_tricks.sliding_window_view(a1, int(FRAME * sr))[::int(HOP * sr)]
        e1 = np.sqrt((st_e ** 2).mean(1))
        st_e = np.lib.stride_tricks.sliding_window_view(a2, int(FRAME * sr))[::int(HOP * sr)]
        e2 = np.sqrt((st_e ** 2).mean(1))
        k = min(len(e1), len(e2))
        e1, e2 = e1[:k], e2[:k]
        t = np.arange(k) * HOP
        rows = []
        for i, s in enumerate(segs):
            m = (t >= s["xmin"]) & (t <= s["xmax"])
            if m.sum() < 5:
                continue
            i0, i1 = int(s["xmin"] * sr), int(s["xmax"] * sr)
            mx = float(np.abs(a1[i0:i1] - a2[i0:i1]).max())
            r = (float(np.corrcoef(e1[m], e2[m])[0, 1])
                 if e1[m].std() > 0 and e2[m].std() > 0 else float("nan"))
            h1 = float((e1[m] > args.thr).mean())
            h2 = float((e2[m] > args.thr).mean())
            rows.append((i, h1, h2, mx, r, s["xmax"] - s["xmin"]))
            print("%-32s %-4d %7.1f%% %7.1f%% %10.6f %8.3f %9.2f"
                  % ("%s/%s" % (d.replace("test/", ""), st), i, 100 * h1, 100 * h2,
                     mx, r, s["xmax"] - s["xmin"]))
        if rows:
            verdict.append((st, rows))

    print("\n" + "=" * 90)
    print("汇总：第 1 段 vs 第 2 段")
    print("=" * 90)
    first = [(r[3], r[1], r[2]) for _, rows in verdict for r in rows if r[0] == 0]
    later = [(r[3], r[1], r[2]) for _, rows in verdict for r in rows if r[0] > 0]
    if first:
        mx = np.array([x[0] for x in first])
        h1 = np.array([x[1] for x in first])
        h2 = np.array([x[2] for x in first])
        print("  第 1 段 (n=%d): 主版高能 %.1f%%  clean高能 %.1f%%  逐样本maxΔ 中位 %.6f  max %.6f"
              % (len(first), 100 * h1.mean(), 100 * h2.mean(),
                 float(np.median(mx)), float(mx.max())))
        print("     clean 高能 / 主版高能 = %.3f"
              % (h2.mean() / h1.mean() if h1.mean() else float("nan")))
    if later:
        h1 = np.array([x[1] for x in later])
        h2 = np.array([x[2] for x in later])
        print("  第 2 段 (n=%d): 主版高能 %.1f%%  clean高能 %.1f%%"
              % (len(later), 100 * h1.mean(), 100 * h2.mean()))
        print("     clean 高能 / 主版高能 = %.3f"
              % (h2.mean() / h1.mean() if h1.mean() else float("nan")))

    if first and later:
        f_ratio = np.array([x[2] for x in first]).mean() / max(
            1e-9, np.array([x[1] for x in first]).mean())
        l_ratio = np.array([x[2] for x in later]).mean() / max(
            1e-9, np.array([x[1] for x in later]).mean())
        print("\n  判读：第 1 段 clean/主版 = %.3f ；第 2 段 clean/主版 = %.3f" % (f_ratio, l_ratio))
        if f_ratio > 0.7 and l_ratio < 0.1:
            print("  ⇒ ✅ **clean = 主版掐掉后续轮次**（同一段录音，只留首轮）。")
            print("     前几遍那个「clean 是另一路音频」是**用错掩码造出来的假象**。")
        elif f_ratio < 0.5 and l_ratio < 0.5:
            print("  ⇒ clean 两段都弱 —— 不是「只留首轮」，如实报。")
        else:
            print("  ⇒ **不落在两种干净形态上** —— 不许二选一，要看更多样本。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
