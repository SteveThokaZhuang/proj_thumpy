"""HumDial 第四遍：**把能量包络打出来**，回答「这文件里到底有谁在说」。

## 为什么必须做到这一步

第三遍量出两件互相矛盾的事：

1. `clean_X.wav` 在用户区间**外**的 RMS 恒为 **0.0000**（数字静音）。
2. 但 `X.wav`（主版）在用户区间**外**的 RMS 也 ≈ **0.0000–0.0005**（同样基本静音）。

第 2 条推翻了「主版 = 双工混音，空隙 = 助手在说」这个我一直默认的模型 ——
**如果助手的声音不在空隙里，那助手在哪？** 这一条不搞清楚，HumDial 能不能用、
怎么用，全是空的。

⚠️ 而且第三遍还量出：`clean` 与主版在用户区间**内** 逐样本 maxΔ 高达 0.98、
相关系数只有 0.62–0.98 ⇒ **clean 不是「把主版置零」**（那个假设已证伪）。

## 本遍要量什么（全部逐帧，20 ms 帧，75% 重叠）

- **E1** 主版 `X.wav` 的能量包络 vs `speech_segments`：
  区间内有多少帧在阈值上、区间外有多少帧在阈值上。
  **区间外只要有一批帧在阈值上 ⇒ 文件里有第二个人。**
- **E2** 同一张图对 `clean_X.wav` 做一遍 —— 与 E1 并排。
- **E3** 主版与 clean 的**逐帧能量**相关（不是逐样本）：
  若 ≈1 ⇒ 两者是同一段语音的不同渲染（降噪/编码）；若低 ⇒ 是两段不同的录音。
- **E4** 空隙里到底有什么：把主版在「区间外」的那部分单独拿出来，
  报它的能量分布（而不是一个平均 RMS —— 平均会把零星的高能帧淹掉）。

## 判据（跑之前写死）

- E1 区间外高能帧占比 **> 5%** ⇒ 文件里**有第二个人**（助手/第三方在轨）。
  **< 1%** ⇒ 文件基本只有标注的那个人。
- E3 逐帧能量相关 **> 0.9** ⇒ clean 是主版的另一种渲染（同一段语音）。
  **< 0.7** ⇒ 两段不同录音。
- ⚠️ 两者可以同时成立，也可以都不成立 —— **不许把结论预先捏成二选一**。

用法（纯读盘，0 GPU）：
  python scripts/humdial_energy_probe.py --zip <zip> --n 14
"""
import argparse
import json
import os
import struct
import sys
import zipfile

import numpy as np

FRAME = 0.020          # 20 ms
HOP = 0.005            # 75% 重叠


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
    if off is None or sr is None:
        return None, None
    a = np.frombuffer(raw[off:off + sz], dtype="<i2").astype(np.float32) / 32768.0
    return a, sr


def frames(a, sr):
    """返回 (每帧 RMS, 每帧起点秒)。"""
    fl, hl = int(FRAME * sr), int(HOP * sr)
    if len(a) < fl:
        return np.array([]), np.array([])
    idx = np.arange(0, len(a) - fl + 1, hl)
    # 用 stride 技巧一次算完
    st = np.lib.stride_tricks.sliding_window_view(a, fl)[::hl]
    return np.sqrt((st ** 2).mean(axis=1)), idx / sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--thr", type=float, default=0.003,
                    help="帧 RMS 阈值（相对满量程）；下面会同时报多个阈值")
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.zip)
    names = set(i.filename for i in zf.infolist() if i.file_size > 0)
    jset = set(n for n in names if n.endswith(".json"))

    dirs = sorted({os.path.dirname(n) for n in names
                   if os.path.dirname(n).startswith("test/")})
    picks = []
    for d in dirs:
        ids = sorted(os.path.basename(n)[:-5] for n in names
                     if os.path.dirname(n) == d and n.endswith(".json")
                     and not n.endswith("_timestamp.json")
                     and not os.path.basename(n).startswith("clean_"))
        if ids:
            picks.append((d, ids[0]))
    picks = picks[:args.n]

    print("=" * 92)
    print("E1/E2 能量包络 vs speech_segments（帧 %.0f ms / 跳 %.0f ms）"
          % (FRAME * 1000, HOP * 1000))
    print("=" * 92)
    print("%-34s %-6s %7s %7s %7s %7s %8s" %
          ("场景/id/文件", "阈值", "内帧数", "内高能", "外帧数", "外高能", "外占比"))
    thr_list = [0.001, 0.003, 0.01]
    agg = {t: {"main_out": [], "clean_out": [], "main_in": [], "clean_in": []}
           for t in thr_list}
    e3 = []
    for d, st in picks:
        w1, w2 = "%s/%s.wav" % (d, st), "%s/clean_%s.wav" % (d, st)
        sp = "%s/%s.json" % (d, st)
        if w1 not in names or sp not in jset:
            print("%-34s 缺主版" % ("%s/%s" % (d.replace("test/", ""), st)))
            continue
        a1, sr = read_wav(zf, w1)
        if a1 is None:
            continue
        segs = json.loads(zf.read(sp))["speech_segments"]
        e1, t1 = frames(a1, sr)
        m1 = np.zeros(len(e1), dtype=bool)
        for s in segs:
            m1 |= (t1 >= s["xmin"]) & (t1 <= s["xmax"])
        has_c = w2 in names
        if has_c:
            a2, _ = read_wav(zf, w2)
            nmin = min(len(a1), len(a2))
            e2, t2 = frames(a2[:nmin], sr)
            m2 = np.zeros(len(e2), dtype=bool)
            for s in segs:
                m2 |= (t2 >= s["xmin"]) & (t2 <= s["xmax"])
            k = min(len(e1), len(e2))
            if e1[:k].std() > 0 and e2[:k].std() > 0:
                e3.append((st, float(np.corrcoef(e1[:k], e2[:k])[0, 1])))
                # 只比两边都有能量的帧，免得静音段把相关拉高
                mm = (e1[:k] > args.thr) | (e2[:k] > args.thr)
                if mm.sum() > 10 and e1[:k][mm].std() > 0 and e2[:k][mm].std() > 0:
                    e3[-1] = (st, float(np.corrcoef(e1[:k][mm], e2[:k][mm])[0, 1]))

        for t in thr_list:
            io, oo = int((e1[m1] > t).sum()), int((e1[~m1] > t).sum())
            ni, no = int(m1.sum()), int((~m1).sum())
            agg[t]["main_in"].append(io / ni if ni else 0)
            agg[t]["main_out"].append(oo / no if no else 0)
            if has_c:
                io2, oo2 = int((e2[m2] > t).sum()), int((e2[~m2] > t).sum())
                agg[t]["clean_in"].append(io2 / max(1, int(m2.sum())))
                agg[t]["clean_out"].append(oo2 / max(1, int((~m2).sum())))
            if t == args.thr:
                print("%-34s %-6.3f %7d %7d %7d %7d %7.2f%%"
                      % ("%s/%s%s" % (d.replace("test/", ""), st,
                                      "" if has_c else " (无clean)"),
                         t, ni, io, no, oo, 100 * oo / no if no else float("nan")))
    print("\n  汇总（各阈值下，区间外高能帧占比的中位数）：")
    print("  %-8s %14s %14s %14s %14s" %
          ("阈值", "主版·区间内", "主版·区间外", "clean·区间内", "clean·区间外"))
    for t in thr_list:
        row = []
        for k in ("main_in", "main_out", "clean_in", "clean_out"):
            v = agg[t][k]
            row.append("%.2f%%" % (100 * float(np.median(v))) if v else "—")
        print("  %-8.3f %14s %14s %14s %14s" % (t, *row))

    print("\n" + "=" * 92)
    print("E3 主版 vs clean 的**逐帧能量**相关（只在两边至少一边有能量的帧上算）")
    print("=" * 92)
    if e3:
        r = np.array([x[1] for x in e3])
        print("  n=%d  中位 %.4f  均值 %.4f  [%.4f, %.4f]"
              % (len(r), np.median(r), r.mean(), r.min(), r.max()))
        for st, v in e3:
            print("     %-20s %.4f" % (st, v))
        med = float(np.median(r))
        print("  ⇒ %s" % (">0.9 ⇒ clean 是主版的另一种渲染（同一段语音）" if med > 0.9
                          else "<0.7 ⇒ 两段不同录音" if med < 0.7
                          else "0.7~0.9 ⇒ **落在中间**，不许二选一"))
    else:
        print("  ⚠️ 一个都没算成 —— 检查 clean 是否存在")

    print("\n" + "=" * 92)
    print("E4 空隙（区间外）里到底有什么：把主版区间外的帧单独看分布")
    print("=" * 92)
    for d, st in picks[:5]:
        w1, sp = "%s/%s.wav" % (d, st), "%s/%s.json" % (d, st)
        if w1 not in names or sp not in jset:
            continue
        a1, sr = read_wav(zf, w1)
        e1, t1 = frames(a1, sr)
        m1 = np.zeros(len(e1), dtype=bool)
        for s in json.loads(zf.read(sp))["speech_segments"]:
            m1 |= (t1 >= s["xmin"]) & (t1 <= s["xmax"])
        out = e1[~m1]
        if not len(out):
            continue
        q = np.percentile(out, [50, 90, 99, 99.9])
        print("  %-30s 外帧 %5d  中位 %.5f  p90 %.5f  p99 %.5f  p99.9 %.5f  max %.5f"
              % ("%s/%s" % (d.replace("test/", ""), st), len(out),
                 q[0], q[1], q[2], q[3], out.max()))
        print("      区间内帧 RMS 中位 %.5f（作对照）"
              % float(np.median(e1[m1])) if m1.sum() else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
