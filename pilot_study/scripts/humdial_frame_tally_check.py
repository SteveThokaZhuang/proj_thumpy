"""为什么两个探针对**同一批帧**报出了不同的数？

## 症状

同一份 zip、同一个 5 ms 网格、同一批 4172 条：

  `humdial_timing_probe.py`     R 语音帧 7,090,241 ｜ A 覆盖 6,051,014 ｜ 未覆盖 1,039,227
  `humdial_timing_decompose.py` R 语音帧 7,090,339 ｜ A 覆盖 6,051,101 ｜ 未覆盖 1,039,238
                                        +98                +87               +11

差异 **98 / 7,090,339 = 0.0014%** —— 小到不影响任何结论，但按本项目 §四.21
（「同一个数据，换个仪器，答案差 6 倍」）的规矩：**两个仪器给出不同的数，
就要说清楚引用的是哪一个**，而不是当没看见。

## 待验的假设（跑之前写死）

两者唯一的实质差别是**计数口径**：

  - **probe**：全网格上 `mR = mask_of(R_int)` —— 区间**先合并成掩码再数**
    ⇒ 若两条 R 区间重叠，重叠部分**只算一次**（并集）。
  - **decompose**：逐条 R 区间 `n_r = len(seg)` 再求和
    ⇒ 重叠部分**算两次**（长度和）。

⇒ **假设 H**：这 98 帧全部来自**重叠的 R 区间**，
   且 `Σ(区间长度) − |并集| == 98`。

## 判据

  H 成立 ⇔ 「逐条求和 − 并集」**恰好等于 98**（两个脚本各自的取整口径可能再贡献
  极少数帧 ⇒ 允许 ±5 的解释余量，但必须是**同一个符号、同一个量级**；
  差出几百帧以上说明假设错了，**如实报，不许圆**）。

## 顺带查一件更要紧的事

若 R 区间真的有重叠 ⇒ **"R 覆盖了多少语音帧"这个量本身取决于口径**。
本项目的帧级 F1 用的是 **probe 的并集口径** ⇒ 正文引用一律用 7,090,241。

用法（纯读盘，0 GPU，必须经 srun --overlap --ntasks=1）：
  python scripts/humdial_frame_tally_check.py --zip <zip>
"""
import argparse
import json
import os
import sys
import zipfile

import numpy as np

HOP = 0.005


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.zip)
    names = [i.filename for i in zf.infolist() if i.file_size > 0]
    jset = set(n for n in names if n.endswith(".json"))

    pairs, seen = [], set()
    for n in sorted(jset):
        d, b = os.path.dirname(n), os.path.basename(n)
        if b.endswith("_timestamp.json") or b.startswith("clean_"):
            continue
        st = b[:-5]
        if "_add" in st or (d, st) in seen:
            continue
        seen.add((d, st))
        sp, tp = "%s/%s.json" % (d, st), "%s/%s_timestamp.json" % (d, st)
        if sp in jset and tp in jset:
            pairs.append((d, st, sp, tp))

    print("可配对 %d 条\n" % len(pairs))

    n_turns = 0
    n_ov_dlg = 0          # 至少含一对重叠区间的对话数
    n_ov_pairs = 0        # 重叠区间**对**数
    sum_len = 0           # 逐条区间长度之和（decompose 口径）
    sum_union = 0         # 并集（probe 口径）
    ov_frames_union = 0   # 重叠处的**并集帧数**（即被重复计的那些）
    worst = []

    for d, st, sp, tp in pairs:
        R = json.loads(zf.read(sp))
        segs = R.get("speech_segments") or []
        if not segs:
            continue
        R_int = [(float(s["xmin"]), float(s["xmax"])) for s in segs]
        T = max(b for _, b in R_int)
        ngrid = int(T / HOP) + 1

        # --- 逐条求和（decompose 口径）---
        for a, b in R_int:
            if b <= a:
                continue
            n_turns += 1
            i0 = int(np.ceil(a / HOP)); i1 = int(np.floor(b / HOP))
            if i1 < i0:
                i0 = i1 = max(0, min(ngrid - 1, int(round(a / HOP))))
            sum_len += (min(ngrid, i1 + 1) - max(0, i0))

        # --- 并集（probe 口径）---
        m = np.zeros(ngrid, dtype=bool)
        for a, b in R_int:
            if b <= a:
                continue
            i0 = int(np.ceil(a / HOP)); i1 = int(np.floor(b / HOP))
            if i1 < i0:
                i0 = i1 = max(0, min(ngrid - 1, int(round(a / HOP))))
            m[max(0, i0):min(ngrid, i1 + 1)] = True
        sum_union += int(m.sum())

        # --- 重叠检测：按起点排序后看相邻 ---
        srt = sorted(R_int)
        dlg_ov = False
        for k in range(1, len(srt)):
            if srt[k][0] < srt[k - 1][1]:      # 后一个的起点落在前一个里面
                n_ov_pairs += 1
                dlg_ov = True
                ov_s = srt[k][0]
                ov_e = min(srt[k - 1][1], srt[k][1])
                if ov_e > ov_s:
                    j0 = max(0, int(np.ceil(ov_s / HOP)))
                    j1 = min(ngrid - 1, int(np.floor(ov_e / HOP)))
                    if j1 >= j0:
                        ov_frames_union += (j1 - j0 + 1)
                worst.append((ov_e - ov_s, d, st))
        if dlg_ov:
            n_ov_dlg += 1

    print("=" * 92)
    print("R 侧帧数：两种口径")
    print("=" * 92)
    print("  轮次（R 区间）合计        %12d" % n_turns)
    print("  逐条求和（decompose 口径） %12d" % sum_len)
    print("  并集      （probe 口径）   %12d" % sum_union)
    diff = sum_len - sum_union
    print("  ⇒ **差 %d 帧**（= decompose 多算的）" % diff)
    print("  其中的重叠帧（并集口径）    %12d" % ov_frames_union)
    print()
    print("  含重叠区间的对话  %d ／ 重叠区间对 %d"
          % (n_ov_dlg, n_ov_pairs))

    print("\n" + "=" * 92)
    print("判据")
    print("=" * 92)
    print("  实测差 %d  vs  重叠帧 %d" % (diff, ov_frames_union))
    if abs(diff - ov_frames_union) <= 5:
        print("  ⇒ ✅ **假设 H 成立** —— 那 98 帧就是重叠区间被 decompose 算了两遍。")
        print("     两个脚本各自都对，**是口径不同不是 bug**；正文引用以并集（7,090,241）为准。")
    else:
        print("  ⇒ 🔴 **假设 H 不成立** —— 差 %d 而重叠只有 %d 帧。"
              % (diff, ov_frames_union))
        print("     说明另有来源（取整？零长度？），**如实报，不许圆**。")
    if worst:
        w = sorted(worst, reverse=True)[:5]
        print("\n  最长的 5 处重叠（秒）：",
              " ｜ ".join("%.3f (%s/%s)" % (a, b, c) for a, b, c in w))

    print("\n  ⚠️ 口径提醒：「R 覆盖多少语音帧」这个量**取决于口径**。")
    print("     probe 的 F1 用的是**并集** ⇒ 正文一律引用 **7,090,241**。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
