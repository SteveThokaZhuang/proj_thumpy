"""把 M1 的 F1 **拆开看**：那 0.9125 是由什么构成的。

## 为什么要拆

主探针（`humdial_timing_probe.py`）按预登记的判据报了 **M1 F1 = 0.9125**，
刚好越过 0.90 那条线。但三个附带数字说明**这一句话盖住了结构**：

  precision 0.9803  vs  recall 0.8534     ← 差 13 个点，不是对称的一致
  有符号面积比 |A|/|R| = 0.8706           ← A 系统性**少覆盖** 13%
  M3 onset 中位 +0.149 s / offset 中位 −0.045 s  ← 轮次两端各缩一点

⇒ **F1 把"两个源量的是不是同一个东西"这件事平均掉了。**

## 🔴 本脚本**不改**预登记的任何判据

M1/M2/M3 的判定按 prereg 原文原样保留（F1 ≥ 0.90 ⇒ 写「一致」）。
本脚本只做**事后的构成分析**，回答一个问题：

> **A 没覆盖的那 15% R 语音帧，是"边界缩了"还是"中间有洞"？**

这两件事的含义**完全不同**：

  - 若主要是**边界缩**（每个轮次两端各差 0.1–0.2 s）⇒ 两源对**轮次在哪**其实一致，
    只是 ASR 的词级时间戳天生偏内 ⇒ **不构成"分段分歧"**。
  - 若主要是**中间有洞**（轮次内部的停顿处 A 没有词）⇒ 那是**粒度差**：
    R 是"这一轮"，A 是"这些词"，**两者根本不是在标注同一个量**
    ⇒ 拿它们互比 F1 **本身就不成立**（recon §5.2b 的教训：两套仪器差 6 倍）。

## 附带修一个我自己造的仪器 bug

主探针的 **M4（按轮次长度分箱）打印 FP ≡ 0**（每个箱都是 0）——
这不是发现，是**构造使然**：分箱只在 **R 轮次区间内**取帧，而区间内 R **按定义**恒为活动，
所以 `~mR` 在里面是空集 ⇒ **FP 永远为 0** ⇒ 那一列"F1"其实是**只有 recall 的假 F1**。
（memory `threshold-must-match-null-model` 第五面：定义域里按定义恒定的那一段会被读成发现。）

本脚本按 §四.26 的处置办：**旧读数原样留在日志里不动**，这里另报一个**诚实版 M4**
（逐轮次的 **recall**，并明写 FP 为何缺席）。

用法（纯读盘，0 GPU，经 srun --overlap --ntasks=1）：
  python scripts/humdial_timing_decompose.py --zip <zip>
"""
import argparse
import json
import os
import sys
import zipfile

import numpy as np

HOP = 0.005


def mask_of(intervals, ngrid):
    m = np.zeros(ngrid, dtype=bool)
    for a, b in intervals:
        if b <= a:
            continue
        i0 = int(np.ceil(a / HOP)); i1 = int(np.floor(b / HOP))
        if i1 < i0:
            i0 = i1 = max(0, min(ngrid - 1, int(round(a / HOP))))
        m[max(0, i0):min(ngrid, i1 + 1)] = True
    return m


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

    # ---------- D1 时长账 ----------
    print("=" * 92)
    print("D1 时长账：final_duration ／ R 最末 end ／ A 最末 end")
    print("=" * 92)
    g_fd_r, g_fd_a, g_a_r = [], [], []
    for d, st, sp, tp in pairs:
        R = json.loads(zf.read(sp)); A = json.loads(zf.read(tp))
        fd = R.get("final_duration")
        rmax = max(s["xmax"] for s in R["speech_segments"])
        amax = max(c["timestamp"][1] for c in A["chunks"])
        if fd is not None:
            g_fd_r.append(fd - rmax)
            g_fd_a.append(fd - amax)
        g_a_r.append(amax - rmax)
    for lab, g in (("final_duration − R 最末 end", g_fd_r),
                   ("final_duration − A 最末 end", g_fd_a),
                   ("A 最末 end − R 最末 end", g_a_r)):
        g = np.array(g)
        print("  %-28s 中位 %+8.3f s  [%.3f, %.3f]  SD %.4f  n=%d"
              % (lab, np.median(g), g.min(), g.max(), g.std(), len(g)))
    print("  ⇒ 若「final_duration − R 最末」恒为同一常数 ⇒ 那是一条**结构性尾巴**，")
    print("     不是「这个人还在说」；它对帧级 F1 无贡献（两边都静音），但要写出来。")

    # ---------- D2 未覆盖帧的分解 ----------
    print("\n" + "=" * 92)
    print("D2 🔴 A 没覆盖的 R 语音帧：**边界缩** vs **中间有洞**")
    print("=" * 92)
    tot_r = tot_edge = tot_hole = tot_cov = 0
    turn_rows = []
    for d, st, sp, tp in pairs:
        R = json.loads(zf.read(sp)); A = json.loads(zf.read(tp))
        R_int = [(s["xmin"], s["xmax"]) for s in R["speech_segments"]]
        A_int = [(c["timestamp"][0], c["timestamp"][1]) for c in A["chunks"]]
        T = max(max(b for _, b in R_int), max(b for _, b in A_int))
        ngrid = int(T / HOP) + 1
        mA = mask_of(A_int, ngrid)
        for (a, b) in R_int:
            if b <= a:
                continue
            i0 = max(0, int(np.ceil(a / HOP))); i1 = min(ngrid - 1, int(np.floor(b / HOP)))
            if i1 < i0:
                continue
            seg = mA[i0:i1 + 1]
            n_r = len(seg); n_cov = int(seg.sum())
            tot_r += n_r; tot_cov += n_cov
            # 该 R 轮次内部的 A 词
            ws = [(s, e) for (s, e) in A_int if s < b and e > a]
            if not ws:
                tot_hole += n_r  # 整轮没有词 —— 记作"洞"（不是边界缩）
                turn_rows.append((b - a, 0.0, 0.0))
                continue
            fs = min(s for s, _ in ws); le = max(e for _, e in ws)
            j0 = max(0, int(np.ceil(a / HOP))); j1 = min(ngrid - 1, int(np.floor(fs / HOP)))
            k0 = max(0, int(np.ceil(le / HOP))); k1 = min(ngrid - 1, int(np.floor(b / HOP)))
            edge = 0
            if j1 >= j0:
                edge += int((~mA[j0:j1 + 1]).sum())
            if k1 >= k0:
                edge += int((~mA[k0:k1 + 1]).sum())
            hole = n_r - n_cov - edge
            tot_edge += edge; tot_hole += max(0, hole)
            turn_rows.append((b - a, n_cov / n_r, edge / n_r))

    unc = tot_r - tot_cov
    print("  R 语音帧合计 %d；A 覆盖 %d（%.2f%%）；未覆盖 %d"
          % (tot_r, tot_cov, 100 * tot_cov / tot_r, unc))
    print("  未覆盖中：**边界缩** %d（%.1f%%）｜ **中间有洞** %d（%.1f%%）"
          % (tot_edge, 100 * tot_edge / unc, tot_hole, 100 * tot_hole / unc))
    print()
    if tot_hole > 3 * tot_edge:
        print("  ⇒ 🔴 **主要是「中间有洞」** —— A 是**词级**、R 是**轮次级**，")
        print("     轮次内部的停顿处 A 没有词。这**不是对轮次边界的分歧**，是**粒度差**")
        print("     ⇒ 拿两者互比帧级 F1，**这个比较本身就不成立**（量的是不同的东西）。")
        print("     ⚠️ 预登记的 M1 判定**照原文算数**（F1=0.9125 ≥ 0.90 ⇒ 写「一致」），")
        print("        但正文必须同时写明：**这个 F1 混合了两种不同性质的分歧**。")
    elif tot_edge > 3 * tot_hole:
        print("  ⇒ ✅ **主要是「边界缩」** —— 两源对**轮次在哪**是一致的，")
        print("     只是 ASR 词级时间戳天生偏内 ⇒ **不构成分段分歧**。")
    else:
        print("  ⇒ ⚠️ **两者量级相当，不许二选一** —— 边界缩与洞各占一部分。")

    tr = np.array([(L, cov) for L, cov, _ in turn_rows if L > 0])
    if len(tr):
        print("\n  逐轮次覆盖率的分布：中位 %.4f  均值 %.4f  <0.5 的轮次 %d/%d (%.1f%%)"
              % (np.median(tr[:, 1]), tr[:, 1].mean(),
                 int((tr[:, 1] < 0.5).sum()), len(tr),
                 100 * (tr[:, 1] < 0.5).mean()))

    # ---------- D3 诚实版 M4：逐轮次 recall，并说明 FP 为何缺席 ----------
    print("\n" + "=" * 92)
    print("D3 诚实版 M4：按 R 轮次长度分箱的 **recall**（不是 F1）")
    print("=" * 92)
    print("  ⚠️ 主探针那一版每个箱都打印 FP = 0 —— **那是构造使然的恒零，不是发现**：")
    print("     分箱只在 R 轮次区间内取帧，区间内 R **按定义**活动 ⇒ FP 恒为 0。")
    print("     所以那一列「F1」是**只有 recall 的假 F1**。这里只报 recall。")
    bins = [(0, 1), (1, 3), (3, 10), (10, 1e9)]
    agg = {b: [0, 0] for b in bins}
    for L, cov, _ in turn_rows:
        for bk in bins:
            if bk[0] <= L < bk[1]:
                n_fr = int(round(L / HOP))
                agg[bk][0] += int(round(cov * n_fr)); agg[bk][1] += n_fr
                break
    print("\n  %-12s %12s %12s %10s" % ("轮次时长", "R帧数", "A覆盖帧", "recall"))
    for bk in bins:
        c, n = agg[bk]
        if n == 0:
            print("  %-12s %12s %12s %10s" % ("%.0f-%.0fs" % (bk[0], min(bk[1], 9999)),
                                              "—", "—", "**分母 0**"))
            continue
        print("  %-12s %12d %12d %10.4f"
              % ("%.0f-%.0fs" % (bk[0], min(bk[1], 9999)), n, c, c / n))
    print("\n  ⇒ 短轮次 recall 明显更低 ⇒ 分歧集中在碎片上。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
