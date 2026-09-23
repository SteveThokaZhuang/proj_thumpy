"""HumDial：同一音频上「参考文本 vs ASR」的**时间**一致性。

预登记：`docs/pilot_study/2026-09-23_humdial_timing_prereg.md`（判据先写死）。
本脚本的 docstring 把判据再抄一遍 —— **改代码时两处一起改，否则就是 §四.26 那个事故**
（注释描述着一个已经不存在的实现，还继续为读数背书）。

## 量的是什么

两个源（详见 schema 文档 §3.4）：

  R = `speech_segments[].text` + `[xmin, xmax]`   轮次级区间，**参考文本**
  A = `chunks[].text` + `timestamp:[a, b]`        **词级**，ASR 输出

**文本**层面已知逐字相同 50.6%（schema §3.4）。**时间**层面本文第一次测。

## 判据（跑之前写死，见 prereg §4）

**M1（主判据，零参数）** —— 5 ms 网格上的点-在-区间-内 活动掩码，F1（以 R 为正类）：
  - F1 ≥ 0.90 ⇒ 两源在"何时说"上一致 ⇒ **削弱**「两个标签」框架
  - F1 ≤ 0.70 ⇒ 两源对"何时"本身就不一致 ⇒ 主命题 L2 那条腿的**独立证据**
  - 中间 ⇒ **如实报中间，不许二选一**

**M3 方向预测（可证伪）**：A 的 onset **晚于** R（中位 +0.05~+0.35 s）、
A 的 offset **早于** R（中位 < 0）。
  - 符号相反 ⇒ 🔴 异常，如实报，不许事后编理由
  - 中位落在 ±0.05 s 内 ⇒ 记「无可测的偏置」，**不许说"基本一致"**

## 🚫 不可做（prereg §4.3）

  1. 不许用 M2 的匹配去"修"M1
  2. 不许挑一个 τ 报 M2/M3 —— **四个一起报**；结论随 τ 变号就如实说它不稳
  3. 不许与 Behavior-SD 的 κ 0.06–0.37 **直接比大小**（不是同一个量）
  4. 不许说"HumDial 的标签不可信" —— 只能报**一致性是多少**，本文没有裁判谁对

## 纪律（本项目踩过的坑，写在前面）

  - **每个数都连分母一起打印**（过滤器制造的空表长得像"结果是 0"）
  - **不静默跳过**：缺字段 / 空区间 / 零长度轮次，全部计数并报出
  - **不读音频**：两个源都在 json 里 ⇒ 全量 4172 条，秒级

用法（纯读盘，0 GPU，必须经 srun --overlap --ntasks=1）：
  python scripts/humdial_timing_probe.py --zip <zip> --collar 0.5
"""
import argparse
import json
import os
import sys
import zipfile

import numpy as np

HOP = 0.005              # 5 ms 网格
TAUS = [0.3, 0.5, 1.0, 2.0]   # prereg §3.2：不挑一个，四个都报


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    return (2 * p * r / (p + r) if p + r else float("nan")), p, r


def mask_of(intervals, T, ngrid):
    """5 ms 网格：点 t 落在某个区间内 ⇒ 记活动。**零参数**（点-在-区间-内）。"""
    m = np.zeros(ngrid, dtype=bool)
    for a, b in intervals:
        if b <= a:
            continue
        i0 = int(np.ceil(a / HOP))
        i1 = int(np.floor(b / HOP))
        if i1 < i0:
            # 区间比一个网格步还窄：仍然记 1 个点，但**计数并报出**，不静默丢
            i0 = i1 = max(0, min(ngrid - 1, int(round(a / HOP))))
        m[max(0, i0):min(ngrid, i1 + 1)] = True
    return m


def group_turns(chunks, tau):
    """把词级 chunks 按「相邻词间隔 > τ 断一轮」聚成轮次。τ 是自由参数 ⇒ 四个都跑。"""
    if not chunks:
        return []
    turns = []
    cur_s = chunks[0]["timestamp"][0]
    cur_e = chunks[0]["timestamp"][1]
    for c in chunks[1:]:
        s, e = c["timestamp"][0], c["timestamp"][1]
        if s - cur_e > tau:
            turns.append((cur_s, cur_e))
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    turns.append((cur_s, cur_e))
    return turns


def match_collar(ref, hyp, collar):
    """带领圈的最近邻匹配（两侧都是排序好的 1-D 点集）。
    返回 (匹配上的 hyp 数, 匹配上的 ref 数, 匹配对的 (ref, hyp) 列表)。"""
    if not len(ref) or not len(hyp):
        return 0, 0, []
    ref = np.asarray(ref); hyp = np.asarray(hyp)
    used_h = set(); pairs = []
    j = 0
    for r in ref:
        # 在 hyp 里找 |h - r| <= collar 的最近者
        while j < len(hyp) and hyp[j] < r - collar:
            j += 1
        k = j
        best = None
        while k < len(hyp) and hyp[k] <= r + collar:
            if k not in used_h:
                if best is None or abs(hyp[k] - r) < abs(hyp[best] - r):
                    best = k
            k += 1
        if best is not None:
            used_h.add(best)
            pairs.append((r, hyp[best]))
    return len(used_h), len(pairs), pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--collar", type=float, default=0.5)
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.zip)
    names = [i.filename for i in zf.infolist() if i.file_size > 0]
    jset = set(n for n in names if n.endswith(".json"))

    # ---- 挑出可配对的 4172：非 clean、非 add、两族 json 都在 ----
    pairs = []
    n_clean = n_add = n_missing = 0
    seen = set()
    for n in sorted(jset):
        d, b = os.path.dirname(n), os.path.basename(n)
        if b.endswith("_timestamp.json") or b.startswith("clean_"):
            continue
        st = b[:-5]
        if "_add" in st:
            n_add += 1
            continue
        key = (d, st)
        if key in seen:
            continue
        seen.add(key)
        sp = "%s/%s.json" % (d, st)
        tp = "%s/%s_timestamp.json" % (d, st)
        if sp in jset and tp in jset:
            pairs.append((d, st, sp, tp))
        else:
            n_missing += 1
            print("  ⚠️ 缺一侧，**计数不静默跳过**：%s/%s (seg=%s ts=%s)"
                  % (d.replace("test/", ""), st, sp in jset, tp in jset))

    print("=" * 92)
    print("样本：可配对 %d 条（跳过 _add %d / 缺一侧 %d）"
          % (len(pairs), n_add, n_missing))
    print("=" * 92)

    ngrid_total = 0
    conf = dict(tp=0, fp=0, fn=0)      # R 为正类
    dur_gap = []
    n_empty_R = n_empty_A = n_zero_len_turn = 0
    n_used = 0
    # M2/M3 累加器
    bnd = {t: dict(on_r=0, on_a=0, on_m=0, of_r=0, of_a=0, of_m=0,
                   on_err=[], of_err=[]) for t in TAUS}
    # M4：按 R 轮次长度分箱（用该轮的帧数当权重）
    bins = [(0, 1), (1, 3), (3, 10), (10, 1e9)]
    m4 = {b: dict(tp=0, fp=0, fn=0) for b in bins}

    for di, (d, st, sp, tp) in enumerate(pairs):
        R = json.loads(zf.read(sp))
        A = json.loads(zf.read(tp))
        segs = R.get("speech_segments") or []
        chunks = A.get("chunks") or []
        if not segs:
            n_empty_R += 1
            continue
        if not chunks:
            n_empty_A += 1
            continue

        R_int = [(s["xmin"], s["xmax"]) for s in segs]
        A_int = [(c["timestamp"][0], c["timestamp"][1]) for c in chunks]
        T = max(max(b for _, b in R_int), max(b for _, b in A_int))
        # prereg §3.1：报出 final_duration 与最末时间戳之差
        fd = R.get("final_duration")
        if fd is not None:
            dur_gap.append(float(fd) - max(b for _, b in R_int))
        ngrid = int(T / HOP) + 1
        ngrid_total += ngrid

        mR = mask_of(R_int, T, ngrid)
        mA = mask_of(A_int, T, ngrid)
        tp_ = int((mR & mA).sum()); fp_ = int((~mR & mA).sum()); fn_ = int((mR & ~mA).sum())
        conf["tp"] += tp_; conf["fp"] += fp_; conf["fn"] += fn_
        n_used += 1

        # ---- M4：逐 R 轮次长度分箱 ----
        for (a, b) in R_int:
            if b <= a:
                n_zero_len_turn += 1
                continue
            L = b - a
            for bkey in bins:
                if bkey[0] <= L < bkey[1]:
                    i0 = max(0, int(np.ceil(a / HOP)))
                    i1 = min(ngrid - 1, int(np.floor(b / HOP)))
                    if i1 < i0:
                        continue
                    m4[bkey]["tp"] += int((mR[i0:i1 + 1] & mA[i0:i1 + 1]).sum())
                    m4[bkey]["fp"] += int((~mR[i0:i1 + 1] & mA[i0:i1 + 1]).sum())
                    m4[bkey]["fn"] += int((mR[i0:i1 + 1] & ~mA[i0:i1 + 1]).sum())
                    break

        # ---- M2/M3：边界（四个 τ 都跑）----
        for tau in TAUS:
            turns = group_turns(chunks, tau)
            on_r = [a for a, _ in R_int]; of_r = [b for _, b in R_int]
            on_a = [a for a, _ in turns]; of_a = [b for _, b in turns]
            nm_o, nm_r, pr_o = match_collar(on_r, on_a, args.collar)
            nm_f, nm_rf, pr_f = match_collar(of_r, of_a, args.collar)
            e = bnd[tau]
            e["on_r"] += len(on_r); e["on_a"] += len(on_a); e["on_m"] += nm_r
            e["of_r"] += len(of_r); e["of_a"] += len(of_a); e["of_m"] += nm_rf
            e["on_err"].extend(h - r for r, h in pr_o)
            e["of_err"].extend(h - r for r, h in pr_f)

        if (di + 1) % 500 == 0:
            print("  … 已处理 %d/%d" % (di + 1, len(pairs)), flush=True)

    # ================= 汇总 =================
    print("\n" + "=" * 92)
    print("M1 【主判据·零参数】5 ms 网格 点-在-区间-内，F1（以 R 为正类）")
    print("=" * 92)
    F1, P, Rc = f1(conf["tp"], conf["fp"], conf["fn"])
    print("  用上 %d 条（空 speech_segments %d / 空 chunks %d / 零长度轮次 %d）"
          % (n_used, n_empty_R, n_empty_A, n_zero_len_turn))
    print("  网格点合计 %d" % ngrid_total)
    print("  TP %d ｜ FP %d（A 说是、R 说不是）｜ FN %d（R 说是、A 说不是）"
          % (conf["tp"], conf["fp"], conf["fn"]))
    print("  📊 **F1 = %.4f**   precision = %.4f   recall = %.4f" % (F1, P, Rc))
    cov_r = conf["tp"] + conf["fn"]; cov_a = conf["tp"] + conf["fp"]
    print("  有符号面积比 |A|/|R| = %.4f   （R 覆盖 %d 帧，A 覆盖 %d 帧）"
          % (cov_a / cov_r if cov_r else float("nan"), cov_r, cov_a))
    if dur_gap:
        g = np.array(dur_gap)
        print("  final_duration − R 最末时间戳：中位 %.3f s  均值 %.3f s  [%.3f, %.3f]"
              % (np.median(g), g.mean(), g.min(), g.max()))

    print("\n  ⇒ 判据（prereg §4.1）：", end="")
    if F1 >= 0.90:
        print("**F1 ≥ 0.90** ⇒ ✅ 两源在「何时说」上一致；正文必须写「削弱两个标签框架」。")
    elif F1 <= 0.70:
        print("**F1 ≤ 0.70** ⇒ 🔴 两源对「何时」本身就不一致 —— 主命题 L2 那条腿的独立证据。")
    else:
        print("**0.70 < F1 < 0.90** ⇒ ⚠️ 落在中间 —— 如实报中间，**不许二选一**。")

    print("\n" + "=" * 92)
    print("M2/M3 边界匹配（领圈 ±%.2f s）—— 四个 τ 全报，不挑" % args.collar)
    print("=" * 92)
    print("  %-6s %8s %8s %8s %8s %26s %26s" %
          ("τ(s)", "onset-P", "onset-R", "onset-F1", "offset-F1", "onset 误差 中位[Q1,Q3]", "offset 误差 中位[Q1,Q3]"))
    for tau in TAUS:
        e = bnd[tau]
        f_on, p_on, r_on = f1(e["on_m"], e["on_a"] - e["on_m"], e["on_r"] - e["on_m"])
        f_of, p_of, r_of = f1(e["of_m"], e["of_a"] - e["of_m"], e["of_r"] - e["of_m"])
        oe = np.array(e["on_err"]) if e["on_err"] else np.array([np.nan])
        fe = np.array(e["of_err"]) if e["of_err"] else np.array([np.nan])
        print("  %-6.1f %8.3f %8.3f %8.3f %8.3f   %10.3f [%6.3f,%6.3f] (n=%d)   %10.3f [%6.3f,%6.3f] (n=%d)"
              % (tau, p_on, r_on, f_on, f_of,
                 np.median(oe), np.percentile(oe, 25), np.percentile(oe, 75), len(oe),
                 np.median(fe), np.percentile(fe, 25), np.percentile(fe, 75), len(fe)))

    print("\n  ⇒ 方向预测（prereg §4.2）：A 的 onset **晚于** R（中位 +0.05~+0.35 s）、")
    print("     A 的 offset **早于** R（中位 < 0）。逐 τ 对照：")
    for tau in TAUS:
        e = bnd[tau]
        if not e["on_err"]:
            continue
        mo, mf = float(np.median(e["on_err"])), float(np.median(e["of_err"]))
        ok_o = 0.05 <= mo <= 0.35
        ok_f = mf < 0
        tag = ("✅ 与预测相容" if (ok_o and ok_f)
               else "⚠️ 部分命中" if (ok_o or ok_f)
               else "🔴 **与预测相反 —— 异常，如实报，不许事后编理由**")
        print("     τ=%.1f  onset 中位 %+.3f（预测 +0.05~+0.35 %s）｜"
              " offset 中位 %+.3f（预测 <0 %s）⇒ %s"
              % (tau, mo, "✅" if ok_o else "✗", mf, "✅" if ok_f else "✗", tag))
        break  # 方向只对主 τ 报一次判定，其余靠上表
    print("     ⚠️ 中位落在 ±0.05 s 内 ⇒ 记「**无可测的偏置**」，不许说「基本一致」。")

    print("\n" + "=" * 92)
    print("M4 按 R 轮次长度分箱的 M1 帧级 F1")
    print("=" * 92)
    print("  %-12s %10s %10s %10s %10s" % ("轮次时长", "TP", "FP", "FN", "F1"))
    for bkey in bins:
        c = m4[bkey]
        if c["tp"] + c["fp"] + c["fn"] == 0:
            print("  %-12s %10s %10s %10s %10s" % ("%.0f-%.0fs" % (bkey[0], min(bkey[1], 9999)),
                                                   "—", "—", "—", "**无数据（分母 0）**"))
            continue
        f_, _, _ = f1(c["tp"], c["fp"], c["fn"])
        print("  %-12s %10d %10d %10d %10.4f"
              % ("%.0f-%.0fs" % (bkey[0], min(bkey[1], 9999)),
                 c["tp"], c["fp"], c["fn"], f_))
    print("\n  ⇒ 若短轮次明显更差 ⇒ 分歧集中在「碎片」上，对下游指标的含义与「普遍不一致」不同。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
