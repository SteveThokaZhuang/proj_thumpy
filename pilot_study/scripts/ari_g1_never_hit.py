"""#51: 为什么这 9 个块"从未命中"?

§7.4 把含 GT 的 25 块分成三层: 5 块总是命中 / 11 块时灵时不灵 (噪声来源) /
9 块 7 次重跑全做不对。前两层已解释, 第三层与种子运气无关, 更像标签或音频问题。

对照设计: 同时打印"总是命中"的 5 块。单看失败样本说不出所以然, 有对照组才能
指出**差异在哪一列上**。

对每块回答四件事:
  1. 该 chunk 覆盖到哪些 BC 事件 (含 realized=None 的), 它们落在 chunk 内什么位置
  2. 这些事件里哪些算 GT (realized != None, 与评估口径一致)
  3. 模型 7 个种子分别预测了几个 (全 0 = 系统性失败, 而非运气)
  4. own 声道音频在 GT 位置到底有没有能量 —— 区分"标签错"和"音频里没有"

用法: python ari_g1_never_hit.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_g1_common import (ANNOT, bc_windows, eval_chunk_ids,  # noqa: E402
                           load_manifest)

OWN = f"{ANNOT}/g1_own_eval/own"
MIX = f"{ANNOT}/g1_sub/mixnorm"

NEVER = [
    "0020a0c5_ch1_t1710", "01a4c01c_ch0_t1705", "01e64778_ch1_t50",
    "02053f7c_ch1_t235", "0480a711_ch0_t400", "0736fa95_ch1_t1105",
    "0777bb85_ch1_t860", "0d4a795e_ch1_t540", "0f918385_ch0_t225",
]

# 每个种子一份评估结果 (标签 -> 文件)
SEEDS = {
    "s42":    f"{ANNOT}/g1_eval_own10.json",
    "s7":     f"{ANNOT}/g1_eval_own10_s7.json",
    "s1234":  f"{ANNOT}/g1_eval_own10_s1234.json",
    "s2024":  f"{ANNOT}/g1_eval_own10_s2024.json",
    "s3407":  f"{ANNOT}/g1_eval_own10_s3407.json",
    "s31337": f"{ANNOT}/g1_eval_own10_s31337.json",
    "s55555": f"{ANNOT}/g1_eval_own10_s55555.json",
}


def load_seed(path):
    d = json.load(open(path))
    return next(iter(d.values()))["per_chunk"]


def frame_rms(audio, sr=16000, win=0.05):
    """50ms 窗 RMS 包络, 返回 (times, rms)."""
    n = int(sr * win)
    k = len(audio) // n
    a = audio[:k * n].reshape(k, n)
    return (np.arange(k) + 0.5) * win, np.sqrt((a ** 2).mean(axis=1))


def main():
    mlook_own = load_manifest(OWN)
    mlook_mix = load_manifest(MIX)
    ids = set(eval_chunk_ids(300))

    # ── 建立三层分组的并集: 每块在 7 个种子里命中过几次 ──────────────────
    per = {t: load_seed(p) for t, p in SEEDS.items()}
    gt_chunks = [c for c in sorted(ids)
                 if per["s42"].get(c, {}).get("n_gt", 0) > 0]
    hit_count = {}
    for c in gt_chunks:
        hit_count[c] = sum(1 for t in per if per[t][c]["tp"] > 0)

    always = sorted([c for c in gt_chunks if hit_count[c] == len(per)])
    never = sorted([c for c in gt_chunks if hit_count[c] == 0])
    swing = sorted([c for c in gt_chunks if 0 < hit_count[c] < len(per)])

    print(f"含 GT 的块 {len(gt_chunks)}: 总是命中 {len(always)} / "
          f"时灵时不灵 {len(swing)} / 从未命中 {len(never)}")
    print(f"（本脚本写死的 NEVER 列表与实测 never 是否一致: "
          f"{sorted(NEVER) == never}）")

    # ── 事件表: 一次性建索引 ─────────────────────────────────────────────
    D = f"{ANNOT}/../analysis/ari"
    import glob
    e1 = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{D}/candor_e1_w*.csv"))])
    ev = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{D}/candor_events_w*.csv"))])
    # ev 提供 start/end/dur/能量类; max_span/both_active_frac 本来就在 e1 里,
    # 从 ev 再取会撞名 (e1 无 start/end, 所以这两列合并后就是 ev 的)
    e1 = e1.merge(ev[["session", "event_id", "ch_event", "ch_other",
                      "start", "end", "dur", "energy_ratio", "voiced_ratio"]],
                  on="event_id", suffixes=("", "_ev"))
    e1["session"] = e1["session"].astype(str)
    e1["ch_event"] = e1["ch_event"].astype(int)

    def events_in(m, chunk_s=10.0):
        """chunk 内所有 BC 事件 (不过滤 realized) 的相对区间."""
        sub = e1[(e1["session"] == str(m["session"]))
                 & (e1["ch_event"] == int(m["ch"]))
                 & (e1["cls"] == "BC")]
        out = []
        for _, r in sub.iterrows():
            if r["end"] > m["t0"] and r["start"] < m["t0"] + chunk_s:
                out.append({
                    "rel_start": round(r["start"] - m["t0"], 2),
                    "rel_end": round(r["end"] - m["t0"], 2),
                    "realized": r["realized"],
                    "dur": round(float(r["dur"]), 2),
                    "energy_ratio": round(float(r["energy_ratio"]), 3),
                    "voiced_ratio": round(float(r["voiced_ratio"]), 3),
                    "max_span": round(float(r["max_span"]), 2),
                    "both_active_frac": round(float(r["both_active_frac"]), 3),
                })
        return sorted(out, key=lambda x: x["rel_start"])

    def audio_probe(cid, lo, hi, chunk_s=10.0, sr=16000):
        """own 声道在 [lo,hi) 相对区间内 vs 整块 的 RMS。"""
        f = f"{OWN}/{cid}.npz"
        if not os.path.exists(f):
            return None
        a = np.load(f)["audio"]
        t, r = frame_rms(a, sr)
        lo_i = max(0, int(lo / 0.05))
        hi_i = min(len(r), max(lo_i + 1, int(hi / 0.05)))
        seg = r[lo_i:hi_i]
        return {
            "rms_win": round(float(seg.mean()), 5) if len(seg) else 0.0,
            "rms_win_max": round(float(seg.max()), 5) if len(seg) else 0.0,
            "rms_chunk": round(float(r.mean()), 5),
            "rms_chunk_max": round(float(r.max()), 5),
            "peak_t": round(float(t[int(np.argmax(r))]), 2),
        }

    def report(title, cids):
        print("\n" + "=" * 78)
        print(f"  {title}  (n={len(cids)})")
        print("=" * 78)
        for cid in cids:
            m = mlook_own.get(cid) or mlook_mix.get(cid)
            preds = {t: per[t][cid]["n_pred"] for t in per}
            gtw = bc_windows(cid, mlook_own)
            evs = events_in(m)
            print(f"\n▌ {cid}")
            print(f"   session {str(m['session'])[:8]}  ch{m['ch']}  t0={m['t0']}")
            print(f"   预测数/种子: " +
                  " ".join(f"{t}={v}" for t, v in preds.items()))
            print(f"   GT 窗口 (realized!=None): "
                  f"{[(round(a,2), round(b,2)) for a, b in gtw]}")
            if not evs:
                print("   ⚠ 该 (session,ch) 在 chunk 内没有任何 BC 事件行")
            for e in evs:
                mark = "GT" if e["realized"] != "None" else "  "
                pr = audio_probe(cid, e["rel_start"], e["rel_end"])
                rms = (f"ownRMS {pr['rms_win']:.5f} / 块均 {pr['rms_chunk']:.5f}"
                       f" (块峰@{pr['peak_t']}s)" if pr else "no audio")
                print(f"     [{mark}] {e['rel_start']:6.2f}-{e['rel_end']:6.2f}s "
                      f"realized={e['realized']:<12} dur={e['dur']:.2f} "
                      f"er={e['energy_ratio']:.3f} vr={e['voiced_ratio']:.3f} "
                      f"span={e['max_span']:.2f}")
                print(f"           {rms}")

    report("A. 从未命中", never)
    report("B. 总是命中（对照组）", always)

    print("\n" + "=" * 78)
    print("  汇总: 音频可闻性")
    print("=" * 78)
    print("\n(块内 GT 窗口 RMS / 全块 RMS 的比值; <1 说明该窗口比平均更安静)\n")
    for label, cids in [("从未命中", never), ("总是命中", always),
                        ("时灵时不灵", swing)]:
        ratios = []
        for cid in cids:
            m = mlook_own.get(cid) or mlook_mix.get(cid)
            gtw = bc_windows(cid, mlook_own)
            pr_ch = audio_probe(cid, 0, 10.0)
            if not gtw or not pr_ch or pr_ch["rms_chunk"] == 0:
                continue
            r = np.mean([audio_probe(cid, a, b)["rms_win"] for a, b in gtw])
            ratios.append(r / pr_ch["rms_chunk"])
        if ratios:
            print(f"  {label:<12} n={len(ratios):2d}  "
                  f"中位数 {np.median(ratios):.2f}  "
                  f"范围 {min(ratios):.2f}–{max(ratios):.2f}")


if __name__ == "__main__":
    main()
