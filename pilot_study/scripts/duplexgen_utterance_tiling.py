"""§5.2d：`utterances/*.wav` 到底是「谁」—— 用**精确样本相等**定址，不用互相关。

## 为什么要重做

§5.2 写过一句「utterance 声道占用严格交替 (0=L, 1=R, 2=L, …)」。
那是 `argmax([0,0]) → 0` 的伪影：某 utterance 在 L 上全零、在 R 上也全零时，
两条候选得分打平，`argmax` 取下标 0，于是「无法判定」被读成了「匹配 L」。

第一次尝试用互相关定址，但跨场景要 ~19,000 次 fftconvolve（几小时），停掉了。
换一个**不需要相关、不需要阈值**的判据：如果是拼接，波形会**逐样本精确相等**
（同一个 int16 序列除以同一个 32768.0），那就直接比整数。

## 判据（跑之前写死）

  T1 切片性: 每个**非全零** utterance 都能在 dialogue 的某个声道上找到一段
             **逐样本精确相等**（`np.array_equal`）的连续区段。
  T2 单侧性: 全部命中集中在**同一个**声道上（即 utterances 只承载一方）。
  T3 计数恒等: Σ_k (utt_k != 0).sum() == (承载声道 != 0).sum()，**精确整数相等**。
  T4 不重叠: 命中的区段两两不重叠（精确判据），并报出相邻区段间的空隙分布。

反面教材同族：claims_ledger §四.21（仪器选择）/ §四.22（同义读数）。

用法 (fd_analysis, 0 GPU, 必须经 srun --overlap):
  python duplexgen_utterance_tiling.py --per-scenario 4 --out <json>
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np

DS = "/share/workspace3/shared_dataset/duplexgen-spoken"
SCEN = ["INT", "NEG", "PER", "PLN", "SOC", "TEA"]


def read_wav(path):
    with wave.open(path) as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        raw = w.readframes(n)
    d = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    return d.reshape(-1, ch), sr


def locate(seg, chan):
    """seg 在 chan 里**精确相等**的所有起点下标。

    先按首样本筛候选，再用 4 个等距点向量化预筛，最后对幸存者做全等比较 ——
    全等比较是 O(n)，不能对全部候选做。
    """
    n = len(seg)
    if n < 16:
        return []
    cand = np.nonzero(chan == seg[0])[0]
    cand = cand[cand + n <= len(chan)]
    if cand.size == 0:
        return []
    ok = np.ones(cand.size, dtype=bool)
    for j in (n // 4, n // 2, (3 * n) // 4, n - 1):
        ok &= chan[cand + j] == seg[j]
    out = []
    for c in cand[ok]:
        c = int(c)
        if np.array_equal(chan[c:c + n], seg):
            out.append(c)
    return out


def probe(root):
    dlg, sr = read_wav(os.path.join(root, "dialogues", "dialogue.wav"))
    utts = sorted(glob.glob(os.path.join(root, "utterances", "*.wav")))
    n_ch = int(dlg.shape[1])
    rec = {"key": root.split("/shards/")[-1] if "/shards/" in root else root,
           "sr": sr, "n_ch_dlg": n_ch, "n_utt": len(utts),
           "dlg_len": int(len(dlg)),
           "nonzeros": [int((dlg[:, c] != 0).sum()) for c in range(n_ch)]}

    hits, n_zero, n_lost, sum_nz, utt_ch = [], 0, 0, 0, set()
    for p in utts:
        x, _ = read_wav(p)
        utt_ch.add(int(x.shape[1]))
        u = x[:, 0]
        nz = np.nonzero(u)[0]
        sum_nz += int(nz.size)
        if nz.size == 0:
            n_zero += 1
            continue
        i0 = int(nz[0])
        seg = u[i0:i0 + 512]
        found = None
        for ch in range(n_ch):
            for c in locate(seg, dlg[:, ch]):
                off = c - i0
                if off < 0 or off + len(u) > len(dlg):
                    continue
                if np.array_equal(dlg[off:off + len(u), ch], u):
                    found = (ch, off)
                    break
            if found:
                break
        if found:
            hits.append({"name": os.path.basename(p), "ch": found[0],
                         "off": found[1], "len": int(len(u))})
        else:
            n_lost += 1
            hits.append({"name": os.path.basename(p), "ch": -1,
                         "off": -1, "len": int(len(u))})
    rec.update(hits=hits, n_zero_utt=n_zero, n_unlocated=n_lost,
               sum_utt_nonzero=sum_nz, utt_n_ch=sorted(utt_ch))
    return rec


def tile_stats(hs):
    """落在同一声道上的命中按起点排序，报重叠与空隙。"""
    hs = sorted([h for h in hs if h["ch"] >= 0], key=lambda h: h["off"])
    ov, gaps = [], []
    for a, b in zip(hs, hs[1:]):
        end = a["off"] + a["len"]
        if b["off"] < end:
            ov.append(end - b["off"])
        else:
            gaps.append(b["off"] - end)
    return hs, ov, gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scenario", type=int, default=4)
    ap.add_argument("--scenarios", default=",".join(SCEN))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    scen = [s for s in a.scenarios.split(",") if s]

    recs = []
    for sc in scen:
        tar = os.path.join(DS, "shards", sc, "%s-00000.tar" % sc)
        tmp = tempfile.mkdtemp(prefix="dgut_")
        try:
            r = subprocess.run(
                ["tar", "-xf", tar, "-C", tmp, "--wildcards",
                 "*/utterances/*.wav", "*/dialogues/dialogue.wav"],
                stderr=subprocess.PIPE)
            if r.returncode != 0:
                print("  ⚠️ %s 解包失败: %s" % (tar, r.stderr.decode()[:200]))
                continue
            roots = []
            for dirpath, _d, files in os.walk(tmp):
                if os.path.basename(dirpath) == "dialogues" \
                        and "dialogue.wav" in files:
                    roots.append(os.path.dirname(dirpath))
            roots.sort()
            print("  %s: 分片内 %d 个对话, 取前 %d 个"
                  % (sc, len(roots), a.per_scenario))
            for root in roots[:a.per_scenario]:
                rec = probe(root)
                rec["scenario"] = sc
                recs.append(rec)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ---- 汇总: 判读**算出来**, 不写死 ----
    print("\n%-4s %-28s %5s %5s %5s %6s %9s %9s %s"
          % ("场景", "对话", "utt", "全零", "失配", "命中ch", "Σutt非零",
             "该声道非零", "T3"))
    t1_ok = t2_ok = t3_ok = t4_ok = 0
    all_ov, all_gap = [], []
    for r in recs:
        hs, ov, gaps = tile_stats(r["hits"])
        chs = sorted(set(h["ch"] for h in hs))
        carrier = chs[0] if len(chs) == 1 else -1
        t1 = (r["n_unlocated"] == 0)
        t2 = (len(chs) == 1 and r["n_unlocated"] == 0)
        t3 = (carrier >= 0 and r["sum_utt_nonzero"] == r["nonzeros"][carrier])
        t4 = (len(ov) == 0)
        t1_ok += t1
        t2_ok += t2
        t3_ok += t3
        t4_ok += t4
        all_ov += ov
        all_gap += gaps
        print("%-4s %-28s %5d %5d %5d %6s %9d %9d %s"
              % (r["scenario"], r["key"].split("/")[-2] + "/"
                 + r["key"].split("/")[-1], r["n_utt"], r["n_zero_utt"],
                 r["n_unlocated"], ("L" if carrier == 0 else
                                    "R" if carrier == 1 else "混合")
                 if carrier >= 0 else "—",
                 r["sum_utt_nonzero"],
                 r["nonzeros"][carrier] if carrier >= 0 else -1,
                 "✅" if t3 else "🔴"))

    n = len(recs)
    print("\n===== §5.2d 汇总 (n=%d 对话) =====" % n)
    print("T1 切片性(全部 utterance 都精确定位)      %2d/%d  %s"
          % (t1_ok, n, "✅" if t1_ok == n else "🔴"))
    print("T2 单侧性(命中只落在一个声道)             %2d/%d  %s"
          % (t2_ok, n, "✅" if t2_ok == n else "🔴"))
    print("T3 计数恒等(Σutt非零 == 该声道非零, 精确) %2d/%d  %s"
          % (t3_ok, n, "✅" if t3_ok == n else "🔴"))
    print("T4 不重叠(命中区段两两不重叠)             %2d/%d  %s"
          % (t4_ok, n, "✅" if t4_ok == n else "🔴"))
    if all_ov:
        print("   ⚠️ 重叠区段数 %d, 重叠样本中位 %d" % (len(all_ov),
                                                int(np.median(all_ov))))
    if all_gap:
        g = np.array(all_gap)
        print("   相邻区段空隙: n=%d  中位 %d 样本 (%.3f s)  P90 %d  最大 %d"
              % (g.size, int(np.median(g)), np.median(g) / recs[0]["sr"],
                 int(np.percentile(g, 90)), int(g.max())))
    else:
        print("   相邻区段空隙: 无")
    tot_utt = sum(r["n_utt"] for r in recs)
    tot_zero = sum(r["n_zero_utt"] for r in recs)
    print("全零 utterance: %d/%d (%.1f%%)" % (tot_zero, tot_utt,
                                          100.0 * tot_zero / max(tot_utt, 1)))
    print("utterance 自身声道数: %s"
          % sorted(set(c for r in recs for c in r["utt_n_ch"])))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"records": recs, "per_scenario": a.per_scenario}, f)
    print("\n写出 %s" % a.out)


if __name__ == "__main__":
    main()
