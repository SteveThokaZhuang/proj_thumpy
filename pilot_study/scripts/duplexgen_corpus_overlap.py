"""§5.2c：把「数字拼接 / 重叠占比」从 6 个对话做成**语料级 base rate**。

判据见 docs/pilot_study/2026-09-23_duplexgen_humdial_recon.md §5.2c（**跑之前写死的**）。
本脚本只实现它，不含任何判据选择。

仪器 = **样本 ≠ 0 即有人**。零参数、零阈值。
（`duplexgen_overlap_probe.py` 的 Otsu 版本已证伪，见 claims_ledger §四.21。）

## 两个省时间的实现选择

1. **抽样按 tar 走，不按对话走**：tar 的整体扫描是主要成本 —— 一个 1.4 GB 的分片，
   不管抽里面 1 个成员还是全部成员，都要读完整个归档。所以一次 `tar -xf` 抽**一个 tar
   的全部** `dialogue.wav`，而不是为了配样本量东抽一个西抽一个。
2. **用 `--wildcards` 一次成型**，不先 `tar -tf` 列目录 —— 那会让每个分片被扫两遍。

用法 (fd_analysis, 0 GPU, 必须经 srun --overlap):
  python duplexgen_corpus_overlap.py --n-tars 3 --out <json>
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


def measure(path):
    d, sr = read_wav(path)
    out = {"sr": sr, "n_ch": d.shape[1], "dur_s": len(d) / sr}
    if d.shape[1] != 2:
        return out                              # §5.2c: 剔除并计数, 不猜哪两个声道
    L, R = d[:, 0], d[:, 1]
    nl, nr = L != 0.0, R != 0.0
    both = nl & nr
    # 🔴 第一版这里写的是 `one_zero = float((~both).mean())` 当作"主读数 1"，
    #    而它与"主读数 2"(overlap) **恒等互补** —— 两条"独立读数"其实是同一个量。
    #    于是 P1(one_zero≥99%) ⟺ P2(overlap≤1%)，不可能一真一假 (冒烟即暴露)。
    #    P1 真正想问的是「静止声道是不是**精确的零**」，即有没有混响/串音/本底噪声，
    #    那与重叠无关 ⇒ 换成下面这两个量 (事后更正, 已在文档里标明)。
    nz = np.abs(d[d != 0.0])
    out.update(
        l_nonzero=float(nl.mean()), r_nonzero=float(nr.mean()),
        l_zero=float((~nl).mean()),             # 主读数 1': 该声道数字静音的比例
        r_zero=float((~nr).mean()),
        both_zero=float((~nl & ~nr).mean()),    # 双方都不出声(真静音)
        overlap=float(both.mean()),             # 主读数 2 (注意: = 1 - l_zero - r_zero + both_zero)
        min_abs_nonzero=float(nz.min()) if nz.size else 0.0,
        frac_below_m60=float((nz < 1e-3).mean()) if nz.size else 0.0,  # 有声样本里低于 -60dBFS 的
        corr=float(np.corrcoef(L, R)[0, 1]) if nl.any() and nr.any() else float("nan"),
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tars", type=int, default=3,
                    help="每个场景取前 N 个 tar (§5.2c 预登记写死为 3)")
    ap.add_argument("--scenarios", default=",".join(SCEN),
                    help="只跑这些场景 (冒烟用; 正式跑用默认全部)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    scen = [s for s in a.scenarios.split(",") if s]

    meta = {}
    with open(os.path.join(DS, "metadata.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            meta[r["key"]] = r

    recs, skipped = [], []
    for sc in scen:
        for i in range(a.n_tars):
            tar = os.path.join(DS, "shards", sc, "%s-%05d.tar" % (sc, i))
            if not os.path.exists(tar):
                skipped.append({"tar": tar, "why": "分片不存在"})
                print("  ⚠️ 缺分片 %s" % tar); continue
            tmp = tempfile.mkdtemp(prefix="dgco_")
            try:
                # 一次扫描抽全部分片的 dialogue.wav
                r = subprocess.run(
                    ["tar", "-xf", tar, "-C", tmp, "--wildcards",
                     "*/dialogues/dialogue.wav"],
                    stderr=subprocess.PIPE)
                if r.returncode != 0:
                    skipped.append({"tar": tar, "why": "tar 解包失败: %s"
                                    % r.stderr.decode()[:200]})
                    print("  ⚠️ 解包失败 %s" % tar); continue
                # 递归找, **不假定 key 是固定层数** (本项目的「凭记忆拼路径」家族)。
                # 只认父目录叫 dialogues 的那个。
                paths = []
                for dirpath, _dirs, files in os.walk(tmp):
                    if os.path.basename(dirpath) == "dialogues" \
                            and "dialogue.wav" in files:
                        paths.append(os.path.join(dirpath, "dialogue.wav"))
                paths.sort()
                n_ok = 0
                for p in paths:
                    key = p[len(tmp) + 1:].rsplit("/dialogues/", 1)[0]
                    try:
                        m = measure(p)
                    except Exception as e:                  # 坏文件要看得见
                        skipped.append({"key": key, "why": "读失败: %r" % (e,)})
                        continue
                    m["key"] = key
                    m["scenario"] = sc
                    md = meta.get(key, {})
                    m["n_backchannels"] = md.get("n_backchannels")
                    m["meta_dur_s"] = md.get("duration")
                    recs.append(m)
                    n_ok += 1
                print("  %s %s: %d 条" % (sc, os.path.basename(tar), n_ok))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

    # ---- 汇总 (只报预登记里的三个主读数, 不另挑统计量) ----
    two = [r for r in recs if r["n_ch"] == 2]
    good = [r for r in two if not (r["corr"] != r["corr"]) and abs(r["corr"]) <= 0.5]
    drop_ch = [r["key"] for r in recs if r["n_ch"] != 2]
    drop_corr = [r["key"] for r in two
                 if (r["corr"] != r["corr"]) or abs(r["corr"]) > 0.5]
    ov = np.array([r["overlap"] for r in good])
    lz = np.array([r["l_zero"] for r in good])
    rz = np.array([r["r_zero"] for r in good])
    minnz = np.array([r["min_abs_nonzero"] for r in good])
    fb60 = np.array([r["frac_below_m60"] for r in good])
    nbc = np.array([r["n_backchannels"] if r["n_backchannels"] is not None else -1
                    for r in good], dtype=float)
    ovs = ov * np.array([r["dur_s"] for r in good])

    print("\n===== §5.2c 汇总 =====")
    print("总条数 %d；可用(2ch 且 |corr|<=0.5) %d；声道数!=2 剔除 %d；相关超限剔除 %d"
          % (len(recs), len(good), len(drop_ch), len(drop_corr)))
    print("跳过/失败 %d 条: %s" % (len(skipped), skipped[:5]))
    if len(good):
        # 主读数 1'（**事后更正**，见 §5.2c：原来的 one_zero 与 overlap 恒等互补）
        print("\n[P1'] 「数字静音」结构（问的是有没有混响/串音/本底噪声，与重叠无关）:")
        print("     每声道精确零的样本占比  中位 L=%.1f%%  R=%.1f%%"
              % (100 * np.median(lz), 100 * np.median(rz)))
        print("     有声样本里 < -60 dBFS 的占比  中位 %.4f%%  最大 %.3f%%"
              % (100 * np.median(fb60), 100 * np.max(fb60)))
        print("     最小非零幅度  中位 %.3e  (= 1 个 16bit 量化步长 = %.3e)"
              % (np.median(minnz), 1 / 32768))
        # 🔴 判读必须**算出来**，不许写死。第一版这里印的是一句硬编码的
        #    「几乎无 -60dBFS 以下的样本 ⇒ 拼接无本底噪声」，而它正上方的数字
        #    是 12%（最大 21.9%）—— **硬编码的结论与同一屏的现算量直接矛盾**，
        #    而且因为它长得像"判读"就很容易被当成结论读走。老账
        #    `hardcoded-conclusions-escape-reproduction` 家族，第九面。
        verdict = []
        verdict.append("精确零占比 %s（判据 ≥20%%）"
                       % ("✅" if np.median(lz) >= 0.20 and np.median(rz) >= 0.20 else "🔴"))
        verdict.append("最小非零幅度 %s（判据 ≈1 个量化步长）"
                       % ("✅" if 0.5e-4 < np.median(minnz) < 2e-4 else "🔴"))
        print("     ⇒ 逐条判读: " + "; ".join(verdict))
        print("     ⚠️ 注意 < -60dBFS 那一列**不构成判据** —— 见文档 §5.2c 的更正 2:")
        print("        清语音本身就有大量低电平样本(擦音/衰减尾), 12%% 属正常;")
        print("        原判据「<1%%」是拍的, 没有任何零模型支撑。")
        print("\n[P2] 主读数2「重叠占比」: 中位 %.3f%%  四分位 [%.3f%%, %.3f%%]  最大 %.3f%%"
              % (100 * np.median(ov), 100 * np.percentile(ov, 25),
                 100 * np.percentile(ov, 75), 100 * ov.max()))
        print("     判据 P2 = 中位落在 [1%%,10%%] 且 无一条 >30%%  ⇒ %s"
              % ("✅" if (0.01 <= np.median(ov) <= 0.10 and ov.max() <= 0.30) else "🔴 未达"))
        m = nbc >= 0
        if m.sum() > 1:
            # 手写 Spearman: scipy 有, 但这里避免再引一个依赖, 排序秩即可
            def rank(x):
                o = np.argsort(x); rk = np.empty(len(x)); rk[o] = np.arange(len(x))
                return rk
            rho = float(np.corrcoef(rank(ovs[m]), rank(nbc[m]))[0, 1])
            print("\n[P3] 重叠秒数 vs n_backchannels: Spearman ρ = %+.3f (n=%d)"
                  % (rho, int(m.sum())))
            print("     判据 P3 = ρ > 0  ⇒ %s" % ("✅" if rho > 0 else "🔴 未达"))
        print("\n逐场景重叠占比中位:")
        for sc in scen:
            v = np.array([r["overlap"] for r in good if r["scenario"] == sc])
            if v.size:
                print("   %s  n=%4d  中位 %.3f%%  最大 %.3f%%"
                      % (sc, v.size, 100 * np.median(v), 100 * v.max()))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"records": recs, "skipped": skipped,
                   "dropped_n_ch": drop_ch, "dropped_corr": drop_corr,
                   "n_tars_per_scenario": a.n_tars}, f)
    print("\n写出 %s" % a.out)


if __name__ == "__main__":
    main()
