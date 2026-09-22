"""G1 操纵检验: own 声道里到底能听到多少"对方"?

消融的逻辑前提是: BC 依赖的"对方还在说"这个变量, 在 own 单声道观测空间里
几乎不可得。但 CANDOR 是双麦录制, 对方的声音可能以串扰泄漏进本人声道。
若泄漏很强, mixnorm 相对 own 的边际信息就很小, 消融是个弱检验。

为什么按 **session 级** 汇总而不是按 chunk: X2-Turn 的 p_speak 极度稀疏
(实测中位数 0, p90 仅 0.04), 约 90% 的帧被判双方静默。10s chunk 内常常只
够一类说话帧, 按 chunk 过滤会丢掉 ~93% 的样本。按 session 汇总则每段有
上千个说话帧, 统计量充足。

两个测量, 各自针对一种混淆:
  A. **包络互相关** (主): 两声道来自独立录音设备, 有时钟偏移, 样本级波形
     相关必然 ~0, 不能作为"无泄漏"的证据。改用 20ms 能量的包络, 在
     ±250ms 内搜峰 —— 对时钟偏移稳健; 真串扰是对方信号的线性滤波,
     会表现为该滞后上的强正相关。
  B. **帧级能量** (辅): 只保留"仅本人说"与"仅对方说"两类帧, 比较本人声道
     在这两类帧上的能量。若"仅对方说"时本人声道并不更响, 说明对方听不见。

用法: python scripts/ari_g1_leakage.py --n-sessions 25
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_g1_prep_subset import ANNOT, CANDOR, parse_id  # noqa: E402

FRAME_S = 0.08
HOP = 320          # 20ms @16k 的包络帧移
WIN = 640          # 40ms 窗
MAX_LAG = 15       # ±300ms


def db(x):
    return 10 * np.log10(max(float(x), 1e-12))


def envelope(x):
    n = (len(x) - WIN) // HOP + 1
    idx = np.arange(n) * HOP
    e = np.array([np.mean(x[i:i + WIN] ** 2) for i in idx])
    return 10 * np.log10(e + 1e-12)


def env_xcorr(a, b):
    """包络互相关峰值与滞后 (对时钟偏移稳健)."""
    ea, eb = envelope(a), envelope(b)
    n = min(len(ea), len(eb))
    ea, eb = ea[:n], eb[:n]
    ea, eb = ea - ea.mean(), eb - eb.mean()
    sa, sb = ea.std(), eb.std()
    if sa < 1e-9 or sb < 1e-9:
        return 0.0, 0
    best, bl = -1.0, 0
    for L in range(-MAX_LAG, MAX_LAG + 1):
        x, y = (ea[L:], eb[: n - L]) if L >= 0 else (ea[: n + L], eb[-L:])
        if len(x) < 100:
            continue
        c = float(np.mean(x * y) / (sa * sb))
        if c > best:
            best, bl = c, L
    return best, bl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=f"{ANNOT}/g1_ids.txt")
    ap.add_argument("--n-sessions", type=int, default=25)
    ap.add_argument("--chunk-s", type=float, default=10.0)
    ap.add_argument("--thr", type=float, default=0.5, help="p_speak 判定阈值")
    args = ap.parse_args()

    ids = [l.strip() for l in open(args.ids) if l.strip()]
    sess = sorted({parse_id(i)[0] for i in ids})[: args.n_sessions]
    x2_files = {f[:-5]: f"{ANNOT}/e4_raw/{f}"
                for f in os.listdir(f"{ANNOT}/e4_raw") if f.endswith(".json")}
    s8_to_full = {k[:8]: k for k in x2_files}

    ccs, lags, own_e, par_e, non_e = [], [], [], [], []
    cls_tot = np.zeros(4)          # 双方 / 仅本人 / 仅对方 / 都静
    for s8 in sess:
        full = s8_to_full.get(s8)
        if full is None:
            continue
        d = json.load(open(x2_files[full]))
        mp3 = f"{CANDOR}/{full}/processed/{full}.mp3"
        y, sr = sf.read(mp3, dtype="float32", always_2d=True)
        g = np.gcd(16000, int(sr))
        chans = {c: resample_poly(y[:, int(c)], 16000 // g, int(sr) // g)
                 .astype(np.float32) for c in ("0", "1")}
        del y
        fr = {c: np.array([[f[0], f[3], f[4]] for f in d["channels"][c]["frames"]],
                          dtype=float) for c in ("0", "1")}
        n = min(len(fr["0"]), len(fr["1"]))
        for ch in ("0", "1"):
            other = "1" if ch == "0" else "0"
            a, b = chans[ch], chans[other]
            L = min(len(a), len(b), int(n * FRAME_S * 16000))
            # A. 全段包络互相关 (切成 30 段分别算, 取中位)
            step = L // 30
            for k in range(30):
                seg_c, seg_l = env_xcorr(a[k * step:(k + 1) * step],
                                         b[k * step:(k + 1) * step])
                ccs.append(seg_c)
                lags.append(seg_l)
            # B. 帧级能量 (仅两类说话帧)
            so, sp = fr[ch][:n, 2], fr[other][:n, 2]
            m_own = (so > args.thr) & (sp <= args.thr)
            m_par = (sp > args.thr) & (so <= args.thr)
            m_non = (so <= args.thr) & (sp <= args.thr)
            m_both = (so > args.thr) & (sp > args.thr)
            cls_tot += np.array([m_both.sum(), m_own.sum(),
                                 m_par.sum(), m_non.sum()], dtype=float)
            fi = np.round(fr[ch][:n, 0] * 16000).astype(int)
            n_s = int(FRAME_S * 16000)
            ok = (fi >= 0) & (fi + n_s <= len(a))
            fi = fi[ok]
            e = np.array([float(np.mean(a[i:i + n_s] ** 2)) for i in fi])
            for mask, acc in ((m_own[ok], own_e), (m_par[ok], par_e),
                              (m_non[ok], non_e)):
                if mask.any():
                    acc.append(db(e[mask].mean()))
        print(f"{s8}: {len(ccs)} 段", flush=True)

    ccs = np.array(ccs)
    tot = cls_tot.sum()
    print(f"\n=== 帧四分类占比 (session 级汇总, {int(tot)} 帧) ===")
    print(f"   双方说 {cls_tot[0]/tot:.4f}   仅本人说 {cls_tot[1]/tot:.4f}   "
          f"仅对方说 {cls_tot[2]/tot:.4f}   都静默 {cls_tot[3]/tot:.4f}")
    print(f"\n=== A. own 声道内的串扰 (包络互相关 {len(ccs)} 段) ===")
    print(f"   峰值相关  中位 {np.median(ccs):6.3f}   p90 {np.percentile(ccs,90):6.3f}   "
          f"p99 {np.percentile(ccs,99):6.3f}")
    for th in (0.3, 0.5, 0.7):
        print(f"   相关 > {th}: {(ccs > th).mean()*100:5.1f}% 的段")
    print(f"\n=== B. 帧级能量 (dB, own 声道) ===")
    print(f"   仅本人说话帧 中位 {np.median(own_e):7.1f}  (n={len(own_e)})")
    print(f"   仅对方说话帧 中位 {np.median(par_e):7.1f}  (n={len(par_e)})")
    print(f"   都静默帧     中位 {np.median(non_e):7.1f}  (n={len(non_e)})")
    print(f"   对方 - 本人  中位 {np.median(par_e)-np.median(own_e):7.1f}   "
          f"<- 若 << 0, 对方在本人声道里被压得很低")
    print(f"   对方 - 静默  中位 {np.median(par_e)-np.median(non_e):7.1f}   "
          f"<- 若 <= 0, 对方话音没高出底噪")
    json.dump({"n_segments": len(ccs), "n_sessions": len(sess),
               "frame_class_frac": {
                   "both": round(float(cls_tot[0]/tot), 4),
                   "own_only": round(float(cls_tot[1]/tot), 4),
                   "partner_only": round(float(cls_tot[2]/tot), 4),
                   "silence": round(float(cls_tot[3]/tot), 4)},
               "env_xcorr_median": round(float(np.median(ccs)), 4),
               "env_xcorr_p90": round(float(np.percentile(ccs, 90)), 4),
               "frac_corr_gt_0.5": round(float((ccs > 0.5).mean()), 4),
               "partner_minus_own_db": round(
                   float(np.median(par_e) - np.median(own_e)), 2),
               "partner_minus_silence_db": round(
                   float(np.median(par_e) - np.median(non_e)), 2)},
              open(f"{ANNOT}/g1_leakage.json", "w"), indent=2)
    print("\nsaved g1_leakage.json")


if __name__ == "__main__":
    main()
