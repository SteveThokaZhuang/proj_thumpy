"""校验 mixnorm 臂里对方到底占多少 —— 全 300 个评估 chunk, 而非抽样。

为什么必须做: ΔF1≈0 (且略负) 的零结果, 最容易被质疑的一点就是"你根本没把对方
送进去, 所以这个零什么也说明不了"。12 个 chunk 的抽样已经暴露出一个系统性失效:
make_audio 的增益 gain = min(p95(own)/p95(part), 30), 当**本人声道静音**时
p95(own)≈0 -> gain≈0 -> 对方被压成静音, 该 chunk 的 mix == own。这个比例不量
化就不能下结论, 所以这里跑全量。

对每个 chunk:
  a = <mix,own>/<own,own>        (最小二乘, 因 mix = s*(own+d) 只差一个全局尺度)
  res = mix - a*own              (≈ s*d, 即对方 + 底噪)
  corr(res, part)                -> 应≈1, 证明残差确实就是对方声道
  e(res)/e(mix)                  -> 对方在送入模型的音频里占多少能量
  gain 与 20log10(对方/本人)      -> 对方相对本人低多少 dB (可听性的直接度量)
并**按 chunk 是否含真值 BC 事件分层** —— 只在有 BC 的 chunk 上, 可听性才有意义。
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts")
from ari_g1_prep_subset import ANNOT, CANDOR

EVAL_IDS = f"{ANNOT}/g1_eval_ids.txt"
OWN_DIR = f"{ANNOT}/g1_own_eval/own"
MIX_DIR = f"{ANNOT}/g1_sub/mixnorm"
OUT = f"{ANNOT}/g1_mix_check.json"
EPS = 1e-8
CHUNK_S = 10.0


def p95(x):
    return float(np.percentile(np.abs(x), 95))


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def load_bc_chunks():
    """哪些 chunk 含真值 BC 事件 —— 只在有 BC 的 chunk 上谈可听性才有意义."""
    try:
        from ari_g1_common import bc_windows, load_manifest
        mlook = load_manifest(MIX_DIR)
        return {cid: len(bc_windows(cid, mlook, CHUNK_S)) for cid in mlook}
    except Exception as e:                                   # noqa: BLE001
        print(f"** 无法取真值 BC 窗口 ({e}); 跳过分层 **")
        return {}


def main():
    ids = [l.strip() for l in open(EVAL_IDS) if l.strip()]
    man = {}
    for l in open(f"{MIX_DIR}/manifest.jsonl"):
        d = json.loads(l)
        man[d["id"]] = d
    ids = [c for c in ids if c in man]
    print(f"评估 chunk {len(ids)}")

    bc_n = load_bc_chunks()
    cache = {}
    rows = []
    for k, cid in enumerate(ids):
        m = man[cid]
        ch = str(m["ch"])
        other = "1" if ch == "0" else "0"
        full, t0 = m["session"], m["t0"]
        if full not in cache:
            y, sr = sf.read(f"{CANDOR}/{full}/processed/{full}.mp3",
                            dtype="float32", always_2d=True)
            g = np.gcd(16000, int(sr))
            cache[full] = {c: resample_poly(y[:, int(c)], 16000 // g, int(sr) // g)
                           .astype(np.float32) for c in ("0", "1")}
            if len(cache) > 8:               # 控制内存: 每个 session 约几十 MB
                cache.pop(next(iter(cache)))
        i0, i1 = int(t0 * 16000), int((t0 + CHUNK_S) * 16000)
        part = cache[full][other][i0:i1]
        own = np.load(f"{OWN_DIR}/{cid}.npz")["audio"]
        mix = np.load(f"{MIX_DIR}/{cid}.npz")["audio"]
        n = min(len(part), len(own), len(mix))
        part, own, mix = part[:n], own[:n], mix[:n]

        a = float(np.dot(mix, own) / (np.dot(own, own) + EPS))
        res = mix - a * own
        e_mix, e_res = rms(mix) ** 2, rms(res) ** 2
        e_own, e_part = rms(own) ** 2, rms(part) ** 2
        sd = lambda x: float(np.std(x))
        # corr 在任一声道近似全零时无定义 -> nan, 统计时用 nan* 版本
        cr = (float(np.corrcoef(res, part)[0, 1])
              if sd(res) > 1e-9 and sd(part) > 1e-9 else np.nan)
        gain = min(p95(own) / (p95(part) + EPS), 30.0)
        rows.append({
            "id": cid, "corr_mix_own": (float(np.corrcoef(mix, own)[0, 1])
                                        if sd(mix) > 1e-9 and sd(own) > 1e-9 else np.nan),
            "corr_res_part": cr,
            "res_share": e_res / (e_mix + EPS),
            "gain": gain,
            # 对方相对本人的能量比 (dB); 本人静音时无意义 -> 交给分层统计
            "part_rel_own_db": (10 * np.log10((e_part + EPS) / (e_own + EPS))),
            "own_rms_db": 10 * np.log10(e_own + EPS),
            "n_bc": bc_n.get(cid, -1),
        })
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(ids)}")

    R = lambda key: np.array([r[key] for r in rows], dtype=float)  # noqa: E731
    res_share, corr = R("res_share"), R("corr_res_part")
    # 失效判据: 残差几乎不含能量, 说明对方被压没了 (mix ≈ own)
    dead = res_share < 0.01

    def stat(mask, name):
        if mask.sum() == 0:
            return
        print(f"\n--- {name} (n={int(mask.sum())}) ---")
        print(f"  对方占 mix 能量   中位 {np.nanmedian(res_share[mask])*100:6.1f}%"
              f"   p10 {np.nanpercentile(res_share[mask], 10)*100:6.1f}%"
              f"   p90 {np.nanpercentile(res_share[mask], 90)*100:6.1f}%")
        c = corr[mask]
        print(f"  corr(残差,对方)   中位 {np.nanmedian(c):.4f}   "
              f"有效 {int(np.sum(~np.isnan(c)))}/{len(c)}")
        print(f"  对方被压没(res<1%)  {int(dead[mask].sum())}/{int(mask.sum())}"
              f"  ({dead[mask].mean()*100:.1f}%)")

    stat(np.ones(len(rows), bool), "全部评估 chunk")
    has_bc = R("n_bc") > 0
    if bc_n:
        stat(has_bc, "含真值 BC 的 chunk  <- 关键")
        stat(~has_bc, "不含 BC 的 chunk")
        print(f"\n含 BC 的 chunk 共 {int(has_bc.sum())}, 承载真值事件 {int(R('n_bc')[has_bc].sum())} 个")

    print(f"\n对方相对本人的能量 (dB)  中位 {np.nanmedian(R('part_rel_own_db')):+.1f}"
          f"  p10 {np.nanpercentile(R('part_rel_own_db'), 10):+.1f}"
          f"  p90 {np.nanpercentile(R('part_rel_own_db'), 90):+.1f}")
    print(f"gain 中位 {np.nanmedian(R('gain')):.3f}  取到 30dB 上限的 {int((R('gain') >= 29.99).sum())} 个")
    verdict = ("操纵有效" if np.nanmedian(corr) > 0.9 and np.nanmedian(res_share) > 0.05
               else "**操纵可疑, 需修 make_audio**")
    print(f"\n结论: {verdict}  (中位 corr={np.nanmedian(corr):.3f}, "
          f"中位能量占比={np.nanmedian(res_share)*100:.1f}%)")

    json.dump({"rows": rows,
               "summary": {"n": len(rows), "n_has_bc": int(has_bc.sum()) if bc_n else None,
                           "median_res_share": float(np.nanmedian(res_share)),
                           "median_corr_res_part": float(np.nanmedian(corr)),
                           "n_dead": int(dead.sum())}},
              open(OUT, "w"), indent=2, default=float)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
