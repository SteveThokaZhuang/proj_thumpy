"""§5.2c 事后稳健性：P3 的 ρ=0.90 里有多少是「时长」？

**非预登记** —— 预登记的 P3（ρ>0）在看到结果前就写死了，这里只是追问它的**幅度**。
动机：长的对话什么都多（重叠秒数多、backchannel 也多），跨对话的
「重叠秒数 ↔ n_BC」相关很可能只是在测**时长**。

纯读盘，0 GPU，读 §5.2c 的 corpus_overlap.json。

用法:
  python duplexgen_overlap_robust.py --in real_data/results/duplexgen_annot/corpus_overlap.json
"""
import argparse
import json

import numpy as np


def rank(x):
    o = np.argsort(x)
    rk = np.empty(len(x))
    rk[o] = np.arange(len(x))
    return rk


def rho(a, b):
    """Spearman。手写排序秩, 避免为这一件事再引 scipy。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def partial(x, y, z):
    """控住 z 之后 x 与 y 的偏相关 (Pearson 口径, 与上面的秩相关不同, 注意标注)。"""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    rxy, ryz, rxz = (np.corrcoef(p, q)[0, 1]
                     for p, q in ((x, y), (y, z), (x, z)))
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    a = ap.parse_args()
    D = json.load(open(a.inp))
    recs = [r for r in D["records"]
            if r.get("n_backchannels") is not None and r.get("dur_s")]
    ov = np.array([r["overlap"] * r["dur_s"] for r in recs])   # 重叠**秒数**
    ovf = np.array([r["overlap"] for r in recs])               # 重叠**占比**
    dur = np.array([r["dur_s"] for r in recs])
    nbc = np.array([r["n_backchannels"] for r in recs], float)
    sc = np.array([r["scenario"] for r in recs])
    print("n=%d 条对话\n" % len(recs))

    print("=== 主读数复算（应与 §5.2c 的 P3 一致）===")
    print("  ρ(重叠秒数, n_backchannels) = %+.3f" % rho(ov, nbc))
    print("\n=== 这条相关是不是「时长」在背后驱动？ ===")
    print("  ρ(时长,     n_backchannels) = %+.3f" % rho(dur, nbc))
    print("  ρ(时长,     重叠秒数)       = %+.3f" % rho(dur, ov))
    print("\n=== 把时长除掉之后还在不在 ===")
    print("  ρ(重叠占比, n_backchannels) = %+.3f   ← 按对话时长归一" % rho(ovf, nbc))
    print("  ρ(重叠秒数, n_BC/时长)      = %+.3f" % rho(ov, nbc / dur))
    print("  ρ(重叠占比, n_BC/时长)      = %+.3f" % rho(ovf, nbc / dur))
    print("  偏相关 ρ(重叠秒数, n_BC | 时长) = %+.3f  (Pearson 口径)" % partial(ov, nbc, dur))

    print("\n=== 合并会不会是「场景」造出来的？ ===")
    print("  %-5s %5s  %-13s %-13s %-13s"
          % ("场景", "n", "ρ(秒,n_BC)", "ρ(时长,n_BC)", "ρ(占比,n_BC)"))
    for s in sorted(set(sc)):
        m = sc == s
        print("  %-5s %5d  %+13.3f %+13.3f %+13.3f"
              % (s, m.sum(), rho(ov[m], nbc[m]), rho(dur[m], nbc[m]),
                 rho(ovf[m], nbc[m])))

    print("\n判读: 方向站得住(偏相关强、逐场景全正), 但**幅度含时长**")
    print("      ⇒ 引用必须写清是「重叠秒数」还是「重叠占比」。")


if __name__ == "__main__":
    main()
