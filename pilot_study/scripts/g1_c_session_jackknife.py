"""逐会话 jackknife（§3.4 那套检查）—— **预登记 §4.3 明令在 C 上重做一遍**。

## 为什么必须重做

prereg §4.3 写死：
> ❌ 不许只看均值。§3.4 那套「效应不是被少数会话撑起来的」检查
> （留一、会话级分布）**在 C 上要重做一遍**。

C 是**另外 100 个会话**，§3.4 的结论（效应弥散、不是 3–5 个会话撑的）
**不能自动继承** —— 换了一批会话，就换了一个"会不会被少数会话带走"的问题。

## 做法

去掉一个会话，在剩下的并集上重算 ΔF1（7 个种子取均值），做 100 次。
报：LOO 范围、有没有翻号、SD、以及**单会话最大影响 / SE**。

## ⚠️ 先验自己

本脚本第一次跑必须**先在 e4c 上复现 §3.4 公布的三个数**：
  - LOO 范围 `[+0.0235, +0.0268]`
  - 100 次没有一次翻号
  - SD ≈ 0.00055
复现不了 ⇒ 说明我这份实现和当初那份不是同一个检查，**C 上的数不许引用**。
(这正是 memory `hardcoded-conclusions-escape-reproduction` 的用法：
 拿旧结论当**测试**，而不是拿它当**依据**。)

用法 (srun 内, fd_analysis):
  python scripts/g1_c_session_jackknife.py --set e4c   # 先验
  python scripts/g1_c_session_jackknife.py --set c
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1_c_analyze as G  # noqa: E402

# §3.4 在 e4c 上公布的数 —— 用作自检，不是用作结论
E4C_EXPECT = dict(lo=0.0235, hi=0.0268, sd=0.00055)


def f1_from(tp, fp, gt):
    fn = gt - tp
    den = 2 * tp + fp + fn
    return 0.0 if den == 0 else 2 * tp / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="c")
    args = ap.parse_args()

    data = G.load_set(args.set)
    G.assert_complete(data, args.set)
    _, _, cids, seeds = G.delta_by_seed(data)
    sess_of = {c: c.split("_")[0] for c in cids}
    sessions = sorted({sess_of[c] for c in cids})
    by_sess = {s: [c for c in cids if sess_of[c] == s] for s in sessions}
    print(f"{args.set}: {len(cids)} 块 / {len(sessions)} 会话 / {len(seeds)} 种子")

    # 逐会话聚合, 免得 100 次 LOO 各扫一遍 4800 块
    agg = {}          # (arm,seed) -> {sess: (tp,fp,gt)}
    tot = {}          # (arm,seed) -> (tp,fp,gt)
    for k in data:
        d = {}
        tt = tf = tg = 0
        for s in sessions:
            tp = sum(data[k][c]["tp"] for c in by_sess[s])
            fp = sum(data[k][c]["fp"] for c in by_sess[s])
            gt = sum(data[k][c]["n_gt"] for c in by_sess[s])
            d[s] = (tp, fp, gt)
            tt += tp; tf += fp; tg += gt
        agg[k] = d
        tot[k] = (tt, tf, tg)

    def df1(scale, drop=None):
        """scale: {sess: 权重}（jackknife 时全 1）; drop: 去掉的会话。"""
        per = []
        for s_ in seeds:
            f = {}
            for arm in G.ARMS:
                tp, fp, gt = tot[(arm, s_)]
                if drop is not None:
                    a, b, c = agg[(arm, s_)][drop]
                    tp -= a; fp -= b; gt -= c
                f[arm] = f1_from(tp, fp, gt)
            per.append(f["own10"] - f["mixnorm"])
        return float(np.mean(per))

    base = df1(None)
    loos = {s: df1(None, drop=s) for s in sessions}
    v = np.array([loos[s] for s in sessions])
    infl = np.abs(v - base)
    print(f"\n并集 ΔF1 = {base:+.4f}")
    print(f"  LOO 范围 [{v.min():+.4f}, {v.max():+.4f}]  SD {v.std(ddof=1):.5f}")
    print(f"  翻号次数: {int((v <= 0).sum())} / {len(v)}")
    worst = sessions[int(np.argmax(infl))]
    print(f"  单会话最大影响 {infl.max():.4f}  ({worst[:8]}…)")
    comp = {}
    for name, lv in (("sess", ("sess",)), ("all", ("sess", "block", "seed"))):
        _, sd, _, _ = G.boot_clustered(data, cids, B=600, lv=lv)
        comp[name] = sd
    _, sd_all, _, _ = G.boot_clustered(data, cids, B=600)
    comp["all"] = sd_all
    print(f"  影响 / 会话级 SE = {infl.max()/comp['sess']:.2f} ×")
    print(f"  影响 / 合并 SE   = {infl.max()/comp['all']:.2f} ×")
    print(f"  (SE: 会话 {comp['sess']:.4f} | 合并 {comp['all']:.4f}, "
          f"估计量 boot_clustered, B=600)")

    if args.set == "e4c":
        tol = 0.0008        # 允许实现差异, 但方向/量级必须对上
        ok = (abs(v.min() - E4C_EXPECT["lo"]) < tol
              and abs(v.max() - E4C_EXPECT["hi"]) < tol
              and abs(v.std(ddof=1) - E4C_EXPECT["sd"]) < tol / 2)
        print(f"\n  §3.4 公布值: [{E4C_EXPECT['lo']:+.4f}, {E4C_EXPECT['hi']:+.4f}] "
              f"SD {E4C_EXPECT['sd']:.5f}")
        print("  " + ("✅ 复现 —— 这份实现就是当初那个检查, C 上的数可引用"
                      if ok else
                      "🔴 复现不了 —— C 上的数**不许引用**, 先查实现差异"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
