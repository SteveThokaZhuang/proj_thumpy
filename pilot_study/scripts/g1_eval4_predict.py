"""k=4 到底能不能判定? —— 用**已有**数据把 SE_chunk 的标度律测出来, 0 GPU 成本.

## 为什么要先算这一步

§7.5 把 k=4 成本估成「1200 块 ≈ 80 事件」。实际按自然流行率选块后只有
**43 个事件**(3.58%), 不是 80。于是外推口径决定结论:

  若 SE_chunk 由**块数**驱动 -> 1200 块是 4×, SE 减半, t ≈ 2.2 (可判定)
  若 SE_chunk 由**事件数**驱动 -> 只有 43/26 = 1.65×, t ≈ 1.9 (仍不可判定)

差的这 0.3 正好跨过 t=2, 而 7.8 GPU 小时就押在它上面。

## 怎么把两个口径分开

只在一套集上子采样是分不开的 —— 流行率恒定时事件数 ∝ 块数, 两条律重合。
必须比较**流行率不同**的两套集在**相同块数**下的 SE_chunk:

  旧 300 块: GT 26 事件, 流行率 8.7%
  E2 1600 块: GT 1640 事件, 流行率 100%

在 k=300 处对齐比 SE_chunk:
  - 两者接近        -> 块数驱动 -> k=4 乐观 (t≈2.2)
  - E2 明显更小     -> 事件驱动 -> k=4 悲观 (t≈1.9)
  - 介于两者之间    -> 用拟合出的指数直接外推到 1200 块

子采样时**跨 14 个 adapter 共用同一份块子集** —— 块抽样是一个跨种子共有的
方差源, 每臂各自重抽会把这一项平均掉, 系统性低估 SE。
见 [[permutation-test-covers-one-variance-source]]。

用法: python scripts/g1_eval4_predict.py [--n-boot 2000]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_g1_common import ANNOT  # noqa: E402

SEEDS = [42, 1234, 7, 2024, 3407, 31337, 55555]
# (集合名, 文件名中的臂后缀, tag) —— tag 是 json 顶层键的后缀
SETS = {"old": ("", ""), "filt": ("_filt", "_filt"), "e2": ("_e2", "_e2")}


def load_set(arm, tag, suffix):
    """-> {seed: {cid: (tp, fp, n_gt)}}"""
    out = {}
    for s in SEEDS:
        ss = "" if s == 42 else f"_s{s}"
        p = f"{ANNOT}/g1_eval_{arm}{ss}{suffix}.json"
        if not os.path.exists(p):
            raise SystemExit(f"❌ 缺 {p}")
        d = json.load(open(p))[f"{arm}{ss}{tag}"]
        out[s] = {c: (v["tp"], v["fp"], v["n_gt"]) for c, v in d["per_chunk"].items()}
    return out


def f1_pair(own, mix, cids):
    """共用块子集上的配对 ΔF1(own10 − mixnorm): 逐种子配对后取均值。"""
    ds = []
    for s in SEEDS:
        fs = []
        for arm in (own, mix):
            tp = sum(arm[s][c][0] for c in cids)
            fp = sum(arm[s][c][1] for c in cids)
            gt = sum(arm[s][c][2] for c in cids)
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / gt if gt else 0.0
            fs.append(2 * p * r / (p + r) if p + r else 0.0)
        ds.append(fs[0] - fs[1])
    return float(np.mean(ds)), ds


def scaling(own, mix, cids, ks, n_boot, rng, label):
    """在若干块数上测 SE_chunk(配对 ΔF1 的子采样 SD)。"""
    print(f"\n── {label}: {len(cids)} 块 / "
          f"GT {sum(own[SEEDS[0]][c][2] for c in cids)} 事件")
    d_full, d_seed = f1_pair(own, mix, cids)
    se_seed = float(np.std(d_seed, ddof=1)) / np.sqrt(len(SEEDS))
    print(f"   全量 ΔF1 = {d_full:+.4f}; SE_seed = {se_seed:.4f}")
    rows = []
    for k in ks:
        k = min(k, len(cids))
        if k >= len(cids):
            # 无放回抽满 = 每次都返回全集, SD 恒为 0。这一行没有信息, 且 log(0)
            # 会把后面的拟合整个带偏 —— 直接跳过。
            print(f"   k={k:5d}: 等于全集, 跳过 (退化点)")
            continue
        sds, evs = [], []
        for _ in range(n_boot):
            sub = [cids[i] for i in rng.choice(len(cids), size=k, replace=False)]
            evs.append(sum(own[SEEDS[0]][c][2] for c in sub))
            sds.append(f1_pair(own, mix, sub)[0])
        rows.append((k, float(np.std(sds, ddof=1)), float(np.mean(evs))))
        print(f"   k={k:5d} (事件 {np.mean(evs):6.1f}): SE_chunk = {rows[-1][1]:.4f}")
    return d_full, se_seed, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1500)
    ap.add_argument("--base", default="filt", choices=["old", "filt"],
                    help="用哪套集做拟合基准; filt 是边界过滤后的**正式口径** "
                         "(ΔF1=+0.0340, SE_seed=0.0117 与 §7.2 一致)。"
                         "plain 没做边界过滤, 事件多 6 个、SE_seed 0.0180, 会系统性低估 t。")
    ap.add_argument("--target-k", type=int, default=1200)
    ap.add_argument("--target-ev", type=float, default=43.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = {}
    for name, (tag, suf) in SETS.items():
        own, mix = load_set("own10", tag, suf), load_set("mixnorm", tag, suf)
        cids = sorted(set(own[SEEDS[0]]) & set(mix[SEEDS[0]]))
        for s in SEEDS:
            assert set(own[s]) == set(mix[s]) == set(cids), f"{name} seed {s} 块集不一致"
        data[name] = (own, mix, cids)

    rng = np.random.default_rng(args.seed)
    res = {}
    for name, (own, mix, cids) in data.items():
        ks = [len(cids)] + [len(cids) // d for d in (2, 4, 8)]
        res[name] = scaling(own, mix, cids, ks, args.n_boot, rng, name)

    # ── 判据: 相同块数下比较两套集的 SE_chunk (流行率 8.7% vs 100%) ──────
    print("\n" + "=" * 64)
    k_cmp = 200
    for name, (own, mix, cids) in data.items():
        if len(cids) < k_cmp:
            continue
        sds, evs = [], []
        for _ in range(args.n_boot):
            sub = [cids[i] for i in rng.choice(len(cids), size=k_cmp, replace=False)]
            evs.append(sum(own[SEEDS[0]][c][2] for c in sub))
            sds.append(f1_pair(own, mix, sub)[0])
        print(f"  k={k_cmp}: {name:4s} 事件 {np.mean(evs):6.1f}  "
              f"SE_chunk = {np.std(sds, ddof=1):.4f}")
    print("  两者接近 => 块数驱动; old 明显更大 => 事件驱动")

    # ── 用 old 集的拟合曲线外推到 k=4 ─────────────────────────────────
    # 单一集合上测不出"块"与"事件"哪个是驱动量 (流行率恒定时二者成正比),
    # 所以不预设, 而是把两种口径**各自**代进同一条拟合曲线, 给出预测区间。
    # 真实值落在两者之间或某一侧, 无论哪种都远小于 §7.5 假设的乐观值。
    d_full, se_seed, rows = res[args.base]
    lg_k, lg_s = np.log([r[0] for r in rows]), np.log([r[1] for r in rows])
    alpha, b = np.polyfit(lg_k, lg_s, 1)
    pi_old = rows[0][2] / rows[0][0]                    # 老集每块事件数
    k_equiv = args.target_ev / pi_old                   # 43 事件相当于多少老集块
    print("\n" + "=" * 64)
    print(f"{args.base} 集拟合: SE_chunk ∝ k^({alpha:.3f}) "
          f"(在 {[r[0] for r in rows]} 上)")
    for lab, k_arg in ((f"按块数 ({args.target_k} 块)", args.target_k),
                       (f"按事件数 (43 事件 ≈ {k_equiv:.0f} 块)", k_equiv)):
        pred = float(np.exp(b + alpha * np.log(k_arg)))
        se = float(np.hypot(se_seed, pred))
        print(f"  {lab}: SE_chunk = {pred:.4f} -> 合并 SE = {se:.4f} "
              f"-> t = {d_full/se:.2f}")
    print(f"  天花板 (SE_chunk→0): t = {d_full/se_seed:.2f}")
    print("\n  注意 k=200 处跨集对比 (old/filt vs e2) **不可用**: E2 的 F1 量级"
          "\n  高 3–7 倍, 绝对噪声随量级走, 比出来的是量级差不是事件数差。"
          "\n  判驱动量要用**同一集合内**的标度指数。")


if __name__ == "__main__":
    main()
