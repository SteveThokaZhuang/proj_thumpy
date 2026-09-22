"""C 的判读: 在**全新的 100 个会话**上复现 ΔF1 > 0 (docs/pilot_study/2026-09-22_planC_prereg.md).

## 这个脚本回答的唯一问题

§2–§3 的所有结论都建立在**固定的那 100 个会话**上 (所有 SE 都是"从这 100 个里重抽")。
C 跑 100 个**全新**会话、用**同一批已训好的 adapter** (0 训练), 按预登记判据检验
ΔF1 > 0 —— 这才是把"在这 100 个上成立"升级成"在 CANDOR 上成立"。

## 三条口径 (缺一不可, 顺序不能换)

1. **主判据 = 符号**。新池流行率是 e4c 的两倍 (§5.3) ⇒ **幅度不可直接比**。
2. **会话簇 bootstrap** 才是诚实 SE。会话是"换一批"时的抽样单元;
   只按块重抽会漏掉会话那一项 (memory `permutation-test-covers-one-variance-source`)。
3. **同仪器读数**: 把新集**事后子采样到 e4c 的流行率**再算一次 ΔF1,
   这一个数才能和 e4c 的 ΔF1 并排。纯读盘, 0 GPU。

## 为什么必须用并集算 F1

F1 是比值 ⇒ 不能"各部分算完再平均" (memory `delta-f1-does-not-decompose`)。
所有分组 (arm × seed) 的 F1 都在**同一批 chunk 的并集**上从 tp/fp/gt 求和后算。

用法 (fd_analysis, 纯读盘):
  python scripts/g1_c_analyze.py --set e4c                 # 只算基线
  python scripts/g1_c_analyze.py --set c                   # 只算 C (新 100 会话)
  python scripts/g1_c_analyze.py --set union               # 并集 (§4.3 的 S=200)
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_g1_common import ANNOT  # noqa: E402

SEEDS = ["42", "1234", "7", "2024", "3407", "31337", "55555"]
ARMS = ["own10", "mixnorm"]
SUFFIX = {s: ("" if s == "42" else f"_s{s}") for s in SEEDS}

# 文件名的 tag 形状: {arm}{suffix}_{set}{batch:02d}
#   e4c:  own10_s1234_e4c07     (12 批)
#   c  :  own10_s1234_c         (不分批)
TAG_RE = re.compile(r"^(own10|mixnorm)(_s\d+)?_([a-z0-9]+?)(\d{2})?$")


def load_set(name):
    """把某一套评估结果归拢成 {(arm, seed): {cid: per_chunk}}。

    不假设批数: 同一 (arm, seed) 下多个文件按 cid 合并; 若同一 cid 出现两次
    (批次划分重叠) 就炸 —— 静默覆盖会让并集悄悄少块。
    """
    out = {}
    for p in sorted(glob.glob(f"{ANNOT}/g1_eval_*_{name}*.json")):
        fn = os.path.basename(p)[len("g1_eval_"):-len(".json")]
        m = TAG_RE.match(fn)
        if not m or m.group(3) != name:
            continue
        arm, suf, _, _ = m.groups()
        seed = "42" if suf is None else suf[2:]
        if seed not in SEEDS:
            continue
        blk = out.setdefault((arm, seed), {})
        d = json.load(open(p))
        # 每个文件里只有一个 key (就是 tag 本身)
        payload = d[list(d)[0]] if len(d) == 1 and "per_chunk" not in d else d
        for cid, rec in payload["per_chunk"].items():
            assert cid not in blk, f"{cid} 在 {arm}/{seed} 下出现两次 ({p})"
            blk[cid] = rec
    return out


def f1_over(recs):
    """在一组 per_chunk 记录上算 F1 (并集口径: 先求和再算比值)。"""
    tp = sum(r["tp"] for r in recs)
    fp = sum(r["fp"] for r in recs)
    gt = sum(r["n_gt"] for r in recs)
    p = tp / max(1, tp + fp)
    r_ = tp / max(1, gt)
    return (2 * p * r_ / (p + r_)) if (p + r_) else 0.0, tp, fp, gt


def delta_by_seed(data):
    """{(arm,seed): {cid: rec}} -> (per-seed ΔF1 列表, 并集上的 F1 表)。

    ⚠️ 并集 = **全部 arm/seed 共用**的 cid 集合。缺块会让并集不等价。
    """
    cids = None
    for k, v in data.items():
        s = set(v)
        cids = s if cids is None else (cids & s)
    assert cids, "没有共同的 chunk"
    cids = sorted(cids)
    f1 = {}
    for (arm, seed), v in data.items():
        f1[(arm, seed)] = f1_over([v[c] for c in cids])[0]
    d = [f1[("own10", s)] - f1[("mixnorm", s)] for s in SEEDS
         if ("own10", s) in f1 and ("mixnorm", s) in f1]
    return d, f1, cids


def _f1_from_counts(tp, fp, gt):
    """由 (Σtp, Σfp, Σgt) 算 F1 —— 并集口径。"""
    p = tp / np.maximum(1.0, tp + fp)
    r_ = tp / np.maximum(1.0, gt)
    s = p + r_
    return np.where(s > 0, 2 * p * r_ / np.maximum(s, 1e-12), 0.0)


def _draw_sums(a, si, ci, B, chunk=500):
    """对 (S, C, 3) 的账做 B 次重抽, 返回 (B, 3) 的和。

    ⚠️ 会话重抽是**抽行**、块重抽是**抽列** —— 两者形状不同, 不能各写一个
    `x[idx]`。`ci` 是 (B, C) 而 x 是 (B, S, C, 3), 直接 `x[:, :, ci]` 会广播错
    (本脚本第一版就是这么炸的: "index 40 is out of bounds for axis 2 with size 3")。
    正确做法是把会话维折进 batch 再 `take_along_axis`。分块是为了不把
    B×S×C×3 全铺开 (4000×100×48×3 ≈ 460 MB/份, 还要复制好几份)。
    """
    S, C, _ = a.shape
    out = np.empty((B, 3), dtype=np.float64)
    for b0 in range(0, B, chunk):
        n = min(chunk, B - b0)
        x = a[si[b0:b0 + n]] if si is not None else np.broadcast_to(
            a, (n, S, C, 3))
        if ci is not None:
            x = x.reshape(n * S, C, 3)
            cols = np.broadcast_to(ci[b0:b0 + n, None, :], (n, S, C))
            x = np.take_along_axis(x, cols.reshape(n * S, C)[:, :, None], axis=1)
            x = x.reshape(n, S, C, 3)
        out[b0:b0 + n] = x.sum(axis=(1, 2))
    return out


def boot(data, cids, B=4000, seed=0, lv=("sess", "block", "seed")):
    """三级 bootstrap 的 ΔF1 分布: 抽**会话** × 抽**会话内的块** × 抽**种子**。

    ⚠️ 三级缺一不可, 而且**每一级都有人漏过** (memory
    `permutation-test-covers-one-variance-source`):
      - 只抽种子 ⇒ 系统性乐观 (G1 的 p=0.0469 就是这么来的);
      - 只抽块   ⇒ 漏掉种子那一路 (19.9 → 合并 42.4);
      - 只抽会话 ⇒ 漏掉块那一路 (本脚本初版实测 0.0054 vs 三级 ~0.0092)。
    `lv` 控制开哪几级, 用来把总 SE 分解回三个分量。

    实现要点: 每个会话的块数相同 (e4c/C 都是 48) ⇒ reshape 成 (S, N, 3),
    会话重抽 = 抽行、块重抽 = 抽列, 都是 fancy-index, 与逐块重抽等价且快得多。
    """
    S = len({c.split("_")[0] for c in cids})
    C = len(cids) // S
    assert S * C == len(cids), f"块数 {len(cids)} 不能被会话数 {S} 整除"
    order = sorted(range(len(cids)), key=lambda i: (cids[i].split("_")[0], cids[i]))
    keys = [s for s in SEEDS if ("own10", s) in data and ("mixnorm", s) in data]
    rng = np.random.default_rng(seed)

    arr = {}
    for k in data:
        a = np.array([[data[k][cids[i]]["tp"], data[k][cids[i]]["fp"],
                       data[k][cids[i]]["n_gt"]] for i in order], dtype=np.float64)
        arr[k] = a.reshape(S, C, 3)

    si = rng.integers(0, S, size=(B, S)) if "sess" in lv else None
    ci = rng.integers(0, C, size=(B, C)) if "block" in lv else None
    gi = rng.integers(0, len(keys), size=(B, len(keys))) if "seed" in lv else None

    per = {}
    for k, a in arr.items():
        tot = _draw_sums(a, si, ci, B)
        per[k] = _f1_from_counts(tot[:, 0], tot[:, 1], tot[:, 2])
    ds = np.stack([per[("own10", s)] - per[("mixnorm", s)] for s in keys])  # (K,B)
    # ds 是 (K, B): 取 ds[种子下标, 第b次] ⇒ 下标矩阵要广播成 (B, K), 再对 K 取均值
    d = (ds.mean(axis=0) if gi is None
         else ds[gi, np.arange(B)[:, None]].mean(axis=1))
    return float(d.mean()), float(d.std()), float(np.percentile(d, 2.5)), \
        float(np.percentile(d, 97.5))


def downsample_to(data, cids, target_prev, seed=0):
    """把块集子采样到接近 target_prev 的流行率 (纯读盘, 0 GPU)。

    ⚠️ 流行率用 **事件数/块数** (与 `g1_eval4c_meta.json:prevalence` 同定义),
    不是「含事件块数/块数」—— 两者在 e4c 上是 3.9375% vs 3.7708%, 别混。

    分层: 含事件的块全留, 不含事件的按概率 q 抽。
        流行率 = gt / (n_pos + n_neg·q) = target  ⇒  q = (gt/target − n_pos)/n_neg
    块内事件多的块不该被更重地保留(它们只有一个 tp 上限), 所以按块分层而非按事件。
    """
    gt = sum(data[("own10", "42")][c]["n_gt"] for c in cids)
    n_pos = [c for c in cids if data[("own10", "42")][c]["n_gt"] > 0]
    n_neg = [c for c in cids if data[("own10", "42")][c]["n_gt"] == 0]
    cur = gt / max(1, len(cids))
    if target_prev >= cur:
        return cids, cur, len(cids)
    q = (gt / target_prev - len(n_pos)) / max(1, len(n_neg))
    q = min(1.0, max(0.0, q))
    rng = np.random.default_rng(seed)
    keep = sorted(n_pos + [c for c in n_neg if rng.random() < q])
    got = sum(data[("own10", "42")][c]["n_gt"] for c in keep) / max(1, len(keep))
    return keep, got, len(keep)


def report(name, data, B=4000):
    print(f"\n{'='*70}\n[{name}]  (arm,seed) 组数 = {len(data)}")
    if not data:
        print("  无数据"); return None
    d, f1, cids = delta_by_seed(data)
    n_ev = sum(1 for c in cids if data[("own10", "42")][c]["n_gt"] > 0)
    gt = sum(data[("own10", "42")][c]["n_gt"] for c in cids)
    print(f"  并集 {len(cids)} 块 / 覆盖 {len({c.split('_')[0] for c in cids})} 会话 / "
          f"{gt} 个真值事件")
    print(f"  含事件块 {n_ev} = {n_ev/len(cids):.4%}  ← 流行率, 跨集比之前必看")
    for arm in ARMS:
        v = [f1[(arm, s)] for s in SEEDS if (arm, s) in f1]
        if v:
            print(f"  F1[{arm:8s}] = {np.mean(v):.4f} ± {np.std(v, ddof=1):.4f} "
                  f"(跨 {len(v)} 种子)")
    d = np.array(d)
    print(f"  ΔF1 (逐种子) = {np.round(d, 4).tolist()}")
    # 三级分解: 每一级单独开一次, 得到该级的 sd; 三级全开得到总 SE
    comp = {}
    for name, lv in (("sess", ("sess",)), ("block", ("block",)),
                     ("seed", ("seed",)),
                     ("all", ("sess", "block", "seed"))):
        _, s_, _, _ = boot(data, cids, B=B, lv=lv)
        comp[name] = float(s_)
    m, sd, lo, hi = boot(data, cids, B=B)
    print(f"  ΔF1 均值 = {d.mean():+.4f}  (逐种子 SE "
          f"{d.std(ddof=1)/np.sqrt(len(d)):.4f})")
    print(f"  SE 分解: 会话 {comp['sess']:.4f} | 块 {comp['block']:.4f} | "
          f"种子 {comp['seed']:.4f}  ⇒ 合并 {comp['all']:.4f}")
    print(f"  三级 bootstrap: {m:+.4f} ± {sd:.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"   t = {m/sd if sd else float('nan'):+.2f}")
    return dict(n_chunks=len(cids), n_events=int(gt),
                prevalence=gt / len(cids),
                delta_mean=float(d.mean()), delta_sd=float(d.std(ddof=1)),
                delta_by_seed=[float(x) for x in d],
                se_components=comp,
                boot_mean=m, boot_se=sd, boot_ci=[lo, hi],
                f1={f"{a}_{s}": float(f1[(a, s)]) for a in ARMS for s in SEEDS
                    if (a, s) in f1})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="e4c / c / union")
    ap.add_argument("--match-prevalence", type=float, default=0.039375,
                    help="同仪器读数的目标流行率 (默认 e4c 实测值)")
    ap.add_argument("--boot", type=int, default=4000)
    args = ap.parse_args()

    if args.set == "union":
        a, b = load_set("e4c"), load_set("c")
        data = {}
        for k in set(a) | set(b):
            m = dict(a.get(k, {}))
            for cid, rec in b.get(k, {}).items():
                assert cid not in m, f"并集: {cid} 两侧都有 ⇒ 新旧会话不独立!"
                m[cid] = rec
            data[k] = m
    else:
        data = load_set(args.set)

    res = {"set": args.set, "seeds": SEEDS}
    r = report(args.set, data, args.boot)
    if r is None:
        return
    res["main"] = r

    # ---- 同仪器读数: 子采样到 e4c 的流行率 ----
    d, f1, cids = delta_by_seed(data)
    keep, got, n = downsample_to(data, cids, args.match_prevalence)
    sub = {k: {c: v[c] for c in keep} for k, v in data.items()}
    print(f"\n  [同仪器] 子采样到 {args.match_prevalence:.4%}: "
          f"{len(cids)} -> {n} 块, 实际流行率 {got:.4%}")
    rs = report(f"{args.set} (matched)", sub, args.boot)
    res["matched"] = rs

    out = f"{ANNOT}/g1_c_analyze_{args.set}.json"
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
