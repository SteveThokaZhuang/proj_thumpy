"""池间 SD 实验的**零模型探针** —— 纯读盘, 0 GPU。跑在预登记之前, 用来定「对照是什么」。

## 为什么需要它

即将做的实验是: 从池子里再抽 J 批、每批 B=400 块, 看 **ΔF1 跨批的 SD** 有多大。
要判读它, 必须有一个「若块在池内可交换, SD 应当是多少」的**零模型**。

项目现成的零模型是 `g1_e4_analyze.boot_se` —— **对块做有放回简单随机重抽(SRS)**。
但评估集的构造**不是 SRS**: `ari_g1_eval4_ids.py` 是**按 (session, ch) 轮转**取样,
每个 key 取同样多块(分层)。分层抽样的方差**低于** SRS —— 所以直接拿 SRS 的
`SE_chunk` 当零模型会**高估**应有的跨批 SD, 把结论推向「ρ<1 ⇐ bootstrap 诚实」。
这是 memory `threshold-must-match-null-model` 那一族: **对照必须与设计同构**。

本脚本把两个零模型都算出来:

  (a) SRS    : 现有 `boot_se`(对块有放回重抽 n 个) —— 项目历史口径
  (b) 分层   : 按 key (session,ch) 分层重抽 —— 与评估集构造同构
               (每个 key 抽 m 个, m = 该集每 key 的块数)

若 (b) ≈ (a) ⇒ 分层无影响, 用哪个都行;
若 (b) ≪ (a) ⇒ **必须用 (b)**, 否则新的跨批 SD 会被 SRS 的过高对照判成「更稳」。

用法 (srun 内, fd_analysis):
  python scripts/g1_pool_null_probe.py --set e4x --set e4
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ⚠️ 口径一律 import, 不重写 (memory: hardcoded-conclusions-escape-reproduction)
from g1_e4_analyze import paired, d_of, f1_of, SEEDS  # noqa: E402

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")


def key_of(cid):
    """'0020a0c5_ch0_t1005' -> ('0020a0c5','0')  —— 与 ari_g1_eval4_ids.parse_id 同口径。"""
    head, ch, _t = cid.rsplit("_", 2)
    return (head, ch[2:])


def strat_boot_se(vec, cids, seeds, m_per_key, boot, rng_seed=0):
    """分层 bootstrap: 每个 key 有放回地重抽 `m_per_key` 个块。

    `m_per_key` = 该评估集每个 key 实际取的块数 (e4x: 8) —— 这样重抽出来的
    集合与真实评估集**同构**(同样分层、同样每 key 块数), 只是换了 key 内的块。
    这正是「再抽一批同样构造的块」的零模型。
    """
    by_key = collections.defaultdict(list)
    for i, c in enumerate(cids):
        by_key[key_of(c)].append(i)
    keys = sorted(by_key)
    per = np.array([by_key[k] for k in keys], dtype=object)
    rng = np.random.default_rng(rng_seed)
    db = np.empty(boot)
    for b in range(boot):
        ii = np.concatenate([rng.choice(per[j], m_per_key, replace=True)
                             for j in range(len(keys))])
        db[b] = np.mean(list(d_of(vec, seeds, ii).values()))
    return float(db.std(ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="sets", action="append", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    sets = args.sets or ["e4x", "e4"]

    for tag in sets:
        D = paired(tag)
        if D is None:
            print(f"=== {tag}: 配对数不足, 跳过")
            continue
        P, seeds, cids, vec = D["P"], D["seeds"], D["cids"], D["vec"]
        n = len(cids)
        by_key = collections.Counter(key_of(c) for c in cids)
        m = collections.Counter(by_key.values())
        d = D["d"]
        dv = np.array([d[s] for s in seeds])
        print(f"\n{'='*74}\n=== {tag}: {n} 块 / {len(by_key)} 个 key / "
              f"每 key 块数分布 {dict(sorted(m.items()))} ===")
        print(f"  ΔF1 = {dv.mean():+.4f}  (n={len(dv)} 种子)")

        # (a) SRS —— 现有口径
        rng = np.random.default_rng(0)
        db = np.empty(args.boot)
        for b in range(args.boot):
            ii = rng.integers(0, n, n)
            db[b] = np.mean(list(d_of(vec, seeds, ii).values()))
        se_srs = float(db.std(ddof=1))

        # (b) 分层 —— 每 key 重抽 m0 个 (m0 = 众数, 正常只有一种取值)
        m0 = m.most_common(1)[0][0]
        se_str = strat_boot_se(vec, cids, seeds, m0, args.boot)

        print(f"  零模型 (a) SRS 重抽 {n} 块      : SE = {se_srs:.5f}   ← 项目历史口径")
        print(f"  零模型 (b) 分层重抽 {len(by_key)} key × {m0} : SE = {se_str:.5f}   ← 与设计同构")
        print(f"  比值 (b)/(a) = **{se_str/se_srs:.3f}**"
              f"   {'✅ 分层无影响' if 0.95 <= se_str/se_srs <= 1.05 else '🔴 分层影响大, 必须用 (b)'}")

        # 每个 key 的块数若不等, (b) 就只是近似 —— 明说
        if len(m) > 1:
            print(f"  ⚠️ 每 key 块数不等 {dict(sorted(m.items()))}, (b) 按众数 {m0} 重抽, 是近似")

        # 附带: 这个集自己的「批」能不能切 (供后续比大小用)
        json.dump({"set": tag, "n_chunks": n, "n_keys": len(by_key),
                   "delta_f1": float(dv.mean()), "se_srs": se_srs,
                   "se_strat": se_str, "ratio": se_str / se_srs},
                  open(f"{ANNOT}/g1_pool_null_probe_{tag}.json", "w"), indent=1)
        print(f"  -> {ANNOT}/g1_pool_null_probe_{tag}.json")


if __name__ == "__main__":
    main()
