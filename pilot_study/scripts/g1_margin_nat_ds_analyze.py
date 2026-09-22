"""§7.4f #72 对账: B/A 的第二格换成**探针 × 自然块集**。

## 预登记在哪

判据写在 `docs/pilot_study/2026-09-17_status.md` **§7.4f** 的
「🔒 #72 预登记」, 封存时刻 **2026-09-18 02:01:33** —— 当时
`g1_margin_nat_{arm}_ds{seed}.json` **一个都还没跑出来**(0/14)。
本脚本**只做对账, 不挑说法**: 判据命中就写命中, 落空就写落空。

**为什么补这一格。** B/A 的两格原本是「代理×自然(0.94)」与「探针×100%正例(0.97)」,
观测量和块集**同时变**。跨臂比那一行干净, 是因为它的三格都是同一个观测量(探针)、
只换块集。这一格就是给 B/A 补上「探针×自然」。

## 统计量 (封存口径, 不得改)

    R_nat = (B_nat / A_nat) ÷ (B_E2 / A_E2)

其中 A 组变 seed (data_seed≡42)、B 组变 data_seed (seed≡42), 各组都取**全部 7 个**。
`B/A_E2` 是探针在 E2-400 上的值 —— 封存时实测 own10 1.52 / mixnorm 0.97。
⚠️ 本脚本**从 E2 产物现算**这两个数并**断言**它们等于封存值:
把封存值直接写进代码, 它就绕过了复现
(memory: hardcoded-conclusions-escape-reproduction)。

## 口径

必须与 `g1_margin_ds_analyze.py` / `g1_margin_natural.py` 逐字一致:
  每适配器 → 该适配器在**共同块**上的 **块平均 margin_max** → 跨适配器求 **SD**。
自然集的采集参数与 A 组那份完全相同(`--ids-file g1_eval_ids.txt --n-chunks 300
--sample-seed 0`), 只换了 lora 与输出名, 所以 A/B 两组的块集应当相同。

用法: python scripts/g1_margin_nat_ds_analyze.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
SEEDS = [42, 1234, 2024, 31337, 3407, 55555, 7]
ARMS = ["own10", "mixnorm"]

# 封存值 —— **只用来断言, 不用来代入计算**
FROZEN_BA_E2 = {"own10": 1.52, "mixnorm": 0.97}
FROZEN_NAT_A_SD = {"own10": 0.180, "mixnorm": 0.546}

# 场次内核 (见 g1_ab_node_confound.py): 两边各 5 连跑, 种子逐一对齐。
# ⚠️ 这一版是**事后**的, 不属于封存判据, 只作附注。
SESS_CORE = [7, 2024, 3407, 31337, 55555]


def load(arm, key, grp, nat=True):
    pre = "g1_margin_nat_" if nat else "g1_margin_"
    p = (f"{ANNOT}/{pre}{arm}_s{key}.json" if grp == "A"
         else f"{ANNOT}/{pre}{arm}_ds{key}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def blockmeans(M, keys, cids):
    return np.array([np.mean([M[k][c]["margin_max"] for c in cids])
                     for k in keys])


def sd_ci(v):
    from scipy import stats as st
    n, s = len(v), np.std(v, ddof=1)
    return (s, s * np.sqrt((n - 1) / st.chi2.ppf(0.975, n - 1)),
            s * np.sqrt((n - 1) / st.chi2.ppf(0.025, n - 1)))


def main():
    print("=" * 78)
    print("  §7.4f #72 对账: B/A 的探针 × 自然块集那一格")
    print("=" * 78)

    # ---------- 载入两套块集 ----------
    M, missing = {}, []
    for nat, tag in ((True, "nat"), (False, "E2")):
        for grp in ("A", "B"):
            for arm in ARMS:
                for s in SEEDS:
                    d = load(arm, s, grp, nat)
                    if d is None:
                        missing.append(f"{tag}:{arm}:{grp}:{s}")
                    else:
                        M[(tag, grp, arm, s)] = d
    n_expect = 2 * 2 * 2 * len(SEEDS)
    print(f"\n  产物 {len(M)}/{n_expect}" + (f"   ❌ 缺: {missing}" if missing else "   ✅"))
    if missing:
        print(f"  🔴 **数据不完整 —— 下面的判据一律不作数**, 只当管线自检看。")
        print(f"     (memory: 跑批没结束就不算最终统计量)")

    # ---------- 自检 0: 各自块集内的共同块 ----------
    cids = {}
    for tag in ("nat", "E2"):
        cids[tag] = sorted(set.intersection(
            *(set(d) for k, d in M.items() if k[0] == tag)))
        sizes = sorted({len(d) for k, d in M.items() if k[0] == tag})
        print(f"\n  ── 自检 0 ({tag}): 各文件块数 {sizes}   交集 {len(cids[tag])} "
              f"{'✅' if sizes == [len(cids[tag])] else '⚠️ 取交集'}")

    # ---------- 自检 1: 自然集 A 组应复现封存时印下的 SD ----------
    print(f"\n  ── 自检 1: 自然集 A 组 SD 应等于封存值 (现算, 不代入) ──")
    ok1 = True
    for arm in ARMS:
        v = blockmeans(M, [("nat", "A", arm, s) for s in SEEDS], cids["nat"])
        s = float(np.std(v, ddof=1))
        f = FROZEN_NAT_A_SD[arm]
        good = abs(s - f) < 0.002
        ok1 &= good
        print(f"    {arm:<8} " + " ".join(f"{x:+.3f}" for x in v)
              + f"   SD {s:.3f}  (封存 {f:.3f})  {'✅' if good else '❌'}")
    print(f"    {'✅ 与封存一致' if ok1 else '🔴 对不上 —— 口径被改过了, 停'}")

    # ---------- 自检 2: E2 的 B/A 应复现封存值 1.52 / 0.97 ----------
    print(f"\n  ── 自检 2: E2 的 B/A 应等于封存值 (现算, 不代入) ──")
    ok2 = True
    ba = {}
    for arm in ARMS:
        a = blockmeans(M, [("E2", "A", arm, s) for s in SEEDS], cids["E2"])
        b = blockmeans(M, [("E2", "B", arm, s) for s in SEEDS], cids["E2"])
        r = float(np.std(b, ddof=1) / np.std(a, ddof=1))
        ba[arm] = r
        f = FROZEN_BA_E2[arm]
        good = abs(r - f) < 0.015
        ok2 &= good
        print(f"    {arm:<8} A SD {np.std(a,ddof=1):.3f}   B SD {np.std(b,ddof=1):.3f}"
              f"   B/A {r:.2f}  (封存 {f:.2f})  {'✅' if good else '❌'}")
    print(f"    {'✅ 与封存一致' if ok2 else '🔴 对不上 —— 停, 先查口径'}")

    # ---------- 主量: R_nat ----------
    print(f"\n  ── 🔒 主量 R_nat = (B_nat/A_nat) ÷ (B_E2/A_E2) ──")
    print(f"    {'arm':<8}{'A_nat':>8}{'B_nat':>8}{'B/A nat':>10}"
          f"{'B/A E2':>9}{'R_nat':>9}   判读")
    verdict = {}
    for arm in ARMS:
        a = blockmeans(M, [("nat", "A", arm, s) for s in SEEDS], cids["nat"])
        b = blockmeans(M, [("nat", "B", arm, s) for s in SEEDS], cids["nat"])
        sa, sb = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
        r_nat = (sb / sa) / ba[arm]
        if 0.7 <= r_nat <= 1.4:
            v = "∈[0.7,1.4] ⇒ B/A **跨两个块集都成立**"
        elif r_nat < 0.5 or r_nat > 1.6:
            v = "<0.5 或 >1.6 ⇒ B/A **对块集敏感**, ③ 撤回至仅 E2-400"
        elif (0.5 < r_nat <= 0.7) or (1.4 <= r_nat < 1.6):
            v = "灰区 ⇒ 不改判, 只记录"
        else:
            v = "兜底 ⇒ 不改判"
        verdict[arm] = (r_nat, v)
        print(f"    {arm:<8}{sa:>8.3f}{sb:>8.3f}{sb/sa:>10.2f}"
              f"{ba[arm]:>9.2f}{r_nat:>9.2f}   {v}")

    # ---------- 附注: 场次内核版 (事后, 不属封存判据) ----------
    print(f"\n  ── 附注(事后): 场次内核版 (两边各取 5 连跑, 种子对齐) ──")
    print(f"     不属封存判据; 封存统计量用的是全部 7 个。")
    kc = [("nat", "A", arm, s) for arm in ARMS for s in SESS_CORE]
    if all(k in M for k in kc) and all(
            ("nat", "B", arm, s) in M for arm in ARMS for s in SESS_CORE):
        for arm in ARMS:
            a = blockmeans(M, [("nat", "A", arm, s) for s in SESS_CORE], cids["nat"])
            b = blockmeans(M, [("nat", "B", arm, s) for s in SESS_CORE], cids["nat"])
            print(f"     {arm:<8} A SD {np.std(a,ddof=1):.3f}   B SD {np.std(b,ddof=1):.3f}"
                  f"   B/A = {np.std(b,ddof=1)/np.std(a,ddof=1):.2f}   (n=5 vs n=5)")
        print(f"     ⚠️ n=5 的 SD 自身 CI 很宽, 只看方向")
    else:
        print(f"     (场次内核的 nat 产物不齐, 跳过)")

    # ---------- 判据表 (照抄封存页, 便于对读) ----------
    print(f"\n  ── 封存判据表 (2026-09-18 02:01:33, 逐字) ──")
    for row in ("∈ [0.7, 1.4]  ⇒ probe 口径的 B/A 跨两个块集都成立 ⇒ ③ 的限定词"
                "从「在 100% 正例的块集上」放宽为「在两个块集上都成立」",
                "< 0.5 或 > 1.6 ⇒ B/A 对块集敏感 ⇒ ③ 撤回至「仅 E2-400 上成立」",
                "(0.5, 0.7] ∪ [1.4, 1.6] ⇒ 灰区 ⇒ 不改判, 只记录",
                "以上都不满足 (产物不齐 / NaN / 其它) ⇒ 不改判 (兜底)"):
        print(f"    {row}")

    print(f"\n  ── 🔴 这个设计补不上什么 (封存时已写) ──")
    print(f"    E2-400 与 filt-300 是**不同的录音、不同的时刻**, 流行率与「块身份」")
    print(f"    在这里**不可分离** —— 标签是块的属性, 而 E2 按构造就是 100% 正例,")
    print(f"    **不存在自然流行率的子集**。所以这是一个**块集稳健性检验**,")
    print(f"    **不是流行率隔离**。跨臂比那一行有同样的残余混淆 ——")
    print(f"    它「隔离」的是**观测量**这一维, 不是流行率。")


if __name__ == "__main__":
    main()
