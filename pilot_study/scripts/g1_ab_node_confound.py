"""§7.4f 的 A/B 分组是否与「训练节点 / 日期」混杂 —— 以及能做哪些检查。

## 发现 (2026-09-18 02:0x)

从每个 adapter 的 `training_args.bin:logging_dir` 解出训练时刻与节点:

    own10    A 组  节点 ['gpu04']             日期 Sep10/12/13
    own10    B 组  节点 ['gpu01']             日期 Sep17
    mixnorm  A 组  节点 ['gpu01','gpu04']     日期 Sep10/12/16
    mixnorm  B 组  节点 ['gpu01']             日期 Sep17

**own10 的 A/B 与节点完全重合** —— A 全是 gpu04、B 全是 gpu01。
所以「B/A = 1.52」测的不只是「数据顺序 vs 初值」, 还混着
「gpu04 上训的 vs gpu01 上训的」(以及 Sep10-13 vs Sep17)。

## 为什么 SD 之比**部分**免疫

- **纯平移不进门**: 若两个节点只差一个常数偏移, 那么它只移动**均值**,
  组内 SD 不变 ⇒ B/A 不受影响。
- **只有「方差本身不同」才进门**: 若某个节点的 run 之间彼此更像
  (例如同一天连续启动、驱动/热状态更接近), 那一组的 SD 会被**压低**,
  B/A 就跟着偏。这不是理论顾虑 —— B 组全部是 Sep17 同一天连续几小时跑的。

## 能做的唯一杠杆

mixnorm 的 A 组**同时含两个节点** (42/1234 在 gpu04, 其余 5 个在 gpu01),
所以可以在 A 组内部做**留组检查**: 去掉那两个 gpu04 的点, 看 SD 崩不崩。

⚠️ **n=2 不许算 SD 比** (memory: sd-from-few-points-manufactures-anomalies)。
所以只报「去掉后 A 组 SD 变成多少」, **不报**「gpu04 组 vs gpu01 组的 SD 比」。
这是一个**探索性**的混杂检查, 不是预登记判据 —— 结论只能用来**限定** §7.4f,
不能用来改判。

own10 的 A 组全是 gpu04 ⇒ **它没有任何组内杠杆**, 这个检查对它做不了。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ⚠️ 一律 **import**, 不重写 (本项目约定: 判据类函数只此一处定义)。
#    `var_ratio_perm` 在 `g1_boundary_dispersion.py` 里固定 seed=0, 可复跑;
#    自己抄一份就会像 0.0160/0.0161 那样长出第二个"结果"来。
from g1_boundary_dispersion import var_ratio_perm  # noqa: E402

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
SEEDS = [42, 1234, 7, 2024, 3407, 31337, 55555]
ARMS = ["own10", "mixnorm"]

# 从 logging_dir 解出的事实 (2026-09-18 02:0x, 见 _train_where.py)。
# ⚠️ 必填 **(arm, grp) 两级** —— 第一版只按 arm 给了一个 {42,1234},
#    于是对 own10 的 A 组也去掉了 42/1234, 并印出「A 组的离散不是那两个
#    gpu04 点撑的」。**那句话是假的**: own10 的 A 组**全部**是 gpu04,
#    根本没有「非 gpu04 的点」可去 —— 去掉的只是两个任意点。
#    留组检查只有在**一个组里同时含两个节点**时才有意义。
ON_GPU04 = {
    ("own10", "A"): {42, 1234, 7, 2024, 3407, 31337, 55555},   # 全 gpu04
    ("own10", "B"): set(),                                      # 全 gpu01
    ("mixnorm", "A"): {42, 1234},                               # gpu04 只有这 2 个
    ("mixnorm", "B"): set(),                                    # 全 gpu01
}


# 两个集合的文件名前缀只差 `nat_`, 但**参数顺序曾经写反过**
# (E2 那份是 (arm, grp, k), 自然集那份是 (arm, k, grp), 调用处却统一写成
#  (arm, s, grp)) —— 结果两边都静默读不到文件、全报「产物不齐」。
# 统一成一个签名, 前缀做成参数, 从结构上消掉这个错。
def load(arm, grp, k, nat=False):
    pre = "g1_margin_nat_" if nat else "g1_margin_"
    p = (f"{ANNOT}/{pre}{arm}_s{k}.json" if grp == "A"
         else f"{ANNOT}/{pre}{arm}_ds{k}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def blockmeans(ds):
    cids = sorted(set.intersection(*(set(d) for d in ds)))
    return np.array([np.mean([d[c]["margin_max"] for c in cids]) for d in ds])


def main():
    print("=" * 78)
    print("  §7.4f A/B 分组 vs 训练节点/日期 —— 混杂检查")
    print("=" * 78)
    print("  节点归属 (来自 training_args.bin):")
    print("    own10   A=gpu04 全, B=gpu01 全   ⇒ **完全重合, 组内无杠杆**")
    print("    mixnorm A=gpu04{42,1234} + gpu01{其余5}, B=gpu01 全")
    print()

    for tag, nat, grps in (("E2-400 (100% 正例)", False, ("A", "B")),
                           ("自然 300", True, ("A", "B"))):
        print(f"  ── {tag} ──")
        for arm in ARMS:
            out = {}
            for grp in grps:
                ds = [d for d in (load(arm, grp, s, nat) for s in SEEDS) if d]
                if len(ds) != len(SEEDS):
                    out[grp] = None
                    continue
                out[grp] = blockmeans(ds)
            if out.get("A") is None or out.get("B") is None:
                print(f"    {arm:<8} (产物不齐, 跳过)")
                continue
            va, vb = out["A"], out["B"]
            sa, sb = np.std(va, ddof=1), np.std(vb, ddof=1)
            hdr = (f"    {arm:<8} A 组 SD {sa:.3f}   B 组 SD {sb:.3f}   "
                   f"B/A {sb/sa:.2f}")
            print(hdr)
            # 留组检查: **只有当一个组里同时含两个节点**时才可做。
            g4 = ON_GPU04[(arm, "A")]
            if not g4 or len(g4) == len(SEEDS):
                which = "全 gpu04" if g4 else "全 gpu01"
                print(f"             ⛔ **做不了留组检查** —— A 组{which}, "
                      f"组内没有第二个节点可去掉。")
                print(f"                本设计的 A/B 在这里**完全重合**, 混杂分不开。")
                continue
            keep = [i for i, s in enumerate(SEEDS) if s not in g4]
            if len(keep) < 3:
                print(f"             (保留点 <3, 不做留组)")
                continue
            sa2 = np.std(va[keep], ddof=1)
            print(f"             留组: 去掉 A 组里 gpu04 的 {len(g4)} 个点 "
                  f"({sorted(g4)}, 剩 n={len(keep)}):")
            print(f"               A 组 SD {sa:.3f} -> {sa2:.3f}   "
                  f"B/A {sb/sa:.2f} -> {sb/sa2:.2f}")
            if abs(sa2 - sa) / sa > 0.30:
                print(f"               ⚠️ A 组 SD 掉了 {abs(sa2-sa)/sa:.0%} ⇒ "
                      f"**节点/批次可能确实在撑 A 组的离散**, B/A 要打折看")
            else:
                print(f"               ✅ A 组 SD 只动了 {abs(sa2-sa)/sa:.0%} ⇒ "
                      f"A 组的离散**不是那两个 gpu04 点撑的**")
            print(f"             (⚠️ 仅探索性; n={len(keep)} 的 SD 本身很不稳, 只当方向看)")
            print(f"             ⚠️ 这只查了「A 组被 gpu04 点撑大」这一个方向;")
            print(f"                「B 组全是同一天同一节点的连续 run ⇒ 彼此更像 ⇒ SD 被压低」")
            print(f"                这个反方向**没有任何杠杆**可查 (B 组组内无节点变化)。")

            # 更有用的一步: A 组的 **gpu01 子集** 与 B 组(全 gpu01) 比 —— 节点配平了。
            # ⚠️ 残余的仍是「Sep16 vs Sep17」这一天; 但节点这一维被消掉了。
            sb2 = np.std(vb, ddof=1)
            print(f"             🔧 **节点+场次配平的 B/A**: 用 A 组的 gpu01 子集 "
                  f"(n={len(keep)}, Sep16) 对 B 组 (全 gpu01, Sep17):")
            print(f"                A(gpu01) SD {sa2:.3f}   B SD {sb2:.3f}   "
                  f"B/A {sb2/sa2:.2f}   (原值 {sb/sa:.2f})")
            # SD 自己的 CI 必须一起报 (memory: 报 SD 就报它的卡方区间)
            from scipy import stats as st
            def ci(v):
                n_, s_ = len(v), np.std(v, ddof=1)
                return (s_ * np.sqrt((n_-1)/st.chi2.ppf(0.975, n_-1)),
                        s_ * np.sqrt((n_-1)/st.chi2.ppf(0.025, n_-1)))
            la, ha = ci(va[keep])
            lb, hb = ci(vb)
            print(f"                A(gpu01) SD 95%CI [{la:.3f}, {ha:.3f}] (n={len(keep)})")
            print(f"                B        SD 95%CI [{lb:.3f}, {hb:.3f}] (n={len(vb)})")
            print(f"                ⚠️ n=5 的 SD 自身就有 ±60% 量级的不确定度 —— "
                  f"1.05 与 0.97 的差别**远在这条不确定度之内**, 别当两个结果。")
            print(f"                ⇒ 这一配平把**节点**和**场次**两维都消掉了 "
                  f"(两组都是单节点单场次的连续 run), 残余只有 Sep16 vs Sep17。")
        print()

    # ---------- 🔧 场次内核对照: 本设计里最干净的跨臂比 ----------
    # 时间戳 (见 _train_where.py) 显示两边 A 组各有一个**同节点同场次**的
    # 连续 5 连跑内核, 且**种子集合完全相同**:
    #
    #   own10   A: Sep13 22:44-23:58 gpu04  {7,2024,3407,31337,55555}  74 分钟
    #   mixnorm A: Sep16 00:19-01:37 gpu01  {同 5 个}                   78 分钟
    #
    # 未配平的 A 组 (n=7) 两边都混了早先几天/另一节点的点, 而**混进来的
    # 点不一样** (own10 混 Sep10/12 的 gpu04 点, mixnorm 混 Sep10/12 的
    # gpu04 点) —— 于是「A 组的离散」里含一份「跨天混批」的量, 两边不等。
    # 只取内核 ⇒ 两边都是单节点单场次连续 run, 且种子逐一对齐。
    #
    # ⚠️ 残余: own10 内核在 gpu04/Sep13, mixnorm 内核在 gpu01/Sep16 ——
    #    **节点和天都还没配平**, 只是把「组内混批」这一维消掉了。
    SESS = [7, 2024, 3407, 31337, 55555]
    print("  ── 🔧 场次内核对照 (两边各取同节点同场次的连续 5 连跑, 种子逐一对齐) ──")
    for tag, nat in (("E2-400", False), ("自然 300", True)):
        vs = {}
        for arm in ARMS:
            ds = [d for d in (load(arm, "A", s, nat) for s in SESS) if d]
            if len(ds) == len(SESS):
                vs[arm] = blockmeans(ds)
        if len(vs) != 2:
            print(f"    {tag}: (产物不齐, 跳过)")
            continue
        so, sm = np.std(vs["own10"], ddof=1), np.std(vs["mixnorm"], ddof=1)
        _, p = var_ratio_perm(vs["own10"], vs["mixnorm"])
        print(f"    {tag}:  own10 SD {so:.3f}   mixnorm SD {sm:.3f}   "
              f"比 {sm/so:.2f}×   置换 p = {p:.4f}   (n=5 vs n=5)")
    print("    ⚠️ 10 个值分 5/5 只有 C(10,5)=252 种分法 ⇒ p 的**分辨率下限** "
          "≈ 1/252 = 0.004。这条对照的分辨力上限就到这里, 报比值即可。")

    print("  ── 结论能说到哪 ──")
    print("    **纯平移不影响 SD 之比** —— 所以若两节点只差一个常数偏移,")
    print("    B/A 不受影响; 只有「某一节点内部 run 之间更像」才会压低那一组的 SD。")
    print("    🔴 但 own10 的 A/B 与节点**完全重合**, 本设计**分不开**")
    print("    「数据顺序的贡献」与「gpu01-vs-gpu04 / Sep17-vs-Sep10-13 的批次效应」。")
    print("    这个混杂必须写进 §7.4f, 且**不能靠一次留组检查消掉**。")


if __name__ == "__main__":
    main()
