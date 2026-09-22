"""纯重跑噪声 (pure replication) 在 margin 空间有多大 —— s42 vs ds42 这一对。

## 这一对为什么是"纯重跑"

`g1_<arm>_sft`      是 seed=42, data_seed=None
`g1_<arm>_ds42_sft` 是 seed=42, data_seed=42

`SeedableRandomSampler` 里 `data_seed=None` 等价于 `set_seed(42)` 后的 42,
所以**两者是同一份有效配置**, 只是两次独立训练。两次的差因此**全部**来自:
  - `full_determinism=False` 下 cuBLAS/cuDNN 的非确定性 kernel (浮点重排)
  - 以及由它级联出来的 dropout / 数据顺序的微小分歧
（已核: 两个训练目录 `g1_margin_all.sh` 用 `g1_${arm}_sft`,
  `g1_margin_ds.sh` 用 `g1_${arm}_ds${s}_sft` —— 是**两个不同的目录都在盘上**,
  所以这不是"同一个权重探了两次"的不确定性, 是**两次真训练**的差。）

## 它为什么重要 —— 第三个方差源

§7.4f 把 margin 的跨适配器方差拆成 **初值 (A 组)** vs **数据顺序 (B 组)**。
这一对测的是**第三样东西**, 两者都不变时仍然存在的底噪。
它是 A、B **共同**的底噪, 所以会同时抬高两组的 SD ——
**B/A 比值**因此对它近似免疫 (下面打印这个修正量)。

## ⚠️ n = 1 对 —— 这条纪律必须写在最前面

只有**一对**, 所以:
  - **没有误差棒**。Δmean 是**一次抽样**, 不是 SD 的估计。
  - 能说「这一次位移占组内 SD 的 x%」, **不能**说「重跑噪声的 SD 是 x%」。
  - 「A、B 两组的底噪同量级」是**假设**, 不是本脚本测出来的 (n=1 根本测不了)。
(memory: sd-from-few-points-manufactures-anomalies —— 点数太少时,
 噪声会被包装成发现; 这里反过来, 用一个点去主张一个 SD 同样不行。)

用法: python scripts/g1_replicate_margin_pair.py
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


def load(arm, grp, k):
    p = (f"{ANNOT}/g1_margin_{arm}_s{k}.json" if grp == "A"
         else f"{ANNOT}/g1_margin_{arm}_ds{k}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    print("=" * 78)
    print("  纯重跑噪声 (margin 空间): s42 vs ds42 —— 同一份有效配置的两次独立训练")
    print("=" * 78)
    print("  ⚠️ n = **1 对**: 只报量级与占比, 不报 SD、不配误差棒。")

    pair, groups = {}, {}
    for arm in ARMS:
        pair[arm] = (load(arm, "A", 42), load(arm, "B", 42))
        groups[arm] = {
            "A": [d for d in (load(arm, "A", s) for s in SEEDS) if d],
            "B": [d for d in (load(arm, "B", s) for s in SEEDS) if d],
        }

    # ---------- 组内 SD (对照基准) ----------
    # 口径与 §5.1 / §7.4f **完全一致**: 每适配器 → 该适配器在块上的平均 margin → 跨适配器求 SD。
    print(f"\n  ── 对照基准: 各组跨适配器 SD (口径同 §5.1) ──")
    gsd = {}
    for arm in ARMS:
        line = f"    {arm:<8}"
        for g in ("A", "B"):
            ds = groups[arm][g]
            if not ds:
                line += f"  {g}: (无产物)"
                continue
            cids = sorted(set.intersection(*(set(d) for d in ds)))
            m = [np.mean([d[c]["margin_max"] for c in cids]) for d in ds]
            gsd[(arm, g)] = (float(np.std(m, ddof=1)), len(ds), len(cids))
            line += f"  {g}: SD {np.std(m, ddof=1):.3f} (n={len(ds)}, {len(cids)}块)"
        print(line)

    # ---------- 主量 ----------
    print(f"\n  ── 逐块 Δ = margin(ds42) − margin(s42) ──")
    print(f"    {'arm':<8}{'共同块':>8}{'Δmean':>10}{'SD(Δ)':>10}"
          f"{'中位|Δ|':>10}{'判定翻转':>10}")
    delta = {}
    for arm in ARMS:
        d1, d2 = pair[arm]
        if d1 is None or d2 is None:
            print(f"    {arm:<8} (缺产物)")
            continue
        cids = sorted(set(d1) & set(d2))
        a = np.array([d1[c]["margin_max"] for c in cids])
        b = np.array([d2[c]["margin_max"] for c in cids])
        dd = b - a
        flip = np.mean([d1[c]["argmax_is_event"] != d2[c]["argmax_is_event"]
                        for c in cids])
        delta[arm] = (dd, a, b, cids)
        print(f"    {arm:<8}{len(cids):>8}{dd.mean():>+10.4f}{dd.std(ddof=1):>10.4f}"
              f"{np.median(np.abs(dd)):>10.4f}{flip:>9.1%}")

    # ---------- 占比: 相对两组 SD ----------
    print(f"\n  ── 一次重跑位移 |Δmean| 相对组内 SD 的占比 ──")
    print(f"    {'arm':<8}{'组':>4}{'组内 SD':>10}{'|Δmean|':>10}{'占比':>9}")
    for arm in ARMS:
        if arm not in delta:
            continue
        dm = abs(delta[arm][0].mean())
        for g in ("A", "B"):
            if (arm, g) not in gsd:
                continue
            s = gsd[(arm, g)][0]
            print(f"    {arm:<8}{g:>4}{s:>10.3f}{dm:>10.4f}{dm/s:>9.1%}")

    # ---------- Δmean 是不是「整条边界挪了」而不是「几个块抖了」 ----------
    # ⚠️ 这里**不能**用块级 SE 报 p 值: 400 块是**重叠窗 + 同一批录音**,
    #    彼此不独立, 有效 n 远小于 400 ⇒ 块级 t 检验的 p 会**系统性乐观**。
    #    (memory: threshold-must-match-null-model —— 阈值/检验必须配它的零模型;
    #     这里零模型不是「400 个独立抽样」而是「几十个相关块」。)
    #    所以只报**描述量**, 不报 p: 看位移是不是**一致同号**,
    #    以及它与块级散布 (median|Δ|) 的相对大小。
    print(f"\n  ── Δmean 是『整条边界挪了』还是『几个块抖了』──")
    print(f"    (不报 p 值: 400 块是重叠窗、彼此不独立, 块级 SE 会让 p 系统性乐观)")
    for arm in ARMS:
        if arm not in delta:
            continue
        dd = delta[arm][0]
        pos = float(np.mean(dd > 0))
        se = dd.std(ddof=1) / np.sqrt(len(dd))
        print(f"    {arm:<8} 同号率 {pos:.1%}   Δmean {dd.mean():+.4f}   "
              f"块级散布 median|Δ| {np.median(np.abs(dd)):.4f}")
        print(f"             ⇒ 位移 = 小幅**一致平移** ({dd.mean():+.4f}) "
              f"叠加大量块级散布 ({dd.std(ddof=1):.4f})。")
        print(f"             🔴 但**别因此小看它**: 适配器层面的统计量是 400 块的**均值**,")
        print(f"                块级散布平均后只剩 {se:.4f} (÷√400), 而那个一致平移")
        print(f"                **完全不被平均掉** ⇒ 在适配器层面它比块噪声大 "
              f"{abs(dd.mean())/se:.0f}×。")
        print(f"                这就是它能在组内 SD 里占到几十个百分点的原因。")
    print(f"    ⚠️ 两臂**都是正的** 不构成共同原因的证据: 2/2 同号")
    print(f"       在纯随机下 p = 0.25, 单靠这个说明不了什么。")

    # ---------- 是不是「块级共同漂移」 ----------
    print(f"\n  ── 两臂的逐块位移是否同向 (= 块级共同漂移)? ──")
    if all(a in delta for a in ARMS):
        c = sorted(set(delta[ARMS[0]][3]) & set(delta[ARMS[1]][3]))
        x = np.array([dict(zip(delta[ARMS[0]][3], delta[ARMS[0]][0]))[k] for k in c])
        y = np.array([dict(zip(delta[ARMS[1]][3], delta[ARMS[1]][0]))[k] for k in c])
        r = float(np.corrcoef(x, y)[0, 1])
        rcrit = 1.96 / np.sqrt(len(c))          # n=400 时双侧 0.05 的近似临界
        print(f"    corr(Δown10, Δmixnorm) 逐块 = {r:+.4f}  (n={len(c)} 块,"
              f" 近似临界 ±{rcrit:.3f} ⇒ {'显著' if abs(r) > rcrit else '不显著'})")
        print(f"    R² = {r*r:.4f} ⇒ 位移里只有 {r*r:.1%} 能被『块共同难易』解释")
        # 对照: 同一 run 内两臂的**水平**相关 —— 若位移是共同漂移, 两者该同量级
        d1o, d1m = pair["own10"][0], pair["mixnorm"][0]
        c2 = sorted(set(d1o) & set(d1m))
        lo = np.array([d1o[k]["margin_max"] for k in c2])
        lm = np.array([d1m[k]["margin_max"] for k in c2])
        rl = float(np.corrcoef(lo, lm)[0, 1])
        print(f"    对照: 同一 run 内两臂 margin **水平**相关 = {rl:+.4f}"
              f"  ⇒ 位移相关只有它的 {abs(r)/abs(rl):.0%}")
        print(f"    ⇒ 位移**基本是各臂自己的 run 噪声**, 不是块级共同难易。")

    # ---------- 对 B/A 比值的影响 (本脚本最该说清的一句) ----------
    print(f"\n  ── 这个底噪对 §7.4f 的 B/A 比值有什么影响 ──")
    print(f"    纯重跑噪声是 A、B **共同**的底噪, 若记它占观测方差的 f,")
    print(f"    则两组 SD 各被抬高约 1/√(1−f), **比值不变**。")
    print(f"    以 own10 的 |Δmean| 最粗地当上界 (它把一次位移当成了整份噪声):")
    for arm in ARMS:
        if arm not in delta:
            continue
        dm = abs(delta[arm][0].mean())
        for g in ("A", "B"):
            if (arm, g) not in gsd:
                continue
            s = gsd[(arm, g)][0]
            f = (dm / s) ** 2          # 一个**上界**式估计: 把一次位移当成噪声 SD
            print(f"      {arm:<8}{g} 组: f ≤ {f:.3f} ⇒ σ 最多缩到 "
                  f"{np.sqrt(1-f):.3f} × 组内 SD ({s:.3f} → {s*np.sqrt(1-f):.3f})")
    print(f"    🔴 但『A、B 两组底噪同量级』是**假设** —— n=1 对测不出这个, 别当结论用。")

    # ---------- 与 ΔF1 空间对照 ----------
    print(f"\n  ── 对照 §7.4c: 同一对在 **ΔF1** 空间是 0.0040 (= 跨种子 SD 的 13%) ──")
    print(f"    margin 空间的占比明显更大, 与『边界散 ≠ F1 散』同向")
    print(f"    (边界在动, 但 ROC 在移动区间是平的 ⇒ 不传到 F1)。")
    print(f"    ⚠️ 两个空间的口径不同 (margin 是 400 块 E2 / F1 是 filt 集),")
    print(f"       受 n=1 限制, **只能并排看方向, 不能当成一个定量的比值**。")

    # ---------- 设计教训: 这个对是**偶然**存在的 ----------
    print(f"\n  ── 🔴 设计教训: 这个纯重复对是**偶然**存在的, 不是设计出来的 ──")
    print(f"    整套 G1 里**只有这一对** s42/ds42, 它存在的原因仅仅是")
    print(f"    `data_seed=None` 恰好在 `SeedableRandomSampler` 里等价于 42。")
    print(f"    换言之: **噪声底是我们撞上的, 不是量出来的。**")
    print(f"    Q4 看起来像第二个重复, 但它是 `train_textonly` (带 DisableSpeechCallback,")
    print(f"    少跑一次 audio_invert_tower 前向) ⇒ RNG 流被扰动, 是**混沌敏感度**不是复现性。")
    print(f"    ⇒ 下次设计消融时, **把重复 run 写进矩阵** (每个配置跑 2 次),")
    print(f"      否则组内 SD 里永远混着一个无法分离的底噪。")
    print(f"      本脚本能把底噪量出来, 纯属运气好。")


if __name__ == "__main__":
    main()
