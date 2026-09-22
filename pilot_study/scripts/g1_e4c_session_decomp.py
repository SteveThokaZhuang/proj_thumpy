"""会话这一路方差有多大? —— 零 GPU, 只用已落盘的 per_chunk 计数。

## 为什么必须问

§2.10 直接量了 `SE(N)`: 抽 N 轮 → 取并集 → 算 ΔF1 → B 次 → 取 SD, 反解出
σ̂ = 0.0201, 三档一致 ⇒ 判「轮可交换、无成簇」。

**但那个检验看不见会话。** e4c 的抽法是**轮转**: 每轮从 200 个
(session, channel) key 里各取 1 块。于是**每一轮都包含全部 100 个会话** ——
会话构成在轮与轮之间**完全相同**。若存在"某些会话上效应大、某些会话上没有"
这种结构, 它在轮间方差里**被抵消掉了**, 因为换一轮并不换会话。

⇒ σ̂ = 0.0201 是**块(会话内) + 种子**的方差, 会话那一项**结构上不可见**。
   这正是 [[permutation-test-covers-one-variance-source]] 的形状:
   把一个因子固定住, 它就永远不进零分布。第三次犯, 换了个端点。

## 做法 (两条, 互为正交)

① **簇 bootstrap over 会话** —— 有放回抽 100 个会话(带重数), 把它们的
   per-chunk 计数按重数相加, 在**并集**上算 F1 (F1 是比值, 不能先算再平均),
   ΔF1 再对 7 个种子取均值。B 次 → SD。这就是**含会话聚类**的 SE。
   同时做只抽种子的版本, 两向一起抽的版本, 三者对比。

② **逐会话 jackknife** —— 去掉一个会话, 在剩下的并集上重算 ΔF1。
   100 个 LOO 值的散布 = 单个会话的影响力量级; 挑出最影响的几个会话。
   若少数会话决定结论, 那这个效应就**不是会话级别的性质**。

③ **逐会话 ΔFP / ΔTP 的符号** —— 每个会话上 own10 与 mixnorm 谁多报。
   分布对称 ⇒ 效应不在会话这一层。

⚠️ 不报"每个会话的 ΔF1": 4800 块只有 189 个真值事件, 摊到 100 个会话是
**~1.9 个/会话** ⇒ 绝大多数会话 F1 退化(无事件或 0 命中), 报出来是
[[sd-from-few-points-manufactures-anomalies]] 的第六种形态。**只在并集上算。**

用法 (srun 内, fd_analysis): python scripts/g1_e4c_session_decomp.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
SEEDS = ["42", "1234", "7", "2024", "3407", "31337", "55555"]
NB = 12
B = 600


def tag(arm, seed, b):
    sfx = "" if seed == "42" else f"_s{seed}"
    return f"{arm}{sfx}_e4c{b:02d}"


def load_counts():
    """-> counts[arm][seed][cid] = (tp, fp, n_gt)。n_gt 两臂相同, 取 own 的。"""
    counts = {"own10": {s: {} for s in SEEDS}, "mixnorm": {s: {} for s in SEEDS}}
    seen_ngt = {}
    for arm in ("own10", "mixnorm"):
        for s in SEEDS:
            for b in range(NB):
                f = f"{ANNOT}/g1_eval_{tag(arm, s, b)}.json"
                d = json.load(open(f))
                pc = next(iter(d.values()))["per_chunk"]
                for cid, v in pc.items():
                    counts[arm][s][cid] = (v["tp"], v["fp"], v["n_gt"])
                    if cid in seen_ngt:
                        assert seen_ngt[cid] == v["n_gt"], \
                            f"{cid} 的 n_gt 在两臂/两批间不一致"
                    seen_ngt[cid] = v["n_gt"]
    return counts, seen_ngt


def f1_from(tp, fp, ngt):
    fn = ngt - tp
    den = 2 * tp + fp + fn
    return 0.0 if den == 0 else 2 * tp / den


def main():
    counts, seen_ngt = load_counts()
    cids = sorted(seen_ngt)
    assert len(cids) == 4800, f"应有 4800 块, 实得 {len(cids)}"
    sess_of = {c: c.split("_")[0] for c in cids}
    sessions = sorted({sess_of[c] for c in cids})
    by_sess = {s: [c for c in cids if sess_of[c] == s] for s in sessions}
    n_sess = len(sessions)
    print(f"e4c: {len(cids)} 块 / {n_sess} 个会话 / {len(SEEDS)} 个种子")
    print(f"  会话块数: 中位 {int(np.median([len(v) for v in by_sess.values()]))}, "
          f"范围 {min(len(v) for v in by_sess.values())}–"
          f"{max(len(v) for v in by_sess.values())}")

    # ---- 预先把每个会话 × 每个种子 × 每臂 的计数聚合好 ----
    # agg[arm][seed][sess] = (tp, fp, ngt)
    agg = {arm: {s: {} for s in SEEDS} for arm in ("own10", "mixnorm")}
    for arm in ("own10", "mixnorm"):
        for s in SEEDS:
            cc = counts[arm][s]
            for sess in sessions:
                tp = fp = ngt = 0
                for c in by_sess[sess]:
                    t, f_, g = cc[c]
                    tp += t; fp += f_; ngt += g
                agg[arm][s][sess] = (tp, fp, ngt)

    def df1(mult):
        """mult: {sess: 重数} -> 7 个种子均值下的 ΔF1 (在**并集**上算一次)。"""
        vals = []
        for s in SEEDS:
            r = {}
            for arm in ("own10", "mixnorm"):
                tp = fp = ngt = 0
                for sess, m in mult.items():
                    if not m:
                        continue
                    t, f_, g = agg[arm][s][sess]
                    tp += m * t; fp += m * f_; ngt += m * g
                r[arm] = f1_from(tp, fp, ngt)
            vals.append(r["own10"] - r["mixnorm"])
        return float(np.mean(vals)), np.array(vals)

    full_mult = {s: 1 for s in sessions}
    full, per_seed = df1(full_mult)
    print(f"\n全池 ΔF1 = {full:+.4f}  (逐种子 "
          f"{', '.join(f'{v:+.4f}' for v in per_seed)})")

    # ---------- ① 簇 bootstrap ----------
    print(f"\n=== ① bootstrap (B={B}) ===")
    rng = np.random.default_rng(0)
    idx = np.arange(n_sess)
    sess_arr = np.array(sessions)

    def boot(resample_sess, resample_seed):
        vals = np.empty(B)
        for b in range(B):
            if resample_sess:
                pick = sess_arr[rng.choice(idx, n_sess, replace=True)]
                mult = {s: 0 for s in sessions}
                for s in pick:
                    mult[s] += 1
            else:
                mult = full_mult
            if resample_seed:
                sub = [SEEDS[i] for i in rng.choice(len(SEEDS), len(SEEDS),
                                                    replace=True)]
                out = []
                for s in sub:
                    r = {}
                    for arm in ("own10", "mixnorm"):
                        tp = fp = ngt = 0
                        for sess, m in mult.items():
                            if not m:
                                continue
                            t, f_, g = agg[arm][s][sess]
                            tp += m * t; fp += m * f_; ngt += m * g
                        r[arm] = f1_from(tp, fp, ngt)
                    out.append(r["own10"] - r["mixnorm"])
                vals[b] = float(np.mean(out))
            else:
                vals[b], _ = df1(mult)
        return vals

    v_sess = boot(True, False)
    v_seed = boot(False, True)
    v_both = boot(True, True)
    se_sess = float(v_sess.std(ddof=1))
    se_seed = float(v_seed.std(ddof=1))
    se_both = float(v_both.std(ddof=1))
    print(f"  只重抽会话 : SE = {se_sess:.5f}   均值 {v_sess.mean():+.4f}")
    print(f"  只重抽种子 : SE = {se_seed:.5f}   均值 {v_seed.mean():+.4f}")
    print(f"  两向都重抽 : SE = {se_both:.5f}   均值 {v_both.mean():+.4f}")
    print(f"  §2.10 解析式 (块+种子, 不含会话) : SE = 0.00557")
    print(f"\n  ⇒ t(含会话) = {full/se_both:.2f}    "
          f"t(不含会话, §2.10 口径) = {full/0.00557:.2f}")

    # 拆开看: 会话项能不能解释两向与只抽种子的差
    extra = np.sqrt(max(se_both ** 2 - se_seed ** 2, 0.0))
    print(f"  合成: √(SE_种子² + SE_会话²) = "
          f"{np.hypot(se_seed, se_sess):.5f} (两向实得 {se_both:.5f})")
    print(f"  由两向反推的**会话增量** ≈ {extra:.5f}")

    # ---------- ② 逐会话 jackknife ----------
    print(f"\n=== ② 逐会话 jackknife (LOO) ===")
    loo = np.empty(n_sess)
    for i, sess in enumerate(sessions):
        m = dict(full_mult); m[sess] = 0
        loo[i], _ = df1(m)
    print(f"  LOO ΔF1: 均值 {loo.mean():+.4f}, SD {loo.std(ddof=1):.5f}, "
          f"范围 [{loo.min():+.4f}, {loo.max():+.4f}]")
    # 影响 = 去掉它会动多少
    infl = full - loo
    order = np.argsort(-np.abs(infl))
    print(f"  单会话影响 |Δ| 最大 {np.abs(infl).max():.4f} "
          f"(≈ {np.abs(infl).max()/se_both:.2f} × SE_两向)")
    print(f"  最影响的 5 个会话:")
    for i in order[:5]:
        sess = sessions[i]
        print(f"    {sess}  块数 {len(by_sess[sess]):>2}  "
              f"ΔF1(全) {full:+.4f} -> 去它后 {loo[i]:+.4f}  (影响 {infl[i]:+.4f})")
    n_up = int((infl > 0).sum())
    print(f"  去掉后有 {n_up}/{n_sess} 个会话让 ΔF1 **下降** "
          f"(即这些会话在**支撑**效应)")

    # ---------- ③ 逐会话的计数方向 ----------
    print(f"\n=== ③ 逐会话 own10 − mixnorm 的计数 (7 种子均值) ===")
    dtp = np.zeros(n_sess); dfp = np.zeros(n_sess)
    for i, sess in enumerate(sessions):
        a = np.mean([agg["own10"][s][sess] for s in SEEDS], axis=0)
        b_ = np.mean([agg["mixnorm"][s][sess] for s in SEEDS], axis=0)
        dtp[i] = a[0] - b_[0]; dfp[i] = a[1] - b_[1]
    print(f"  ΔTP: 均值 {dtp.mean():+.3f}, 正号会话 {int((dtp>0).sum())}/{n_sess}")
    print(f"  ΔFP: 均值 {dfp.mean():+.3f}, 正号会话 {int((dfp>0).sum())}/{n_sess}"
          f"   (正 = own10 多报假阳性)")
    tot_tp = np.mean([sum(agg['own10'][s][ss][0] for ss in sessions)
                      for s in SEEDS])
    print(f"  全池 TP(own10) = {tot_tp:.0f}, 而 ΔTP 合计 {dtp.sum():+.0f} "
          f"⇒ 效应占 TP 的 {dtp.sum()/max(tot_tp,1):.1%}")

    # ---------- ④ 计数必须逐种子列出来 ----------
    # 计数类量最危险: 单次 run 的摆幅可能盖过臂间差
    # ([[permutation-test-covers-one-variance-source]] 同一天栽过两次)。
    print(f"\n=== ④ ΔTP / ΔFP 逐种子 (计数类量: 先看它翻不翻号) ===")
    print(f"{'seed':>7}{'TP_own':>8}{'TP_mix':>8}{'ΔTP':>7}"
          f"{'FP_own':>8}{'FP_mix':>8}{'ΔFP':>7}{'ΔF1':>9}")
    rows = []
    for i, s in enumerate(SEEDS):
        a_tp = sum(agg["own10"][s][ss][0] for ss in sessions)
        a_fp = sum(agg["own10"][s][ss][1] for ss in sessions)
        b_tp = sum(agg["mixnorm"][s][ss][0] for ss in sessions)
        b_fp = sum(agg["mixnorm"][s][ss][1] for ss in sessions)
        rows.append((a_tp, b_tp, a_fp, b_fp))
        print(f"{s:>7}{a_tp:>8.0f}{b_tp:>8.0f}{a_tp-b_tp:>+7.0f}"
              f"{a_fp:>8.0f}{b_fp:>8.0f}{a_fp-b_fp:>+7.0f}"
              f"{per_seed[i]:>+9.4f}")
    arr = np.array(rows, dtype=float)
    for j, nm in enumerate(("TP_own", "TP_mix", "FP_own", "FP_mix")):
        print(f"  {nm}: 跨种子 SD = {arr[:, j].std(ddof=1):.1f} "
              f"(均值 {arr[:, j].mean():.1f})")
    dtp_s = arr[:, 0] - arr[:, 1]
    dfp_s = arr[:, 2] - arr[:, 3]
    print(f"  ΔTP 跨种子: 均值 {dtp_s.mean():+.1f}, SD {dtp_s.std(ddof=1):.1f}, "
          f"正号 {int((dtp_s>0).sum())}/7")
    print(f"  ΔFP 跨种子: 均值 {dfp_s.mean():+.1f}, SD {dfp_s.std(ddof=1):.1f}, "
          f"正号 {int((dfp_s>0).sum())}/7 (正 = own10 多报)")
    print(f"  ⇒ 臂间差 / 跨种子SD: ΔTP {abs(dtp_s.mean())/max(dtp_s.std(ddof=1),1e-9):.2f}"
          f"  ΔFP {abs(dfp_s.mean())/max(dfp_s.std(ddof=1),1e-9):.2f}")

    # ---------- ⑤ 会话是第二个地板: 加块到底还能降多少 ----------
    # §2.10 的 σ̂ 反解**看不见会话** —— 轮转抽样把会话构成钉死了。
    # 这里用**多阶段 bootstrap** (会话簇 → 会话内块 → 种子) 直接量总 SE,
    # 并在三档「每会话块数」上做, 好把**随 N 缩的那一项**与**不缩的地板**分开。
    print(f"\n=== ⑤ 总 SE 的标度 (会话簇 → 块 → 种子, B={B}) ===")
    M = {}
    for arm in ("own10", "mixnorm"):
        for s in SEEDS:
            t_ = np.zeros((n_sess, 48)); f_ = np.zeros((n_sess, 48))
            g_ = np.zeros((n_sess, 48))
            for i, sess in enumerate(sessions):
                for j, c in enumerate(by_sess[sess]):
                    tp, fp, ngt = counts[arm][s][c]
                    t_[i, j] = tp; f_[i, j] = fp; g_[i, j] = ngt
            M[(arm, s)] = (t_, f_, g_)

    NS = (12, 24, 48)
    se_by_n = {}
    for n_ps in NS:
        rng2 = np.random.default_rng(1)
        vals = np.empty(B)
        for b in range(B):
            si = rng2.choice(n_sess, n_sess, replace=True)
            ci = rng2.integers(0, 48, size=(n_sess, n_ps))
            sub = [SEEDS[i] for i in rng2.choice(len(SEEDS), len(SEEDS),
                                                 replace=True)]
            out = []
            for s in sub:
                rr = {}
                for arm in ("own10", "mixnorm"):
                    t_, f_, g_ = M[(arm, s)]
                    rr[arm] = f1_from(t_[si[:, None], ci].sum(),
                                      f_[si[:, None], ci].sum(),
                                      g_[si[:, None], ci].sum())
                out.append(rr["own10"] - rr["mixnorm"])
            vals[b] = float(np.mean(out))
        se = float(vals.std(ddof=1))
        se_by_n[n_ps] = se
        print(f"  每会话 {n_ps:>2} 块 (共 {n_ps*n_sess:>5} 块): "
              f"SE = {se:.5f}   均值 {vals.mean():+.4f}")

    # SE² = A/N + F ⇒ 对 (1/N, SE²) 拟合直线, **截距就是加块加不到的地板**。
    xs = np.array([1.0 / n for n in NS])
    ys = np.array([se_by_n[n] ** 2 for n in NS])
    A_fit, F_fit = np.polyfit(xs, ys, 1)
    F_fit = max(float(F_fit), 0.0)
    floor = float(np.sqrt(F_fit))
    print(f"\n  拟合 SE² = A/N + F:  A = {A_fit:.3e}, F = {F_fit:.3e}  "
          f"(3 点拟 2 参, 只剩 1 df —— 只能当量级用)")
    print(f"  ⇒ **加块加不到的地板 = √F = {floor:.5f}**")
    print(f"     §2.10 只算种子那一项给 0.00376 ⇒ **漏了会话, 偏小 "
          f"{0.00376/max(floor,1e-9):.2f}×**")
    print(f"  ⇒ **t 的上限 = {full/floor:.2f}**   "
          f"(§2.10 报的 6.8 是同一个漏项) ")
    print(f"  现在 (每会话 48 块): SE = {se_by_n[48]:.5f}, "
          f"t = {full/se_by_n[48]:.2f}")
    if F_fit > 0:
        n_star = A_fit / F_fit
        print(f"  块项 = 地板 ⇒ 每会话 N* = A/F = {n_star:.1f} 块 "
              f"(现在 48; 全部候选只够每会话 ~718 块)")
    # 3 点拟 2 参只剩 1 df ⇒ 必须看**换一对点**会不会翻结论
    print(f"\n  两两配对的敏感性 (别信单次拟合):")
    for i in range(len(NS)):
        for j in range(i + 1, len(NS)):
            a2, f2 = np.polyfit([xs[i], xs[j]], [ys[i], ys[j]], 1)
            f2 = max(float(f2), 0.0)
            print(f"    只用 N={NS[i]},{NS[j]}: 地板 {np.sqrt(f2):.5f}, "
                  f"t 上限 {full/max(np.sqrt(f2),1e-9):.2f}")

    out = {"full_df1": full, "n_sess": n_sess, "n_chunks": len(cids),
           "se_by_nps": {str(k): v for k, v in se_by_n.items()},
           "floor": floor, "t_ceiling": full / floor,
           "se_three_way_48": se_by_n[48],
           "se_sess_boot": se_sess, "se_seed_boot": se_seed,
           "se_both_boot": se_both, "se_analytic_no_sess": 0.00557,
           "session_increment": float(extra),
           "t_with_sess": full / se_both, "t_without_sess": full / 0.00557,
           "loo_mean": float(loo.mean()), "loo_sd": float(loo.std(ddof=1)),
           "loo_min": float(loo.min()), "loo_max": float(loo.max()),
           "n_sessions_supporting": n_up,
           "dtp_mean": float(dtp.mean()), "dtp_pos": int((dtp > 0).sum()),
           "dfp_mean": float(dfp.mean()), "dfp_pos": int((dfp > 0).sum()),
           "per_seed_dfp": [float(v) for v in dfp_s],
           "per_seed_dtp": [float(v) for v in dtp_s],
           "dfp_seed_sd": float(dfp_s.std(ddof=1)),
           "dtp_seed_sd": float(dtp_s.std(ddof=1))}
    json.dump(out, open(f"{ANNOT}/g1_e4c_session_decomp.json", "w"), indent=1)
    print(f"\n-> {ANNOT}/g1_e4c_session_decomp.json")


if __name__ == "__main__":
    main()
