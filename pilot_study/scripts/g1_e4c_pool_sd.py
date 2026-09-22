"""池间 SD: ΔF1 跨「再抽一批块」的分布 —— 纯读盘, 0 GPU。

对应预登记 docs/pilot_study/2026-09-18_pool_sd.md §1.4。判据在跑之前就写死了,
本脚本**按预登记实现**, 不挑说法。

主统计量:
    SD_batch(400) = SD({ΔF1_j}, ddof=1)        J=12, df=11
    ρ = SD_batch(400) / SE_null(400)
    主检验 (单侧, H0 = 块在 key 内可交换):  (J−1)·ρ² ~ χ²_{J−1}

零模型一律**分层**(每 key 重抽 m 块, 与一批同构), 理由见 2026-09-18_pool_sd.md §1.3:
项目的 SRS 口径把零模型高估 7–11%, 拿它当对照会系统性偏向「bootstrap 诚实」。

口径全部 import, 不重写 (memory: hardcoded-conclusions-escape-reproduction)。

用法 (srun 内, fd_analysis):
  python scripts/g1_e4c_pool_sd.py                 # 全 12 批
  python scripts/g1_e4c_pool_sd.py --n-batches 8   # 只用前 8 批 (中断后的退化方案)
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from g1_e4_analyze import paired, d_of, f1_of, SEEDS  # noqa: E402
from g1_pool_null_probe import key_of, strat_boot_se  # noqa: E402

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
NB_MAX = 12
B_BATCH = 400


def tag_of(j):
    return f"e4c{j:02d}"


def load_batches(nb):
    """逐批载入; 缺种子的批直接报出来, 不静默少算。"""
    out = []
    for j in range(nb):
        D = paired(tag_of(j))
        if D is None:
            print(f"  🔴 批 {j} 配对不足 2 个种子, 跳过")
            continue
        if len(D["seeds"]) != len(SEEDS) or D["missing"]:
            print(f"  🔴 批 {j} 只有 {len(D['seeds'])}/{len(SEEDS)} 对种子 "
                  f"(缺 {D['missing']}) —— 该批 **不完整**, 跳过")
            continue
        out.append((j, D))
    return out


def pooled_vec(batches):
    """把各批拼成一份 (cids, vec) —— 只为算零模型, 不用于算 ΔF1。"""
    cids, vec = [], None
    for _j, D in batches:
        cids.extend(D["cids"])
        if vec is None:
            vec = {k: [] for k in D["vec"]}
        for k, v in D["vec"].items():
            vec[k].extend(v)
    return cids, vec


def srs_boot_se(vec, cids, seeds, k, boot, rng_seed=0):
    """SRS 零模型 (项目历史口径), 在合并集上按 k 块重抽 —— 只作并报。"""
    n = len(cids)
    rng = np.random.default_rng(rng_seed)
    db = np.empty(boot)
    for b in range(boot):
        ii = rng.integers(0, n, k)
        db[b] = np.mean(list(d_of(vec, seeds, ii).values()))
    return float(db.std(ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-batches", type=int, default=NB_MAX)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    seeds = list(SEEDS)
    batches = load_batches(args.n_batches)
    J = len(batches)
    print(f"=== 池间 SD: 可用批 {J}/{args.n_batches} ===")
    if J < 4:
        print("  批数 < 4, SD 无意义, 退出。")
        return

    # ---------- ① 逐批 ΔF1 ----------
    print(f"\n{'批':>4}{'块数':>7}{'事件':>6}{'ΔF1':>10}"
          f"{'own10 F1':>11}{'mixnorm F1':>12}")
    djs = []
    for j, D in batches:
        dv = np.array([D["d"][s] for s in D["seeds"]])
        e0 = D["P"][("own10", D["seeds"][0])]
        ngt = sum(e0["per_chunk"][c]["n_gt"] for c in D["cids"])
        fo = f1_of(D["vec"][("own10", D["seeds"][0])], range(len(D["cids"])))
        fm = f1_of(D["vec"][("mixnorm", D["seeds"][0])], range(len(D["cids"])))
        print(f"{j:>4}{len(D['cids']):>7}{ngt:>6}{dv.mean():>+10.4f}"
              f"{fo:>11.4f}{fm:>12.4f}")
        djs.append(float(dv.mean()))
    djs = np.array(djs)

    # ---------- ② 主统计量 ----------
    SD_batch = float(djs.std(ddof=1))
    cids_p, vec_p = pooled_vec(batches)
    n_per_key = collections.Counter(key_of(c) for c in cids_p)
    K = len(n_per_key)
    print(f"\n合并池: {len(cids_p)} 块 / {K} 个 key / 每 key "
          f"{min(n_per_key.values())}–{max(n_per_key.values())} 块")

    # 每批内部的**轮**结构 —— k<400 的组必须按轮切, 不能按下标切:
    # 一轮 = 每个 key 各 1 块, 按下标切会让半批只覆盖前一半 key,
    # 于是"组间差"里混进"哪几个会话", 与分层零模型不再同构。
    #
    # ⚠️ 轮序只能取自**抽样时的 id 文件**。`paired()` 返回的 cids 是
    # `sorted(...)`(按 id 字符串排), 同一个 key 的 24 块会连续排在一起, 于是
    # "前 200 个下标"= 前几个会话, 不是一轮。2026-09-19 第一次运行就栽在这里:
    # 断言报 "批 0 的轮 0 只覆盖 100/200 个 key" —— id 文件里前 200 行确实是
    # 200 个 key(抽样脚本按轮追加), 排序把它毁掉了。
    # ⚠️ 下标一律用**合并池**的坐标系 (vec_p / cids_p), 不用各批自己的 D:
    # k>400 的组会横跨多批, 而每一批是**独立一次评估**(各自的 vec), 不存在
    # "把两批的 vec 拼起来"这回事 —— 只有 pooled_vec 是同一套数组。
    pos_p = {c: i for i, c in enumerate(cids_p)}
    rounds = []                       # [idx_list, ...] 每项是一轮 K 块 (合并池下标)
    for _j, D in batches:
        order = [l.strip() for l in
                 open(f"{ANNOT}/g1_eval4c_ids_b{_j:02d}_ids.txt") if l.strip()]
        assert len(order) == len(D["cids"]), \
            f"批 {_j}: id 文件 {len(order)} 行 vs 产物 {len(D['cids'])} 块"
        assert set(order) == set(D["cids"]), f"批 {_j}: id 文件与产物不是同一组块"
        for r in range(len(order) // K):
            chunk = order[r * K:(r + 1) * K]
            keys = {key_of(c) for c in chunk}
            assert len(keys) == K, \
                f"批 {_j} 的轮 {r} 只覆盖 {len(keys)}/{K} 个 key ⇒ id 文件没保住轮序"
            rounds.append([pos_p[c] for c in chunk])
    per_batch_rounds = len(rounds) // len(batches)
    print(f"  轮结构: {len(rounds)} 轮 × {K} 块, 每批 {per_batch_rounds} 轮 ✅")

    def df1_of(idx):
        """一组的 ΔF1 —— 在**合并池**上按 `d_of` 的定义算 (口径只有一处)。

        ⚠️ 必须对 idx 的**并集**算一次 F1, **不能**各组先算再平均:
        F1 是比值, `ΔF1(A∪B) ≠ [ΔF1(A)+ΔF1(B)]/2`。2026-09-19 的第一版就是
        按轮平均的, k=400 的自检当场报出 0.009194(vs paired 0.009308)、
        0.013725(vs 0.013502) —— 差在第四位, 肉眼极易放过。
        (同一条坑本文件 88 行的 `d_of` 文档里就写着, 我在这里又踩了一次。)
        """
        return float(np.mean(list(d_of(vec_p, seeds, idx).values())))

    results = {}
    print(f"\n=== 标度扫描 (零模型 = 分层 bootstrap, 与一组同构) ===")
    print(f"{'k':>6}{'m/key':>7}{'组数':>6}{'df':>4}{'SD_batch':>11}"
          f"{'SE_null分层':>13}{'ρ=SD/SE':>10}{'SE_null_SRS':>13}{'ρ_SRS':>9}")
    for k, m in ((200, 1), (400, 2), (800, 4), (1600, 8)):
        gsize = k // K                     # 每组几轮
        if gsize < 1 or len(rounds) % gsize:
            continue
        groups = [rounds[i:i + gsize] for i in range(0, len(rounds), gsize)]
        if len(groups) < 3:
            continue
        vals = [df1_of([i for r in g for i in r]) for g in groups]
        sd_k = float(np.std(vals, ddof=1))
        # 自检: k=400 的"分组算出来的 ΔF1" 必须与 ① 里 `paired(tag)` 直接给的**逐位一致**。
        # 对不上说明"轮切/聚合"这套推广口径与 `paired` 的口径不是同一个 —— 那就停,
        # 别拿推广出来的 ρ 去判读 (memory: 口径只有一处 / 自检必须能失败)。
        # ⚠️ 这条自检**管不住轮序**: k=400 时 gsize=2, 一组正好是一整批, 于是轮序
        # 怎么错它都通过。轮序由上面那条 `len(keys)==K` 断言守 —— 那才是会失败的。
        # 它管的是**聚合口径**(并集 vs 逐轮平均), 2026-09-19 正是它逮到的。
        if k == B_BATCH:
            bad = [(a, b) for a, b in zip(vals, djs) if abs(a - b) > 1e-12]
            assert not bad, (f"k=400 的分组值与 paired() 不一致 {bad[:2]} "
                             f"⇒ 轮切口径有 bug, 停止判读")
        df = len(groups) - 1
        se_str = strat_boot_se(vec_p, cids_p, seeds, m, args.boot)
        se_srs = srs_boot_se(vec_p, cids_p, seeds, k, args.boot)
        star = "  ← 主" if k == B_BATCH else ""
        print(f"{k:>6}{m:>7}{len(groups):>6}{df:>4}{sd_k:>11.5f}{se_str:>13.5f}"
              f"{sd_k/se_str:>10.3f}{se_srs:>13.5f}{sd_k/se_srs:>9.3f}{star}")
        results[k] = {"sd": sd_k, "se_strat": se_str, "se_srs": se_srs,
                      "rho": sd_k / se_str, "rho_srs": sd_k / se_srs,
                      "df": df, "n_groups": len(groups)}

    # ---------- ③ 预登记判读 (主点 k=400) ----------
    R = results.get(B_BATCH)
    print(f"\n{'='*74}\n=== 预登记判读 (2026-09-18_pool_sd.md §1.4) ===")
    if R is None:
        print("  k=400 那一行没算出来, 无法判读。")
        return
    rho, df = R["rho"], R["df"]
    chi2_stat = df * rho ** 2
    crit = float(chi2.ppf(0.95, df))
    print(f"  ΔF1 跨批: 均值 {djs.mean():+.4f}, SD_batch(400) = {SD_batch:.5f}, "
          f"df={df}, J={J}")
    print(f"  零模型 (分层, m=2/key): SE_null(400) = {R['se_strat']:.5f}")
    print(f"  **ρ = {rho:.3f}**   主检验 (J−1)ρ² = {chi2_stat:.2f} vs 临界 {crit:.2f} "
          f"⇒ {'🔴 拒绝 H0 (bootstrap 低估)' if chi2_stat > crit else '✅ 不拒绝 H0'}")
    print(f"  95% CI (χ² 反解): ρ ∈ "
          f"[{rho*np.sqrt(df/float(chi2.ppf(0.975, df))):.3f}, "
          f"{rho*np.sqrt(df/float(chi2.ppf(0.025, df))):.3f}]")
    lo, hi = 1.3, 2.0
    if rho <= lo:
        verdict = ("**≤1.3: bootstrap 诚实** ⇒ e4x 的线下与天花板 t=1.667 站得住; "
                   "下一步是「还要多少块」的算术题")
    elif rho >= hi:
        verdict = ("**≥2.0: bootstrap 低估 ≥2×** ⇒ 真实池间 SD 与效应同量级, "
                   "判据线被系统性低估; 点估计自身漂移 ±22% ⇒ 加块无意义")
    else:
        verdict = ("**1.3–2.0: 灰区** ⇒ 如实报 ρ 与 CI, 不下二值判决, "
                   "并报两种 ρ 下各自反解出的 k_min")
    print(f"  预登记判读: {verdict}")

    # ---------- ④ 「还要多少块」反解 (只作诊断, 不算判据) ----------
    eff = float(djs.mean())
    print(f"\n=== 诊断: 若按 SD_strat(k) = SD_batch(400)·√(400/k) 外推 ===")
    if abs(eff) > 1e-9:
        k_min = B_BATCH * (2 * SD_batch / abs(eff)) ** 2
        print(f"  以 ΔF1={eff:+.4f} 让「均值 > 2×SD_strat(k)」成立, 需 k ≈ "
              f"{k_min:,.0f} 块 ({k_min/B_BATCH:.1f}× 本实验规模)")
        # k_min 对 ΔF1 是平方反比 ⇒ 把三个口径的点估计都代进去, 看它有多脆
        print(f"  ⚠️ 这个反解**继承 ΔF1 点估计的全部噪声** —— 同一构造的三次"
              f"独立抽取给出:")
        for nm, v in (("e4c(本次)", eff), ("e4(1200块)", 0.0210),
                      ("e4x(1600块)", 0.0163)):
            if abs(v) > 1e-9:
                print(f"       ΔF1={v:+.4f} ({nm}) ⇒ k_min ≈ "
                      f"{B_BATCH * (2 * SD_batch / abs(v)) ** 2:,.0f} 块")
        print(f"     ⇒ k_min 在 ~650–1600 之间漂, **换一个点估计就换一个答案**;"
              f" 这不是判据, 只说明「加块」这条路的天花板比想象的软。")
    else:
        print("  批均值 ≈ 0, 无反解可言。")

    # ---------- ⑤ 批序漂移检查 (时间混杂的代理) ----------
    js = np.array([j for j, _ in batches], dtype=float)
    r = float(np.corrcoef(js, djs)[0, 1]) if J > 2 and djs.std() > 0 else float("nan")
    print(f"\n=== 批序 vs ΔF1 相关 (评估时刻混杂的代理; 外层是 (arm,seed) 内层 batch, "
          f"本应无趋势) ===\n  r = {r:+.3f}  (n={J})")
    # 置换零模型: 同一批 ΔF1 值随机配批序, |r| 能有多大 (n=12 时 ±0.5 都可能出现)
    rng_p = np.random.default_rng(0)
    null_r = np.array([abs(np.corrcoef(js, rng_p.permutation(djs))[0, 1])
                       for _ in range(20000)])
    p_r = float((null_r >= abs(r)).mean())
    print(f"  置换零模型 (20000 次, 打乱 ΔF1 与批序的对应): "
          f"|r| ≥ {abs(r):.3f} 的概率 = {p_r:.3f} ⇒ "
          f"{'⚠️ 有趋势嫌疑' if p_r < 0.05 else '✅ 与无趋势相容'}")
    print(f"  (n=12 时 |r| 的 95% 分位约 {np.quantile(null_r, 0.95):.3f} —— "
          f"不查零分布就把 {abs(r):.2f} 读成'有趋势'会误报)")

    # ---------- ⑥ 事后诊断 (**不在预登记里**, 只作解释用) ----------
    ngts = np.array([sum(D["P"][("own10", D["seeds"][0])]["per_chunk"][c]["n_gt"]
                         for c in D["cids"]) for _j, D in batches], dtype=float)
    r_ev = float(np.corrcoef(ngts, djs)[0, 1])
    print(f"\n=== 事后诊断 (不在预登记内): 批的ΔF1 与其**事件数**的相关 ===\n"
          f"  每批事件数 {ngts.min():.0f}–{ngts.max():.0f}, r(事件数, ΔF1) = "
          f"{r_ev:+.3f}  (n={J})\n"
          f"  ⇒ {'事件多的批 ΔF1 也大, 批间差可能只是「这批有几个事件」' if abs(r_ev) > 0.5 else '无强关系'}")

    json.dump({"n_batches": J, "delta_f1": djs.tolist(),
               "mean": float(djs.mean()), "sd_batch_400": SD_batch,
               "rho": rho, "chi2_stat": chi2_stat, "crit": crit, "df": df,
               "rho_ci": [float(rho * np.sqrt(df / float(chi2.ppf(0.975, df)))),
                          float(rho * np.sqrt(df / float(chi2.ppf(0.025, df))))],
               "batch_order_r": r, "n_gt": [int(sum(
                   D["P"][("own10", D["seeds"][0])]["per_chunk"][c]["n_gt"]
                   for c in D["cids"])) for _j, D in batches],
               "scan": {str(k): v for k, v in results.items()}},
              open(f"{ANNOT}/g1_e4c_pool_sd.json", "w"), indent=1)
    print(f"\n-> {ANNOT}/g1_e4c_pool_sd.json")


if __name__ == "__main__":
    main()
