"""#55 P0: mixnorm 不稳的**优化侧**排查 (免费, 只读已有产物).

动机: E2 上 mixnorm 的 F1 跨种子 SD 0.102, 是 own10 (0.024) 的 4.3 倍。在花机时做
决策边界探针 (P1) 之前, 先用已有产物排掉"某些种子没训好"这个最平凡的解释。

本脚本查两件事:
  A. grad_norm 轨迹 (trainer_state.json) —— 塌掉的种子是否收敛更差/更抖?
  B. LoRA adapter 的权重范数 —— 塌掉的种子是否学到的更新更小 (欠拟合)?

注意: **训练 loss 在本项目里全是 0.0**, trainer_state.json / trainer_log.jsonl /
train_results.json 三个 sink 都是 0.0 (日志里有 `loss_type=None ... unrecognised`)。
所以 loss 轨迹不可用, grad_norm 是唯一的优化侧观测量 —— 这也是任务 #57 (P4) 的由来。

用法: python scripts/ari_g1_instability_p0.py
"""
import json
import os

import numpy as np

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
SEEDS = [42, 1234, 7, 2024, 3407, 31337, 55555]
ARMS = ["own10", "mixnorm"]


def run_dir(arm, seed):
    return f"{ANNOT}/g1_{arm}_sft" if seed == 42 else f"{ANNOT}/g1_{arm}_s{seed}_sft"


def suffix(seed):
    return "" if seed == 42 else f"_s{seed}"


def e2_scores(arm, seed):
    p = f"{ANNOT}/g1_eval_{arm}{suffix(seed)}_e2.json"
    r = next(iter(json.load(open(p)).values()))
    return r["f1"], r["recall"], r["precision"], r["n_pred"]


def grad_stats(arm, seed):
    p = f"{run_dir(arm, seed)}/saves/trainer_state.json"
    h = json.load(open(p))["log_history"]
    g = np.array([e["grad_norm"] for e in h if "grad_norm" in e], float)
    return {"first": g[0], "last": g[-1], "mean": g.mean(),
            "max": g.max(), "sd": g.std(ddof=1), "n": len(g)}


def adapter_norm(arm, seed):
    """LoRA adapter 的全参数 L2 范数 (流式累加, 不把 266MB 全读进内存)。"""
    from safetensors import safe_open
    p = f"{run_dir(arm, seed)}/saves/adapter_model.safetensors"
    if not os.path.exists(p):
        return None
    ss, n, per = 0.0, 0, {}
    with safe_open(p, framework="np") as f:
        for k in f.keys():
            t = f.get_tensor(k).astype(np.float64)
            s = float((t ** 2).sum())
            ss += s
            n += t.size
            # 按模块类型聚合 (如 q_proj / v_proj / ...)
            mod = k.split(".")[-2] if k.count(".") >= 2 else k
            per[mod] = per.get(mod, 0.0) + s
    return {"l2": float(np.sqrt(ss)), "n_params": n,
            "per_module": {k: float(np.sqrt(v)) for k, v in sorted(per.items())}}


def main():
    out = {}
    print("=" * 92)
    print("  A. grad_norm 轨迹 vs E2 表现")
    print("=" * 92)
    print(f"  {'arm':<8}{'seed':>7}{'F1':>9}{'recall':>9}{'n_pred':>8} | "
          f"{'gn首':>8}{'gn末':>8}{'gn均值':>8}{'gnSD':>8}")
    for arm in ARMS:
        for s in SEEDS:
            f1, rc, pr, npd = e2_scores(arm, s)
            g = grad_stats(arm, s)
            out[f"{arm}_{s}"] = {"f1": f1, "recall": rc, "precision": pr,
                                 "n_pred": npd, "grad": g}
            print(f"  {arm:<8}{s:>7}{f1:>9.4f}{rc:>9.4f}{npd:>8} | "
                  f"{g['first']:>8.3f}{g['last']:>8.3f}{g['mean']:>8.3f}{g['sd']:>8.3f}")

    print("\n  ── 组内相关 (F1 与 grad_norm 统计量的相关) ──")
    for arm in ARMS:
        f1 = np.array([out[f"{arm}_{s}"]["f1"] for s in SEEDS])
        for key in ("first", "last", "mean", "sd"):
            v = np.array([out[f"{arm}_{s}"]["grad"][key] for s in SEEDS])
            r = np.corrcoef(f1, v)[0, 1] if v.std() > 1e-12 else float("nan")
            print(f"    {arm:<8} corr(F1, gn_{key:<5}) = {r:+.3f}   "
                  f"(gn SD = {v.std(ddof=1):.4f})")

    print("\n" + "=" * 92)
    print("  B. LoRA adapter 权重范数 (欠拟合检验)")
    print("=" * 92)
    print(f"  {'arm':<8}{'seed':>7}{'F1':>9} | {'‖ΔW‖':>10}{'参数量':>12}")
    for arm in ARMS:
        for s in SEEDS:
            a = adapter_norm(arm, s)
            if a is None:
                print(f"  {arm:<8}{s:>7}  (缺 adapter_model.safetensors)")
                continue
            out[f"{arm}_{s}"]["adapter"] = a
            print(f"  {arm:<8}{s:>7}{out[f'{arm}_{s}']['f1']:>9.4f} | "
                  f"{a['l2']:>10.4f}{a['n_params']:>12,}")

    print("\n  ── ‖ΔW‖ 与 F1 的关系 ──")
    for arm in ARMS:
        f1 = np.array([out[f"{arm}_{s}"]["f1"] for s in SEEDS])
        l2 = np.array([out[f"{arm}_{s}"]["adapter"]["l2"] for s in SEEDS])
        r = np.corrcoef(f1, l2)[0, 1] if l2.std() > 1e-12 else float("nan")
        print(f"    {arm:<8} ‖ΔW‖ 范围 {l2.min():.3f}-{l2.max():.3f} "
              f"(SD {l2.std(ddof=1):.4f})   corr(F1, ‖ΔW‖) = {r:+.3f}")

    json.dump(out, open(f"{ANNOT}/g1_instability_p0.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"\n-> {ANNOT}/g1_instability_p0.json")


if __name__ == "__main__":
    main()
