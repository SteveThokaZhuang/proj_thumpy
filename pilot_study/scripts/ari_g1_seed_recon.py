"""Q1 溯源: mixnorm 的各种子为什么把阈值画在不同地方?

§5 只测出现象 —— 各种子**平均 margin** 的 SD: own10 0.183 vs mixnorm 0.525。
三个候选来源:
    (a) 数据顺序 / shuffle    —— 若 data_seed 随 seed 变, 每个 run 看到的样本顺序不同
    (b) LoRA 初值             —— 若初始化随 seed 变, 每个 run 起点不同
    (c) 两声道相加的增益不确定性 —— mixnorm 独有的数据侧性质

关键判别: **同 seed 跨臂**(own10_s42 vs mixnorm_s42) 这一对。
它俩若共享同一套 seed, 那么 (a) 和 (b) 都被扣掉 —— 同样的初始化、同样的
样本顺序、同样的超参, **唯一差异是音频内容**。若这一对的阈值仍然不同,
那来源只能是 (c) 或"同一初值在两种数据上的优化路径发散"。

反过来, 若同 seed 跨臂的 lora_A 就不同, 那说明连初始化都没对齐。

零成本: 纯读盘 (training_args.bin + adapter_model.safetensors), 不需要 GPU。

用法: srun --jobid=<JOBID> --overlap --ntasks=1 \
        /share/home/zhuangruicen/miniconda3/envs/funaudiochat/bin/python \
        scripts/ari_g1_seed_recon.py
"""
import glob
import os
import re

import torch
from safetensors.torch import load_file

ROOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
        "real_data/results/annotator")
SEEDS = [42, 1234, 7, 2024, 3407, 31337, 55555]
# c1024/c2048 是 micro-batch 排查用的旁支, 不属于 7 种子主实验
SKIP = ("c1024", "c2048", "stale")


def parse_run(d):
    """'g1_own10_s1234_sft' -> ('own10', 1234)"""
    m = re.match(r"g1_(own10|mixnorm)(?:_s(\d+))?_sft$", os.path.basename(d))
    if not m:
        return None
    arm, s = m.group(1), m.group(2)
    return arm, (int(s) if s else 42)


def find_runs():
    out = {}
    for cfg in sorted(glob.glob(f"{ROOT}/g1_*_sft/g1_config.yaml")):
        d = os.path.dirname(cfg)
        if any(k in d for k in SKIP):
            continue
        p = parse_run(d)
        if p:
            out[p] = d
    return out


def main():
    runs = find_runs()
    print(f"找到 {len(runs)} 个主实验 run\n")

    # ---------- 1. training_args.bin ----------
    print("=" * 92)
    print("  1. training_args.bin —— 实际跑的参数")
    print("=" * 92)
    KEYS = ["seed", "data_seed", "per_device_train_batch_size",
            "gradient_accumulation_steps", "learning_rate", "num_train_epochs",
            "lr_scheduler_type", "warmup_ratio", "cutoff_len", "lora_rank",
            "lora_alpha", "max_steps"]
    print(f"  {'arm':<9}{'seed':>7}" + "".join(f"{k:>14}" for k in KEYS[1:7]))
    args_by_run = {}
    for (arm, seed) in sorted(runs, key=lambda x: (x[0], x[1])):
        p = f"{runs[(arm, seed)]}/saves/training_args.bin"
        if not os.path.exists(p):
            print(f"  {arm:<9}{seed:>7}   ⚠️ 缺 training_args.bin")
            continue
        a = torch.load(p, map_location="cpu", weights_only=False)
        args_by_run[(arm, seed)] = a
        vals = [str(getattr(a, k, "N/A"))[:13] for k in KEYS[1:7]]
        print(f"  {arm:<9}{seed:>7}" + "".join(f"{v:>14}" for v in vals))

    print("\n  ── 关键字段 ──")
    for (arm, seed), a in sorted(args_by_run.items(), key=lambda x: (x[0], x[1])):
        print(f"    {arm:<9}{seed:>7}   seed={a.seed}  "
              f"data_seed={getattr(a, 'data_seed', 'N/A')}  "
              f"full_determinism={getattr(a, 'full_determinism', 'N/A')}")

    # ---------- 2. 参数是否真的只有 seed 不同 ----------
    print("\n" + "=" * 92)
    print("  2. 同 seed 跨臂的配置差异 (排除 seed 本身)")
    print("=" * 92)
    for seed in SEEDS:
        o, n = args_by_run.get(("own10", seed)), args_by_run.get(("mixnorm", seed))
        if o is None or n is None:
            continue
        diff = []
        for k, v in vars(o).items():
            if k.startswith("_") or k in ("output_dir", "seed"):
                continue
            w = getattr(n, k, "<missing>")
            if repr(v) != repr(w):
                diff.append(f"{k}: {v!r} vs {w!r}")
        print(f"  seed {seed:<6} " + ("（无差异）" if not diff else ""))
        for d in diff:
            print(f"           {d}")

    # ---------- 3. lora_A 比对 ----------
    print("\n" + "=" * 92)
    print("  3. adapter 权重 —— lora_A 是否共享同一初值")
    print("=" * 92)
    # P0 测得 lora_A 近似冻结 (‖A‖≈59.3 基本不动), 所以 lora_A ≈ 初值。
    # 若同 seed 跨臂的 lora_A 逐位相同 -> 两臂共享初始化, 初始化不是差异来源。
    As, Bs = {}, {}
    for (arm, seed), d in sorted(runs.items()):
        p = f"{d}/saves/adapter_model.safetensors"
        if not os.path.exists(p):
            continue
        sd = load_file(p)
        As[(arm, seed)] = {k: v for k, v in sd.items()
                           if "lora_A" in k and v.dim() == 2}
        Bs[(arm, seed)] = {k: v for k, v in sd.items()
                           if "lora_B" in k and v.dim() == 2}
    if not As:
        print("  ⚠️ 没读到 adapter 权重")
        return

    # lora_A 与 lora_B 的 key 集合并不相同(有些层只挂了一个), 取交集
    allruns = sorted(set(As) & set(Bs))
    ksA = sorted(set.intersection(*(set(As[r]) for r in allruns)))
    ksB = sorted(set.intersection(*(set(Bs[r]) for r in allruns)))
    print(f"  共同 run 数: {len(allruns)}   lora_A 公共 key: {len(ksA)}   "
          f"lora_B 公共 key: {len(ksB)}")

    def rel_diff(x, y):
        return (x - y).norm().item() / max(y.norm().item(), 1e-12)

    def pairdiff(store, ks, ra, rb):
        return sum(rel_diff(store[ra][k], store[rb][k]) for k in ks) / len(ks)

    print("\n  ── 同 seed 跨臂: own10_s<seed> vs mixnorm_s<seed> ──")
    print(f"  {'seed':>7}{'lora_A 相对差':>18}{'lora_B 相对差':>18}   判定")
    same_init = {}
    for seed in SEEDS:
        ka, kb = ("own10", seed), ("mixnorm", seed)
        if ka not in As or kb not in As:
            continue
        da = pairdiff(As, ksA, ka, kb)
        db = pairdiff(Bs, ksB, ka, kb)
        verdict = "共享初值" if da < 1e-6 else "初值不同"
        same_init[seed] = da < 1e-6
        print(f"  {seed:>7}{da:>18.3e}{db:>18.3e}   {verdict}")

    print("\n  ── 对照: 同臂跨 seed (初值本来就该不同) ──")
    print(f"  {'arm':<9}{'seed 对':>18}{'lora_A 相对差':>18}")
    for arm in ("own10", "mixnorm"):
        base = (arm, 42)
        if base not in As:
            continue
        for seed in SEEDS[1:]:
            if (arm, seed) not in As:
                continue
            d = pairdiff(As, ksA, base, (arm, seed))
            print(f"  {arm:<9}{f'42 vs {seed}':>18}{d:>18.3e}")

    # ---------- 4. 结论 ----------
    print("\n" + "=" * 92)
    print("  4. 结论")
    print("=" * 92)
    if same_init and all(same_init.values()):
        print("  ✅ 全部同 seed 跨臂都共享同一份 lora_A 初值。")
        print("     -> 对每个 seed, own10 与 mixnorm 是**同初值、同样本顺序、同超参**")
        print("        的一对, 唯一差异是音频内容。")
        print("     -> (a) 数据顺序 与 (b) LoRA 初值 **都不能解释两臂的阈值差异**;")
        print("        差异必然来自 (c) 数据侧, 或同初值在两种数据上的优化路径发散。")
        print("     -> 但注意: 这**不能**解释**同臂跨 seed** 的阈值离散 —— 那部分")
        print("        仍然可以来自 (a)/(b)。要分开需要 data_seed 实验 (Q3)。")
    elif same_init:
        print(f"  ⚠️ 部分 seed 共享初值: {same_init}")
    else:
        print("  ❌ 同 seed 跨臂的 lora_A 并不相同 —— 初始化没有对齐,")
        print("     那么 (b) 初值 仍是一个活着的候选解释。")


if __name__ == "__main__":
    main()
