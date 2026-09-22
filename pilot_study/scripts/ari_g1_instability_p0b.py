"""#55 P0-B (修正版): LoRA adapter 范数 —— 用 ‖B‖ 而不是 ‖(A,B)‖.

⚠️ **自我更正**: 本文件的前一版把 LoRA 的"权重范数"算成了全部参数的 L2
(sqrt(‖A‖² + ‖B‖²)), 得到 14 个 run 全是 59.91–60.06 (SD 0.03)。那个数**没有意义**:
  - lora_A 是**随机初始化**且几乎不动 (跨 run 变化 0.007%), 其范数由随机矩阵范数
    集中效应决定, 跟训练好坏无关;
  - ΔW = B@A 才是真正学到的量, 而 **B 从 0 初始化**, 所以学习全在 B 上。
把两者平方和开根, 结果是"初始化主导", 于是"范数几乎不变"是必然的, 不是发现。

正确做法: 分开报 ‖A‖ / ‖B‖, 并算每个模块的 ‖B@A‖ (= 实际的权重增量)。

用法: python scripts/ari_g1_instability_p0b.py
"""
import json

import numpy as np
from safetensors import safe_open

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
SEEDS = [42, 1234, 7, 2024, 3407, 31337, 55555]
ARMS = ["own10", "mixnorm"]


def run_dir(arm, seed):
    return f"{ANNOT}/g1_{arm}_sft" if seed == 42 else f"{ANNOT}/g1_{arm}_s{seed}_sft"


def suffix(seed):
    return "" if seed == 42 else f"_s{seed}"


def f1_of(arm, seed):
    return next(iter(json.load(open(
        f"{ANNOT}/g1_eval_{arm}{suffix(seed)}_e2.json")).values()))["f1"]


def lora_stats(arm, seed):
    """分开累计 ‖A‖ / ‖B‖, 并把 A/B 按模块配对算 ‖B@A‖。"""
    p = f"{run_dir(arm, seed)}/saves/adapter_model.safetensors"
    pairs = {}          # 模块前缀 -> [A, B]
    with safe_open(p, framework="np") as f:
        for k in f.keys():
            if ".lora_A." in k:
                pairs.setdefault(k.replace(".lora_A.", ".LORA."), [None, None])[0] = \
                    f.get_tensor(k).astype(np.float64)
            elif ".lora_B." in k:
                pairs.setdefault(k.replace(".lora_B.", ".LORA."), [None, None])[1] = \
                    f.get_tensor(k).astype(np.float64)

    sa = sb = sd = 0.0
    n_a = n_b = n_paired = n_zeroB = 0
    for pre, (A, B) in sorted(pairs.items()):
        if A is not None:
            sa += float((A ** 2).sum()); n_a += 1
        if B is not None:
            sb += float((B ** 2).sum()); n_b += 1
            if np.count_nonzero(B) == 0:
                n_zeroB += 1
        if A is not None and B is not None:
            sd += float(((B @ A) ** 2).sum()); n_paired += 1
    return {"normA": float(np.sqrt(sa)), "normB": float(np.sqrt(sb)),
            "normBA": float(np.sqrt(sd)), "n_modules": n_paired,
            "n_zeroB": n_zeroB}


def main():
    print("=" * 84)
    print("  P0-B (修正): LoRA 范数 —— ‖A‖ 是初始化, ‖B‖ / ‖B@A‖ 才是学到的")
    print("=" * 84)
    print(f"  {'arm':<8}{'seed':>7}{'F1':>9} | {'‖A‖':>9}{'‖B‖':>9}{'‖B@A‖':>10}"
          f"{'零B模块':>9}{'模块数':>8}")
    rows = {}
    for arm in ARMS:
        for s in SEEDS:
            d = lora_stats(arm, s)
            d["f1"] = f1_of(arm, s)
            rows[f"{arm}_{s}"] = d
            print(f"  {arm:<8}{s:>7}{d['f1']:>9.4f} | {d['normA']:>9.4f}"
                  f"{d['normB']:>9.4f}{d['normBA']:>10.4f}"
                  f"{d['n_zeroB']:>9}{d['n_modules']:>8}")

    print("\n  ── 与 F1 的关系 ──")
    for arm in ARMS:
        f1 = np.array([rows[f"{arm}_{s}"]["f1"] for s in SEEDS])
        print(f"    {arm}:")
        for key in ("normA", "normB", "normBA"):
            v = np.array([rows[f"{arm}_{s}"][key] for s in SEEDS])
            sd = v.std(ddof=1)
            r = np.corrcoef(f1, v)[0, 1] if sd > 1e-9 else float("nan")
            print(f"      {key:<7} 范围 {v.min():8.4f}-{v.max():8.4f} "
                  f"(SD {sd:.4f}, 相对 %.3f%%)   corr(F1, ·) = {r:+.3f}"
                  % (sd / abs(v.mean()) * 100))

    json.dump(rows, open(f"{ANNOT}/g1_instability_p0b.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"\n-> {ANNOT}/g1_instability_p0b.json")


if __name__ == "__main__":
    main()
