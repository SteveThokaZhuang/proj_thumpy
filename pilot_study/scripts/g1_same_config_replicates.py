"""同配置重复 vs 跨种子差异 —— 量出「运行噪声」到底占多少。纯读盘, 0 GPU。

## 🔴 本脚本第一版把两组东西**混为一谈**, 结论会反向, 记在这里

第一版把下面三个 run 都当成「同配置重跑」:

  | 标签        | 产物 tag          | 走的代码路径       | data_seed |
  |-------------|-------------------|--------------------|-----------|
  | A 原始      | `<arm>_filt`      | `train_plain`      | None      |
  | Q4 重跑     | `<arm>_q4_filt`   | **`train_textonly`** | None    |
  | Q3 的 B 组  | `<arm>_ds42_filt` | `train_plain`      | **42**    |

于是算出「同配置极差 = 0.0267, 是跨种子 SD 的 86%」, 判读写成
「种子间差异可能主要是运行噪声, Q3 拆解框架要重新审视」——
**那个结论是错的**, 而且恰好把整篇的主旨往反方向带。

**错在哪**: Q4 走的是 `train_textonly`, 带 `DisableSpeechCallback`。
§1.3 早就写明, 该回调**少跑一次 `audio_invert_tower` 前向**,
因而**改变了 dropout 消耗的 RNG 流** —— 它不是同配置重跑,
而是**被扰动过 RNG 序列的另一次抽样**。把它当重复, 就把
「RNG 流扰动的敏感度」错记成了「运行噪声」。

## 正确的分组 (现在这么算)

  **① 纯重复** —— 同代码同配置, 只有 `data_seed` 字段不同
     (A: None vs B: 42), 而 `SeedableRandomSampler.__init__` 里
     `initial_seed = data_seed if data_seed is not None else torch.random.initial_seed()`,
     `set_seed(42)` 之后那个值就是 42 ⇒ **同一份有效配置** (§2.3 逐环验证)。
     本脚本会**读盘核对** `training_args.bin`, 不靠推断。

  **② RNG 流被扰动** —— Q4。它量的是另一个量:
     **结果对 RNG 消耗顺序有多敏感**, 与运行噪声是两回事。

两者混在一起 = 把「混沌敏感度」记成「复现性差」。

## 为什么这直接关乎主旨

Q3 的全部立论是「把 Var(seed) 拆成 Var(init) + Var(shuffle)」。
如果同配置重跑的离散与跨种子 SD 同量级, 那拆出来的两块就是**噪声的分解**,
没有意义。所以**必须**先量出纯重复的噪声量级, 拆解才可解读。

用法:
  python scripts/g1_same_config_replicates.py
"""
import json
import os

import numpy as np

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
ROOT = ANNOT
ARMS = ["own10", "mixnorm"]
SD_ACROSS_SEEDS = 0.0309   # §7.2: 跨 7 个**不同**种子的 SD

# ① 纯重复: 同代码 (train_plain) 同配置, 只差 data_seed 字段
PURE = [("A 原始 (ds=None)", "{}_filt", "g1_{}_sft"),
        ("B 组 ds42       ", "{}_ds42_filt", "g1_{}_ds42_sft")]
# ② RNG 流被扰动 (train_textonly, 回调改变 dropout 的 RNG 消耗)
PERTURBED = [("Q4 重跑 (RNG流扰动)", "{}_q4_filt", "g1_{}_q4_sft")]

CHECK_KEYS = ["seed", "data_seed", "per_device_train_batch_size",
              "gradient_accumulation_steps", "learning_rate",
              "num_train_epochs", "save_steps", "full_determinism"]


def load(tag):
    p = f"{ANNOT}/g1_eval_{tag}.json"
    if not os.path.exists(p):
        return None
    return next(iter(json.load(open(p)).values()))


def train_args(adir):
    """读**落盘**的 training_args.bin —— 别信 config 文件 (memory: config-differs)。"""
    p = os.path.join(ROOT, adir, "saves", "training_args.bin")
    if not os.path.exists(p):
        return None
    try:
        import torch
        return torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        return None


def collect(runs):
    out = []
    for label, tmpl, adir_tmpl in runs:
        vals = {a: load(tmpl.format(a)) for a in ARMS}
        if any(v is None for v in vals.values()):
            print(f"  {label}: ⚠️ 缺产物 "
                  f"{[a for a in ARMS if vals[a] is None]}")
            continue
        out.append((label, vals))
    return out


def main():
    print("=" * 74)
    print("  纯重复 vs RNG流扰动 —— 运行噪声到底占多少?")
    print("=" * 74)

    # ---------- 先核对配置 (读落盘产物, 不靠推断) ----------
    print("\n  ── 配置核对 (读 training_args.bin) ──")
    base = None
    for _, _, adir_tmpl in PURE + PERTURBED:
        a = train_args(adir_tmpl.format(ARMS[0]))
        if a is None:
            print(f"    {adir_tmpl}: ⚠️ 读不到 training_args.bin")
            continue
        kv = {k: getattr(a, k, "<无>") for k in CHECK_KEYS}
        print(f"    {adir_tmpl.format(ARMS[0]):<26} " +
              "  ".join(f"{k}={kv[k]}" for k in
                        ["seed", "data_seed", "full_determinism"]))
        if base is None:
            base = kv
    if base is not None:
        print(f"\n    (完整比对见下方「纯重复」段的字段差异)")

    # ---------- ① 纯重复 ----------
    print(f"\n  ── ① 纯重复: 同代码同配置 (只差 data_seed 字段) ──")
    pure = collect(PURE)
    if len(pure) >= 2:
        # 字段级核对
        a0 = train_args(PURE[0][2].format(ARMS[0]))
        a1 = train_args(PURE[1][2].format(ARMS[0]))
        if a0 is not None and a1 is not None:
            d = [(k, getattr(a0, k, None), getattr(a1, k, None))
                 for k in CHECK_KEYS
                 if getattr(a0, k, None) != getattr(a1, k, None)]
            if not d:
                print(f"    ✅ 两者逐字段相同")
            else:
                print(f"    字段差异: " +
                      ", ".join(f"{k}: {x}->{y}" for k, x, y in d))
                print(f"    (data_seed None↔42 是**等价**的, 见模块 docstring;")
                print(f"     其它字段若不同则本比较不成立!)")
        for label, v in pure:
            print(f"    {label}  own10={v['own10']['f1']:.4f}  "
                  f"mixnorm={v['mixnorm']['f1']:.4f}  "
                  f"ΔF1={v['own10']['f1']-v['mixnorm']['f1']:+.4f}")
        dp = np.array([v["own10"]["f1"] - v["mixnorm"]["f1"] for _, v in pure])
        rng_pure = float(dp.max() - dp.min())
        print(f"\n    纯重复的 ΔF1 之差 = **{rng_pure:.4f}**  (n={len(dp)})")
        print(f"    对照跨种子 SD = {SD_ACROSS_SEEDS:.4f}"
              f"   ⇒  比值 **{rng_pure/SD_ACROSS_SEEDS:.2f}**")
    else:
        rng_pure = None
        print("    ⚠️ 纯重复的产物不足两份 —— 等 B 组评估跑完再执行本段。")

    # ---------- ② RNG 流扰动 ----------
    print(f"\n  ── ② RNG 流被扰动 (Q4, 走 train_textonly) ──")
    pert = collect(PERTURBED)
    if pert and pure:
        for label, v in pert:
            d_pert = v["own10"]["f1"] - v["mixnorm"]["f1"]
            print(f"    {label}  ΔF1={d_pert:+.4f}")
            for plabel, pv in pure:
                dp0 = pv["own10"]["f1"] - pv["mixnorm"]["f1"]
                print(f"      与「{plabel.strip()}」差 = {d_pert-dp0:+.4f}")
        print(f"\n    这个量与运行噪声**不是一回事**: 它量的是"
              f"「换掉 RNG 消耗顺序, 结果动多少」。")
    else:
        print("    ⚠️ 产物不足, 跳过。")

    # ---------- 判读 ----------
    if rng_pure is not None:
        print(f"\n  ── 判读 (只针对 ① 纯重复) ──")
        if rng_pure > SD_ACROSS_SEEDS:
            print(f"    ⚠️ 纯重复的差 > 跨种子 SD ⇒ 种子间差异可能主要是噪声。")
            print(f"       Q3 的方差拆解要按这个前提读。")
        elif rng_pure > 0.5 * SD_ACROSS_SEEDS:
            print(f"    🟡 纯重复的差与跨种子 SD 同量级 ⇒ 拆解仍可做,")
            print(f"       但**不能**把 Var(seed) 全部归因给「运气」。")
        else:
            print(f"    ✅ **纯重复的差远小于跨种子 SD** (比值 "
                  f"{rng_pure/SD_ACROSS_SEEDS:.2f})")
            print(f"       ⇒ `seed=42` 把运行**钉得相当死**: 同配置重跑几乎重现。")
            print(f"       ⇒ 跨种子的 0.0309 是**真实的种子间差异**, 不是运行噪声。")
            print(f"       ⇒ **Q3 的方差拆解框架成立**, 拆出来的两块有物理含义。")
        print(f"\n  ⚠️ 纯重复只有 n={len(pure)} —— **极不可靠**, 只报量级不报 CI。")
        print(f"     但它是**下界**性质的证据: 运行噪声不会小于 {rng_pure:.4f}。")
        print(f"     要收紧得跑更多次重跑, 或开 `full_determinism` 后再比。")


if __name__ == "__main__":
    main()
