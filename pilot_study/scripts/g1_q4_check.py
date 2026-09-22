"""Q4 中性校验 —— 纯读盘, 0 GPU。结果落地前先写好。

## Q4 是什么实验

Q4 = **同配置重跑一次**, 唯一改动是加 `DisableSpeechCallback` 修 loss 记录
(修前 23 个 run 的 `trainer_log.jsonl` 里 loss 恒为 0.0, 见 §1.1)。
训练数据、seed、data_seed 全部不变。

所以 Q4 是一次**零假设实验**: 它应该什么都改变不了。
它的价值全在"校验"二字 —— 它同时校验两件事:

1. **那个 callback 修复是纯记录性的**, 没有意外改动训练。
2. **`seed` 到底控制住了多少东西。**

## 判据口径 (2026-09-17 23:5x **更正过一次**, 经过见下)

先说结论: **尺子就是跨种子 SD (§1.3 定的那个), 判据是「|差| < 跨种子 SD」。**

> 🔴 **本脚本第一版把判据写错了, 而且错得自相矛盾。**
> 它写的是「Q4 与原来的差应**远小于**跨种子 SD, 因为 seed 同时锁住初值
> 和数据顺序, 重跑只差非确定性 kernel」。**但 §1.3 早就写明**
> 「少跑一次 `audio_invert_tower` 前向**可能改变 dropout 消耗的 RNG 流,
> 轨迹会漂移**」—— 也就是说 Q4 这次**根本不是纯重跑**,
> 它是 RNG 流被挪位之后的**另一次抽样**。抽样之间差一个跨种子 SD
> 是正常的, 不是异常。
>
> 实测出来后 (差 +0.0267, 跨种子 SD 0.0309) 第一版吐出一句
> 「**seed 没锁住它看起来锁住的东西**」—— 那是**误报**。
> 按 §1.3 的预登记判据, 0.0267 < 0.0309, **Q4 是通过的**。
>
> **教训**: 把检查写得更严 ≠ 更好。严错了模型, 就会在正常结果上
> 拉警报, 而**误报会侵蚀真警报的可信度**。定阈值前先问
> 「这个系统的噪声模型是什么」—— 答案往往已经写在更早的文档里了。

## 自检

脚本自己算 A 组(7 个原 seed)的跨种子 SD, **必须复现 0.0309**。
对不上说明读盘/配对逻辑有 bug, 那么 Q4 的比较也不可信。

用法:
  python scripts/g1_q4_check.py
"""
import json
import os

import numpy as np

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
SEEDS = [42, 1234, 7, 2024, 3407, 31337, 55555]
ARMS = ["own10", "mixnorm"]

BASE_SD = 0.0309          # §7.2: 原 7 个种子的跨种子 SD
BASE_DELTA_42 = 0.0440    # 原 ds42 (即 tag 无后缀那个) 的配对 ΔF1


def suffix(seed):
    return "" if seed == 42 else f"_s{seed}"


def load(name):
    p = f"{ANNOT}/g1_eval_{name}.json"
    return next(iter(json.load(open(p)).values())) if os.path.exists(p) else None


def main():
    print("=" * 74)
    print("  Q4 中性校验 —— 同配置重跑应什么都改变不了")
    print("=" * 74)

    # ---------- 自检: 跨种子 SD 必须复现 ----------
    d_seed = {}
    per_arm = {a: [] for a in ARMS}          # 各臂自己的跨种子 F1, 用来定单臂的噪声尺子
    for s in SEEDS:
        a, b = load(f"own10{suffix(s)}_filt"), load(f"mixnorm{suffix(s)}_filt")
        if a and b:
            d_seed[s] = a["f1"] - b["f1"]
            per_arm["own10"].append(a["f1"])
            per_arm["mixnorm"].append(b["f1"])
    sd_arm = {a: float(np.std(v, ddof=1)) for a, v in per_arm.items()}
    if len(d_seed) < 2:
        print("❌ 原 7 个种子的 filt 产物不全, 无法建立噪声基准")
        return
    dv = np.array(list(d_seed.values()))
    sd = dv.std(ddof=1)
    ok_sd = abs(sd - BASE_SD) < 0.0025
    print(f"\n  自检 @ 原 7 种子: 跨种子 SD = {sd:.4f}  期望 {BASE_SD:.4f}  "
          f"{'✅' if ok_sd else '❌ 差 %+.4f' % (sd - BASE_SD)}")
    if not ok_sd:
        print("  ❌ 自检失败 —— 先修代码, 别看 Q4 结果")
        return
    print(f"  (原 ds42 配对 ΔF1 = {d_seed.get(42, float('nan')):+.4f}, "
          f"基准 {BASE_DELTA_42:+.4f})")

    # ---------- Q4 vs 原 ds42 ----------
    q4 = {a: load(f"{a}_q4_filt") for a in ARMS}
    if any(v is None for v in q4.values()):
        miss = [a for a in ARMS if q4[a] is None]
        print(f"\n  ⏳ Q4 产物还没齐 (缺 {miss}) —— 等评估跑完再执行本脚本")
        print(f"  但脚本本身已通过自检, 届时可直接采信。")
        return

    orig = {a: load(f"{a}_filt") for a in ARMS}
    print("\n=== 逐臂 F1: Q4 重跑 vs 原 ds42 ===")
    print(f"  {'arm':>8}{'原 ds42':>12}{'Q4 重跑':>12}{'差':>12}")
    for a in ARMS:
        print(f"  {a:>8}{orig[a]['f1']:>12.4f}{q4[a]['f1']:>12.4f}"
              f"{q4[a]['f1'] - orig[a]['f1']:>+12.4f}")

    d_q4 = q4["own10"]["f1"] - q4["mixnorm"]["f1"]
    d_or = orig["own10"]["f1"] - orig["mixnorm"]["f1"]
    gap = d_q4 - d_or

    print(f"\n=== 关键: 配对 ΔF1 的重跑稳定性 ===")
    print(f"  原 ds42  ΔF1 = {d_or:+.4f}")
    print(f"  Q4 重跑  ΔF1 = {d_q4:+.4f}")
    print(f"  两者之差     = {gap:+.4f}")
    # ⚠️ 判据口径 (2026-09-17 23:5x 更正)
    #
    # 本脚本最初写的是「Q4 与原来的差应**远小于**跨种子 SD, 因为 seed 同时
    # 锁住初值和数据顺序, 重跑只差非确定性 kernel」。**那个预判是错的**,
    # 而且它自相矛盾: §1.3 早就写明「少跑一次 `audio_invert_tower` 前向
    # **可能改变 dropout 消耗的 RNG 流, 轨迹会漂移**」。也就是说 Q4 这次
    # **根本不是纯重跑** —— 它是 RNG 流被挪位后的**另一次抽样**, 而抽样
    # 之间差一个跨种子 SD 是正常的, 不是异常。
    #
    # 所以正确的比较是: 「Q4 这一次, 像不像同一分布里抽出来的一个点」,
    # 而不是「它是否几乎没变」。§1.3 定的判据 (落在跨种子噪声内) 才是对的。
    #
    # 单臂用**该臂自己**的跨种子 SD 当尺子 (own10 与 mixnorm 的尺度不同),
    # 配对差用配对的跨种子 SD。
    print(f"\n  判据: Q4 是**另一次抽样**(RNG 流被回调挪位), 不是纯重跑。")
    print(f"        故比的是「像不像同一分布里的一个点」, 而非「是否几乎没变」。")
    print(f"\n  单臂 (尺子 = 各臂自己的跨种子 SD):")
    for a in ARMS:
        ch = q4[a]["f1"] - orig[a]["f1"]
        print(f"    {a:>8}: 改 {ch:+.4f}, 该臂跨种子 SD = {sd_arm[a]:.4f}"
              f"  ->  {ch/sd_arm[a]:+.2f} SD")
    print(f"\n  配对 ΔF1: 差 {gap:+.4f}, 配对跨种子 SD = {sd:.4f}"
          f"  ->  {gap/sd:+.2f} SD")
    print(f"  预登记判据 (§1.3): |差| < 跨种子 SD = {sd:.4f}"
          f"   ->  {'✅ 通过' if abs(gap) < sd else '❌ 未通过'}")

    print(f"\n  ⚠️ 功效上限 (必须一起报): **只有 n=1 次重跑**, 尺子 σ≈{sd:.3f}。")
    print(f"     这个检验只能发现**量级 >= 一个跨种子 SD 的粗大改动**,")
    print(f"     发现不了更小的。所以「通过」= **未检出粗大改动**,")
    print(f"     不等于「已证明该回调在训练层面完全惰性」。")
    print(f"     要坐实惰性, 得跑**多次**重跑去看分布, 或用逐位可复现的设置重做。")

    # ---------- Q4 的 loss 是否可读 ----------
    print(f"\n=== 附带: Q4 的 loss 记录是否修好 ===")
    for a in ARMS:
        l = f"{ANNOT}/g1_{a}_q4_sft/saves/trainer_log.jsonl"
        if not os.path.exists(l):
            print(f"  {a}: ⚠️ 没找到 {l}")
            continue
        tot = zero = 0
        vals = []
        for line in open(l):
            try:
                e = json.loads(line)
            except Exception:
                continue
            if "loss" in e:
                tot += 1
                if e["loss"] == 0.0:
                    zero += 1
                else:
                    vals.append(e["loss"])
        print(f"  {a}: {tot} 行, 其中恰为 0.0 的 {zero} 行")
        if vals:
            print(f"       非零 loss 样例: {[round(v, 4) for v in vals[:5]]}")
            print(f"       ⇒ ✅ 修复生效 (修复前全项目 23 个 run 无一例外全是 0.0)")
        elif tot:
            print(f"       ⇒ ❌ 仍然全 0 —— 修复没生效, 与 §1.1 修复前同样。")


if __name__ == "__main__":
    main()
