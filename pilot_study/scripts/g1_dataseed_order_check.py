"""Q3 的空操作检查 —— **直接看采样顺序**, 0 GPU, 几秒钟。

## 为什么需要重做这个检查

`g1_q3q4_train.sh` 末尾原本用「7 个 adapter 的 md5 是否全同」来判断
`data_seed` 有没有生效。**这个检查是无效的**, 2026-09-17 20:15 发现:

    自然实验: `g1_own10_sft`(seed=42, data_seed 未设)
            vs `g1_own10_ds42_sft`(seed=42, data_seed=42)
    两者 config **只差 output_dir 与那两行追加**——
    而 `data_seed=None` 在 `SeedableRandomSampler.__init__` 里等价于
    `torch.random.initial_seed()`, 即 `set_seed(42)` 之后的 **42**。
    也就是说这**是同一份有效配置**。
    但两个 adapter 的 md5 **不同**。

原因: 基座 config 没有 `full_determinism` ⇒ 默认 False ⇒ cuBLAS/cuDNN
的非确定性 kernel 让**每次重跑都产生不同的比特**。
于是「7 个 md5 互不相同」在 `data_seed` **完全无效**时也照样成立 ——
这个断言永远不会 fail, 提供的是**虚假保证**。

## 换成的检查

不猜产物, **直接迭代真实的 sampler**, 打印每个 data_seed 下前若干个样本下标。
顺序不同 ⇒ `data_seed` 确实在控制打乱; 顺序相同 ⇒ 空操作, Q3 无效。
判据是确定性的、可复跑的, 且与 GPU 无关。

顺带验证 `data_seed=None ≡ 42`(即 ds=42 那个 run 应当复现原始 run 的**数据顺序**)。
"""
import torch
from torch.utils.data import RandomSampler
from accelerate.data_loader import SeedableRandomSampler

N = 2100                       # g1-own10 / g1-mixnorm 的训练集大小 (wc -l train.jsonl)
DS_LIST = [42, 1234, 7, 2024, 3407, 31337, 55555]
SHOW = 8


def order(data_seed, epoch=0, n_show=SHOW):
    """复刻 transformers -> accelerate 的实际包装方式, 取前 n_show 个下标。"""
    s = SeedableRandomSampler(data_source=RandomSampler(range(N)), data_seed=data_seed)
    s.set_epoch(epoch)
    it = iter(s)
    return [next(it) for _ in range(n_show)]


def main():
    print(f"训练集 N = {N}, 前 {SHOW} 个样本下标 (epoch 0)\n")
    seen = {}
    for ds in DS_LIST:
        o = order(ds)
        seen[ds] = o
        print(f"  data_seed={ds:<6} {' '.join(f'{i:5d}' for i in o)}")

    # None 走的是 torch.random.initial_seed(); HF 的 set_seed(42) 之后就是 42
    torch.manual_seed(42)
    o_none = order(None)
    print(f"\n  data_seed=None   {' '.join(f'{i:5d}' for i in o_none)}"
          f"   <- 未设时的等价物")

    print("\n" + "=" * 70)
    uniq = {tuple(v) for v in seen.values()}
    print(f"  7 个 data_seed 的不同顺序数: {len(uniq)}/7")
    if len(uniq) == 1:
        print("  ❌ 全部相同 —— data_seed 是空操作, Q3 结果不可解读!")
        raise SystemExit(1)
    print("  ✅ 顺序互不相同 —— data_seed 确实在控制打乱顺序。")

    o42 = seen[42]
    print(f"\n  data_seed=None 与 data_seed=42 的顺序是否相同: "
          f"{'是 ✅ (ds=42 应复现原始 run 的数据顺序)' if o_none == o42 else '否 ❌'}")

    # 该检查与 md5 检查的对照: 说明为什么旧的无效
    print("\n" + "=" * 70)
    print("  注: 本检查是**确定性**的, 同一份代码反复跑结果逐位相同;")
    print("      而 md5 检查受 full_determinism=False 的 kernel 非确定性支配,")
    print("      data_seed 无效时它也会给出 7 个不同的 md5 —— 故已弃用。")


if __name__ == "__main__":
    main()
