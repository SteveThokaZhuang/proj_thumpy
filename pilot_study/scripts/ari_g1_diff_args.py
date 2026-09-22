"""逐字段对比两个 run 的 training_args.bin, 找 f8c_v2 与 g1_own10 之间那 0.052 的来源.

背景 (报告 §7.3): 两套训练集 2,095/2,100 共享且音频逐字节相同、micro-batch 相同、
cutoff_len 已证为空操作, 但同一批 300 块上 F1 = 0.1203 vs 0.1727。人工挑字段比对了
好多项都相同, 所以这里改成**全字段机械对比** —— 不再靠我想得到哪些字段。

只跑一次, 需要 torch, 所以要在 conda funaudiochat 里跑 (srun --overlap 内)。
"""
import os

import torch

A = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
     "real_data/results/annotator")

# 这两个 run 的 output_dir / run_name 天然不同, 不是差异来源
IGNORE = {"output_dir", "run_name", "logging_dir", "overwrite_output_dir"}

# 即使相同也值得亲眼确认的关键字段 (确认"相同"是这个脚本的结论之一, 要留证据)
KEY = ("seed", "data_seed", "per_device_train_batch_size",
       "gradient_accumulation_steps", "num_train_epochs", "learning_rate",
       "lr_scheduler_type", "warmup_ratio", "cutoff_len", "bf16", "fp16",
       "gradient_checkpointing", "packing", "neftune_noise_alpha",
       "disable_shuffling", "model_name_or_path", "template", "dataset",
       "train_on_prompt", "mask_history", "preprocessing_num_workers",
       "dataloader_num_workers", "max_steps", "save_steps", "logging_steps",
       "resume_from_checkpoint", "do_train", "eval_strategy", "optim")


def load(name):
    p = f"{A}/{name}/saves/training_args.bin"
    return torch.load(p, map_location="cpu", weights_only=False) if os.path.exists(p) else None


def main():
    v2, g1 = load("f8c_v2_sft"), load("g1_own10_sft")
    if v2 is None or g1 is None:
        print("缺 training_args.bin, 无法对比")
        return
    dv, dg = vars(v2), vars(g1)
    keys = sorted(set(dv) | set(dg))

    diffs = [(k, dv.get(k, "<缺>"), dg.get(k, "<缺>"))
             for k in keys if dv.get(k, "<缺>") != dg.get(k, "<缺>")]
    real = [d for d in diffs if d[0] not in IGNORE]

    print(f"字段总数 {len(keys)}; 值不同 {len(diffs)} 个 "
          f"(其中路径类 {len(diffs) - len(real)} 个已忽略)")
    print()
    if not real:
        print("  ✓ 除路径外**没有任何字段不同**")
    for k, a, b in real:
        print(f"  {k:34}\n      v2    = {str(a)[:110]}\n      own10 = {str(b)[:110]}")

    print("\n--- 关键字段 (相同也要留证据) ---")
    for k in KEY:
        a, b = str(dv.get(k, "<缺>"))[:38], str(dg.get(k, "<缺>"))[:38]
        print(f"  {k:28} v2={a:40} own10={b:40}{'' if a == b else '  <<< 不同'}")


if __name__ == "__main__":
    main()
