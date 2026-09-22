"""G1 消融: 从指定观测空间的 chunk 集构建 F8c-v2 事件任务训练数据.

与 ari_f8c_v2_build.py 完全同口径 (TAU=0.1 / 正例 3/4 / 硬负例 1/8),
额外**排除冻结评估集**, 音频取自 --npz-dir。

用法 (fd_analysis):
  python scripts/ari_g1_build.py --npz-dir .../g1_mixnorm10 \
      --out .../g1_mixnorm10_sft --chunk-s 10 --n-samples 2100
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_g1_common import (eval_chunk_ids, select_train_chunks,  # noqa: E402
                           write_jsonl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-s", type=float, default=10.0)
    ap.add_argument("--n-samples", type=int, default=2100)
    args = ap.parse_args()

    excl = frozenset(eval_chunk_ids(300))
    chosen = select_train_chunks(args.npz_dir, args.n_samples,
                                 chunk_s=args.chunk_s, exclude_ids=excl)
    n, n_pos = write_jsonl(chosen, args.npz_dir, args.out,
                           chunk_s=args.chunk_s)
    print(f"samples: {n} (pos {n_pos}, neg {n - n_pos}), "
          f"eval-excluded {len(excl)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
