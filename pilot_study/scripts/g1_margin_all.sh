#!/usr/bin/env bash
# #56 P1: 对 14 个 adapter 全跑首 token logit 间隔探针 (同样的 400 块, 保证配对)。
#
# 约定 (与 g1_e2_eval.sh 一致):
#   mixnorm 臂的 npz/wav 在 g1_e2/mixnorm, own10 臂在 g1_e2/own。
#   adapter 目录: 种子 42 是 g1_<arm>_sft, 其余是 g1_<arm>_s<seed>_sft。
set -u
source /share/home/zhuangruicen/miniconda3/etc/profile.d/conda.sh
conda activate funaudiochat
cd /share/workspace3/zhuangruicen/proj-thumpy/pilot_study

A=real_data/results/annotator
N=400
SEEDS="42 1234 7 2024 3407 31337 55555"

npzdir () { case $1 in own10) echo "$A/g1_e2/own";; mixnorm) echo "$A/g1_e2/mixnorm";; esac; }
runname () { if [ "$2" = 42 ]; then echo "g1_$1_sft"; else echo "g1_$1_s$2_sft"; fi; }

for arm in own10 mixnorm; do
  for s in $SEEDS; do
    out="$A/g1_margin_${arm}_s${s}.json"
    if [ -s "$out" ]; then
      echo "[skip] $out 已存在"
      continue
    fi
    echo "=== $arm seed $s ==="
    python scripts/ari_g1_margin_probe.py \
      --lora "$A/$(runname $arm $s)/saves" \
      --npz-dir "$(npzdir $arm)" \
      --ids-file "$A/g1_eval2_ids.txt" \
      --tag "${arm}_s${s}" --n-chunks $N --sample-seed 0 \
      --out "$out" || echo "!!! $arm $s 失败, 继续"
  done
done
echo "全部完成 $(date +%H:%M:%S)"
