#!/usr/bin/env bash
# Q3 的 B 组 (seed 固定 42, 变 data_seed) 补跑 margin 探针 —— 14 个 ds adapter。
#
# 为什么必须补: ari_g1_margin_analyze.py Q5 报「mixnorm 各种子平均 margin 的 SD
# 比 own10 大 2.88×」, 而 margin 产物只按 `_s{seed}` 命名 —— 变 data_seed 的那一半
# **根本没有探针**。没有它就没法回答「那个离散是初值还是数据顺序」。
#
# 🔴 块必须与 A 组**逐个相同**: 用从现有 margin 文件里抽出的
#    g1_margin_ds_ids.txt (400 行), 而不是 g1_eval2_ids.txt + 重新抽样 ——
#    后者依赖 sample-seed 复现同一批块, 一旦探针脚本的抽样逻辑变了就静默错位。
#    (n_chunks=400 且 ids 也是 400 ⇒ probe 里 `400 < 400` 为假 ⇒ 不再抽样。)
#
# 约定与 g1_margin_all.sh 完全一致: npz/wav 分臂, adapter 目录 g1_<arm>_ds<ds>_sft。
set -u
source /share/home/zhuangruicen/miniconda3/etc/profile.d/conda.sh
conda activate funaudiochat
cd /share/workspace3/zhuangruicen/proj-thumpy/pilot_study

A=real_data/results/annotator
N=400
SEEDS="42 1234 7 2024 3407 31337 55555"

npzdir () { case $1 in own10) echo "$A/g1_e2/own";; mixnorm) echo "$A/g1_e2/mixnorm";; esac; }

echo "开始 $(date +%F' '%H:%M:%S)"
for arm in mixnorm own10; do
  for s in $SEEDS; do
    out="$A/g1_margin_${arm}_ds${s}.json"
    if [ -s "$out" ]; then
      echo "[skip] $out 已存在"
      continue
    fi
    t0=$(date +%s)
    echo "=== $arm ds$s  $(date +%H:%M:%S) ==="
    python scripts/ari_g1_margin_probe.py \
      --lora "$A/g1_${arm}_ds${s}_sft/saves" \
      --npz-dir "$(npzdir $arm)" \
      --ids-file "$A/g1_margin_ds_ids.txt" \
      --tag "${arm}_ds${s}" --n-chunks $N --sample-seed 0 \
      --out "$out" || echo "!!! $arm ds$s 失败, 继续"
    echo "    用时 $(( $(date +%s) - t0 ))s"
  done
done
echo "全部完成 $(date +%F' '%H:%M:%S)"
