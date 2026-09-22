#!/usr/bin/env bash
# 多种子重复的收尾: 评估两臂 + 量化噪声下限。
# 前置: g1_train_seeds.sh 已产出两个 adapter (自己验一遍, 不信上游退出码)。
# 用法 (gpu04, srun --overlap 内, **conda funaudiochat** —— 评估要 transformers,
# fd_analysis 里没有; 噪声分析只用 numpy, funaudiochat 里也有):
#   G1_SEED=1234 bash g1_seeds_finish.sh
set -uo pipefail
S=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts
A=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator
SEED=${G1_SEED:-1234}
cd "$S"

for arm in own10 mixnorm; do
  f="$A/g1_${arm}_s${SEED}_sft/saves/adapter_model.safetensors"
  [ -f "$f" ] || { echo "FINISH_ABORT: 缺 $f"; exit 2; }
  echo "  [OK ] $arm adapter $(stat -c%s "$f") B"
done

bash g1_eval_seeds.sh || { echo "FINISH_ABORT: 评估失败"; exit 3; }
python ari_g1_noise.py || { echo "FINISH_ABORT: 噪声分析失败"; exit 4; }
echo "SEEDSFINISH_DONE $(date '+%F %T')"
