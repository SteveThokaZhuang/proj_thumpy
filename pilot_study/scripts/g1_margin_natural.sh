#!/usr/bin/env bash
# Q5: 把首 token logit 间隔探针跑到**自然流行率**的旧 300 块集上 (14 个 adapter 全跑)。
#
# 为什么需要这个
#   §5 的 margin 数字全部来自 E2 (400 块, 100% 正例)。§3.4 说 E2 的放大机制是
#   「阈值离散 × 无负例」—— 那么在自然流行率集上, 如果机制成立, mixnorm 的
#   **跨 0 率应当回落**, 而不是继续停在 60%。
#   这是一个**预测**: 跑之前就知道两种结果各自意味着什么。
#
# 顺带解锁的第二个用途
#   g1_mix_check.json (300 块, 自然流行率集) 里每块有 mixnorm 的归一化 gain。
#   探针跑完后就有一套同块号的 margin —— 于是可以问
#   「gain 的极端值 (0 或 30) 是否对应 margin 的跨 0 / 各种子分歧」。
#   这是对 §7 候选 (c) 增益不确定性 的直接检验, 而 E2 的 margin 与这份 gain
#   块号完全不相交 (交集 = 0), 所以之前做不了。
#
# 路径约定 (与 g1_margin_all.sh 对偶, 注意旧集臂目录名不同):
#   mixnorm -> $A/g1_sub/mixnorm       wav 在 $A/g1_sub/g1_eval_wavs_mixnorm
#   own10   -> $A/g1_own_eval/own      wav 在 $A/g1_own_eval/g1_eval_wavs_own
#   探针内部算 wav_dir = {npz_dir}/../g1_eval_wavs_{basename(npz_dir)},
#   所以这两种 npz_dir 的 basename 必须正好是 mixnorm / own。
#
# 用法: setsid nohup srun --jobid=67471 --overlap --ntasks=1 \
#         bash scripts/g1_margin_natural.sh > /tmp/q5.log 2>&1 < /dev/null &
set -u
source /share/home/zhuangruicen/miniconda3/etc/profile.d/conda.sh
conda activate funaudiochat
cd /share/workspace3/zhuangruicen/proj-thumpy/pilot_study

A=real_data/results/annotator
N=300
SEEDS="42 1234 7 2024 3407 31337 55555"

npzdir () { case $1 in own10) echo "$A/g1_own_eval/own";; mixnorm) echo "$A/g1_sub/mixnorm";; esac; }
runname () { if [ "$2" = 42 ]; then echo "g1_$1_sft"; else echo "g1_$1_s$2_sft"; fi; }

echo "=== Q5 开始 $(date '+%F %T') ==="
for arm in own10 mixnorm; do
  for s in $SEEDS; do
    out="$A/g1_margin_nat_${arm}_s${s}.json"
    if [ -s "$out" ]; then
      echo "[skip] $out 已存在"
      continue
    fi
    echo "=== $arm seed $s  开始 $(date '+%T') ==="
    python scripts/ari_g1_margin_probe.py \
      --lora "$A/$(runname $arm $s)/saves" \
      --npz-dir "$(npzdir $arm)" \
      --ids-file "$A/g1_eval_ids.txt" \
      --tag "nat_${arm}_s${s}" --n-chunks $N --sample-seed 0 \
      --out "$out" && echo "=== $arm seed $s  完成 $(date '+%T') ===" \
      || echo "!!! $arm $s 失败, 继续"
  done
done
echo "=== Q5 全部完成 $(date '+%F %T') ==="
