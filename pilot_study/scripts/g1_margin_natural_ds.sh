#!/usr/bin/env bash
# #72: 自然流行率集的 **ds 版** margin 探针 —— 补上 §7.4f B/A 缺的那一格。
#
# 判据**已封存在跑之前**:
#   docs/pilot_study/2026-09-17_status.md §7.4f 「🔒 #72 预登记」,
#   封存时刻 **2026-09-18 02:01:33**（当时 nat A 组 14 个在盘、ds 版 0 个）。
#   统计量 R_nat = (B_nat/A_nat) ÷ (B_E2/A_E2), 基准 B/A_E2 = own10 1.52 / mixnorm 0.97。
#   本脚本**只做采集, 不挑说法**。
#
# 它补的是「观测量」那一维:
#   §7.4f 表里 B/A 的两格是 代理×自然(0.94) 与 探针×100%正例(0.97) —— 两维同时变。
#   补完这 14 个产物后, B/A 可以在**同一观测量(探针)** 下跨两个块集比。
#
# 🔴 它补**不上**的: 块集这一维本身。E2-400 与 filt-300 是不同的录音/时刻,
#   流行率与「块身份」不可分离（标签是块的属性, 而 E2 按构造 100% 正例,
#   不存在自然流行率的子集）。所以这是**块集稳健性检验, 不是流行率隔离**。
#
# 口径**逐字对齐** `g1_margin_natural.sh`（A 组那批已经在盘上, 块集必须一模一样）:
#   同一个 --ids-file、--n-chunks 300、--sample-seed 0、同一对 npz-dir。
#   ⚠️ 只换两处: --lora 指向 ds adapter、--out/--tag 换成 nat_ds 命名。
#   若这两批的块集不同, 下面的 B/A 就是在比两个不同的块集 —— 静默错位。
#
# 路径约定 (自然集与 E2 集**不同**, 别照抄 g1_margin_ds.sh):
#   own10   -> $A/g1_own_eval/own     wav 在 $A/g1_own_eval/g1_eval_wavs_own
#   mixnorm -> $A/g1_sub/mixnorm      wav 在 $A/g1_sub/g1_eval_wavs_mixnorm
#
# 用法 (gpu01 空闲; **不要在 login01 跑**):
#   setsid nohup srun --jobid=69621 --overlap --ntasks=1 \
#     bash scripts/g1_margin_natural_ds.sh > /tmp/nat_ds.log 2>&1 < /dev/null &
set -u
source /share/home/zhuangruicen/miniconda3/etc/profile.d/conda.sh
conda activate funaudiochat
cd /share/workspace3/zhuangruicen/proj-thumpy/pilot_study

A=real_data/results/annotator
N=300
SEEDS="42 1234 7 2024 3407 31337 55555"

npzdir () { case $1 in own10) echo "$A/g1_own_eval/own";; mixnorm) echo "$A/g1_sub/mixnorm";; esac; }

echo "=== #72 开始 $(date '+%F %T') ==="
for arm in own10 mixnorm; do
  for s in $SEEDS; do
    out="$A/g1_margin_nat_${arm}_ds${s}.json"
    if [ -s "$out" ]; then
      echo "[skip] $out 已存在"
      continue
    fi
    echo "=== $arm ds$s  开始 $(date '+%T') ==="
    python scripts/ari_g1_margin_probe.py \
      --lora "$A/g1_${arm}_ds${s}_sft/saves" \
      --npz-dir "$(npzdir $arm)" \
      --ids-file "$A/g1_eval_ids.txt" \
      --tag "nat_${arm}_ds${s}" --n-chunks $N --sample-seed 0 \
      --out "$out" && echo "=== $arm ds$s  完成 $(date '+%T') ===" \
      || echo "!!! $arm ds$s 失败, 继续"
  done
done
echo "=== #72 全部完成 $(date '+%F %T') ==="
echo "NAT_DS_DONE $(ls $A/g1_margin_nat_*_ds*.json 2>/dev/null | wc -l)/14"
