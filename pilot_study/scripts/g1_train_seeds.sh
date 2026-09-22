#!/usr/bin/env bash
# G1 多种子重复: 同臂、同数据、**只换随机种子**重训两臂, 干净地量化 run-to-run 噪声。
#
# 为什么需要: 报告 §7 发现 f8c_v2 与 g1_own10 的训练数据几乎同源 (2,095/2,100 块
# 逐字节相同、答案全同、同为 132 步), 却差 0.052 F1 —— 说明噪声下限比待检效应
# (0.033) 还大。但那一对对照还混着 5 个块的数据差异和 micro-batch 切分 (8×4 vs
# 4×8), 只能算噪声的**上界估计**。这里换种子重跑同一条流水线, 数据逐字节不变,
# 唯一变量确实只有种子 —— 给出干净的噪声测量。
#
# 数据复用: 直接沿用已注册的 g1-own10 / g1-mixnorm 数据集 (它们的 train.jsonl 未变,
# 仍指向 g1_own10_sft/wavs 与 g1_mixnorm_sft/wavs), 所以**不需要**复制 4G wavs,
# 也不需要重新注册。config 里 dataset 名保持不变, 只改 output_dir 与 seed。
#
# 用法 (gpu04, srun --overlap 内, conda funaudiochat):
#   G1_SEED=1234 bash g1_train_seeds.sh              # 两臂
#   G1_SEED=7 G1_ARMS=own10 bash g1_train_seeds.sh   # 只 own10 臂
set -uo pipefail
A=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator
S=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts
PROJECT_ROOT=/share/workspace3/zhuangruicen/proj-thumpy/third_party/Fun-Audio-Chat
LOGS=$A/g1_logs
SEED=${G1_SEED:-1234}
mkdir -p "$LOGS"

# 与主实验同一个基座 config, 同一套 sed 覆写 —— 保证除了 output_dir/seed 之外
# 一切逐字相同 (batch 4×accum 8 / save_steps 25 都沿用)。
make_config () {
  local arm=$1 out=$2 cfg="$2/g1_config.yaml"
  mkdir -p "$out/saves"
  sed -e "s|^dataset:.*|dataset: g1-$arm|" \
      -e "s|^output_dir:.*|output_dir: $out/saves|" \
      -e "s|^save_steps:.*|save_steps: 25|" \
      -e "s|^per_device_train_batch_size:.*|per_device_train_batch_size: 4|" \
      -e "s|^gradient_accumulation_steps:.*|gradient_accumulation_steps: 8|" \
      "$S/f8c_v2_config.yaml" > "$cfg"
  # 基座 config 里没有 seed 行 -> 追加即是唯一来源, 不会与既有值冲突。
  # LLaMA-Factory 的 seed 同时决定权重初始化与数据打乱顺序。
  echo "seed: $SEED" >> "$cfg"
  echo "save_total_limit: 2" >> "$cfg"
  # 残留的旧 adapter 会骗过末尾的成功判定
  rm -f "$out/saves/adapter_model.safetensors"
}

run_arm () {
  local arm=$1 out=$2 rc
  echo "########## $arm (seed=$SEED)  $(date '+%F %T') ##########"
  make_config "$arm" "$out"
  llamafactory-cli train "$out/g1_config.yaml" 2>&1 | tee "$LOGS/train_${arm}_s${SEED}.log"
  rc=${PIPESTATUS[0]}          # tee 会吞掉真实退出码
  echo "########## $arm 结束 rc=$rc  $(date '+%F %T') ##########"
  RC[$arm]=$rc
}

export AUDIO_PLACEHOLDER="<|audio_bos|><|AUDIO|><|audio_eos|>"
export DISABLE_VERSION_CHECK=1
unset FORCE_TORCHRUN                  # NCCL 会炸, 单进程
export PYTHONPATH="$PROJECT_ROOT/training/plugin:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJECT_ROOT/training"

declare -A RC
# 默认两臂; 只关心单臂时设 G1_ARMS=own10 (脚本 g1_seeds_extend.sh 就是这么用的)。
# 臂列表走同一个 make_config, 所以单臂跑出来的 config 与两臂跑时逐字相同。
ARMS=${G1_ARMS:-"own10 mixnorm"}
for arm in $ARMS; do
  run_arm "$arm" "$A/g1_${arm}_s${SEED}_sft"
done

echo "=== 汇总 (以 adapter 文件为准, 不信退出码) ==="
fail=0
for arm in $ARMS; do
  f="$A/g1_${arm}_s${SEED}_sft/saves/adapter_model.safetensors"
  if [ -f "$f" ]; then
    echo "  [OK ] $arm rc=${RC[$arm]} $(stat -c%s "$f") B"
  else
    echo "  [FAIL] $arm rc=${RC[$arm]} 无 adapter"
    fail=1
  fi
done
[ $fail -eq 0 ] && echo "SEEDS_DONE $(date '+%F %T')" || echo "SEEDS_FAILED $(date '+%F %T')"
exit $fail
