#!/usr/bin/env bash
# 历史对照臂: 拆开 f8c_v2 与 g1_own10 之间那 0.052 F1 差距到底来自哪个配置项。
#
# 缘起(2026-09-13): 文档 §7.2 曾推断"这 0.052 是 micro-batch 8x4 vs 4x8 造成的"。
# **该推断是错的** —— 读 f8c_v2_sft/saves/training_args.bin 发现它实际跑的是
# micro=4/accum=8, 与 g1_own10 相同。真正的差异是 f8c_v2 用了 config_small
# (cutoff_len 1024, 而非 2048)。
#
# 所以本脚本在**同一个臂、同一份数据、同一个种子(默认 42)**上各只改一个变量,
# 两臂都与既有的 g1_own10 (cutoff 2048 / 4x8 / seed 42 / F1 0.1727) 对比:
#   c1024_4x8 -> 隔离 cutoff_len   (v2 的真实配置差异)
#   c2048_8x4 -> 隔离 micro-batch  (用户最初要问的那个; 注意 24GB 上是临界配置)
#
# 数据复用已注册的 g1-own10 数据集, 不需要复制 wavs。
# 用法 (gpu04, srun --overlap 内, conda funaudiochat): bash g1_train_confound.sh
set -uo pipefail
A=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator
S=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts
PROJECT_ROOT=/share/workspace3/zhuangruicen/proj-thumpy/third_party/Fun-Audio-Chat
LOGS=$A/g1_logs
mkdir -p "$LOGS"

# $1=tag $2=cutoff $3=micro $4=accum
make_config () {
  local tag=$1 cutoff=$2 micro=$3 accum=$4 out="$A/g1_own10_${1}_sft" cfg
  cfg="$out/g1_config.yaml"
  mkdir -p "$out/saves"
  sed -e "s|^dataset:.*|dataset: g1-own10|" \
      -e "s|^output_dir:.*|output_dir: $out/saves|" \
      -e "s|^cutoff_len:.*|cutoff_len: $cutoff|" \
      -e "s|^save_steps:.*|save_steps: 25|" \
      -e "s|^per_device_train_batch_size:.*|per_device_train_batch_size: $micro|" \
      -e "s|^gradient_accumulation_steps:.*|gradient_accumulation_steps: $accum|" \
      "$S/f8c_v2_config.yaml" > "$cfg"
  echo "save_total_limit: 2" >> "$cfg"
  # 不写 seed 行 -> 沿用 LLaMA-Factory 默认 42, 与 g1_own10 一致 (这才是对照)
  rm -f "$out/saves/adapter_model.safetensors"
}

run_arm () {
  local tag=$1 cutoff=$2 micro=$3 accum=$4 rc
  echo "########## $tag (cutoff=$cutoff, ${micro}x${accum})  $(date '+%F %T') ##########"
  make_config "$tag" "$cutoff" "$micro" "$accum"
  llamafactory-cli train "$A/g1_own10_${tag}_sft/g1_config.yaml" \
    2>&1 | tee "$LOGS/train_confound_${tag}.log"
  rc=${PIPESTATUS[0]}          # tee 会吞掉真实退出码
  echo "########## $tag 结束 rc=$rc  $(date '+%F %T') ##########"
  RC[$tag]=$rc
}

export AUDIO_PLACEHOLDER="<|audio_bos|><|AUDIO|><|audio_eos|>"
export DISABLE_VERSION_CHECK=1
unset FORCE_TORCHRUN                  # NCCL 会炸, 单进程
export PYTHONPATH="$PROJECT_ROOT/training/plugin:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJECT_ROOT/training"

declare -A RC
run_arm c1024_4x8 1024 4 8
run_arm c2048_8x4 2048 8 4

echo "=== 汇总 (以 adapter 文件为准, 不信退出码) ==="
fail=0
for tag in c1024_4x8 c2048_8x4; do
  f="$A/g1_own10_${tag}_sft/saves/adapter_model.safetensors"
  if [ -f "$f" ]; then
    echo "  [OK ] $tag rc=${RC[$tag]} $(stat -c%s "$f") B"
  else
    echo "  [FAIL] $tag rc=${RC[$tag]} 无 adapter"
    fail=1
  fi
done
[ $fail -eq 0 ] && echo "CONFOUND_DONE $(date '+%F %T')" \
                || echo "CONFOUND_PARTIAL $(date '+%F %T')"
exit $fail
