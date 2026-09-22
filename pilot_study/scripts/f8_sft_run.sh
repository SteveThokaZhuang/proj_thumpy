#!/usr/bin/env bash
# F8: Fun-Audio-Chat 话轮状态 LoRA SFT 启动脚本
# 用法 (gpu02, kimi2 作业, funaudiochat 环境):
#   bash /share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts/f8_sft_run.sh [SMOKE=1]
set -u
PROJECT_ROOT=/share/workspace3/zhuangruicen/proj-thumpy/third_party/Fun-Audio-Chat
export AUDIO_PLACEHOLDER="<|audio_bos|><|AUDIO|><|audio_eos|>"
export DISABLE_VERSION_CHECK=1
export FORCE_TORCHRUN=1
export PYTHONPATH="$PROJECT_ROOT/training/plugin:$PROJECT_ROOT:${PYTHONPATH:-}"
export TIMEOUT=180000000
cd "$PROJECT_ROOT/training"

CONFIG=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts/f8_sft_config.yaml
if [ "${SMOKE:-0}" = "1" ]; then
  # 冒烟: 覆盖为 1 步验证
  CONFIG=$(
    sed -e 's/^max_steps.*//' -e 's/^num_train_epochs.*/num_train_epochs: 0.001/' \
        -e 's/^save_steps.*/save_steps: 100000/' \
        -e 's/^per_device_train_batch_size.*/per_device_train_batch_size: 2/' \
        "$CONFIG"
  )
  echo "$CONFIG" > /tmp/f8_smoke_config.yaml
  CONFIG=/tmp/f8_smoke_config.yaml
fi

llamafactory-cli train "$CONFIG" 2>&1 | tee /share/workspace3/zhuangruicen/proj-thumpy/pilot_study/ari_logs/f8_sft.log
