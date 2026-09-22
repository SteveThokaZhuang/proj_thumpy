#!/usr/bin/env bash
# 在 gpu02 (kimi tmux 窗口, 4×RTX 4090 D) 启动 4 个 worker 并行跑指定 split
# 用法: bash run_real_workers.sh validation   (或 test)
# 每张卡 1 个 worker (分 4 片, 断点续跑); 两个 split 分两批跑, 避免 32G 内存吃紧
set -e
cd "$(dirname "$0")"
source ./env.sh

SPLIT="${1:-validation}"
for gpu in 0 1 2 3; do
  LOG="real_${SPLIT}_w${gpu}.log"
  CUDA_VISIBLE_DEVICES=$gpu SPLIT=$SPLIT SHARD_ID=$gpu NUM_SHARDS=4 \
    nohup python scripts/run_real_pipeline.py > "$LOG" 2>&1 &
  echo "worker gpu$gpu: split=$SPLIT shard=$gpu/4 log=$LOG pid=$!"
done
echo "4 个 worker 已启动 (split=$SPLIT)"
