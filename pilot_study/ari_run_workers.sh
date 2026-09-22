#!/usr/bin/env bash
# ARI 实验全量特征提取启动脚本
# 用法: 在 kimi 窗格 (gpu02 srun 交互作业) 内直接执行, 或通过持久步骤:
#   srun --jobid=<kimi作业号> --overlap bash -c 'bash ari_run_workers.sh; wait'
# 注意: 勿用 "srun --overlap bash -c '... & exit'" 短步骤模式 —
#       SLURM 步骤结束会按 cgroup 清理全部子进程 (见 docs/log/2026-08-24_ari_engineering.md)
# CANDOR: 8 worker (seek-read 窗口级解码, I/O 密集)
# Behavior-SD: 8 worker, validation+test 全量 + train 前 8 tar
set -u
cd /share/workspace3/zhuangruicen/proj-thumpy/pilot_study
PY=/share/home/zhuangruicen/miniconda3/envs/fd_analysis/bin/python
OUT=real_data/results/analysis/ari
mkdir -p "$OUT" ari_logs

for w in 0 1 2 3 4 5 6 7; do
  nohup nice -n 10 $PY scripts/ari_extract_candor.py --worker $w --n-workers 8 \
    --out "$OUT/candor_events_w$w.csv" > ari_logs/candor_w$w.log 2>&1 &
  echo "candor w$w pid $!"
done

for w in 0 1 2 3 4 5 6 7; do
  nohup nice -n 10 $PY scripts/ari_extract_behavior.py --worker $w --n-workers 8 \
    --splits validation,test,train --tar-limit 8 \
    --out "$OUT/behavior_events_w$w.csv" > ari_logs/behavior_w$w.log 2>&1 &
  echo "behavior w$w pid $!"
done

echo LAUNCHED
