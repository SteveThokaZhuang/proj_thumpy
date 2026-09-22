#!/usr/bin/env bash
# F2 链式任务: X2-Turn 100 会话全会话推理 -> SoulX 30 会话融合层推理
# 在 gpu01 持久步骤内执行 (srun --jobid=45764 --overlap bash -c 'bash ari_f2_run.sh')
set -u
cd /share/workspace3/zhuangruicen/proj-thumpy/pilot_study
A=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator

# 阶段 1: X2-Turn (断点续跑, 已含 E4 的 20 会话)
source /share/home/zhuangruicen/miniconda3/bin/activate x2-turn
python -u scripts/ari_e4_infer.py --n-sessions 100 > ari_logs/f2_x2.log 2>&1
echo "F2_X2_DONE"

# 阶段 2: SoulX 融合层 (30 会话, 独立区域目录)
source /share/home/zhuangruicen/miniconda3/bin/activate fd_analysis
python -u scripts/ari_e5_prep.py --n-sessions 30 --out-dir $A/f2_soulx_regions > ari_logs/f2_prep.log 2>&1
echo "F2_PREP_DONE"

source /share/home/zhuangruicen/miniconda3/bin/activate kimiev-soulx
cd /share/workspace3/zhuangruicen/proj-thumpy/third_party/SoulX-Duplug-training
python -u scripts/duplex_inference.py \
  --config_path /share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts/e5_soulx_config.yaml \
  --eval_dir $A/f2_soulx_regions > /share/workspace3/zhuangruicen/proj-thumpy/pilot_study/ari_logs/f2_soulx.log 2>&1
echo "F2_ALL_DONE"
