#!/usr/bin/env bash
# #53 扩大评估集: 为 g1_eval2_ids.txt 构建两臂观测空间音频.
#
# **不重训任何模型**: 新 id 与 g1_ids.txt (2100 训练 + 300 旧评估) 完全不相交,
# 所以现有 14 个 adapter 都没见过它们, 可以直接评。
#
# 用法 (gpu01, srun --overlap 内, **conda fd_analysis** —— 建库需要 soundfile/scipy):
#   bash scripts/g1_e2_build.sh
set -uo pipefail
A=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator
S=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts
LOG=$A/g1_logs/e2_build.log
mkdir -p "$A/g1_logs"
cd "$S"

echo "=== E2 BUILD START $(date '+%F %T') ===" | tee "$LOG"
# 一趟解码同时产出两种观测空间 (mp3 解码是主要开销)
python ari_g1_prep_subset.py --ids "$A/g1_eval2_ids.txt" \
    --input-mode own,mixnorm --out "$A/g1_e2" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

for m in own mixnorm; do
  n=$(ls "$A/g1_e2/$m"/*.npz 2>/dev/null | wc -l)
  echo "  $m: $n npz" | tee -a "$LOG"
done
need=$(wc -l < "$A/g1_eval2_ids.txt")
have=$(ls "$A/g1_e2/own"/*.npz 2>/dev/null | wc -l)
if [ "$rc" -eq 0 ] && [ "$have" -eq "$need" ]; then
  echo "E2_BUILD_DONE $(date '+%F %T')" | tee -a "$LOG"
else
  echo "E2_BUILD_FAILED rc=$rc have=$have need=$need $(date '+%F %T')" | tee -a "$LOG"
  exit 1
fi
