#!/usr/bin/env bash
# #64 k=4 扩集建库: 为 g1_eval4_ids.txt 构建两臂观测空间音频。
#
# **不重训任何模型**: id 与 g1_ids.txt (2400) 及 g1_eval2_ids.txt (1600) 完全
# 不相交, **且不与任何已用块时间重叠** (5s 步进 / 10s chunk 意味着相邻块共享
# 50% 音频, 只查 id 不相交是不够的 —— 见 ari_g1_eval4_ids.py 的说明)。
#
# 与 E2 的关键差别: 本集**不要求块内有事件**, 所以是自然流行率 (3.58%,
# 1200 块 / 43 事件)。E2 的 100% 正例率正是 §3.3 那个方差伪影的来源。
#
# 用法 (kimi2 / gpu04, srun --overlap 内, **conda fd_analysis** —— 建库要
# soundfile/scipy; funaudiochat 没有这些):
#   bash scripts/g1_e4_build.sh
set -uo pipefail
A=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator
S=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts
LOG=$A/g1_logs/e4_build.log
mkdir -p "$A/g1_logs"
cd "$S"

echo "=== E4 BUILD START $(date '+%F %T') ===" | tee "$LOG"
# 一趟解码同时产出两种观测空间 (mp3 解码是主要开销)
python ari_g1_prep_subset.py --ids "$A/g1_eval4_ids.txt" \
    --input-mode own,mixnorm --out "$A/g1_e4" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

for m in own mixnorm; do
  n=$(ls "$A/g1_e4/$m"/*.npz 2>/dev/null | wc -l)
  echo "  $m: $n npz" | tee -a "$LOG"
done
need=$(wc -l < "$A/g1_eval4_ids.txt")
have=$(ls "$A/g1_e4/own"/*.npz 2>/dev/null | wc -l)
if [ "$rc" -eq 0 ] && [ "$have" -eq "$need" ]; then
  echo "E4_BUILD_DONE $(date '+%F %T')" | tee -a "$LOG"
else
  echo "E4_BUILD_FAILED rc=$rc have=$have need=$need $(date '+%F %T')" | tee -a "$LOG"
  exit 1
fi
