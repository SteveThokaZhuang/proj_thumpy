#!/usr/bin/env bash
# k=4b 加块建库: 只建**新增的 400 块** (g1_eval4b_ids.txt)。
#
# 为什么只建 400: `ari_g1_eval4_ids.py --n-eval 1600` 的轮转取样是**前缀稳定**的
# —— 1600 块里的前 1200 块与既有 `g1_eval4_ids.txt` **逐位相同** (已 assert)。
# 所以既有 1200 块的 14 个产物一个都不用重跑, 只补新增的 400 块, 再 merge。
# 这不是省事, 是**必须**: 重跑会把 1200 块的逐块结果换成另一批随机数,
# 而 §5.8b 的一切 (SE_chunk = 0.009725, ΔF1 = +0.0210) 都是挂在那些块上的。
#
# 用法 (srun --overlap 内, **conda fd_analysis** —— 建库要 soundfile/scipy):
#   bash scripts/g1_e4b_build.sh
set -uo pipefail
# 解释器**写死**: 建库要 soundfile/scipy, 只有 fd_analysis 有。旧脚本用裸 `python`
# 是隐式依赖调用者的 activate —— 换个人换台机器就静默跑错环境。
PY=/share/home/zhuangruicen/miniconda3/envs/fd_analysis/bin/python
A=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator
S=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts
LOG=$A/g1_logs/e4b_build.log
mkdir -p "$A/g1_logs"
cd "$S"
echo "py: $PY" | tee -a "$LOG"
"$PY" -c "import soundfile, scipy; print('  deps ok', soundfile.__version__)" \
  | tee -a "$LOG" || { echo "E4B_BUILD_FAILED 环境不对" | tee -a "$LOG"; exit 1; }

echo "=== E4B BUILD START $(date '+%F %T') ===" | tee "$LOG"
"$PY" ari_g1_prep_subset.py --ids "$A/g1_eval4b_ids.txt" \
    --input-mode own,mixnorm --out "$A/g1_e4b" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

for m in own mixnorm; do
  n=$(ls "$A/g1_e4b/$m"/*.npz 2>/dev/null | wc -l)
  echo "  $m: $n npz" | tee -a "$LOG"
done
need=$(grep -c . "$A/g1_eval4b_ids.txt")
have=$(ls "$A/g1_e4b/own"/*.npz 2>/dev/null | wc -l)
# 自检: 新块必须与旧 1200 块零交集, 否则 merge 时会把同一块算两遍
inter=$(comm -12 <(sort "$A/g1_eval4b_ids.txt") <(sort "$A/g1_eval4_ids.txt") | wc -l)
echo "  与 e4 的 id 交集: $inter (必须为 0)" | tee -a "$LOG"
if [ "$rc" -eq 0 ] && [ "$have" -eq "$need" ] && [ "$inter" -eq 0 ]; then
  echo "E4B_BUILD_DONE $(date '+%F %T')" | tee -a "$LOG"
else
  echo "E4B_BUILD_FAILED rc=$rc have=$have need=$need inter=$inter $(date '+%F %T')" | tee -a "$LOG"
  exit 1
fi
