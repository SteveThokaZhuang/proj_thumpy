#!/usr/bin/env bash
# #64 k=4 扩集评估: 用**已有的 14 个 adapter** 评第 4 套评估集, 一行都不用重训。
#
# 为什么是 k=4 而不是重训: 要判的是 §7.2 那个 ΔF1 = +0.0340 站不站得住。
# 卡在功效上的是**块抽样**那一项方差 (SE_chunk = 0.0198, 比 SE_seed = 0.0117
# 还大), 而它随评估集增大而缩小, 不需要新模型。
#
# 预测 (scripts/g1_eval4_predict.py, 用 filt 集拟合出 SE_chunk ∝ k^-0.672):
#   1200 块 / 43 事件 -> SE_chunk 0.0052~0.0078, 合并 SE 0.0128~0.0141
#   -> 预期 t = 2.42 ~ 2.66 (天花板 2.91)
#   对照现状: SE = 0.0230, t = 1.48
# **这是预测不是结论** —— 跑完要重新报, 尤其要复核流行率与 F1 量级。
#
# 与 E2 的差别: 本集是**自然流行率** (3.58%), 不是事件密集集。多报要吃 FP,
# 所以 F1 量级会显著低于 E2, 两者不可直接比大小。
#
# 产出 (每 arm×seed 一个 json): g1_eval_<arm>[_s<seed>]_e4.json
#
# 用法 (kimi2 / gpu04, srun --overlap 内, **conda funaudiochat** ——
# fd_analysis 没有 transformers):
#   bash scripts/g1_e4_eval.sh
set -uo pipefail
A=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator
S=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts
LOG=$A/g1_logs/e4_eval.log
mkdir -p "$A/g1_logs"
cd "$S"

STAGE=${1:-e4}
SEEDS="42 1234 7 2024 3407 31337 55555"

echo "=========== E4 EVAL $(date '+%F %T') ===========" | tee -a "$LOG"

npz_e4 () { case $1 in own10) echo "$A/g1_e4/own";; mixnorm) echo "$A/g1_e4/mixnorm";; esac; }
# manifest 必须**按评估集分开**: 1200 个新 id 不在任何旧 manifest 里, 用错会
# KeyError (bc_windows 要查 m["session"/"ch"/"t0"])。两个臂的 manifest 对同一批
# 块是一致的, 所以臂间可互换。
mani () { echo "$A/g1_e4/mixnorm"; }
# seed 42 的产物没有 _s42 后缀 (与既有命名一致)
suffix () { [ "$1" = "42" ] && echo "" || echo "_s$1"; }

# 建库是否齐 —— 不齐就别开跑, 否则会静默评出一个不完整的集
need=$(wc -l < "$A/g1_eval4_ids.txt")
have=$(ls "$A/g1_e4/own"/*.npz 2>/dev/null | wc -l)
if [ "$have" -ne "$need" ]; then
  echo "❌ E4 建库不全: $have/$need npz —— 先跑 g1_e4_build.sh" | tee -a "$LOG"
  exit 1
fi
echo "建库齐全 $have/$need" | tee -a "$LOG"

run_one () {   # arm seed tag npzdir extra_args...
  local arm=$1 seed=$2 tag=$3 npzdir=$4; shift 4
  local dir="$A/g1_${arm}$(suffix "$seed")_sft/saves"
  local out="$A/g1_eval_${tag}.json"
  if [ -f "$out" ]; then
    echo "  [跳过] $tag 已存在" | tee -a "$LOG"; return 0
  fi
  echo "########## $tag  ($(date '+%F %T')) ##########" | tee -a "$LOG"
  python ari_g1_eval.py --npz-dir "$npzdir" \
      --manifest-from "$(mani)" \
      --lora "$dir" --tag "$tag" --require-inside \
      --ids-file "$A/g1_eval4_ids.txt" "$@" 2>&1 \
    | grep -vE "Loading checkpoint|it/s\]|^ *$" | tee -a "$LOG"
  return ${PIPESTATUS[0]}
}

fail=0
for seed in $SEEDS; do
  for arm in own10 mixnorm; do
    sfx=$(suffix "$seed")
    run_one "$arm" "$seed" "${arm}${sfx}_e4" "$(npz_e4 "$arm")" \
      || { echo "  !! ${arm}${sfx}_e4 失败" | tee -a "$LOG"; fail=1; }
  done
done

n=$(ls "$A"/g1_eval_*_e4.json 2>/dev/null | wc -l)
if [ "$fail" -eq 0 ] && [ "$n" -eq 14 ]; then
  echo "E4_EVAL_DONE $n/14 $(date '+%F %T')" | tee -a "$LOG"
else
  echo "E4_EVAL_INCOMPLETE $n/14 fail=$fail $(date '+%F %T')" | tee -a "$LOG"
  exit 1
fi
