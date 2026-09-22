#!/usr/bin/env bash
# 评估多种子重复的两臂 —— 每臂仍用**各自的观测空间**音频, instruction 与主实验一致。
# 用法 (gpu04, srun --overlap 内, **conda funaudiochat** —— fd_analysis 没有
# transformers/torch, 在里面跑会 ModuleNotFoundError): G1_SEED=1234 bash g1_eval_seeds.sh
# 只评单臂: G1_SEED=7 G1_ARMS=own10 bash g1_eval_seeds.sh
#
# 容错: 早先这里是裸 `set -u`, 一臂崩掉脚本照样往下走并打印 SEEDEVAL_DONE ——
# 下游拿到"成功"字样却缺 json。改成逐个查退出码 + 末尾验产物。
set -uo pipefail
A=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator
S=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts
SEED=${G1_SEED:-1234}
cd "$S"

# 注意: ari_g1_prep.py 是平铺写入的, npz 在 {out}/{mode}/ 下 —— own 臂在
# g1_own_eval/own, mixnorm 臂在 g1_sub/mixnorm。manifest 一律用 mixnorm 那份
# (两臂 chunk id 完全相同, 用它只为取 session/t0 做真值窗口查询)。
eval_arm () {
  local arm=$1 npzdir=$2
  echo "########## $arm seed=$SEED  $(date '+%F %T') ##########"
  python ari_g1_eval.py \
    --npz-dir "$npzdir" \
    --manifest-from "$A/g1_sub/mixnorm" \
    --lora "$A/g1_${arm}_s${SEED}_sft/saves" \
    --tag "${arm}_s${SEED}"
  return $?
}

# 默认两臂; G1_ARMS=own10 时只评 own10 (与 g1_train_seeds.sh 同名同义)
ARMS=${G1_ARMS:-"own10 mixnorm"}
rc=0
for arm in $ARMS; do
  case $arm in
    own10)   eval_arm own10   "$A/g1_own_eval/own" || rc=1 ;;
    mixnorm) eval_arm mixnorm "$A/g1_sub/mixnorm"  || rc=1 ;;
    *) echo "未知臂: $arm"; rc=1 ;;
  esac
done

# 以产物为准, 不信退出码 (与训练侧同一条纪律)
for arm in $ARMS; do
  f="$A/g1_eval_${arm}_s${SEED}.json"
  [ -f "$f" ] || { echo "  缺 $f"; rc=1; }
done
[ $rc -eq 0 ] && echo "SEEDEVAL_DONE $(date '+%F %T')" || { echo "SEEDEVAL_FAILED $(date '+%F %T')"; exit 1; }
