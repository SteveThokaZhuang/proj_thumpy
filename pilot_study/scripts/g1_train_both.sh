#!/usr/bin/env bash
# G1 消融: 顺序训练两臂 (own -> mixnorm), 同一 config / 同一数据管线.
# 单卡 24GB 放不下两个 8B 训练, 只能串行。
# 用法 (gpu04, srun --overlap 内): bash g1_train_both.sh
set -u
SCRIPTS=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts
D=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator
LOGS=$D/g1_logs
mkdir -p "$LOGS"

# 两臂唯一的差别是音频来源目录; id/答案/指令已由 ari_g1_check_arms.py 验证一致
declare -A RC
run_arm () {
  local arm=$1 out=$2 rc
  echo "########## $arm  ($(date '+%F %T')) ##########"
  bash "$SCRIPTS/g1_run.sh" "$arm" "$D/g1_sub/mixnorm" "$out"
  rc=$?          # 必须立刻接住: 下面 echo 里的 $(date) 会覆盖 $?
  RC[$arm]=$rc
  echo "########## $arm 结束 rc=$rc ($(date '+%F %T')) ##########"
}

run_arm own10   "$D/g1_own10_sft"
run_arm mixnorm "$D/g1_mixnorm_sft"

echo "=== 汇总 (以 adapter 文件为准, 不信退出码) ==="
fail=0
for arm in own10 mixnorm; do
  out="$D/g1_${arm}_sft/saves/adapter_model.safetensors"
  if [ -f "$out" ]; then
    echo "  [OK ] $arm  rc=${RC[$arm]}  $(stat -c%s "$out") B"
  else
    echo "  [FAIL] $arm  rc=${RC[$arm]}  无 adapter"
    fail=1
  fi
done
[ $fail -eq 0 ] && echo "ALL_DONE $(date '+%F %T')" || echo "ALL_FAILED $(date '+%F %T')"
exit $fail
