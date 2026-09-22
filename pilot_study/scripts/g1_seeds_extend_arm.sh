#!/usr/bin/env bash
# 通用版多种子扩展: 任意臂 + 任意种子列表, 可续跑。
#
# 与 g1_seeds_extend.sh 的关系: 那个是 own10 专用 (2026-09-13 跑第一批时写的,
# 当时还只关心 own10 臂)。本脚本把它参数化到任意臂, **不修改那个文件** ——
# 它当时正在被运行中的作业执行, 而 bash 是边读边执行的, 中途改文件有风险。
# 等那批跑完可以把两者合并。
#
# 为什么要通用版: 量化 ΔF1 = mixnorm − own10 的误差棒需要**两臂各自的 run-to-run
# 方差**。own10 那边已有多点, mixnorm 只有 2 个 (0.1400 / 0.1164), 拿 2 个点估的
# 方差去当误差棒太薄。补 mixnorm 种子:
#   G1_ARMS=mixnorm G1_MORE_SEEDS="7 2024 3407" bash g1_seeds_extend_arm.sh
#
# 用法 (gpu04, srun --overlap 内, **conda funaudiochat**):
#   G1_ARMS=own10   G1_MORE_SEEDS="1 2 3" bash g1_seeds_extend_arm.sh
#   G1_ARMS=mixnorm G1_MORE_SEEDS="7 2024 3407" bash g1_seeds_extend_arm.sh
set -uo pipefail
A=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator
S=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts
ARMS=${G1_ARMS:-own10}
SEEDS=${G1_MORE_SEEDS:-"7 2024 3407 31337 55555"}

case "$ARMS" in
  own10|mixnorm) ;;
  *) echo "未知臂: $ARMS (只支持 own10 / mixnorm)"; exit 2 ;;
esac

echo "=========== $ARMS 多种子扩展: seeds = $SEEDS  $(date '+%F %T') ==========="

train_fail=""; eval_fail=""
for seed in $SEEDS; do
  echo ""
  echo "########## $ARMS seed=$seed  $(date '+%F %T') ##########"
  adapter="$A/g1_${ARMS}_s${seed}_sft/saves/adapter_model.safetensors"
  evjson="$A/g1_eval_${ARMS}_s${seed}.json"

  if [ -f "$adapter" ]; then
    echo "  [跳过训练] 已有 adapter ($(stat -c%s "$adapter") B)"
  else
    G1_SEED=$seed G1_ARMS="$ARMS" bash "$S/g1_train_seeds.sh" \
      || echo "  ⚠ 训练脚本返回非 0"
    if [ -f "$adapter" ]; then
      echo "  [OK ] 训练完成 adapter $(stat -c%s "$adapter") B"
    else
      echo "  [FAIL] 训练未产出 adapter, 跳过本种子的评估"
      train_fail="$train_fail $seed"
      continue
    fi
  fi

  if [ -f "$evjson" ]; then
    echo "  [跳过评估] 已有 $evjson"
  else
    G1_SEED=$seed G1_ARMS="$ARMS" bash "$S/g1_eval_seeds.sh" \
      || echo "  ⚠ 评估脚本返回非 0"
  fi
  [ -f "$evjson" ] || { echo "  [FAIL] 缺 $evjson"; eval_fail="$eval_fail $seed"; }
done

echo ""
echo "=========== 汇总 (以产物为准)  $(date '+%F %T') ==========="
for seed in $SEEDS; do
  evjson="$A/g1_eval_${ARMS}_s${seed}.json"
  if [ -f "$evjson" ]; then
    line=$(python3 -c "
import json
d = json.load(open('$evjson'))
tag = '$ARMS' + '_s$seed'
v = d.get(tag)
print('F1=%.4f P=%.4f R=%.4f n_pred=%d' % (v['f1'], v['precision'], v['recall'], v['n_pred'])) if v else print('(json 里无该 tag)')
" 2>/dev/null) || line="(解析失败)"
    printf "  [OK ] %s seed=%-6s %s\n" "$ARMS" "$seed" "$line"
  else
    printf "  [FAIL] %s seed=%-6s 无 eval json\n" "$ARMS" "$seed"
  fi
done

if [ -z "$train_fail$eval_fail" ]; then
  echo "EXTEND_DONE $(date '+%F %T')"; exit 0
else
  echo "EXTEND_PARTIAL 训练失败:$train_fail  评估失败:$eval_fail  $(date '+%F %T')"; exit 1
fi
