#!/usr/bin/env bash
# G1 多种子扩展: 只跑 own10 臂, 把 F1 的 run-to-run 分布描厚 (2026-09-13 起)。
#
# 动机: 报告 §7.3 发现 f8c_v2 (0.1203) 比三条 own10 臂 (0.1727 / 0.1887 / 0.1805,
# 均值 0.181 / SD 0.008) 低 7.6 个标准差, 而那 0.052 的成因排查后仍未找到。
# 但三个点估 SD 太薄 —— 万一分布是重尾的, "0.1203 是异常值"这个判断本身就不成立,
# 而且"噪声下限 0.016"这个结论也得跟着修。所以再跑若干种子, 把分布描厚。
#
# 为什么只 own10: 待解释的那对是 v2 vs own10; mixnorm 的噪声已由 2 个种子刻画过,
# 再跑对本题没有增量。少跑一臂 = 省一半时间。
#
# 种子是**事先固定**的任意整数 (不是挑出来的), 与既有的 42 / 1234 不同即可。
# LLaMA-Factory 的 seed 同时决定权重初始化与数据打乱, 所以换种子即换一次独立重跑。
#
# 可续跑: 已有 adapter 的种子跳过训练, 已有 eval json 的种子跳过评估。
# 用法 (gpu04, srun --overlap 内, **conda funaudiochat**):
#   bash g1_seeds_extend.sh
#   G1_MORE_SEEDS="7 2024" bash g1_seeds_extend.sh     # 只跑指定种子
set -uo pipefail
A=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator
S=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts
SEEDS=${G1_MORE_SEEDS:-"7 2024 3407 31337 55555"}

echo "=========== own10 多种子扩展: seeds = $SEEDS  $(date '+%F %T') ==========="

train_fail=""; eval_fail=""
for seed in $SEEDS; do
  echo ""
  echo "########## seed=$seed  $(date '+%F %T') ##########"
  adapter="$A/g1_own10_s${seed}_sft/saves/adapter_model.safetensors"
  evjson="$A/g1_eval_own10_s${seed}.json"

  # --- 训练 (已有 adapter 就跳过) ---
  if [ -f "$adapter" ]; then
    echo "  [跳过训练] 已有 adapter ($(stat -c%s "$adapter") B)"
  else
    if G1_SEED=$seed G1_ARMS=own10 bash "$S/g1_train_seeds.sh"; then
      echo "  训练脚本返回 0"
    else
      echo "  ⚠ 训练脚本返回非 0"
    fi
    # 以产物为准 —— 退出码和日志字样都不可信 (见基础设施笔记第 3、12 条)
    if [ -f "$adapter" ]; then
      echo "  [OK ] 训练完成 adapter $(stat -c%s "$adapter") B"
    else
      echo "  [FAIL] 训练未产出 adapter, 跳过本种子的评估"
      train_fail="$train_fail $seed"
      continue
    fi
  fi

  # --- 评估 (已有 json 就跳过) ---
  if [ -f "$evjson" ]; then
    echo "  [跳过评估] 已有 $evjson"
  else
    if G1_SEED=$seed G1_ARMS=own10 bash "$S/g1_eval_seeds.sh"; then
      echo "  评估脚本返回 0"
    else
      echo "  ⚠ 评估脚本返回非 0"
    fi
  fi
  [ -f "$evjson" ] || { echo "  [FAIL] 缺 $evjson"; eval_fail="$eval_fail $seed"; }
done

# --- 汇总: 逐个种子核产物 ---
echo ""
echo "=========== 汇总 (以产物为准)  $(date '+%F %T') ==========="
for seed in $SEEDS; do
  evjson="$A/g1_eval_own10_s${seed}.json"
  if [ -f "$evjson" ]; then
    line=$(python3 -c "
import json
v = json.load(open('$evjson'))['own10_s$seed']
print('F1=%.4f P=%.4f R=%.4f n_pred=%d' % (v['f1'], v['precision'], v['recall'], v['n_pred']))
" 2>/dev/null) || line="(解析失败)"
    printf "  [OK ] seed=%-6s %s\n" "$seed" "$line"
  else
    printf "  [FAIL] seed=%-6s 无 eval json\n" "$seed"
  fi
done

if [ -z "$train_fail$eval_fail" ]; then
  echo "EXTEND_DONE $(date '+%F %T')"
  exit 0
else
  echo "EXTEND_PARTIAL 训练失败:$train_fail  评估失败:$eval_fail  $(date '+%F %T')"
  exit 1
fi
