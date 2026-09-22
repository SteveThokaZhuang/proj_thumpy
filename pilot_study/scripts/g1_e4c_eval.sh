#!/usr/bin/env bash
# 池间 SD 实验的评估 (docs/pilot_study/2026-09-18_pool_sd.md §1.2)
#   12 批 × 400 块 × 14 个 adapter, **0 训练**。既有 adapter 一个都不重训。
#
# ## 循环顺序是设计的一部分, 不是实现细节
# 外层 `(arm, seed)`、内层 `batch`。每个 adapter 连着跑完 12 批(≈1.9h),
# 于是**每一批都由 14 个 adapter 在 14 个不同时刻评过** ⇒ 评估时刻与批次身份
# **不混杂**。反过来(外层 batch)会让每一批落在一个连续的时间窗里,
# 那么"批间差"里就混着"什么时候评的" —— §7.4f 那个
# 「A/B 分组与节点+场次几乎完全重合 ⇒ 跑再多评估也补不上」就是这个形状。
#
# ## 分片 —— 分片键必须让 **arm 与节点不嵌套**
# 按 (seed, arm) 列表的下标奇偶分两片, 每片 7 个 adapter。
# ⚠️ 枚举顺序**不能**用 `for seed; do for arm; done` 配 `i%2`: 那样 arm 是内层,
# 奇偶就等于按 **arm** 分片 ⇒ own10 全在 gpu04、mixnorm 全在 gpu01,
# **arm 完全嵌套在节点里**, 于是 ΔF1 = own10 − mixnorm 会带上节点间的系统差。
# 这正是 §7.4f「A/B 分组与节点+场次几乎完全重合」的形状。
# 改为: 偶数号 seed 先 own10, 奇数号 seed 先 mixnorm ⇒ 每片都拿到两种 arm,
# 且同一个 seed 的两个 arm **同时**在两张卡上跑(配对更 contemporaneous)。
#   SHARD=0/2 bash scripts/g1_e4c_eval.sh
#
# 用法 (srun --overlap 内, **conda funaudiochat** —— fd_analysis 没有 transformers):
#   srun --overlap --jobid=<j> -n 1 -c 2 bash scripts/g1_e4c_eval.sh
set -uo pipefail
PY=/share/home/zhuangruicen/miniconda3/envs/funaudiochat/bin/python
A=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator
S=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts
SHARD=${SHARD:-0/1}
sk=${SHARD%%/*}; sn=${SHARD##*/}
LOG=$A/g1_logs/e4c_eval_${sk}of${sn}.log
mkdir -p "$A/g1_logs"
cd "$S"

SEEDS="42 1234 7 2024 3407 31337 55555"
NB=12                                   # 批数, 与 --split-batches 一致

echo "=========== E4C EVAL shard $sk/$sn  $(date '+%F %T') ===========" | tee -a "$LOG"

npz_e4c () { case $1 in own10) echo "$A/g1_e4c/own";; mixnorm) echo "$A/g1_e4c/mixnorm";; esac; }
mani () { echo "$A/g1_e4c/mixnorm"; }
suffix () { [ "$1" = "42" ] && echo "" || echo "_s$1"; }

# 建库是否齐 —— 不齐就别开跑, 否则会静默评出一个不完整的集 (项目踩过: 缺产物但印"成功")
need=$(grep -c . "$A/g1_eval4c_ids.txt")
have=$(ls "$A/g1_e4c/own"/*.npz 2>/dev/null | wc -l)
if [ "$have" -ne "$need" ]; then
  echo "❌ E4C 建库不全: $have/$need npz —— 先跑 ari_g1_prep_subset.py" | tee -a "$LOG"
  exit 1
fi
echo "建库齐全 $have/$need" | tee -a "$LOG"

# (seed, arm) 的执行序 —— 偶数号 seed 先 own10, 奇数号先 mixnorm。
# 一个文件, 数数/执行/判读都读它, 保证三处口径一致 (memory: 口径只有一处)。
PAIRF=$(mktemp)
si=0
for seed in $SEEDS; do
  if [ $((si % 2)) -eq 0 ]; then order="own10 mixnorm"; else order="mixnorm own10"; fi
  for arm in $order; do echo "$seed $arm" >> "$PAIRF"; done
  si=$((si+1))
done

n_mine=0; i=0
while read -r _s _a; do
  [ $((i % sn)) -eq "$sk" ] && n_mine=$((n_mine+1)); i=$((i+1))
done < "$PAIRF"
echo "本片负责 $n_mine 个 adapter × $NB 批 = $((n_mine*NB)) 个 run (共 $((i*NB)))" | tee -a "$LOG"
echo "本片 adapter: $(awk -v sk="$sk" -v sn="$sn" '{i=NR-1; if (i%sn==sk) printf "%s/%s ", $1, $2}' "$PAIRF")" | tee -a "$LOG"

run_one () {   # arm seed batch
  local arm=$1 seed=$2 b=$3
  local tag="${arm}$(suffix "$seed")_e4c$(printf '%02d' "$b")"
  local dir="$A/g1_${arm}$(suffix "$seed")_sft/saves"
  local ids="$A/g1_eval4c_ids_b$(printf '%02d' "$b")_ids.txt"
  local out="$A/g1_eval_${tag}.json"
  if [ -f "$out" ]; then
    echo "  [跳过] $tag 已存在" | tee -a "$LOG"; return 0
  fi
  echo "########## $tag  ($(date '+%F %T')) ##########" | tee -a "$LOG"
  "$PY" ari_g1_eval.py --npz-dir "$(npz_e4c "$arm")" \
      --manifest-from "$(mani)" \
      --lora "$dir" --tag "$tag" --require-inside \
      --ids-file "$ids" 2>&1 \
    | grep -vE "Loading checkpoint|it/s\]|^ *$" | tee -a "$LOG"
  return ${PIPESTATUS[0]}
}

# 先数一遍: 若某批的 id 文件不在, 现在就说, 别跑到一半才炸
for b in $(seq 0 $((NB-1))); do
  f="$A/g1_eval4c_ids_b$(printf '%02d' "$b")_ids.txt"
  [ -s "$f" ] || { echo "❌ 缺 $f" | tee -a "$LOG"; exit 1; }
done

fail=0; i=0
while read -r seed arm; do
  if [ $((i % sn)) -ne "$sk" ]; then i=$((i+1)); continue; fi
  i=$((i+1))
  for b in $(seq 0 $((NB-1))); do
    run_one "$arm" "$seed" "$b" \
      || { echo "  !! ${arm}$(suffix "$seed")_e4c$(printf '%02d' "$b") 失败" | tee -a "$LOG"; fail=1; }
  done
done < "$PAIRF"

# 判据按**本片**算 (分片时另一片的产物不在本片的判据里) —— 2026-09-19 修。
# 原来这里遍历全部 14 个 adapter 算全局完成度, 与本片无关 ⇒ 先跑完的那一片
# 必然打印 E4C_EVAL_INCOMPLETE (自己 84 个全绿也一样), 与上一行的注释矛盾。
# 2026-09-18 那一轮就真的这样报了一次: shard=1/2 161/168 fail=0。
# 现在只数本片的 adapter (复用上面那份 $PAIRF, 所以 rm 挪到后面)。
mine=0; all=0; i=0
while read -r seed arm; do
  if [ $((i % sn)) -ne "$sk" ]; then i=$((i+1)); continue; fi
  i=$((i+1))
  for b in $(seq 0 $((NB-1))); do
    all=$((all+1))
    [ -f "$A/g1_eval_${arm}$(suffix "$seed")_e4c$(printf '%02d' "$b").json" ] && mine=$((mine+1))
  done
done < "$PAIRF"
rm -f "$PAIRF"
echo "本片完成度 (仅本片 $n_mine 个 adapter): $mine/$all; fail=$fail  $(date '+%F %T')" | tee -a "$LOG"
if [ "$fail" -eq 0 ] && [ "$mine" -eq "$all" ]; then
  echo "E4C_EVAL_DONE shard=$sk/$sn $mine/$all $(date '+%F %T')" | tee -a "$LOG"
else
  echo "E4C_EVAL_INCOMPLETE shard=$sk/$sn $mine/$all fail=$fail $(date '+%F %T')" | tee -a "$LOG"
  exit 1
fi
