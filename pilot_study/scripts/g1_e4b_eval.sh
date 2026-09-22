#!/usr/bin/env bash
# k=4b 加块评估: 用**已有的 14 个 adapter** 只评**新增的 400 块**, 一行都不用重训。
#
# 为什么不重评 1200 块: 选块是前缀稳定的 (见 g1_e4b_build.sh 的说明), 既有
# 1200 块的逐块结果原样并入 1600 块, 重评只会把 §5.8b 挂着的数换成另一批随机数。
#
# 本脚本产出 g1_eval_<arm>[_s<seed>]_e4b.json (每份 400 块),
# 之后跑 `g1_e4x_merge.py` 合成 1600 块的 `_e4x`。
#
# 分片: 14 个 (arm, seed) 相互独立, 可拆到两张卡上跑。用环境变量
#   SHARD=0/2 bash scripts/g1_e4b_eval.sh     # 第 0 片 (共 2 片)
# 分片只影响**跑哪些**, 不影响每个 run 的内容 —— 所以两片合起来与单片等价。
#
# 用法 (srun --overlap 内, **conda funaudiochat** —— fd_analysis 没有 transformers):
#   bash scripts/g1_e4b_eval.sh
set -uo pipefail
# 解释器**写死**: 评估要 transformers/funaudiochat。旧脚本用裸 `python` 是隐式
# 依赖调用者的 activate —— 换个人换台机器就静默跑错环境 (甚至跑成 CPU)。
PY=/share/home/zhuangruicen/miniconda3/envs/funaudiochat/bin/python
A=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator
S=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts
SHARD=${SHARD:-0/1}
sk=${SHARD%%/*}; sn=${SHARD##*/}
LOG=$A/g1_logs/e4b_eval_${sk}of${sn}.log
mkdir -p "$A/g1_logs"
cd "$S"

SEEDS="42 1234 7 2024 3407 31337 55555"

echo "=========== E4B EVAL shard $sk/$sn  $(date '+%F %T') ===========" | tee -a "$LOG"

npz_e4b () { case $1 in own10) echo "$A/g1_e4b/own";; mixnorm) echo "$A/g1_e4b/mixnorm";; esac; }
# manifest 按评估集分开: 400 个新 id 不在任何旧 manifest 里, 用错会 KeyError
mani () { echo "$A/g1_e4b/mixnorm"; }
suffix () { [ "$1" = "42" ] && echo "" || echo "_s$1"; }

# 建库是否齐 —— 不齐就别开跑, 否则会静默评出一个不完整的集
need=$(grep -c . "$A/g1_eval4b_ids.txt")
have=$(ls "$A/g1_e4b/own"/*.npz 2>/dev/null | wc -l)
if [ "$have" -ne "$need" ]; then
  echo "❌ E4B 建库不全: $have/$need npz —— 先跑 g1_e4b_build.sh" | tee -a "$LOG"
  exit 1
fi
echo "建库齐全 $have/$need" | tee -a "$LOG"

# 每片自己数一遍要跑几个, 末尾的完成判据按片算 (单片时即 14)
n_mine=0; i=0
for seed in $SEEDS; do for arm in own10 mixnorm; do
  [ $((i % sn)) -eq "$sk" ] && n_mine=$((n_mine+1)); i=$((i+1))
done; done
echo "本片负责 $n_mine 个 run (共 $i)" | tee -a "$LOG"

run_one () {   # arm seed tag npzdir
  local arm=$1 seed=$2 tag=$3 npzdir=$4
  local dir="$A/g1_${arm}$(suffix "$seed")_sft/saves"
  local out="$A/g1_eval_${tag}.json"
  if [ -f "$out" ]; then
    echo "  [跳过] $tag 已存在" | tee -a "$LOG"; return 0
  fi
  echo "########## $tag  ($(date '+%F %T')) ##########" | tee -a "$LOG"
  "$PY" ari_g1_eval.py --npz-dir "$npzdir" \
      --manifest-from "$(mani)" \
      --lora "$dir" --tag "$tag" --require-inside \
      --ids-file "$A/g1_eval4b_ids.txt" 2>&1 \
    | grep -vE "Loading checkpoint|it/s\]|^ *$" | tee -a "$LOG"
  return ${PIPESTATUS[0]}
}

fail=0; i=0
for seed in $SEEDS; do
  for arm in own10 mixnorm; do
    if [ $((i % sn)) -ne "$sk" ]; then i=$((i+1)); continue; fi
    i=$((i+1))
    sfx=$(suffix "$seed")
    run_one "$arm" "$seed" "${arm}${sfx}_e4b" "$(npz_e4b "$arm")" \
      || { echo "  !! ${arm}${sfx}_e4b 失败" | tee -a "$LOG"; fail=1; }
  done
done

# 判据按**本片**算: 分片时 14 个全在另一片, 用 14 判会假报 INCOMPLETE
n=$(ls "$A"/g1_eval_*_e4b.json 2>/dev/null | wc -l)
if [ "$fail" -eq 0 ]; then
  echo "E4B_SHARD_DONE shard=$sk/$sn 本片 $n_mine 个; 全局已有 $n 份 _e4b $(date '+%F %T')" | tee -a "$LOG"
else
  echo "E4B_SHARD_INCOMPLETE shard=$sk/$sn fail=$fail 全局已有 $n 份 _e4b $(date '+%F %T')" | tee -a "$LOG"
  exit 1
fi
