#!/usr/bin/env bash
# C 方案的评估: 100 个**全新会话** × 4800 块 × 14 个 adapter, **0 训练**。
# 预登记: docs/pilot_study/2026-09-22_planC_prereg.md
#
# ## 与 g1_e4c_eval.sh 的三点差别
#   1. **不分批**。e4c 的 12 批是为了量"批间 SD"; C 只有一个集, 切批只会白增复杂度。
#      循环顺序仍是 **adapter 外层**, 于是每个 adapter 连续跑完 —— 但 C 的判据
#      ΔF1 = own10 − mixnorm 是**同种子内配对**, 而配对的两个 arm 在 $PAIRF 里相邻,
#      所以 arm 与"什么时候评的"不混杂。这是 e4c 那条注释保留下来的部分。
#   2. **分片可选**。只有一张卡时 SHARD=0/1。若第二张卡可用, 用 0/2 与 1/2 同时跑;
#      分片键是 (seed, arm) 下标奇偶 ⇒ 每片都拿到两种 arm, arm 不嵌套在节点里。
#   3. **不依赖 e4c 的 id 文件**。ids 换成 g1_c_ids.txt。
#
# 用法 (srun --overlap 内, **conda funaudiochat** —— fd_analysis 没有 transformers):
#   srun --overlap --jobid=<j> -n 1 -c 2 bash scripts/g1_c_eval.sh
#   SHARD=0/2 srun --overlap --jobid=<j2> -n 1 -c 2 bash scripts/g1_c_eval.sh
set -uo pipefail
PY=/share/home/zhuangruicen/miniconda3/envs/funaudiochat/bin/python
A=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator
S=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts
SHARD=${SHARD:-0/1}
sk=${SHARD%%/*}; sn=${SHARD##*/}
mkdir -p "$A/g1_logs"
LOG=$A/g1_logs/c_eval_${sk}of${sn}.log
cd "$S"

SEEDS="42 1234 7 2024 3407 31337 55555"

echo "=========== C EVAL shard $sk/$sn  $(date '+%F %T') ===========" | tee -a "$LOG"

npz_c () { case $1 in own10) echo "$A/g1_c/own";; mixnorm) echo "$A/g1_c/mixnorm";; esac; }
mani () { echo "$A/g1_c/mixnorm"; }
suffix () { [ "$1" = "42" ] && echo "" || echo "_s$1"; }

# --- 建库是否齐 —— 不齐就别开跑 (项目踩过: 缺产物但印"成功") ---
need=$(grep -c . "$A/g1_c_ids.txt")
have=$(ls "$A/g1_c/own"/*.npz 2>/dev/null | wc -l)
if [ "$have" -ne "$need" ]; then
  echo "❌ C 建库不全: $have/$need npz —— 先跑 ari_g1_prep_subset.py" | tee -a "$LOG"
  exit 1
fi
# npz 必须只有 audio 一个键 (--no-states): 若混进 states 说明建库口径不对
"$PY" - "$A/g1_c/own" <<'EOF' 2>&1 | tee -a "$LOG"
import glob, sys, numpy as np
fs = sorted(glob.glob(f"{sys.argv[1]}/*.npz"))
keys, short = set(), 0
for f in fs[:200]:
    d = np.load(f)
    keys |= set(d.files)
    short += (d["audio"].shape[0] != 160000)
print(f"抽样 200 个的键 = {sorted(keys)}; 非 10s 块 {short}")
assert keys == {"audio"}, f"npz 键不是只有 audio: {sorted(keys)}"
assert short == 0, f"{short} 个块不是 160000 采样"
print("建库口径自检 ✅")
EOF
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "❌ 建库自检失败" | tee -a "$LOG"; exit 1; }

# --- (seed, arm) 的执行序: 偶数号 seed 先 own10, 奇数号先 mixnorm ---
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
echo "本片负责 $n_mine 个 adapter × $need 块 (共 $((i)) 个 adapter)" | tee -a "$LOG"
echo "本片 adapter: $(awk -v sk="$sk" -v sn="$sn" '{i=NR-1; if (i%sn==sk) printf "%s/%s ", $1, $2}' "$PAIRF")" | tee -a "$LOG"

run_one () {   # arm seed
  local arm=$1 seed=$2
  local tag="${arm}$(suffix "$seed")_c"
  local dir="$A/g1_${arm}$(suffix "$seed")_sft/saves"
  local out="$A/g1_eval_${tag}.json"
  if [ -f "$out" ]; then
    echo "  [跳过] $tag 已存在" | tee -a "$LOG"; return 0
  fi
  local t0=$(date +%s)
  echo "########## $tag  ($(date '+%F %T')) ##########" | tee -a "$LOG"
  "$PY" ari_g1_eval.py --npz-dir "$(npz_c "$arm")" \
      --manifest-from "$(mani)" \
      --lora "$dir" --tag "$tag" --require-inside \
      --ids-file "$A/g1_c_ids.txt" 2>&1 \
    | grep -vE "Loading checkpoint|it/s\]|^ *$" | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  echo "  -> $tag 用时 $(( ($(date +%s) - t0) / 60 )) min  $(date '+%F %T')" | tee -a "$LOG"
  return $rc
}

fail=0; i=0
while read -r seed arm; do
  if [ $((i % sn)) -ne "$sk" ]; then i=$((i+1)); continue; fi
  i=$((i+1))
  run_one "$arm" "$seed" \
    || { echo "  !! ${arm}$(suffix "$seed")_c 失败" | tee -a "$LOG"; fail=1; }
done < "$PAIRF"
rm -f "$PAIRF"

# 完成度: 数**本方案**的产物 (文件名以 `_c.json` 收尾, 不会误收 e4c 的 `_e4cNN.json`)
nf=$(ls "$A"/g1_eval_own10*_c.json "$A"/g1_eval_mixnorm*_c.json 2>/dev/null | wc -l)
echo "完成度 (全部 adapter): $nf/14; fail=$fail  $(date '+%F %T')" | tee -a "$LOG"
if [ "$fail" -eq 0 ] && [ "$nf" -eq 14 ]; then
  echo "C_EVAL_DONE shard=$sk/$sn $nf/14 $(date '+%F %T')" | tee -a "$LOG"
else
  echo "C_EVAL_INCOMPLETE shard=$sk/$sn $nf/14 fail=$fail $(date '+%F %T')" | tee -a "$LOG"
  exit 1
fi
