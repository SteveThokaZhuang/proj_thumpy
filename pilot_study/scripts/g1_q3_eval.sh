#!/usr/bin/env bash
# Q3/Q4 产物的评估驱动 —— 把 14 个 data_seed adapter (+ 2 个 Q4 adapter) 评到
# **与 §7.2 同一套评估集**上, 这样新旧方差才可比。
#
# ## 为什么必须用旧 300 的 `filt` 口径, 而不是 E2
#
# Q3 要问的是「纯 shuffle 分量有多大」, 方法是把 `Var(data_seed | seed=42)`
# 与既有的 `Var(seed)` 联立。既有那个方差是在**旧 300 + `--require-inside`**
# 上算的 (`g1_eval_<arm>[_s<seed>]_filt.json`, ΔF1=+0.0340, SE_seed=0.0117)。
# 换评估集会让两个方差落在不同尺度上, 联立就失去意义。
#
# 不用 E2 还有个成本原因: E2 是 1600 块, 按实测 1.666 s/块 要 44 min/adapter,
# 14 个就是 10 GPU 小时。旧 300 只要 8.3 min/adapter。
#
# ## 为什么这几个 adapter 没见过评估集
#
# 评估集是固定的旧 300 块; Q3/Q4 的 adapter 只是**换种子重训**,
# 训练数据 (g1-own10 / g1-mixnorm) 没变, 所以不会引入新的泄漏。
#
# 产出:
#   g1_eval_<arm>_ds<ds>_filt.json   × 14   (Q3)
#   g1_eval_<arm>_q4_filt.json       ×  2   (Q4 中性校验, 与 g1_eval_<arm>_filt.json 比)
#
# 用法 (kimi / gpu01, srun --overlap 内, **conda funaudiochat**):
#   bash scripts/g1_q3_eval.sh q3     # 只评 Q3 的 14 个 (~1.9h)
#   bash scripts/g1_q3_eval.sh q4     # 只评 Q4 的 2 个 (~17min)
#   bash scripts/g1_q3_eval.sh both
set -uo pipefail
A=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator
S=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts
LOG=$A/g1_logs/q3_eval.log
mkdir -p "$A/g1_logs"
cd "$S"

STAGE=${1:-both}
DS_LIST=${G1_DS_LIST:-"42 1234 7 2024 3407 31337 55555"}
ARMS=${G1_ARMS:-"own10 mixnorm"}

# 旧 300 块的 npz 与 manifest (与 g1_e2_eval.sh 的 filt 分支完全一致)
npz_old () { case $1 in own10) echo "$A/g1_own_eval/own";; mixnorm) echo "$A/g1_sub/mixnorm";; esac; }
MANI=$A/g1_sub/mixnorm
N_CHUNKS=$(wc -l < "$A/g1_eval_ids.txt")

echo "=========== Q3 EVAL stage=$STAGE $(date '+%F %T') ===========" | tee -a "$LOG"
echo "评估集: 旧 300 块 + --require-inside (n_chunks=$N_CHUNKS)" | tee -a "$LOG"

run_one () {   # adapter_dir tag
  local dir=$1 tag=$2 npzdir=$3
  local out="$A/g1_eval_${tag}.json"
  if [ -f "$out" ]; then
    echo "  [跳过] $tag 已存在" | tee -a "$LOG"; return 0
  fi
  if [ ! -f "$dir/adapter_model.safetensors" ]; then
    echo "  [跳过] $tag —— adapter 不存在 ($dir)" | tee -a "$LOG"; return 1
  fi
  echo "########## $tag  ($(date '+%F %T')) ##########" | tee -a "$LOG"
  python ari_g1_eval.py --npz-dir "$npzdir" \
      --manifest-from "$MANI" \
      --lora "$dir" --tag "$tag" --require-inside 2>&1 \
    | grep -vE "Loading checkpoint|it/s\]|^ *$" | tee -a "$LOG"
  return ${PIPESTATUS[0]}
}

rc=0
if [ "$STAGE" = "q4" ] || [ "$STAGE" = "both" ]; then
  for arm in $ARMS; do
    run_one "$A/g1_${arm}_q4_sft/saves" "${arm}_q4_filt" "$(npz_old "$arm")" \
      || { echo "  !! ${arm}_q4_filt 失败" | tee -a "$LOG"; rc=1; }
  done
fi

if [ "$STAGE" = "q3" ] || [ "$STAGE" = "both" ]; then
  for ds in $DS_LIST; do
    for arm in $ARMS; do
      run_one "$A/g1_${arm}_ds${ds}_sft/saves" "${arm}_ds${ds}_filt" "$(npz_old "$arm")" \
        || { echo "  !! ${arm}_ds${ds}_filt 失败" | tee -a "$LOG"; rc=1; }
    done
  done
fi

# ── 以**内容**为准核对 (与 g1_e2_eval.sh 同一条纪律) ───────────────────────
# 文件存在不等于评对了: 曾有过 `--n-eval` 默认 300 的截断 bug, 会让扩集
# 静默退化成"只评了前 300 块"而 json 照样生成。这里额外核对 n_gt:
# 同一套评估集下所有 adapter 的 n_gt 必须**完全相同** (20), 对不上说明
# manifest / ids-file 用错了 —— 那会让 ΔF1 比的是两套不同的真值。
echo "" | tee -a "$LOG"
echo "=== 内容核对 (n_chunks 应=$N_CHUNKS, n_gt 应一致) ===" | tee -a "$LOG"
py_check () {
  python - "$1" "$N_CHUNKS" <<'PY'
import json, sys
p, want = sys.argv[1], int(sys.argv[2])
d = json.load(open(p))
e = next(iter(d.values()))
print(f"  {p.split('/')[-1]:<34} n_chunks={e['n_chunks']:<6} n_gt={e['n_gt']}")
if e["n_chunks"] != want:
    sys.exit(1)
PY
}
tags=""
[ "$STAGE" = "q4" ] || [ "$STAGE" = "both" ] && for arm in $ARMS; do tags="$tags ${arm}_q4_filt"; done
[ "$STAGE" = "q3" ] || [ "$STAGE" = "both" ] && for ds in $DS_LIST; do for arm in $ARMS; do tags="$tags ${arm}_ds${ds}_filt"; done; done

for t in $tags; do
  f="$A/g1_eval_${t}.json"
  [ -f "$f" ] || { echo "  缺 $f" | tee -a "$LOG"; rc=1; continue; }
  py_check "$f" || { echo "  !! $f 块数不对" | tee -a "$LOG"; rc=1; }
done

# n_gt 一致性: 同一评估集下必须只有一个取值
ngt=$(for t in $tags; do
        f="$A/g1_eval_${t}.json"; [ -f "$f" ] || continue
        python -c "import json,sys;print(next(iter(json.load(open(sys.argv[1])).values()))['n_gt'])" "$f"
      done | sort -u | tr '\n' ' ')
echo "  全部 n_gt 取值集合: {$ngt}" | tee -a "$LOG"
[ "$(echo $ngt | wc -w)" -gt 1 ] && { echo "  !! n_gt 不一致 —— 评估集用错了" | tee -a "$LOG"; rc=1; }

n=$(ls $A/g1_eval_*_ds*_filt.json $A/g1_eval_*_q4_filt.json 2>/dev/null | wc -l)
[ $rc -eq 0 ] && echo "Q3EVAL_DONE 共 $n 个 json $(date '+%F %T')" | tee -a "$LOG" \
              || { echo "Q3EVAL_FAILED $(date '+%F %T')" | tee -a "$LOG"; exit 1; }
