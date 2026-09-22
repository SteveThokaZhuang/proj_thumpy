#!/usr/bin/env bash
# #53 扩大评估集: 用**已有的 14 个 adapter** 评两套评估集, 一行都不用重训。
#
# 为什么不用重训: g1_eval2_ids.txt 是从 g1_ids.txt (2100 训练 + 300 旧评估)
# **之外**的窗口里选的, 所以这 14 个 adapter 从没见过它们。
#
# 两套评估集回答不同的问题, 都要报:
#   E2   (1600 块 / 1640 事件, 事件密集) —— **有功效**的那套, 用来判 ΔF1
#   旧 300 (26 事件, 自然流行率 8.3%) —— 与既有数字可比, 但欠功效
# 两套都加 --require-inside (横跨 chunk 边界的真值窗口不可答, 见 §7.5)。
#
# 产出 (每个 arm×seed 两个 json):
#   g1_eval_<arm>[_s<seed>]_e2.json     扩集
#   g1_eval_<arm>[_s<seed>]_filt.json   旧 300 + 边界过滤
#
# 用法 (gpu01, srun --overlap 内, **conda funaudiochat** —— fd_analysis 没有
# transformers): bash scripts/g1_e2_eval.sh <e2|filt|both>
set -uo pipefail
A=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator
S=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts
LOG=$A/g1_logs/e2_eval.log
mkdir -p "$A/g1_logs"
cd "$S"

STAGE=${1:-both}
SEEDS="42 1234 7 2024 3407 31337 55555"

echo "=========== E2 EVAL stage=$STAGE $(date '+%F %T') ===========" | tee -a "$LOG"

# arm -> 该臂的 npz 目录 (两种评估集各一套)
npz_e2 ()  { case $1 in own10) echo "$A/g1_e2/own";; mixnorm) echo "$A/g1_e2/mixnorm";; esac; }
npz_old () { case $1 in own10) echo "$A/g1_own_eval/own";; mixnorm) echo "$A/g1_sub/mixnorm";; esac; }
# manifest 必须**按评估集分开**: 扩集的 1600 个 id 不在旧 manifest 里, 旧 300 个
# id 也不在扩集 manifest 里, 用错会 KeyError (bc_windows 要查 m["session"/"ch"/"t0"])。
# 两个 manifest 的 id/session/ch/t0 对同一批块是一致的, 所以臂间仍可互换。
mani () { case $1 in e2) echo "$A/g1_e2/mixnorm";; filt) echo "$A/g1_sub/mixnorm";; esac; }
# seed 42 的产物没有 _s42 后缀 (与既有命名一致)
suffix () { [ "$1" = "42" ] && echo "" || echo "_s$1"; }

run_one () {   # arm seed stage out_tag npzdir extra_args...
  local arm=$1 seed=$2 st=$3 tag=$4 npzdir=$5; shift 5
  local dir="$A/g1_${arm}$(suffix "$seed")_sft/saves"
  local out="$A/g1_eval_${tag}.json"
  if [ -f "$out" ]; then
    echo "  [跳过] $tag 已存在" | tee -a "$LOG"; return 0
  fi
  echo "########## $tag  ($(date '+%F %T')) ##########" | tee -a "$LOG"
  python ari_g1_eval.py --npz-dir "$npzdir" \
      --manifest-from "$(mani "$st")" \
      --lora "$dir" --tag "$tag" --require-inside "$@" 2>&1 \
    | grep -vE "Loading checkpoint|it/s\]|^ *$" | tee -a "$LOG"
  return ${PIPESTATUS[0]}
}

for seed in $SEEDS; do
  for arm in own10 mixnorm; do
    sfx=$(suffix "$seed")
    if [ "$STAGE" = "e2" ] || [ "$STAGE" = "both" ]; then
      run_one "$arm" "$seed" e2 "${arm}${sfx}_e2" "$(npz_e2 "$arm")" \
        --ids-file "$A/g1_eval2_ids.txt" || echo "  !! ${arm}${sfx}_e2 失败" | tee -a "$LOG"
    fi
    if [ "$STAGE" = "filt" ] || [ "$STAGE" = "both" ]; then
      run_one "$arm" "$seed" filt "${arm}${sfx}_filt" "$(npz_old "$arm")" \
        || echo "  !! ${arm}${sfx}_filt 失败" | tee -a "$LOG"
    fi
  done
done

# 以产物为准 (与训练侧同一条纪律: 不信退出码)
want=""; [ "$STAGE" = "e2" ] && want=e2; [ "$STAGE" = "filt" ] && want=filt
need_chunks=$(wc -l < "$A/g1_eval2_ids.txt")
rc=0
for seed in $SEEDS; do
  for arm in own10 mixnorm; do
    sfx=$(suffix "$seed")
    for w in $want; do
      f="$A/g1_eval_${arm}${sfx}_${w}.json"
      [ -f "$f" ] || { echo "  缺 $f" | tee -a "$LOG"; rc=1; }
    done
    # 文件存在还不够: --n-eval 曾有个默认 300 的截断 bug, 会让扩集静默退化成
    # "只评了前 300 块" 而 json 照样生成。所以按**内容**核对块数。
    if [ "$STAGE" = "e2" ] || [ "$STAGE" = "both" ]; then
      f="$A/g1_eval_${arm}${sfx}_e2.json"
      [ -f "$f" ] && python - "$f" "$need_chunks" <<'PY' || rc=1
import json, sys
d = json.load(open(sys.argv[1]))
n = next(iter(d.values()))["n_chunks"]
if n != int(sys.argv[2]):
    print(f"  !! {sys.argv[1]} 只评了 {n} 块, 应为 {sys.argv[2]}", flush=True)
    sys.exit(1)
PY
    fi
    [ -z "$want" ] && for w in e2 filt; do
      f="$A/g1_eval_${arm}${sfx}_${w}.json"
      [ -f "$f" ] || { echo "  缺 $f" | tee -a "$LOG"; rc=1; }
    done
  done
done
[ $rc -eq 0 ] && echo "E2_EVAL_DONE stage=$STAGE $(date '+%F %T')" | tee -a "$LOG" \
              || { echo "E2_EVAL_FAILED $(date '+%F %T')" | tee -a "$LOG"; exit 1; }
