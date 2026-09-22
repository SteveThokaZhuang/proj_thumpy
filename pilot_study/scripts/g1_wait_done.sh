#!/usr/bin/env bash
# 等 Q3 旧集评估 (14) 与 k=4 评估 (14) 都齐活, 然后退出。
#
# 🔴 2026-09-18 00:0x 修过一次 —— 第一版是**结构性误报**:
#    它 grep 了整个 g1_logs/ 目录找 Traceback, 于是命中了
#    `train_both_crashed.log`(09-10)、`seeds_finish.log`(09-12)、
#    `confound_train.log`(09-13) 三个**别的实验的历史日志**。
#    那种写法**永远会失败**, 与「永远为真」的 md5 检查是同一族的两面。
#    判据统一为一句: **问它「在什么情况下会失败 / 会通过」。答不上来就等于没有。**
#
# 现在的设计:
#   1. **只看当前链自己的 3 个日志** —— 不扫目录, 不碰历史文件。
#      实测这三个文件**全文 0 处** Traceback, 所以扫全文是安全的,
#      而且比只看尾部**更灵敏** —— 能抓到"某个 eval 崩了但 for 循环继续跑"
#      这种中途失败 (那时产物会少一个, 尾部却是干净的)。
#   2. **产物停滞检测** —— 「安静」和「死了」必须能区分。每轮记产物数,
#      连续 STALL_MIN 分钟不增长才判停滞。(单个评估约 8~20 分钟。)
#   3. 覆盖成功、失败、停滞、超时四条路径, 一条都不静默。
set -uo pipefail
A=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator
LOGS=$A/g1_logs
# 只认这三个 —— 当前两条链自己写的
CUR_LOGS=("$LOGS/q4_chain.log" "$LOGS/q3_eval.log" "$LOGS/e4_eval.log")
FATAL_RE="Traceback \(most recent call last\)|CUDA out of memory|Q3Q4_FAILED|E4_FAILED|Killed$"
STALL_MIN=30
TIMEOUT_H=4

deadline=$(( $(date +%s) + TIMEOUT_H*3600 ))
last_n=-1; last_change=$(date +%s)

# 当前链的日志里**应当一处致命错误都没有** —— 有就是这次真的出事了
has_fatal () {
  [ -f "$1" ] || return 1
  grep -qE "$FATAL_RE" "$1" 2>/dev/null
}

while :; do
  nds=$(ls -1 $A/g1_eval_*_ds*_filt.json 2>/dev/null | wc -l)
  ne4=$(ls -1 $A/g1_eval_*_e4.json 2>/dev/null | wc -l)
  n=$(( nds + ne4 ))

  # 产物在动就重置停滞计时
  if [ "$n" -ne "$last_n" ]; then
    last_n=$n; last_change=$(date +%s)
  fi

  # ── 路径 1: 当前链的日志里出现了致命错误 ──
  for f in "${CUR_LOGS[@]}"; do
    if has_fatal "$f"; then
      echo "❌ 当前日志出现致命错误: $f"
      grep -nE "$FATAL_RE" "$f" | head -5
      exit 1
    fi
  done

  # ── 路径 2: 成功 ──
  if [ "$nds" -ge 14 ] && [ "$ne4" -ge 14 ]; then
    echo "✅ 全部到齐: ds_filt=$nds/14  e4=$ne4/14   $(date '+%F %T')"
    exit 0
  fi

  # ── 路径 3: 产物停滞 (安静 ≠ 死了) ──
  if [ $(( ($(date +%s) - last_change) / 60 )) -ge "$STALL_MIN" ]; then
    echo "⚠️ 停滞 ${STALL_MIN} 分钟无新产物: ds_filt=$nds/14  e4=$ne4/14"
    for f in "${CUR_LOGS[@]}"; do
      echo "  --- $(basename "$f") 末 3 行 ---"
      tail -n 3 "$f" 2>/dev/null | cut -c1-160
    done
    exit 3
  fi

  # ── 路径 4: 超时 ──
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "⏰ 超时 ${TIMEOUT_H}h: ds_filt=$nds/14  e4=$ne4/14   $(date '+%F %T')"
    exit 2
  fi

  sleep 60
done
