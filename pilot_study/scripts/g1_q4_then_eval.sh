#!/usr/bin/env bash
# 接在**正在跑的** Q3 训练之后: 补跑 Q4 -> 评 Q3+Q4。一条自驱动的链。
#
# ## 为什么要排队而不是现在就起
#
# gpu01 那张 4090 现在被 Q3 训练占着 (21.9/23.5 GB)。此刻起第二个进程
# 就是重演 2026-09-17 19:54 那次抢卡事故 —— 后面起的每个 run 都会在
# ~14 秒内 CUDA OOM。**所以本脚本第一件事是等, 不是跑。**
#
# ## 等待判据 (在 gpu01 上判断, 不看别的节点)
#
#   1. `nvidia-smi --query-compute-apps` 为空 —— GPU 真的空了
#   2. 14 个 `g1_*_ds*_sft/saves/adapter_model.safetensors` 齐 —— 训练循环
#      真的走完了, 而不是崩在半路 (只看进程在不在会把"崩了"也当成"完了")
#
# ## §3.2 的教训编进了脚本
#
# 上一次 Q4 之所以丢失, 是因为活下来的进程是**修复前**启动的。
# 所以这里在跑 Q4 之前把 `g1_text_only_train.py` 的 mtime 打出来,
# 让「本脚本启动时刻必须晚于它」这件事**可见**, 而不是靠记性。
#
# 用法 (kimi / gpu01, srun --overlap 内, conda funaudiochat):
#   bash scripts/g1_q4_then_eval.sh
set -uo pipefail
A=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator
S=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts
LOG=$A/g1_logs/q4_chain.log
mkdir -p "$A/g1_logs"
cd /share/workspace3/zhuangruicen/proj-thumpy/pilot_study

log () { echo "$*" | tee -a "$LOG"; }
n_adapter () { ls $A/g1_*_ds*_sft/saves/adapter_model.safetensors 2>/dev/null | wc -l; }
n_gpu_app () { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l; }

log ""
log "############################################################"
log "=== Q4 补跑 + 评估链启动 $(date '+%F %T') ==="
log "############################################################"

# 先确认这个步骤真的看得见 GPU —— 否则下面的等待判据会把
# "nvidia-smi 失败" 误读成 "GPU 空了", 然后一头撞进 OOM
nvidia-smi -L >/dev/null 2>&1 || { log "❌ 本步骤看不到 GPU (nvidia-smi 失败), 拒绝继续"; exit 1; }

# ───────────────────── 阶段 0: 等 GPU 空 + Q3 跑完 ─────────────────────
DEADLINE=$(( $(date +%s) + 6 * 3600 ))
waited=0
while :; do
  na=$(n_gpu_app); nad=$(n_adapter)
  if [ "$na" -eq 0 ] && [ "$nad" -ge 14 ]; then
    log "✅ 条件满足: GPU 空, $nad 个 Q3 adapter 齐 ($(date '+%F %T'), 等了 ${waited} 分钟)"
    break
  fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    log "❌ 等待超时 (6 小时): GPU 上 $na 个进程, $nad 个 adapter。"
    log "   不硬上 —— 硬上就是抢卡 OOM。请人工确认 Q3 那棵树的状态。"
    exit 1
  fi
  [ $(( waited % 15 )) -eq 0 ] && log "  等待中... GPU 进程=$na, Q3 adapter=$nad/14"
  sleep 60; waited=$(( waited + 1 ))
done
sleep 30   # 让刚退出的进程把显存真正还回去

# ───────────────────── 阶段 1: 补跑 Q4 ─────────────────────
log ""
log "=== 阶段 1: 补跑 Q4 (2 run, ~27 min) $(date '+%F %T') ==="
# §3.2 的教训: 把源码 mtime 与本次启动时刻都打出来, 让「读的是修复后的版本」可见
log "  g1_text_only_train.py mtime = $(ls -l --time-style=+%F_%T "$S/g1_text_only_train.py" | awk '{print $6}')"
log "  本阶段启动时刻             = $(date '+%F_%T')   <- 必须更晚"
log "  且这是**新起的进程**, 不存在'旧进程还活着'的问题"

G1_ONLY=q4 bash "$S/g1_q3q4_train.sh" 2>&1 | tee -a "$LOG"
rc4=${PIPESTATUS[0]}
log "  Q4 训练 rc=$rc4"

# 以产物为准, 不信退出码 (本项目通用纪律)
q4_ok=1
for arm in own10 mixnorm; do
  f="$A/g1_${arm}_q4_sft/saves/adapter_model.safetensors"
  if [ -f "$f" ]; then log "  ✅ $arm Q4 adapter 存在"; else log "  ❌ $arm Q4 adapter 缺失"; q4_ok=0; fi
done
[ "$q4_ok" -eq 0 ] && { log "❌ Q4 没产出, 不再往下跑评估 (评了也没有对应 adapter)。"; exit 1; }

# ── Q4 的核心问题: loss 到底能不能读了 ──
log ""
log "=== Q4 关键指标: trainer_log.jsonl 的 loss 是否变成有限值 ==="
log "  (修复前: 全项目 23 个 run 每一行 loss 都是 0.0 —— 见 §1.1)"
for arm in own10 mixnorm; do
  l="$A/g1_${arm}_q4_sft/saves/trainer_log.jsonl"
  if [ -f "$l" ]; then
    # 注意 `|| echo 0`: grep -c 无匹配时会**同时**打印 0 并退出 1,
    # 不加 `|| true` 的话命令替换会变成 "0\n0" 两行
    tot=$(grep -c '"loss"' "$l" 2>/dev/null || true)
    z=$(grep -o '"loss": 0\.0' "$l" 2>/dev/null | wc -l)
    log "  $arm: 共 $tot 行, 其中 loss 恰为 0.0 的有 $z 行"
    log "       $(grep -o '"loss": [0-9.]*' "$l" | head -5 | tr '\n' ' ')"
  else
    log "  $arm: ⚠️ 没找到 $l"
  fi
done

# ───────────────────── 阶段 2: 评 Q3 + Q4 ─────────────────────
log ""
log "=== 阶段 2: 评 Q3(14) + Q4(2), 旧 300 + --require-inside (~2.2h) $(date '+%F %T') ==="
bash "$S/g1_q3_eval.sh" both 2>&1 | tee -a "$LOG"
rce=${PIPESTATUS[0]}
log "  评估 rc=$rce"

log ""
log "############################################################"
if [ "$rc4" -eq 0 ] && [ "$rce" -eq 0 ]; then
  log "CHAIN_DONE $(date '+%F %T') —— Q4 已补跑, 16 个评估 json 已产出"
  log "   下一步: python scripts/g1_q3_analyze.py --group both"
else
  log "CHAIN_FAILED rc4=$rc4 rce=$rce $(date '+%F %T')"
fi
log "############################################################"
[ "$rc4" -eq 0 ] && [ "$rce" -eq 0 ]
