#!/usr/bin/env bash
# 隔离"prompt 不匹配"这一混淆.
#
# 背景: v2 **训练**用 "...with confidence...", 但 ari_f8c_evaluate.py 评估时
# 漏了 "with confidence" —— 已发表的 0.127 是在不匹配 prompt 下测的。
#
# 做法: 拿**产生 0.127 的那个 v2 adapter**, 在同一批 300 个冻结 chunk、同一份
# own 声道音频上, 只换 prompt 各评一次:
#   ① 旧 prompt (不带 with confidence) -> 应当复现 ≈0.127, 这是对评估管线的
#      端到端校验 (复现不了就说明我的 harness 和当初那版不等价, 结论不能采信)
#   ② 匹配 prompt -> 与 ① 的差就是纯 prompt 效应
#
# 注意: 只跑 v2 一个 adapter, 所以与 G1 两臂的差异里还混着训练集差异
# (f8c_v2 是 2100 行, g1_* 也是 2100 行, 共享 2095 块 —— 2026-09-14 核实;
#  此前这里写的"v2 2400"是错的, 2400 是 f8c_sft/v1 的行数), 不能直接归因。
# 另: §7.4 已证明这条 0.05 量级的差距本就是普通 run-to-run 波动, 见报告 §7。
set -u
A=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator
S=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts
OLD="When does the listener produce backchannels in this segment? List the times in seconds, or answer 'no backchannel'."
cd "$S"

echo "########## v2 adapter + 旧 prompt (应复现 0.127) $(date '+%F %T') ##########"
python ari_g1_eval.py \
  --npz-dir "$A/g1_own_eval/own" \
  --manifest-from "$A/g1_sub/mixnorm" \
  --lora "$A/f8c_v2_sft/saves" \
  --tag v2_oldprompt \
  --instruction "$OLD"

echo "########## v2 adapter + 匹配 prompt $(date '+%F %T') ##########"
python ari_g1_eval.py \
  --npz-dir "$A/g1_own_eval/own" \
  --manifest-from "$A/g1_sub/mixnorm" \
  --lora "$A/f8c_v2_sft/saves" \
  --tag v2_matchedprompt
echo "PROMPTCHECK_DONE $(date '+%F %T')"
