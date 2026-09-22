#!/usr/bin/env bash
# G1 消融: 两臂各自在自己的观测空间里评估 (匹配训练 instruction).
#
# 对称性: 两臂用**同一个** eval 脚本、**同一批** 300 个冻结 chunk id、
# 同一套真值 (manifest 取自 mixnorm, 各臂 id/session/ch/t0 完全一致),
# 只有喂进去的音频不同 —— 即各臂训练时看到的那个观测空间。
#
# 注意 v2 基线 0.127 只能当参考: 它是用**不匹配**的 prompt 评的
# (训练带 "with confidence", 评估漏了), 本脚本两臂都用匹配 prompt。
set -u
A=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator
S=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts
cd "$S"

echo "########## own10 (own 声道观测空间) $(date '+%F %T') ##########"
# 注意: 构建脚本把 npz 写在 {out}/{mode}/ 下, 不是 {out}/ 下
python ari_g1_eval.py \
  --npz-dir "$A/g1_own_eval/own" \
  --manifest-from "$A/g1_sub/mixnorm" \
  --lora "$A/g1_own10_sft/saves" \
  --tag own10

echo "########## mixnorm (双声道混合观测空间) $(date '+%F %T') ##########"
python ari_g1_eval.py \
  --npz-dir "$A/g1_sub/mixnorm" \
  --manifest-from "$A/g1_sub/mixnorm" \
  --lora "$A/g1_mixnorm_sft/saves" \
  --tag mixnorm

echo "=== 两臂对比 ==="
python - "$A" <<'PY'
import json, sys
A = sys.argv[1]
try:
    o = json.load(open(f"{A}/g1_eval_own10.json"))["own10"]
    m = json.load(open(f"{A}/g1_eval_mixnorm.json"))["mixnorm"]
except FileNotFoundError as e:
    print("评估结果缺失:", e); sys.exit(1)
print(f"{'arm':10} {'F1':>7} {'P':>7} {'R':>7} {'n_pred':>7} {'n_gt':>6}")
for nm, r in (("own10", o), ("mixnorm", m)):
    print(f"{nm:10} {r['f1']:7.4f} {r['precision']:7.4f} {r['recall']:7.4f} "
          f"{r['n_pred']:7d} {r['n_gt']:6d}")
d = m["f1"] - o["f1"]
print(f"\nΔF1 (mixnorm - own10) = {d:+.4f}")
print("结论:", "观测对方**有帮助**" if d > 0.02 else
      ("观测对方**有害**" if d < -0.02 else "两臂**无实质差异**"))
PY
echo "EVAL_DONE $(date '+%F %T')"
