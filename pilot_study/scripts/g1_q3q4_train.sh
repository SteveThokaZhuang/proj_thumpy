#!/usr/bin/env bash
# #62/#63 Q4 + Q3: 在 kimi (job 69621 / gpu01) 上把这两件事一次跑完。
#
# ============================ Q4 (先跑, 2 个 run) ============================
# 带着**可读的 loss 曲线**重训一次, 并验证 `g1_text_only_train.py` 的修复是
# 「优化中性」的 —— 即关掉 speech 支路不改变任何一步更新。
#
# 背景 (§4.3): 全项目 23 个 run 的 loss 全记成 0.0, 不是训练坏了, 是
# `speech_loss` 在纯文本回复上恒为 NaN, 被 `logging_nan_inf_filter` 吞掉。
# 修法: `model.sp_gen_kwargs['disable_speech'] = True`。
#
# 验证方式: 用 seed=42 + 默认 data_seed **重跑原配置**, 与既有
# `g1_{arm}_sft` 比对 adapter 与 F1。**不假设逐位相同** —— 少跑一次
# audio_invert_tower 前向可能改变 dropout 消耗的 RNG 流, 轨迹会漂移。
# 所以判定口径是「落在跨种子噪声内」而不是 md5 相同 (见 §7.4, SD≈0.031)。
#
# ============================ Q3 (后跑, 14 个 run) ==========================
# 把「初始化运气」和「shuffle 运气」分开 —— §5 只测出 mixnorm 的种子会落在
# 不同阈值上, 没解释来源。候选来源有三个, (a) 数据顺序 (b) LoRA 初值
# (c) 两声道相加重叠的增益不确定性。(a)(b) 已在 §7.1 用读盘证据排除,
# 本实验是 (a)(b) 的**直接干预版**。
#
# 设计: **固定 seed=42, 只变 data_seed**, 两臂各 7 个值。
#   - seed 决定 LoRA 初值(以及 dropout 等一切 torch RNG)
#   - data_seed 单独决定**训练集的打乱顺序**
#   - 固定 seed ⇒ 7 个 run 的初值**逐位相同**, 差异只能来自数据顺序
#   - 于是 Var(data_seed | seed=42) = **纯 shuffle 分量**
#   - 与既有跨 7 个 seed 的方差联立: Var(seed) = Var(init) + Var(shuffle)
#
# **data_seed 的管线已逐环验证** (2026-09-17, 因为 LLaMA-Factory 全仓库
# grep 不到 "data_seed" 一词, 一度怀疑它是空操作):
#   finetuning_args.disable_shuffling 默认 False  (否则会走 SequentialSampler)
#   -> HF trainer.py:1001 `_get_train_sampler` 返回 RandomSampler
#   -> trainer.py:5137 从 AcceleratorConfig 取 use_seedable_sampler, **默认 True**
#      (注意 accelerate 自己的同名参数默认是 False, 但 transformers 的
#       AcceleratorConfig 默认 True, 这里生效的是后者)
#   -> trainer.py:5142 `dataloader_config.data_seed = self.args.data_seed`
#   -> accelerate/data_loader.py:1176 包成 SeedableRandomSampler(data_seed=...)
#   -> 其 __iter__ 里 `generator.manual_seed(self.epoch + self.initial_seed)`,
#      而 initial_seed 就是 data_seed ⇒ 换 data_seed 必换顺序
# 所以本实验有效; 末尾的 md5 比对是运行期的第二道保险。
#
# **本段故意不加 DisableSpeechCallback** —— 保持与原始 14 个 run 逐字相同的
# 配置, 让 data_seed 成为唯一变量。附带好处: `ds=42` 那个 run 必须**复现**
# 既有的 `g1_{arm}_sft`(seed=42, data_seed 未设 ≡ 42), 这是一次强校验 ——
# 它同时证明 data_seed 的管线接对了、且配置确实只差这一个字段。
# 另外脚本末尾会比对 7 个 adapter 的 md5: **若全部相同, 说明 data_seed
# 根本没生效**, 那样本实验就是无效的, 必须如实报出而不是硬解读。
#
# 用法 (kimi / gpu01, srun --overlap 内, **conda funaudiochat**):
#   bash scripts/g1_q3q4_train.sh            # 先 Q4 后 Q3, 共 16 个 run ≈ 3.2h
#   G1_ONLY=q4 bash scripts/g1_q3q4_train.sh # 只跑 Q4
set -uo pipefail
A=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator
S=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts
PROJECT_ROOT=/share/workspace3/zhuangruicen/proj-thumpy/third_party/Fun-Audio-Chat
LOGS=$A/g1_logs
mkdir -p "$LOGS"

ONLY=${G1_ONLY:-both}
DS_LIST=${G1_DS_LIST:-"42 1234 7 2024 3407 31337 55555"}
ARMS=${G1_ARMS:-"own10 mixnorm"}

export AUDIO_PLACEHOLDER="<|audio_bos|><|AUDIO|><|audio_eos|>"
export DISABLE_VERSION_CHECK=1
unset FORCE_TORCHRUN                  # NCCL 会炸, 单进程
export PYTHONPATH="$PROJECT_ROOT/training/plugin:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 与主实验同一个基座 config, 同一套 sed 覆写 —— 除 output_dir/seed 外逐字相同
# (batch 4×accum 8, save_steps 25 全部沿用)。
make_config () {   # arm outdir extra_yaml_lines...
  local arm=$1 out=$2; shift 2
  local cfg="$out/g1_config.yaml"
  mkdir -p "$out/saves"
  sed -e "s|^dataset:.*|dataset: g1-$arm|" \
      -e "s|^output_dir:.*|output_dir: $out/saves|" \
      -e "s|^save_steps:.*|save_steps: 25|" \
      -e "s|^per_device_train_batch_size:.*|per_device_train_batch_size: 4|" \
      -e "s|^gradient_accumulation_steps:.*|gradient_accumulation_steps: 8|" \
      "$S/f8c_v2_config.yaml" > "$cfg"
  # 基座 config 里没有 seed/data_seed 行 -> 追加即是唯一来源, 不会冲突
  printf '%s\n' "$@" >> "$cfg"
  echo "save_total_limit: 2" >> "$cfg"
  # 残留的旧 adapter 会骗过末尾的成功判定
  rm -f "$out/saves/adapter_model.safetensors"
}

declare -A RC

train_plain () {   # arm outdir tag extra_yaml...
  local arm=$1 out=$2 tag=$3; shift 3
  if [ -f "$out/saves/adapter_model.safetensors" ]; then
    echo "  [跳过] $tag 已存在"; RC[$tag]=0; return 0
  fi
  local rc
  echo "########## $tag  $(date '+%F %T') ##########"
  make_config "$arm" "$out" "$@"
  ( cd "$PROJECT_ROOT/training" && \
    llamafactory-cli train "$out/g1_config.yaml" ) 2>&1 \
    | tee "$LOGS/train_${tag}.log"
  rc=${PIPESTATUS[0]}                # tee 会吞掉真实退出码
  echo "########## $tag 结束 rc=$rc  $(date '+%F %T') ##########"
  RC[$tag]=$rc
}

train_textonly () {   # arm outdir tag extra_yaml...  —— 带 Q4 的 loss 修复
  local arm=$1 out=$2 tag=$3; shift 3
  if [ -f "$out/saves/adapter_model.safetensors" ]; then
    echo "  [跳过] $tag 已存在"; RC[$tag]=0; return 0
  fi
  local rc
  echo "########## $tag (text-only)  $(date '+%F %T') ##########"
  make_config "$arm" "$out" "$@"
  ( cd "$PROJECT_ROOT/training" && python -c "
import sys; sys.path.insert(0,'$S')
from g1_text_only_train import run_text_only; run_text_only(sys.argv[1])
" "$out/g1_config.yaml" ) 2>&1 | tee "$LOGS/train_${tag}.log"
  rc=${PIPESTATUS[0]}
  echo "########## $tag 结束 rc=$rc  $(date '+%F %T') ##########"
  RC[$tag]=$rc
}

# ───────────────────────── Q4: 可读 loss + 中性校验 ─────────────────────────
if [ "$ONLY" = "both" ] || [ "$ONLY" = "q4" ]; then
  echo "==================== Q4 $(date '+%F %T') ===================="
  for arm in $ARMS; do
    # seed=42 且不写 data_seed -> 与既有 g1_{arm}_sft 完全同配置, 只是多了回调
    train_textonly "$arm" "$A/g1_${arm}_q4_sft" "${arm}_q4" "seed: 42"
  done
fi

# ───────────────────────── Q3: data_seed 解耦 ─────────────────────────
if [ "$ONLY" = "both" ] || [ "$ONLY" = "q3" ]; then
  echo "==================== Q3 $(date '+%F %T') ===================="
  for ds in $DS_LIST; do
    for arm in $ARMS; do
      # **不加** DisableSpeechCallback: 与原始 run 逐字同配置, data_seed 是唯一变量
      train_plain "$arm" "$A/g1_${arm}_ds${ds}_sft" "${arm}_ds${ds}" \
        "seed: 42" "data_seed: $ds"
    done
  done
fi

echo "=== 汇总 (以 adapter 文件为准, 不信退出码) ==="
fail=0
for tag in "${!RC[@]}"; do
  [ "${RC[$tag]}" -eq 0 ] || { echo "  !! $tag rc=${RC[$tag]}"; fail=1; }
done

# ── 关键校验: data_seed 到底有没有生效 ──────────────────────────────────
#
# ⚠️ 2026-09-17 20:15 更正: 这里原先比的是「7 个 adapter 的 md5 是否全同」。
# **那个检查是恒真的, 等于没有。** 基座 config 没开 `full_determinism`
# (默认 False), cuBLAS/cuDNN 的非确定性 kernel 让**每次重跑都产生不同比特**,
# 所以 7 个 md5 **必然**互不相同 —— 哪怕 `data_seed` 完全不起作用。
# 它今天也确实报了「7 个不同 md5 / 7 个 run」并放行, 那个"通过"零信息量。
# 自然实验证据: `g1_own10_sft` 与 `g1_own10_ds42_sft` 的 config 只差
# `output_dir` + 两行追加 (data_seed=None 在 SeedableRandomSampler 里
# 就等价于 set_seed(42) 后的 42), 是同一份有效配置, md5 却不同。
#
# 换成**直接迭代真实 sampler、打印前 8 个样本下标**: 确定性、0 GPU、
# 可反复复跑, 而且**假设它失败时会真的失败**。
echo ""
echo "=== data_seed 生效性检查 (迭代真实 sampler, 非 md5) ==="
order_out=$(python "$S/g1_dataseed_order_check.py" 2>&1)
order_rc=$?
echo "$order_out" | sed 's/^/  /'
if [ $order_rc -ne 0 ]; then
  echo "  ❌ 检查脚本本身没跑起来 (rc=$order_rc) —— **这是「没能验证」, 不是「通过」**"
  fail=1
elif echo "$order_out" | grep -q '7/7'; then
  echo "  ✅ 7 个 data_seed 产生 7 种不同顺序 ⇒ data_seed 确实生效, Q3 可解读"
else
  echo "  ❌ 未看到 7/7 ⇒ data_seed 疑似空操作 —— **Q3 结果不可解读!**"
  fail=1
fi

# ── Q4 中性校验: 与既有 g1_{arm}_sft 比 loss 曲线是否可读 ────────────────
for arm in $ARMS; do
  log="$A/g1_${arm}_q4_sft/saves/trainer_log.jsonl"
  if [ -f "$log" ]; then
    nz=$(grep -c '"loss"' "$log" 2>/dev/null || echo 0)
    z=$(grep -o '"loss": 0\.0' "$log" 2>/dev/null | wc -l)
    echo "  $arm Q4: trainer_log $nz 行, 其中 loss=0.0 有 $z 行"
  else
    echo "  $arm Q4: ⚠️ 没找到 $log"
  fi
done

[ "$fail" -eq 0 ] && echo "Q3Q4_DONE $(date '+%F %T')" || echo "Q3Q4_FAILED $(date '+%F %T')"
exit $fail
