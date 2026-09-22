#!/usr/bin/env bash
# 复现 f8c_v2: 用**逐字相同的数据集与配置**重训一次, 看 0.1203 是否可复现。
#
# 动机 (报告 §7.3): f8c_v2 (F1 0.1203) 比 own10 (0.1727/0.1887/0.1805) 低 7.6 个
# 标准差, 而那 0.052 排查后仍未找到原因。已经**逐项排除**的:
#   - 训练超参: 145 个字段全量对比 (ari_g1_diff_args.py), 除 save_steps /
#     save_total_limit / generation_max_length 外**无任何实质差异**, seed 均为 42;
#   - micro-batch (都是 4x8)、cutoff_len (恒 250 audio token, 空操作, 实证 Δ+0.008);
#   - 数据: 2100 行对 2100 行, 2,095 共享, 音频 md5 逐字节相同, audio 元数据
#     (含 token pad 字段) 全同, dataset_info.json 注册项结构全同;
#   - LoRA: adapter_config 集合全同, adapter 文件大小逐字节同 (265,962,288 B);
#   - 训练过程: trainer_state 都是 132 步 / epoch 2.0 / 无 resume, total_flos 差 0.2%;
#   - 评估: load_model 走 text_greedy=True (贪心, 确定性), 与已发表 0.127 的逐位
#     复现相符 —— 所以评估本身不是噪声源。
#
# 剩下的可能只有两种, 而它们指向下一步完全不同的动作:
#   (A) v2 那套数据/配置**确实**产出 0.12 -> 数据是原因, 去二分那 5 个块的差异;
#   (B) 重跑得到 0.17 附近 -> **原来那次 v2 是一个离群 run**, 与数据/配置无关,
#       那 0.052 就不能归给任何已记录的变量。
# 本脚本就是判据。注意用的是**原始 f8c-v2 数据集本身**, 不是 own10 的数据。
#
# 只改 output_dir (写到新目录, **绝不覆盖原始 f8c_v2_sft**), 其余逐字沿用
# f8c_v2_config_small.yaml。save_steps 500 保持原样 —— 最终 adapter 在训练结束时
# 仍会保存, 中间 checkpoint 是否落盘不影响结论。
#
# 用法 (gpu04, srun --overlap 内, conda funaudiochat): bash g1_train_v2repro.sh
set -uo pipefail
A=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator
S=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts
PROJECT_ROOT=/share/workspace3/zhuangruicen/proj-thumpy/third_party/Fun-Audio-Chat
LOGS=$A/g1_logs
OUT=$A/f8c_v2_repro_sft
CFG=$OUT/g1_config.yaml
mkdir -p "$OUT/saves" "$LOGS"

# 原始 v2 adapter 必须完好 —— 这是不可再生的产物, 先确认它在
ORIG=$A/f8c_v2_sft/saves/adapter_model.safetensors
[ -f "$ORIG" ] || { echo "V2REPRO_ABORT: 原始 v2 adapter 不见了, 不敢动"; exit 2; }
echo "原始 v2 adapter 在位: $(stat -c%s "$ORIG") B"

sed -e "s|^output_dir:.*|output_dir: $OUT/saves|" \
    "$S/f8c_v2_config_small.yaml" > "$CFG"
echo "save_total_limit: 2" >> "$CFG"
rm -f "$OUT/saves/adapter_model.safetensors"     # 残留旧 adapter 会骗过成功判定

echo "--- 实际用于复现的 config (只改了 output_dir) ---"
grep -E "^(dataset|cutoff_len|per_device_train_batch_size|gradient_accumulation_steps|learning_rate|num_train_epochs|seed|lora_rank):" "$CFG"
echo "(无 seed 行 = LLaMA-Factory 默认 42, 与原 v2 一致)"

export AUDIO_PLACEHOLDER="<|audio_bos|><|AUDIO|><|audio_eos|>"
export DISABLE_VERSION_CHECK=1
unset FORCE_TORCHRUN
export PYTHONPATH="$PROJECT_ROOT/training/plugin:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJECT_ROOT/training"

echo "########## v2 复现训练  $(date '+%F %T') ##########"
llamafactory-cli train "$CFG" 2>&1 | tee "$LOGS/train_v2repro.log"
rc=${PIPESTATUS[0]}
echo "########## 训练结束 rc=$rc  $(date '+%F %T') ##########"

# 以产物为准
if [ ! -f "$OUT/saves/adapter_model.safetensors" ]; then
  echo "V2REPRO_FAILED: 训练未产出 adapter"
  exit 3
fi
echo "[OK] 复现 adapter $(stat -c%s "$OUT/saves/adapter_model.safetensors") B"

# 评估: 与 g1_eval_promptcheck.sh 里 v2_matchedprompt 逐字同参数 (只换 adapter 与 tag),
# 这样两者的 F1 才可直接比。
cd "$S"
echo "########## v2 复现评估  $(date '+%F %T') ##########"
python ari_g1_eval.py \
  --npz-dir "$A/g1_own_eval/own" \
  --manifest-from "$A/g1_sub/mixnorm" \
  --lora "$OUT/saves" \
  --tag v2_repro || { echo "V2REPRO_FAILED: 评估报错"; exit 4; }

[ -f "$A/g1_eval_v2_repro.json" ] || { echo "V2REPRO_FAILED: 缺 eval json"; exit 5; }
python3 -c "
import json
r = json.load(open('$A/g1_eval_v2_repro.json'))['v2_repro']
print('复现结果: F1=%.4f P=%.4f R=%.4f n_pred=%d' % (r['f1'], r['precision'], r['recall'], r['n_pred']))
print('对照 原始v2 = 0.1203 (P 0.0748 R 0.3077 n_pred 107)')
print('对照 own10 = 0.1727 (P 0.1062 R 0.4615 n_pred 113)')
"
echo "V2REPRO_DONE $(date '+%F %T')"
