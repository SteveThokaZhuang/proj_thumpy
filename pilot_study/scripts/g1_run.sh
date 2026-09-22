#!/usr/bin/env bash
# G1 观测空间消融: 单臂训练启动脚本.
# 用法 (gpu 节点, srun --overlap 内, funaudiochat 环境):
#   bash g1_run.sh <arm> <npz_dir> <out_dir>
# 例: bash g1_run.sh mixnorm10 .../g1_mixnorm10 .../g1_mixnorm10_sft
# pipefail 必须有: 末尾 `llamafactory-cli train | tee` 的退出码默认是 tee 的,
# 训练崩了也会返回 0 —— 上一轮 own10 就是这样静默失败、adapter 没保存。
set -uo pipefail
ARM=$1; NPZ=$2; OUT=$3
PROJECT_ROOT=/share/workspace3/zhuangruicen/proj-thumpy/third_party/Fun-Audio-Chat
SCRIPTS=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts
LOGS=/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator/g1_logs
mkdir -p "$LOGS" "$OUT"

# 注册数据集 (共享路径写 dataset_info.json)
python - "$ARM" "$OUT/train.jsonl" <<'PY'
import json, sys
arm, path = sys.argv[1], sys.argv[2]
p = ("/share/workspace3/zhuangruicen/proj-thumpy/third_party/"
     "Fun-Audio-Chat/training/data/dataset_info.json")
d = json.load(open(p))
d[f"g1-{arm}"] = {
    "file_name": path, "formatting": "sharegpt",
    "columns": {"system": "system", "messages": "messages", "audios": "audio"},
    "tags": {"role_tag": "role", "content_tag": "content",
             "user_tag": "user", "assistant_tag": "assistant"}}
json.dump(d, open(p, "w"), ensure_ascii=False, indent=2)
print("registered dataset g1-" + arm)
PY

# 生成配置 (写共享路径, 不用 /tmp —— /tmp 是按节点的)
# save_steps 从 500 降到 25: 总步数只有 132, 原设置等于全程不存盘, 一旦
# 末步崩溃 (已发生过一次 CUBLAS 错误) 就丢掉整个训练。快照不影响优化轨迹。
# micro-batch 8 -> 4 (accum 4 -> 8, 全局 batch 仍是 32):
# 两次失败都是"差一点"的内存失败, 且发生在**不同**步数 (开 flag 时 step 131
# 报 CUBLAS, 关 flag 时 step ~57 报 OOM) —— 说明不是随训练稳定增长, 而是
# 动态 padding 下某一批的序列特别长导致的峰值尖刺。降 micro-batch 直接把峰值
# 砍半 (实测 batch=8 时 PyTorch 分配 22.51 GiB, 其中 16 GB 是 8B 权重, 留给
# 激活的只有 ~6 GB)。全局 batch 不变, 梯度在数学上等价, 两臂同样处理。
CONFIG="$OUT/g1_config.yaml"
sed -e "s|^dataset:.*|dataset: g1-$ARM|" \
    -e "s|^output_dir:.*|output_dir: $OUT/saves|" \
    -e "s|^save_steps:.*|save_steps: 25|" \
    -e "s|^per_device_train_batch_size:.*|per_device_train_batch_size: 4|" \
    -e "s|^gradient_accumulation_steps:.*|gradient_accumulation_steps: 8|" \
    "$SCRIPTS/f8c_v2_config.yaml" > "$CONFIG"
echo "save_total_limit: 2" >> "$CONFIG"
mkdir -p "$OUT/saves"
# 删掉上一轮可能残留的 adapter: 否则末尾的"是否成功"检查会被旧文件骗过
rm -f "$OUT/saves/adapter_model.safetensors"

export AUDIO_PLACEHOLDER="<|audio_bos|><|AUDIO|><|audio_eos|>"
export DISABLE_VERSION_CHECK=1
# 注意: gpu02/gpu04 上 FORCE_TORCHRUN 会触发 NCCL 错误 -> 单进程
unset FORCE_TORCHRUN
export PYTHONPATH="$PROJECT_ROOT/training/plugin:$PROJECT_ROOT:${PYTHONPATH:-}"
# expandable_segments 必须开。上一轮我把它当成 CUBLAS 崩溃的嫌疑犯去掉了,
# 结果同一份配置在 step ~57 就 OOM, 而报错原文 ("Tried to allocate 20.00 MiB",
# 且仅剩 7.75 MiB) 正是碎片化的经典签名 —— 它是缓解手段, 不是病因。开着时
# 同一配置能跑到 step 131。两臂都设, 不构成臂间差异。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJECT_ROOT/training"

echo "=== arm=$ARM config=$CONFIG ==="
llamafactory-cli train "$CONFIG" 2>&1 | tee "$LOGS/train_$ARM.log"
rc=${PIPESTATUS[0]}
echo "=== arm=$ARM llamafactory 退出码 $rc ==="

# 退出码不可信: 上一轮 own10 报 rc=0 但 adapter 根本没落盘。以文件为准。
if [ -f "$OUT/saves/adapter_model.safetensors" ]; then
  echo "=== arm=$ARM OK: adapter 已保存 ($(stat -c%s "$OUT/saves/adapter_model.safetensors") B) ==="
  exit 0
fi
echo "=== arm=$ARM **失败**: 无 adapter_model.safetensors; saves/ 内容: $(ls "$OUT/saves" | tr '\n' ' ') ==="
exit 1
