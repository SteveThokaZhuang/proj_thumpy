# F8：Fun-Audio-Chat 8B 话轮状态 LoRA SFT——数据飞轮的第一次闭环

> **日期**：2026-08-27
> **动机**：[数据飞轮计划](2026-08-26_fd_flywheel_plan.md) 的训练侧首验：
> 用 CANDOR-FD 的 L3 标注（X2-Turn 话轮状态）对半双工模型做轻量后训练，
> 检验"标注数据 → 训练增益"的传导率。
>
> **脚本**：`pilot_study/scripts/ari_f8_build_dataset.py`（数据）+
> `f8_sft_config.yaml` + `f8_sft_run.sh`（训练）+ `ari_f8_evaluate.py`（评估）
> **产物**：LoRA adapter `annotator/f8_sft/saves/`；数据集 `annotator/f8_sft/`

---

## 1. 设置

- **基座**：Fun-Audio-Chat-8B（半双工语音对话模型，5Hz 语音表示；官方 LLaMA-Factory
  SFT 管线，LoRA rank 16）
- **任务**（B 变体）：输入 = 对方说话人 10s 语音 + 指令；输出 = 每 1 秒一个
  话轮状态词（listen/speak/backchannel），标签来自 X2-Turn 80ms 软概率的阈值离散化
- **数据**：CANDOR-FD 的 3,600 个 10s 块；单卡 24GB（训练时 23.3GB 显存、98% 利用率、
  4.87s/步、18.5 分钟完成 2 epochs）
- **评估**：留出 100 块（非训练集），逐词 accuracy / 序列 EM / 逐类 recall，
  基座 vs LoRA 对比

## 2. 结果

| 模型 | 词准确率 | EM | listen recall | speak recall | backchannel recall |
|---|---|---|---|---|---|
| 基座（无训练） | 0.651 | 0.21 | 0.682 | 0.333 | 0.222 |
| v1 LoRA（随机采样，94% listen 词主导） | 0.910 | 0.67 | 0.992 | 0.075 | **0.0（多数类坍缩）** |
| **v2 LoRA（类别平衡采样）** | **0.898** | **0.58** | **0.938** | **0.487** | **0.444** |

## 3. 裁决

1. **传导率成立**：v2 平衡训练相对基座全面提升——speak recall +46%
   （0.33→0.49）、**backchannel recall 翻倍（0.22→0.44）**、EM 0.21→0.58。
   18 分钟单卡 LoRA 即可让半双工模型获得显著的话轮状态预测能力——"低训练成本
   获得全双工能力"的第一个实证。
2. **v1 坍缩是重要发现**：随机采样下 listen 词占 94%，训练坍缩到多数类
   （整体准确率反升到 0.91 但稀有类归零）。**类别平衡采样是话轮状态 SFT 的
   必要条件**——这对 F9 的合成基线对照同样适用（合成 metadata 标注训练
   也必须用平衡采样，否则对照不公平）。
3. **工程发现**：LLaMA-Factory 此版本的 loss 恒显 0.0（真实训练确在发生——
   适配器权重 91% 非零、max|w|=0.042）——**训练监控必须用生成评估而非
   报告 loss**。
4. 数据管线全链路打通：CANDOR-FD 标注 → 平衡采样 → sharegpt 格式（官方
   audio 字段：path+pad token）→ SFT → 留出评估。F9（真实 vs 合成标注双基线）
   可直接复用。

## 4. 局限与下一步

- 100 样本留出集、单 seed；backchannel 词仅 734 个（数据侧上限）。
- 任务为"状态描述"而非"状态驱动的回应生成"——F9 后应增加
  "预测 BC 内容/时机并生成回应"的变体（任务 B 完整版）。
- 下一步：F9 双基线（合成 metadata 标注同任务同量级训练，检验
  "真实标注 > 合成标注"预言）；然后 F7 评测 harness 应用于训练前后模型。

## 5. 复现

```bash
# 数据 (fd_analysis 环境)
python scripts/ari_f8_build_dataset.py --n-samples 3600
# 训练 (funaudiochat 环境, gpu02)
bash scripts/f8_sft_run.sh
# 评估
python scripts/ari_f8_evaluate.py --n-eval 100 \
    --lora-dir real_data/results/annotator/f8_sft/saves
```
