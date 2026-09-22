# 2026-08-20 GPU 正式运行日志 (pilot_study)

> 承接 8/20 白天的 CPU 冒烟测试 (桩 diarization + whisper tiny, 3 条)。本日志记录当晚在 GPU 节点
> (gpu02, RTX 4090 D) 上用真实 pyannote + whisper large-v3 跑 20 条样本的完整过程与结果。

## 1. 前置准备

### 1.1 Hugging Face token 与 gated 模型

- 用户提供新 token (填入 `pilot_study/env.sh`, 同时显式声明 `HF_ENDPOINT=https://hf-mirror.com`)。
- 本集群 (login01 / gpu02) 直连 huggingface.co 被墙 (DNS 污染返回 Facebook IP, 连接超时),
  **hf-mirror 是唯一可用通道**。
- 关键验证结论 (**推翻了冒烟测试时的误判**): hf-mirror **会透传 token 鉴权**, 此前
  "镜像对 gated 仓库带 token 也 403" 是用无效 token 测出的错误结论。判别方法:
  - 伪造 token → 报 *"must be authenticated ... Please log in"* (凭据无效)
  - 有效 token 但未点协议 → 报 *"you are not in the authorized list"* (凭据通过, 未授权)
- gated 协议状态变化: `speaker-diarization-3.1` 用户已获批准; `segmentation-3.0` 用户当晚点击
  "Agree" 后生效 (有约一分钟的授权缓存延迟)。
- 依赖清单 (来自 diarization-3.1 的 `config.yaml`):
  - `pyannote/speaker-diarization-3.1` (gated) ✅
  - `pyannote/segmentation-3.0` (gated, MIT) ✅
  - `pyannote/wespeaker-voxceleb-resnet34-LM` (公开) ✅

### 1.2 GPU 与环境

- GPU: 用户在 **kimi tmux 窗口**内 `srun --gres=gpu:1 --mem=32G -c 8 -p gpu2node -w gpu02 --pty bash`
  申请到 RTX 4090 D (24GB)。Claude Code 会话在 login01 的 claude 窗口, 通过
  `tmux send-keys -t kimi:0` 驱动 gpu02 上的 shell; `/share` 为共享存储。
- 环境: conda `fd_pilot` (py3.10), torch 2.13.0+cu130, CUDA 可用 ✅
- whisper large-v3 (2.88GB) 在 gpu02 上后台下载, 4m25s @ ~11.6MiB/s, 缓存至 `~/.cache/whisper/`
  (whisper 走 Azure 源, 不受 HF 墙影响)。
- 启动封装脚本 `pilot_study/run_on_gpu.sh`: cd 到 pilot_study → source env.sh → nohup 跑 pipeline。

## 2. 运行过程

```bash
# gpu02 (kimi 窗口):
bash /share/workspace3/zhuangruicen/proj-thumpy/pilot_study/run_on_gpu.sh
# login01 (本窗口):
/share/home/zhuangruicen/miniconda3/envs/fd_pilot/bin/python pilot_study/scripts/analyze_results.py
```

**耗时**: 20/20 条共 2m24s (~7.2s/条, 含首条模型暖机; 后续 ~4.5s/条)。

## 3. 踩坑与修复

1. **pyannote 4.x API 不兼容 (首次全量运行暴露)**:
   `pipeline()` 在 pyannote 4.x 返回 `DiarizeOutput` dataclass 而非 3.x 的 `Annotation`,
   直接调用 `.itertracks()` 报 `'DiarizeOutput' object has no attribute 'itertracks'`。
   修复: 取 `.speaker_diarization` 字段 (含重叠话轮的 Annotation) 后再做重叠提取。
   **教训**: 冒烟测试用桩数据绕过了真实 diarization 返回对象, 该兼容性问题只能靠真实模型暴露。
2. **跨节点监控教训**: login01 上用 `kill -0 <PID>` 检查 gpu02 进程永远为假, 曾误判 pipeline 已死。
   改为基于日志标记 ("All results saved" / Traceback) 的监控。
3. **tmux 驱动多次漏 cd**: 改为共享盘上的封装脚本一条命令启动, 避免 send-keys 引号/路径纠缠。

## 4. 结果

### 4.1 总览

- `results/pilot_results.json`: 20 条全部成功, **无任何 error 条目**。
- `results/diarization/*.json` × 20, `results/transcription/*.json` × 20 ✅
- `results/analysis/confusion_matrix.png`, `analysis_report.md` ✅

### 4.2 标签一致性

| 指标 | 值 |
|------|-----|
| 一致 (match) | 12/20 (60.0%) |
| **不一致率** | **40.0%** (> 20% 目标, 假设得到支撑) |
| 检出双说话人 | 7/20 条 (全部为 Interruption 标签) |

不一致 8 条明细: 3 条 Interruption → None (重叠 0s), 5 条 Backchannel → None (重叠 0s)。

### 4.3 根因分析 (已用 RMS 能量剖面验证)

不是 pipeline bug, 而是**真实 pyannote 对合成谐波信号的说话人分离能力限制**:

| 标签 | 检出 | 根因 |
|------|------|------|
| None (5) | 5/5 ✅ | 无重叠, 1 说话人, 判定一致 |
| Interruption (10) | 7/10 | B 音量 1.05–1.2 时能分离; 漏检 3 条中 B 能量存在 (RMS 有凸起) 但被合并进 A |
| Backchannel (5) | 0/5 ❌ | B 音量仅 0.5–0.65 (轻声回应设计), 淹没在 A 包络中, pyannote 完全听不到第二人 |

另: whisper 对非语音谐波信号产生少量幻觉转写 (如 "C-M Master"), 属预期 — 合成数据无真实词。

### 4.4 解读

- **pipeline 流程与判定规则已验证可靠** (对能检出的重叠, 时长提取准确: 如 sample_002 检出
  [1.23–1.99s] 0.76s 与设计 [~1.2s, ~0.9s] 吻合)。
- 40% 不一致率中,**Backchannel 全军覆没是合成数据音量设计导致的系统性检测下限**, 不等于
  真实语音场景的表现; 若用于组会汇报, 需注明这一点, 否则会高估 "标签-实现不一致"。
- 该结果同时恰好佐证了本研究的假设方向之一: **行为意图 (标签) 与声学可感知的实现之间的
  gap 是真实存在的** — 但需要真实语音数据才能给出可外推的量化。

## 5. 下一步建议

1. **替换真实数据** (doc §2.2 方案 A): 下载 Behavior-SD 公开音频 20–30 条重跑本 pipeline
   (代码与流程无需改动), 才是可汇报的正式结论。
2. 若继续用合成数据, 需调高 Backchannel 音量 (≥0.9) 或加长重叠, 但合成数据的结论仍不可外推。
3. whisper 幻觉转写不影响重叠判定 (判定只看 diarization), 无需处理。
4. GPU 使用已结束, 可释放 kimi 窗口的 srun 分配 (或留给后续真实数据运行)。
