# Behavior-SD Pilot Study

参照 `../docs/0820guidance.md`。真实数据评测 (validation + test + train 前 30%) 已完成:

- 结论: `../docs/log/2026-08-23_train30_final.md`
- 报告: `../docs/reports/analysis_report.md` (pilot_study 内不保存分析报告)
- 运行流程: `../docs/2026-08-23_pipeline_flow.md`

## 目录结构 (2026-08-23 清理后)

```
proj-thumpy/
├── docs/
│   ├── 0820guidance.md              # 研究设计 (guidance)
│   ├── 2026-08-23_pipeline_flow.md  # 评测运行流程文档
│   ├── log/                         # 按日期的运行/验证/勘误日志
│   └── reports/                     # 分析报告 (统一存放)
│       ├── analysis_report.md + 混淆矩阵 PNG ×2
│       └── verification/            # Backchannel 声道级验证报告 + 波形图
└── pilot_study/
    ├── env.sh                       # LD_LIBRARY_PATH 修复 + HF_TOKEN + hf-mirror
    ├── run_real_workers.sh          # gpu02 上 4×4090 启动 4 worker
    ├── scripts/                     # 评测/分析/验证脚本 (见 pipeline_flow §6)
    └── real_data/
        ├── metadata_{validation,test,train}.json   # 官方原始格式 + file_id
        ├── results/{split}/{diarization,transcription}/*.json  # per-file 结果
        ├── results/{split}/pilot_results_shard{0..3}.json      # 汇总 (无重复)
        ├── results/analysis/{statistics,per_split_summary}.csv # 分析数据
        └── verification/            # 声道级验证数据 (stats json + 试听 wav)
```

## 环境

- conda env `fd_pilot` (Python 3.10): pyannote.audio 4.0.7、whisper、librosa、pandas 等。
- 运行前 `source env.sh`: 修复 `CXXABI_1.3.15` 库路径问题 + 设置 HF_TOKEN (gated 模型,
  需在 HF 接受协议) + `HF_ENDPOINT=https://hf-mirror.com` 镜像加速。
- GPU: kimi 窗口 gpu02 (4×RTX 4090 D), `srun --gres=gpu:4 --mem=32G -c 8 -p gpu2node
  -w gpu02 --pty bash`。**login01 禁止跑大负载任务**。

## 快速命令

```bash
cd pilot_study && source env.sh
# (可选) 提取音频 — real_data/audio/ 已清理, 重跑 pipeline 前先重建 (flac -> 16k mono wav)
TAR_LIMIT=64 SPLITS=train python scripts/extract_audio.py
# 重建官方 metadata
SPLITS=validation,test python scripts/rebuild_metadata_full.py
# 跑 pipeline (每卡 1 worker)
bash run_real_workers.sh validation
# 分析 (报告 -> ../docs/reports/, csv -> real_data/results/analysis/)
SPLITS=validation,test,train python scripts/analyze_real_results.py
```

## 关键结果 (33,857 条: train 32,000 前 30% + val 932 + test 925)

- 标签 vs GT (区间时间线) 一致率 **80.4%**; 标签 vs pyannote 检测一致率 **67.2%**;
  检测器有效性 Pearson **r = 0.857** (三 split 稳定: 0.849–0.862)。
- 三层框架: ① 标签→元数据基本诚实 (剩余 19.6% 为 BC↔Int 阈值错配, 非标注错误);
  ② 元数据区间→声学渲染有 ~2× 系统性落差 (GT 4.59s vs 检测 2.30s; Backchannel 标称
  4.49s, 声道级实测真正同时发声仅 ~15% ≈ 0.6s/条, 其余渲染进宿主停顿);
  ③ pyannote 是合格的相关性检测器, 但事件级判定不宜直接当 realized 基准
  (后续研究以数据集 utterance/backchannel 区间时间线为准)。

## 🔴 重大勘误 (2026-08-21)

旧预处理 `prepare_real_data.py` (已删除) 丢弃嵌套 backchannels 字段, 导致
"Backchannel 零重叠""元数据标注错误""pyannote 过分割高估"均为伪象。
官方格式重建 (rebuild_metadata_full.py) 后: 标签→元数据 100% 同源, 真正的不一致在
元数据区间 vs 声学渲染。详见 `../docs/log/2026-08-21_metadata_rebuild_and_reeval.md`
与 `../docs/log/2026-08-21_backchannel_stereo_verification.md`。

## 历史

- 2026-08-20 合成数据 smoke test (20 条) 已完成并清理; 记录见
  `../docs/log/2026-08-20_gpu_full_run.md`。
- 2026-08-23 项目清理 (75G -> 1.4G): 删除提取音频/smoke test/备份/临时文件,
  报告统一移至 docs/。见 `../docs/log/2026-08-23_train30_final.md` §9。
