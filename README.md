# proj-thumpy

全双工语音对话中的 **backchannel（应和语）事件检测** 研究项目。

一句话概括当前结论：在 CANDOR 多人对话上做行为事件检测，**观测空间的选择**
（送入模型的音频表征）确实影响事件级 F1，且这个效应**不是少数会话撑起来的**；
但效应量级很小、方差结构复杂，项目花了大量精力在**先把噪声底噪量清楚**上——
详见下面的「研究记录」。

> 🔑 **要读结论，从这里进：[`docs/pilot_study/syllabus.md`](docs/pilot_study/syllabus.md)** ——
> 那是本项目的**总索引**（现状、被推翻的结论、待办、指针）。
> 引用任何旧结论之前，先查
> [`docs/pilot_study/2026-09-22_claims_ledger.md`](docs/pilot_study/2026-09-22_claims_ledger.md)（**结论总账**：
> 已证实 / 已证伪 / 悬而未决，按状态重排过）。
> README 只讲「怎么把环境跑起来」，**不复制结论**——结论只有一处出处，改起来才不会打架。

---

## 目录结构

```
proj-thumpy/
├── docs/                          # 研究记录 (markdown, 会入库)
│   ├── pilot_study/               # ★ 主战场
│   │   ├── syllabus.md            #   🔑 总索引, 从这进
│   │   ├── 2026-09-22_claims_ledger.md   # 结论总账 (状态机)
│   │   ├── 2026-09-11_g1_observation_ablation.md  # G1 主报告
│   │   ├── 2026-09-18_pool_sd.md  # 池间 SD
│   │   ├── 2026-09-22_planC_prereg.md   # C 方案预登记 (跑之前写死)
│   │   ├── log/                   #   按日期的运行/勘误日志
│   │   ├── reports/               #   分析报告
│   │   └── figures/               #   图
│   ├── paper/                     # 论文草稿
│   ├── finegrid_label/            # 细粒度标注设计
│   └── log/
├── pilot_study/                   # 代码与运行产物
│   ├── scripts/                   #   ★ 全部脚本 (py + sh)
│   ├── env.sh.example             #   环境模板 → 复制成 env.sh 再填 token
│   ├── *.sh                       #   worker 启动脚本
│   ├── real_data/                 #   ⛔ 不入库 (146 G, 见「数据从哪来」)
│   ├── ari_logs/  g1_logs/        #   ⛔ 不入库 (运行日志)
│   └── README.md                  #   早期 pipeline (pyannote 质检) 的说明
└── third_party/                   # ⛔ 不入库 (按下面的 URL 自行 clone)
```

`docs/pilot_study/` 与 `pilot_study/` 是**平级**的两个目录，不是同一个——
前者是**记录**，后者是**代码和产物**。文档里引用路径时两者都从仓库根写起。

---

## 环境

三个 conda 环境，分工不同。脚本里的解释器路径是**写死**的
（`/share/home/zhuangruicen/miniconda3/envs/<env>/bin/python`），换机器要改。

| 环境 | 用途 |
|---|---|
| `funaudiochat` | 训练与 GPU 评估（有 `transformers` / `peft`） |
| `fd_analysis` | 数据构建与判读（纯 CPU、纯读盘） |
| `fd_pilot` | 早期 pyannote 质检 pipeline |

```bash
conda create -n funaudiochat python=3.11   # 具体依赖见 docs/pilot_study/ 各文档
conda create -n fd_analysis  python=3.10

cd pilot_study
cp env.sh.example env.sh    # ⚠️ 然后填入自己的 HF token
source env.sh               # 修 LD_LIBRARY_PATH + 设 HF_TOKEN + hf-mirror
```

> ⚠️ **`env.sh` 里是明文凭据，已被 `.gitignore` 排除。**
> `git add -f pilot_study/env.sh` 也不行——那是给公开仓库。

### 第三方仓库（不入库，自行 clone）

`third_party/` 下的五个仓库按以下 URL 与 commit 取（版本对齐过，换版本可能复现不出）：

```bash
cd third_party
git clone https://github.com/pengyizhou/FD-Bench.git            && git -C FD-Bench checkout 8a4b7df
git clone https://github.com/QwenAudio/Fun-Audio-Chat.git       && git -C Fun-Audio-Chat checkout 8ba984b
git clone https://github.com/Soul-AILab/SoulX-Duplug.git        && git -C SoulX-Duplug checkout a0b9063
git clone https://github.com/X-Square-Robot/X2-Turn.git         && git -C X2-Turn checkout c3c3a0d
# SoulX-Duplug-training: 非 git 仓库 (内部训练代码分支, 无公开 URL)
```

## 数据从哪来

数据**不在仓库里**（`real_data/` 有 146 G，是 CANDOR 的评测产物与观测空间 npz）。
需要自备：

| 数据 | 路径（脚本里写死） |
|---|---|
| CANDOR 主库 | `/share/workspace3/shared_dataset/CANDOR/files/` |
| 共享模型权重 | `/share/workspace3/shared_models/` |

依赖的数据家底（1,656 个会话、100% 有 mp3、1,655 已带 GT 标注），
以及「已用了哪 100 个会话」的准确说法，记在
`docs/pilot_study/2026-09-22_claims_ledger.md` §5 与 syllabus §5.4——
**扩集之前先查那里**，别照抄二手数字（项目在这上面栽过两次）。

## 计算纪律

**登录节点禁止跑任务。** 所有计算走已有分配的 srun 步骤：

```bash
srun --overlap --jobid=<JOBID> -n 1 -c 2 bash scripts/<脚本>.sh
```

⚠️ `-n 1` 不能省：`srun -c N` 不加 `-n 1` 会**并发起 N 份**，共享同一份
`> file` 重定向产生竞态（输出行数成倍）。另：0 字节日志 ≠ 失败，可能只是块缓冲。

## 复现一个具体结果

每个实验文档的 §「复现命令」小节都写了**从哪一步到哪一步**的命令序列。
最新的一个是 C 方案，见
[`docs/pilot_study/2026-09-22_planC_prereg.md`](docs/pilot_study/2026-09-22_planC_prereg.md) §7：
建 id 列表 → 建观测空间 npz → 评估 → 判读，四步，各步用哪个 conda 环境也标了。

## 关于「代码部分」的一句说明

本仓库以**代码 + 研究记录**为主。判读脚本之间共享一套口径
（F1 必须在 chunk 的**并集**上算、bootstrap 必须**三级**同时抽会话/块/种子），
这些口径写在脚本头部的注释里，改脚本前请先读——项目在这些点上各翻过一次车，
注释里记的是**为什么**，不是**是什么**。
