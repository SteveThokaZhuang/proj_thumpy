# ARI 实验工程排查记录（2026-08-24）

配套：[docs/pilot_study/2026-08-23_ari_results.md](../pilot_study/2026-08-23_ari_results.md)

## 1. pandas 读 CANDOR 转写 CSV 段错误（已解决）

**现象**：`ari_extract_candor.py` 在 gpu02 上以段错误 (SIGSEGV) 死亡，且**每个 worker 都死在
自己分片里的固定会话上**（w1 死在第 140 个会话 `aced00ba-…`，w4/w5/w6/w7 各自死在不同进度）。
重跑后 resume 跳过好文件、立刻撞上坏文件 → 表现为"秒死"。

**定位过程**：
- `python -X faulthandler` 显示 C 引擎崩在 `pandas/io/parsers/c_parser_wrapper.py`；
- 改 `engine="python"` 后崩点变为 `base_parser.py:_infer_types`（字符串转数值时）；
- 文件本身"正常"：无 NUL 字节、无非 ASCII，最长行 775 字节——但有**未闭合引号字段**；
- 同样的文件在 login01 读正常，两个引擎在不同内部点崩溃 → 无法在进程内捕获。

**解决**：转写 CSV 全部改用**标准库 `csv.DictReader`**（纯 Python 解析，无 C 扩展参与），
脚本 `read_transcript()`；输出 CSV 的 resume 检查同样改用 csv 模块。
解析性能差异可忽略（文件 ≤1000 行）。

**教训**：pandas 对真实世界脏 CSV（未闭合引号等）在两个引擎下都可能段错误，且 segfault
不可捕获。读外部来源的 CSV 时优先 csv 模块，或至少不要在生产循环里裸调 `pd.read_csv`。

## 2. `srun --overlap` 短步骤清理子进程（已解决）

**现象**：`srun --jobid=<job> --overlap bash -c 'nohup worker & ...; exit'` 方式启动的 worker
在启动后几十秒内集体消失（`setsid`/`nohup` 均无效）。

**原因**：SLURM 步骤结束时按其 cgroup 清理该步骤的**全部**进程，setsid 脱离会话无法逃出
cgroup 清理。首次启动时 `ps` 还能看到进程，是因为清理发生在步骤 epilog 阶段（延迟数秒）。

**解决**：把 worker 作为 srun 步骤的**前台子进程**，步骤内 `bash -c '... & wait'` 保持存活，
worker 自然结束后步骤才退出：

```bash
srun --jobid=<job> --overlap bash -c '
  cd /share/workspace3/zhuangruicen/proj-thumpy/pilot_study
  for w in 0 1 2 3 4 5 6 7; do
    python scripts/ari_extract_candor.py --worker $w --n-workers 8 ... &
  done
  wait'
```

**教训**：交互窗格（`srun --pty bash`）里 nohup 起的进程能存活（步骤常驻）；通过
`srun --overlap bash -c` 一键式启动必须用 `&` + `wait` 的持久步骤模式。

## 3. 其他要点

- **login01 纪律**：B=500 bootstrap 一度在 login01 跑了 20+ CPU 分钟（违反
  pipeline_flow 的禁令），经提醒后杀掉并在 gpu02 重跑。此后所有计算（含分析、出图）
  一律走 gpu02 持久步骤，login01 仅做轻量文件检查。
- **gpu02 为共享节点**：其他用户的训练任务曾在节点上造成内存/CPU 压力；
  提取脚本自带断点续跑（按输出 CSV 已有的 session/file_id 跳过），进程被外部扫掉后
  重跑即可续上，无需从头开始。
- **CANDOR mp3 读取优化**：全量解码 45min 会话需 ~76s（解码 34s + soxr 重采样 36s），
  改用 libsndfile 窗口级 seek 读取 + 每窗口独立重采样后降到 ~14s/会话（6.2×）。
  实测 seek 精度良好（包络相关 ≥0.999），事件窗口留 0.15s 余量即可。
- **pandas 读自身输出的 CSV 时**：类别名 "None" 会被默认 na_values 吞掉，
  统一用 `read_csv(..., keep_default_na=False)`（或 csv 模块）。
- **tmux send-keys 与用户输入冲突**：向用户正在使用的窗格注入命令会与用户打字交错，
  产生拼接错误命令。需要与交互窗格协作时优先用 `srun --jobid --overlap` 通道。
