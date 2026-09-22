# Interruption 声学重叠验证 [validation] (n=818)

- 方法: 原始 flac 双声道 RMS 能量, 阈值 0.1×声道最大, 事件 ≥0.1s
- **双声道同时活跃 (声学重叠)**: 817/818 条有重叠 (100%)
- 重叠总时长: 均值 0.916s | 中位 0.790s | 最大 3.440s | 有重叠样本均值 0.917s
- 重叠事件数(≥0.1s): 均值 3.86 / 条
- **标注-声学失配** (音频有声音但 GT 说话人静音): 均值 1.672s/条
- pyannote 检出重叠 (有结果 818 条): 均值 2.321s vs 声道实测 0.916s

## 逐样本 (前 15 条)

| file_id | dur(s) | 重叠(s) | 事件 | 失配A(s) | 失配B(s) | pyannote(s) |
|---------|--------|---------|------|----------|----------|-------------|
| validation_0000000005 | 78.42 | 0.83 | 4 | 1.45 | 0.0 | 2.45 |
| validation_0000000012 | 57.98 | 1.62 | 7 | 1.36 | 2.51 | 3.59 |
| validation_0000000013 | 41.92 | 0.52 | 3 | 1.35 | 0.0 | 1.77 |
| validation_0000000023 | 122.68 | 1.54 | 6 | 2.06 | 3.12 | 4.62 |
| validation_0000000033 | 39.38 | 0.32 | 2 | 0.0 | 0.33 | 0.41 |
| validation_0000000035 | 83.11 | 1.24 | 5 | 0.25 | 1.79 | 3.12 |
| validation_0000000036 | 71.72 | 1.02 | 4 | 1.57 | 0.0 | 2.51 |
| validation_0000000037 | 62.69 | 0.21 | 0 | 0.0 | 1.09 | 1.30 |
| validation_0000000062 | 46.82 | 0.68 | 3 | 0.83 | 0.72 | 1.94 |
| validation_0000000065 | 49.78 | 0.75 | 3 | 0.51 | 0.63 | 2.31 |
| validation_0000000067 | 74.95 | 0.6 | 2 | 1.85 | 0.0 | 2.31 |
| validation_0000000074 | 66.6 | 1.08 | 5 | 2.32 | 0.0 | 1.35 |
| validation_0000000077 | 95.51 | 1.23 | 4 | 1.75 | 2.39 | 3.37 |
| validation_0000000079 | 42.59 | 0.41 | 2 | 0.2 | 0.45 | 1.06 |
| validation_0000000082 | 69.62 | 0.27 | 0 | 0.21 | 0.0 | 0.42 |

图: ['/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/verification/plot_validation_validation_0000000005.png', '/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/verification/plot_validation_validation_0000000012.png', '/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/verification/plot_validation_validation_0000000013.png', '/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/verification/plot_validation_validation_0000000023.png', '/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/verification/plot_validation_validation_0000000033.png']

VERIFY DONE