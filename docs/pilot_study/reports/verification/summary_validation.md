# Backchannel 声学重叠验证 [validation] (n=100)

- 方法: 原始 flac 双声道 RMS 能量, 阈值 0.1×声道最大, 事件 ≥0.1s
- **双声道同时活跃 (声学重叠)**: 91/100 条有重叠 (91%)
- 重叠总时长: 均值 0.605s | 中位 0.455s | 最大 3.000s | 有重叠样本均值 0.665s
- 重叠事件数(≥0.1s): 均值 2.52 / 条
- **标注-声学失配** (音频有声音但 GT 说话人静音): 均值 1.903s/条
- pyannote 检出重叠 (有结果 100 条): 均值 1.711s vs 声道实测 0.605s

## 逐样本 (前 15 条)

| file_id | dur(s) | 重叠(s) | 事件 | 失配A(s) | 失配B(s) | pyannote(s) |
|---------|--------|---------|------|----------|----------|-------------|
| validation_0000000019 | 101.2 | 0.11 | 0 | 1.52 | 0.13 | 0.00 |
| validation_0000000032 | 70.48 | 0 | 0 | 0.0 | 0.4 | 0.00 |
| validation_0000000055 | 35.84 | 0.27 | 2 | 0.22 | 0.0 | 0.66 |
| validation_0000000121 | 24.39 | 0.67 | 2 | 1.1 | 0.43 | 2.40 |
| validation_0000000150 | 95.66 | 0.79 | 2 | 1.52 | 0.0 | 1.21 |
| validation_0000000151 | 50.74 | 0.21 | 1 | 1.32 | 0.0 | 0.17 |
| validation_0000000156 | 71.38 | 0.64 | 3 | 1.98 | 0.0 | 1.64 |
| validation_0000000216 | 140.31 | 1.08 | 6 | 1.01 | 1.08 | 2.45 |
| validation_0000000234 | 121.11 | 1.6 | 6 | 2.8 | 2.0 | 3.56 |
| validation_0000000324 | 97.81 | 3.0 | 14 | 0.2 | 6.12 | 5.79 |
| validation_0000000334 | 73.09 | 0.2 | 1 | 0.36 | 2.47 | 0.78 |
| validation_0000000389 | 62.46 | 0.06 | 0 | 0.0 | 0.47 | 0.66 |
| validation_0000000410 | 104.97 | 1.3 | 4 | 2.84 | 2.32 | 1.87 |
| validation_0000000430 | 77.99 | 0.69 | 1 | 1.53 | 0.0 | 2.02 |
| validation_0000000454 | 52.13 | 0 | 0 | 0.23 | 0.0 | 0.00 |

图: ['/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/verification/plot_validation_validation_0000000019.png', '/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/verification/plot_validation_validation_0000000032.png', '/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/verification/plot_validation_validation_0000000055.png', '/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/verification/plot_validation_validation_0000000121.png', '/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/verification/plot_validation_validation_0000000150.png']

VERIFY DONE