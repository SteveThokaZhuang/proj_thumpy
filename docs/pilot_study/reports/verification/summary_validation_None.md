# None 声学重叠验证 [validation] (n=14)

- 方法: 原始 flac 双声道 RMS 能量, 阈值 0.1×声道最大, 事件 ≥0.1s
- **双声道同时活跃 (声学重叠)**: 0/14 条有重叠 (0%)
- 重叠总时长: 均值 0.000s | 中位 0.000s | 最大 0.000s | 有重叠样本均值 nans
- 重叠事件数(≥0.1s): 均值 0.00 / 条
- **标注-声学失配** (音频有声音但 GT 说话人静音): 均值 0.157s/条
- pyannote 检出重叠 (有结果 14 条): 均值 0.013s vs 声道实测 0.000s

## 逐样本 (前 15 条)

| file_id | dur(s) | 重叠(s) | 事件 | 失配A(s) | 失配B(s) | pyannote(s) |
|---------|--------|---------|------|----------|----------|-------------|
| validation_0000000048 | 106.39 | 0 | 0 | 0.09 | 0.0 | 0.00 |
| validation_0000000238 | 139.57 | 0 | 0 | 0.16 | 0.1 | 0.00 |
| validation_0000000281 | 146.69 | 0 | 0 | 0.14 | 0.11 | 0.00 |
| validation_0000000648 | 74.8 | 0 | 0 | 0.0 | 0.03 | 0.00 |
| validation_0000001067 | 165.81 | 0 | 0 | 0.49 | 0.24 | 0.00 |
| validation_0000001206 | 63.13 | 0 | 0 | 0.0 | 0.16 | 0.00 |
| validation_0000001302 | 114.88 | 0 | 0 | 0.04 | 0.17 | 0.00 |
| validation_0000001323 | 34.83 | 0 | 0 | 0.0 | 0.0 | 0.00 |
| validation_0000002308 | 86.05 | 0 | 0 | 0.0 | 0.1 | 0.19 |
| validation_0000002915 | 38.79 | 0 | 0 | 0.0 | 0.0 | 0.00 |
| validation_0000002934 | 38.58 | 0 | 0 | 0.0 | 0.0 | 0.00 |
| validation_0000003185 | 143.22 | 0 | 0 | 0.17 | 0.17 | 0.00 |
| validation_0000003260 | 70.13 | 0 | 0 | 0.0 | 0.0 | 0.00 |
| validation_0000003369 | 99.98 | 0 | 0 | 0.01 | 0.02 | 0.00 |

图: ['/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/verification/plot_validation_validation_0000000048.png', '/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/verification/plot_validation_validation_0000000238.png', '/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/verification/plot_validation_validation_0000000281.png', '/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/verification/plot_validation_validation_0000000648.png', '/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/verification/plot_validation_validation_0000001067.png']

VERIFY DONE