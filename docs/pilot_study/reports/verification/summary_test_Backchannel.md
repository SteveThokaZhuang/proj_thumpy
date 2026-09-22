# Backchannel 声学重叠验证 [test] (n=115)

- 方法: 原始 flac 双声道 RMS 能量, 阈值 0.1×声道最大, 事件 ≥0.1s
- **双声道同时活跃 (声学重叠)**: 104/115 条有重叠 (90%)
- 重叠总时长: 均值 0.625s | 中位 0.500s | 最大 2.430s | 有重叠样本均值 0.692s
- 重叠事件数(≥0.1s): 均值 2.57 / 条
- **标注-声学失配** (音频有声音但 GT 说话人静音): 均值 2.121s/条
- pyannote 检出重叠 (有结果 115 条): 均值 1.761s vs 声道实测 0.625s

## 逐样本 (前 15 条)

| file_id | dur(s) | 重叠(s) | 事件 | 失配A(s) | 失配B(s) | pyannote(s) |
|---------|--------|---------|------|----------|----------|-------------|
| test_0000000003 | 67.24 | 1.61 | 7 | 0.2 | 3.5 | 5.75 |
| test_0000000010 | 76.07 | 0 | 0 | 0.01 | 0.21 | 0.00 |
| test_0000000012 | 115.52 | 1.93 | 6 | 1.08 | 4.07 | 4.56 |
| test_0000000072 | 79.01 | 1.53 | 5 | 4.56 | 0.03 | 5.21 |
| test_0000000140 | 102.26 | 0 | 0 | 0.61 | 0.07 | 0.00 |
| test_0000000158 | 76.92 | 0.69 | 2 | 2.7 | 0.65 | 1.84 |
| test_0000000230 | 105.61 | 0.05 | 0 | 1.54 | 0.1 | 0.39 |
| test_0000000243 | 95.97 | 0.5 | 1 | 0.88 | 2.1 | 2.78 |
| test_0000000253 | 73.36 | 0.95 | 5 | 0.0 | 2.24 | 2.18 |
| test_0000000277 | 79.39 | 1.3 | 5 | 3.02 | 0.68 | 5.21 |
| test_0000000305 | 88.57 | 0 | 0 | 0.34 | 0.04 | 0.00 |
| test_0000000312 | 42.74 | 0.29 | 1 | 0.4 | 0.96 | 1.01 |
| test_0000000322 | 106.06 | 1.0 | 3 | 3.05 | 0.06 | 3.26 |
| test_0000000365 | 88.25 | 0.95 | 3 | 4.29 | 0.14 | 1.50 |
| test_0000000368 | 130.11 | 0.57 | 2 | 1.3 | 0.5 | 1.18 |

图: ['/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/verification/plot_test_test_0000000003.png', '/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/verification/plot_test_test_0000000010.png', '/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/verification/plot_test_test_0000000012.png', '/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/verification/plot_test_test_0000000072.png', '/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/verification/plot_test_test_0000000140.png']

VERIFY DONE