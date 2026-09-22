"""ARI 实验共享特征提取函数 (声道级, 修复 mono 混音问题).

设计要点 (2026-08-23, 修复版):
- 所有特征按**声道**计算: event 声道 = 事件说话人, other 声道 = 对方说话人.
  两个数据集都是真立体声, 每声道一个说话人 (CANDOR: channel_map.json;
  Behavior-SD: speaker_idx -> 声道, 已验证 96.6% 可分离).
- 三类事件 (BC/Int/None) 完全同口径, 杜绝"非 BC 样本特征为常数"的问题.
- 不包含 duration 特征 (与标签定义循环, 见 CANDOR Int 判定规则 duration>0.3).
- 能量比取 log10 (None 事件对方声道近静音 -> 比值极值, log 压缩).
"""
import numpy as np
import librosa
from scipy.signal import resample_poly

SR = 16000
FMIN, FMAX = 50.0, 500.0
FRAME_LEN, HOP_LEN = 2048, 256
MIN_WINDOW = 0.15   # 事件窗口最短 (s), 更短无法可靠提 F0
MAX_WINDOW = 20.0   # 事件窗口最长 (s)
MIN_JOINT_VOICED = 10   # f0_correlation 所需最少联合有声帧
MIN_VOICED_SLOPE = 3    # f0_slope 所需最少有声帧
EPS = 1e-6
# 有声帧能量门控 (绝对阈值): 两个数据集均为响度归一化录音
# (声道 RMS ~0.08-0.13 @16k float32), 0.01 ≈ -20dB, 排除静音/串扰.
# 注意: librosa.yin 对静音帧也常给出 F0 估计, 必须加能量门控.
VOICE_RMS = 0.01

FEATURE_COLS = ["energy_ratio", "f0_correlation", "f0_slope",
                "spectral_centroid", "voiced_ratio"]


def frame_rms(y):
    """与 yin 同帧率的帧 RMS (stride view, 无拷贝)."""
    frames = librosa.util.frame(y, frame_length=FRAME_LEN, hop_length=HOP_LEN)
    return np.sqrt(np.mean(frames ** 2, axis=0))


def f0_contour(y):
    """yin 提取 F0 轮廓. 比 pyin 快 ~3x, 对斜率/相关性特征足够."""
    if len(y) < FRAME_LEN:
        return np.full(0, np.nan)
    return librosa.yin(y, fmin=FMIN, fmax=FMAX, sr=SR,
                       frame_length=FRAME_LEN, hop_length=HOP_LEN)


def voiced_mask(y):
    """有声帧 mask = F0 存在 且 帧能量 > VOICE_RMS."""
    f0 = f0_contour(y)
    rms = frame_rms(y)
    n = min(len(f0), len(rms))
    return (~np.isnan(f0[:n])) & (rms[:n] > VOICE_RMS), f0[:n]


def event_features(y_e, y_o):
    """计算单个事件的声道级声学特征.

    Args:
        y_e: event 说话人声道的事件窗口 (16k)
        y_o: 对方声道同一时间窗口 (16k)

    Returns:
        dict: 5 个特征 (FEATURE_COLS), 无 duration
    """
    rms_e = np.sqrt(np.mean(y_e ** 2))
    rms_o = np.sqrt(np.mean(y_o ** 2))
    # log10 能量比 (dB-ish), clip 防静音声道导致的极值
    energy_ratio = float(np.clip(np.log10(rms_e / (rms_o + EPS)), -8.0, 8.0))

    mask_e, f0_e = voiced_mask(y_e)
    mask_o, f0_o = voiced_mask(y_o)

    # voiced_ratio: event 声道有声帧占比
    voiced_ratio = float(np.mean(mask_e)) if len(mask_e) else 0.0

    # f0_slope: event 声道 F0 随时间的斜率 (Hz/s), 只拟合有声帧
    t = np.arange(len(mask_e)) * (HOP_LEN / SR)
    if mask_e.sum() >= MIN_VOICED_SLOPE:
        f0_slope = float(np.polyfit(t[mask_e], f0_e[mask_e], 1)[0])
    else:
        f0_slope = 0.0

    # f0_correlation: 双方同时有声帧上的 F0 相关性 (BC 音高随动, Levitan/Ward)
    joint = mask_e & mask_o
    if joint.sum() >= MIN_JOINT_VOICED:
        r = np.corrcoef(f0_e[joint], f0_o[joint])[0, 1]
        f0_correlation = 0.0 if (r is None or np.isnan(r)) else float(r)
    else:
        f0_correlation = 0.0

    # spectral_centroid: event 声道频谱质心均值 (Hz)
    spectral_centroid = float(np.mean(
        librosa.feature.spectral_centroid(y=y_e, sr=SR)))

    return {
        "energy_ratio": energy_ratio,
        "f0_correlation": f0_correlation,
        "f0_slope": f0_slope,
        "spectral_centroid": spectral_centroid,
        "voiced_ratio": voiced_ratio,
    }


def slice_window(y, ch, start, end):
    """切片 [start, end) 的声道 ch 音频; 窗口超界/过短/过长返回 None."""
    a, b = int(start * SR), int(end * SR)
    n = y.shape[1]
    if a < 0 or b > n:
        return None
    if (b - a) < int(MIN_WINDOW * SR) or (b - a) > int(MAX_WINDOW * SR):
        return None
    return y[ch, a:b]


def load_event_window(sf, start, end, margin=0.15):
    """从已打开的 SoundFile (48k CANDOR mp3) seek-read 事件窗口并重采样到 16k.

    实测: libsndfile mp3 seek 帧级 (24ms), margin 0.15s 足够; 取重采样后
    窗口中间 dur*SR 样本, 避开滤波暂态与 seek 偏差 (±30ms 不影响特征).
    全量解码 45min 会话需 ~34s, 窗口级读取 ~0.13s/事件.
    """
    sr_native = sf.samplerate
    a = max(0.0, start - margin)
    b = min(sf.frames / sr_native, end + margin)
    if (b - a) < MIN_WINDOW:
        return None
    sf.seek(int(a * sr_native))
    n_req = int((b - a) * sr_native)
    y = sf.read(n_req, dtype="float32", always_2d=True).T  # (2, N)
    if y.shape[1] < int(n_req * 0.95):  # 读到文件尾 (时间戳可能越界)
        return None
    z = resample_poly(y, 1, 3, axis=1).astype(np.float32)  # 48k -> 16k
    n_target = int((end - start) * SR)
    n_mid = z.shape[1] // 2
    half = n_target // 2
    z = z[:, max(0, n_mid - half): n_mid - half + n_target]
    if z.shape[1] < int(MIN_WINDOW * SR):
        return None
    return z
