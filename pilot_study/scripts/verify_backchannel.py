#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 Backchannel 标签样本是否存在声学重叠 (元数据 vs 实际音频)

背景: 元数据 utterance 时间线显示 Backchannel 样本零重叠 (间隙插入),
      但听感疑似有重叠。本脚本用原始 flac 声道级能量客观裁决。

方法: 原始 flac 为 22050Hz 真立体声 (已实测 corr(L,R)≈0, 每声道一个说话人):
  1. 每声道短时 RMS (25ms 窗 / 10ms 步进), 语音活跃阈值 = 0.1 × 该声道最大 RMS
  2. 双声道同时活跃 = 声学重叠 (合并 ≤50ms 间隙的帧, 事件 ≥0.1s 计入)
  3. 与 GT utterance 时间线对照: 每声道"音频活跃但 GT 该说话人未说话"的时长
     (这才是真正的标注-声学失配)
  4. 与 pyannote 检出重叠总量对比 (验证检测器是否其实是对的)

输出:
  real_data/verification/plot_{split}_{file_id}.png   前 5 个样本波形图
  real_data/verification/stats_{split}.json           逐样本统计
  real_data/verification/summary_{split}.md           汇总

用法: python scripts/verify_backchannel.py [validation|test]
"""
import io
import json
import sys
import tarfile
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]
REAL = BASE / "real_data"
SRC = Path("/share/workspace3/shared_dataset/behavior-sd")
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "validation"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "Backchannel"  # 也用于 None 对照
OUT = REAL / "verification"
OUT.mkdir(parents=True, exist_ok=True)

FRAME, HOP = 0.025, 0.010   # 25ms 窗 / 10ms 步进
ACTIVE_RATIO = 0.1          # 活跃阈值 = 0.1 × 声道最大 RMS
MIN_EVENT = 0.1             # 重叠事件最小时长 (与 pilot 判定同口径)
MERGE_GAP = 0.05            # 帧合并间隙


def derive_label(rec):
    total_int = sum(b.get("interruptions", 0) for b in rec.get("behaviors", []))
    total_bc = sum(b.get("backchannels", 0) for b in rec.get("behaviors", []))
    if total_int > 0:
        return "Interruption"
    if total_bc > 0:
        return "Backchannel"
    return "None"


def load_flac(soda_id):
    """从 tar 中读取单条 flac -> (y(L,R), sr)"""
    for tar_path in sorted((SRC / SPLIT).glob("*.tar")):
        try:
            with tarfile.open(tar_path) as tar:
                if f"{soda_id}.flac" not in tar.getnames():
                    continue
                data = tar.extractfile(f"{soda_id}.flac").read()
                y, sr = librosa.load(io.BytesIO(data), sr=None, mono=False)
                return y, sr
        except tarfile.TarError:
            continue
    return None, None


def merge_segments(mask, hop):
    """bool 帧掩码 -> 合并后的 (start,end) 段列表"""
    times = np.where(mask)[0] * hop
    if len(times) == 0:
        return []
    segs = [[times[0], times[0] + hop]]
    for t in times[1:]:
        if t - segs[-1][1] <= MERGE_GAP:
            segs[-1][1] = t + hop
        else:
            segs.append([t, t + hop])
    return [(s, e) for s, e in segs]


def speaker_activity(meta, n_frames, hop):
    """GT: 每帧哪个 speaker 在说话 -> (spk0_mask, spk1_mask)"""
    spk = [np.zeros(n_frames, dtype=bool) for _ in range(2)]
    for u in meta.get("utterances", []):
        s, e = u["start_time"], u["end_time"]
        i0, i1 = int(s / hop), int(e / hop)
        spk[u["speaker_idx"] % 2][i0:min(i1 + 1, n_frames)] = True
    return spk


def analyze(meta):
    """单样本分析 -> 统计 dict"""
    y, sr = load_flac(meta["file_name"].rsplit(".", 1)[0])
    if y is None:
        return None
    L, R = y[0].astype(np.float32), y[1].astype(np.float32)
    rl = librosa.feature.rms(y=L, frame_length=int(FRAME * sr),
                             hop_length=int(HOP * sr))[0]
    rr = librosa.feature.rms(y=R, frame_length=int(FRAME * sr),
                             hop_length=int(HOP * sr))[0]
    n = len(rl)
    hop = HOP

    act_l = rl > ACTIVE_RATIO * rl.max()
    act_r = rr > ACTIVE_RATIO * rr.max()
    both = act_l & act_r

    # 声道 ↔ 说话人映射: 每个说话人说话时段内哪声道能量更高
    spk = speaker_activity(meta, n, hop)
    e_l = rl[spk[0]].sum() if spk[0].any() else 0.0
    e_r = rr[spk[0]].sum() if spk[0].any() else 0.0
    spk0_ch = "L" if e_l >= e_r else "R"

    # 音频活跃但 GT 对应说话人静音 (标注-声学失配)
    if spk0_ch == "L":
        gt_l, gt_r = spk[0], spk[1]
    else:
        gt_l, gt_r = spk[1], spk[0]
    mismatch_l = float((act_l & ~gt_l).sum() * hop)
    mismatch_r = float((act_r & ~gt_r).sum() * hop)

    ov_segs = merge_segments(both, hop)
    ov_total = sum(e - s for s, e in ov_segs)
    ov_events = [seg for seg in ov_segs if seg[1] - seg[0] >= MIN_EVENT]

    # pyannote 检出 (real_data/results/{split}/diarization/{file_id}.json)
    det = {}
    dj = REAL / "results" / SPLIT / "diarization" / f"{meta['file_id']}.json"
    if dj.exists():
        d = json.load(open(dj))
        if "overlaps" in d:
            det["total"] = sum(o.get("duration", 0) for o in d["overlaps"]
                               if o.get("duration", 0) > MIN_EVENT)
            det["events"] = len([o for o in d["overlaps"]
                                 if o.get("duration", 0) > MIN_EVENT])
        else:
            det = {"total": None, "events": None}

    return {
        "file_id": meta["file_id"],
        "duration_s": round(max((u["end_time"] for u in meta.get("utterances", [])),
                               default=0.0), 2),
        "ch_active_total": round(float(act_l.sum() * hop), 2),
        "ch2_active_total": round(float(act_r.sum() * hop), 2),
        "overlap_total_s": round(ov_total, 3),
        "overlap_events_0.1": len(ov_events),
        "mismatch_spk0_s": round(mismatch_l, 3),
        "mismatch_spk1_s": round(mismatch_r, 3),
        "spk0_channel": spk0_ch,
        "pyannote_overlap_s": det.get("total"),
        "pyannote_events": det.get("events"),
    }, {
        "act_l": act_l, "act_r": act_r, "both": both, "rl": rl, "rr": rr,
        "spk": spk, "spk0_ch": spk0_ch, "ov_segs": ov_segs,
    }


def plot_one(meta, a, s, fig_path):
    """波形 + 能量 + 重叠区域 + GT 边界"""
    y, sr = load_flac(meta["file_name"].rsplit(".", 1)[0])
    L, R = y[0], y[1]
    t_wav = np.linspace(0, len(L) / sr, len(L))
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(t_wav, L, color="C0", alpha=0.6, linewidth=0.4, label="Ch L (speaker A)")
    axes[0].plot(t_wav, R, color="C1", alpha=0.6, linewidth=0.4, label="Ch R (speaker B)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right")
    axes[0].set_title(
        f"{meta['file_id']}  [{derive_label(meta)}]  "
        f"重叠 {s['overlap_total_s']:.2f}s / {s['overlap_events_0.1']} 事件 | "
        f"GT 失配 {s['mismatch_spk0_s']:.2f}s/{s['mismatch_spk1_s']:.2f}s")

    axes[1].plot(a["rl"], color="C0", linewidth=1.2, label="RMS Ch L")
    axes[1].plot(a["rr"], color="C1", linewidth=1.2, label="RMS Ch R")
    thr_l = ACTIVE_RATIO * a["rl"].max()
    thr_r = ACTIVE_RATIO * a["rr"].max()
    axes[1].axhline(thr_l, color="C0", ls=":", alpha=0.6)
    axes[1].axhline(thr_r, color="C1", ls=":", alpha=0.6)
    axes[1].set_ylabel("RMS energy")
    axes[1].legend(loc="upper right")

    axes[2].fill_between(np.arange(len(a["rl"])) * HOP, 0, a["rl"],
                         color="C0", alpha=0.35, label="Ch L energy")
    axes[2].fill_between(np.arange(len(a["rr"])) * HOP, 0, a["rr"],
                         color="C1", alpha=0.35, label="Ch R energy")
    for k, (seg_s, seg_e) in enumerate(a["ov_segs"]):
        axes[2].axvspan(seg_s, seg_e, color="red", alpha=0.45,
                        label="Overlap" if k == 0 else None)
    for i, u in enumerate(meta["utterances"]):
        c = "C0" if u["speaker_idx"] % 2 == 0 else "C1"
        axes[2].axvspan(u["start_time"], u["end_time"], color=c, alpha=0.08)
        if i == 0:
            axes[2].plot([], [], color="C0", alpha=0.3, label="GT speaker 0")
            axes[2].plot([], [], color="C1", alpha=0.3, label="GT speaker 1")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Energy")
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)


def main():
    meta_all = json.load(open(REAL / f"metadata_{SPLIT}.json"))
    bc = [m for m in meta_all if derive_label(m) == LABEL]
    print(f"[{SPLIT}] {LABEL} 样本 {len(bc)} 条", flush=True)

    stats, fig_meta = [], []
    for i, m in enumerate(bc, 1):
        r = analyze(m)
        if r is None:
            print(f"  ⚠️ 读取失败: {m['file_id']}", flush=True)
            continue
        s, a = r
        stats.append(s)
        if i <= 5:
            fig_path = OUT / f"plot_{SPLIT}_{m['file_id']}.png"
            plot_one(m, a, s, fig_path)
            fig_meta.append(str(fig_path))
        if i % 20 == 0 or i == len(bc):
            print(f"  {i}/{len(bc)} 完成", flush=True)

    json.dump(stats, open(OUT / f"stats_{SPLIT}_{LABEL}.json", "w"), indent=1)
    n = len(stats)
    ov = [s["overlap_total_s"] for s in stats]
    ev = [s["overlap_events_0.1"] for s in stats]
    ml = [s["mismatch_spk0_s"] for s in stats]
    mr = [s["mismatch_spk1_s"] for s in stats]
    det_ok = [s for s in stats if s["pyannote_overlap_s"] is not None]
    pd_total = [s["pyannote_overlap_s"] for s in det_ok]

    lines = [
        f"# {LABEL} 声学重叠验证 [{SPLIT}] (n={n})",
        "",
        f"- 方法: 原始 flac 双声道 RMS 能量, 阈值 0.1×声道最大, 事件 ≥{MIN_EVENT}s",
        f"- **双声道同时活跃 (声学重叠)**: {sum(s > 0 for s in ov)}/{n} 条有重叠 "
        f"({100 * sum(s > 0 for s in ov) / n:.0f}%)",
        f"- 重叠总时长: 均值 {np.mean(ov):.3f}s | 中位 {np.median(ov):.3f}s | "
        f"最大 {max(ov):.3f}s | 有重叠样本均值 {np.mean([s for s in ov if s > 0]):.3f}s",
        f"- 重叠事件数(≥0.1s): 均值 {np.mean(ev):.2f} / 条",
        f"- **标注-声学失配** (音频有声音但 GT 说话人静音): 均值 {np.mean(ml) + np.mean(mr):.3f}s/条",
        f"- pyannote 检出重叠 (有结果 {len(det_ok)} 条): 均值 {np.mean(pd_total):.3f}s vs 声道实测 {np.mean(ov):.3f}s",
        "",
        "## 逐样本 (前 15 条)",
        "",
        "| file_id | dur(s) | 重叠(s) | 事件 | 失配A(s) | 失配B(s) | pyannote(s) |",
        "|---------|--------|---------|------|----------|----------|-------------|",
    ]
    for s in stats[:15]:
        p = f"{s['pyannote_overlap_s']:.2f}" if s["pyannote_overlap_s"] is not None else "-"
        lines.append(
            f"| {s['file_id']} | {s['duration_s']} | {s['overlap_total_s']} | "
            f"{s['overlap_events_0.1']} | {s['mismatch_spk0_s']} | "
            f"{s['mismatch_spk1_s']} | {p} |")
    lines += ["", f"图: {fig_meta}", "", "VERIFY DONE"]
    (OUT / f"summary_{SPLIT}_{LABEL}.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
