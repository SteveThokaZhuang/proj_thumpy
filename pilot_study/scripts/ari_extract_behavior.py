"""ARI 实验: Behavior-SD 声道级事件特征提取 (多 worker 分片).

事件定义 (与 CANDOR 同口径, 声道级; speaker_idx -> 声道: spk0=L, spk1=R,
已验证 96.6% 可分离):
  BC   : 嵌套 backchannels[] 事件 (host utterance 内的 BC, 各带 start/end),
         speaker = 1 - host.speaker_idx (听者发出)
  Int  : 自身在对方说话人 utterance 进行中开始 (剩余重叠 >=0.1s) 且自身
         dur>0.3 的 utterance (打断者一侧), 每文件限 CAP
  None : 与对方说话人 utterance 无 >0.1s 重叠且 dur>0.5 的 utterance, 限 CAP

额外输出 realized_label (声道级 RMS 重叠裁决, 与 verify_backchannel.py 同口径:
25ms 窗/10ms 步进, 阈值 = 0.1x 该声道全文件最大 RMS; 连续双声道活跃段
>0.5s -> Interruption, 0.1-0.5s -> Backchannel, 无 -> None)
用于 Behavior-SD 内部 generation vs realized 标签对照.

用法:
  python ari_extract_behavior.py --worker 0 --n-workers 8 \
      --splits validation,test --out .../behavior_events_w0.csv
"""
import argparse
import csv
import io
import json
import os
import sys
import tarfile
import time

import numpy as np
import pandas as pd
import librosa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_features import event_features, slice_window, SR  # noqa: E402

BEHAVIOR_ROOT = "/share/workspace3/shared_dataset/behavior-sd"

CAP_INT = 30
CAP_NONE = 30

COLS = ["file_id", "event_id", "cls", "realized", "both_active_frac",
        "ch_event", "ch_other", "start", "end", "dur",
        "energy_ratio", "f0_correlation", "f0_slope",
        "spectral_centroid", "voiced_ratio"]


def load_flac(tar, name):
    data = tar.extractfile(name).read()
    y, _ = librosa.load(io.BytesIO(data), sr=SR, mono=False)
    if y.ndim == 1:
        return None
    return np.ascontiguousarray(y)


def realized_overlap(y, start, end, ch_max):
    """窗口内双声道同时活跃的连续段 -> (事件列表, both_active_frac)."""
    a, b = int(start * SR), int(end * SR)
    seg = y[:, a:b]
    if seg.shape[1] == 0:
        return [], 0.0
    win, hop = int(0.025 * SR), int(0.01 * SR)
    frames = []
    for i in range(0, seg.shape[1] - win + 1, hop):
        r0 = np.sqrt(np.mean(seg[0, i:i + win] ** 2))
        r1 = np.sqrt(np.mean(seg[1, i:i + win] ** 2))
        frames.append(r0 > 0.1 * ch_max[0] and r1 > 0.1 * ch_max[1])
    if not frames:
        return [], 0.0
    frames = np.array(frames)
    frac = float(frames.mean())
    # 连续活跃段
    spans = []
    in_span = False
    for i, act in enumerate(frames):
        if act and not in_span:
            s = i * hop / SR
            in_span = True
        elif not act and in_span:
            spans.append((s, i * hop / SR))
            in_span = False
    if in_span:
        spans.append((s, len(frames) * hop / SR))
    return spans, frac


def realized_label(y, start, end, ch_max):
    spans, frac = realized_overlap(y, start, end, ch_max)
    if any(e - s > 0.5 for s, e in spans):
        return "Interruption", frac
    if any(0.1 < e - s <= 0.5 for s, e in spans):
        return "Backchannel", frac
    return "None", frac


def build_events(rec):
    """从官方元数据构造事件列表. rec: metadata JSON 的一条 (file 级)."""
    utts = rec["utterances"]
    events = []

    for u in utts:
        # BC 事件 (全部)
        for i, bc in enumerate(u.get("backchannels", [])):
            dur = bc["end_time"] - bc["start_time"]
            if dur < 0.15 or dur > 20.0:
                continue
            events.append({
                "cls": "BC",
                "speaker_idx": 1 - u["speaker_idx"],
                "start": bc["start_time"], "stop": bc["end_time"],
                "dur": dur,
                "uidx": u["uttr_idx"], "bidx": i,
            })

    # utterance 级重叠判定 (其他说话人的 utterance)
    by_spk = {}
    for u in utts:
        by_spk.setdefault(u["speaker_idx"], []).append(u)

    def other_span_overlap(u, min_overlap=0.1):
        for v in by_spk.get(1 - u["speaker_idx"], []):
            ov = min(u["end_time"], v["end_time"]) - max(u["start_time"], v["start_time"])
            if ov > min_overlap:
                return True
        return False

    def starts_during_other(u, min_left=0.1):
        for v in by_spk.get(1 - u["speaker_idx"], []):
            if v["start_time"] < u["start_time"] < v["end_time"] - min_left:
                return True
        return False

    rng = np.random.default_rng(42)
    for cls, cap in [("Int", CAP_INT), ("None", CAP_NONE)]:
        cand = []
        for u in utts:
            dur = u["end_time"] - u["start_time"]
            if dur > 20.0:
                continue
            if cls == "Int":
                ok = dur > 0.3 and starts_during_other(u)
            else:
                ok = dur > 0.5 and not other_span_overlap(u)
            if ok:
                cand.append(u)
        if len(cand) > cap:
            cand = rng.choice(cand, cap, replace=False).tolist()
        for u in cand:
            events.append({
                "cls": cls,
                "speaker_idx": u["speaker_idx"],
                "start": u["start_time"], "stop": u["end_time"],
                "dur": u["end_time"] - u["start_time"],
                "uidx": u["uttr_idx"], "bidx": -1,
            })

    return events


def process_tar(tar_path, split, worker, n_workers, writer, done_files=()):
    tf = tarfile.open(tar_path)
    names = sorted(n for n in tf.getnames() if n.endswith(".flac"))
    mine = [n for i, n in enumerate(names) if i % n_workers == worker]
    mine = [n for n in mine if f"{split}_{n[:-5]}" not in done_files]
    stats = {"files": 0, "events": 0, "skip": 0, "errors": 0}
    t_start = time.time()
    for j, name in enumerate(mine):
        try:
            json_member = name.replace(".flac", ".json")
            rec = json.loads(tf.extractfile(json_member).read().decode("utf-8"))
            y = load_flac(tf, name)
            if y is None:
                stats["errors"] += 1
                continue
            win = int(0.025 * SR)
            ch_max = []
            for c in range(2):
                n = y.shape[1] // win
                seg = y[c, : n * win].reshape(n, win)
                ch_max.append(float(np.max(np.sqrt(np.mean(seg ** 2, axis=1)))))
            file_id = f"{split}_{name[:-5]}"
            events = build_events(rec)
            stats["files"] += 1
            for i, ev in enumerate(events):
                ch_e = ev["speaker_idx"]
                ch_o = 1 - ch_e
                y_e = slice_window(y, ch_e, ev["start"], ev["stop"])
                y_o = slice_window(y, ch_o, ev["start"], ev["stop"])
                if y_e is None or y_o is None:
                    stats["skip"] += 1
                    continue
                feats = event_features(y_e, y_o)
                rz, frac = realized_label(y, ev["start"], ev["stop"], ch_max)
                writer.writerow({
                    "file_id": file_id, "event_id": f"{file_id}_{i}",
                    "cls": ev["cls"], "realized": rz,
                    "both_active_frac": round(frac, 4),
                    "ch_event": ch_e, "ch_other": ch_o,
                    "start": round(ev["start"], 3), "end": round(ev["stop"], 3),
                    "dur": round(ev["dur"], 3), **feats,
                })
                stats["events"] += 1
        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] <= 5:
                print(f"[w{worker}] ERROR {name}: {repr(e)[:150]}", flush=True)
        if j % 50 == 0:
            print(f"[w{worker}] {os.path.basename(tar_path)} {j}/{len(mine)} "
                  f"{stats} elapsed={time.time()-t_start:.0f}s", flush=True)
    tf.close()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--n-workers", type=int, required=True)
    ap.add_argument("--splits", type=str, default="validation,test")
    ap.add_argument("--tar-limit", type=int, default=0,
                    help="每 split 最多处理的 tar 数 (0=全部)")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # 断点续跑: 跳过输出 CSV 中已完成的 file_id
    done_files = set()
    if os.path.exists(args.out):
        try:
            done_files = set(pd.read_csv(
                args.out, usecols=["file_id"])["file_id"])
        except Exception:
            pass
    new_file = (not os.path.exists(args.out)) or (not done_files)
    f = open(args.out, "a" if not new_file else "w", newline="")
    writer = csv.DictWriter(f, fieldnames=COLS)
    if new_file:
        writer.writeheader()

    for split in args.splits.split(","):
        tars = sorted(os.path.join(BEHAVIOR_ROOT, split, t)
                      for t in os.listdir(f"{BEHAVIOR_ROOT}/{split}")
                      if t.endswith(".tar"))
        if args.tar_limit:
            tars = tars[: args.tar_limit]
        for tp in tars:
            st = process_tar(tp, split, args.worker, args.n_workers,
                             writer, done_files)
            f.flush()
            print(f"[w{args.worker}] DONE {tp}: {st}", flush=True)
    f.close()
    print(f"[w{args.worker}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
