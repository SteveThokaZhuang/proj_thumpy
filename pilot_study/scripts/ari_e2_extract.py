"""E2 提取: FD-Bench 区间规则的标签源敏感性 + 串扰压力测试.

在 Behavior-SD (validation+test) 上, 对每条对话用四种"片段来源"跑同一套
FD-Bench 打断规则 (忠实简化版, 阈值与 L597-851 一致, 16k 样本):
  GT   : metadata utterances 的起止 (完美时间戳)
  VAD  : 每声道 Silero VAD (threshold 0.5, min_silence 1500ms, FD-Bench 同参)
  VAD10: 声道加 10% 串扰 (A'=A+0.1B) 后 VAD
  VAD30: 声道加 30% 串扰后 VAD

规则 (秒, 复刻自 FD-Bench benchmarking.py L597-851 的 16k 样本常量):
  user_interrupt : a_s < u_s < a_e           (用户 AI 说话中途开口)
  success_int    : 上述且 a_e < u_e          (AI 在用户说完前停)
  wrong_int      : u_s < a_s < u_e - 0.5     (AI 抢话, 用户还有 >0.5s)
  noise_int      : u_e < a_e < next_u_s 且 next_u_s - a_e >= 2.5 且 a_e - a_s < 5.5
  指标: n_round=len(U), n_interrupt=len(U)-1, SIR=succ/n_int, EIR=wrong/n_round,
        NIR=noise/n_gap (n_gap = len(U)-1)

另对每个 GT user_interrupt 事件检查 [u_s, u_s+0.5s] 内双声道 RMS 双活跃
(>0.01) -> gt_overlap_frac (真实重叠率).

输出: e2_rows_w{W}.csv 每行 (file_id, source, n_round, n_interrupt, n_succ,
n_wrong, n_noise, sir, eir, nir, gt_overlap_frac[仅 GT 行])
用法: python scripts/ari_e2_extract.py --worker W --n-workers 8 --out .../e2_rows_wW.csv
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
import librosa
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_features import SR  # noqa: E402

BEHAVIOR_ROOT = "/share/workspace3/shared_dataset/behavior-sd"
COLS = ["file_id", "source", "n_round", "n_interrupt", "n_succ", "n_wrong",
        "n_noise", "sir", "eir", "nir", "gt_overlap_frac"]


def fdbench_rules(U, A):
    """FD-Bench 区间规则 (忠实简化). U/A: [(s,e),...] 已排序. 返回计数 dict."""
    U = sorted(U)
    A = sorted(A)
    n_round = len(U)
    n_int = max(0, n_round - 1)
    n_gap = n_int
    n_succ = n_wrong = n_noise = 0
    for a_s, a_e in A:
        # user interrupt: 用户在该 AI 段中途开口
        for u_s, u_e in U:
            if a_s < u_s < a_e:
                if a_e < u_e:
                    n_succ += 1
            # wrong interrupt: AI 抢话 (用户还有 >0.5s)
            if u_s < a_s < u_e - 0.5:
                n_wrong += 1
        # noise interrupt: AI 段结束于轮间空隙 (>=2.5s 距下一用户) 且短于 5.5s
        next_u = [u[0] for u in U if u[0] >= a_e]
        if next_u and next_u[0] - a_e >= 2.5 and (a_e - a_s) < 5.5:
            prev_u = [u[1] for u in U if u[1] <= a_s]
            if prev_u and prev_u[-1] < a_s:
                n_noise += 1
    return dict(n_round=n_round, n_interrupt=n_int, n_succ=n_succ,
                n_wrong=n_wrong, n_noise=n_noise,
                sir=(n_succ / n_int if n_int else float("nan")),
                eir=(n_wrong / n_round if n_round else float("nan")),
                nir=(n_noise / n_gap if n_gap else float("nan")))


def gt_segments(rec):
    """(user=A=spk0, AI=B=spk1) 的 GT 片段."""
    U = [(u["start_time"], u["end_time"]) for u in rec["utterances"]
         if u["speaker_idx"] == 0]
    A = [(u["start_time"], u["end_time"]) for u in rec["utterances"]
         if u["speaker_idx"] == 1]
    return U, A


def gt_overlap_frac(y, U, A):
    """GT user_interrupt 事件在 [u_s, u_s+0.5s] 内双活跃占比."""
    n, n_ov = 0, 0
    win_l = int(0.025 * SR)
    for a_s, a_e in A:
        for u_s, u_e in U:
            if a_s < u_s < a_e:
                n += 1
                i0, i1 = int(u_s * SR), int((u_s + 0.5) * SR)
                if i1 <= y.shape[1]:
                    seg = y[:, i0:i1]
                    both = ((np.sqrt(np.mean(seg ** 2, axis=1)) > 0.01)).all()
                    n_ov += int(both)
    return (n_ov / n) if n else float("nan")


def process_file(tar, split, stem, writer, vad_model, get_ts):
    rec = json.loads(tar.extractfile(f"{stem}.json").read().decode())
    data = tar.extractfile(f"{stem}.flac").read()
    y, _ = librosa.load(io.BytesIO(data), sr=SR, mono=False)
    if y.ndim == 1:
        return
    y = np.ascontiguousarray(y)
    U, A = gt_segments(rec)
    file_id = f"{split}_{stem}"

    def vad(seg_y, ts_fn, model, **kw):
        try:
            out = ts_fn(seg_y, model, sampling_rate=SR, threshold=0.5,
                        min_speech_duration_ms=100,
                        min_silence_duration_ms=1500)
            return [(t["start"] / SR, t["end"] / SR) for t in out]
        except Exception:
            return []

    sources = {}
    sources["GT"] = (U, A)
    sources["VAD"] = (vad(y[0], get_ts, vad_model), vad(y[1], get_ts, vad_model))
    for xt in (0.1, 0.3):
        c0 = y[0] + xt * y[1]
        c1 = y[1] + xt * y[0]
        sources[f"VAD_xt{int(xt*100)}"] = (vad(c0, get_ts, vad_model),
                                           vad(c1, get_ts, vad_model))

    for src, (Uu, Aa) in sources.items():
        m = fdbench_rules(Uu, Aa)
        row = {"file_id": file_id, "source": src,
               "n_round": m["n_round"], "n_interrupt": m["n_interrupt"],
               "n_succ": m["n_succ"], "n_wrong": m["n_wrong"],
               "n_noise": m["n_noise"],
               "sir": round(m["sir"], 4) if m["sir"] == m["sir"] else "",
               "eir": round(m["eir"], 4) if m["eir"] == m["eir"] else "",
               "nir": round(m["nir"], 4) if m["nir"] == m["nir"] else "",
               "gt_overlap_frac": ""}
        if src == "GT":
            row["gt_overlap_frac"] = round(gt_overlap_frac(y, U, A), 4)
        writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--n-workers", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    tars = []
    for split in ["validation", "test"]:
        d = f"{BEHAVIOR_ROOT}/{split}"
        tars += sorted(os.path.join(split, t) for t in os.listdir(d)
                       if t.endswith(".tar"))

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        try:
            with open(args.out, newline="") as f:
                done = {r["file_id"] for r in csv.DictReader(f)}
        except Exception:
            pass
    new_file = (not os.path.exists(args.out)) or (not done)
    f = open(args.out, "a" if not new_file else "w", newline="")
    writer = csv.DictWriter(f, fieldnames=COLS)
    if new_file:
        writer.writeheader()

    print(f"[w{args.worker}] loading silero vad...", flush=True)
    vad_model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad",
                                      trust_repo=True)
    get_ts = utils[0]

    t0 = time.time()
    n_done = 0
    for tp in tars:
        split, tname = tp.split("/", 1)
        tf = tarfile.open(f"{BEHAVIOR_ROOT}/{split}/{tname}")
        names = sorted(n for n in tf.getnames() if n.endswith(".flac"))
        mine = [n for i, n in enumerate(names)
                if i % args.n_workers == args.worker]
        for name in mine:
            stem = name[:-5]
            fid = f"{split}_{stem}"
            if fid in done:
                continue
            try:
                process_file(tf, split, stem, writer, vad_model, get_ts)
            except Exception as e:
                print(f"[w{args.worker}] ERR {name}: {repr(e)[:100]}", flush=True)
            n_done += 1
            f.flush()
            if n_done % 20 == 0:
                print(f"[w{args.worker}] {n_done} files "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
        tf.close()
    f.close()
    print(f"[w{args.worker}] DONE {n_done}", flush=True)


if __name__ == "__main__":
    main()
