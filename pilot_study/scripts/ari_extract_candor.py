"""ARI 实验: CANDOR 声道级事件特征提取 (多 worker 分片).

事件定义 (三类同口径, 声道级):
  BC   : transcript_backbiter.csv 中 backchannel_start 非空的 BC 事件窗口,
         speaker = backchannel_speaker (BC 发出者)
  Int  : transcript_audiophile.csv 中 overlap=True 且 dur>0.3 的 turn,
         排除与任何 BC 窗口时间重合的 turn (避免同一事件进两个类), 每会话限 CAP
  None : audiophile 中 overlap=False 且 dur>0.5 的 turn,
         同样排除与 BC 窗口重合的 turn, 每会话限 CAP
  (overlap=True 且 dur<=0.3 的歧义短重叠不采样)

特征 (ari_features.py): 5 个, 无 duration; 按声道计算 (channel_map.json 定位说话人).

用法:
  python ari_extract_candor.py --worker 0 --n-workers 8 --out .../candor_events_w0.csv
  SMOKE=1 时只处理前 SMOKE 个会话 (冒烟测试).
"""
import argparse
import csv
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf_lib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_features import event_features, load_event_window  # noqa: E402


def _f(x, default=float("nan")):
    """安全转 float; 空串/异常 -> default (NaN)."""
    if x is None or str(x).strip() == "":
        return default
    try:
        return float(x)
    except ValueError:
        return default


def read_transcript(path):
    """读转写 CSV (标准库 csv 模块).

    实测: 1656 个会话中若干转写文件会让 pandas 的 C 引擎 AND python 引擎
    都段错误 (未闭合引号字段 + 类型推断触发, 见 2026-08-24 排查记录),
    且 segfault 无法在进程内捕获. csv 模块为纯 Python 解析, 不受影响.
    """
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

DATA_ROOT = "/share/workspace3/shared_dataset/CANDOR/files"
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

CAP_INT = 30    # 每会话 Int 采样上限
CAP_NONE = 30   # 每会话 None 采样上限
MAXW = 20.0     # 事件窗口最长 (s), 与 ari_features.MAX_WINDOW 一致

COLS = ["session", "event_id", "cls", "ch_event", "ch_other",
        "start", "end", "dur",
        "energy_ratio", "f0_correlation", "f0_slope",
        "spectral_centroid", "voiced_ratio"]


def list_sessions():
    return sorted(d for d in os.listdir(DATA_ROOT) if UUID_RE.match(d))


def build_events(session):
    """构造该会话的事件列表 (尚未提取特征).

    转写解析用标准库 csv 模块 (pandas 对若干畸形转写文件会段错误, 见
    read_transcript 注释). 事件定义与旧版一致:
      BC   : backbiter 的 backchannel_start 非空事件 (dur 0.15-3.0s)
      Int  : audiophile overlap=True 且 dur>0.3, 排除"即 BC 事件本身"的 turn
      None : audiophile overlap=False 且 dur>0.5, 同上排除, Int/None 各限 CAP
    """
    bb = read_transcript(
        f"{DATA_ROOT}/{session}/transcription/transcript_backbiter.csv")
    au = read_transcript(
        f"{DATA_ROOT}/{session}/transcription/transcript_audiophile.csv")
    cm = json.load(open(f"{DATA_ROOT}/{session}/processed/channel_map.json"))
    ch_by_user = {u: (0 if cm.get("L") == u else 1) for u in cm.values()}

    events = []

    # BC 事件 (全部)
    for row in bb:
        bs = _f(row.get("backchannel_start"))
        be = _f(row.get("backchannel_stop"))
        if bs != bs or be != be:  # NaN
            continue
        spk = row.get("backchannel_speaker", "")
        if spk not in ch_by_user:
            continue
        dur = be - bs
        # backbiter 的 BC 窗口有 33% 超过 3s (多 BC 合并标注), 截断保类纯度
        if dur < 0.15 or dur > 3.0:
            continue
        events.append({
            "cls": "BC", "speaker": spk,
            "start": bs, "stop": be, "dur": dur, "session": session,
        })

    # BC 窗口集合 (用于排除 Int/None 采样中"就是该 BC 事件本身"的 turn:
    # 仅当 turn 与 BC 窗口起止均接近 (±0.3s) 才视为同一事件排除;
    # 包含 BC 窗口的长宿主 turn 不排除)
    bc_windows = [(e["start"], e["stop"]) for e in events if e["cls"] == "BC"]

    def is_bc_turn(s, e):
        return any(abs(s - b_s) < 0.3 and abs(e - b_e) < 0.3
                   for b_s, b_e in bc_windows)

    # Int / None (限 CAP)
    for cls, cap in [("Int", CAP_INT), ("None", CAP_NONE)]:
        cand = []
        for row in au:
            s = _f(row.get("start"))
            e = _f(row.get("stop"))
            if s != s or e != e:
                continue
            dur = e - s
            if dur > MAXW:
                continue
            ov = str(row.get("overlap", "")).strip().lower() == "true"
            if cls == "Int":
                ok = ov and dur > 0.3
            else:
                ok = (not ov) and dur > 0.5
            if ok and not is_bc_turn(s, e):
                cand.append({"cls": cls, "speaker": row.get("speaker"),
                             "start": s, "stop": e, "dur": dur,
                             "session": session})
        if len(cand) > cap:
            cand = list(np.random.default_rng(42).choice(
                cand, cap, replace=False))
        events.extend(cand)

    return events, ch_by_user


def process_session(session, writer):
    t0 = time.time()
    try:
        events, ch_by_user = build_events(session)
        if not events:
            return {"session": session, "n": 0, "skipped_load": True}
        mp3 = f"{DATA_ROOT}/{session}/processed/{session}.mp3"
        sf = sf_lib.SoundFile(mp3)
        if sf.channels != 2:  # 防御: 单声道会话跳过
            sf.close()
            return {"session": session, "n": 0, "skipped_mono": True}
        n_ok, n_skip = 0, 0
        for i, ev in enumerate(events):
            ch_e = ch_by_user[ev["speaker"]]
            ch_o = 1 - ch_e
            win = load_event_window(sf, ev["start"], ev["stop"])
            if win is None:
                n_skip += 1
                continue
            feats = event_features(win[ch_e], win[ch_o])
            writer.writerow({
                "session": session, "event_id": f"{session[:8]}_{i}",
                "cls": ev["cls"], "ch_event": ch_e, "ch_other": ch_o,
                "start": round(ev["start"], 3), "end": round(ev["stop"], 3),
                "dur": round(ev["dur"], 3), **feats,
            })
            n_ok += 1
        sf.close()
        return {"session": session, "n": n_ok, "skip": n_skip,
                "t": time.time() - t0}
    except Exception as e:
        return {"session": session, "error": repr(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--n-workers", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    sessions = list_sessions()
    mine = [s for i, s in enumerate(sessions) if i % args.n_workers == args.worker]
    if os.environ.get("SMOKE"):
        mine = mine[: int(os.environ["SMOKE"])]

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # 断点续跑: 跳过输出 CSV 中已完成的会话 (gpu02 为共享节点, 进程可能被扫掉)
    done_sessions = set()
    if os.path.exists(args.out):
        try:
            with open(args.out, newline="") as f:
                done_sessions = {row["session"] for row in csv.DictReader(f)}
        except Exception:
            pass
    mine = [s for s in mine if s not in done_sessions]
    new_file = (not os.path.exists(args.out)) or (not done_sessions)
    f = open(args.out, "a" if not new_file else "w", newline="")
    writer = csv.DictWriter(f, fieldnames=COLS)
    if new_file:
        writer.writeheader()

    t_start = time.time()
    for j, s in enumerate(mine):
        res = process_session(s, writer)
        f.flush()
        if j % 10 == 0 or "error" in res:
            print(f"[w{args.worker}] {j}/{len(mine)} {res} "
                  f"elapsed={time.time()-t_start:.0f}s", flush=True)
    f.close()
    print(f"[w{args.worker}] DONE {len(mine)} sessions", flush=True)


if __name__ == "__main__":
    main()
