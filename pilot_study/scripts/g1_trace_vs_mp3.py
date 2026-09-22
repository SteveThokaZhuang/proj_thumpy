"""§4.5 检查 1：chunk 网格能不能只用 mp3 时长重建，从而省掉帧轨迹？

## 背景

`ari_g1_eval4_ids.py` 里块网格是这么铺的（第 132-137 行）：

    t0a  = np.array([f[0] for f in fr])     # 轨迹的帧时刻
    tmax = float(t0a[-1])                   # 最后一帧的时刻
    t = 0.0
    while t + CHUNK_S <= tmax:              # CHUNK_S = 10.0, STRIDE = 5.0
        cid = f"{s8}_ch{ch}_t{int(t)}"
        ...
        t += STRIDE

⇒ **网格完全由 `tmax` 这一个标量决定**（id 就是 t = 0,5,10,… 的前缀）。
`tmax` 之外，轨迹还供一个 `fm.sum() < MIN_FRAMES` 的稀疏闸门。

若 `tmax` 能由 mp3 时长替代，**帧轨迹（实测 981 s/会话）就可以省掉**。
本脚本量化：两者差多少、会不会改变块数、闸门有没有真的在拦块。

## 口径

与 `ari_g1_eval4_ids.py` **逐字同构**：
- `n_chunks(tmax) = |{t = 0,5,10,… : t + 10 <= tmax}|`
- 块数相同 ⇒ id 列表逐字相同（id 只由 t 决定）

只读盘，零 GPU。用法：
  python scripts/g1_trace_vs_mp3.py
  # -> g1_logs/trace_vs_mp3.log + results/annotator/g1_trace_vs_mp3.json
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_g1_common import ANNOT  # noqa: E402

E4RAW = f"{ANNOT}/e4_raw"
CANDOR = "/share/workspace3/shared_dataset/CANDOR/files"
OUT = f"{ANNOT}/g1_trace_vs_mp3.json"

CHUNK_S = 10.0
STRIDE = 5.0
MIN_FRAMES = 20


def n_chunks(tmax):
    """与 eval4_ids 同构：t = 0,5,10,… while t + CHUNK_S <= tmax。"""
    if tmax + 1e-9 < CHUNK_S:
        return 0
    return int(np.floor((tmax - CHUNK_S) / STRIDE + 1e-9)) + 1


def ids_of(s8, ch, tmax):
    return [f"{s8}_ch{ch}_t{int(t)}"
            for t in np.arange(0.0, n_chunks(tmax) * STRIDE, STRIDE)]


def main():
    rows = []
    miss_mp3 = []
    files = sorted(f for f in os.listdir(E4RAW) if f.endswith(".json"))
    print(f"e4_raw 轨迹 {len(files)} 个会话", flush=True)

    for i, fn in enumerate(files):
        full = fn[:-5]
        s8 = full[:8]
        d = json.load(open(f"{E4RAW}/{fn}"))
        mp3 = f"{CANDOR}/{full}/processed/{full}.mp3"
        if not os.path.exists(mp3):
            miss_mp3.append(full)
            continue
        try:
            info = sf.info(mp3)
            dur = float(info.frames) / float(info.samplerate)
        except Exception as e:                       # noqa: BLE001
            miss_mp3.append(f"{full} ({type(e).__name__})")
            continue

        for ch in ("0", "1"):
            fr = d["channels"].get(ch, {}).get("frames") or []
            if not len(fr):
                continue
            t0a = np.array([f[0] for f in fr], dtype=float)
            tmax_tr = float(t0a[-1])
            n_tr, n_mp = n_chunks(tmax_tr), n_chunks(dur)
            # 稀疏闸门：轨迹里有多少个网格位置帧数不足
            sparse = 0
            for t in np.arange(0.0, n_tr * STRIDE, STRIDE):
                if int(((t0a >= t) & (t0a < t + CHUNK_S)).sum()) < MIN_FRAMES:
                    sparse += 1
            rows.append(dict(
                session=full, s8=s8, ch=ch,
                tmax_trace=tmax_tr, mp3_dur=dur, offset=tmax_tr - dur,
                n_chunks_trace=n_tr, n_chunks_mp3=n_mp,
                n_frames=len(fr),
                fps=len(fr) / max(tmax_tr, 1e-9),
                sparse_chunks=sparse,
            ))
        if (i + 1) % 25 == 0:
            print(f"  ... {i+1}/{len(files)}", flush=True)

    if not rows:
        print("!! 没有可比的 (session,ch)"); return
    off = np.array([r["offset"] for r in rows])
    same = np.array([r["n_chunks_trace"] == r["n_chunks_mp3"] for r in rows])
    sparse_tot = sum(r["sparse_chunks"] for r in rows)
    n_grid = sum(r["n_chunks_trace"] for r in rows)

    print("\n" + "=" * 62)
    print(f"(session,ch) 对: {len(rows)}   缺 mp3: {len(miss_mp3)}")
    print(f"\n[tmax 差值]  轨迹 − mp3时长  (秒)")
    print(f"  mean {off.mean():+.4f}   sd {off.std():.4f}   "
          f"min {off.min():+.4f}   max {off.max():+.4f}")
    print(f"  |offset| ≤ 0.1s 的占比: {(np.abs(off) <= 0.1).mean():.1%}")
    print(f"  offset 恒定的猜测: 分钟 {off.min()/60:.3f} / 最大 {off.max()/60:.3f}"
          f"   帧率 mean {np.mean([r['fps'] for r in rows]):.4f} fps")

    print(f"\n[块数是否相同]  {same.sum()}/{len(same)} = {same.mean():.1%}")
    if not same.all():
        diff = [(r['s8'], r['ch'], r['n_chunks_trace'], r['n_chunks_mp3'], r['offset'])
                for r in rows if r['n_chunks_trace'] != r['n_chunks_mp3']]
        print(f"  不同的 {len(diff)} 对（前 10）：")
        for s8, ch, a, b, o in diff[:10]:
            print(f"    {s8} ch{ch}: 轨迹 {a} 块 vs mp3 {b} 块  (offset {o:+.3f}s)")
        d = np.array([b - a for _, _, a, b, _ in diff])
        print(f"  块数差: mean {d.mean():+.3f}  min {d.min()}  max {d.max()}")

    print(f"\n[稀疏闸门 MIN_FRAMES={MIN_FRAMES}]")
    print(f"  网格位置总数 {n_grid}，其中帧数不足被拦 {sparse_tot} "
          f"({sparse_tot/max(n_grid,1):.4%})")
    per = [r["sparse_chunks"] for r in rows]
    print(f"  有被拦的 (session,ch) 数: {sum(1 for x in per if x)}/{len(per)}")

    print(f"\n[每会话块数]  ", end="")
    nc = np.array([r["n_chunks_trace"] for r in rows])
    print(f"min {nc.min()}  median {int(np.median(nc))}  max {nc.max()}")

    if miss_mp3:
        print(f"\n[缺 mp3] {len(miss_mp3)}: {miss_mp3[:10]}")

    json.dump(dict(
        n_pairs=len(rows), miss_mp3=miss_mp3,
        offset_mean=float(off.mean()), offset_sd=float(off.std()),
        offset_min=float(off.min()), offset_max=float(off.max()),
        frac_within_0p1s=float((np.abs(off) <= 0.1).mean()),
        n_same_chunkcount=int(same.sum()), n_pairs_total=len(same),
        sparse_total=int(sparse_tot), n_grid_total=int(n_grid),
        fps_mean=float(np.mean([r["fps"] for r in rows])),
        n_chunks_min=int(nc.min()), n_chunks_median=int(np.median(nc)),
        n_chunks_max=int(nc.max()),
        rows=rows,
    ), open(OUT, "w"), indent=1)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
