"""G1 定向构建: 只造指定 chunk id 的观测空间音频.

为什么不用全量: 全量 prep 要解 100 个 mp3 并对整段做重采样 (~45 min);
但 build 实际只用 2400 个 chunk。按 id 定向构建只需解 mp3 + 重采样这些片段
(~10 min), 且**天然保证各观测空间臂的 chunk id 逐条相同**。

两个正确性要点:
  1. **先整段重采样再切片** (不是切片再重采样): 与 ari_f8_prep_data.py 的
     `resample_poly(y[:, ch], 1, 3)` 完全同路径, 因此 own 臂的音频与
     f8_training 逐位一致 —— 已用 687 个同 id chunk 验证 max|diff| = 0.0。
  2. 多模式一趟解码: mp3 解码是主要开销, `--input-mode own,mixnorm` 只解一次。

用法 (fd_analysis):
  python scripts/ari_g1_prep_subset.py --ids g1_ids.txt \
      --input-mode mixnorm --out .../g1_sub
  # -> .../g1_sub/mixnorm/{cid}.npz  +  .../g1_sub/mixnorm/manifest.jsonl
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_g1_prep import make_audio  # noqa: E402

CANDOR = "/share/workspace3/shared_dataset/CANDOR/files"
ANNOT = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator"
FRAME_S = 0.08
W = 0.7


def parse_id(cid):
    """'0020a0c5_ch0_t1005' -> (session8, ch, t0)."""
    head, ch, t = cid.rsplit("_", 2)
    return head, ch[2:], float(t[1:])


def nearest_idx(sorted_ref, q):
    """sorted_ref 已排序; 返回每个 q 的最近邻下标, O(m log n).

    等价于原实现 `np.abs(ref[:,None] - q[None,:]).argmin(axis=0)`, 但后者会
    构造 (n_seg x n_frame) 的稠密矩阵 —— 45 分钟的 CANDOR 会话有 ~34k 帧,
    一次就是 800MB / 1.5s, 且每 chunk 重算一遍, 是最大的性能坑。
    """
    j = np.searchsorted(sorted_ref, q)
    j = np.clip(j, 1, len(sorted_ref) - 1)
    left, right = sorted_ref[j - 1], sorted_ref[j]
    return np.where(q - left <= right - q, j - 1, j)


def session_channel_feats(d, full, ch, soulx, soulx_sessions):
    """会话级量: 每个 chunk 都要用, 但**与 t0 无关**, 因此只算一次.

    返回 (t_frame, p_bc, p_sp, fused)。原实现把这段放在 chunk 循环里,
    每 chunk 重算一次 fused 匹配, 实测占 2.58s/chunk 中的绝大部分。
    """
    fr = d["channels"][ch]["frames"]
    t0a = np.array([f[0] for f in fr])
    p_bc = np.array([f[3] for f in fr])
    p_sp = np.array([f[4] for f in fr])
    has_fused = full in soulx_sessions and ch in soulx.get(full, {})
    if not has_fused:
        return t0a, p_bc, p_sp, np.zeros(len(t0a), dtype=np.float32), False
    st0, st1, sp = soulx[full][ch]
    sct = (st0 + st1) / 2
    order = np.argsort(sct)          # searchsorted 要求参考序列有序
    sct, sp = sct[order], sp[order]
    idx = nearest_idx(sct, t0a + FRAME_S / 2)
    return t0a, p_bc, p_sp, W * p_bc + (1 - W) * sp[idx], True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="每行一个 chunk id")
    ap.add_argument("--input-mode", default="mixnorm",
                    help="逗号分隔, 如 own,mix,mixnorm")
    ap.add_argument("--out", required=True, help="基目录, npz 落在 {out}/{mode}/")
    ap.add_argument("--chunk-s", type=float, default=10.0)
    ap.add_argument("--limit-sessions", type=int, default=0,
                    help=">0 时只处理前 N 个 session (调试用)")
    ap.add_argument("--no-states", action="store_true",
                    help="不读帧轨迹/soulx, npz 只写 audio。评估路径只读 audio "
                         "(make_eval_wavs -> predict_times), states/fused 是死的; "
                         "C 方案用它省掉 27 GPU·h 的帧轨迹 (见 pool_sd §5.1)。"
                         "⚠️ 副作用: manifest 少 n_frames/has_fused 两个字段")
    args = ap.parse_args()

    modes = [m.strip() for m in args.input_mode.split(",") if m.strip()]
    ids = [l.strip() for l in open(args.ids) if l.strip()]
    by_sess = defaultdict(list)
    for cid in ids:
        s8, ch, t0 = parse_id(cid)
        by_sess[s8].append((cid, ch, t0))
    if args.limit_sessions:
        keep = sorted(by_sess)[: args.limit_sessions]
        by_sess = {s: by_sess[s] for s in keep}
    n_tot = sum(len(v) for v in by_sess.values())
    print(f"{n_tot} ids over {len(by_sess)} sessions, modes={modes}", flush=True)

    for m in modes:
        os.makedirs(f"{args.out}/{m}", exist_ok=True)

    # ---- 会话全名映射: 用**主库目录**建, 不用 e4_raw ----
    # ⚠️ 2026-09-22 改。原实现只从 `e4_raw/*.json` 建这张表, 于是**没有帧轨迹的
    # 新会话会全部落到 `full is None` 被静默跳过** (打印一行"无 e4_raw, 跳过"),
    # 而 C 方案的全部 100 个会话都属于这一类 ⇒ 会得到一个空的 npz 集。
    # 主库目录是完备的, 且 id 里只有 8 字符前缀, 所以这里必须查碰撞。
    all_sess = [d for d in os.listdir(CANDOR) if len(d) == 36 and d[8] == "-"]
    s8_to_full = {}
    for s in all_sess:
        prev = s8_to_full.setdefault(s[:8], s)
        assert prev == s, f"8 字符前缀碰撞: {prev} vs {s}"
    x2_files = {f[:-5]: f"{ANNOT}/e4_raw/{f}"
                for f in os.listdir(f"{ANNOT}/e4_raw") if f.endswith(".json")}
    print(f"主库 {len(all_sess)} 会话 (有帧轨迹 {len(x2_files)})", flush=True)

    soulx, soulx_sessions = {}, set()
    if not args.no_states:
        from ari_f8_prep_data import load_soulx
        mani = json.load(open(f"{ANNOT}/e5_regions_manifest.json"))
        soulx = load_soulx(mani)
        soulx_sessions = set(m["session"] for m in mani.values())

    # 已写过的 manifest 行 (续跑时不重复写; 也用于补写被打断会话缺的行)
    written = set()
    for m in modes:
        p = f"{args.out}/{m}/manifest.jsonl"
        if os.path.exists(p):
            for line in open(p):
                written.add((m, json.loads(line)["id"]))
    handles = {m: open(f"{args.out}/{m}/manifest.jsonl", "a") for m in modes}

    n_done, n_skip = 0, 0
    for s8, items in sorted(by_sess.items()):
        full = s8_to_full.get(s8)
        if full is None:
            print(f"{s8}: 不在主库里, 跳过", flush=True)
            continue
        # 断点续跑: 全模式都已落盘则跳过音频合成 (srun 步骤可能被清理)。
        # 但 manifest 行仍要补 —— 被打断的会话常常是 npz 写了、行没写。
        todo = {cid for cid, ch, t0 in items
                if not all(os.path.exists(f"{args.out}/{m}/{cid}.npz")
                           for m in modes)}
        n_skip += len(items) - len(todo)
        need_audio = bool(todo)

        chans = None
        if need_audio:
            mp3 = f"{CANDOR}/{full}/processed/{full}.mp3"
            y, sr = sf.read(mp3, dtype="float32", always_2d=True)
            g = np.gcd(16000, int(sr))     # CANDOR 均为 48k; 非整数比时退化
            chans = {ch: resample_poly(y[:, int(ch)], 16000 // g, int(sr) // g)
                     .astype(np.float32) for ch in ("0", "1")}
            del y

        if args.no_states:
            feats = None
        else:
            d = json.load(open(x2_files[full]))
            # 会话级量按信道预计算一次 (与 t0 无关), chunk 循环里只做切片
            feats = {ch: session_channel_feats(d, full, ch, soulx, soulx_sessions)
                     for ch in sorted({c for _, c, _ in items})}

        for cid, ch, t0 in items:
            if args.no_states:
                meta = {"id": cid, "session": full, "ch": ch, "t0": t0}
            else:
                t0a, p_bc, p_sp, fused, has_fused = feats[ch]
                fm = (t0a >= t0) & (t0a < t0 + args.chunk_s)
                states = np.stack([1 - p_sp[fm] - p_bc[fm],
                                   p_sp[fm], p_bc[fm]], axis=1).astype(np.float32)
                meta = {"id": cid, "session": full, "ch": ch, "t0": t0,
                        "n_frames": int(fm.sum()), "has_fused": has_fused}

            if cid in todo:
                other = "1" if ch == "0" else "0"
                i0, i1 = int(t0 * 16000), int((t0 + args.chunk_s) * 16000)
                own, part = chans[ch][i0:i1], chans[other][i0:i1]
                for m in modes:
                    audio = make_audio(own, part, m)
                    if args.no_states:
                        np.savez_compressed(f"{args.out}/{m}/{cid}.npz", audio=audio)
                    else:
                        np.savez_compressed(f"{args.out}/{m}/{cid}.npz",
                                            audio=audio, states=states,
                                            fused=fused[fm])
            for m in modes:
                if (m, cid) not in written:
                    handles[m].write(json.dumps({**meta, "input_mode": m}) + "\n")
                    written.add((m, cid))
            n_done += 1
        for m in modes:
            handles[m].flush()
        print(f"{s8}: {len(items)} chunks (total {n_done}, 续跑跳过 {n_skip})",
              flush=True)

    for m in modes:
        handles[m].close()
    print(f"modes={modes} total {n_done} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
