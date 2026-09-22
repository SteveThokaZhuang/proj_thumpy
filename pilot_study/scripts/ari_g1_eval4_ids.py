"""k=4 扩集: 从自然流行率池里选 1200 块, 且与**所有已用块时间上不重叠**.

## 与 ari_g1_eval2_ids.py (E2) 的三点差别

**1. 不要求「块内至少 1 个事件」→ 自然流行率。**
   E2 额外要求 `inside` 非空, 于是 1600 块的 GT 正例率是 **1.00**(零负例),
   `corr(报的比例, F1)` 被推到 +0.935 —— 这正是 §3.3 那个伪影的来源。
   这里去掉那一条, 正例率回到池子本身的 ~5.8%, 多报要吃 FP。

**2. 排除「时间重叠」, 不只是「id 不同」。**
   块按 5s 步进生成, 而 chunk 长 10s —— 相邻两块**共享 50% 的音频**。
   实测 `g1_ids.txt` 里大量 `t=45,50` `295,300` `320,325,330` 这样的成对块。
   只查 id 不相交会漏掉这一层: 同一个 (session, ch) 上一个 t=450 的训练块
   与 t=455 的评估块, 有 5 秒音频是同一段。
   本脚本对每个候选块, 查同 (session, ch) 下**所有已用块的 t**, 只要存在
   `|Δt| < CHUNK_S` 就丢弃。
   (附注: E2 与旧 300 都没做这层排除, 所以它们的"不相交"只是 id 层面的。
    这一点在报告里如实标注, 不改动既有产物。)

**3. 每个 (session, ch) 内部先随机打散再轮转取样。**
   eval2 的候选是按 t 升序排的, 轮转取样每轮取 `v[idx]` 就等于**每个会话都只取
   最前面几个块** —— 评估集会挤在每个会话的开头。这里用固定 RNG 打散后再轮转,
   块在会话内均匀铺开。

用法 (fd_analysis 环境, 纯表操作不碰 GPU):
  python scripts/ari_g1_eval4_ids.py --n-eval 1200 \
      --used g1_ids.txt,g1_eval2_ids.txt
  # -> g1_eval4_ids.txt + g1_eval4_meta.json

## C 方案 (新 100 会话) 的两个新开关 —— 2026-09-22

`--sessions-from range --start 100 --n-sessions 100`
  扫描 `sorted(os.listdir(CANDOR))[100:200]`, 而不是 `e4_raw/` 里那 100 个。
  新会话**没有帧轨迹**, 所以必须配 `--grid-from mp3`。

`--grid-from mp3`
  块网格由 **mp3 时长**定, 不读 X2-Turn 帧轨迹。
  依据见 `docs/pilot_study/2026-09-18_pool_sd.md` §5.1 (实测 200 对):
    - `tmax(轨迹) − 时长(mp3)` = +0.740 s, **0% 落在 0.1 s 内**;
    - 网格块数 **84% 完全相同**, 其余 32 个**只差末端 1 块**;
    - `MIN_FRAMES=20` 闸门 **0/84,944 一次没拦** ⇒ 该口径下直接跳过 (死代码)。
  ⚠️ **这是构造变更**, 不是"同一构造换个算法": 16% 的 (session,ch) 会少一个末端块。
  好处是**不会**产生越过 mp3 结尾的短块 (轨迹网格下 15% 的末端块会)。

**默认值不变** ⇒ 旧的调用逐位复现旧产物 (跑完请照 §7 的锚点自检核一遍)。
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_g1_common import ANNOT  # noqa: E402

CHUNK_S = 10.0
STRIDE = 5.0
MIN_FRAMES = 20
CANDOR = "/share/workspace3/shared_dataset/CANDOR/files"


def sessions_sorted():
    """主库里全部会话, 字母序。与 `ari_e4_infer.py` 同口径。"""
    return sorted(d for d in os.listdir(CANDOR) if len(d) == 36 and d[8] == "-")


def load_bc_index():
    """(session, ch_event) -> [(start, end)] : 只要 realized != None 的 BC。"""
    import glob

    import pandas as pd
    D = f"{ANNOT}/../analysis/ari"
    ev = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{D}/candor_events_w*.csv"))])
    e1 = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{D}/candor_e1_w*.csv"))])
    m = ev.merge(e1[["event_id", "realized"]], on="event_id",
                 suffixes=("", "_e1"), how="left")
    m = m[(m["cls"] == "BC") & (m["realized"] != "None")]
    by = collections.defaultdict(list)
    for s, ch, a, b in zip(m["session"], m["ch_event"], m["start"], m["end"]):
        by[(str(s), int(ch))].append((float(a), float(b)))
    return by


def parse_id(cid):
    """'0020a0c5_ch0_t1005' -> (session8, '0', 1005.0)。"""
    head, ch, t = cid.rsplit("_", 2)
    return head, ch[2:], float(t[1:])


def key_of(cid):
    """分层用的 key: ('0020a0c5', '0')。与 `g1_pool_null_probe.key_of` 同口径。"""
    head, ch, _ = parse_id(cid)
    return (head, ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=1200)
    ap.add_argument("--used", default=f"{ANNOT}/g1_ids.txt",
                    help="逗号分隔的已用 id 清单; 候选必须与它们**id 及时间**都不重叠")
    ap.add_argument("--out-ids", default=f"{ANNOT}/g1_eval4_ids.txt")
    ap.add_argument("--out-meta", default=f"{ANNOT}/g1_eval4_meta.json")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--sessions-from", choices=("e4_raw", "range"), default="e4_raw",
                    help="e4_raw = 扫 e4_raw/*.json 那 100 个会话 (旧行为, 默认); "
                         "range = 扫主库字母序的 [start, start+n) 区间")
    ap.add_argument("--start", type=int, default=0, help="--sessions-from range 时用")
    ap.add_argument("--n-sessions", type=int, default=100,
                    help="--sessions-from range 时用")
    ap.add_argument("--grid-from", choices=("trace", "mp3"), default="trace",
                    help="trace = 用帧轨迹最后一帧时刻定网格 (旧行为, 默认); "
                         "mp3 = 用 mp3 时长 (省 27 GPU·h, 但是构造变更, 见文件头)")
    ap.add_argument("--split-batches", type=int, default=0,
                    help=">0 时把抽到的块按**连续整轮**切成这么多批, 每批写成 "
                         "<out-ids> 的同目录 `..._b00_ids.txt`。用于量池间 SD "
                         "(见 docs/pilot_study/2026-09-18_pool_sd.md): 每批必须"
                         "是完整轮次、铺满全部 key, 否则批间差会混进会话组成。")
    args = ap.parse_args()

    used = set()
    per_file = {}
    for p in [x.strip() for x in args.used.split(",") if x.strip()]:
        s = set(l.strip() for l in open(p) if l.strip())
        per_file[os.path.basename(p)] = len(s)
        used |= s
    # 已用块按 (session, ch) 归拢 t 值 —— 用来查时间重叠
    used_t = collections.defaultdict(list)
    for cid in used:
        s8, ch, t0 = parse_id(cid)
        used_t[(s8, ch)].append(t0)
    used_t = {k: np.sort(np.array(v)) for k, v in used_t.items()}
    print(f"已用池 {len(used)} 块 {per_file}", flush=True)

    def overlaps_used(s8, ch, t):
        """同 (session,ch) 下任一已用块与本块 [t, t+10) 有交集?"""
        arr = used_t.get((s8, ch))
        if arr is None or not len(arr):
            return False
        i = np.searchsorted(arr, t)
        lo = max(0, i - 1)
        hi = min(len(arr), i + 1)
        return bool(np.any(np.abs(arr[lo:hi] - t) < CHUNK_S - 1e-9))

    by = load_bc_index()
    if args.sessions_from == "e4_raw":
        sess_list = [f[:-5] for f in sorted(os.listdir(f"{ANNOT}/e4_raw"))
                     if f.endswith(".json")]
    else:
        all_s = sessions_sorted()
        sess_list = all_s[args.start:args.start + args.n_sessions]
        assert len(sess_list) == args.n_sessions, \
            f"主库只有 {len(all_s)} 个会话, 取不到 [{args.start}, " \
            f"{args.start + args.n_sessions})"
        # ⚠️ 新会话必须与已用块**所属会话**零重叠, 否则 C 就不是样本外了。
        # 这一条光靠下面那个 id 层面的 assert 拦不住 (新会话本来就不会有旧 id)。
        clash = {s[:8] for s in sess_list} & {s8 for s8, _ in used_t}
        assert not clash, f"新会话与已用块同会话: {sorted(clash)[:5]}"
    # mp3 网格会改变块集。允许对任意会话集用它 (比如"同一批会话两种网格"的对照),
    # 但**不许**落进默认输出名 —— 那会静默改掉已发表产物的口径。
    if args.grid_from == "mp3":
        assert args.out_ids != f"{ANNOT}/g1_eval4_ids.txt", (
            "--grid-from mp3 会改变块集, 不许写进默认的 g1_eval4_ids.txt; "
            "请显式给 --out-ids")
    print(f"扫描 {len(sess_list)} 个会话 (from={args.sessions_from}"
          + (f", 下标 [{args.start},{args.start+len(sess_list)})"
             if args.sessions_from == "range" else "")
          + f"), 网格 from={args.grid_from}; "
          f"BC(realized!=None) 索引 {len(by)} 个 (session,ch)", flush=True)

    rng = np.random.default_rng(args.shuffle_seed)
    per_key = collections.OrderedDict()
    n_scanned = n_offpool = n_adjacent = n_gated = 0
    n_no_trace = 0
    for full in sess_list:
        s8 = full[:8]
        d = None
        if args.grid_from == "trace":
            d = json.load(open(f"{ANNOT}/e4_raw/{full}.json"))
        for ch in ("0", "1"):
            if args.grid_from == "trace":
                fr = d["channels"].get(ch, {}).get("frames") or []
                if not len(fr):
                    n_no_trace += 1
                    continue
                t0a = np.array([f[0] for f in fr])
                tmax = float(t0a[-1])
            else:
                mp3 = f"{CANDOR}/{full}/processed/{full}.mp3"
                info = sf.info(mp3)
                t0a = None
                tmax = float(info.frames) / float(info.samplerate)
            wins_all = by.get((full, int(ch)), [])
            cand = []
            t = 0.0
            while t + CHUNK_S <= tmax:
                cid = f"{s8}_ch{ch}_t{int(t)}"
                n_scanned += 1
                if cid in used:
                    t += STRIDE
                    continue
                n_offpool += 1
                if overlaps_used(s8, ch, t):
                    n_adjacent += 1
                    t += STRIDE
                    continue
                if t0a is not None:
                    fm = (t0a >= t) & (t0a < t + CHUNK_S)
                    if int(fm.sum()) < MIN_FRAMES:
                        n_gated += 1
                        t += STRIDE
                        continue
                # else: mp3 口径没有帧可数 ⇒ 跳过闸门。
                # 依据 §5.1: 该闸门在 84,944 个网格位置上一次没拦过, 是死代码。
                # 完整落在块内的窗口; 这里**不**要求非空 (自然流行率)
                inside = [(a, b) for a, b in wins_all
                          if a >= t - 1e-9 and b <= t + CHUNK_S + 1e-9]
                cand.append((cid, len(inside)))
                t += STRIDE
            if cand:
                cand.sort(key=lambda x: x[0])          # 先定序, 保证打散可复现
                order = rng.permutation(len(cand))     # 再打散, 避免只取会话开头
                per_key[(s8, ch)] = [cand[i] for i in order]

    n_cand = sum(len(v) for v in per_key.values())
    n_ev = sum(n for _, n in (c for v in per_key.values() for c in v))
    n_pos = sum(1 for v in per_key.values() for _, n in v if n)
    print(f"扫描 {n_scanned} 个窗口位置: 池内 {n_scanned - n_offpool}, "
          f"池外但时间重叠被弃 {n_adjacent}, 稀疏闸门拦下 {n_gated}", flush=True)
    if args.grid_from == "mp3":
        print(f"  (mp3 网格: 跳过 MIN_FRAMES 闸门; "
              f"{n_no_trace} 个 (会话,ch) 无帧轨迹 —— 本口径下不需要)", flush=True)
    print(f"候选 {n_cand} 块 / {n_ev} 个事件 (正例率 {n_ev/max(n_cand,1):.3%}, "
          f"含事件块 {n_pos}), 分布在 {len(per_key)} 个 (session,ch)", flush=True)

    picked, picked_round, idx = [], [], {k: 0 for k in per_key}
    rnd = 0
    while len(picked) < args.n_eval:
        progressed = False
        for k, v in per_key.items():
            if len(picked) >= args.n_eval:
                break
            if idx[k] < len(v):
                picked.append(v[idx[k]])
                picked_round.append(rnd)
                idx[k] += 1
                progressed = True
        if not progressed:
            break
        rnd += 1
    n_rounds = rnd

    ids = [c for c, _ in picked]
    n_ev_sel = sum(n for _, n in picked)
    n_pos_sel = sum(1 for _, n in picked if n)
    sess = collections.Counter(c.split("_")[0] for c in ids)
    print(f"\n选定 {len(ids)} 块 / {n_ev_sel} 个事件")
    print(f"  正例率 {n_ev_sel/len(ids):.3%}; 含事件块 {n_pos_sel}/{len(ids)}")
    print(f"  覆盖 {len(sess)} 个会话, 每会话 {min(sess.values())}–"
          f"{max(sess.values())} 块 (中位 {sorted(sess.values())[len(sess)//2]})")
    print(f"  每块事件数分布: "
          f"{dict(sorted(collections.Counter(n for _, n in picked).items()))}")

    overlap = set(ids) & used
    assert not overlap, f"与已用池 id 重叠 {len(overlap)} 块!"
    assert len(set(ids)) == len(ids), "id 有重复"
    for cid in ids:                      # 时间重叠必须为零, 这是本脚本的主要理由
        s8, ch, t0 = parse_id(cid)
        assert not overlaps_used(s8, ch, t0), f"{cid} 与已用块时间重叠"

    # ---------- 切成互不重叠的批 (池间 SD 实验用) ----------
    if args.split_batches:
        J = args.split_batches
        assert args.n_eval % J == 0, "--n-eval 必须能被 --split-batches 整除"
        B = args.n_eval // J
        per_round = collections.Counter(picked_round)
        K = len(per_key)
        # 每一轮都必须**满**(每个 key 各取 1 块)。不满的轮次说明某些 key 先抽干了,
        # 那样的轮次会偏袒还有货的 key ⇒ 批间差里混进"哪几个 key"的成分。
        # 这正是 memory `threshold-must-match-null-model` 那一族: 对照/切分必须与设计同构。
        bad = {r: c for r, c in per_round.items() if c != K}
        assert not bad, (f"有 {len(bad)} 轮不是满轮 (每轮应 {K} 块): "
                         f"{dict(list(bad.items())[:5])} ⇒ 池子对这批块数不够, 加大 --used 或减块数")
        assert n_rounds % J == 0, f"轮数 {n_rounds} 不能被 {J} 整除, 会切出一大一小"
        rpb = n_rounds // J                                # rounds per batch
        batches = [[] for _ in range(J)]
        for cid, r in zip(ids, picked_round):
            batches[r // rpb].append(cid)
        stem = args.out_ids[:-4] if args.out_ids.endswith(".txt") else args.out_ids
        for j, b in enumerate(batches):
            assert len(b) == B, f"批 {j} 有 {len(b)} 块, 应为 {B}"
            # 每批都必须铺满全部 key —— 这是"批间差 = 块差, 不是会话组成差"的前提
            nk = len(set(key_of(c) for c in b))
            assert nk == K, f"批 {j} 只覆盖 {nk}/{K} 个 key"
            with open(f"{stem}_b{j:02d}_ids.txt", "w") as f:
                f.write("\n".join(b) + "\n")
        print(f"\n切成 {J} 批 × {B} 块 (每批 {rpb} 轮, 覆盖全部 {K} 个 key); "
              f"轮数 {n_rounds}, 每轮 {K} 块 ✅")
        print(f"  -> {stem}_b00_ids.txt … {stem}_b{J-1:02d}_ids.txt")

    with open(args.out_ids, "w") as f:
        f.write("\n".join(ids) + "\n")
    json.dump({
        "n_chunks": len(ids), "n_events": n_ev_sel, "n_positive_chunks": n_pos_sel,
        "prevalence": n_ev_sel / len(ids), "n_sessions": len(sess),
        "require_inside": True, "require_events": False,
        "temporal_overlap_excluded": True,
        "chunk_s": CHUNK_S, "stride": STRIDE, "min_frames": MIN_FRAMES,
        # 口径必须落盘: 换了 grid_from 就是换了构造, 事后判读要认这个字段
        "sessions_from": args.sessions_from, "grid_from": args.grid_from,
        "session_range": ([args.start, args.start + len(sess_list)]
                          if args.sessions_from == "range" else None),
        "gate_skipped": args.grid_from == "mp3",
        "n_gated": n_gated,
        "excluded_pools": per_file, "excluded_pool_size": len(used),
        "dropped_temporal_overlap": n_adjacent,
        "candidates_available": n_cand, "candidate_events": n_ev,
        "events_per_chunk": dict(sorted(
            collections.Counter(n for _, n in picked).items())),
        "chunks_per_session": dict(sorted(sess.items())),
        **({"split_batches": args.split_batches,
            "chunks_per_batch": args.n_eval // args.split_batches,
            "rounds_per_batch": n_rounds // args.split_batches,
            "n_keys": len(per_key), "n_rounds": n_rounds}
           if args.split_batches else {}),
    }, open(args.out_meta, "w"), indent=2, ensure_ascii=False)
    print(f"\n-> {args.out_ids}\n-> {args.out_meta}")


if __name__ == "__main__":
    main()
