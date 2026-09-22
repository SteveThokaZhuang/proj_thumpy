"""§4.5 检查 3：新会话池的正例率与 e4c 是不是同一个问题？

## 为什么必须先看这个

`2026-09-18_pool_sd.md` §4.4 的 C 方案要用 **100 个全新会话**去复现 ΔF1 > 0。
但 §3.3 那次翻车的教训是：**评估集的流行率能直接翻转结论**
（E2 是 100% 正例集，把「mixnorm 不稳」这个伪影制造了出来）。
换一批会话，若流行率差太多，那 C 就不是「复现」，而是**换了一个问题**。
⇒ 按 §5.8 的教训：**开跑前先报正例率。**

## 口径（与 `ari_g1_eval4_ids.py` 一致）

- 块网格 `t = 0,5,10,…` while `t + CHUNK_S <= tmax`，`CHUNK_S=10, STRIDE=5`
- GT 事件 = `cls == 'BC'` 且 `realized != 'None'`
- 「落在块内」= `require_inside`：`a >= t` 且 `b <= t + CHUNK_S`
- 正例率 = **事件数 / 块数**（与 `g1_eval4_meta.json:prevalence` 同定义）

⚠️ 本脚本用 **mp3 时长**当 `tmax`（见 §4.5 检查 1 的结论），
不是帧轨迹 —— 这样新会话**不需要跑轨迹也能算流行率**。

## 自查（会失败的那种）

`g1_eval4c_ids.txt` 是用**轨迹网格**选的，本脚本用**mp3 网格**重算它的事件数，
应当落在 189 附近（文档 §3/§4 记的 e4c 事件数）。若差很远，说明口径没对齐。

只读盘，零 GPU。用法：
  python scripts/g1_newpool_prevalence.py
  # -> g1_logs/newpool_prevalence.log + results/annotator/g1_newpool_prevalence.json
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_g1_common import ANNOT                       # noqa: E402
from ari_g1_eval4_ids import load_bc_index            # noqa: E402

CANDOR = "/share/workspace3/shared_dataset/CANDOR/files"
OUT = f"{ANNOT}/g1_newpool_prevalence.json"
E4C_IDS = f"{ANNOT}/g1_eval4c_ids.txt"

CHUNK_S = 10.0
STRIDE = 5.0
N_NEW = 100

E4RAW = f"{ANNOT}/e4_raw"


def sessions_sorted():
    """与 ari_e4_infer.py 完全同口径: 36 字符且第 9 位是 '-'。"""
    return sorted(d for d in os.listdir(CANDOR) if len(d) == 36 and d[8] == "-")


def split_sessions():
    """返回 (已用 100, 新 100)。

    ⚠️ 2026-09-22 实测更正：已用的 100 个 = **sorted[0:100]**，
    不是文档 §4.1 一直写的 sorted[1:101]。这里以 `e4_raw/` 目录为准。
    """
    sess = sessions_sorted()
    used = sorted(f[:-5] for f in os.listdir(E4RAW) if f.endswith(".json"))
    start = sess.index(used[-1]) + 1
    new = sess[start:start + N_NEW]
    assert not (set(used) & set(new)), "新旧会话有重叠"
    return used, new, start


def grid_positions(dur):
    """与 eval4_ids 同构的网格: 返回起始时刻列表。"""
    if dur + 1e-9 < CHUNK_S:
        return []
    n = int(np.floor((dur - CHUNK_S) / STRIDE + 1e-9)) + 1
    return [i * STRIDE for i in range(n)]


def scan_session(full, s8, by, cache):
    """扫一个会话的两个声道, 返回 (n_chunks, n_events, n_pos_chunks, per_chunk)。"""
    mp3 = f"{CANDOR}/{full}/processed/{full}.mp3"
    if full not in cache:
        try:
            info = sf.info(mp3)
            cache[full] = float(info.frames) / float(info.samplerate)
        except Exception:                              # noqa: BLE001
            cache[full] = None
    dur = cache[full]
    if dur is None:
        return None
    out = {}
    for ch in ("0", "1"):
        wins = by.get((full, int(ch)), [])
        per = []
        for t in grid_positions(dur):
            inside = [1 for a, b in wins
                      if a >= t - 1e-9 and b <= t + CHUNK_S + 1e-9]
            per.append(len(inside))
        out[ch] = per
    return out


def summarize(per_by_ch):
    """把 {ch: [每块事件数]} 汇总。"""
    flat = [n for per in per_by_ch.values() for n in per]
    n_ch = sum(len(per) for per in per_by_ch.values())
    n_ev = int(sum(flat))
    n_pos = int(sum(1 for n in flat if n))
    return dict(n_chunks=n_ch, n_events=n_ev, n_pos_chunks=n_pos,
                prevalence=n_ev / max(n_ch, 1),
                pos_block_rate=n_pos / max(n_ch, 1))


def per_session_prevalence(per_by_ch):
    """按会话算 (事件/块), 供会话簇 bootstrap 用。"""
    n_ch = sum(len(per) for per in per_by_ch.values())
    n_ev = sum(sum(per) for per in per_by_ch.values())
    return n_ch, n_ev


def cluster_boot(vals, B=4000, seed=0):
    """vals = [(n_ch, n_ev)] 每会话一个; 会话簇 bootstrap 出 prevalence 的 SE。"""
    rng = np.random.default_rng(seed)
    a = np.array(vals, dtype=float)
    n = len(a)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    idx = rng.integers(0, n, size=(B, n))
    c = a[idx, 0].sum(axis=1)
    e = a[idx, 1].sum(axis=1)
    p = np.where(c > 0, e / np.maximum(c, 1), 0.0)
    return float(p.mean()), float(p.std()), float(np.percentile(p, 2.5)), \
        float(np.percentile(p, 97.5))


def main():
    sess = sessions_sorted()
    used, new, start = split_sessions()
    print(f"主库会话 {len(sess)}  已用 {len(used)} (下标 0–{start-1})  "
          f"新取 {len(new)} (下标 {start}–{start+len(new)-1})", flush=True)

    by = load_bc_index()
    print(f"GT 索引: {len(by)} 个 (session, ch)", flush=True)

    # ---- 自查: 用 mp3 网格重算 e4c 的 4800 块 ----
    # ⚠️ id 里只有 **8 字符前缀**, 而 GT 索引按**完整 UUID** 建 ⇒ 必须先映射。
    print("\n" + "=" * 62)
    print("[自查] 用 mp3 网格重算 g1_eval4c_ids.txt 的事件数")
    s8map = {s[:8]: s for s in sess}
    assert len(s8map) == len(sess), "8 字符前缀有碰撞, id 无法唯一还原会话"
    ids = [l.strip() for l in open(E4C_IDS) if l.strip()]
    n_ev_ids = 0
    for cid in ids:
        head, ch, t = cid.rsplit("_", 2)
        full = s8map.get(head)
        assert full is not None, f"id 前缀 {head} 不在主库里"
        t0 = float(t[1:])
        wins = by.get((full, int(ch[2:])), [])
        n_ev_ids += sum(1 for a, b in wins
                        if a >= t0 - 1e-9 and b <= t0 + CHUNK_S + 1e-9)
    print(f"  e4c id 数 {len(ids)}  事件数 {n_ev_ids}  "
          f"正例率 {n_ev_ids/max(len(ids),1):.4%}")
    print(f"  文档记的是 4800 块 / 189 事件 / 3.94%  ⇒ 差 {n_ev_ids - 189}")

    # ---- 两组各扫一遍 ----
    cache, res = {}, {}
    for name, group in (("used100", used), ("new100", new)):
        print(f"\n[扫描] {name} ({len(group)} 会话) ...", flush=True)
        per_sess, per_ch_all = [], {}
        for i, full in enumerate(group):
            s8 = full[:8]
            got = scan_session(full, s8, by, cache)
            if got is None:
                continue
            per_sess.append(per_session_prevalence(got))
            for ch, per in got.items():
                per_ch_all.setdefault(ch, []).extend(per)
            if (i + 1) % 25 == 0:
                print(f"  ... {i+1}/{len(group)}", flush=True)
        s = summarize(per_ch_all)
        mean, sd, lo, hi = cluster_boot(per_sess)
        res[name] = dict(**s, n_sessions=len(per_sess),
                         boot_mean=mean, boot_se=sd, boot_ci=[lo, hi],
                         per_session=per_sess)
        print(f"  -> 块 {s['n_chunks']}  事件 {s['n_events']}  "
              f"正例率(事件/块) {s['prevalence']:.4%}  "
              f"含事件块 {s['pos_block_rate']:.4%}")
        print(f"     会话簇 bootstrap: {mean:.4%} ± {sd:.4%}  "
              f"CI [{lo:.4%}, {hi:.4%}]")

    # ---- 对比 ----
    print("\n" + "=" * 62)
    print("[对比] 会话簇 bootstrap (会话是抽样单元)")
    a, b = res["used100"], res["new100"]
    d = b["boot_mean"] - a["boot_mean"]
    se_d = float(np.hypot(a["boot_se"], b["boot_se"]))
    print(f"  已用 100: {a['boot_mean']:.4%} ± {a['boot_se']:.4%}")
    print(f"  新取 100: {b['boot_mean']:.4%} ± {b['boot_se']:.4%}")
    print(f"  差 = {d:+.4%}   SE(差) = {se_d:.4%}   t = {d/se_d:+.2f}")
    print(f"  ⇒ {'与 0 无异' if abs(d) < 2*se_d else '差异超过 2×SE'}")

    # 事件密度(每块多少个事件) 也报一下
    print(f"\n  事件密度: 已用 {a['n_events']/a['n_chunks']:.4f}   "
          f"新取 {b['n_events']/b['n_chunks']:.4f} 个事件/块")

    json.dump(dict(
        n_sessions_total=len(sess), n_used=len(used), n_new=len(new),
        selfcheck=dict(n_ids=len(ids), n_events_recount=n_ev_ids,
                       documented=189, diff=n_ev_ids - 189),
        used100={k: v for k, v in a.items() if k != "per_session"},
        new100={k: v for k, v in b.items() if k != "per_session"},
        diff=d, se_diff=se_d, t_diff=d / se_d,
    ), open(OUT, "w"), indent=1)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
