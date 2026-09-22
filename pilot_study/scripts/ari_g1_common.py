"""G1 消融公用: 观测空间无关的 chunk/评估集工具.

关键不变式: 所有观测空间臂使用**同一批 chunk id、同一批标签、同一冻结评估集**,
因此臂间差异只能归因于音频内容 (观测空间)。
"""
import glob
import json
import os
import random

import numpy as np
import pandas as pd

# soundfile 只在下面两个写音频的函数里用, 所以**延迟导入**: 本模块的其余部分
# (id 清单、真值窗口查询) 是纯表操作, 分析脚本常在没装音频库的环境里跑
# (fd_analysis 没有 soundfile), 顶层导入会让它们连 import 都过不去。

ANNOT = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator"
DEFAULT_OUT = f"{ANNOT}/../analysis/ari"
EVAL_JSON = f"{ANNOT}/f8_eval_set.json"


def load_manifest(npz_dir):
    """{id: meta} from a chunk set."""
    with open(f"{npz_dir}/manifest.jsonl") as f:
        return {json.loads(l)["id"]: json.loads(l) for l in f}


def eval_chunk_ids(total=300):
    """冻结评估集的 chunk id (臂间完全一致)."""
    frozen = json.load(open(EVAL_JSON))
    ids = [os.path.basename(w)[:-4] for (w, _) in frozen]
    return ids[:total]


def load_ids_file(path, total=0):
    """读一份 chunk id 清单 (每行一个); total>0 时截断."""
    ids = [l.strip() for l in open(path) if l.strip()]
    return ids[:total] if total else ids


def make_eval_wavs(npz_dir, out_dir, ids):
    """从某臂的 npz 导出评估 wav; 返回 [(wav, id)]."""
    import soundfile as sf          # 延迟导入, 见文件头
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for cid in ids:
        wav = f"{out_dir}/{cid}.wav"
        if not os.path.exists(wav):
            d = np.load(f"{npz_dir}/{cid}.npz")
            sf.write(wav, d["audio"], 16000)
        out.append((wav, cid))
    return out


_E1_CACHE = {}


def _e1_by_session_ch():
    """(session, ch_event) -> 该组事件行的字典, 惰性构建一次.

    原来这段 concat+merge 写在 bc_windows 里, 而 bc_windows 每个 chunk 调一次:
    16 个 CSV 共 47MB 会被重复解析 300 遍 (实测卡死 >3 分钟且一个 GPU 都不占),
    且每个 chunk 都对整张表做一次布尔掩码 —— O(chunks x rows)。这里一次建好
    并按 (session, ch_event) 预分组, 查询变 O(1)。
    注意 key 统一成 (str, int): keep_default_na=False 会让某些列变成 object,
    不归一化的话 groupby 出来的键可能与 manifest 里的类型对不上。
    """
    if "by" not in _E1_CACHE:
        ev = pd.concat([pd.read_csv(f, keep_default_na=False)
                        for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_events_w*.csv"))])
        e1 = pd.concat([pd.read_csv(f, keep_default_na=False)
                        for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_e1_w*.csv"))])
        e1 = e1.merge(ev[["session", "event_id", "cls", "ch_event", "start", "end"]],
                      on="event_id", suffixes=("", "_ev"))
        by = {}
        for (sess, ch), g in e1.groupby(["session", "ch_event"], sort=False):
            by[(str(sess), int(ch))] = g
        _E1_CACHE["by"] = by
    return _E1_CACHE["by"]


def bc_windows(cid, mlook, chunk_s=10.0, realized_only=True,
               require_inside=False, return_dropped=False):
    """chunk 内的 AWS BC 窗口 -> 相对 chunk 起点的 [(start, end)].

    require_inside (#53, 2026-09-16): 只保留**完整落在块内**的窗口
    (start >= t0 且 end <= t0 + chunk_s)。

    为什么需要它: 真值窗口横跨 chunk 边界时, 声学证据有一大半在**相邻块**里,
    却要求模型在本块内报出时刻 —— 本质不可答。G1 的 26 个真值事件里有 3 个
    (11.5%) 属于这种情况 (§7.5), 它们把 F1 的绝对值和排序都拉偏了。
    两臂评的是同一批块同一批事件, 所以这个过滤**不影响臂间公平性**。

    过滤掉的是"窗口的一部分在块内"的事件, 不是"完全没有" —— 后者本来就不在
    返回值里 (下面的 overlap 判断已排除)。
    """
    m = mlook[cid]
    sub = _e1_by_session_ch().get((str(m["session"]), int(m["ch"])))
    if sub is None:
        return ([], []) if return_dropped else []
    sub = sub[sub["cls"] == "BC"]
    if realized_only:
        sub = sub[sub["realized"] != "None"]
    t0 = m["t0"]
    kept, dropped = [], []
    for _, r in sub.iterrows():
        if not (r["end"] > t0 and r["start"] < t0 + chunk_s):
            continue                      # 与块完全不相交
        rel = (r["start"] - t0, r["end"] - t0)
        if require_inside and not (r["start"] >= t0 - 1e-9
                                   and r["end"] <= t0 + chunk_s + 1e-9):
            dropped.append(rel)           # 相交但横跨边界 -> 不可答
        else:
            kept.append(rel)
    if return_dropped:
        return kept, dropped
    return kept


def select_train_chunks(npz_dir, n_samples, chunk_s=10.0, seed=42,
                        exclude_ids=frozenset()):
    """F8c-v2 同款选取: 事件正例 3/4 + 硬负例 1/8 + 普通负例.

    复用 v2 的口径 (TAU=0.1, 只保留落在 realized!=None 窗口内的事件);
    额外排除冻结评估集, 避免污染。
    """
    TAU = 0.1
    mlook = load_manifest(npz_dir)
    e1 = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_e1_w*.csv"))])
    ev = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_events_w*.csv"))])
    e1 = e1.merge(ev[["session", "event_id", "cls", "ch_event", "start", "end"]],
                  on="event_id", suffixes=("", "_ev"))
    # 预建索引: (session, ch) -> [(start, end, realized)]  (BC 类)
    # 原实现每轮对整张表做 pandas 过滤, 85k chunk 量级下要十几分钟
    bc = e1[e1["cls"] == "BC"]
    wins_by_key = {}
    for s, ch, st, en, rz in zip(bc["session"], bc["ch_event"],
                                 bc["start"], bc["end"], bc["realized"]):
        wins_by_key.setdefault((s, int(ch)), []).append((st, en, rz))
    print(f"indexed {len(bc)} BC windows over {len(wins_by_key)} (session,ch)",
          flush=True)

    manifest = list(mlook.values())
    random.Random(seed).shuffle(manifest)

    x2_cache = {}
    pos, hard_neg, neg = [], [], []
    for n_seen, m in enumerate(manifest):
        if n_seen % 5000 == 0 and n_seen:
            print(f"  scanned {n_seen}: pos {len(pos)} hard_neg "
                  f"{len(hard_neg)} neg {len(neg)}", flush=True)
        if m["id"] in exclude_ids:
            continue
        if m["session"] not in x2_cache:
            try:
                d = json.load(open(f"{ANNOT}/e4_raw/{m['session']}.json"))
            except Exception:
                continue
            fr = d["channels"][str(m["ch"])]["frames"]
            x2_cache[m["session"]] = (
                np.array([f[0] for f in fr]),
                np.array([f[1] for f in fr]),
                np.array([f[3] for f in fr]))
        t0a, t1a, p_bc = x2_cache[m["session"]]
        fm = (t0a >= m["t0"]) & (t1a < m["t0"] + chunk_s)
        if fm.sum() < 50:
            continue
        t = (t0a[fm] + t1a[fm]) / 2
        p = p_bc[fm]
        wins = [w for w in wins_by_key.get((m["session"], int(m["ch"])), [])
                if w[1] > m["t0"] and w[0] < m["t0"] + chunk_s]

        def peak_in_realized(t_peak):
            return any(a <= t_peak <= b and rz != "None" for a, b, rz in wins)

        events, run = [], []
        for i in range(len(t)):
            if p[i] >= TAU:
                run.append(i)
            elif run:
                pk = run[int(np.argmax([p[j] for j in run]))]
                if peak_in_realized(t[pk]):
                    events.append((t[pk], float(p[pk])))
                run = []
        if run:
            pk = run[int(np.argmax([p[j] for j in run]))]
            if peak_in_realized(t[pk]):
                events.append((t[pk], float(p[pk])))
        has_fake = any(rz == "None" for a, b, rz in wins)
        if events:
            pos.append((m, events))
        elif has_fake:
            hard_neg.append(m)
        else:
            neg.append(m)
        if (len(pos) >= n_samples * 3 // 4
                and len(hard_neg) + len(neg) >= n_samples // 4):
            break

    return (pos[: n_samples * 3 // 4]
            + hard_neg[: n_samples // 8]
            + neg[: max(0, n_samples // 4 - len(hard_neg))])


INSTRUCTION = ("When does the listener produce backchannels in this segment? "
               "List the times in seconds with confidence, or answer "
               "'no backchannel'.")


def write_jsonl(chosen, npz_dir, out_dir, chunk_s=10.0, seed=42):
    """chosen -> LLaMA-Factory sharegpt JSONL (音频取自 npz_dir)."""
    import soundfile as sf          # 延迟导入, 见文件头
    wav_dir = f"{out_dir}/wavs"
    os.makedirs(wav_dir, exist_ok=True)
    chosen = list(chosen)
    random.Random(seed).shuffle(chosen)
    rows = []
    for item in chosen:
        if isinstance(item, tuple):
            m, evts = item
            answer = ", ".join(f"{e - m['t0']:.1f}s ({c:.2f})" for e, c in evts)
        else:
            m = item
            answer = "no backchannel"
        d = np.load(f"{npz_dir}/{m['id']}.npz")
        wav = f"{wav_dir}/{m['id']}.wav"
        sf.write(wav, d["audio"], 16000)
        audio_field = json.dumps({
            "path": os.path.realpath(wav), "text": "",
            "token": "<|audio_pad|>" * int(chunk_s * 25),
            "ref_path": "", "ref_text": "",
        }, ensure_ascii=False, sort_keys=True)
        rows.append({
            "system": "",
            "messages": [
                {"role": "user",
                 "content": "<|audio_bos|><|AUDIO|><|audio_eos|>" + INSTRUCTION},
                {"role": "assistant", "content": answer},
            ],
            "audio": audio_field,
        })
    with open(f"{out_dir}/train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_pos = sum(1 for r in rows if r["messages"][1]["content"] != "no backchannel")
    return len(rows), n_pos
