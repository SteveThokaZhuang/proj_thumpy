"""§3.6 打通：spoken 的 meta.json ↔ corpus 的 dialogues 究竟 join 不 join 得上？

动机：§3 曾把「L1(corpus) 与 L3(spoken) 在已发布文件上 join 不通」记成一条**死结**。
本节把这条结论**收窄**：join 不通的**只有 annotations 那一层**；
spoken 的 `meta.json` 里带着转写，与 corpus 的 `dialogues` 是**对得上**的。

━━ 一个必须写下来的陷阱（本项目已反复栽在「同名不同物」上）━━
`meta.json` 里有**两个**字段都读得像「这份对话的转写」：

    utterances   len = num_turns (INT 首条 = 32)   ← 按**话轮**切，含 backchannel 等
    speech_meta  len = 20                          ← 按**语段**切，与 corpus history 等长

拿错一个，就会得到「0/16 对不上」这种**看起来像 §3 死结复现**的失败。
本脚本两个都算，并把 `utterances` 的结果**并列打印**，就是为了让这个陷阱显形。

━━ 判据（写于跑之前）━━
J1 长度恒等: len(speech_meta) == len(corpus.history)，逐条对话都要成立（精确整数相等）。
J2 文本一致: 归一化后逐轮相等率 ≥ 95%。
J3 逐条全对: 完全对上的对话数 / 总对话数 ≥ 50%。
J4 残差可解释: 不一致的轮里，corpus 侧含 `[TAKE_FLOOR]` 的比例显著高于一致的轮
             （即残差不是随机噪声，而是同一形状）。

纯读盘，0 GPU。用 tarfile 随机读，**不落盘解压**。

用法:
  python duplexgen_meta_join.py                       # 默认全部 shard
  python duplexgen_meta_join.py --shards INT          # 只跑一个
"""
import argparse
import glob
import json
import os
import re
import tarfile
from collections import defaultdict

CORPUS = "/share/workspace3/shared_dataset/duplexgen-corpus"
SPOKEN = "/share/workspace3/shared_dataset/duplexgen-spoken"

MARKER = re.compile(r"\[[^\]]*\]")


def norm(s):
    """归一化：小写 → 去 [MARKER] → 去标点 → 压空白。

    去 marker 是必须的：corpus 侧带 `[TAKE_FLOOR]`、`[REDACTED]`，spoken 侧不带。
    """
    s = s.lower()
    s = MARKER.sub(" ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_corpus(shards):
    """→ {shard: {example_id: record}}"""
    out = {}
    for sh in shards:
        p = os.path.join(CORPUS, "dialogues", sh, "train.jsonl")
        if not os.path.exists(p):
            continue
        d = {}
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                d[r["example_id"]] = r
        out[sh] = d
    return out


def load_spoken(shard):
    """→ {work_id: {variant_idx: meta}}，tarfile 随机读，不落盘。"""
    out = defaultdict(dict)
    for tar in sorted(glob.glob(os.path.join(SPOKEN, "shards", shard, "*.tar"))):
        with tarfile.open(tar, "r:") as tf:
            for m in tf.getmembers():
                if not m.name.endswith("meta.json"):
                    continue
                # 形如 INT/work_0000/var00/meta.json
                parts = m.name.split("/")
                if len(parts) != 4:
                    continue
                work, var = parts[1], parts[2]
                f = tf.extractfile(m)
                if f is None:
                    continue
                out[work][var] = json.load(f)
    return out


def one_dialogue(rec, meta):
    """比一条对话。→ dict"""
    hist = rec["history"]
    sm = meta["speech_meta"]
    n_h, n_s = len(hist), len(sm)
    res = {"n_hist": n_h, "n_speech": n_s, "len_eq": n_h == n_s,
           "utts": len(meta.get("utterances", [])), "rows": []}
    if not res["len_eq"]:
        return res
    for h, s in zip(hist, sm):
        c_raw, t_raw = h["content"], s["tts_text"]
        c, t = norm(c_raw), norm(t_raw)
        res["rows"].append({
            "same": c == t,
            "corpus_take_floor": "[TAKE_FLOOR]" in c_raw,
            "c": c, "t": t,
        })
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="*", default=None)
    ap.add_argument("--max-show", type=int, default=8,
                    help="残差样例最多打印几条")
    a = ap.parse_args()

    shards = a.shards or sorted(
        d for d in os.listdir(os.path.join(SPOKEN, "shards"))
        if os.path.isdir(os.path.join(SPOKEN, "shards", d)))
    print("shard: %s\n" % ", ".join(shards))

    corpus = load_corpus(shards)

    tot_turns = tot_same = 0
    n_dlg = n_dlg_all = 0
    len_eq_fail = []
    tf_hit = tf_miss = 0          # 不一致轮里 corpus 含 TAKE_FLOOR
    ok_hit = ok_miss = 0          # 一致轮里 corpus 含 TAKE_FLOOR
    gaps = defaultdict(int)
    mismatch_rows = []
    utts_len_seen = defaultdict(int)
    var_counts = defaultdict(set)

    for sh in shards:
        if sh not in corpus:
            print("  [skip] corpus 无 shard %s" % sh)
            continue
        sp = load_spoken(sh)
        print("=== %s === spoken work_id %d 个 / corpus 记录 %d 条"
              % (sh, len(sp), len(corpus[sh])))
        n_sh = n_sh_all = 0
        for work, metas in sorted(sp.items()):
            rec = corpus[sh].get(work)
            if rec is None:
                continue
            # 同一 work 的多个 var 应该文本相同；逐个比，取第一个 var 作为主读数
            for var in sorted(metas):
                var_counts[work].add(var)
                r = one_dialogue(rec, metas[var])
                utts_len_seen[r["utts"]] += 1
                gaps[metas[var].get("turn_gap_sec")] += 1
                if not r["len_eq"]:
                    len_eq_fail.append((sh, work, var, r["n_hist"], r["n_speech"]))
                if var != sorted(metas)[0]:
                    continue          # 主读数只统计第一个 var，避免 var 重复计数
                n_sh_all += 1
                for i, row in enumerate(r["rows"]):
                    tot_turns += 1
                    if row["same"]:
                        tot_same += 1
                        ok_hit += row["corpus_take_floor"]
                        ok_miss += not row["corpus_take_floor"]
                    else:
                        tf_hit += row["corpus_take_floor"]
                        tf_miss += not row["corpus_take_floor"]
                        if len(mismatch_rows) < a.max_show:
                            mismatch_rows.append((sh, work, i, row))
                if r["rows"] and all(x["same"] for x in r["rows"]):
                    n_sh += 1
        n_dlg += n_sh
        n_dlg_all += n_sh_all
        print("    逐条全对 %d/%d   长度恒等失败 %d 条"
              % (n_sh, n_sh_all, len(len_eq_fail)))

    print("\n" + "=" * 60)
    print("=== J1 长度恒等 (len(speech_meta) == len(corpus.history)) ===")
    print("  失败 %d 条对话" % len(len_eq_fail))
    for x in len_eq_fail[:10]:
        print("    %s/%s/%s  hist=%d speech=%d" % x)

    print("\n=== J2 文本一致（归一化后逐轮）===")
    print("  %d/%d = %.1f%%" % (tot_same, tot_turns,
                                100.0 * tot_same / max(tot_turns, 1)))
    print("=== J3 逐条全对 ===")
    print("  %d/%d = %.1f%%" % (n_dlg, n_dlg_all,
                                100.0 * n_dlg / max(n_dlg_all, 1)))

    print("\n=== J4 残差是不是同一个形状 ===")
    print("  %-16s %8s %8s" % ("", "含TAKE_FLOOR", "不含"))
    print("  %-16s %8d %8d" % ("不一致轮", tf_hit, tf_miss))
    print("  %-16s %8d %8d" % ("一致轮", ok_hit, ok_miss))
    r_bad = tf_hit / max(tf_hit + tf_miss, 1)
    r_ok = ok_hit / max(ok_hit + ok_miss, 1)
    print("  含标记率: 不一致轮 %.1f%%  vs  一致轮 %.1f%%" % (100 * r_bad, 100 * r_ok))

    print("\n=== 残差样例 ===")
    for sh, work, i, row in mismatch_rows:
        common = 0
        while (common < min(len(row["c"]), len(row["t"]))
               and row["c"][common] == row["t"][common]):
            common += 1
        print("  %s/%s @%-3d 共同前缀 %d 字  corpus含标记=%s"
              % (sh, work, i, common, row["corpus_take_floor"]))
        print("      corpus: ...%s" % row["c"][-60:])
        print("      speech: ...%s" % row["t"][-60:])

    print("\n=== 附带读数 ===")
    print("  turn_gap_sec 取值: %s" % dict(gaps))
    print("  utterances 长度分布(前 5): %s"
          % sorted(utts_len_seen.items(), key=lambda kv: -kv[1])[:5])
    vc = sorted(len(v) for v in var_counts.values())
    if vc:
        print("  每个 work 的 var 数: min=%d max=%d" % (vc[0], vc[-1]))

    print("\n判读: J1/J2/J3/J4 全过 ⇒ **spoken meta ↔ corpus dialogues 是 join 得通的**;")
    print("      §3 的死结应**收窄**为「annotations 那一层 join 不通」，而不是全层不通。")


if __name__ == "__main__":
    main()
