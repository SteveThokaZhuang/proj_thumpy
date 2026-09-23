"""§3.6 三源对照：corpus `dialogues` / spoken `meta.json` / corpus `annotations`。

`duplexgen_meta_join.py` 已经确立：**corpus ↔ spoken.speech_meta 是 join 得通的**
（INT 917 条，长度恒等 917/917，逐轮一致 98.8%，残差 208/208 全是 `[TAKE_FLOOR]`）。
本脚本回答剩下的两问：

  ① 那个「拿错字段」的陷阱到底会伪造出什么？（`utterances` vs `speech_meta`）
  ② `annotations` 那一层对得上谁？（对不上 corpus，那对得上 spoken 吗？）

━━ 为什么单独一个脚本 ━━
因为 `var00` 不是每个 work 都有（实测 `work_0502` 只有 var01/var09）。
上一版按 `variant=="var00"` 筛，会得到「只有 25/50 有 spoken」这种**看起来像缺口、
其实是自己筛出来的**数字。本脚本按「**任意变体**」算，并把 var00 命中率**并列打印**，
让这个自造缺口显形 —— 与 `tiebreak-default-read-as-evidence` 同族：
一致性陈述先问「是不是我的代码在替我选」。

━━ 判据（写于跑之前）━━
K1 覆盖盘点（**不设阈值**，只报数）：50 个被标注 id 中，有 spoken 变体的比例。
K2 `annotations` 首轮文本 ⊄ corpus 全文 ⇒ 判**不通**；一致率必须报出具体数。
K3 `annotations` 首轮文本 ⊄ spoken 全文 ⇒ 同上。
K4 **陷阱量化**：拿 `utterances` 当转写去比 corpus，一致率应 ≈ 0 ⇒
   证明「同一个 meta.json 里两个字段都像转写，拿错就伪造出 join 失败」。

纯读盘，0 GPU。用 metadata.jsonl 定位 tar（比扫全库快，且不会静默漏）。
用法:
  python duplexgen_three_source_join.py --shard INT
"""
import argparse
import json
import os
import re
import tarfile
from collections import defaultdict

CORPUS = "/share/workspace3/shared_dataset/duplexgen-corpus"
SPOKEN = "/share/workspace3/shared_dataset/duplexgen-spoken"
MARKER = re.compile(r"\[[^\]]*\]")


def norm(s):
    s = MARKER.sub(" ", s.lower())
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def read_metas(shard, want=None, variant=None):
    """→ {example_id: {variant: meta}}。用 metadata.jsonl 定位，按 tar 分组，每个 tar 只开一次。"""
    key2tar, key2var = {}, defaultdict(list)
    with open(f"{SPOKEN}/metadata.jsonl") as f:
        for line in f:
            x = json.loads(line)
            if x["scenario"] != shard:
                continue
            if want is not None and x["example_id"] not in want:
                continue
            if variant is not None and x["variant"] != variant:
                continue
            key2tar[x["example_id"]] = x["shard"]
            key2var[x["shard"]].append((x["example_id"], x["variant"]))

    out = defaultdict(dict)
    for tar, items in key2var.items():
        needs = defaultdict(set)
        for eid, var in items:
            needs[var].add(eid)
        with tarfile.open(f"{SPOKEN}/{tar}", "r:") as tf:
            for m in tf.getmembers():
                p = m.name.split("/")
                if len(p) != 4 or not m.name.endswith("meta.json"):
                    continue
                eid, var = p[1], p[2]
                if eid in needs.get(var, ()):
                    out[eid][var] = json.load(tf.extractfile(m))
    return out


def first_text(rec, field):
    """corpus 记录 → 首轮文本"""
    return norm(rec["history"][0]["content"])


def contained(needle, haystack, n=40):
    n = min(n, len(needle))
    return bool(n) and needle[:n] in haystack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="INT")
    ap.add_argument("--ann-split", default="test")
    a = ap.parse_args()

    ann_path = f"{CORPUS}/annotations/{a.shard}/{a.ann_split}.jsonl"
    ann = [json.loads(l) for l in open(ann_path)]
    dlg = {}
    with open(f"{CORPUS}/dialogues/{a.shard}/train.jsonl") as f:
        for line in f:
            r = json.loads(line)
            dlg[r["example_id"]] = r

    print("=== 家底 ===")
    print("  annotations(%s) 条数: %d" % (a.ann_split, len(ann)))
    print("  corpus dialogues 条数: %d" % len(dlg))

    ann_ids = [x["example_id"] for x in ann]
    print("\n" + "=" * 60)
    print("=== K1 覆盖盘点（不设阈值）===")
    any_var = read_metas(a.shard, want=set(ann_ids))
    v00 = read_metas(a.shard, want=set(ann_ids), variant="var00")
    print("  被标注 id 中 有**任意** spoken 变体: %d/%d" % (len(any_var), len(ann_ids)))
    print("  被标注 id 中 有 **var00**  变体: %d/%d   ← 上一版就是按这个筛的"
          % (len(v00), len(ann_ids)))
    no_var = [i for i in ann_ids if i not in any_var]
    if no_var:
        print("  完全无 spoken 的 id (%d): %s" % (len(no_var), no_var[:10]))
    nv0 = [i for i in ann_ids if i in any_var and i not in v00]
    if nv0:
        print("  有 spoken 但**没有 var00** 的 id (%d): %s   ← 自造缺口的来源"
              % (len(nv0), nv0[:10]))

    if not any_var:
        return
    # 每个 id 取任意一个变体作为主读数
    main_meta = {i: any_var[i][sorted(any_var[i])[0]] for i in any_var}
    spoken_full = {i: " ".join(norm(x["tts_text"]) for x in m["speech_meta"])
                   for i, m in main_meta.items()}

    print("\n" + "=" * 60)
    print("=== K2/K3 annotations 对得上谁 ===")
    n = hit_c = hit_s = 0
    turnrows = []
    for x in ann:
        eid = x["example_id"]
        if eid not in main_meta:
            continue
        n += 1
        at = norm(x["history"][0]["content"])
        ct = " ".join(norm(h["content"]) for h in dlg[eid]["history"]) if eid in dlg else ""
        hit_c += contained(at, ct)
        hit_s += contained(at, spoken_full[eid])
        turnrows.append((len(x["history"]), len(main_meta[eid]["speech_meta"])))
    print("  样本 %d 条（有 spoken 的）" % n)
    print("  ann 首轮 ⊂ corpus 全文 : %d/%d = %.1f%%" % (hit_c, n, 100.0 * hit_c / max(n, 1)))
    print("  ann 首轮 ⊂ spoken 全文 : %d/%d = %.1f%%" % (hit_s, n, 100.0 * hit_s / max(n, 1)))
    if turnrows:
        ac = [t[0] for t in turnrows]
        sc = [t[1] for t in turnrows]
        print("  ann 轮数: min=%d max=%d 均值=%.1f" % (min(ac), max(ac), sum(ac) / len(ac)))
        print("  spk 轮数: min=%d max=%d 均值=%.1f" % (min(sc), max(sc), sum(sc) / len(sc)))
        print("  轮数相同的条数: %d/%d"
              % (sum(1 for t in turnrows if t[0] == t[1]), len(turnrows)))

    print("\n" + "=" * 60)
    print("=== K4 陷阱量化：拿 `utterances` 当转写会怎样 ===")
    tot_u = same_u = tot_s = same_s = 0
    for eid, m in main_meta.items():
        u, s, c = m["utterances"], m["speech_meta"], dlg[eid]["history"]
        # 只比前 min(len) 轮，两个字段长度不同
        k = min(len(u), len(c))
        for i in range(k):
            tot_u += 1
            same_u += norm(u[i]["texts"]) == norm(c[i]["content"])
        k2 = min(len(s), len(c))
        for i in range(k2):
            tot_s += 1
            same_s += norm(s[i]["tts_text"]) == norm(c[i]["content"])
    print("  用 speech_meta 比 corpus: %d/%d = %.1f%%   ← 真转写"
          % (same_s, tot_s, 100.0 * same_s / max(tot_s, 1)))
    print("  用 utterances  比 corpus: %d/%d = %.1f%%   ← 拿错字段"
          % (same_u, tot_u, 100.0 * same_u / max(tot_u, 1)))
    print("  两个字段长度: utterances=%d  speech_meta=%d（同一条对话）"
          % (len(main_meta[sorted(main_meta)[0]]["utterances"]),
             len(main_meta[sorted(main_meta)[0]]["speech_meta"])))

    print("\n判读:")
    print("  K2/K3 都 ≈0 ⇒ annotations 与 corpus、spoken **都**对不上（孤立层）。")
    print("  K4 的落差 ⇒ 「join 不通」有可能是**字段选错**造出来的；报之前先两个都算。")


if __name__ == "__main__":
    main()
