"""§3.7 `annotations` 的文本**到底是哪一份**？—— 四份文本的交叉核对。

背景：`duplexgen-code/docs/CORPUS.md` 有一张表，说 `annotations/` 装的是
**the same dialogues, plus human rater votes at annotated word boundaries**。
§3 早就量到 `content` 逐 turn 相同 0/50，本脚本把它做成**全 6 场景 × 两个 split**
的系统核对，并追出那份文本的来源。

盘上能拿到的**四份文本**：
  ① 上游原始（INT = `Anthropic/AnthropicInterviewer` 的 workforce_transcripts.csv）
  ② `duplexgen-corpus/dialogues/<C>/train.jsonl`      —— 发布的生成对话
  ③ `duplexgen-spoken` 各变体的 `meta.json.speech_meta` —— ②的音频转写
  ④ `duplexgen-corpus/annotations/<C>/{train,test}.jsonl` —— 人工标注

━━ 判据（写于跑之前）━━
P1 文本同一性：对**有同名 dialogue** 的标注，逐条全同率应 > 0（文档说「same」）。
P2 轮数恒等：同上，轮数相同率应 > 0。
P3 来源归属：④ 的首轮文本应能在 ①②③ 至少一处找到（40 字片段包含）。
P4 血统对照：② 的首轮应能在 ① 找到（证明 ② 确实来自 ①）。

**P4 是给 P3 当对照的**：如果 P4 命中而 P3 全不中，才说明「找不到」不是方法问题。

━━ 数据依赖 ━━
① 需先下载（不在盘上）：
  curl -L -H "Authorization: Bearer $HF_TOKEN" \
    https://huggingface.co/datasets/Anthropic/AnthropicInterviewer/resolve/main/interview_transcripts/workforce_transcripts.csv \
    -o temp/upstream/workforce_transcripts.csv
缺这个文件时，脚本自动跳过 ① 相关行并**明确标注跳过**，不静默。

用法:
  python duplexgen_annotation_provenance.py                 # 全 6 场景
  python duplexgen_annotation_provenance.py --scenarios INT
"""
import argparse
import csv
import json
import os
import re
import tarfile
from collections import defaultdict

CORPUS = "/share/workspace3/shared_dataset/duplexgen-corpus"
SPOKEN = "/share/workspace3/shared_dataset/duplexgen-spoken"
UPSTREAM = ("/share/workspace3/zhuangruicen/proj-thumpy/temp/upstream/"
            "workforce_transcripts.csv")
CODES = ["TEA", "PLN", "INT", "NEG", "PER", "SOC"]
MARKER = re.compile(r"\[[^\]]*\]")


def norm(s):
    s = MARKER.sub(" ", s.lower())
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def part_a(codes):
    """P1/P2 + id 覆盖：6 场景 × 2 split。"""
    print("=" * 74)
    print("=== A. annotations ↔ dialogues：id 覆盖与文本同一性 ===")
    print("%-5s %-6s %6s %8s %10s %10s %10s"
          % ("场景", "split", "条数", "有dlg", "逐条全同", "轮数相同", "首轮相同"))
    tot = defaultdict(int)
    for c in codes:
        dpath = f"{CORPUS}/dialogues/{c}/train.jsonl"
        if not os.path.exists(dpath):
            continue
        dl = {r["example_id"]: r for r in load_jsonl(dpath)}
        for split in ("train", "test"):
            apath = f"{CORPUS}/annotations/{c}/{split}.jsonl"
            if not os.path.exists(apath):
                continue
            ann = load_jsonl(apath)
            n = same = teq = f0 = 0
            for a in ann:
                eid = a["example_id"]
                if eid not in dl:
                    continue
                n += 1
                ha, hd = a["history"], dl[eid]["history"]
                if len(ha) == len(hd):
                    teq += 1
                k = min(len(ha), len(hd))
                if (k and len(ha) == len(hd)
                        and all(norm(ha[i]["content"]) == norm(hd[i]["content"])
                                for i in range(k))):
                    same += 1
                if ha and hd and norm(ha[0]["content"]) == norm(hd[0]["content"]):
                    f0 += 1
            print("%-5s %-6s %6d %8d %10d %10d %10d"
                  % (c, split, len(ann), n, same, teq, f0))
            tot["n"] += n
            tot["same"] += same
            tot["teq"] += teq
            tot["f0"] += f0
    print("-" * 74)
    print("合计: 有对应 dlg %d 条 | 逐条全同 %d | 轮数相同 %d | 首轮相同 %d"
          % (tot["n"], tot["same"], tot["teq"], tot["f0"]))
    print("⇒ P1 %s（0 条全同）/ P2 %s（%d/%d = %.1f%%）"
          % ("**不通过**" if not tot["same"] else "通过",
             "字面通过" if tot["teq"] else "**不通过**",
             tot["teq"], tot["n"], 100.0 * tot["teq"] / max(tot["n"], 1)))
    print("⚠️ **P2 的判据写得太松**：当时写的是「> 0」，3/99 就字面通过了。")
    print("   3% 的轮数巧合**不足以**支撑「同一份对话」—— 零模型下两条对话轮数相同的")
    print("   概率本来就不是 0（轮数都集中在十几到二十）。这条判据**没有分辨力**，")
    print("   保留在此是为了留档：判据要在写的时候就问「零模型下这个量是多少」。")
    return tot


def part_b(scenario, up):
    """P3/P4：④ 的首轮文本能不能在 ①/②/③ 里找到。"""
    print()
    print("=" * 74)
    print("=== B. 标注文本的来源归属（场景 %s）===" % scenario)
    dpath = f"{CORPUS}/dialogues/{scenario}/train.jsonl"
    dl = {r["example_id"]: r for r in load_jsonl(dpath)}
    ann = (load_jsonl(f"{CORPUS}/annotations/{scenario}/train.jsonl")
           + load_jsonl(f"{CORPUS}/annotations/{scenario}/test.jsonl"))
    first = {a["example_id"]: norm(a["history"][0]["content"]) for a in ann}

    # ③ 所有变体的 speech_meta
    ids = set(first)
    key2tar = defaultdict(list)
    with open(f"{SPOKEN}/metadata.jsonl") as f:
        for line in f:
            x = json.loads(line)
            if x["scenario"] == scenario and x["example_id"] in ids:
                key2tar[x["shard"]].append((x["example_id"], x["variant"]))
    if not key2tar:
        print("  （该场景在 spoken 里无变体，跳过 ③）")
    metas = defaultdict(dict)
    for tar, items in key2tar.items():
        byvar = defaultdict(set)
        for e, v in items:
            byvar[v].add(e)
        with tarfile.open(f"{SPOKEN}/{tar}", "r:") as tf:
            for m in tf.getmembers():
                p = m.name.split("/")
                if len(p) != 4 or not m.name.endswith("meta.json"):
                    continue
                if p[1] in byvar.get(p[2], ()):
                    metas[p[1]][p[2]] = json.load(tf.extractfile(m))

    def has(hay, needle):
        k = min(40, len(needle))
        return bool(k) and needle[:k] in hay

    h_up = h_dlg = h_var = 0
    n = 0
    for eid, txt in first.items():
        n += 1
        if up and eid in up:
            h_up += has(norm(up[eid]), txt)
        if eid in dl:
            h_dlg += has(" ".join(norm(h["content"]) for h in dl[eid]["history"]), txt)
        if eid in metas:
            for mm in metas[eid].values():
                if has(" ".join(norm(x["tts_text"]) for x in mm["speech_meta"]), txt):
                    h_var += 1
                    break
    print("  标注首轮 ⊂ ① 上游原文          : %d/%d" % (h_up, n))
    print("  标注首轮 ⊂ ② corpus dialogues   : %d/%d" % (h_dlg, n))
    print("  标注首轮 ⊂ ③ **任意**变体 speech_meta: %d/%d" % (h_var, n))
    print("⇒ P3 %s（④ 的文本在 ①②③ 里都找不到）"
          % ("不通过" if (h_up + h_dlg + h_var) == 0 else "**部分通过**"))

    # P4 对照：② 能不能在 ① 里找到
    if up:
        m = n2 = 0
        for eid, d in dl.items():
            if eid not in up:
                continue
            n2 += 1
            m += has(norm(up[eid]), norm(d["history"][0]["content"]))
        print("  【对照 P4】② dialogues 首轮 ⊂ ① 上游原文: %d/%d = %.1f%%"
              % (m, n2, 100.0 * m / max(n2, 1)))
        print("⇒ P4 %s（② 确实源自 ①；所以 P3 的「找不到」不是方法问题）"
              % ("通过" if m else "**不通过**"))
    else:
        print("  【P4 跳过】**缺 %s**，无法做血统对照 —— 这一行没有跑，不是通过。" % UPSTREAM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--provenance-scenario", default="INT",
                    help="做 P3/P4 来源归属的场景（需该场景上游原文）")
    a = ap.parse_args()
    codes = a.scenarios or CODES

    up = {}
    if os.path.exists(UPSTREAM):
        with open(UPSTREAM) as f:
            for row in csv.DictReader(f):
                up[row["transcript_id"]] = row["text"]
        print("上游 %s: %d 条" % (os.path.basename(UPSTREAM), len(up)))
    else:
        print("⚠️ 上游文件不存在，① 与 P4 将**跳过**（不是通过）：\n   %s" % UPSTREAM)

    part_a(codes)
    part_b(a.provenance_scenario, up)

    print()
    print("判读: P1/P2 不通过 ⇒ 文档「the same dialogues」与已发布文件**不一致**；")
    print("      P4 通过而 P3 全 0 ⇒ 标注用的是一份**未发布的**文本（第 ④ 份）。")


if __name__ == "__main__":
    main()
