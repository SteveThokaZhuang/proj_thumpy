"""F3 人工验证（judgments2.csv）分析：用户填写的 100 条判断 + 自由文本 note。

用户提出的核心观点（2026-09-15 原话）：
  - "发现了一些 bc 词出现时并不是 bc 场合"
  - "有些词会根据语调和语境决定其 bc 和 interruption 的程度"
  - "bc 和 interruption 是有程度的……可以分为不对说话人产生干扰的 bc 和会对说话人
     产生干扰的 bc-interruption，同时还有不带 bc 词的强硬 interruption"
  - "他们对话轮的影响并非二元"

本脚本做三件事：
  ① 与管线真值（truth.json 的 realized 标签）对表，看人工判断在哪些格子上偏离；
  ② **把 note 里的自由文本编码成用户描述的那套分级**（这一步的编码是我读 note 得出的
     判断，不是用户自己的勾选，所以每条都回显原文供核对）；
  ③ 找出「同一个表层词承担不同功能」的对照对 —— 这是分级说最硬的证据。

用 csv 模块读，不用 pandas（见 gpu02-infra-gotchas：CANDOR 转写 CSV 上 pandas 会段错误）。
"""
import csv
import json
import os
import re
from collections import Counter, defaultdict

KIT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
       "real_data/results/annotator/f3_kit")
JUDG = os.path.join(KIT, "judgments2.csv")
TRUTH = os.path.join(KIT, "truth.json")

# ---------------------------------------------------------------------------
# note → 分级编码。这一层是我读 note 后的判断，留了 raw 字段供用户复核。
# 分级（用户描述的那套）：
#   BC_PURE  不对说话人产生干扰的 bc（典型附和，明确不打断）
#   BC_MID   会对说话人产生干扰的 bc，即 bc-interruption（"介于两者之间"）
#   INTR     不带 bc 词的强硬 interruption，或 bc 词但功能是抢话轮
#   TURN     单纯话轮转换：有 bc 词但没重叠、说话人本来就说完了
#   MIXED    同一条 clip 里既有真 bc 又有 interruption
#   NOISE    音频/录音问题（听不清、回声、时间戳异常）
# 另外单独标记 two 个横切属性（可与其他类共存）：
#   NONOVER  明确说"没有重叠但仍是 bc"（换气间隙）
#   INTAUD   文本里有 bc 词，但听不出声音
#   INTENT   意图与实际效果不一致（想说打断但被忽略）
# ---------------------------------------------------------------------------
CODING = {
    "clip_000": ("NOISE", ["NONOVER"], "有overlap但是声音不清晰，感觉算是杂音类backchannel"),
    "clip_007": ("BC_PURE", [], "000-007里最清晰和标准的bc"),
    "clip_008": ("BC_PURE", [], "这个也很标准"),
    "clip_009": ("MIXED", [], "说话人0在说话人1插话后的Unhm是bc，但是说话人1（右声道）的i know是interruption"),
    "clip_010": ("BC_PURE", ["NONOVER"], "说话人0插入bc的时候并没有语音上的overlap，但是实际上是在说话人1的换气间隙，这种non overlap bc还是挺常见的感觉"),
    "clip_012": ("BC_MID", [], "这是语气比较强的bc，可能会影响到其他人，感觉得单独算一种bc，这时候在人类语境下可以不被打断也可以被打断"),
    "clip_018": ("BC_PURE", [], "句尾的bc，发生在切换话轮前，本质上也不会影响话论切换"),
    "clip_003": ("INTR", [], "这是interruption"),
    "clip_011": ("INTR", [], "Interruption"),
    "clip_019": ("BC_PURE", [], "同时发笑应该算bc吧？我认为是，但我不确定"),
    "clip_023": ("MIXED", [], "这个比较乱，只有说话人0的oh是真正的bg，前面一个说话人0 got it实际上打断了，还有后一个说话人1的yeah"),
    "clip_024": ("INTR", ["INTAUD"], "不是bc，我听到的是wait，算interruption"),
    "clip_025": ("INTR", [], "单纯两个说话人重叠了，硬要说可以算interruption"),
    "clip_028": ("TURN", ["NONOVER"], "有bc词但是没重叠，单纯的话轮转换"),
    "clip_030": ("INTR", [], "插嘴，算是interruption"),
    "clip_034": ("INTR", [], "Interruption"),
    "clip_041": ("TURN", ["NONOVER"], "有wow这类附和的bc词，不过没有重叠，这种情况比较特殊，wow在说话人断句中插入，比较可能在后续变成interruption"),
    "clip_043": ("TURN", ["NONOVER"], "有OK这类附和的bc词，不过没有重叠，这种情况比较特殊，wow在说话人断句中插入，比较可能在后续变成interruption"),
    "clip_044": ("BC_PURE", [], "这里的ok在说话人中，就明显是纯bc，不会打断"),
    "clip_045": ("TURN", [], "有yeah这种bc词，但是是正常话轮转换"),
    "clip_047": ("INTR", [], "有yeah这种bc词，不过这里yeah更偏向interruption"),
    "clip_048": ("TURN", [], "两个人互相yeah，都在谦让话轮，这也是个比较特殊的情况"),
    "clip_049": ("BC_MID", [], "这里说话人1说了个声音很长的yeah来做backchannel，不过发生bc的时段在说话人0快说完的时候，所以这里的bc虽然没有抢话轮，但是较强的语气也暗示说话人0交换话论给她"),
    "clip_052": ("BC_PURE", [], "说话人1表示惊叹的yeah，语调也很长，但是并不意味着暗示接管话轮，而是单纯表达惊叹，这里涉及的语义信息也是比较多"),
    "clip_053": ("BC_PURE", [], "两边的bc yeah用于附和"),
    "clip_058": ("INTR", [], "说话人1说yeah用来接话轮，有重叠但是是interruption"),
    "clip_059": ("INTR", [], "说话人1说yeah用来接话轮，有重叠但是是interruption"),
    "clip_060": ("MIXED", [], "说话人0的yeah是bc，其他的bc词并非bc，而是interruption"),
    "clip_062": ("MIXED", [], "有bc的yeah也有interruption的yeah"),
    "clip_063": ("MIXED", [], "有bc的yeah也有interruption的yeah"),
    "clip_064": ("INTR", [], "说话人1的ok是interruption"),
    "clip_066": ("MIXED", ["INTENT"], "说话人0的yeah和i know抢话轮本意是interruption，但是被说话人1忽略了，本质上也属于bc"),
    "clip_067": ("TURN", [], "正常话论切换中的yeah这种bc词"),
    "clip_070": ("BC_PURE", ["NONOVER"], "说话人1的wow bc词在说话人0的陈述断句处，虽然没有重叠音频但也算bc"),
    "clip_073": ("TURN", [], "正常话论切换中的bc词，个人认为非bc"),
    "clip_074": ("TURN", [], "正常话论切换中的bc词，个人认为非bc"),
    "clip_077": ("BC_MID", [], "说话人0的ok有轻微打断的意思，介于bc和interruption之间吧"),
    "clip_080": ("NOISE", [], "这条数据有点特殊，说话人0的左声道出现了说话人1的回音，除此之外，就是正常话论切换中的bc词，个人认为非bc"),
    "clip_085": ("INTR", [], "有bc和重叠，但双方都在用bc词打断对方，所以不算bc算interruption"),
    "clip_087": ("INTR", [], "纯interruption"),
    "clip_088": ("TURN", ["INTAUD"], "不清晰，虽然文本是yeah但实际上听不出任何东西，所以没算做bc"),
    "clip_089": ("TURN", [], "正常话论切换中的bc词，个人认为非bc"),
    "clip_091": ("TURN", [], "正常话论切换中的bc词，个人认为非bc"),
    "clip_092": ("TURN", ["INTAUD"], "虽然文本有bc，但是听不清，所以归类为正常话论切换中的bc词，个人认为非bc"),
    "clip_094": ("MIXED", [], "有bc词interruption（说话人0的oh）也有纯bc"),
    "clip_097": ("TURN", [], "正常话论切换中的bc词，个人认为非bc"),
    "clip_098": ("INTR", [], "纯interruption"),
}

# 表层词 → 该词在各 clip 里被判定成的功能（用于"同一个词跨功能"的对照）
# 从 note 里显式提到的词抽取
WORD_CASES = [
    ("yeah", "clip_044", "BC_PURE", "yeah 出现在说话人话轮中间，明显是纯 bc，不会打断"),
    ("yeah", "clip_052", "BC_PURE", "语调很长的 yeah，单纯表达惊叹，不暗示接管话轮"),
    ("yeah", "clip_049", "BC_MID",  "同样是很长的 yeah，但语气较强、暗示交换话轮"),
    ("yeah", "clip_047", "INTR",    "yeah 更偏向 interruption"),
    ("yeah", "clip_058", "INTR",    "yeah 用来接话轮，有重叠，是 interruption"),
    ("yeah", "clip_062", "MIXED",   "同一条里既有 bc 的 yeah 也有 interruption 的 yeah"),
    ("ok",   "clip_044", "BC_PURE", "ok 在说话人中，纯 bc，不会打断"),
    ("ok",   "clip_077", "BC_MID",  "ok 有轻微打断的意思，介于 bc 和 interruption 之间"),
    ("ok",   "clip_064", "INTR",    "说话人1的 ok 是 interruption"),
    ("wow",  "clip_041", "TURN",    "wow 附和词但没重叠，在说话人断句中插入"),
    ("wow",  "clip_070", "BC_PURE", "wow 在陈述断句处，没有重叠但算 bc"),
    ("i know", "clip_066", "MIXED", "i know 本意 interruption，被忽略后本质属 bc"),
]


def load_rows():
    with open(JUDG, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def norm(v):
    return (v or "").strip().rstrip(".").upper() or "?"


def main():
    rows = load_rows()
    truth = json.load(open(TRUTH))
    print(f"judgments2.csv: {len(rows)} 行; truth.json: {len(truth)} 条\n")

    # ---------- ① 与管线真值对表 ----------
    print("=" * 78)
    print("① 人工判断 vs 管线真值（truth.json 的 realized）")
    print("=" * 78)
    ct = defaultdict(Counter)
    for r in rows:
        cid = r["clip_id"]
        realized = truth.get(cid, {}).get("realized", "?")
        ct[realized][norm(r["heard_bc"])] += 1

    labels = sorted({norm(r["heard_bc"]) for r in rows})
    print(f"\n{'realized':16}" + "".join(f"{l:>10}" for l in labels) + f"{'合计':>8}")
    for realized in sorted(ct):
        row = ct[realized]
        print(f"{realized:16}" + "".join(f"{row[l]:>10}" for l in labels)
              + f"{sum(row.values()):>8}")

    # 指南 §3 说得很清楚，这两列问的是**两件不同的事**，要分别对表：
    #   heard_bc      = 窗口内、被标注的声道上，**有没有一个简短的附和 token**
    #   overlaps_host = 这个附和**是否与另一声道真正同时发声**
    # truth.json 的 realized 是"声学 realized 标签（真重叠 vs 停顿内）"，
    # 所以它对应的是 **overlaps_host**，不是 heard_bc。下面两条分开算。
    print("\n  【检查①】人工听到的 BC token —— 100 条里 %d 条听到 (%.0f%%)"
          % (sum(1 for r in rows if norm(r["heard_bc"]) == "Y"),
             100 * sum(1 for r in rows if norm(r["heard_bc"]) == "Y") / len(rows)))

    print("\n  同样对表 overlaps_host（人工听出来的重叠）:")
    ct2 = defaultdict(Counter)
    for r in rows:
        realized = truth.get(r["clip_id"], {}).get("realized", "?")
        ct2[realized][norm(r["overlaps_host"])] += 1
    labels2 = sorted({norm(r["overlaps_host"]) for r in rows})
    print(f"\n{'realized':16}" + "".join(f"{l:>10}" for l in labels2) + f"{'合计':>8}")
    for realized in sorted(ct2):
        row = ct2[realized]
        print(f"{realized:16}" + "".join(f"{row[l]:>10}" for l in labels2)
              + f"{sum(row.values()):>8}")

    # 【检查②】realized 声称的是"真重叠"，所以与 overlaps_host 对表才是正确口径。
    # realized=="Int" 按定义也是重叠，一并算作"应重叠"。
    OV = {"Backchannel", "Int"}
    agree = sum(1 for r in rows
                if norm(r["overlaps_host"]) ==
                ("Y" if truth.get(r["clip_id"], {}).get("realized") in OV else "N"))
    print(f"\n  【检查②】realized（真重叠 vs 停顿内）⇔ overlaps_host 一致率: "
          f"{agree}/{len(rows)} = {agree / len(rows):.1%}")
    for realized in ("Backchannel", "None"):
        sub = [r for r in rows if truth.get(r["clip_id"], {}).get("realized") == realized]
        exp = "Y" if realized in OV else "N"
        ok = sum(1 for r in sub if norm(r["overlaps_host"]) == exp)
        print(f"      realized={realized:12} 人工同意 {ok:2}/{len(sub)} = {ok / len(sub):5.1%}")

    print("\n  ⚠ 注意两个方向的不对称：管线说『重叠』时人工多数同意，"
          "管线说『停顿内(None)』时人工有相当比例听到了重叠。")

    # ---------- ② note 的分级编码 ----------
    print("\n" + "=" * 78)
    print("② note 分级编码（编码是我读 note 得出的，共 %d 条有 note）" % len(CODING))
    print("=" * 78)
    grade = Counter()
    flag = Counter()
    for cid, (g, flags, _) in CODING.items():
        grade[g] += 1
        for fl in flags:
            flag[fl] += 1
    print()
    names = {"BC_PURE": "纯 bc（明确不干扰说话人）",
             "BC_MID": "bc-interruption（有干扰、介于两者之间）",
             "INTR": "interruption（抢话轮）",
             "TURN": "单纯话轮转换（有 bc 词但非 bc）",
             "MIXED": "同一条里 bc 与 interruption 并存",
             "NOISE": "音频/录音问题（听不清、回声、时间戳）"}
    for g, n in grade.most_common():
        print(f"  {names[g]:38} {n:3}  ({n / len(CODING):5.1%})")
    print(f"  {'— 合计':38} {sum(grade.values()):3}")
    print("\n  横切标记（与上面的类可共存）:")
    fnames = {"NONOVER": "明确说『没重叠但仍是 bc』",
              "INTAUD": "文本里有 bc 词但听不出声音",
              "INTENT": "意图与实际效果不一致（想打断却被忽略）"}
    for fl, n in flag.most_common():
        print(f"    {fnames[fl]:34} {n}")

    # 分级 × 管线的二元标签：用户说的"程度"落在现有的哪一格
    print("\n  【分级 × 管线 realized】—— 看二元标签是不是把不同功能压进了一格:")
    cross = defaultdict(Counter)
    for cid, (g, _, _) in CODING.items():
        realized = truth.get(cid, {}).get("realized", "?")
        cross[realized][g] += 1
    gs = [g for g, _ in grade.most_common()]
    print(f"\n{'realized':14}" + "".join(f"{g:>10}" for g in gs) + f"{'合计':>7}")
    for realized in sorted(cross):
        row = cross[realized]
        print(f"{realized:14}" + "".join(f"{row[g]:>10}" for g in gs)
              + f"{sum(row.values()):>7}")

    no_note = [r["clip_id"] for r in rows if not (r["notes"] or "").strip()]
    print(f"\n  未填 note 的 clip: {len(no_note)} 条 {no_note[:10]}")
    uncoded = [r["clip_id"] for r in rows
               if (r["notes"] or "").strip() and r["clip_id"] not in CODING]
    if uncoded:
        print(f"  ⚠ 有 note 但我没编码的: {uncoded}")

    # ---------- ③ 同一个词、不同功能 ----------
    print("\n" + "=" * 78)
    print("③ 同一个表层词承担不同功能（分级说最直接的证据）")
    print("=" * 78)
    by_word = defaultdict(list)
    for w, cid, g, why in WORD_CASES:
        by_word[w].append((cid, g, why))
    for w, cases in sorted(by_word.items()):
        funcs = {g for _, g, _ in cases}
        print(f"\n  「{w}」→ {len(funcs)} 种功能: {', '.join(sorted(funcs))}")
        for cid, g, why in cases:
            print(f"      {cid}  {g:8} {why}")

    # ---------- ④ 关键词扫描：证实编码没漏 ----------
    print("\n" + "=" * 78)
    print("④ 自由文本关键词扫描（防止我的编码漏掉整类）")
    print("=" * 78)
    keys = {
        "interruption": r"interrupt|打断|插嘴|抢话|抢话轮|插话",
        "非bc/不算bc": r"非bc|不算bc|不是bc|并非bc|没算做bc",
        "话轮转换": r"话轮转换|话论切换|话轮切换|交换话论",
        "无重叠": r"没有重叠|没重叠|没有语音上的overlap|non overlap|虽然没有重叠",
        "听不清": r"听不清|不清晰|听不出|声音不清晰",
        "程度/介于": r"程度|介于|偏向|比较强|轻微",
    }
    for name, pat in keys.items():
        hits = [r["clip_id"] for r in rows if re.search(pat, r["notes"] or "")]
        print(f"  {name:14} {len(hits):3} 条  {hits}")

    # ---------- ⑤ 我的编码在多大程度上可信 ----------
    print("\n" + "=" * 78)
    print("⑤ 两条自查：笔记是不是挑着写的 + 笔记与 y/n 列自洽吗")
    print("=" * 78)

    # (a) 选择效应：人只在"有话说"时才写 note，所以有 note 的子集偏向难例。
    #     如果难例占比两边差很多，那 ② 的分级比例就不能外推到全部 100 条。
    print("\n  (a) 选择效应 —— 有 note 的 47 条是不是偏向难例:")
    noted = [r for r in rows if r["clip_id"] in CODING]
    unnoted = [r for r in rows if r["clip_id"] not in CODING]
    for name, sub in (("有 note", noted), ("无 note", unnoted)):
        n_bc = sum(1 for r in sub
                   if truth.get(r["clip_id"], {}).get("realized") in OV)
        y = sum(1 for r in sub if norm(r["heard_bc"]) == "Y")
        ov = sum(1 for r in sub if norm(r["overlaps_host"]) == "Y")
        print(f"      {name:8} n={len(sub):3}   realized=重叠 {n_bc:2} ({n_bc / len(sub):5.1%})"
              f"   heard_bc=Y {y:2} ({y / len(sub):5.1%})"
              f"   overlaps=Y {ov:2} ({ov / len(sub):5.1%})")
    print("      → 两边比例差得越多，② 的分级比例越不能外推到 100 条。")

    # (b) 自洽性：note 说"听不清"的，heard_bc 应该是 N；说"是 bc"的不该填 N。
    print("\n  (b) note 与 y/n 列的自洽性抽查:")
    # 只有两类能做**机械**校验：
    #   - 说"没重叠但仍是 bc"的 → overlaps_host 必须 N
    #   - 明说是 bc 的        → heard_bc 必须 Y
    # "听不清"那类**不设自动判定**：note 常常是"听不清但算是杂音类 bc"这种混合口径
    # （clip_000），或"有回音，但 bc 词本身听得见"（clip_080），拿一个规则去卡它们
    # 会造出假的不一致。只回显，由人看。
    bad = []
    for cid, (g, flags, raw) in sorted(CODING.items()):
        r = next(x for x in rows if x["clip_id"] == cid)
        hb, ov = norm(r["heard_bc"]), norm(r["overlaps_host"])
        if g == "TURN" and "NONOVER" in flags:
            ok, flagtxt = ov == "N", "（说没重叠）"
        elif g in ("BC_PURE", "BC_MID"):
            ok, flagtxt = hb == "Y", "（说是 bc）"
        else:
            continue
        if not ok:
            bad.append(cid)
        print(f"      {cid} {flagtxt} heard_bc={hb} overlaps={ov} {'✓' if ok else '✗ 不一致'}")
    print(f"\n      机械可校验项里不一致: {len(bad)} 条 {bad}")

    print("\n      —— 下面这些**不做自动判定**，请你目视确认填得对不对 ——")
    for cid, (g, flags, raw) in sorted(CODING.items()):
        if not ("INTAUD" in flags or g == "NOISE"):
            continue
        r = next(x for x in rows if x["clip_id"] == cid)
        print(f"      {cid} heard_bc={norm(r['heard_bc'])} "
              f"overlaps={norm(r['overlaps_host'])}  note: {raw}")
    print("      （指南 §3 说『听不清就填 n』，上表里有填 Y 的 —— 如果是有意为之，"
          "说明这条规则在实操中不好用，值得改。）")

    # ---------- ⑥ 产物 ----------
    out = {
        "n_rows": len(rows),
        "n_with_note": len(CODING),
        "heard_bc_x_realized": {k: dict(v) for k, v in ct.items()},
        "overlaps_host_x_realized": {k: dict(v) for k, v in ct2.items()},
        "grade_counts": dict(grade),
        "flag_counts": dict(flag),
        "coding": {c: {"grade": g, "flags": f, "raw": raw}
                   for c, (g, f, raw) in CODING.items()},
    }
    p = os.path.join(KIT, "f3_analysis.json")
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
    print(f"\n已写出 {p}")


if __name__ == "__main__":
    main()
