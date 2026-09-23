"""HumDial-FDBench **内容**勘察（第二遍）—— 解 schema_probe 留下的三个谜。

## 第一遍量出来的、现在必须解释的三件事

1. **`_add` 是什么？** 结构探针把 `X_add.json` 和 `X.json` 配成 287 对，
   结果 `speech_segments` **逐条相同 0 / 不同 287**。
   ⚠️ **但先别下"命名骗人"的结论** —— 也可能是**配对方式错了**
   （`X_add` 未必对应 `X`）。本项目在 DuplexGen 上刚犯过一模一样的错：
   `CORPUS.md` 说 annotations 是 "the same dialogues"，实测 0/99。
   要区分"命名骗人"和"我配错了对"，唯一办法是**看真名字**。

2. **时间戳族恰好是分段族的 2 倍**（如 cn/ask：`timestamp` 472 vs `plain` 236，
   `add_timestamp` 128 vs `add` 64）。2× 是个**太整齐**的比例，不像巧合。
   可能是 (a) 每个 stem 两个 wav，或 (b) stem 里还有一层我们没剥的后缀。
   → **必须把真文件名打出来**，不许从计数反推。

3. **`chunks`/`text` 是什么？** 结构探针只看到键名。它是转写？是逐轮？
   有没有说话人？—— 这直接决定 HumDial 接不接得上本项目的 L1/L3。

## 判据（跑之前写死）

- **N1**：打印**整目录**的文件名（不抽样 —— 目录才几十个文件）。
  若 `timestamp` 真是 `plain` 的 2 倍，文件名列表里必须**肉眼可见**那个结构。
- **N2**：stem 集合关系算出来：`stems(wav)` vs `stems(timestamp)` vs `stems(plain)`。
  **任何一边多出来的 stem，逐个列出**（memory：过滤器会制造缺口，
  反过来「凭空多出来的」也必须落到名字上）。
- **N3**：`_add` 的**正确配对**由名字定，不由后缀替换定。配好对之后再比一次
  `speech_segments`，**若还是全不同，才**可以写「`_add` 是另一批对话」。
- **N4**：把每个 json 族的**全文**打出一条完整样本（不截断），逐键看。
- **N5**：全库扫一遍**所有** json 的所有键（递归），确认**没有**说话人/角色字段被漏掉。

⚠️ 本脚本仍然**只做勘察，不写实验设计**（recon §5.5）。

用法（纯读盘，0 GPU）：
  python scripts/humdial_content_probe.py --zip <zip> --dir en_test_nondev/ask
"""
import argparse
import collections
import json
import os
import sys
import zipfile


def stem(name):
    """文件名 → stem：去扩展名，再去已知后缀。"""
    b = os.path.basename(name)
    for ext in (".json", ".wav"):
        if b.endswith(ext):
            b = b[:-len(ext)]
            break
    for suf in ("_add_timestamp", "_add", "_timestamp"):
        if b.endswith(suf):
            return b[:-len(suf)]
    return b


def all_keys(o, prefix=""):
    """递归收集所有键路径。"""
    out = set()
    if isinstance(o, dict):
        for k, v in o.items():
            p = prefix + "." + k if prefix else k
            out.add(p)
            out |= all_keys(v, p)
    elif isinstance(o, list):
        for v in o[:3]:
            out |= all_keys(v, prefix + "[]")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--dir", default="en_test_nondev/ask",
                    help="打印整目录文件名的那个目录")
    ap.add_argument("--lang", default="en_test_nondev")
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.zip)
    info = [i for i in zf.infolist() if i.file_size > 0]
    names = [i.filename for i in info]

    # ---------- N1 整目录文件名 ----------
    print("=" * 76)
    print("N1 整目录文件名：%s" % args.dir)
    print("=" * 76)
    ins = sorted(n for n in names if os.path.dirname(n) == "test/" + args.dir)
    print("共 %d 个文件；按 (stem, 后缀) 列：" % len(ins))
    bystem = collections.defaultdict(list)
    for n in ins:
        bystem[stem(n)].append(os.path.basename(n))
    print("  %d 个 stem" % len(bystem))
    for s in sorted(bystem)[:6]:
        print("   stem %-14s -> %s" % (s, sorted(bystem[s])))
    print("   …（只列 6 个；下面按后缀统计）")

    suf_ct = collections.Counter()
    for n in ins:
        b = os.path.basename(n)
        for suf in ("_add_timestamp.json", "_timestamp.json", "_add.wav",
                    ".wav", "_add.json", ".json"):
            if b.endswith(suf):
                suf_ct[suf] += 1
                break
    for k, v in suf_ct.most_common():
        print("   %-22s %4d" % (k, v))

    # ---------- N2 stem 集合关系 ----------
    print("\n" + "=" * 76)
    print("N2 各目录的 stem 集合关系")
    print("=" * 76)
    dirs = sorted({os.path.dirname(n) for n in names
                   if os.path.dirname(n).startswith("test/")})
    # ⚠️ 文件数 与 stem 数 是**两个不同的量**（同名 stem 可有 1 个或 2 个文件）。
    #    第一遍结构探针报的是**文件数**，这里必须两个都报 —— 只报一个的话
    #    「600 个 wav / 540 个 stem」看起来像矛盾，其实不是。
    print("%-38s %s" % ("目录", " ".join("%14s" % x for x in
          ("wav 文/stem", "addwav 文/stem", "plain 文/stem", "ts 文/stem"))))
    extra_report = []
    for d in dirs:
        fs = [n for n in names if os.path.dirname(n) == d]
        cnt = collections.Counter()
        sw, sa, sp, st = set(), set(), set(), set()
        for n in fs:
            b = os.path.basename(n)
            if b.endswith("_add.wav"):
                cnt["sa"] += 1; sa.add(stem(n))
            elif b.endswith(".wav"):
                cnt["sw"] += 1; sw.add(stem(n))
            elif b.endswith("_add_timestamp.json") or b.endswith("_timestamp.json"):
                cnt["st"] += 1; st.add(stem(n))
            elif b.endswith(".json"):
                cnt["sp"] += 1; sp.add(stem(n))
        print("%-38s %s" % (d.replace("test/", ""), " ".join(
            "%7d/%-6d" % (cnt[k], len(s)) for k, s in
            (("sw", sw), ("sa", sa), ("sp", sp), ("st", st)))))
        # 缺族的 stem 逐个记下
        for lab, s in (("wav", sw | sa), ("jsonspeech", sp), ("ts", st)):
            miss = (sw | sa | sp | st) - s
            if miss:
                extra_report.append((d.replace("test/", ""), lab,
                                     len(miss), sorted(miss)[:4]))
    print("\n  有 stem 但缺某一族的：")
    for d_, lab, n_, ex in extra_report:
        print("    %-38s 缺 %-10s %3d 个  例 %s" % (d_, lab, n_, ex))
    if not extra_report:
        print("    （无 —— 三族 stem 完全一致）")

    # ---------- N3 _add 的正确配对 ----------
    print("\n" + "=" * 76)
    print("N3 `_add` 到底配谁（由**名字**定，不由后缀替换定）")
    print("=" * 76)
    # 先看 add 的 stem 在不在非 add 的 stem 集合里
    a_stems, n_stems = set(), set()
    for n in names:
        b = os.path.basename(n)
        if b.endswith("_add.json") or b.endswith("_add_timestamp.json"):
            a_stems.add(stem(n))
        elif b.endswith(".json"):
            n_stems.add(stem(n))
    both = a_stems & n_stems
    print("  add 族 stem %d；非 add 族 stem %d；**交集 %d**"
          % (len(a_stems), len(n_stems), len(both)))
    print("  ⇒ add 的编号 %s 与非 add 的编号"
          % ("落在同一编号空间" if both else "**落在不同编号空间**"))
    if both:
        print("     交集的例子：%s" % sorted(both)[:6])
    else:
        print("     非 add 例子：%s" % sorted(n_stems)[:6])
        print("     add  例子：%s" % sorted(a_stems)[:6])
        print("     ⇒ `X_add` 与 `X` **不是同一编号空间**，")
        print("        第一遍那个「0/287 相同」是**配对假设错**，不是命名骗人。")

    # ---------- N4 各族一条全文 ----------
    print("\n" + "=" * 76)
    print("N4 各族全文样本（不截断）")
    print("=" * 76)
    shown = {}
    for n in names:
        b = os.path.basename(n)
        kind = ("add_timestamp" if b.endswith("_add_timestamp.json")
                else "timestamp" if b.endswith("_timestamp.json")
                else "add" if b.endswith("_add.json")
                else "plain" if b.endswith(".json") else None)
        if kind and kind not in shown and os.path.dirname(n).startswith("test/"):
            shown[kind] = n
        if len(shown) == 4:
            break
    for kind in ("plain", "timestamp", "add", "add_timestamp"):
        n = shown.get(kind)
        if not n:
            print("\n[%s] 无样本" % kind)
            continue
        o = json.loads(zf.read(n))
        print("\n[%s] %s" % (kind, n))
        print(json.dumps(o, ensure_ascii=False, indent=1)[:2600])
        print("  键: %s" % sorted(o))
        if "chunks" in o:
            c = o["chunks"]
            print("  chunks 类型 %s, 长度 %s" % (type(c).__name__, len(c)))
            if c and isinstance(c[0], dict):
                print("  chunks[0] 键: %s" % sorted(c[0]))
        if "speech_segments" in o:
            s = o["speech_segments"]
            print("  speech_segments %d 段, 段键 %s"
                  % (len(s), sorted(s[0]) if s else []))

    # ---------- N5 全库递归键 ----------
    print("\n" + "=" * 76)
    print("N5 全库所有 json 的递归键集合（找有没有说话人/角色字段）")
    print("=" * 76)
    kk = collections.defaultdict(collections.Counter)
    js = [n for n in names if n.endswith(".json")]
    for n in js:
        b = os.path.basename(n)
        kind = ("add_timestamp" if b.endswith("_add_timestamp.json")
                else "timestamp" if b.endswith("_timestamp.json")
                else "add" if b.endswith("_add.json") else "plain")
        try:
            o = json.loads(zf.read(n))
        except Exception:
            continue
        for k in all_keys(o):
            kk[kind][k] += 1
    for kind in ("plain", "timestamp", "add", "add_timestamp"):
        if kind not in kk:
            continue
        print("  [%s] 共 n=%d 条" % (kind, max(kk[kind].values())))
        for k, c in kk[kind].most_common():
            print("      %-34s %d" % (k, c))
    print("\n  ⚠️ 若上面**没有**任何 speaker/role/spk 字段 ⇒ HumDial 不提供逐轮说话人。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
