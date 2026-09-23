"""把同一个 id 的**所有**文件摊开看（base / clean / add / clean_add），逐字段全文。

## 为什么要这一步

前四遍都是**统计**（占比、相关、相等率），每一条都留了个说不通的尾巴：

- `clean` 在用户区间内与主版逐样本 maxΔ 高达 0.98、逐帧能量相关只有 0.43
  ⇒ **不是**「把主版置零」，也**不像**同段语音的另一种渲染。
- `clean_X_timestamp` 的文本与 `X_timestamp` 逐字相同**正好 50.0%**（1635/3272）。
  正好一半，这个数**长得像配对错位**，不像自然现象。
- 音频里只有标注的那个人（区间外 p99 RMS ≈ 1e-4，数字静音）。

**统计量解释不了的时候，就去看那条数据本身。**
（memory `tiebreak-default-read-as-evidence`：同一容器里两个字段都像目标物时，
选错的那个会给出「像结果」的坏数 —— 这里同理，`clean`/`add` 到底指什么，
只有把文件摊开才知道。）

## 做法

给定若干 id，打印：
1. 该 id 下**所有**文件名（base / clean / add / clean_add 四种前缀组合）
2. 每个 json 的**全文**（文本类给全文，分段类给全部段）
3. 段数、时长
**不抽样、不截断、不挑**。

用法（纯读盘，0 GPU）：
  python scripts/humdial_group_dump.py --zip <zip> --dir en_test_nondev/ask \
      --mode differ --k 3
"""
import argparse
import json
import os
import re
import sys
import zipfile

# 🔴 第一版写的是 `re.compile(r"[^a-z0-9]+")` —— 那个正则会把**汉字整个删掉**，
#    于是所有 cn_* 的文本都规范化成空串，两两"相等"。
#    症状：Q1b 报出「逐字相同 1635/3272 = 正好 50.0%」——**正好一半**就是因为
#    语料一半中文，而中文那半全被规范化没了。恒真检查的典型形态
#    （memory `hardcoded-conclusions-escape-reproduction`：恒真自检）。
#    改用 `str.isalnum()`（对 CJK 返回 True），保留汉字。
def norm(s):
    return "".join(c for c in s.lower() if c.isalnum())


def stem_of(b):
    for ext in (".json", ".wav"):
        if b.endswith(ext):
            b = b[:-len(ext)]
            break
    for suf in ("_add_timestamp", "_add", "_timestamp"):
        if b.endswith(suf):
            return b[:-len(suf)]
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--dir", default="en_test_nondev/ask")
    ap.add_argument("--mode", default="differ", choices=["differ", "same"],
                    help="differ=挑 clean 文本与主版不同的；same=挑相同的")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.zip)
    names = [i.filename for i in zf.infolist() if i.file_size > 0]
    mine = [n for n in names if os.path.dirname(n) == "test/" + args.dir]
    jset = set(n for n in names if n.endswith(".json"))

    def txt(n):
        return json.loads(zf.read(n))["text"] if n in jset else None

    ids = sorted({stem_of(os.path.basename(n)) for n in mine
                  if not os.path.basename(n).startswith("clean_")})
    picked = []
    for st in ids:
        b = "test/%s/%s_timestamp.json" % (args.dir, st)
        c = "test/%s/clean_%s_timestamp.json" % (args.dir, st)
        if b in jset and c in jset:
            eq = norm(txt(b)) == norm(txt(c))
            if (args.mode == "same") == eq:
                picked.append(st)
        if len(picked) >= args.k:
            break

    print("目录 %s  模式 %s  挑中 %d 个 id：%s\n"
          % (args.dir, args.mode, len(picked), picked))
    for st in picked:
        print("=" * 88)
        print("id = %s" % st)
        print("=" * 88)
        group = sorted(os.path.basename(n) for n in mine if stem_of(os.path.basename(n)) == st)
        print("文件（%d 个）:" % len(group))
        for g in group:
            print("    %s" % g)

        for prefix in ("", "clean_"):
            for tag, suf in (("分段 speech_segments", ".json"),
                             ("词级 chunks", "_timestamp.json"),
                             ("分段·add", "_add.json"),
                             ("词级·add", "_add_timestamp.json")):
                fn = "test/%s/%s%s%s" % (args.dir, prefix, st, suf)
                if fn not in jset:
                    continue
                o = json.loads(zf.read(fn))
                print("\n  [%s%s] %s" % (prefix, tag, os.path.basename(fn)))
                if "speech_segments" in o:
                    print("    final_duration = %s；%d 段"
                          % (o["final_duration"], len(o["speech_segments"])))
                    for s in o["speech_segments"]:
                        print("      [%7.2f, %7.2f] %s"
                              % (s["xmin"], s["xmax"], s["text"]))
                else:
                    print("    text = %s" % o["text"])
                    print("    chunks %d 个，首尾: %s … %s"
                          % (len(o["chunks"]), o["chunks"][0], o["chunks"][-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
