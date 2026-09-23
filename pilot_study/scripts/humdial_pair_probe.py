"""HumDial-FDBench 第三遍：**四族文件之间是什么关系**。

## 前两遍已经确定的（都从字节现算）

- 音频 **9098 个 wav，一律 ch=1 / sr=16000 / bits=16** ⇒ recon §4 的「双声道」是**假的**。
- 只有 **2 种** json 形状：
  - 分段族 `{final_duration, speech_segments:[{xmin,xmax,text}]}` —— 5198 条
  - 词级族 `{text, chunks:[{text,timestamp:[a,b]}]}` —— 9098 条
- **全库递归键里没有任何 speaker / role 字段**（N5 扫了 9098 条 json）。
- 文件名 = `[clean_]{DDDD}_{NNNN}[_add]{.wav|.json|_timestamp.json}`。

## 三遍要回答的（按重要性排）

### Q1 🔴 `clean_` 是什么 —— 这决定 HumDial 对本项目有没有用

若 `clean_X.wav` 是「同一条对话、但只剩用户那一侧」⇒ 那就是本项目一直缺的
**真人双工 + 分轨**素材，价值极高。
若只是降噪/重录的另一条 ⇒ 价值一般。
**不许靠名字猜**（本项目在 DuplexGen `CORPUS.md` 上刚栽过），用音频本身判：
在 `speech_segments` 说「用户正在说」的区间与「空隙（助手在说）」的区间上分别量能量。
- 若 `clean` 的两类区间能量**差很多**（空隙接近静音）⇒ clean 去掉了助手 ⇒ 分轨
- 若两类区间能量**差不多** ⇒ clean 是整条对话的另一种版本，没分轨

### Q2 `speech_segments` 与 `chunks` 是不是**同一份标注的两种粒度**

若是 ⇒ 只是一个字段的两种渲染，不构成第二个标注源。
若不是 ⇒ HumDial 自带**两个标注源**，本项目的 L1/L2/L3 三源对照可以直接搬过来。

### Q3 `_add` 与主版本的关系（第二遍只量出 speech_segments 287/287 不同，
      但**没看它们是不是同一个场景/同一个模板**）—— 逐对看文本相似度。

## 判据（跑之前写死）

- **Q1**：`clean` 与主版**时长比**的中位数；以及**用户区间 vs 空隙区间的 RMS 比**。
  判据：空隙/用户 RMS < 0.25 ⇒ 记作「clean ≈ 去助手」；> 0.6 ⇒ 记作「clean ≈ 同一条」。
  落在中间 ⇒ **如实报中间，不许二选一**。
- **Q2**：把 `speech_segments[].text` 拼起来，与 `chunks.text` 做**规范化后**比较
  （去标点、统一小写、压空白）。报**逐字相等率**。>0.95 ⇒ 同一份；<0.8 ⇒ 两个源。
- **Q3**：`X` 与 `X_add` 的 speech_segments 文本的字符相似度分布。
  高 ⇒ 同场景换模板；低 ⇒ 无关。

用法（纯读盘，0 GPU）：
  python scripts/humdial_pair_probe.py --zip <zip> --audio-n 20
"""
import argparse
import collections
import difflib
import json
import os
import re
import struct
import sys
import zipfile

import numpy as np

# 🔴 第一版写的是 `re.compile(r"[^a-z0-9]+")` —— 那个正则会把**汉字整个删掉**，
#    于是所有 cn_* 的文本都规范化成空串，两两"相等"。
#    症状：Q1b 报出「逐字相同 1635/3272 = 正好 50.0%」——**正好一半**就是因为
#    语料一半中文，而中文那半全被规范化没了。恒真检查的典型形态
#    （memory `hardcoded-conclusions-escape-reproduction`：恒真自检）。
#    改用 `str.isalnum()`（对 CJK 返回 True），保留汉字。
def norm(s):
    return "".join(c for c in s.lower() if c.isalnum())


def read_wav(zf, name):
    """读整条 wav（16-bit PCM）—— 只有 16k 单声道才走这条快路。"""
    with zf.open(name) as f:
        raw = f.read()
    if raw[:4] != b"RIFF":
        return None, None
    p, ch, sr, bits, off = 12, None, None, None, None
    while p < len(raw) - 8:
        cid = raw[p:p + 4]
        sz = struct.unpack("<I", raw[p + 4:p + 8])[0]
        if cid == b"fmt ":
            _, ch, sr, br, ba, bits = struct.unpack("<HHIIHH", raw[p + 8:p + 24])
        elif cid == b"data":
            off = p + 8
            break
        p += 8 + sz + (sz & 1)
    if off is None or sr is None:
        return None, None
    a = np.frombuffer(raw[off:off + sz], dtype="<i2").astype(np.float32) / 32768.0
    return a, sr


def rms(a):
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else 0.0


def stem_of(name):
    b = os.path.basename(name)
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
    ap.add_argument("--audio-n", type=int, default=20,
                    help="Q1 真读音频的对话数（每个场景各取，确定性不随机）")
    ap.add_argument("--dir", default="en_test_nondev/ask")
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.zip)
    names = [i.filename for i in zf.infolist() if i.file_size > 0]
    jset = set(n for n in names if n.endswith(".json"))

    def J(n):
        return json.loads(zf.read(n)) if n in jset else None

    # ---------- P0 全库 id 清单 ----------
    print("=" * 76)
    print("P0 全库 id 形态（[clean_]{id}[_add] × {wav, seg, ts}）")
    print("=" * 76)
    inv = collections.defaultdict(lambda: collections.Counter())
    for n in names:
        d = os.path.dirname(n)
        b = os.path.basename(n)
        st = stem_of(n)
        clean = st.startswith("clean_")
        add = "_add." in b or "_add_timestamp." in b
        kind = ("wav" if b.endswith(".wav")
                else "ts" if b.endswith("_timestamp.json")
                else "seg")
        inv[(d, st, clean, add)][kind] += 1
    pat = collections.Counter()
    for (d, st, clean, add), c in inv.items():
        pat[(clean, add, tuple(sorted(c)))] += 1
    print("  (clean, add, 有的族) -> 多少个 id")
    for k, v in pat.most_common():
        print("    clean=%-5s add=%-5s %-24s %6d" % (k[0], k[1], k[2], v))

    # ---------- Q2 speech_segments vs chunks ----------
    print("\n" + "=" * 76)
    print("Q2 `speech_segments` 与 `chunks` 是不是同一份标注的两种粒度")
    print("=" * 76)
    same, diff, miss = 0, [], 0
    for (d, st, clean, add), c in inv.items():
        if clean or add:
            continue
        # ⚠️ 必须拼上目录 —— jset 里存的是**全路径**，只给基名会全部落进 miss
        #    （第一版就是这么写的，报出「缺一侧 4172」这个假数字）。
        sp = "%s/%s.json" % (d, st)
        tp = "%s/%s_timestamp.json" % (d, st)
        if sp not in jset or tp not in jset:
            miss += 1
            continue
        a = "".join(s["text"] for s in J(sp)["speech_segments"])
        b = J(tp)["text"]
        if norm(a) == norm(b):
            same += 1
        else:
            diff.append((st, a, b))
    print("  非 clean / 非 add 的 id：逐字相等 %d，不等 %d，缺一侧 %d"
          % (same, len(diff), miss))
    if same + len(diff):
        print("  相等率 %.1f%%" % (100 * same / (same + len(diff))))
    for st, a, b in diff[:5]:
        print("   ✗ %s" % st)
        print("      seg : %s" % a[:110])
        print("      ts  : %s" % b[:110])
        r = difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()
        print("      规范后相似度 %.3f" % r)

    # ---------- Q3 X vs X_add ----------
    print("\n" + "=" * 76)
    print("Q3 `X` 与 `X_add` 的 speech_segments 像不像")
    print("=" * 76)
    ratios = []
    for (d, st, clean, add), c in inv.items():
        if clean or not add or "seg" not in c:
            continue
        sp, ap_ = "%s/%s.json" % (d, st), "%s/%s_add.json" % (d, st)
        if sp not in jset or ap_ not in jset:
            continue
        a = "".join(s["text"] for s in J(sp)["speech_segments"])
        b = "".join(s["text"] for s in J(ap_)["speech_segments"])
        ratios.append(difflib.SequenceMatcher(None, norm(a), norm(b)).ratio())
    if ratios:
        r = np.array(ratios)
        print("  n=%d  相似度 中位 %.3f  均值 %.3f  [%.3f, %.3f]"
              % (len(r), np.median(r), r.mean(), r.min(), r.max()))
        print("  完全一样(>0.99)的: %d" % int((r > 0.99).sum()))
        print("  毫不相干(<0.3)的: %d" % int((r < 0.3).sum()))

    # ---------- Q1 clean 是不是分轨 ----------
    print("\n" + "=" * 76)
    print("Q1 🔴 `clean_` 是不是「只剩用户那一侧」（真读音频判，不靠名字）")
    print("=" * 76)
    dirs = sorted({d for (d, _, _, _) in inv})
    picks = []
    for d in dirs:
        ids = sorted(st for (dd, st, cl, ad) in inv
                     if dd == d and not cl and not ad and "seg" in inv[(dd, st, cl, ad)])
        if ids:
            picks.append((d, ids[0]))          # 确定性：每场景第一个
    picks = picks[:args.audio_n]

    print("  每个场景取第 1 个 id（确定性，不随机）：")
    print("  %-38s %8s %8s %8s %8s" %
          ("场景/id", "时长比", "用户RMS", "空隙RMS", "比值"))
    seg_ratios, ratios_q1, dur_ratios = [], [], []
    for d, st in picks:
        w1 = "%s/%s.wav" % (d, st)
        w2 = "%s/clean_%s.wav" % (d, st)
        sp = "%s.json" % st if False else "%s/%s.json" % (d, st)
        if w1 not in set(names) or w2 not in set(names) or sp not in jset:
            print("  %-38s 缺文件" % ("%s/%s" % (d.replace("test/", ""), st)))
            continue
        a1, sr1 = read_wav(zf, w1)
        a2, sr2 = read_wav(zf, w2)
        segs = J(sp)["speech_segments"]
        if a1 is None or a2 is None:
            continue
        dur_ratios.append(len(a2) / len(a1))
        # clean 上：用户区间 vs 空隙区间
        m = np.zeros(len(a2), dtype=bool)
        for s in segs:
            i0, i1 = int(s["xmin"] * sr2), int(s["xmax"] * sr2)
            m[max(0, i0):min(len(a2), i1)] = True
        ru, rg = rms(a2[m]), rms(a2[~m])
        seg_ratios.append((ru, rg))
        ratios_q1.append(rg / ru if ru > 0 else float("nan"))
        print("  %-38s %8.4f %8.4f %8.4f %8.3f"
              % ("%s/%s" % (d.replace("test/", ""), st), len(a2) / len(a1),
                 ru, rg, ratios_q1[-1]))

    # ---------- Q1c 🔴 clean 与 main 逐样本比：是「静音掩码」还是「分出来的轨」 ----------
    #    判据：在 speech_segments 说用户正在说的区间里，
    #      - clean == main（逐样本相等） ⇒ clean 只是把区间外**置零**，是标注的衍生物
    #      - clean != main            ⇒ clean 是另录/分离出来的一路
    #    另一半：main 在区间**外**是不是有能量（助手在说）—— 那才算真双工。
    print("\n  Q1c clean 与 main 逐样本比（用户区间内 / 区间外）：")
    print("  %-38s %12s %12s %12s %12s" %
          ("场景/id", "区间内 maxΔ", "区间内 r", "区间外 mainRMS", "区间外 cleanRMS"))
    inside_same, inside_diff = 0, 0
    for d, st in picks:
        w1 = "%s/%s.wav" % (d, st)
        w2 = "%s/clean_%s.wav" % (d, st)
        sp = "%s/%s.json" % (d, st)
        if w1 not in set(names) or w2 not in set(names) or sp not in jset:
            continue
        a1, sr1 = read_wav(zf, w1)
        a2, sr2 = read_wav(zf, w2)
        if a1 is None or a2 is None:
            continue
        # ⚠️ 两条时长**不严格相等**（clean/main ≈ 1.0004，差 ~12 ms），
        #    所以按 min 截齐再比 —— 第一版写的 `len(a1) != len(a2): continue`
        #    把所有样本都筛掉了，却打印出「相等 0 / 不相等 0」这种
        #    **看起来像结果**的空表（memory：过滤器会制造缺口 / 空表被读成结论）。
        n = min(len(a1), len(a2))
        a1, a2 = a1[:n], a2[:n]
        segs = J(sp)["speech_segments"]
        m = np.zeros(len(a2), dtype=bool)
        for s in segs:
            m[max(0, int(s["xmin"] * sr2)):min(len(a2), int(s["xmax"] * sr2))] = True
        if m.sum() == 0 or (~m).sum() == 0:
            continue
        mx = float(np.abs(a1[m] - a2[m]).max())
        if a1[m].std() > 0 and a2[m].std() > 0:
            r = float(np.corrcoef(a1[m], a2[m])[0, 1])
        else:
            r = float("nan")
        if mx == 0.0:
            inside_same += 1
        else:
            inside_diff += 1
        print("  %-38s %12.6f %12.4f %12.4f %12.4f"
              % ("%s/%s" % (d.replace("test/", ""), st), mx, r,
                 rms(a1[~m]), rms(a2[~m])))
    print("\n  用户区间内逐样本**完全相等** %d 个 / 不相等 %d 个" % (inside_same, inside_diff))
    if inside_same and not inside_diff:
        print("  ⇒ **clean = 把 main 在用户区间外置零**（标注的衍生物，不是独立分轨）。")
        print("     佐证：区间外 clean 的 RMS 恒为 0，而 main 的区间外**有能量**。")
    elif inside_diff and not inside_same:
        print("  ⇒ **clean 是另一路音频**（区间内波形就不同）—— 可能是真分轨，也可能是重录。")
    elif inside_same or inside_diff:
        print("  ⇒ **两种都有** —— 不许一句话概括，要按类报。")

    if ratios_q1:
        arr = np.array([x for x in ratios_q1 if np.isfinite(x)])
        print("\n  clean/main 时长比 中位 %.4f" % float(np.median(dur_ratios)))
        print("  空隙/用户 RMS 比：中位 %.3f  均值 %.3f  [%.3f, %.3f]"
              % (float(np.median(arr)), float(arr.mean()),
                 float(arr.min()), float(arr.max())))
        med = float(np.median(arr))
        if med < 0.25:
            print("  ⇒ **clean ≈ 去掉了助手** —— 空隙基本是静音。这对本项目价值极高。")
        elif med > 0.6:
            print("  ⇒ **clean ≈ 同一条对话的另一种版本**，没有分轨。")
        else:
            print("  ⇒ **落在中间**：既不接近 0 也不接近 1，"
                  "不许二选一 —— 要么加样本、要么按区间分布报。")

    # ---------- Q1b clean 的 timestamp 文本 vs 主版 ----------
    print("\n  Q1b `clean_X_timestamp` 的 text 与 `X_timestamp` 比：")
    cs, cd = 0, 0
    for (d, st, clean, add), c in inv.items():
        if clean or add:
            continue
        t1 = "%s/%s_timestamp.json" % (d, st)
        t2 = "%s/clean_%s_timestamp.json" % (d, st)
        if t1 in jset and t2 in jset:
            if norm(J(t1)["text"]) == norm(J(t2)["text"]):
                cs += 1
            else:
                cd += 1
    print("     非空配对 %d 对：文本逐字相同 %d / 不同 %d" % (cs + cd, cs, cd))
    return 0


if __name__ == "__main__":
    sys.exit(main())
