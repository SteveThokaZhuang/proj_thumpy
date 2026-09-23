"""HumDial-FDBench test 集的**字段表勘察** —— 回答 recon §5.5「它接 L1 还是接 L3」。

## 为什么要有这个脚本

recon §4 的表把三件事记成了「事实」，但**来源是 GitHub/HF 的说明文字**，不是数据：

| §4 的断言 | 本次要验的 |
|---|---|
| 「真实人类录音、**双声道**」 | zip 里每个 wav 的 `fmt` 块到底几个声道 |
| 「test 5,000 条」 | zip 里实际有多少条 |
| 「两类 **9** 子场景」 | 实际有多少个场景目录 |
| 「README 没给字段表」 | 逐 json 统计键形状，把字段表**打出来** |

本项目反复吃过「描述物 ≠ 被描述物」（memory `config-differs-from-config-file`），
所以这里**一律从 zip 字节里现算**，不引用任何说明文字。

## 判据（写在这里，跑之前定好）

- **C1 字段表**：所有 `.json` 的顶层键形状必须收敛到有限几种。若出现许多种形状 ⇒
  说明存在我们没发现的标注种类，**必须逐个 print 出来**，不许只报最常见的。
- **C2 音频布局**：对**每一个** wav 读 `fmt` 块（不是抽样）。报告 (ch, sr, bits) 的
  分布；**只要有一种不是 (1, 16000, 16)，就必须单独列出**。
- **C3 段数分布**：`len(speech_segments)` 的直方图。若恒为 2，那「2 段 + 中间空隙」
  就是这套数据的**结构**（空隙 = 模型该说话的地方），可以直接写进结论。
- **C4 自洽**：`final_duration` 与同 id wav 的**实际时长**之比。若普遍 ≈1，
  说明 `final_duration` 就是音频长度，`speech_segments` 的时间轴与之同源。
- **C5 `_add` 是什么**：`_add` 变体的覆盖率，以及它的 `speech_segments` 与
  非 `_add` 版本**是否相同**。相同 ⇒ `_add` 只换音频不换标注。

⚠️ 本脚本**只做勘察，不写任何实验设计**（recon §5.5 的明文要求）。

用法（纯读盘，0 GPU）：
  python scripts/humdial_schema_probe.py --zip /share/workspace3/shared_dataset/humdial-fdbench/Humdial-Track2-Test.zip
"""
import argparse
import collections
import io
import json
import os
import re
import struct
import sys
import wave
import zipfile


def wav_fmt(zf, name):
    """只读 wav 的 fmt 块 —— 不把整条音频读进来。"""
    with zf.open(name) as f:
        head = f.read(1024)
    if head[:4] != b"RIFF":
        return None, "非 RIFF: %r" % head[:8]
    p = 12
    while p < len(head) - 8:
        cid = head[p:p + 4]
        sz = struct.unpack("<I", head[p + 4:p + 8])[0]
        if cid == b"fmt ":
            fmt, ch, sr, br, ba, bits = struct.unpack("<HHIIHH", head[p + 8:p + 24])
            return (ch, sr, bits, fmt), None
        p += 8 + sz + (sz & 1)
    return None, "没有 fmt 块"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--max-json", type=int, default=0,
                    help=">0 时只解析前 N 个 json（调试用）；0 = 全解析")
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.zip)
    info = [i for i in zf.infolist() if i.file_size > 0]
    print("=" * 72)
    print("zip: %s" % args.zip)
    print("条目 %d（非空），解压后 %.2f GB" % (
        len(info), sum(i.file_size for i in info) / 1e9))

    by_dir = collections.defaultdict(lambda: collections.Counter())
    for i in info:
        d = os.path.dirname(i.filename)
        b = os.path.basename(i.filename)
        e = os.path.splitext(b)[1].lower()
        if e == ".wav":
            k = "wav_add" if b.endswith("_add.wav") else "wav"
        elif e == ".json":
            for suf, kk in (("_add_timestamp.json", "add_timestamp"),
                            ("_timestamp.json", "timestamp"),
                            ("_add.json", "add"),
                            (".json", "plain")):
                if b.endswith(suf):
                    k = kk
                    break
        else:
            k = "other"
        by_dir[d][k] += 1

    print("\n--- 目录清单（%d 个）---" % len(by_dir))
    kinds = ["plain", "add", "timestamp", "add_timestamp", "wav", "wav_add", "other"]
    print("%-44s %s" % ("目录", " ".join("%10s" % k for k in kinds)))
    for d in sorted(by_dir):
        c = by_dir[d]
        print("%-44s %s" % (d.replace("test/", ""),
                            " ".join("%10d" % c[k] for k in kinds)))

    # ---------- C1 字段表 ----------
    print("\n=== C1 字段表：所有 json 的顶层键形状 ===")
    shapes = collections.defaultdict(collections.Counter)
    seg_hist = collections.Counter()
    add_same, add_diff, add_pairs = 0, 0, 0
    nseg_by_kind = collections.defaultdict(collections.Counter)
    jsons = [i.filename for i in info if i.filename.endswith(".json")]
    if args.max_json:
        jsons = jsons[:args.max_json]
    parsed = {}
    for n, nm in enumerate(jsons):
        try:
            o = json.loads(zf.read(nm).decode("utf-8", "replace"))
        except Exception as ex:
            shapes["<解析失败>"][repr(ex)[:60]] += 1
            continue
        b = os.path.basename(nm)
        kind = ("add_timestamp" if b.endswith("_add_timestamp.json")
                else "timestamp" if b.endswith("_timestamp.json")
                else "add" if b.endswith("_add.json") else "plain")
        shape = tuple(sorted(o)) if isinstance(o, dict) else ("<非 dict>",)
        shapes[kind][shape] += 1
        parsed[nm] = o
        if "speech_segments" in (o if isinstance(o, dict) else {}):
            ns = len(o["speech_segments"])
            seg_hist[ns] += 1
            nseg_by_kind[kind][ns] += 1
    for kind in ("plain", "add", "timestamp", "add_timestamp"):
        if kind not in shapes:
            continue
        print("  [%s]" % kind)
        for shape, c in shapes[kind].most_common():
            print("     %-38s x%d" % (str(shape), c))
    if "<解析失败>" in shapes:
        print("  🔴 有解析失败的 json —— 必须逐个看")

    # ---------- C3 段数 ----------
    print("\n=== C3 speech_segments 段数直方图 ===")
    for k, v in sorted(seg_hist.items()):
        print("    %d 段: %d 条" % (k, v))
    if len(seg_hist) > 1:
        print("  ⚠️ 不止一种段数 —— 逐 kind 看：")
        for kind, c in nseg_by_kind.items():
            print("     %-14s %s" % (kind, dict(c)))

    # ---------- C5 _add 是否只是换音频 ----------
    print("\n=== C5 `_add` 变体与主版本是否同标注 ===")
    for nm, o in parsed.items():
        if not nm.endswith("_add.json"):
            continue
        base = nm[:-len("_add.json")] + ".json"
        if base not in parsed:
            continue
        add_pairs += 1
        if parsed[base].get("speech_segments") == o.get("speech_segments"):
            add_same += 1
        else:
            add_diff += 1
    if add_pairs:
        print("    能配上对的 %d 对：speech_segments **逐条相同** %d / **不同** %d"
              % (add_pairs, add_same, add_diff))
        if add_diff:
            ex = [nm for nm in parsed if nm.endswith("_add.json")
                  and nm[:-len("_add.json")] + ".json" in parsed
                  and parsed[nm[:-len("_add.json")] + ".json"].get("speech_segments")
                  != parsed[nm].get("speech_segments")][:1]
            if ex:
                b = ex[0][:-len("_add.json")] + ".json"
                print("    例：%s" % ex[0].replace("test/", ""))
                print("      主版: %s" % json.dumps(
                    parsed[b]["speech_segments"], ensure_ascii=False)[:200])
                print("      add : %s" % json.dumps(
                    parsed[ex[0]]["speech_segments"], ensure_ascii=False)[:200])
    else:
        print("    ⚠️ 一对都没配上 —— 检查命名假设")

    # ---------- C2 音频布局（全量，不抽样） ----------
    print("\n=== C2 音频布局：逐个 wav 读 fmt（不是抽样）===")
    fmts = collections.Counter()
    bad = []
    ch_by_dir = collections.defaultdict(collections.Counter)
    wavs = [i.filename for i in info if i.filename.endswith(".wav")]
    for nm in wavs:
        f, err = wav_fmt(zf, nm)
        if f is None:
            bad.append((nm, err))
            continue
        fmts[f[:3]] += 1
        ch_by_dir[os.path.dirname(nm)][f[0]] += 1
    for k, v in fmts.most_common():
        print("    ch=%d sr=%d bits=%d : %d 个" % (k[0], k[1], k[2], v))
    if len(fmts) > 1:
        print("  🔴 音频格式不唯一 —— 必须列出异常目录")
        for d in sorted(ch_by_dir):
            if len(ch_by_dir[d]) > 1 or list(ch_by_dir[d])[0] != 1:
                print("     %s -> %s" % (d, dict(ch_by_dir[d])))
    if bad:
        print("  🔴 读不出 fmt 的 %d 个（前 5）: %s" % (len(bad), bad[:5]))

    # ---------- C4 final_duration vs 实际时长 ----------
    print("\n=== C4 final_duration 与 wav 实际时长之比 ===")
    names = set(zf.namelist())          # ⚠️ 别在循环里调 namelist(): 每次都是 O(n)
    ratios, missing = [], 0
    for nm, o in parsed.items():
        if not isinstance(o, dict) or "final_duration" not in o:
            continue
        w = nm[:-5] + ".wav"
        if w not in names:
            missing += 1
            continue
        with zf.open(w) as f:
            head = f.read(64)
        if head[:4] != b"RIFF":
            continue
        p = 12
        ch = sr = bits = None
        while p < len(head) - 8:
            cid = head[p:p + 4]
            sz = struct.unpack("<I", head[p + 4:p + 8])[0]
            if cid == b"fmt ":
                _, ch, sr, br, ba, bits = struct.unpack("<HHIIHH", head[p + 8:p + 24])
                break
            p += 8 + sz + (sz & 1)
        if not sr:
            continue
        dur = zf.getinfo(w).file_size / (sr * ch * bits // 8)
        ratios.append(o["final_duration"] / dur)
    if ratios:
        import statistics
        ratios.sort()
        print("    n=%d  中位 %.4f  [%.4f, %.4f]" % (
            len(ratios), statistics.median(ratios), ratios[0], ratios[-1]))
        near = sum(1 for r in ratios if abs(r - 1) < 0.02)
        print("    |比 − 1| < 0.02 的占 %d/%d = %.1f%%"
              % (near, len(ratios), 100 * near / len(ratios)))
    if missing:
        print("    ⚠️ %d 个 json 没有同名 wav" % missing)
    print("\n(以上全部由 zip 字节现算；§4 的说明文字一律未采信)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
