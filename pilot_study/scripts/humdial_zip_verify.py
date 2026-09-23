"""下载完整性核对：**不采信下载日志**，直接拿 zip 自己的中央目录跟 HF 的清单对。

## 为什么要这一步

`download.log` 只写了一行 `rc=7`（curl 连不上），**那不是一次成功的记录** ——
zip 是后来另一个进程写的，**没有任何日志为它背书**。本项目反复吃过
「拿一次失败的日志/一次没跑完的批当成功」（memory `sd-from-few-points-manufactures-anomalies`
的六种形态之一：**跑批没跑完**）。

zip 的中央目录在**文件末尾**。若下载被截断，要么中央目录根本读不出来，
要么中央目录里的条目数/偏移对不上 HF 的清单。

## 判据（跑之前写死）

- **P1** `zipfile` 能读出中央目录，且**条目数 == ziplist.json 的条目数**。
- **P2** 逐条比 `(name, file_size)` —— **任何一条对不上都要列出来**，不许只报计数。
- **P3** 最后一个条目的 `offset + csz` 必须 ≤ 实际文件字节数
  （即中央目录真的落在文件里，不是半个）。
- **P4** 抽查 **20 个**条目做真 CRC 校验（`ZipFile.testzip` 太慢，1.8 GB 全量 CRC 单独跑）。

只查结构不查 CRC ⇒ **P1–P3 全绿也只说明"目录自洽"**，所以补 P4 抽真读。

用法（srun 内，纯读盘）：
  python scripts/humdial_zip_verify.py \
      --zip /share/workspace3/shared_dataset/humdial-fdbench/Humdial-Track2-Test.zip \
      --list /share/workspace3/shared_dataset/humdial-fdbench/ziplist.json
"""
import argparse
import json
import os
import random
import sys
import zipfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--crc-n", type=int, default=20)
    ap.add_argument("--crc-seed", type=int, default=0)
    args = ap.parse_args()

    real = os.path.getsize(args.zip)
    ref = json.load(open(args.list))
    ref_map = {e["name"]: e["usz"] for e in ref}
    print("zip 实际字节   = %d" % real)
    print("HF 清单条目数  = %d" % len(ref))
    print("HF 清单总字节  = %d" % sum(e["usz"] for e in ref))
    print("HF 清单最后偏移= %d" % max(e["off"] + e["csz"] for e in ref))

    ok = True

    # ---------- P1 ----------
    try:
        zf = zipfile.ZipFile(args.zip)
        infos = zf.infolist()
    except Exception as ex:
        print("\n🔴 P1 失败：中央目录读不出来 —— %r" % (ex,))
        print("   ⇒ 下载**被截断**，必须重下。不要试图用它。")
        return 1
    print("\nP1 中央目录条目数 = %d" % len(infos))
    if len(infos) != len(ref):
        print("   🔴 与清单不符（差 %d）" % (len(infos) - len(ref)))
        ok = False
    else:
        print("   ✅ 与清单一致")

    # ---------- P2 ----------
    zip_map = {i.filename: i.file_size for i in infos}
    only_ref = [n for n in ref_map if n not in zip_map]
    only_zip = [n for n in zip_map if n not in ref_map]
    size_bad = [(n, ref_map[n], zip_map[n]) for n in ref_map
                if n in zip_map and ref_map[n] != zip_map[n]]
    print("\nP2 逐条比对 (name, usz):")
    print("   只在清单里 : %d" % len(only_ref))
    print("   只在 zip 里: %d" % len(only_zip))
    print("   大小不符   : %d" % len(size_bad))
    for lab, lst in (("只在清单", only_ref[:5]), ("只在 zip", only_zip[:5]),
                     ("大小不符", size_bad[:5])):
        for x in lst:
            print("      %s: %s" % (lab, x))
    if only_ref or only_zip or size_bad:
        ok = False
        print("   🔴 有问题")
    else:
        print("   ✅ 逐条一致")

    # ---------- P3 ----------
    last = max(ref, key=lambda e: e["off"] + e["csz"])
    end = last["off"] + last["csz"]
    print("\nP3 清单里最远的字节位置 = %d  (zip 实际 %d)" % (end, real))
    if end > real:
        print("   🔴 清单声明的数据超出文件末尾 —— 文件是半截的")
        ok = False
    else:
        print("   ✅ 落在文件内（尾隙 %d 字节 = 中央目录）" % (real - end))

    # ---------- P4 ----------
    print("\nP4 抽 %d 个条目真读 + CRC:" % args.crc_n)
    rng = random.Random(args.crc_seed)
    cand = [i.filename for i in infos if i.file_size > 0]
    pick = rng.sample(cand, min(args.crc_n, len(cand)))
    p4bad = []
    for nm in pick:
        try:
            zf.read(nm)
        except Exception as ex:
            p4bad.append((nm, repr(ex)[:80]))
    print("   读了 %d 个，失败 %d 个" % (len(pick), len(p4bad)))
    for nm, e in p4bad[:5]:
        print("      🔴 %s -> %s" % (nm, e))
    if p4bad:
        ok = False
    else:
        print("   ✅ 全部解压且 CRC 通过")

    print("\n" + "=" * 60)
    if ok:
        print("✅ 结构 + 抽样 CRC 全过 —— 可以用。")
        print("   ⚠️ 仍未做全量 CRC；若结论依赖某个具体文件，读它时会自然校验。")
    else:
        print("🔴 有问题 —— 先解决，别往下跑。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
