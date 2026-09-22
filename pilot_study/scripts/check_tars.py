#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 shared_dataset 内 tar 完整性, 并演示不解压直接从 tar 内读取 flac"""
import io
import tarfile
from pathlib import Path

import librosa

SRC = Path("/share/workspace3/shared_dataset/behavior-sd")


def main():
    for split in ["validation", "test", "train"]:
        tars = sorted((SRC / split).glob("*.tar"))
        n_flac = n_json = 0
        bad = []
        for tp in tars:
            try:
                with tarfile.open(tp) as t:
                    names = t.getnames()
                n_flac += sum(1 for n in names if n.endswith(".flac"))
                n_json += sum(1 for n in names if n.endswith(".json"))
            except Exception as e:  # noqa: BLE001
                bad.append((tp.name, repr(e)))
        print(f"[{split}] {len(tars)} 个 tar: 内含 {n_json} json + {n_flac} flac, "
              f"坏 tar: {len(bad)}", flush=True)
        for b in bad[:3]:
            print(f"  BAD: {b}", flush=True)

    # 演示: 不解压, 流式读出一个 flac 并解码
    with tarfile.open(SRC / "test" / "0000.tar") as t:
        raw = t.extractfile("0000000003.flac").read()
    y, sr = librosa.load(io.BytesIO(raw), sr=None, mono=False)
    print(f"\n演示: tar 内流式读出 0000000003.flac ({len(raw)} 字节) "
          f"-> 解码 {y.shape}, {sr}Hz (官方参数 22050Hz 立体声)", flush=True)
    print("TAR CHECK DONE", flush=True)


if __name__ == "__main__":
    main()
