#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 file_id 去重 shard 结果文件 (修复旧版续跑重复追加的条目)。

用法: python scripts/dedup_shards.py train   (split 名, 默认 train)
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "real_data" / "results"
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "train"


def main():
    for shard in sorted((RESULTS / SPLIT).glob("pilot_results_shard*.json")):
        data = json.load(open(shard))
        seen = set()
        deduped = []
        for r in data:
            fid = r.get("file_id")
            if fid in seen:
                continue
            seen.add(fid)
            deduped.append(r)
        removed = len(data) - len(deduped)
        json.dump(deduped, open(shard, "w"), indent=1, ensure_ascii=False)
        print(f"{shard.name}: {len(data)} -> {len(deduped)} 条 (移除重复 {removed})",
              flush=True)
    print("DEDUP DONE", flush=True)


if __name__ == "__main__":
    main()
