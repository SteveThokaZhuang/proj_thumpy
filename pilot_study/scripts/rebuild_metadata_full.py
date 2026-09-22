#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按官方原始格式重建 metadata (保留全部官方字段, 仅附加 file_id)

背景: 旧 metadata (prepare_real_data.py 产出) 只保留顶层 utterances,
      丢弃了嵌套的 backchannels (含各自时间戳) 等字段, 导致 GT 重叠计算错误。
      本脚本从 shared_dataset 的 tar 中重新解出官方 JSON, 原样保留,
      仅附加 "file_id": "{split}_{soda_id}" 用于与 pipeline 结果 join。

环境变量: SPLITS (默认 "validation,test")
输出: real_data/metadata_{split}.json = [官方 JSON + file_id]
"""
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

SRC_DIR = Path("/share/workspace3/shared_dataset/behavior-sd")
OUT_DIR = Path(__file__).resolve().parents[1] / "real_data"
SPLITS = os.environ.get("SPLITS", "validation,test").split(",")
TAR_LIMIT = int(os.environ.get("TAR_LIMIT", "0")) or None  # 0/缺省 = 全部


def main():
    for split in SPLITS:
        staging = OUT_DIR / f"_staging_json_{split}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        tars = sorted((SRC_DIR / split).glob("*.tar"))
        if TAR_LIMIT:
            tars = tars[:TAR_LIMIT]
        print(f"[{split}] 从 {len(tars)} 个 tar 解出 JSON... (TAR_LIMIT={TAR_LIMIT})",
              flush=True)
        for tp in tars:
            with tarfile.open(tp) as tar:
                for m in tar.getmembers():
                    if m.isfile() and m.name.endswith(".json"):
                        tar.extract(m, staging)

        records = []
        for p in sorted(staging.glob("*.json")):
            with open(p, encoding="utf-8") as f:
                rec = json.load(f)
            rec["file_id"] = f"{split}_{p.stem}"  # 唯一附加字段
            records.append(rec)
        records.sort(key=lambda r: (r.get("soda_index", 0), r.get("file_id", "")))

        out = OUT_DIR / f"metadata_{split}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=1)
        # 字段完整性抽查
        missing = []
        for rec in records:
            for u in rec.get("utterances", []):
                if "backchannels" not in u:
                    missing.append(rec["file_id"])
                    break
        print(f"[{split}] ✅ {len(records)} 条 (官方原始格式) -> {out}", flush=True)
        print(f"[{split}] 缺 backchannels 字段的文件: {len(missing)}", flush=True)
        shutil.rmtree(staging, ignore_errors=True)
    print("REBUILD DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
