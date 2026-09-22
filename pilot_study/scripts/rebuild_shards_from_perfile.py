#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 per-file 结果 + 官方 metadata 确定性重建 shard 汇总文件。

背景: 旧版 append_shard_result 断点续跑会重复追加条目, 且多次 kill/重启后
      train 的 shard 文件出现损坏 (JSONDecodeError) 与条目数不一致。
      per-file diarization/transcription JSON 是原子写出的 source of truth,
      因此直接用它们重建汇总。分配规则与 run_real_pipeline 一致
      (samples[SHARD_ID::NUM_SHARDS], 即 idx % NUM_SHARDS), 保证后续
      ONLY_MISSING 续跑兼容。

用法: SPLITS=train python scripts/rebuild_shards_from_perfile.py
"""
import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
REAL = BASE / "real_data"
SPLITS = os.environ.get("SPLITS", "validation,test").split(",")
NUM_SHARDS = 4


def derive_label(record):
    """官方 behaviors 数组 -> Generation Condition 标签。"""
    total_int = sum(b.get("interruptions", 0) for b in record.get("behaviors", []))
    total_bc = sum(b.get("backchannels", 0) for b in record.get("behaviors", []))
    if total_int > 0:
        return "Interruption"
    if total_bc > 0:
        return "Backchannel"
    return "None"


def main():
    for split in SPLITS:
        meta = json.load(open(REAL / f"metadata_{split}.json"))
        res = REAL / "results" / split
        diar_dir, asr_dir = res / "diarization", res / "transcription"
        missing = []
        err_rows = []
        rows_by_shard = {i: [] for i in range(NUM_SHARDS)}
        for idx, rec in enumerate(meta):
            fid = rec["file_id"]
            diar_f, asr_f = diar_dir / f"{fid}.json", asr_dir / f"{fid}.json"
            if not diar_f.exists() or not asr_f.exists():
                missing.append(fid)
                continue
            diar = json.load(open(diar_f))
            asr = json.load(open(asr_f))
            if "error" in diar or "error" in asr:
                err_rows.append(fid)
            row = {
                "file_id": fid,
                "original_label": derive_label(rec),
                "total_interruptions": sum(b.get("interruptions", 0)
                                           for b in rec.get("behaviors", [])),
                "total_backchannels": sum(b.get("backchannels", 0)
                                          for b in rec.get("behaviors", [])),
                "audio_path": str(REAL / "audio" / split / f"{fid}.wav"),
                "diarization": diar,
                "asr": asr,
            }
            rows_by_shard[idx % NUM_SHARDS].append(row)
        for i in range(NUM_SHARDS):
            out = res / f"pilot_results_shard{i}.json"
            json.dump(rows_by_shard[i], open(out, "w"), indent=1, ensure_ascii=False)
            print(f"[{split}] shard{i}: {len(rows_by_shard[i])} 条 -> {out.name}",
                  flush=True)
        total = sum(len(v) for v in rows_by_shard.values())
        print(f"[{split}] ✅ 重建完成: 总 {total} 条 / 应 {len(meta)} 条, "
              f"缺 per-file: {len(missing)}, 含 error: {len(err_rows)}", flush=True)
        for m in missing[:10]:
            print(f"  缺: {m}", flush=True)
        for e in err_rows[:10]:
            print(f"  error: {e}", flush=True)
    print("REBUILD SHARDS DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
