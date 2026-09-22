#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 shared_dataset tar 提取音频: flac (22050 立体声) -> 16k mono wav (模型标准输入)

背景: 原 prepare_real_data.py 同时做 metadata 生成, 但那段逻辑丢弃嵌套
      backchannels 字段, 已被 rebuild_metadata_full.py 取代 (官方原始格式)。
      本脚本只保留其音频部分 (解包 + 并行转码), 供按需重新提取。

环境变量:
  SPLITS      要处理的 split, 逗号分隔 (默认 "validation,test")
  TAR_LIMIT   每个 split 只处理前 N 个 tar (默认全部; train 前 64 tar ≈ 30%)
  CONVERT_JOBS 并行转码进程数 (默认 12)

输出: real_data/audio/{split}/{split}_{soda_id}.wav
      命名与 run_real_pipeline.py 的 AUDIO_DIR/f"{file_id}.wav" 一致。
"""
import os
import shutil
import subprocess
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SRC_DIR = Path("/share/workspace3/shared_dataset/behavior-sd")
OUT_DIR = Path(__file__).resolve().parents[1] / "real_data"
SPLITS = os.environ.get("SPLITS", "validation,test").split(",")
TAR_LIMIT = int(os.environ.get("TAR_LIMIT", "0")) or None
CONVERT_JOBS = int(os.environ.get("CONVERT_JOBS", "12"))

ENV_BIN = Path.home() / "miniconda3/envs/fd_pilot/bin"
FFMPEG = shutil.which("ffmpeg", path=f"{ENV_BIN}:{os.environ.get('PATH', '')}") or "ffmpeg"


def convert_one(args):
    """(flac_path, wav_path) -> ffmpeg 转码 16k mono"""
    flac_path, wav_path = args
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-i", str(flac_path),
         "-ac", "1", "-ar", "16000", str(wav_path)],
        check=True,
    )
    return wav_path


def main():
    for split in SPLITS:
        tar_dir = SRC_DIR / split
        audio_out = OUT_DIR / "audio" / split
        audio_out.mkdir(parents=True, exist_ok=True)
        staging = OUT_DIR / f"_staging_audio_{split}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        tars = sorted(tar_dir.glob("*.tar"))
        if TAR_LIMIT:
            tars = tars[:TAR_LIMIT]
        print(f"[{split}] 处理 {len(tars)} 个 tar (TAR_LIMIT={TAR_LIMIT})", flush=True)

        # 解包 flac 到 staging (json 不需要, 跳过)
        for tar_path in tars:
            with tarfile.open(tar_path) as tar:
                for m in tar.getmembers():
                    if m.isfile() and m.name.endswith(".flac"):
                        tar.extract(m, staging)

        # 并行转码 (已存在的 wav 跳过)
        tasks = []
        for flac in sorted(staging.glob("*.flac")):
            wav_path = audio_out / f"{split}_{flac.stem}.wav"
            if not wav_path.exists():
                tasks.append((flac, wav_path))
        print(f"  转码 {len(tasks)} 个 flac ({CONVERT_JOBS} 并行)...", flush=True)
        with ProcessPoolExecutor(max_workers=CONVERT_JOBS) as ex:
            for i, _ in enumerate(ex.map(convert_one, tasks), 1):
                if i % 2000 == 0:
                    print(f"  转码进度 {i}/{len(tasks)}", flush=True)

        shutil.rmtree(staging, ignore_errors=True)
        print(f"[{split}] ✅ 音频 -> {audio_out}", flush=True)
    print("EXTRACT AUDIO DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
