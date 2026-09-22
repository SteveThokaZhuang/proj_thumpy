#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Behavior-SD 真实数据 pipeline (val/test): 说话人分离 + 重叠检测 + Whisper 转写

用法 (每张卡一个 worker, 通过 CUDA_VISIBLE_DEVICES 绑卡):
  SPLIT=validation SHARD_ID=0 NUM_SHARDS=4 python scripts/run_real_pipeline.py
  SPLIT=test       SHARD_ID=1 NUM_SHARDS=4 python scripts/run_real_pipeline.py

特性:
- 分片: 文件列表按 SHARD_ID/NUM_SHARDS 切分, 多卡并行互不冲突
- 断点续跑: diarization + transcription 两个 per-file JSON 都存在则跳过
- per-file 结果实时追加到 shard 结果文件 (临时文件 + rename, 可随时中断)

输出:
  real_data/results/{split}/diarization/{file_id}.json
  real_data/results/{split}/transcription/{file_id}.json
  real_data/results/{split}/pilot_results_shard{SHARD_ID}.json
"""

import json
import os
import tempfile
from pathlib import Path

import torch
import whisper
from pyannote.audio import Pipeline

# =================配置=================
BASE_DIR = Path(__file__).resolve().parents[1]  # pilot_study/
REAL_DIR = BASE_DIR / "real_data"
HF_TOKEN = os.environ.get("HF_TOKEN")
SPLIT = os.environ.get("SPLIT", "validation")
SHARD_ID = int(os.environ.get("SHARD_ID", "0"))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")

METADATA_FILE = REAL_DIR / f"metadata_{SPLIT}.json"
AUDIO_DIR = REAL_DIR / "audio" / SPLIT
RESULTS_DIR = REAL_DIR / "results" / SPLIT
DIAR_DIR = RESULTS_DIR / "diarization"
ASR_DIR = RESULTS_DIR / "transcription"
SHARD_FILE = RESULTS_DIR / f"pilot_results_shard{SHARD_ID}.json"

for d in [DIAR_DIR, ASR_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =================加载模型=================
print(f"[shard {SHARD_ID}/{NUM_SHARDS}] Loading pyannote pipeline...", flush=True)
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", token=HF_TOKEN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
diarization_pipeline.to(device)

print(f"Loading Whisper model ({WHISPER_MODEL})...", flush=True)
asr_model = whisper.load_model(WHISPER_MODEL)
if torch.cuda.is_available():
    asr_model = asr_model.cuda()

# =================处理函数=================
def extract_overlaps(diarization):
    """从 Annotation 提取不同说话人话轮的两两重叠区域 (>100ms)。"""
    overlaps = []
    tracks = list(diarization.itertracks(yield_label=True))
    for i, (turn_a, _, spk_a) in enumerate(tracks):
        for j, (turn_b, _, spk_b) in enumerate(tracks):
            if i < j and spk_a != spk_b:
                inter = turn_a & turn_b
                if inter.duration > 0.1:
                    overlaps.append({
                        "start": inter.start,
                        "end": inter.end,
                        "duration": inter.duration,
                        "speakers": [spk_a, spk_b],
                    })
    return overlaps, tracks


def derive_label(record):
    """behaviors 数组 -> Generation Condition 标签 (兼容两种 metadata 格式)。"""
    total_int = sum(b.get("interruptions", 0) for b in record.get("behaviors", []))
    total_bc = sum(b.get("backchannels", 0) for b in record.get("behaviors", []))
    if total_int > 0:
        return "Interruption"
    if total_bc > 0:
        return "Backchannel"
    return "None"


def process_sample(sample):
    file_id = sample["file_id"]
    audio_path = AUDIO_DIR / f"{file_id}.wav"
    diar_out = DIAR_DIR / f"{file_id}.json"
    asr_out = ASR_DIR / f"{file_id}.json"
    result = {
        "file_id": file_id,
        # 兼容官方格式 (behaviors) 与 reduced 格式 (直接字段)
        "original_label": sample.get("original_label") or derive_label(sample),
        "total_interruptions": sample.get("total_interruptions")
        if sample.get("total_interruptions") is not None
        else sum(b.get("interruptions", 0) for b in sample.get("behaviors", [])),
        "total_backchannels": sample.get("total_backchannels")
        if sample.get("total_backchannels") is not None
        else sum(b.get("backchannels", 0) for b in sample.get("behaviors", [])),
        "audio_path": str(audio_path),
    }

    # 1. 说话人分离 (断点: 已产出则跳过)
    if diar_out.exists():
        with open(diar_out) as f:
            result["diarization"] = json.load(f)
    else:
        try:
            diarization = diarization_pipeline(str(audio_path))
            # pyannote 4.x 返回 DiarizeOutput, 取含重叠话轮的 Annotation
            if hasattr(diarization, "speaker_diarization"):
                diarization = diarization.speaker_diarization
            overlaps, tracks = extract_overlaps(diarization)
            result["diarization"] = {
                "num_speakers": len({spk for _, _, spk in tracks}),
                "total_duration": sum(t[0].duration for t in tracks),
                "overlaps": overlaps,
                "total_overlap_duration": sum(o["duration"] for o in overlaps),
            }
            tmp = tempfile.NamedTemporaryFile(
                "w", dir=DIAR_DIR, suffix=".tmp", delete=False)
            json.dump(result["diarization"], tmp, indent=1)
            tmp.close()
            Path(tmp.name).replace(diar_out)
        except Exception as e:
            print(f"❌ Diarization failed {file_id}: {e}", flush=True)
            result["diarization"] = {"error": str(e)}

    # 2. Whisper 转写 (断点: 已产出则跳过)
    if asr_out.exists():
        with open(asr_out) as f:
            result["asr"] = json.load(f)
    else:
        try:
            transcription = asr_model.transcribe(
                str(audio_path), word_timestamps=True, verbose=False)
            words = [
                {"text": w["word"], "start": w["start"], "end": w["end"]}
                for seg in transcription.get("segments", [])
                for w in seg.get("words", [])
            ]
            result["asr"] = {
                "text": transcription.get("text", "").strip(),
                "language": transcription.get("language", "unknown"),
                "words": words,
                "num_words": len(words),
            }
            tmp = tempfile.NamedTemporaryFile(
                "w", dir=ASR_DIR, suffix=".tmp", delete=False)
            json.dump(result["asr"], tmp, indent=1, ensure_ascii=False)
            tmp.close()
            Path(tmp.name).replace(asr_out)
        except Exception as e:
            print(f"❌ ASR failed {file_id}: {e}", flush=True)
            result["asr"] = {"error": str(e)}

    return result


def append_shard_result(result):
    """追加一条到 shard 结果文件 (临时文件 + rename, 保证原子性)。"""
    results = []
    if SHARD_FILE.exists():
        try:
            with open(SHARD_FILE) as f:
                results = json.load(f)
        except json.JSONDecodeError:
            results = []
    results.append(result)
    tmp = tempfile.NamedTemporaryFile("w", dir=RESULTS_DIR, suffix=".tmp", delete=False)
    json.dump(results, tmp, indent=1, ensure_ascii=False)
    tmp.close()
    Path(tmp.name).replace(SHARD_FILE)


# =================主流程=================
def main():
    with open(METADATA_FILE) as f:
        samples = json.load(f)

    # 分片 (轮转分配, 保证各卡样本时长均衡)
    shard_samples = samples[SHARD_ID::NUM_SHARDS]
    print(f"[shard {SHARD_ID}/{NUM_SHARDS}] {len(shard_samples)} 条样本 "
          f"(split={SPLIT}, 总 {len(samples)} 条)", flush=True)

    # ONLY_MISSING=1: 只处理 per-file 结果缺失的样本 (收尾模式)
    if os.environ.get("ONLY_MISSING"):
        def _missing(s):
            fid = s["file_id"]
            return (not (DIAR_DIR / f"{fid}.json").exists()
                    or not (ASR_DIR / f"{fid}.json").exists())
        shard_samples = [s for s in shard_samples if _missing(s)]
        print(f"[shard {SHARD_ID}] ONLY_MISSING 过滤后: {len(shard_samples)} 条待处理",
              flush=True)

    # 断点续跑: 已存在于 shard 结果中的 file_id 不再追加 (避免重启产生重复条目)
    seen_ids = set()
    if SHARD_FILE.exists():
        try:
            with open(SHARD_FILE) as f:
                seen_ids = {r.get("file_id") for r in json.load(f)}
        except (json.JSONDecodeError, OSError):
            seen_ids = set()

    for idx, sample in enumerate(shard_samples, 1):
        result = process_sample(sample)
        if result.get("file_id") not in seen_ids:
            append_shard_result(result)
            seen_ids.add(result.get("file_id"))
        if idx % 10 == 0 or idx == len(shard_samples):
            print(f"[shard {SHARD_ID}] 进度 {idx}/{len(shard_samples)}", flush=True)

    print(f"[shard {SHARD_ID}] ✅ 完成, 结果 -> {SHARD_FILE}", flush=True)


if __name__ == "__main__":
    main()
