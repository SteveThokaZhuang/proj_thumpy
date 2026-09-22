"""E3 冒烟: X2-Turn 本地权重加载 + 短音频推理."""
import sys
import time

import torch
from transformers import AutoProcessor

from voxtral_realtime.transformers.inference import infer_asr_turn
from voxtral_realtime.transformers.modeling import load_mtp_checkpoint

MODEL_DIR = "/share/workspace3/shared_models/X2-Turn-4B-0812"
AUDIO = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator/tmp/e3_test.wav"


def main():
    t0 = time.time()
    print("loading model...", flush=True)
    model = load_mtp_checkpoint(MODEL_DIR, device="cuda", dtype=torch.bfloat16)
    model.eval()
    print(f"model loaded {time.time()-t0:.1f}s, "
          f"vram {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_DIR)

    t0 = time.time()
    result = infer_asr_turn(model, processor, AUDIO)
    print(f"infer {time.time()-t0:.1f}s", flush=True)
    print("transcript:", result.transcript[:200])
    print(f"frames: {len(result.turn_frames)}, frame_ms={result.frame_ms}")
    from collections import Counter
    print("label counts:", Counter(f.label for f in result.turn_frames))
    for f in result.turn_frames[:10]:
        print(f"  {f.start_ms:6d}-{f.end_ms:6d} {f.label:12s} "
              f"conf={f.confidence:.2f} probs={ {k: round(v,2) for k,v in f.probabilities.items()} }")


if __name__ == "__main__":
    main()
