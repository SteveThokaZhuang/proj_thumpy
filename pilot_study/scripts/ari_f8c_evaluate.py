"""F8c 评估: 事件式模型的行为级命中 (±0.5s vs realized 真重叠 BC 窗口).

对比: 基座 vs f8c 事件式 LoRA vs real_12k 词序列 LoRA (同冻结评估集).
指标: 事件 precision/recall/F1 (预测 "X.Xs" 落在真重叠 AWS BC 窗口 ±0.5s 内).

用法 (funaudiochat 环境, gpu04):
  python scripts/ari_f8c_evaluate.py
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_f8_evaluate import ANNOT, load_model, MODEL_DIR  # noqa: E402
from ari_f7b_framewise import extend_eval_set, chunk_ov_mask  # noqa: E402
from ari_analyze import DEFAULT_OUT  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from funaudiochat.register import register_funaudiochat  # noqa: E402
register_funaudiochat()

import librosa  # noqa: E402
import torch  # noqa: E402

INSTRUCTION = ("When does the listener produce backchannels in this segment? "
               "List the times in seconds, or answer 'no backchannel'.")
AUDIO_TEMPLATE = "<|audio_bos|><|AUDIO|><|audio_eos|>"
MODELS = [
    ("base", None),
    ("f8c_events", f"{ANNOT}/f8c_sft/saves"),
    ("real_12k_words", f"{ANNOT}/f8_sft_expanded/saves"),
]


def predict_times(model, processor, wav):
    audio = [librosa.load(wav, sr=16000)[0]]
    conversation = [{"role": "system", "content": ""},
                    {"role": "user", "content": AUDIO_TEMPLATE + INSTRUCTION}]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audio, return_tensors="pt",
                       return_token_type_ids=False).to(model.device)
    with torch.no_grad():
        generate_ids, _ = model.generate(**inputs, max_new_tokens=128)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    out = processor.decode(generate_ids[0], skip_special_tokens=True)
    # 只匹配 "X.Xs" (时间带 s 后缀), 避免把置信度 "(0.85)" 误当时间
    times = [float(x) for x in re.findall(r"(\d+\.?\d*)s", out)
             if 0 <= float(x) <= 10.5]
    return times


def ov_windows(cid):
    """chunk 的真重叠 AWS BC 窗口 (会话时间) -> [(start, end)]."""
    with open(f"{ANNOT}/f8_training/manifest.jsonl") as f:
        mlook = {json.loads(l)["id"]: json.loads(l) for l in f}
    m = mlook[cid]
    ev = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_events_w*.csv"))])
    e1 = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_e1_w*.csv"))])
    e1 = e1.merge(ev[["session", "event_id", "cls", "ch_event", "start", "end"]],
                  on="event_id", suffixes=("", "_ev"))
    sub = e1[(e1["session"] == m["session"]) & (e1["cls"] == "BC")
             & (e1["ch_event"] == int(m["ch"])) & (e1["realized"] != "None")]
    return [(r["start"], r["end"]) for _, r in sub.iterrows()
            if r["end"] > m["t0"] and r["start"] < m["t0"] + 10]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-lora", type=str, default="")
    args = ap.parse_args()
    models = list(MODELS)
    if args.extra_lora:
        models.append(("extra", args.extra_lora))
    eval_set = extend_eval_set(300)
    ids = [os.path.basename(w)[:-4] for (w, _) in eval_set]
    processor = AutoProcessor.from_pretrained(MODEL_DIR)

    # 预计算真值窗口
    truths = []
    for cid in ids:
        wins = ov_windows(cid)
        with open(f"{ANNOT}/f8_training/manifest.jsonl") as f:
            mlook = {json.loads(l)["id"]: json.loads(l) for l in f}
        m = mlook[cid]
        truths.append([(w[0] - m["t0"], w[1] - m["t0"]) for w in wins])

    res = {}
    for name, lora in models:
        model = load_model(lora)
        tp = fp = gt = 0
        for i, (wav, _) in enumerate(eval_set):
            times = predict_times(model, processor, wav)
            wins = truths[i]
            matched = [False] * len(wins)
            for t in times:
                hit = False
                for j, (a, b) in enumerate(wins):
                    if a - 0.5 <= t <= b + 0.5 and not matched[j]:
                        matched[j] = True
                        hit = True
                        break
                if hit:
                    tp += 1
                else:
                    fp += 1
            gt += len(wins)
            if (i + 1) % 50 == 0:
                print(f"{name} {i+1}/{len(eval_set)}", flush=True)
        p = tp / max(1, tp + fp)
        r = tp / max(1, gt)
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        res[name] = {"precision": round(p, 4), "recall": round(r, 4),
                     "f1": round(f1, 4), "n_pred_events": tp + fp,
                     "n_gt_windows": gt}
        print(name, res[name], flush=True)
        del model
        torch.cuda.empty_cache()

    json.dump(res, open(f"{ANNOT}/f8c_eval_results.json", "w"), indent=2)
    print("saved f8c_eval_results.json")


if __name__ == "__main__":
    main()
