"""F8c v3 评估: 软分数曲线的帧级行为指标.

指标 (与标注器参照同口径):
  1. 帧级 AUC: 模型输出的 200ms 概率曲线 vs realized 真重叠 BC 窗口隶属
     (直接对标标注器软分数 AUC 0.652-0.667)
  2. 事件级: 曲线上峰值段 (>= τ) vs 真重叠窗口 ±0.5s 命中 P/R/F1
对比: 基座 / f8c v1 事件 / v3 软分数。

用法 (funaudiochat, gpu03): python scripts/ari_f8c_v3_evaluate.py
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_f8_evaluate import ANNOT, MODEL_DIR  # noqa: E402
from ari_f7b_framewise import extend_eval_set  # noqa: E402
from ari_analyze import DEFAULT_OUT  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from funaudiochat.register import register_funaudiochat  # noqa: E402
register_funaudiochat()

import librosa  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

N_BLOCKS = 50
BLOCK_S = 0.2
INSTRUCTION = ("Output the backchannel probability for every 200-millisecond "
               "block of this segment, as 50 numbers from 0.0 to 1.0.")
AUDIO_TEMPLATE = "<|audio_bos|><|AUDIO|><|audio_eos|>"
MODELS = [
    ("base", None),
    ("f8c_v1_events", f"{ANNOT}/f8c_sft/saves"),
    ("f8c_v3_soft", f"{ANNOT}/f8c_v3_sft/saves"),
]


def load_model(lora_dir=None):
    from transformers import AutoConfig, AutoModelForSeq2SeqLM
    config = AutoConfig.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_DIR, config=config, torch_dtype=torch.bfloat16, device_map="cuda")
    if lora_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_dir)
    model.eval()
    return model


def predict_curve(model, processor, wav):
    audio = [librosa.load(wav, sr=16000)[0]]
    conversation = [{"role": "system", "content": ""},
                    {"role": "user", "content": AUDIO_TEMPLATE + INSTRUCTION}]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audio, return_tensors="pt",
                       return_token_type_ids=False).to(model.device)
    with torch.no_grad():
        generate_ids, _ = model.generate(**inputs, max_new_tokens=256)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    out = processor.decode(generate_ids[0], skip_special_tokens=True)
    vals = [min(1.0, max(0.0, float(x)))
            for x in re.findall(r"\d+\.?\d*", out)]
    if len(vals) < N_BLOCKS:
        vals += [0.0] * (N_BLOCKS - len(vals))
    return np.array(vals[: N_BLOCKS])


def chunk_mask(cid):
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
    mask = np.zeros(N_BLOCKS, bool)
    for _, r in sub.iterrows():
        for k in range(N_BLOCKS):
            a, b = m["t0"] + k * BLOCK_S, m["t0"] + (k + 1) * BLOCK_S
            if a < r["end"] and b > r["start"]:
                mask[k] = True
    return mask


def main():
    eval_set = extend_eval_set(300)
    ids = [os.path.basename(w)[:-4] for (w, _) in eval_set]
    masks = [chunk_mask(cid) for cid in ids]
    processor = AutoProcessor.from_pretrained(MODEL_DIR)

    res = {}
    for name, lora in MODELS:
        model = load_model(lora)
        aucs, tps, fps, n_gt = [], 0, 0, 0
        for i, (wav, _) in enumerate(eval_set):
            curve = predict_curve(model, processor, wav)
            mask = masks[i]
            if mask.any() and (~mask).any():
                aucs.append(roc_auc_score(mask, curve))
            pred = curve >= 0.2
            tps += int((pred & mask).sum())
            fps += int((pred & ~mask).sum())
            n_gt += int(mask.sum())
            if (i + 1) % 50 == 0:
                print(f"{name} {i+1}/{len(eval_set)}", flush=True)
        p = tps / max(1, tps + fps)
        r = tps / max(1, n_gt)
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        res[name] = {"auc_mean": round(float(np.mean(aucs)), 4) if aucs else None,
                     "precision": round(p, 4), "recall": round(r, 4),
                     "f1": round(f1, 4)}
        print(name, res[name], flush=True)
        del model
        torch.cuda.empty_cache()

    json.dump(res, open(f"{ANNOT}/f8c_v3_eval_results.json", "w"), indent=2)
    print("saved f8c_v3_eval_results.json")


if __name__ == "__main__":
    main()
