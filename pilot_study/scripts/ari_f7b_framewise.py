"""F7b: 帧级行为评估 — 200ms 词粒度下四方模型 vs 标注器参照.

F7 v0 发现 1s 词粒度太粗 (BC 窗口 ~0.6s, 命中窗口小). F7b 把输出粒度提到
200ms (50 词/块): 模型生成每 200ms 一个状态词; 帧级口径 vs realized 真重叠:
  - bc 词指示器的帧级 AUC (vs 真重叠 BC 窗口隶属)
  - 事件级 precision/recall/F1 (连续 bc 词段 vs 真重叠 BC 窗口)
参照: X2-Turn 融合软分数在 200ms 块上的同样指标 (AUROC + 硬块 F1).
评估集: 冻结 100 + 追加 200 (追加式, 历史对比不受影响).

用法 (funaudiochat 环境, gpu02/gpu04):
  python scripts/ari_f7b_framewise.py
"""
import glob
import json
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_f8_evaluate import (ANNOT, load_model, predict,  # noqa: E402
                             MODEL_DIR, AUDIO_TEMPLATE, INSTRUCTION)
from ari_analyze import DEFAULT_OUT  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from funaudiochat.register import register_funaudiochat  # noqa: E402
register_funaudiochat()

import librosa  # noqa: E402
import torch  # noqa: E402
import soundfile as sf  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

BLOCK_S = 0.2
N_BLOCKS = 50
MODELS = [
    ("base", None),
    ("real_3.6k", f"{ANNOT}/f8_sft/saves"),
    ("real_12k", f"{ANNOT}/f8_sft_expanded/saves"),
    ("synthetic", f"{ANNOT}/f9_sft_synthetic/saves"),
]
PROMPT = (AUDIO_TEMPLATE + "Predict the listener's turn-taking states "
          "(listen / speak / backchannel) over this segment, "
          "one word per 200 milliseconds.")


def extend_eval_set(total=300):
    """冻结集 100 + 追加 200 (f8_training 中非训练集块)."""
    frozen = json.load(open(f"{ANNOT}/f8_eval_set.json"))
    if len(frozen) >= total:
        return frozen[: total]
    train_ids = set()
    for line in open(f"{ANNOT}/f8_sft/train.jsonl"):
        r = json.loads(line)
        train_ids.add(os.path.basename(json.loads(r["audio"])["path"])[:-4])
    with open(f"{ANNOT}/f8_training/manifest.jsonl") as f:
        manifest = [json.loads(l) for l in f]
    random.Random(5678).shuffle(manifest)
    have = {os.path.basename(w)[:-4] for (w, _) in frozen}
    for m in manifest:
        if m["id"] in train_ids or m["id"] in have:
            continue
        d = np.load(f"{ANNOT}/f8_training/{m['id']}.npz")
        wav = f"{ANNOT}/f8_sft/eval_wavs/{m['id']}.wav"
        os.makedirs(os.path.dirname(wav), exist_ok=True)
        if not os.path.exists(wav):
            sf.write(wav, d["audio"], 16000)
        frozen.append([wav, None])
        have.add(m["id"])
        if len(frozen) >= total:
            break
    json.dump(frozen, open(f"{ANNOT}/f8_eval_set.json", "w"))
    return frozen


def chunk_ov_mask(chunk_id):
    """chunk -> 每 200ms 块的 bool (是否与真重叠 BC 窗口相交) + 窗口数."""
    with open(f"{ANNOT}/f8_training/manifest.jsonl") as f:
        mlook = {json.loads(l)["id"]: json.loads(l) for l in f}
    m = mlook[chunk_id]
    ev = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_events_w*.csv"))])
    e1 = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_e1_w*.csv"))])
    e1 = e1.merge(ev[["session", "event_id", "cls", "ch_event", "start", "end"]],
                  on="event_id", suffixes=("", "_ev"))
    sub = e1[(e1["session"] == m["session"]) & (e1["cls"] == "BC")
             & (e1["ch_event"] == int(m["ch"]))]
    mask = np.zeros(N_BLOCKS, bool)
    for _, r in sub.iterrows():
        if r["realized"] == "None":
            continue
        for k in range(N_BLOCKS):
            a, b = m["t0"] + k * BLOCK_S, m["t0"] + (k + 1) * BLOCK_S
            if a < r["end"] and b > r["start"]:
                mask[k] = True
    return mask


def predict_blocks(model, processor, wav, n):
    audio = [librosa.load(wav, sr=16000)[0]]
    conversation = [{"role": "system", "content": ""},
                    {"role": "user", "content": PROMPT}]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audio, return_tensors="pt",
                       return_token_type_ids=False).to(model.device)
    with torch.no_grad():
        generate_ids, _ = model.generate(
            **inputs, max_new_tokens=max(64, n * 8))
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    out_text = processor.decode(generate_ids[0], skip_special_tokens=True)
    words = [w for w in out_text.split()
             if w in ("listen", "speak", "backchannel")][: n]
    words += ["listen"] * (n - len(words))
    return [w == "backchannel" for w in words]


def annotator_ref(eval_ids):
    """X2-Turn 融合分数在 200ms 块上的 AUROC + 硬块 F1 (τ=0.2)."""
    mani = json.load(open(f"{ANNOT}/e5_regions_manifest.json"))
    x2 = {}
    for fp in glob.glob(f"{ANNOT}/e4_raw/*.json"):
        s = os.path.basename(fp)[:-5]
        d = json.load(open(fp))
        x2[s] = {}
        for ch in ("0", "1"):
            fr = d["channels"][ch]["frames"]
            x2[s][ch] = (np.array([f[0] for f in fr]),
                         np.array([f[1] for f in fr]),
                         np.array([f[3] for f in fr]))
    with open(f"{ANNOT}/f8_training/manifest.jsonl") as f:
        mlook = {json.loads(l)["id"]: json.loads(l) for l in f}
    scores, labels = [], []
    for cid in eval_ids:
        m = mlook[cid]
        if m["session"] not in x2 or str(m["ch"]) not in x2[m["session"]]:
            continue
        xt0, xt1, xp = x2[m["session"]][str(m["ch"])]
        mask = chunk_ov_mask(cid)
        for k in range(N_BLOCKS):
            a, b = m["t0"] + k * BLOCK_S, m["t0"] + (k + 1) * BLOCK_S
            fm = (xt0 < b) & (xt1 > a)
            scores.append(float(xp[fm].max()) if fm.any() else 0.0)
            labels.append(bool(mask[k]))
    labels = np.array(labels)
    scores = np.array(scores)
    auc = roc_auc_score(labels, scores)
    # 硬块 F1 (τ=0.2 连续块)
    pred = scores >= 0.2
    tp = int((pred & labels).sum())
    p = tp / max(1, int(pred.sum()))
    r = tp / max(1, int(labels.sum()))
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"auc": round(float(auc), 4), "f1": round(float(f1), 4),
            "precision": round(float(p), 4), "recall": round(float(r), 4)}


def main():
    eval_set = extend_eval_set(300)
    ids = [os.path.basename(w)[:-4] for (w, _) in eval_set]
    processor = AutoProcessor.from_pretrained(MODEL_DIR)

    res = {"annotator_ref": annotator_ref(ids)}
    print("annotator_ref", res["annotator_ref"], flush=True)
    for name, lora in MODELS:
        model = load_model(lora)
        aucs, fps, tps, n_gt = [], 0, 0, 0
        for i, (wav, _) in enumerate(eval_set):
            pred_bc = np.array(predict_blocks(model, processor, wav, N_BLOCKS))
            mask = chunk_ov_mask(ids[i])
            if mask.any() and (~mask).any():
                aucs.append(roc_auc_score(mask, pred_bc.astype(float)))
            fp = int((pred_bc & ~mask).sum())
            tp = int((pred_bc & mask).sum())
            fps += fp
            tps += tp
            n_gt += int(mask.sum())
            if (i + 1) % 50 == 0:
                print(f"{name} {i+1}/{len(eval_set)}", flush=True)
        p = tps / max(1, tps + fps)
        r = tps / max(1, n_gt)
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        res[name] = {"auc_mean": round(float(np.mean(aucs)), 4) if aucs else None,
                     "precision": round(float(p), 4),
                     "recall": round(float(r), 4),
                     "f1": round(float(f1), 4)}
        print(name, res[name], flush=True)
        del model
        torch.cuda.empty_cache()

    json.dump(res, open(f"{ANNOT}/f7b_results.json", "w"), indent=2)
    print("saved f7b_results.json")


if __name__ == "__main__":
    main()
