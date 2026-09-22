"""F7 完整版: S2S 语音回复重放 + VAD 行为指标.

对冻结评估集的每个 chunk, 模型以语音对话模式生成回复音频 (S2S,
CosyVoice3 语音 token 解码), 然后:
  - RMS-VAD 得到回复语音段
  - 行为指标: 回复率、回复起始延迟、BC 类短回复 (<1.5s) 与 realized
    真重叠 BC 窗口 ±0.5s 的命中
  - 三方对比: 基座 / f8c 事件式 / real_12k 词序列

注意: 生成慢 (含语音解码), 每模型取前 60 个 chunk.

用法 (funaudiochat 环境, gpu03):
  python scripts/ari_f7_full.py
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/share/workspace3/zhuangruicen/proj-thumpy/third_party/Fun-Audio-Chat")
from ari_f8_evaluate import ANNOT, MODEL_DIR  # noqa: E402
from ari_f7b_framewise import extend_eval_set  # noqa: E402
from ari_analyze import DEFAULT_OUT  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from funaudiochat.register import register_funaudiochat  # noqa: E402
register_funaudiochat()

import glob  # noqa: E402
import librosa  # noqa: E402
import torch  # noqa: E402

# ruamel 混装修复: composer 引用 Loader.max_depth 但本环境 Loader 无此属性
import ruamel.yaml.loader  # noqa: E402
if not hasattr(ruamel.yaml.loader.Loader, "max_depth"):
    ruamel.yaml.loader.Loader.max_depth = 0

AUDIO_TEMPLATE = "<|audio_bos|><|AUDIO|><|audio_eos|>"
# 官方 S2S 配方: 官方 system prompt + text_greedy=True (同时返回文本与音频)
from utils.constant import SPOKEN_S2M_PROMPT  # noqa: E402
SYSTEM_PROMPT = SPOKEN_S2M_PROMPT
MODELS = [
    ("base", None),
    ("f8c_events", f"{ANNOT}/f8c_sft/saves"),
    ("real_12k_words", f"{ANNOT}/f8_sft_expanded/saves"),
]
N_CHUNKS = 60


def load_model(lora_dir=None):
    from transformers import AutoConfig, AutoModelForSeq2SeqLM
    config = AutoConfig.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_DIR, config=config, torch_dtype=torch.bfloat16, device_map="cuda")
    if lora_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_dir)
    # 官方 S2S 配方: text_greedy=True 同时返回文本与音频
    model.sp_gen_kwargs.update({"text_greedy": True})
    model.eval()
    return model


def generate_speech(model, processor, wav, detok):
    audio = [librosa.load(wav, sr=16000)[0]]
    conversation = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": AUDIO_TEMPLATE}]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audio, return_tensors="pt",
                       return_token_type_ids=False).to(model.device)
    with torch.no_grad():
        generate_ids, audio_ids = model.generate(
            **inputs, max_new_tokens=512, do_sample=False)
    if audio_ids is None or len(audio_ids) == 0 or audio_ids[0].numel() == 0:
        return None
    # 官方配方: CosyVoice 声码器把语音 token 转 wav
    from utils.cosyvoice_detokenizer import token2wav
    token_for_cv = list(filter(lambda x: 0 <= x < 6561, audio_ids[0].tolist()))
    if not token_for_cv:
        return None
    speech = token2wav(detok, token_for_cv, embedding=None,
                       token_hop_len=25 * 30, pre_lookahead_len=3)
    return np.asarray(speech.cpu()).astype(np.float32).reshape(-1)


def vad_segments(y, sr=16000, thr=0.01, min_gap=0.3):
    win = int(0.025 * sr)
    hop = int(0.01 * sr)
    frames = []
    for i in range(0, len(y) - win + 1, hop):
        frames.append(np.sqrt(np.mean(y[i:i + win] ** 2)) > thr)
    segs, s = [], None
    for i, act in enumerate(frames):
        if act and s is None:
            s = i * hop / sr
        elif not act and s is not None:
            if i * hop / sr - s >= 0.2:
                segs.append((s, i * hop / sr))
            s = None
    if s is not None and len(frames) * hop / sr - s >= 0.2:
        segs.append((s, len(frames) * hop / sr))
    return segs


def ov_windows(cid):
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
    return [(r["start"] - m["t0"], r["end"] - m["t0"]) for _, r in sub.iterrows()
            if r["end"] > m["t0"] and r["start"] < m["t0"] + 10]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args()
    models_run = [(n, l) for n, l in MODELS
                  if not args.only or n == args.only]
    eval_set = extend_eval_set(300)[: N_CHUNKS]
    ids = [os.path.basename(w)[:-4] for (w, _) in eval_set]
    truths = [ov_windows(cid) for cid in ids]
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    from utils.cosyvoice_detokenizer import get_audio_detokenizer
    detok = get_audio_detokenizer()

    res = {}
    for name, lora in models_run:
        model = load_model(lora)
        n_resp = n_bc = 0
        onsets = []
        tp = fp = gt = 0
        for i, (wav, _) in enumerate(eval_set):
            y = generate_speech(model, processor, wav, detok)
            if y is None or len(y) == 0:
                continue
            segs = vad_segments(y)
            if not segs:
                continue
            n_resp += 1
            onsets.append(segs[0][0])
            wins = truths[i]
            gt += len(wins)
            for (a, b) in segs:
                if b - a < 1.5:  # BC 类短回复
                    n_bc += 1
                    hit = any(w0 - 0.5 <= a <= w1 + 0.5 for w0, w1 in wins)
                    tp += int(hit)
                    fp += int(not hit)
            if (i + 1) % 20 == 0:
                print(f"{name} {i+1}/{N_CHUNKS}", flush=True)
        p = tp / max(1, tp + fp)
        r = tp / max(1, gt)
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        res[name] = {
            "response_rate": round(n_resp / N_CHUNKS, 4),
            "mean_onset_s": round(float(np.mean(onsets)), 3) if onsets else None,
            "n_short_bc_like": n_bc,
            "bc_precision": round(p, 4), "bc_recall": round(r, 4),
            "bc_f1": round(f1, 4),
        }
        print(name, res[name], flush=True)
        del model
        torch.cuda.empty_cache()

    json.dump(res, open(f"{ANNOT}/f7_full_results.json", "w"), indent=2)
    print("saved f7_full_results.json")


if __name__ == "__main__":
    main()
