"""F7 v2: 零成本 listen/BC 决策层 (基座 + 曲线提示).

v1 的问题: 模型回复率 1.0 (半双工天性, 每段必答).
v2: 两阶段 —
  决策层: 基座模型 + 概率曲线提示 -> 每 200ms 的 p_bc 曲线
     peak >= 0.5  -> backchannel 决策 (生成 BC 类短语音回复)
     peak <  0.5  -> listen 决策 (沉默)
  回复层: 仅 backchannel 决策时走 S2S 短回复生成
指标: listen 率 (vs v1 的 0%)、BC 决策 vs realized 真重叠窗口的 P/R/F1、
回复起始延迟. 零训练成本路线验证.

用法 (funaudiochat, gpu03):
  cd /share/workspace3/zhuangruicen/proj-thumpy/third_party/Fun-Audio-Chat
  python -u .../scripts/ari_f7_v2.py
"""
import glob
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

import librosa  # noqa: E402
import torch  # noqa: E402

# ruamel 混装修复 (同 f7_full)
import ruamel.yaml.loader  # noqa: E402
if not hasattr(ruamel.yaml.loader.Loader, "max_depth"):
    ruamel.yaml.loader.Loader.max_depth = 0

N_BLOCKS = 50
BLOCK_S = 0.2
AUDIO_TEMPLATE = "<|audio_bos|><|AUDIO|><|audio_eos|>"
CURVE_PROMPT = ("Output the backchannel probability for every 200-millisecond "
                "block of this segment, as 50 numbers from 0.0 to 1.0.")
from utils.constant import SPOKEN_S2M_PROMPT  # noqa: E402
BC_PROMPT = (SPOKEN_S2M_PROMPT + " Respond with a brief spoken acknowledgment "
             "like 'yeah' or 'mm-hmm'.")
N_CHUNKS = 40


def load_model():
    from transformers import AutoConfig, AutoModelForSeq2SeqLM
    config = AutoConfig.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_DIR, config=config, torch_dtype=torch.bfloat16, device_map="cuda")
    model.sp_gen_kwargs.update({"text_greedy": True})
    model.eval()
    return model


def predict_curve(model, processor, wav):
    audio = [librosa.load(wav, sr=16000)[0]]
    conversation = [{"role": "system", "content": ""},
                    {"role": "user", "content": AUDIO_TEMPLATE + CURVE_PROMPT}]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audio, return_tensors="pt",
                       return_token_type_ids=False).to(model.device)
    with torch.no_grad():
        generate_ids, _ = model.generate(**inputs, max_new_tokens=256)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    out = processor.decode(generate_ids[0], skip_special_tokens=True)
    vals = [min(1.0, max(0.0, float(x))) for x in re.findall(r"\d+\.?\d*", out)]
    if len(vals) < N_BLOCKS:
        vals += [0.0] * (N_BLOCKS - len(vals))
    return np.array(vals[: N_BLOCKS])


def generate_bc_speech(model, processor, wav, detok):
    audio = [librosa.load(wav, sr=16000)[0]]
    conversation = [{"role": "system", "content": BC_PROMPT},
                    {"role": "user", "content": AUDIO_TEMPLATE}]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audio, return_tensors="pt",
                       return_token_type_ids=False).to(model.device)
    with torch.no_grad():
        generate_ids, audio_ids = model.generate(
            **inputs, max_new_tokens=256, do_sample=False)
    if audio_ids is None or len(audio_ids) == 0 or audio_ids[0].numel() == 0:
        return None
    from utils.cosyvoice_detokenizer import token2wav
    token_for_cv = list(filter(lambda x: 0 <= x < 6561, audio_ids[0].tolist()))
    if not token_for_cv:
        return None
    speech = token2wav(detok, token_for_cv, embedding=None,
                       token_hop_len=25 * 30, pre_lookahead_len=3)
    return np.asarray(speech.cpu()).astype(np.float32).reshape(-1)


def vad_onset(y, sr=16000, thr=0.01):
    win = int(0.025 * sr)
    hop = int(0.01 * sr)
    for i in range(0, len(y) - win + 1, hop):
        if np.sqrt(np.mean(y[i:i + win] ** 2)) > thr:
            return i * hop / sr
    return None


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
    eval_set = extend_eval_set(300)[: N_CHUNKS]
    ids = [os.path.basename(w)[:-4] for (w, _) in eval_set]
    masks = [chunk_mask(cid) for cid in ids]
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    from utils.cosyvoice_detokenizer import get_audio_detokenizer
    detok = get_audio_detokenizer()
    model = load_model()

    n_listen = n_bc = 0
    tps, fps, n_gt = 0, 0, 0
    onsets = []
    for i, (wav, _) in enumerate(eval_set):
        curve = predict_curve(model, processor, wav)
        peak = float(curve.max())
        if peak < 0.5:
            n_listen += 1
        else:
            n_bc += 1
            # BC 决策位置 = 峰值块; 与真重叠窗口 ±0.5s 判定
            pk = int(np.argmax(curve))
            t_peak = pk * BLOCK_S + BLOCK_S / 2
            mask = masks[i]
            hit = mask[max(0, pk - 2): min(N_BLOCKS, pk + 3)].any()
            tps += int(hit)
            fps += int(not hit)
            # 生成 BC 短回复并测起始
            y = generate_bc_speech(model, processor, wav, detok)
            if y is not None:
                on = vad_onset(y)
                if on is not None:
                    onsets.append(on)
        n_gt += int(masks[i].any())
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{N_CHUNKS} listen={n_listen} bc={n_bc}", flush=True)

    p = tps / max(1, tps + fps)
    r = tps / max(1, n_gt)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    res = {
        "listen_rate": round(n_listen / N_CHUNKS, 4),
        "bc_rate": round(n_bc / N_CHUNKS, 4),
        "bc_precision": round(p, 4),
        "bc_recall": round(r, 4),
        "bc_f1": round(f1, 4),
        "mean_response_onset_s": round(float(np.mean(onsets)), 3) if onsets else None,
    }
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(f"{ANNOT}/f7_v2_results.json", "w"), indent=2)
    print("saved f7_v2_results.json")


if __name__ == "__main__":
    main()
