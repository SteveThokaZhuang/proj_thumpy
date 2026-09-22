"""F8 评估: 基座 vs LoRA 后训练模型在留出集上生成话轮状态序列.

指标: 逐词 accuracy + 逐类 recall + 序列级 EM.
留出集: f8_training 中非训练集的 chunks (200 样本, seed 1234 抽样).

用法 (gpu02, funaudiochat 环境):
  python scripts/ari_f8_evaluate.py [--n-eval 200]
"""
import argparse
import json
import os
import random
import sys

import librosa
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/share/workspace3/zhuangruicen/proj-thumpy/third_party/Fun-Audio-Chat")
from funaudiochat.register import register_funaudiochat  # noqa: E402
register_funaudiochat()  # 显式调用注册 (register.py 只在被调用时注册)
from transformers import AutoProcessor  # noqa: E402

ANNOT = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator"
MODEL_DIR = "/share/workspace3/shared_models/Fun-Audio-Chat-8B"
SAVES = f"{ANNOT}/f8_sft/saves"
INSTRUCTION = ("Predict the listener's turn-taking states (listen / speak / "
               "backchannel) over this segment, one word per second.")
AUDIO_TEMPLATE = "<|audio_bos|><|AUDIO|><|audio_eos|>"
STATE_WORDS = ["listen", "speak", "backchannel"]


def build_eval_set(n):
    # 冻结评估集: 首次抽样后持久化, 保证跨模型/跨训练轮公平对比
    frozen = f"{ANNOT}/f8_eval_set.json"
    if os.path.exists(frozen):
        return json.load(open(frozen))
    train_ids = set()
    for line in open(f"{ANNOT}/f8_sft/train.jsonl"):
        r = json.loads(line)
        train_ids.add(os.path.basename(json.loads(r["audio"])["path"])[:-4])
    with open(f"{ANNOT}/f8_training/manifest.jsonl") as f:
        manifest = [json.loads(l) for l in f]
    random.Random(1234).shuffle(manifest)
    out = []
    for m in manifest:
        if m["id"] in train_ids or m["id"] in {x[0] for x in out}:
            continue
        d = np.load(f"{ANNOT}/f8_training/{m['id']}.npz")
        n_sec = int(len(d["audio"]) / 16000)
        words = []
        for i in range(n_sec):
            blk = d["states"][i * 12: (i + 1) * 12 + 1]
            m2 = blk.mean(axis=0)
            words.append("backchannel" if m2[2] >= 0.2 else
                         "speak" if m2[1] >= 0.5 else "listen")
        # 留出集样本的 wav 按需落盘
        import soundfile as sf
        wav = f"{ANNOT}/f8_sft/eval_wavs/{m['id']}.wav"
        os.makedirs(os.path.dirname(wav), exist_ok=True)
        if not os.path.exists(wav):
            sf.write(wav, d["audio"], 16000)
        out.append([wav, words])
        if len(out) >= n:
            break
    json.dump(out, open(frozen, "w"))
    return out


def load_model(lora_dir=None):
    from transformers import AutoConfig, AutoModelForSeq2SeqLM
    config = AutoConfig.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_DIR, config=config, torch_dtype=torch.bfloat16, device_map="cuda")
    if lora_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_dir)
    model.sp_gen_kwargs.update({"text_greedy": True})
    model.eval()
    return model


def predict(model, processor, wav_path, n_words):
    audio = [librosa.load(wav_path, sr=16000)[0]]
    conversation = [
        {"role": "system", "content": ""},
        {"role": "user", "content": AUDIO_TEMPLATE + INSTRUCTION},
    ]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audio, return_tensors="pt",
                       return_token_type_ids=False).to(model.device)
    with torch.no_grad():
        # text_greedy 已通过 model.sp_gen_kwargs 设置 (官方用法)
        generate_ids, _ = model.generate(
            **inputs, max_new_tokens=max(32, n_words * 8))
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    out_text = processor.decode(generate_ids[0], skip_special_tokens=True)
    pred = [w for w in out_text.split() if w in STATE_WORDS][: n_words]
    pred += ["listen"] * (n_words - len(pred))
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--lora-dir", type=str, default="")
    args = ap.parse_args()

    eval_set = build_eval_set(args.n_eval)
    print(f"eval samples: {len(eval_set)}", flush=True)

    results = {}
    for name, lora in [("base", None), ("lora", args.lora_dir or None)]:
        if name == "lora" and lora is None:
            continue
        model = load_model(lora)
        processor = AutoProcessor.from_pretrained(MODEL_DIR)
        acc, em = [], 0
        per_class = {w: [0, 0] for w in STATE_WORDS}
        for i, (wav, words) in enumerate(eval_set):
            pred = predict(model, processor, wav, len(words))
            acc.append(np.mean([p == r for p, r in zip(pred, words)]))
            em += int(pred == words)
            for p, r in zip(pred, words):
                per_class[r][0] += int(p == r)
                per_class[r][1] += 1
            if (i + 1) % 20 == 0:
                print(f"{name} {i+1}/{len(eval_set)}", flush=True)
        results[name] = {
            "word_acc": round(float(np.mean(acc)), 4),
            "em": round(float(em / len(eval_set)), 4),
            "per_class_recall": {k: round(v[0] / max(1, v[1]), 4)
                                 for k, v in per_class.items()},
        }
        print(name, results[name], flush=True)
        del model
        torch.cuda.empty_cache()

    json.dump(results, open(f"{ANNOT}/f8_eval_results.json", "w"), indent=2)
    print("saved f8_eval_results.json")


if __name__ == "__main__":
    main()
