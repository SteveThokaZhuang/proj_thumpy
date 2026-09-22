"""G1 消融评估: 事件级 F1 (预测 "X.Xs" 落在真重叠 AWS BC 窗口 ±0.5s 内).

与 ari_f8c_evaluate.py 同口径, 但音频/窗口长度按观测空间臂参数化。
评估集为**同一批 300 个 chunk**, 各臂只是音频不同 -> 臂间可直接比较。

用法 (funaudiochat 环境, gpu04):
  python scripts/ari_g1_eval.py --npz-dir .../g1_mixnorm10 \
      --lora .../g1_mixnorm10_sft/saves --tag mixnorm10
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_f8_evaluate import ANNOT, load_model, MODEL_DIR  # noqa: E402
from ari_g1_common import (bc_windows, eval_chunk_ids, load_ids_file,  # noqa: E402
                           load_manifest, make_eval_wavs)
from funaudiochat.register import register_funaudiochat  # noqa: E402
register_funaudiochat()

import librosa  # noqa: E402
import torch  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

INSTRUCTION = ("When does the listener produce backchannels in this segment? "
               "List the times in seconds with confidence, or answer "
               "'no backchannel'.")
AUDIO_TEMPLATE = "<|audio_bos|><|AUDIO|><|audio_eos|>"


def predict_times(model, processor, wav, chunk_s, instruction=INSTRUCTION):
    audio = [librosa.load(wav, sr=16000)[0]]
    conversation = [{"role": "system", "content": ""},
                    {"role": "user", "content": AUDIO_TEMPLATE + instruction}]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audio, return_tensors="pt",
                       return_token_type_ids=False).to(model.device)
    with torch.no_grad():
        generate_ids, _ = model.generate(**inputs, max_new_tokens=128)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    out = processor.decode(generate_ids[0], skip_special_tokens=True)
    return [float(x) for x in re.findall(r"(\d+\.?\d*)s", out)
            if 0 <= float(x) <= chunk_s + 0.5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="",
                    help="该臂的 npz 目录 (从中导出评估 wav)")
    ap.add_argument("--eval-wav-dir", default="",
                    help="直接用已有的评估 wav (own 臂可用 f8_sft/eval_wavs)")
    ap.add_argument("--manifest-from", default="",
                    help="读 manifest 的目录; 默认同 --npz-dir。各臂 id/session/ch/t0 "
                         "相同, 所以 own 臂可以借 mixnorm 臂的 manifest 取窗口元数据")
    ap.add_argument("--lora", default="")
    ap.add_argument("--tag", default="arm")
    ap.add_argument("--chunk-s", type=float, default=10.0)
    ap.add_argument("--n-eval", type=int, default=0,
                    help="截断到前 N 块; 0 = 不截断 (默认)。注意 --ids-file 给定时"
                         "这个默认值**必须**是 0 —— 否则扩评估集会静默退化成"
                         "「只评前 300 块」, 而 1600 块的运行要好几个小时, "
                         "等发现时已经白跑")
    ap.add_argument("--ids-file", default="",
                    help="用自定义 chunk id 清单替代冻结的 300 块评估集 (#53 扩集)")
    ap.add_argument("--require-inside", action="store_true",
                    help="真值只取**完整落在块内**的窗口 (#53): 横跨 chunk 边界的"
                         "窗口声学证据在相邻块里, 本质不可答, 会拉偏绝对 F1")
    ap.add_argument("--also-base", action="store_true")
    ap.add_argument("--instruction", default=INSTRUCTION,
                    help="评估用指令; 需与训练一致, 否则基线被 prompt 不匹配污染")
    args = ap.parse_args()

    # 冻结评估集默认 300 块 (既有脚本的行为, 不能变); --n-eval 0 = 不截断
    ids = (load_ids_file(args.ids_file, args.n_eval) if args.ids_file
           else eval_chunk_ids(args.n_eval or 300))
    if args.eval_wav_dir:
        eval_set = [(f"{args.eval_wav_dir}/{cid}.wav", cid) for cid in ids]
        missing = [w for w, _ in eval_set if not os.path.exists(w)]
        if missing:
            raise SystemExit(f"缺 {len(missing)} 个评估 wav, 例如 {missing[:3]}")
    else:
        eval_dir = (f"{args.npz_dir}/../g1_eval_wavs_"
                    f"{os.path.basename(args.npz_dir.rstrip('/'))}")
        eval_set = make_eval_wavs(args.npz_dir, eval_dir, ids)
    mlook = load_manifest(args.manifest_from or args.npz_dir)
    truths, dropped = [], []
    for cid in ids:
        k, d = bc_windows(cid, mlook, chunk_s=args.chunk_s,
                          require_inside=args.require_inside,
                          return_dropped=True)
        truths.append(k)
        dropped.append(d)
    n_drop = sum(len(d) for d in dropped)
    if n_drop:
        print(f"边界过滤: 丢弃 {n_drop} 个横跨 chunk 边界的真值窗口 "
              f"(涉及 {sum(1 for d in dropped if d)} 个块)", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    print(f"eval chunks: {len(ids)} (audio from {args.npz_dir}, "
          f"require_inside={args.require_inside})", flush=True)

    results = {}
    arms = [("base", None)] if args.also_base else []
    arms.append((args.tag, args.lora))
    for name, lora in arms:
        model = load_model(lora)
        tp = fp = gt = 0
        # 逐 chunk 记录: 只有 26 个真值事件, 聚合数字不足以判断两臂差异是否
        # 超出噪声。两臂评的是同一批 chunk, 因此可以做配对 bootstrap。
        per_chunk = {}
        for i, (wav, cid) in enumerate(eval_set):
            times = predict_times(model, processor, wav, args.chunk_s,
                                  args.instruction)
            wins = truths[i]
            matched = [False] * len(wins)
            ctp = cfp = 0
            for t in times:
                hit = False
                for j, (a, b) in enumerate(wins):
                    if a - 0.5 <= t <= b + 0.5 and not matched[j]:
                        matched[j] = True
                        hit = True
                        break
                tp += 1 if hit else 0
                fp += 0 if hit else 1
                ctp += 1 if hit else 0
                cfp += 0 if hit else 1
            gt += len(wins)
            # pred_times/wins 一并落盘 (#53): 没有它们, 任何真值口径的改动
            # (边界过滤、容差、realized 口径) 都得重跑一遍 GPU。存下来之后
            # 重新打分就是纯后处理, 秒级完成。
            per_chunk[cid] = {"tp": ctp, "fp": cfp, "n_gt": len(wins),
                              "n_pred": len(times),
                              "pred_times": [round(t, 2) for t in times],
                              "wins": [[round(a, 2), round(b, 2)]
                                       for a, b in wins]}
            if (i + 1) % 50 == 0:
                print(f"  {name} {i+1}/{len(eval_set)} tp={tp} fp={fp} gt={gt}",
                      flush=True)
        p = tp / max(1, tp + fp)
        r = tp / max(1, gt)
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        results[name] = {"precision": round(p, 4), "recall": round(r, 4),
                         "f1": round(f1, 4), "n_pred": tp + fp, "n_gt": gt,
                         "n_chunks": len(ids),
                         "require_inside": bool(args.require_inside),
                         "ids_file": args.ids_file,
                         "n_dropped_straddling": n_drop,
                         "per_chunk": per_chunk}
        print(f"{name}: {results[name]}", flush=True)
        del model
        torch.cuda.empty_cache()

    out = f"{ANNOT}/g1_eval_{args.tag}.json"
    json.dump(results, open(out, "w"), indent=2)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
