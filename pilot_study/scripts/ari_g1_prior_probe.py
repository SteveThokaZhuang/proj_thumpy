"""#51 续: "8.2–8.8s" 是模型的默认落点吗?

never 组的 4 个块预测落在 8.2–8.8s (0f918385 8.8 / 01a4c01c 8.8 / 0777bb85 8.2 /
0736fa95 8.2), 而 always 组的预测是 0.4/1.6/3.8/4.2/6.8 —— 跟着真值走。
怀疑模型在没有证据时会**退化成某个刻板落点**。

检验: 在**没有任何 BC 事件**的块上跑模型。如果那里也集中在 8.x, 就是位置先验,
而不是"这一块真有个事件我找错了地方"。

用法: python ari_g1_prior_probe.py [--n 40]
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_f8_evaluate import MODEL_DIR, load_model  # noqa: E402
from ari_g1_common import ANNOT, eval_chunk_ids, load_manifest  # noqa: E402
from ari_g1_eval import INSTRUCTION, predict_times  # noqa: E402
from funaudiochat.register import register_funaudiochat  # noqa: E402
register_funaudiochat()

import torch  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

OWN = f"{ANNOT}/g1_own_eval/own"
WAV = f"{ANNOT}/g1_own_eval/g1_eval_wavs_own"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="负块数量")
    ap.add_argument("--lora", default=f"{ANNOT}/g1_own10_sft/saves")
    args = ap.parse_args()

    mlook = load_manifest(OWN)
    per = next(iter(json.load(open(f"{ANNOT}/g1_eval_own10.json")).values()))["per_chunk"]

    # 无 GT 的块, 按 id 排序取前 n (确定性, 不用随机)
    negs = [c for c in eval_chunk_ids(300)
            if per.get(c, {}).get("n_gt", 0) == 0]
    negs = sorted(negs)[:args.n]
    print(f"负块 n={len(negs)} (从 {sum(1 for c in eval_chunk_ids(300) if per[c]['n_gt']==0)} 个里取)")

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = load_model(args.lora)

    times, no_ans = [], 0
    for cid in negs:
        wav = f"{WAV}/{cid}.wav"
        if not os.path.exists(wav):
            continue
        ts = predict_times(model, processor, wav, 10.0, INSTRUCTION)
        if not ts:
            no_ans += 1
        times.extend(ts)

    print(f"\n负块预测 {len(times)} 个时刻, 其中 {no_ans}/{len(negs)} 块答 'no backchannel'")
    if times:
        t = np.array(times)
        print(f"  均值 {t.mean():.2f}s  中位数 {np.median(t):.2f}s  "
              f"范围 {t.min():.2f}–{t.max():.2f}s")
        print("\n  1s 分桶直方图:")
        h = collections.Counter(int(x) for x in t)
        for b in range(10):
            c = h.get(b, 0)
            print(f"    {b}-{b+1}s  {c:4d}  {'█' * min(50, c)}")
        late = sum(1 for x in t if 8.0 <= x < 9.0)
        print(f"\n  落在 8.0–9.0s 的比例: {late}/{len(t)} = {late/len(t)*100:.0f}%"
              f"  (均匀分布期望 10%)")


if __name__ == "__main__":
    main()
