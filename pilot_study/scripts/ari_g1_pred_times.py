"""#51 续: 把 never-hit 块的**预测时刻**打出来.

只用计数看不出问题在哪: 0d4a795e 那类块 7/7 个种子都预测了 1 个事件, 却 tp=0
—— 说明模型**看到了东西, 只是没落在 GT 窗口的 ±0.5s 内**。要知道它落在哪。

跑 14 个块 (9 never + 5 always 对照) × 2 个种子, 秒级。

用法: python ari_g1_pred_times.py
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_f8_evaluate import load_model  # noqa: E402
from ari_g1_common import ANNOT, bc_windows, eval_chunk_ids, load_manifest  # noqa: E402
from ari_g1_eval import INSTRUCTION, predict_times  # noqa: E402
from funaudiochat.register import register_funaudiochat  # noqa: E402
register_funaudiochat()

import torch  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from ari_f8_evaluate import MODEL_DIR  # noqa: E402

OWN = f"{ANNOT}/g1_own_eval/own"
WAV = f"{ANNOT}/g1_own_eval/g1_eval_wavs_own"

NEVER = [
    "0020a0c5_ch1_t1710", "01a4c01c_ch0_t1705", "01e64778_ch1_t50",
    "02053f7c_ch1_t235", "0480a711_ch0_t400", "0736fa95_ch1_t1105",
    "0777bb85_ch1_t860", "0d4a795e_ch1_t540", "0f918385_ch0_t225",
]
ALWAYS = [
    "002d68da_ch0_t280", "010a1b2a_ch1_t1060", "01e64778_ch0_t1500",
    "0542c0f0_ch1_t780", "05aa5dd6_ch1_t1565",
]

ARMS = [
    ("s42", f"{ANNOT}/g1_own10_sft/saves"),
    ("s7",  f"{ANNOT}/g1_own10_s7_sft/saves"),
]


def match_of(t, wins, tol=0.5):
    """该预测是否命中, 命中第几个窗口."""
    for j, (a, b) in enumerate(wins):
        if a - tol <= t <= b + tol:
            return j
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{os.path.dirname(os.path.abspath(__file__))}"
                                     f"/pred_times_out.txt")
    args = ap.parse_args()

    ids = set(eval_chunk_ids(300))
    mlook = load_manifest(OWN)
    chunks = [c for c in NEVER + ALWAYS if c in ids]

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    for name, lora in ARMS:
        model = load_model(lora)
        emit(f"\n{'='*78}\n  臂 {name}  ({lora})\n{'='*78}")
        for cid in chunks:
            wav = f"{WAV}/{cid}.wav"
            if not os.path.exists(wav):
                emit(f"  缺 wav {wav}")
                continue
            wins = bc_windows(cid, mlook)
            times = predict_times(model, processor, wav, 10.0, INSTRUCTION)
            grp = "NEVER" if cid in NEVER else "ALWAYS"
            emit(f"\n▌ [{grp}] {cid}")
            for a, b in wins:
                emit(f"    GT 窗口 {a:6.2f}-{b:6.2f}  可命中区间 "
                     f"[{a-0.5:6.2f}, {b+0.5:6.2f}]")
            if not wins:
                emit("    (无 GT 窗口)")
            if not times:
                emit("    预测: 无 (答了 'no backchannel')")
            for t in times:
                j = match_of(t, wins)
                tag = f"✅ 命中窗口{j}" if j is not None else "❌ 未命中"
                # 离最近窗口边界的距离, 判断"差多少"
                if wins:
                    d = min(min(abs(t - (a - 0.5)), abs(t - (b + 0.5)))
                            if not (a - 0.5 <= t <= b + 0.5) else 0.0
                            for a, b in wins)
                    tag += f"   离最近窗口边界 {d:.2f}s"
                emit(f"    预测 {t:6.2f}s  {tag}")
        del model
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
