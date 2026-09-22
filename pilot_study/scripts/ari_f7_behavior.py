"""F7: 模型话轮行为指标 (BCRR/BC-precision-recall) 端到端评估.

对四个模型 (基座 / 真实3.6k / 真实12k / 合成) 在冻结评估集上:
  1. 生成每 1s 块的话轮状态词 (F8 任务)
  2. BC 行为指标 (vs 真值):
     - BC 窗口 = 该块内与 AWS BC 窗口相交且 realized=真重叠 的 1s 块
     - precision/recall/F1 = 模型预测 bc 块 vs 这些窗口
     - 也报告 vs 全部 AWS BC 窗口 (含停顿内 BC) 的命中
  3. 状态分布 (坍缩检测)
参照: X2-Turn 融合标注器在相同窗口上的覆盖 (F1/F6 口径, 标注器上限).

输出: f7_results.json + docs/pilot_study/figures/f7_behavior.png
用法 (gpu02/gpu04, funaudiochat 环境):
  python scripts/ari_f7_behavior.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_f8_evaluate import (build_eval_set, load_model, predict,  # noqa: E402
                             ANNOT, STATE_WORDS)
from ari_analyze import DEFAULT_OUT  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from funaudiochat.register import register_funaudiochat  # noqa: E402
register_funaudiochat()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = "/share/workspace3/zhuangruicen/proj-thumpy/docs/pilot_study/figures"
MODEL_DIR = "/share/workspace3/shared_models/Fun-Audio-Chat-8B"
C_BLUE, C_ORANGE, C_AQUA, C_RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK2, GRID, BASE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
MODELS = [
    ("base", None),
    ("real_3.6k", f"{ANNOT}/f8_sft/saves"),
    ("real_12k", f"{ANNOT}/f8_sft_expanded/saves"),
    ("synthetic", f"{ANNOT}/f9_sft_synthetic/saves"),
]


def chunk_truth(chunk_id):
    """chunk -> (真重叠 BC 窗口 1s 块索引列表, 全部 AWS BC 窗口列表)."""
    with open(f"{ANNOT}/f8_training/manifest.jsonl") as f:
        mani = {json.loads(l)["id"]: json.loads(l) for l in f}
    m = mani[chunk_id]
    ev = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_events_w*.csv"))])
    e1 = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_e1_w*.csv"))])
    e1 = e1.merge(ev[["session", "event_id", "cls", "ch_event", "start", "end"]],
                  on="event_id", suffixes=("", "_ev"))
    sub = e1[(e1["session"] == m["session"]) & (e1["cls"] == "BC")
             & (e1["ch_event"] == int(m["ch"]))]
    ov_blocks, all_wins = set(), []
    for _, r in sub.iterrows():
        if r["end"] > m["t0"] and r["start"] < m["t0"] + 10:
            all_wins.append((r["start"], r["end"]))
            if r["realized"] != "None":
                for k in range(10):
                    a, b = m["t0"] + k, m["t0"] + k + 1
                    if a < r["end"] and b > r["start"]:
                        ov_blocks.add(k)
    return ov_blocks, all_wins


def behavior_metrics(pred_words, ov_blocks):
    pred_bc = {i for i, w in enumerate(pred_words) if w == "backchannel"}
    tp = len(pred_bc & ov_blocks)
    p = tp / len(pred_bc) if pred_bc else 0.0
    r = tp / len(ov_blocks) if ov_blocks else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"bc_precision": p, "bc_recall": r, "bc_f1": f1,
            "n_pred_bc": len(pred_bc), "n_ov_windows": len(ov_blocks)}


def annotator_reference(eval_set):
    """X2-Turn 融合标注器在相同窗口上的 F1 (τ=0.2, F6 口径)."""
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
    tp = fp_cnt = gt = 0
    for (wav, words) in eval_set:
        cid = os.path.basename(wav)[:-4]
        m = mlook[cid]
        if m["session"] not in x2 or str(m["ch"]) not in x2[m["session"]]:
            continue
        xt0, xt1, xp = x2[m["session"]][str(m["ch"])]
        ov_blocks, _ = chunk_truth(cid)
        gt += len(ov_blocks)
        for k in range(10):
            a, b = m["t0"] + k, m["t0"] + k + 1
            fm = (xt0 < b) & (xt1 > a)
            hit = fm.any() and xp[fm].max() >= 0.2
            pred_bc = bool(hit)
            if pred_bc:
                fp_cnt += 1
                tp += int(k in ov_blocks)
    p = tp / fp_cnt if fp_cnt else 0.0
    r = tp / gt if gt else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"bc_precision": p, "bc_recall": r, "bc_f1": f1}


def main():
    eval_set = build_eval_set(100)
    truths = [chunk_truth(os.path.basename(w)[:-4]) for (w, _) in eval_set]
    processor = AutoProcessor.from_pretrained(MODEL_DIR)

    res = {}
    for name, lora in MODELS:
        model = load_model(lora)
        aggs = {"bc_precision": [], "bc_recall": [], "bc_f1": [],
                "n_pred_bc": [], "n_ov_windows": []}
        dist = {w: 0 for w in STATE_WORDS}
        for i, (wav, words) in enumerate(eval_set):
            pred = predict(model, processor, wav, len(words))
            ov_blocks, _ = truths[i]
            m = behavior_metrics(pred, ov_blocks)
            for k in aggs:
                aggs[k].append(m[k])
            for w in pred:
                dist[w] += 1
            if (i + 1) % 25 == 0:
                print(f"{name} {i+1}/{len(eval_set)}", flush=True)
        res[name] = {k: round(float(np.mean(v)), 4) for k, v in aggs.items()}
        res[name]["state_dist"] = {w: round(v / max(1, sum(dist.values())), 4)
                                   for w, v in dist.items()}
        print(name, res[name], flush=True)
        del model

    res["annotator_ref"] = annotator_reference(eval_set)
    print("annotator_ref", res["annotator_ref"], flush=True)
    json.dump(res, open(f"{ANNOT}/f7_results.json", "w"), indent=2)

    # 图
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    names = [m[0] for m in MODELS] + ["annotator"]
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
    for s_ in ["left", "bottom"]:
        ax.spines[s_].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    prec = [res[n]["bc_precision"] for n in names[:-1]] + \
           [res["annotator_ref"]["bc_precision"]]
    rec = [res[n]["bc_recall"] for n in names[:-1]] + \
          [res["annotator_ref"]["bc_recall"]]
    f1 = [res[n]["bc_f1"] for n in names[:-1]] + \
         [res["annotator_ref"]["bc_f1"]]
    x = np.arange(len(names))
    w = 0.26
    ax.bar(x - w, prec, w, color=C_BLUE, label="precision")
    ax.bar(x, rec, w, color=C_ORANGE, label="recall")
    ax.bar(x + w, f1, w, color=C_AQUA, label="F1")
    ax.set_xticks(x, names, rotation=20, ha="right", fontsize=8.5, color=INK2)
    ax.set_ylabel("vs realized-overlap BC windows", fontsize=10, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model BC behavior vs real overlap", fontsize=11,
                 color=INK, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1]
    ax.set_facecolor(SURFACE)
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
    for s_ in ["left", "bottom"]:
        ax.spines[s_].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for j, wname in enumerate(STATE_WORDS):
        vals = [res[n]["state_dist"][wname] for n in names[:-1]]
        ax.plot(range(len(names) - 1), vals, "o-", markersize=4.5,
                label=wname)
    ax.set_xticks(range(len(names) - 1), [n[0] for n in MODELS],
                  rotation=20, ha="right", fontsize=8.5, color=INK2)
    ax.set_ylabel("predicted state fraction", fontsize=10, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_title("Predicted state distribution (collapse check)",
                 fontsize=11, color=INK, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle("F7: behavior metrics of trained models (frozen eval set)",
                 fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/f7_behavior.png", dpi=300,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved f7_results.json + f7_behavior.png")


if __name__ == "__main__":
    main()
