"""F5 分析: Fisher 跨域标注质量 (fd_analysis 环境).

从 f5_fisher/*.json 计算:
  - p_speak 帧 AUC vs 人工转写 utterance 隶属
  - p_bc 帧 AUC vs 短 utterance (<=1.5s) 隶属 (人工转写的 BC 候选)
  - 子串对齐 WER vs LDC 人工转写
输出: f5_results.json
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_wer_utils import wer  # noqa: E402

ANNOT = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator"
OUT = f"{ANNOT}/f5_fisher"


def main():
    files = sorted(f for f in glob.glob(f"{OUT}/*.json")
                   if not f.endswith("f5_results.json"))
    parts_sp, parts_bc, wers = [], [], []
    n_short_utt = 0
    for fp in files:
        d = json.load(open(fp))
        for ch in ("0", "1"):
            fr = d["frames"][ch]
            if not fr:
                continue
            t0a = np.array([f[0] for f in fr])
            t1a = np.array([f[1] for f in fr])
            p_bc = np.array([f[2] for f in fr])
            p_sp = np.array([f[3] for f in fr])
            utts = [(a, b) for (a, b, t) in d["gt"][ch]]
            bcs = [(a, b) for (a, b, t) in d["gt"][ch] if b - a <= 1.5]
            n_short_utt += len(bcs)
            ins_u = np.zeros(len(t0a), bool)
            for (a, b) in utts:
                ins_u |= (t0a < b) & (t1a > a)
            ins_b = np.zeros(len(t0a), bool)
            for (a, b) in bcs:
                ins_b |= (t0a < b) & (t1a > a)
            if ins_u.any() and (~ins_u).any():
                neg = np.where(~ins_u)[0]
                if len(neg) > 50000:
                    neg = np.random.default_rng(0).choice(
                        neg, 50000, replace=False)
                idx = np.concatenate([np.where(ins_u)[0], neg])
                parts_sp.append((p_sp[idx], ins_u[idx]))
            if ins_b.any() and (~ins_b).any():
                neg = np.where(~ins_b)[0]
                if len(neg) > 50000:
                    neg = np.random.default_rng(0).choice(
                        neg, 50000, replace=False)
                idx = np.concatenate([np.where(ins_b)[0], neg])
                parts_bc.append((p_bc[idx], ins_b[idx]))
            ref = " ".join(t for (a, b, t) in d["gt"][ch])
            w = wer(ref, d["transcript"].get(ch, ""))
            if w == w:
                wers.append(w)

    res = {
        "n_conversations": len(files),
        "n_short_utterances": int(n_short_utt),
        "auc_speak_vs_utterance": round(
            float(np.mean([roc_auc_score(y, s) for (s, y) in parts_sp])), 4)
        if parts_sp else None,
        "auc_bc_vs_short_utterance": round(
            float(np.mean([roc_auc_score(y, s) for (s, y) in parts_bc])), 4)
        if parts_bc else None,
        "wer_median": round(float(np.median(wers)), 4) if wers else None,
        "wer_mean": round(float(np.mean(wers)), 4) if wers else None,
    }
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(f"{OUT}/f5_results.json", "w"), indent=2)
    print("saved f5_results.json")


if __name__ == "__main__":
    main()
