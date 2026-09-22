"""G1 消融不变式检查: 两臂必须只有音频不同.

检查项 (任一不满足即消融无效):
  1. 样本数、顺序、chunk id 完全一致
  2. 每条 assistant 答案 (标签) 完全一致
  3. user 指令完全一致
  4. 音频确实不同 (否则观测空间没生效)
  5. 音频峰值 <= 1 (PCM_16 落盘不削波)

用法: python scripts/ari_g1_check_arms.py <armA_dir> <armB_dir>
"""
import json
import os
import sys

import numpy as np
import soundfile as sf


def load(dir_):
    rows = []
    for line in open(f"{dir_}/train.jsonl"):
        r = json.loads(line)
        rows.append({
            "cid": os.path.basename(json.loads(r["audio"])["path"])[:-4],
            "wav": json.loads(r["audio"])["path"],
            "user": r["messages"][0]["content"],
            "answer": r["messages"][1]["content"],
        })
    return rows


def main():
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    A, B = load(a_dir), load(b_dir)
    name = lambda d: os.path.basename(d.rstrip("/"))
    ok = True

    def chk(label, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK ' if cond else 'FAIL'}] {label} {detail}")

    print(f"=== {name(a_dir)}  vs  {name(b_dir)}  (n={len(A)} / {len(B)}) ===")
    chk("样本数一致", len(A) == len(B), f"{len(A)} vs {len(B)}")
    chk("chunk id 顺序一致", [r["cid"] for r in A] == [r["cid"] for r in B])
    chk("答案逐条一致", [r["answer"] for r in A] == [r["answer"] for r in B])
    chk("指令逐条一致", [r["user"] for r in A] == [r["user"] for r in B])
    for nm, rows in ((name(a_dir), A), (name(b_dir), B)):
        n_pos = sum(1 for r in rows if r["answer"] != "no backchannel")
        print(f"  {nm}: 正例 {n_pos} 负例 {len(rows) - n_pos}")

    # 音频差异 + 峰值 (抽样, 全量太慢)
    idx = list(range(0, len(A), max(1, len(A) // 150)))[:150]
    diff, pka, pkb = [], [], []
    for i in idx:
        x, _ = sf.read(A[i]["wav"], dtype="float32")
        y, _ = sf.read(B[i]["wav"], dtype="float32")
        if x.shape != y.shape:
            print(f"  [FAIL] 形状不一致 {A[i]['cid']}: {x.shape} vs {y.shape}")
            ok = False
            continue
        diff.append(float(np.max(np.abs(x - y))))
        pka.append(float(np.max(np.abs(x))))
        pkb.append(float(np.max(np.abs(y))))
    diff = np.array(diff)
    chk("音频确有差异 (观测空间生效)", diff.min() > 1e-6,
        f"min|diff| {diff.min():.4f} 中位 {np.median(diff):.4f}")
    for nm, pk in ((name(a_dir), pka), (name(b_dir), pkb)):
        pk = np.array(pk)
        chk(f"{nm} 峰值 <= 1 (不削波)", pk.max() <= 1.0,
            f"max {pk.max():.4f} 超1比例 {(pk > 1).mean():.4f}")

    print("\n结论:", "不变式成立, 消融有效" if ok else "**不变式被破坏, 结果不可用**")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
