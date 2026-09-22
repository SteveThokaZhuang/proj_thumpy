"""#56 P1: 首 token logit 间隔探针 —— 直接测量"报/不报"的决策边界.

背景 (P0 的结论):
  mixnorm 的 F1 跨种子不稳定, 且**不是**优化问题 —— grad_norm 轨迹与 F1 无关
  (corr ≈ +0.1), 各 run 的 ‖B‖ 也量级相当。现象集中在**召回** (召回 SD 0.113 vs
  own10 0.029), 且 62% 的块是"7 个种子里 1-6 个报"的抛硬币块。

假设: mixnorm 把"报/不报"的决策整体推到阈值附近 -> 每个种子的 F1 变成对一堆硬币
的高方差抽样。**这是决策边界问题, 不是表征问题。**

本探针测的就是那个连续量。输出是生成式文本:
  「报」  = 第一个 token 开始一个数字 (如 "0.5s" 的 "0")
  「不报」= 第一个 token 开始 "no" (如 "no backchannel")
于是定义间隔
  margin = logsumexp(logits[数字起始 token]) - logsumexp(logits[no 起始 token])
margin 越接近 0 = 决策越边缘。

**为什么用 generate(max_new_tokens=1, output_scores=True) 而不是自己前向**:
  generate 会负责 prompt/decoder 起始 token 的一切细节, 不必关心这个模型是
  encoder-decoder 还是 decoder-only (它是 AutoModelForSeq2SeqLM, 但 eval 里按
  decoder-only 的方式切 generate_ids, 口径容易搞错)。而且只要 1 步解码, 比完整
  128 token 生成便宜一个量级。

用法:
  # 先验证 token 集合定义是否靠谱 (小样本, 打印首 token)
  python scripts/ari_g1_margin_probe.py --validate
  # 正式跑
  python scripts/ari_g1_margin_probe.py --n-chunks 400 --out <path>.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_f8_evaluate import ANNOT, load_model  # noqa: E402
from ari_g1_common import eval_chunk_ids, load_ids_file  # noqa: E402
from ari_g1_eval import AUDIO_TEMPLATE, INSTRUCTION  # noqa: E402
from funaudiochat.register import register_funaudiochat  # noqa: E402

register_funaudiochat()

TOKSET_CACHE = f"{ANNOT}/g1_margin_toksets.json"


def first_token_sets(tokenizer):
    """把词表切成「以数字开头」和「以 no 开头」两类首 token。

    用解码后的**第一个可见字符**判定, 而不是猜 token 边界 —— 音频 LLM 的词表里
    数字常被切成 "0" / " 0" / "0." 等多种形式, 白名单枚举容易漏。
    """
    if os.path.exists(TOKSET_CACHE):
        d = json.load(open(TOKSET_CACHE))
        return d["event"], d["none"]
    ev, none = [], []
    n = len(tokenizer)
    for tid in range(n):
        s = tokenizer.decode([tid])
        t = s.strip()
        if not t:
            continue
        if t[0].isdigit():
            ev.append(tid)
        if t[:2].lower() == "no":
            none.append(tid)
    json.dump({"event": ev, "none": none, "vocab": n},
              open(TOKSET_CACHE, "w"))
    return ev, none


def first_token_margin(model, processor, wav, chunk_s, ev_ids, none_ids):
    """返回 (margin, p_event, p_none, argmax_tid)。"""
    import librosa
    import torch
    audio = [librosa.load(wav, sr=16000)[0]]
    conversation = [{"role": "system", "content": ""},
                    {"role": "user", "content": AUDIO_TEMPLATE + INSTRUCTION}]
    text = processor.apply_chat_template(conversation,
                                         add_generation_prompt=True,
                                         tokenize=False)
    inputs = processor(text=text, audio=audio, return_tensors="pt",
                       return_token_type_ids=False).to(model.device)
    # 不能走 generate 拿 scores: Fun-Audio-Chat 的 _sample 结尾是
    # `return input_ids, speech_ids`, 内部收集的 scores 根本没返回。
    # 改用前向取末位分布 —— 已核对与 generate 的首个生成 token 逐块一致
    # (text_greedy=True ⇒ 贪心 ⇒ argmax 相等), 见 scripts/_check_forward.py。
    with torch.no_grad():
        lg = model(**inputs).logits[0, -1, :].float()

    if not ev_ids or not none_ids:
        return None
    ev_t = torch.tensor(ev_ids, device=lg.device)
    no_t = torch.tensor(none_ids, device=lg.device)

    # ⚠️ 两个口径, 别混用:
    #   margin_lse = logsumexp(数字类) - logsumexp(no 类)   —— 「哪一类总概率大」
    #   margin_max = max(数字类)     - max(no 类)           —— 「贪心会选哪一类」
    # 生成用的是**贪心** (text_greedy=True), 所以决定实际报/不报的是 margin_max。
    # margin_lse 有系统性偏置: 数字类被切成 28 个 token, no 类几乎只有 1 个, 求和的
    # 类概率天然偏向数字类 —— 实测有块 margin_lse=+1.05 而 argmax 仍是 'no'。
    # 因此 **margin_max 为主口径**, margin_lse 仅作参考。
    ev_lse = float(torch.logsumexp(lg[ev_t], 0))
    no_lse = float(torch.logsumexp(lg[no_t], 0))
    ev_max = float(lg[ev_t].max())
    no_max = float(lg[no_t].max())
    lse_all = float(torch.logsumexp(lg, 0))

    am = int(torch.argmax(lg))
    return {"margin_max": ev_max - no_max,
            "margin_lse": ev_lse - no_lse,
            "p_event": float(np.exp(ev_lse - lse_all)),
            "p_none": float(np.exp(no_lse - lse_all)),
            "argmax_tid": am,
            "argmax_is_event": am in set(ev_ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="", help="adapter 目录; 空 = base 模型")
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--npz-dir", default="", help="取 wav 的臂目录 (含 g1_eval_wavs_*)")
    ap.add_argument("--ids-file", default="")
    ap.add_argument("--n-chunks", type=int, default=400)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--chunk-s", type=float, default=10.0)
    ap.add_argument("--validate", action="store_true",
                    help="小样本: 打印首 token 及其解码文本, 用来核对 token 集合")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    processor = None
    model = load_model(args.lora or None)
    from transformers import AutoProcessor
    from ari_f8_evaluate import MODEL_DIR
    processor = AutoProcessor.from_pretrained(MODEL_DIR)

    tok = processor.tokenizer
    ev_ids, none_ids = first_token_sets(tok)
    print(f"token 集合: 数字起始 {len(ev_ids)} 个, no 起始 {len(none_ids)} 个 "
          f"(词表 {len(tok)})", flush=True)

    # 评估块
    if args.ids_file:
        ids = load_ids_file(args.ids_file, 0)
    else:
        ids = eval_chunk_ids(300)
    if args.n_chunks and args.n_chunks < len(ids):
        rng = np.random.default_rng(args.sample_seed)
        ids = sorted(rng.choice(ids, size=args.n_chunks, replace=False).tolist())
    print(f"探针块数: {len(ids)}", flush=True)

    wav_dir = (f"{args.npz_dir}/../g1_eval_wavs_"
               f"{os.path.basename(args.npz_dir.rstrip('/'))}")
    missing = [c for c in ids if not os.path.exists(f"{wav_dir}/{c}.wav")]
    if missing:
        raise SystemExit(
            f"❌ {len(missing)}/{len(ids)} 个块的 wav 不存在于 {wav_dir}\n"
            f"   例如 {missing[:3]}\n"
            f"   —— 评估集与臂目录不匹配? 注意 **E2 的 1600 块与旧 300 完全不相交**"
            f" (E2 ∩ g1_sub = 0), 评 E2 必须给 --ids-file <g1_eval2_ids.txt>。\n"
            f"   这里硬失败, 因为默默 continue 会写出一个 0 块的 json, "
            f"看起来像'跑完了'。")

    rows = {}
    for i, cid in enumerate(ids):
        wav = f"{wav_dir}/{cid}.wav"
        r = first_token_margin(model, processor, wav, args.chunk_s,
                               ev_ids, none_ids)
        if r is None:
            continue
        r["argmax_str"] = tok.decode([r["argmax_tid"]])
        for k in ("margin_max", "margin_lse"):
            r[k] = round(r[k], 4)
        rows[cid] = r
        if args.validate and i < 20:
            print(f"  {cid}  m_max={r['margin_max']:+8.3f} "
                  f"m_lse={r['margin_lse']:+8.3f}  p_ev={r['p_event']:.3f} "
                  f"p_no={r['p_none']:.3f}  argmax={r['argmax_str']!r}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"  {args.tag} {i+1}/{len(ids)}", flush=True)

    if args.validate:
        mm = np.array([v["margin_max"] for v in rows.values()])
        ml = np.array([v["margin_lse"] for v in rows.values()])
        fire = np.array([v["argmax_is_event"] for v in rows.values()])
        print(f"\n  margin_max: mean {mm.mean():+.3f}  median {np.median(mm):+.3f}  "
              f"|m|<1 的比例 {np.mean(np.abs(mm) < 1)*100:.0f}%")
        print(f"  margin_lse: mean {ml.mean():+.3f}  median {np.median(ml):+.3f}  "
              f"|m|<1 的比例 {np.mean(np.abs(ml) < 1)*100:.0f}%")
        # 决定性自检: margin_max 的符号必须与贪心是否报一致 (定义上就该一致)
        agree = ((mm > 0) == fire).mean()
        agree_lse = ((ml > 0) == fire).mean()
        print(f"\n  符号与贪心一致率: margin_max {agree*100:.1f}%   "
              f"margin_lse {agree_lse*100:.1f}%   ← max 口径应当接近 100%")
        n_fire = int(fire.sum())
        print(f"  贪心报的块: {n_fire}/{len(rows)} ({n_fire/len(rows)*100:.0f}%)")

    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"-> {args.out} ({len(rows)} 块)", flush=True)


if __name__ == "__main__":
    main()
