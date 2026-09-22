"""核对: 直接前向的 logits[-1] 是否等价于 generate 的首个生成 token.

Fun-Audio-Chat 的 generate 被改成返回 (input_ids, speech_ids), 丢了 scores,
所以探针不能走 generate 拿 logits。改用前向。但前提是「前向末位分布 == 首生成位
分布」成立 —— 本脚本就是验证这一点 (text_greedy=True, 故 generate 应为 argmax)。
"""
import sys

sys.path.insert(0, "/share/workspace3/zhuangruicen/proj-thumpy/third_party/Fun-Audio-Chat")
sys.path.insert(0, "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts")

import librosa  # noqa: E402
import torch  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

from ari_f8_evaluate import ANNOT, MODEL_DIR, load_model  # noqa: E402
from ari_g1_eval import AUDIO_TEMPLATE, INSTRUCTION  # noqa: E402
from funaudiochat.register import register_funaudiochat  # noqa: E402

register_funaudiochat()

model = load_model(f"{ANNOT}/g1_mixnorm_sft/saves")
proc = AutoProcessor.from_pretrained(MODEL_DIR)
ids = [ln.strip() for ln in open(f"{ANNOT}/g1_eval2_ids.txt") if ln.strip()][:6]
wd = f"{ANNOT}/g1_e2/g1_eval_wavs_mixnorm"

for cid in ids:
    audio = [librosa.load(f"{wd}/{cid}.wav", sr=16000)[0]]
    conv = [{"role": "system", "content": ""},
            {"role": "user", "content": AUDIO_TEMPLATE + INSTRUCTION}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True,
                                    tokenize=False)
    inp = proc(text=text, audio=audio, return_tensors="pt",
               return_token_type_ids=False).to(model.device)
    with torch.no_grad():
        gids, _ = model.generate(**inp, max_new_tokens=1)
        gen_tok = gids[0, inp.input_ids.size(1)].item()
        out = model(**inp)
        lg = out.logits[0, -1, :].float()
        fwd_tok = int(torch.argmax(lg))
    print(f"{cid}  generate={proc.tokenizer.decode([gen_tok])!r:<14} "
          f"forward_argmax={proc.tokenizer.decode([fwd_tok])!r:<14} "
          f"{'MATCH' if gen_tok == fwd_tok else '**DIFF**'}  "
          f"logits={tuple(out.logits.shape)}")
