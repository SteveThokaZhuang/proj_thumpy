"""#57 P4: 查明为什么所有 run 的 loss 都记成 0.0.

症状 (全项目, 不是 mixnorm 特有):
  trainer_log.jsonl / trainer_state.json / train_results.json 里 **每一行** loss 都是 0.0,
  但同一行的 grad_norm 是 0.27–4.04, 而且 adapter 确实学到了东西 (F1 0.25–0.52)。
  全项目 19 个 run 全是这样 (含纯文本的 f8/f9), 所以跟音频/混合观测无关。

已读到的代码路径 (transformers 4.52.3 trainer.py):
  2457  tr_loss = torch.tensor(0.0, device=args.device)
  2562  tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
  2563  if (args.logging_nan_inf_filter and not is_torch_xla_available()
          and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))):
            tr_loss = tr_loss + tr_loss / (...)      # ← tr_loss 保持 0.0
        else:
            tr_loss = tr_loss + tr_loss_step
  3074  tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
  3079  logs["loss"] = round(tr_loss_scalar / (global_step - _globalstep_last_logged), 4)

  两个候选机制:
    (A) loss 每步都是 nan/inf -> 走过滤分支 -> tr_loss 永远 0.0 -> 记为 0.0
    (B) loss 有限但被 round(x, 4) 抹成 0 -> 需要 |loss| < 5e-5
  两者可以靠"打印真实 loss"一刀切开。grad_norm 有限这一点**偏向 (B) 不可能**
  (loss 5e-5 却 grad_norm 4.0 不自然), 但必须实测, 不能再猜。

本脚本做的事:
  在 run_exp() 之前猴补 Trainer.compute_loss / Trainer.training_step,
  打印**训练器真正拿到手**的那个 loss (不是我们另算的), 再跑 6 步。

用法 (必须 srun 到 GPU 节点, 且脚本在共享盘上):
  srun --overlap -j 67471 --gres=gpu:1 python scripts/_p4_loss_probe.py
"""
import os
import sys

REPO = "/share/workspace3/zhuangruicen/proj-thumpy/third_party/Fun-Audio-Chat"
LF = f"{REPO}/third_party/LLaMA-Factory/src"
sys.path.insert(0, REPO)
sys.path.insert(0, LF)

import torch  # noqa: E402
import transformers  # noqa: E402

CONFIG = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts/"
          "_p4_loss_probe.yaml")

_orig_compute = transformers.Trainer.compute_loss
_orig_step = transformers.Trainer.training_step
_state = {"n": 0}


def _f(x):
    """把一个可能是 None / tensor 的字段打印成人能读的短串。"""
    if x is None:
        return "None"
    if torch.is_tensor(x):
        return (f"{x.item():.6g}(nan={torch.isnan(x).item()},"
                f"inf={torch.isinf(x).item()})")
    return repr(x)


def compute_loss(self, model, inputs, return_outputs=False,
                 num_items_in_batch=None, **kw):
    _state["n"] += 1
    first = _state["n"] == 1
    # 第一次多要一次 outputs, 好把 FunAudioChat 的 text_loss / speech_loss 拆开看
    out = _orig_compute(self, model, inputs,
                        return_outputs=(return_outputs or first),
                        num_items_in_batch=num_items_in_batch, **kw)
    if return_outputs or first:
        loss, outputs = out
    else:
        loss, outputs = out, None
    if _state["n"] <= 12:
        labels = inputs.get("labels")
        n_lab = int(labels.numel()) if labels is not None else -1
        n_valid = int((labels != -100).sum()) if labels is not None else -1
        print(f"[P4] compute_loss#{_state['n']} "
              f"loss={_f(loss)} dtype={loss.dtype} "
              f"| labels {n_valid}/{n_lab} 非 -100 "
              f"| model_accepts_loss_kwargs={self.model_accepts_loss_kwargs} "
              f"| num_items_in_batch={num_items_in_batch}", flush=True)
        if first and outputs is not None:
            # ★ 决定性的三个数: 总 loss 是不是被 text/speech 其中之一毒化的
            print(f"[P4]   outputs.loss       = {_f(getattr(outputs, 'loss', 'MISSING'))}",
                  flush=True)
            print(f"[P4]   outputs.text_loss  = {_f(getattr(outputs, 'text_loss', 'MISSING'))}",
                  flush=True)
            print(f"[P4]   outputs.speech_loss= {_f(getattr(outputs, 'speech_loss', 'MISSING'))}",
                  flush=True)
            print(f"[P4]   grad_fn = {type(getattr(outputs, 'loss', None)).__name__} "
                  f"requires_grad={getattr(getattr(outputs, 'loss', None), 'requires_grad', None)}",
                  flush=True)
    return (loss, outputs) if return_outputs else loss


def training_step(self, model, inputs, num_items_in_batch=None):
    out = _orig_step(self, model, inputs, num_items_in_batch)
    if _state["n"] <= 12:
        print(f"[P4] training_step 返回 {out.item()!r} "
              f"(= loss/grad_accum; 这个数才是进 tr_loss 的)", flush=True)
        for name, p in list(model.named_parameters()):
            if p.grad is not None:
                print(f"[P4]   首个有梯度的参数 {name} "
                      f"grad_norm={p.grad.norm().item():.4g} "
                      f"finite={torch.isfinite(p.grad).all().item()}", flush=True)
                break
    return out


transformers.Trainer.compute_loss = compute_loss
transformers.Trainer.training_step = training_step

print(f"[P4] transformers {transformers.__version__}, torch {torch.__version__}",
      flush=True)
print(f"[P4] logging_nan_inf_filter 默认 = "
      f"{transformers.TrainingArguments(output_dir='/tmp/_x').logging_nan_inf_filter}",
      flush=True)

sys.argv = ["p4probe", CONFIG]
from llamafactory.train.tuner import run_exp  # noqa: E402

run_exp()
print("[P4] 结束", flush=True)
