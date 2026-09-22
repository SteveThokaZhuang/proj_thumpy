"""#57 P4 修复: 让训练 loss 变得可读 —— 关掉用不上的 speech 支路.

## 为什么要修

全项目 19 个 run 的 `trainer_log.jsonl` 里, **每一行** loss 都是 0.0, 而同行的
grad_norm 是 0.27–4.04。查下来是一条完整的链:

1. `Fun-Audio-Chat-8B/config.json` 里 `enable_audio_invert_tower=True`, 且
   `modeling_funaudiochat.py:793` 把 `sp_gen_kwargs['disable_speech']` 硬编码成
   False → **每次 forward 都会算 speech_loss**。
2. 我们的响应是纯文本 (`"2.7s (0.76)"` / `"no backchannel"`), 音频 token 全在
   prompt 里 (label=-100) → `speech_labels` 被整片 mask 成 ignore_index。
3. `ForCausalLMLoss` 在全 ignore 时 `reduction='mean'` 除零: **loss 值是 NaN,
   但梯度是 0(不是 NaN)** —— 已用合成实验证实 (`transformers.loss.loss_utils`)。
4. `modeling_funaudiochat.py:1148` 的 `loss = text_loss + 0.0; loss += speech_loss`
   → 总 loss = NaN。**但 speech 支路梯度恒为 0, 所以文本侧梯度完全正常。**
5. transformers 4.52.3 `trainer.py:2563` 的 `logging_nan_inf_filter`(默认 True)
   看到 NaN 就走"累计值不变"分支 → `tr_loss` 永远停在 0.0 → 日志记为 0.0。

⇒ **训练从来没坏过, 坏的是记录**。但也正因为如此, "用 loss 看是否欠拟合"这条路
   在本项目里从来就是走不通的 —— 不是没看, 是看不到。

## 这个修复做了什么

训练开始前把 `model.sp_gen_kwargs['disable_speech'] = True` (与 eval 里
`ari_f8_evaluate.py:79` 设 `text_greedy` 是同一个套路)。效果:

  - **梯度完全等价**: speech 支路梯度恒为 0, 关掉不改变任何一步更新。
  - 省掉 audio_invert_tower 的整次前向 (训练会快一截)。
  - 总 loss 退化成 `text_loss`, 是有限值 → 日志里终于能看到真实的 loss 曲线。

## 用法

不建议直接改 yaml (LLaMA-Factory 的配置里没有这个开关)。用包装器启动:

    from g1_text_only_train import run_text_only
    run_text_only("/path/to/config.yaml")

或在已有训练脚本里把 `llamafactory-cli train cfg.yaml` 换成:

    python -c "import sys; sys.path.insert(0,'scripts'); \
               from g1_text_only_train import run_text_only; run_text_only(sys.argv[1])" cfg.yaml
"""
import sys

REPO = "/share/workspace3/zhuangruicen/proj-thumpy/third_party/Fun-Audio-Chat"
LF = f"{REPO}/third_party/LLaMA-Factory/src"
for p in (REPO, LF):
    if p not in sys.path:
        sys.path.insert(0, p)

from transformers import TrainerCallback  # noqa: E402


def _unwrap(m):
    while hasattr(m, "module"):
        m = m.module
    return m


class DisableSpeechCallback(TrainerCallback):
    """把用不上的 speech 支路关掉, 让 loss 变成可读的 text_loss.

    **必须继承 `TrainerCallback`** (2026-09-17 修)。原先这里是鸭子类型对象,
    以为 LLaMA-Factory 只会按名字回调 `on_train_begin` —— 实际
    `transformers 4.52.3` 的 `trainer_callback.py:556` 是无默认值的
    `getattr(callback, event)`, 在 `on_init_end` 上直接:

        AttributeError: 'DisableSpeechCallback' object has no attribute 'on_init_end'

    继承之后其余事件都有 no-op 默认实现, 只覆写真正需要的那两个。
    """

    def on_train_begin(self, args, state, control, model=None, **kw):
        m = _unwrap(model) if model is not None else None
        if m is None or not hasattr(m, "sp_gen_kwargs"):
            print("[P4] ⚠️ 拿不到 model.sp_gen_kwargs, 没关成 speech —— "
                  "loss 大概率仍会记成 0.0", flush=True)
            return
        was = m.sp_gen_kwargs.get("disable_speech")
        m.sp_gen_kwargs["disable_speech"] = True
        print(f"[P4] disable_speech: {was} -> True "
              f"(梯度等价, 只为让 loss 可读 + 省一次 audio_invert_tower 前向)",
              flush=True)

    def on_log(self, args, state, control, logs=None, **kw):
        # 兜底: 万一还有 NaN 混进来, 要看得见, 而不是被静默写成 0.0。
        if logs and "loss" in logs and logs["loss"] != logs["loss"]:
            print("[P4] ⚠️ 这一步 loss 仍是 NaN —— 说明还有别的支路在出 NaN",
                  flush=True)


def run_text_only(config_path):
    """等价于 `llamafactory-cli train <config_path>`, 但先关掉 speech 支路。"""
    from llamafactory.train.tuner import run_exp

    print(f"[P4] 以 text-only 方式训练: {config_path}", flush=True)
    old_argv = sys.argv
    sys.argv = ["llamafactory-cli", config_path]
    try:
        run_exp(callbacks=[DisableSpeechCallback()])
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else None
    if not cfg:
        raise SystemExit("用法: python scripts/g1_text_only_train.py <config.yaml>")
    run_text_only(cfg)
