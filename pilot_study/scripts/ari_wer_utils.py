"""子串对齐 WER (仅 numpy, 供 x2-turn 等精简环境复用)."""
import re

import numpy as np


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower())


def edit_dist(r, h, free_prefix_del=False):
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    for i in range(len(r) + 1):
        d[i, 0] = 0 if free_prefix_del else i
    for j in range(len(h) + 1):
        d[0, j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d


def wer(ref, hyp, float_align=True):
    """WER; float_align=True 时转写沿参考做子串匹配 (流式转写缺头部 tokens)."""
    r = [w for w in norm(ref).split() if w]
    h = [w for w in norm(hyp).split() if w]
    if not r or not h:
        return float("nan")
    if not float_align:
        return edit_dist(r, h)[-1, -1] / len(r)
    d = edit_dist(r, h, free_prefix_del=True)
    best = min(d[i, len(h)] for i in range(len(r) + 1))
    return best / len(h)
