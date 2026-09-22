"""钉死块 id 里 `t0` 的语义 —— 纯读盘, 0 GPU。

## 为什么要问这个

2026-09-18 一次诊断里, 我把 `t0` 当成了**毫秒**, 于是算出「同会话 12 块的时间跨度
中位数只有 1.6 秒」, 进而怀疑「同会话的块几乎重合 ⇒ 有效样本量是 100 个会话而不是
1600 块 ⇒ 全项目的 SE 都被低估」。为此还跑了会话簇 bootstrap 去检验。

**簇 bootstrap 给 0.97× (块级 0.009772 vs 会话簇 0.009447)** —— 聚类假说被排除。
但那个假说的**前提**本身是我读错的: `t0` 的单位是**秒**, 不是毫秒。
跨度中位数是 **1595 秒 = 26.6 分钟**, 完全正常。

⇒ 本节把 `t0` 的语义**从代码和实测两头钉死**, 免得再有人 (包括我) 按错的单位推理。

## 语义 (三重证据)

1. **代码**: `ari_g1_eval4_ids.py:127` 里 `t = 0.0` 起步、`t += STRIDE`(5.0),
   循环条件 `while t + CHUNK_S(10.0) <= tmax`, 而 `tmax = t0a[-1]` 是**帧时间戳(秒)**。
   id 由 `f"{s8}_ch{ch}_t{int(t)}"` 拼成 ⇒ id 里是**整数秒**。
2. **建库**: `ari_g1_prep_subset.py:161` 用 `int(t0 * 16000)` 切片 ⇒ 16 kHz 下
   `t0` 是秒。
3. **结构**: 轮转取样的键是 **(session, ch)**, e4x 下 200 个键**每个正好 8 块**
   (100 会话 × 2 声道), 1600 = 200 × 8。

## 实测判据 (本脚本做的那条)

`t0` 是秒 ⇒ 两个同 (session,ch) 的块, 当 `Δt < 10` 秒时**必须共享音频**,
且最佳对齐滞后 = `-Δt × 16000` 采样, 归一化互相关 (NCC) 应当很高。
若 `t0` 是毫秒, `Δt=105` 那种"看似很近"的对应该共享 89.5% 音频 —— 实测 NCC ≈ 0。

**自检 (会炸)**: 同一块与自身在 lag=0 处 NCC 必须 = 1.000。

用法: python scripts/g1_e4x_t0_semantics.py
"""
import collections
import os
import sys

import numpy as np

ANNOT = ("/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/"
         "real_data/results/annotator")
SR = 16000
CHUNK_S = 10.0


def ids_of(name):
    return [l.strip() for l in open(f"{ANNOT}/{name}") if l.strip()]


def keyed(ids):
    """(session, ch) -> [t0, ...];  t0 按**秒**解析。

    ⚠️ 这里 `ch` 保留 id 里的原样 `'ch0'`/`'ch1'` (带前缀) —— 早先一版把它当成
    裸的 `'0'`/`'1'`, 拼路径时又补了一次前缀, 于是找的是 `..._chch0_t360`。
    键与路径**用同一个 token**, 不再二次加工。
    """
    g = collections.defaultdict(list)
    for i in ids:
        s, ch, t = i.rsplit("_", 2)
        g[(s, ch)].append(float(t[1:]))
    return g


def ncc(a, b, lag):
    """a 与 b 在给定滞后下的归一化互相关。lag<0 ⇒ b 相对 a **延后** |lag|。"""
    if lag < 0:
        x, y = a[-lag:], b[:len(a) + lag]
    elif lag > 0:
        x, y = a[:len(a) - lag], b[lag:]
    else:
        x, y = a, b
    if len(x) < 1000:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d else float("nan")


def audio(s, ch, t):
    """e4 与 e4b 的 npz 分开存, 两处都找。

    ⚠️ `ch` 是 id 里的原样 token (`'ch0'`), 路径直接拼 `{s}_{ch}_t{...}` ——
    和 `keyed()` 用同一个形状, 别在这里补 `_ch` 前缀。
    """
    name = f"{s}_{ch}_t{int(t)}"
    for sub in ("g1_e4", "g1_e4b"):
        p = f"{ANNOT}/{sub}/mixnorm/{name}.npz"
        if os.path.exists(p):
            return np.load(p)["audio"].astype(np.float64)
    raise FileNotFoundError(f"{name} 在两个库里都没有")


def ncc_scan(a, b, max_shift, stride=160):
    """在 |lag| ≤ max_shift 里扫过所有对齐, 返回最大 NCC。

    **负对照专用**。`ncc(a, b, lag)` 在 |lag| 超过块长时**没有定义** (切片为空
    ⇒ NaN), 而 NaN 掉进 `abs(x) < 0.3` 这种比较里会静默变成 False —— 于是
    「判据落进没定义的区间」被误读成「判据不通过」。这里把搜索范围**卡在数据
    能支撑的区间内**, 值域铺满, 不留 NaN。

    stride=160 ⇒ 步进 10 ms, 对 16 kHz 音频足够密 (块长 10s 时约 201 个滞后)。
    """
    best, at = -1.0, 0
    for lag in range(-max_shift, max_shift + 1, stride):
        v = ncc(a, b, lag)
        if v == v and v > best:   # v == v 排除 NaN
            best, at = v, lag
    return best, at


def main():
    ids = ids_of("g1_eval4x_ids.txt")
    g = keyed(ids)

    print("=" * 78)
    print("  t0 语义钉死: 单位是**秒**, 不是毫秒")
    print("=" * 78)
    print(f"  块 {len(ids)}   键(session,ch) {len(g)}   "
          f"每键块数 {sorted(set(len(v) for v in g.values()))}"
          f"   ⇒ {len(g)} × {len(ids)//len(g)} = {len(ids)}")

    allt = [t for v in g.values() for t in v]
    print(f"  t0 范围 [{min(allt):.0f}, {max(allt):.0f}] 秒 "
          f"= [{min(allt)/60:.1f}, {max(allt)/60:.1f}] 分钟  (会话长度正常)")

    gaps = []
    for v in g.values():
        v = sorted(v)
        gaps += [b - a for a, b in zip(v, v[1:])]
    gaps.sort()
    n_ov = sum(1 for x in gaps if x < CHUNK_S)
    print(f"  同键相邻块间隔(秒): 最小 {gaps[0]:.0f}  中位 {gaps[len(gaps)//2]:.0f}  "
          f"最大 {gaps[-1]:.0f}")
    print(f"  ⇒ 间隔 < {CHUNK_S:.0f}s (=**会共享音频**) 的对: "
          f"{n_ov} / {len(gaps)}")
    print(f"  ⚠️ 若 t0 是毫秒, 间隔 105 的那种对会被读成 1.05s ⇒ 该共享 89.5% 音频;")
    print(f"     按秒读则是 105s ⇒ 毫无重合。下面用波形直接判。")

    # ---------- 自检: 块与自身的 NCC 必须是 1 ----------
    s0, ch0 = next(iter(g))
    t_any = sorted(g[(s0, ch0)])[0]
    a0 = audio(s0, ch0, t_any)
    self_ncc = ncc(a0, a0, 0)
    print(f"\n  ── 自检 ──")
    print(f"    块与自身 lag=0: NCC = {self_ncc:.6f}  "
          f"{'✅' if abs(self_ncc - 1) < 1e-6 else '🔴 NCC 机器坏了, 下面的数不作数'}")
    if abs(self_ncc - 1) > 1e-6:
        return 1

    # ---------- 判据 1: 会重合的对, 必须在预测滞后上高度相关 ----------
    pairs = []
    for k, v in g.items():
        v = sorted(v)
        pairs += [(k, a, b) for a, b in zip(v, v[1:]) if b - a < CHUNK_S]
    pairs.sort(key=lambda x: x[2] - x[1])
    print(f"\n  ── 判据 1: 同键且 Δt<10s 的 {len(pairs)} 对, "
          f"在预测滞后 -Δt×{SR} 上的 NCC ──")
    print(f"    {'块对':<40}{'Δt(s)':>7}{'理论重合':>10}{'NCC@预测':>11}{'NCC@0':>10}")
    vals = []
    for (s, ch), t1, t2 in pairs[:8]:
        a, b = audio(s, ch, t1), audio(s, ch, t2)
        dt = t2 - t1
        v_pred = ncc(a, b, -int(round(dt * SR)))
        v0 = ncc(a, b, 0)
        vals.append(v_pred)
        print(f"    {s + '_' + ch + '_' + str(int(t1)) + '-' + str(int(t2)):<40}"
              f"{dt:>7.0f}{max(0.0, CHUNK_S - dt):>9.1f}s{v_pred:>11.3f}{v0:>10.3f}")
    med = float(np.median(vals))
    print(f"    预测滞后上 NCC 中位数 = {med:.3f}  "
          f"{'✅ 高 ⇒ t0 是秒, 且切片对齐' if med > 0.7 else '🔴 低 —— 语义推断错了'}")

    # ---------- 判据 2: 间隔远大于块长的对, **任何对齐**都找不到重合 ----------
    # ⚠️ 早先一版在这里问「预测滞后上的 NCC 是多少」, 但 Δt > 块长 ⇒ 该滞后
    #    越界 ⇒ ncc() 返回 NaN ⇒ 被 `abs(x) < 0.3` 静默判为不通过, 打出 🔴。
    #    那是**判据落进没定义的区间**, 不是实测发现。改成扫遍数据能支撑的滞后。
    L = int(CHUNK_S * SR)
    far = [(k, a, b) for k, v in g.items() for a, b in zip(sorted(v), sorted(v)[1:])
           if b - a > 100]
    far.sort(key=lambda x: x[2] - x[1])
    SCAN_S = 2.0
    print(f"\n  ── 判据 2 (负对照): 间隔 >100s 的对, Δt > 块长 {CHUNK_S:.0f}s "
          f"⇒ 重合**结构上为 0**;")
    print(f"     在 |lag| ≤ {SCAN_S:.0f}s 里扫遍所有对齐, 都不该出现高 NCC ──")
    print(f"    {'块对':<40}{'Δt(s)':>7}{'重合(s)':>9}{'最佳NCC':>10}{'@lag(s)':>9}")
    vals2 = []
    for (s, ch), t1, t2 in far[:5]:
        a, b = audio(s, ch, t1), audio(s, ch, t2)
        dt = t2 - t1
        ov = max(0.0, CHUNK_S - dt)
        assert ov == 0.0, f"{s}_{ch} Δt={dt} 竟然有重合, 判据 2 的前提不成立"
        v, lag = ncc_scan(a, b, int(SCAN_S * SR))
        assert v == v, f"{s}_{ch} 扫描出了 NaN —— 值域没铺满, 判据不可用"
        vals2.append(v)
        print(f"    {s + '_' + ch + '_' + str(int(t1)) + '-' + str(int(t2)):<40}"
              f"{dt:>7.0f}{ov:>9.1f}{v:>10.3f}{lag / SR:>9.2f}")
    mx2 = float(np.max(vals2))
    tag2 = ("✅ 最大 %.3f ⇒ 无任何对齐能对上一段不存在的共同音频"
            % mx2) if mx2 < 0.7 else "🔴 扫出了高 NCC —— t0 语义或切片有问题"
    print(f"    最大 NCC = {mx2:.3f}  {tag2}")

    # ---------- 判据 3: 上面的对若按**毫秒**读, 该发生什么 ----------
    print(f"\n  ── 判据 3: 把 t0 按**毫秒**读会预测什么 (反证) ──")
    print(f"    取最接近的一对 (Δt={pairs[0][2]-pairs[0][1]:.0f}), "
          f"按毫秒读 Δt={pairs[0][2]-pairs[0][1]:.0f}ms ⇒ 重合 "
          f"{max(0.0, CHUNK_S - (pairs[0][2]-pairs[0][1])/1000):.2f}s ⇒ NCC 该接近 1")
    (s, ch), t1, t2 = pairs[0]
    a, b = audio(s, ch, t1), audio(s, ch, t2)
    v_ms = ncc(a, b, -int(round((t2 - t1) / 1000 * SR)))
    tag3 = "🔴 高 —— 那 t0 真是毫秒" if v_ms > 0.7 else "✅ 低 ⇒ 毫秒假说被证伪"
    print(f"    实测按毫秒滞后的 NCC = {v_ms:.3f}  {tag3}")

    print(f"\n  🔑 结论: `t0` = 该块在**会话内的起点, 单位秒**; id 里取整。")
    print(f"     块长 10s、候选池步进 5s ⇒ **候选池内**相邻窗口共享 50% 音频;")
    print(f"     但**被选中的块**在会话内几乎均匀散开 (间隔中位 {gaps[len(gaps)//2]:.0f}s),")
    print(f"     只有 {n_ov}/{len(gaps)} 对真的重叠 ⇒ **块间音频重合不是常态**。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
