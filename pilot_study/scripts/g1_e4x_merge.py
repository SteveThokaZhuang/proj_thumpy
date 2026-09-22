"""把 k=4 (1200 块) 与 k=4b (新增 400 块) 的产物合并成 1600 块的一整套。

## 为什么是 merge 而不是重跑

`ari_g1_eval4_ids.py` 的轮转取样是**前缀稳定**的: `--n-eval 1600` 选出的前
1200 块与既有 `g1_eval4_ids.txt` **逐位相同** (选块时已 assert)。所以 1600 块
的评估集 = 既有 1200 块 ∪ 新增 400 块, 且两部分**同总体、同程序、同会话覆盖**。

⇒ 只评新增的 400 块, 再把 `per_chunk` 拼起来。重跑 1200 块不仅浪费时间,
而且是**错的**: 那会把 §5.8b 挂着的逐块结果 (SE_chunk = 0.009725,
ΔF1 = +0.0210) 换成另一批随机数, 无谓地切断与既有结论的可追溯性。

## 三条自检 (每条都能失败, 都能抓到一类真 bug)

1. **聚合器先对已知答案**: 对 14 个 e4 源文件, 用本脚本的 `agg()` 从 `per_chunk`
   重算 precision/recall/f1/n_pred/n_gt, 必须与文件里**存着的**聚合值一致
   (f1 到 4 位小数)。对不上说明聚合器写错了 —— 那么合并后的聚合值也不可信。
2. **两部分必须不相交**: 有交集就意味着一块被算了两遍, 合并后的 F1 会被
   那一块按权重扭曲, 而且**不会有任何报错**。
3. **并集必须恰好等于 id 清单**: 防的是"某次评估跑到一半死了", 静默产出一个
   1500 块的集 —— 缺的那 100 块若恰好偏正例, 整个结论会朝一个方向偏。

用法 (纯读盘, 0 GPU, 任一环境):
  python scripts/g1_e4x_merge.py            # 合并
  python scripts/g1_e4x_merge.py --check    # 只自检, 不写盘
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_e4_analyze import ANNOT, ARMS, SEEDS, suffix  # noqa: E402

SRC = ["e4", "e4b"]          # 旧 1200 块 / 新 400 块
DST = "e4x"                  # 合并后的标签 (1600 块)


def agg(pc):
    """从 per_chunk 重算聚合量 —— 与 ari_g1_eval.py 的算法同口径。"""
    tp = sum(e["tp"] for e in pc.values())
    fp = sum(e["fp"] for e in pc.values())
    gt = sum(e["n_gt"] for e in pc.values())
    p = tp / max(1, tp + fp)
    r = tp / max(1, gt)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f1, 4), "n_pred": tp + fp, "n_gt": gt,
            "n_chunks": len(pc)}


def path(tag_set, arm, seed):
    return f"{ANNOT}/g1_eval_{arm}{suffix(seed)}_{tag_set}.json"


def load(tag_set, arm, seed):
    p = path(tag_set, arm, seed)
    if not os.path.exists(p):
        return None
    return next(iter(json.load(open(p)).values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只自检, 不写盘")
    args = ap.parse_args()

    # ---------- 期望的 id 清单 (合并后的全集必须恰好是它) ----------
    ids_e4 = [l.strip() for l in open(f"{ANNOT}/g1_eval4_ids.txt") if l.strip()]
    ids_e4b = [l.strip() for l in open(f"{ANNOT}/g1_eval4b_ids.txt") if l.strip()]
    ids_all = [l.strip() for l in open(f"{ANNOT}/g1_eval4x_ids.txt") if l.strip()]
    print(f"== 合并 {SRC[0]} ({len(ids_e4)}) + {SRC[1]} ({len(ids_e4b)}) "
          f"-> {DST} ({len(ids_all)}) ==")

    # 选块阶段的结论在这里**再查一遍** —— 合并脚本不该相信上游的注释
    if ids_e4 + ids_e4b != ids_all:
        print("🔴 id 清单不是「旧 + 新」的顺序拼接 —— 停。")
        print("   (把 e4b 定义成 e4x 去掉 e4 的差集, 这个等式就该成立)")
        return 1
    if set(ids_e4) & set(ids_e4b):
        print("🔴 两部分 id 有交集!")
        return 1
    print("   ✅ id 清单: e4 + e4b == e4x (顺序拼接), 交集 0")

    # ---------- 自检 1: 聚合器对 14 个源文件 ----------
    print("\n== 自检 1: 聚合器能否复现 14 个 e4 源文件里存着的聚合值 ==")
    bad = []
    for arm in ARMS:
        for s in SEEDS:
            e = load("e4", arm, s)
            if e is None:
                bad.append((arm, s, "缺文件"))
                continue
            got = agg(e["per_chunk"])
            for k in ("precision", "recall", "f1", "n_pred", "n_gt"):
                if got[k] != e.get(k):
                    bad.append((arm, s, f"{k}: 重算 {got[k]} vs 存着 {e.get(k)}"))
    if bad:
        print(f"   🔴 {len(bad)} 处对不上, 前 5:")
        for b in bad[:5]:
            print("     ", b)
        print("   ⇒ 聚合器有 bug, 停 (合并后的聚合值同样会错)。")
        return 1
    print("   ✅ 14/14 源文件的聚合值逐位复现 —— 聚合器可信")

    # ---------- 自检 2 & 3 + 写盘 ----------
    print(f"\n== 自检 2/3: 不相交 + 并集 == id 清单 ==")
    n_written = 0
    for arm in ARMS:
        for s in SEEDS:
            objs = {}
            for t in SRC:
                e = load(t, arm, s)
                if e is None:
                    print(f"   🔴 缺 {path(t, arm, s)} —— 停 (不产生残缺的集)")
                    return 1
                objs[t] = e
            pc_a, pc_b = objs["e4"]["per_chunk"], objs["e4b"]["per_chunk"]
            inter = set(pc_a) & set(pc_b)
            if inter:
                print(f"   🔴 {arm}{suffix(s)}: e4 与 e4b 有 {len(inter)} 块重叠!")
                return 1
            merged = {**pc_a, **pc_b}
            if set(merged) != set(ids_all):
                miss = set(ids_all) - set(merged)
                extra = set(merged) - set(ids_all)
                print(f"   🔴 {arm}{suffix(s)}: 并集 {len(merged)} != 清单 {len(ids_all)}"
                      f" (缺 {len(miss)}, 多 {len(extra)})")
                if miss:
                    print(f"      缺的例如 {sorted(miss)[:3]}")
                return 1
            if args.check:
                n_written += 1
                continue
            a = agg(merged)
            # 溯源: 放在 arm dict **内部** —— 顶层若加第二个 key, 下游
            # `next(iter(...))` 取第一个 value 的写法就会变脆。
            a["per_chunk"] = merged
            a["merge"] = {
                "from": SRC, "n_from": [len(pc_a), len(pc_b)],
                "union": len(merged), "disjoint": True,
                "id_list": f"{ANNOT}/g1_eval4x_ids.txt",
                "note": "e4x = e4(1200) ∪ e4b(400); 前缀稳定, 1200 块的逐块结果原样保留",
            }
            json.dump({f"{arm}_x": a}, open(path(DST, arm, s), "w"),
                      indent=2, ensure_ascii=False)
            n_written += 1
            if s == SEEDS[0]:
                print(f"   {arm}: f1={a['f1']} n={a['n_chunks']} "
                      f"gt={a['n_gt']} pred={a['n_pred']}")

    if args.check:
        print(f"\n   ✅ --check: 14 对全部不相交且并集 == 清单 (未写盘)")
        return 0
    print(f"\n   ✅ 写盘 {n_written} 个 -> g1_eval_*_{DST}.json")

    # ---------- 收尾: 新集的基本量 ----------
    e = load(DST, "own10", SEEDS[0])
    pc = e["per_chunk"]
    n_pos = sum(1 for v in pc.values() if v["n_gt"] > 0)
    print(f"\n== {DST} 基本量 (own10/s42) ==")
    print(f"   块 {len(pc)}   真值事件 {e['n_gt']}   含事件块 {n_pos} "
          f"({n_pos/len(pc):.2%} 正例率)")
    print(f"   ⚠️ 与 e4 的 3.58% 不必逐位相同: 新增 400 块是同一总体的下一次"
          f"轮转抽样, 有小幅抽样波动 (选块时实测全集 4.06%)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
