#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 split 汇总 statistics.csv -> per-split 指标表 (用于日志/文档)。

用法: python scripts/per_split_summary.py
输出: real_data/results/analysis/per_split_summary.csv + 控制台表格
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
CSV = BASE / "real_data" / "results" / "analysis" / "statistics.csv"


def main():
    # keep_default_na=False: CSV 中字符串 'None' 是真实标签, 不能被读成 NaN
    df = pd.read_csv(CSV, keep_default_na=False)
    rows = []
    for split, g in df.groupby("split"):
        ok = g[~g["diar_error"] & (g["duration_s"] > 0)]
        r = np.corrcoef(ok["gt_overlap_total"], ok["det_overlap_total"])[0, 1]
        rows.append({
            "split": split,
            "n": len(g),
            "label=GT 一致率%": round((g["original_label"] == g["gt_realized"]).mean() * 100, 1),
            "label=检测 一致率%": round((g["original_label"] == g["det_realized"]).mean() * 100, 1),
            "r (GT vs 检测重叠总量)": round(float(r), 3),
            "GT 重叠总量均值 s": round(g["gt_overlap_total"].mean(), 2),
            "检测重叠总量均值 s": round(g["det_overlap_total"].mean(), 2),
            "GT 事件数均值": round(g["gt_overlap_events"].mean(), 2),
            "检测事件数均值": round(g["det_overlap_events"].mean(), 2),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False), flush=True)
    out.to_csv(BASE / "real_data" / "results" / "analysis" / "per_split_summary.csv",
               index=False)
    print("PER SPLIT DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
