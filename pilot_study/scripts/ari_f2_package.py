"""F2 打包: CANDOR-FD v0 基准索引与统计.

分层标签 (同一批真实会话):
  AWS      : backbiter/audiophile 派生标签 (L2 参考层)
  realized : 声道级 RMS 双活跃裁决 (声学验证层)
  X2-Turn  : 80ms 帧级 p_bc (L3 主标注层, 100 会话)
  SoulX    : 160ms backchannel 状态 (L3 补充层, 30 会话, BC 邻域区域)
  fused    : mean 融合软分数 (F1 配方, 30 会话可用处)

输出: annotator/candor_fd_v0/index.json + README.md + summary.json
"""
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_analyze import DEFAULT_OUT  # noqa: E402

ANNOT = "/share/workspace3/zhuangruicen/proj-thumpy/pilot_study/real_data/results/annotator"
V0 = f"{ANNOT}/candor_fd_v0"


def main():
    os.makedirs(V0, exist_ok=True)
    x2_files = {os.path.basename(f)[:-5]: f
                for f in glob.glob(f"{ANNOT}/e4_raw/*.json")}
    mani = json.load(open(f"{ANNOT}/e5_regions_manifest.json"))
    soulx_sessions = set(m["session"] for m in mani.values())

    ev = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_events_w*.csv"))])
    e1 = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_e1_w*.csv"))])
    e1 = e1.merge(ev[["session", "event_id", "cls", "ch_event", "start", "end"]],
                  on="event_id", suffixes=("", "_ev"))

    index = {}
    n_tot = {"sessions": 0, "bc_windows": 0, "with_soulx": 0,
             "with_x2": 0, "with_transcript": 0}
    for s, fp in sorted(x2_files.items()):
        d = json.load(open(fp))
        has_tr = any(d["channels"][ch].get("transcript")
                     for ch in ("0", "1"))
        bc = e1[(e1["session"] == s) & (e1["cls"] == "BC")]
        entry = {
            "x2_frames": os.path.relpath(fp, ANNOT),
            "n_bc_windows": int(len(bc)),
            "soulx": s in soulx_sessions,
            "transcript": has_tr,
            "bc_realized": bc["realized"].value_counts().to_dict()
            if len(bc) else {},
        }
        index[s] = entry
        n_tot["sessions"] += 1
        n_tot["bc_windows"] += entry["n_bc_windows"]
        n_tot["with_soulx"] += int(entry["soulx"])
        n_tot["with_x2"] += 1
        n_tot["with_transcript"] += int(entry["transcript"])

    summary = {
        "version": "v0",
        "created": "2026-08-27",
        "counts": n_tot,
        "layers": {
            "AWS": "candor_events_w*.csv + candor_e1_w*.csv (L2 参考层, 全部会话)",
            "realized": "candor_e1_w*.csv 的 realized 列 (声道级双活跃裁决)",
            "X2-Turn": "e4_raw/*.json (80ms 帧级 p_bc, 100 会话)",
            "SoulX": "f2_soulx_regions/*_states.json (160ms backchannel, 30 会话, BC 邻域)",
            "fused": "mean 融合软分数 (F1 配方; SoulX 可用处)",
        },
    }
    with open(f"{V0}/index.json", "w") as f:
        json.dump(index, f, indent=1)
    with open(f"{V0}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=1), flush=True)

    readme = f"""# CANDOR-FD v0 — 真实对话全双工行为标注基准

分层标注同一批真实对话 (CANDOR 会话):

| 层 | 内容 | 覆盖 | 位置 |
|---|---|---|---|
| AWS (L2 参考) | backbiter/audiophile 派生标签 | 全部会话 | candor_events_w*.csv + candor_e1_w*.csv |
| realized (声学验证) | 声道级 RMS 双活跃裁决 | 全部事件 | candor_e1_w*.csv 的 realized 列 |
| X2-Turn (L3 主标注) | 80ms/帧 6 类状态 + p(backchannel)/p(speaking) | {n_tot['with_x2']} 会话 | e4_raw/{{session}}.json |
| SoulX (L3 补充) | 160ms/块 backchannel 状态 | {n_tot['with_soulx']} 会话 (BC 邻域区域) | f2_soulx_regions/{{key}}_states.json |
| fused (L3 融合) | mean 融合软分数 (F1 配方) | SoulX 可用处 | 用 scripts/ari_f1_fusion.py 重算 |

统计: {n_tot['sessions']} 会话, {n_tot['bc_windows']} 个 AWS BC 窗口,
{n_tot['with_transcript']} 会话含 X2-Turn 转写.

用途: 鲁棒全双工标注与评测的基准数据 (flywheel 计划 F2 产物).
"""
    with open(f"{V0}/README.md", "w") as f:
        f.write(readme)
    print("packaged ->", V0)


if __name__ == "__main__":
    main()
