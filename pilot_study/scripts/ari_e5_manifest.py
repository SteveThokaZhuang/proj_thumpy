"""E5 清单: 区域文件名 -> (session, ch, t0, t1) 时间映射."""
import glob
import json
import os
import sys

import pandas as pd
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ari_analyze import DEFAULT_OUT  # noqa: E402

CANDOR = "/share/workspace3/shared_dataset/CANDOR/files"
MARGIN, MIN_GAP = 5.0, 2.0


def main():
    n_sessions = int(os.environ.get("E5_N_SESSIONS", "5"))
    ev = pd.concat([pd.read_csv(f, keep_default_na=False)
                    for f in sorted(glob.glob(f"{DEFAULT_OUT}/candor_events_w*.csv"))])
    sessions = sorted(ev["session"].unique())[:n_sessions]
    mani = {}
    for s in sessions:
        sub = ev[(ev["session"] == s) & (ev["cls"] == "BC")]
        sfh = sf.SoundFile(f"{CANDOR}/{s}/processed/{s}.mp3")
        total = sfh.frames / sfh.samplerate
        sfh.close()
        for ch in (0, 1):
            wins = sorted(zip(sub[sub["ch_event"] == ch]["start"],
                              sub[sub["ch_event"] == ch]["end"]))
            regions = []
            for (a, b) in wins:
                a0, b0 = max(0, a - MARGIN), min(total, b + MARGIN)
                if regions and a0 - regions[-1][1] < MIN_GAP:
                    regions[-1] = (regions[-1][0], b0)
                else:
                    regions.append((a0, b0))
            for ri, (a0, b0) in enumerate(regions):
                mani[f"{s[:8]}_ch{ch}_r{ri}"] = {
                    "session": s, "ch": ch, "t0": round(a0, 3), "t1": round(b0, 3)}
    out = os.path.join(os.path.dirname(DEFAULT_OUT.rsplit("/", 1)[0]),
                       "annotator", "e5_regions_manifest.json")
    with open(out, "w") as f:
        json.dump(mani, f, indent=1)
    print("manifest entries:", len(mani), "->", out)


if __name__ == "__main__":
    main()
