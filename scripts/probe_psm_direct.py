#!/usr/bin/env python3
"""Using the winning hypothesis B (PSM poses already in camera frame),
overlay PSM tip dots on N sampled frames for visual verification.

python scripts/probe_psm_direct.py --episode F_3 --snippet 001 --num 12
python scripts/probe_psm_direct.py --episode E_3 --snippet 002 --num 12
"""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
PARQUET_DIR = Path(
    r"f:\2026 vibes\MPHY Project\CRCD_manual\hub\datasets--SITL-Eng--CRCD"
    r"\snapshots\f597d230356f4e6d46516b83c2baa4f52c923358\data"
)
SEGMENTS_DIR = BASE / "data" / "Segments"
CALIB_PATH = BASE / "cam_cali" / "cam_calib" / "ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl"


def find_parquet_for_frames(ep, wanted):
    """Return list of (parquet_path, set_of_frames_in_it) covering wanted."""
    out = []
    remaining = set(wanted)
    for pq in sorted((PARQUET_DIR / ep).glob("*.parquet")):
        fns = pd.read_parquet(pq, columns=["frame_n"])["frame_n"].astype(int).tolist()
        here = set(fns) & remaining
        if here:
            out.append((pq, here))
            remaining -= here
    return out, remaining


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--snippet", required=True)
    ap.add_argument("--num", type=int, default=10)
    args = ap.parse_args()

    with open(CALIB_PATH, "rb") as f:
        calib = pickle.load(f)
    K = calib["ecm_left_rect_K"]

    snip_dir = SEGMENTS_DIR / args.episode / f"snippet_{args.snippet}"
    left_dir = snip_dir / "frames_left"
    fns = sorted(int(p.stem.split("_")[1]) for p in left_dir.glob("frame_*.webp"))
    step = max(1, len(fns) // args.num)
    sample = fns[::step][: args.num]
    print(f"sampling {len(sample)} frames: {sample[0]}..{sample[-1]}")

    parquets, missing = find_parquet_for_frames(args.episode, set(sample))
    if missing:
        print(f"WARN: {len(missing)} sample frames not in any parquet; skipping them")

    cols = [
        "frame_n",
        "/PSM1/custom/setpoint_cp", "/PSM2/custom/setpoint_cp",
        "/PSM1/jaw/measured_js", "/PSM2/jaw/measured_js",
    ]
    rows = {}
    for pq, here in parquets:
        df = pd.read_parquet(pq, columns=cols)
        df = df[df["frame_n"].isin(here)]
        for _, r in df.iterrows():
            rows[int(r["frame_n"])] = r

    out_dir = BASE / "outputs" / "psm_probe" / f"{args.episode}_snippet_{args.snippet}_direct"
    out_dir.mkdir(parents=True, exist_ok=True)

    inside1 = inside2 = 0
    for fn in sample:
        if fn not in rows:
            continue
        r = rows[fn]
        p1 = np.asarray(r["/PSM1/custom/setpoint_cp"], dtype=float)[:3]
        p2 = np.asarray(r["/PSM2/custom/setpoint_cp"], dtype=float)[:3]
        j1 = float(np.asarray(r["/PSM1/jaw/measured_js"]).flatten()[0])
        j2 = float(np.asarray(r["/PSM2/jaw/measured_js"]).flatten()[0])

        img = cv2.imread(str(left_dir / f"frame_{fn:06d}.webp"))
        h, w = img.shape[:2]

        def proj_and_draw(p, color, label, jaw):
            if p[2] <= 0:
                return False
            uv = K @ (p / p[2])
            x, y = int(round(uv[0])), int(round(uv[1]))
            inside = 0 <= x < w and 0 <= y < h
            col = color if inside else (120, 120, 120)
            cv2.circle(img, (x, y), 10, col, 2)
            cv2.circle(img, (x, y), 3, col, -1)
            cv2.putText(img, f"{label} j={jaw:+.2f}", (x + 12, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            return inside

        in1 = proj_and_draw(p1, (0, 255, 0), "PSM1", j1)
        in2 = proj_and_draw(p2, (0, 128, 255), "PSM2", j2)
        inside1 += int(in1)
        inside2 += int(in2)

        cv2.putText(img, f"frame {fn}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(out_dir / f"direct_{fn:06d}.jpg"), img)

    print(f"PSM1 inside image: {inside1}/{len(sample)}")
    print(f"PSM2 inside image: {inside2}/{len(sample)}")
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
