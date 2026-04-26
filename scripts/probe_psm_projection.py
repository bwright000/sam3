#!/usr/bin/env python3
"""Probe: project PSM1/PSM2 tip 3D kinematics onto left camera image.

Goal: verify whether /PSM{1,2}/custom/setpoint_cp positions expressed in the
'custom' frame can be projected to pixel coordinates using /ECM/custom/setpoint_cp
and the rectified left-camera intrinsics (cam_cali/cam_calib/*.pkl).

If the projected dots land on the visible tool tips, we can use this as automatic
SAM3 point prompts with no manual annotation.

Usage:
    python scripts/probe_psm_projection.py --episode F_3 --snippet 001 --num-samples 6
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


def quat_to_rotmat(q):
    """q = (qx, qy, qz, qw) -> 3x3 R."""
    qx, qy, qz, qw = q
    # normalize
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def pose_to_SE3(pose):
    """pose = (tx, ty, tz, qx, qy, qz, qw) -> 4x4 SE3."""
    t = np.asarray(pose[:3], dtype=float)
    R = quat_to_rotmat(pose[3:7])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def project_point(p_world, T_world_cam, K):
    """Project a 3D world point through camera pose (T_world_cam: camera in world)."""
    T_cam_world = np.linalg.inv(T_world_cam)
    p_h = np.array([p_world[0], p_world[1], p_world[2], 1.0])
    p_cam = T_cam_world @ p_h
    if p_cam[2] <= 0:
        return None, p_cam[2]
    p_img = K @ (p_cam[:3] / p_cam[2])
    return (float(p_img[0]), float(p_img[1])), float(p_cam[2])


def find_parquet_containing(ep, frames_wanted):
    """Scan episode parquets for one that contains the target frames."""
    for pq in sorted((PARQUET_DIR / ep).glob("*.parquet")):
        df = pd.read_parquet(pq, columns=["frame_n"])
        fns = set(df["frame_n"].astype(int).tolist())
        if fns & frames_wanted:
            return pq
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--snippet", required=True)
    ap.add_argument("--num-samples", type=int, default=6)
    args = ap.parse_args()

    # Load calibration
    with open(CALIB_PATH, "rb") as f:
        calib = pickle.load(f)
    K = calib["ecm_left_rect_K"]
    print("Left rect K:")
    print(K)

    # Load snippet frame range
    snip_dir = SEGMENTS_DIR / args.episode / f"snippet_{args.snippet}"
    left_dir = snip_dir / "frames_left"
    frame_ns = sorted(int(p.stem.split("_")[1]) for p in left_dir.glob("frame_*.webp"))
    if not frame_ns:
        raise SystemExit(f"no frames in {left_dir}")
    print(f"snippet range: {frame_ns[0]}..{frame_ns[-1]}  ({len(frame_ns)} frames)")

    step = max(1, len(frame_ns) // args.num_samples)
    sample_fns = frame_ns[::step][: args.num_samples]
    print(f"sampling frames: {sample_fns}")

    # Find parquet containing these frames (may span multiple; simple: use first)
    pq = find_parquet_containing(args.episode, set(sample_fns))
    if pq is None:
        raise SystemExit("no parquet contains any sampled frame")
    print(f"reading {pq.name}")

    cols = [
        "frame_n",
        "/ECM/custom/setpoint_cp",
        "/PSM1/custom/setpoint_cp",
        "/PSM2/custom/setpoint_cp",
        "/PSM1/jaw/measured_js",
        "/PSM2/jaw/measured_js",
    ]
    df = pd.read_parquet(pq, columns=cols)
    df = df[df["frame_n"].isin(sample_fns)].set_index("frame_n")

    out_dir = BASE / "outputs" / "psm_probe" / f"{args.episode}_snippet_{args.snippet}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fn in sample_fns:
        if fn not in df.index:
            print(f"  frame {fn}: not in parquet slice; skip")
            continue
        row = df.loc[fn]
        ecm = np.asarray(row["/ECM/custom/setpoint_cp"], dtype=float)
        psm1 = np.asarray(row["/PSM1/custom/setpoint_cp"], dtype=float)
        psm2 = np.asarray(row["/PSM2/custom/setpoint_cp"], dtype=float)
        jaw1 = float(np.asarray(row["/PSM1/jaw/measured_js"]).flatten()[0])
        jaw2 = float(np.asarray(row["/PSM2/jaw/measured_js"]).flatten()[0])

        T_world_ecm = pose_to_SE3(ecm)

        # Project each PSM tip
        pix1, z1 = project_point(psm1[:3], T_world_ecm, K)
        pix2, z2 = project_point(psm2[:3], T_world_ecm, K)

        img_path = left_dir / f"frame_{fn:06d}.webp"
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  frame {fn}: image missing; skip")
            continue
        h, w = img.shape[:2]

        def draw_pt(p, color, label):
            if p is None:
                return
            x, y = int(round(p[0])), int(round(p[1]))
            inside = 0 <= x < w and 0 <= y < h
            col = color if inside else (120, 120, 120)
            cv2.circle(img, (x, y), 10, col, 2)
            cv2.circle(img, (x, y), 3, col, -1)
            cv2.putText(img, label, (x + 12, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        draw_pt(pix1, (0, 255, 0), f"PSM1 jaw={jaw1:+.2f}")
        draw_pt(pix2, (0, 128, 255), f"PSM2 jaw={jaw2:+.2f}")

        # Annotate image
        meta = f"frame {fn}  z1={z1:.3f} z2={z2:.3f}"
        cv2.putText(img, meta, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out_path = out_dir / f"probe_{fn:06d}.jpg"
        cv2.imwrite(str(out_path), img)
        print(
            f"  frame {fn}: PSM1 pix={pix1} z={z1:.3f} | PSM2 pix={pix2} z={z2:.3f} | "
            f"-> {out_path.name}"
        )

    print(f"\nWrote {len(sample_fns)} probe images to {out_dir}")


if __name__ == "__main__":
    main()
