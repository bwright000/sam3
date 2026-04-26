#!/usr/bin/env python3
"""Try several frame-convention hypotheses for projecting PSM tips onto the left image.

Saves one annotated image per hypothesis for visual inspection.
Hypotheses tried:
  A. PSM/ECM both in common world; cam = ECM; OpenCV axes
  B. PSM expressed directly in ECM tip frame (no inversion); OpenCV axes
  C. Hypothesis A with camera-frame rotation: flip y,z (dVRK z-forward -> OpenCV z-forward,
     but y flipped)
  D. Hypothesis A with 180° around z (flip x,y)
  E. Hypothesis B with y,z flip (dVRK tip frame to OpenCV)
  F. Hypothesis B with 180° around x
  G. Hypothesis B with permutation (x, y, z) -> (z, -x, -y)   [common ROS->OpenCV]
  H. Hypothesis B with (x,y,z) -> (-y, -z, x)
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
    qx, qy, qz, qw = q
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def pose_to_SE3(pose):
    T = np.eye(4)
    T[:3, :3] = quat_to_rotmat(pose[3:7])
    T[:3, 3] = pose[:3]
    return T


def project(p_cam, K):
    if p_cam[2] <= 0:
        return None
    uv = K @ (p_cam / p_cam[2])
    return (float(uv[0]), float(uv[1]))


def get_p_cam(psm_pose, ecm_pose, hypothesis):
    """Return 3D point in (hypothesized) camera frame."""
    p_psm = np.asarray(psm_pose[:3], dtype=float)
    p_ecm = np.asarray(ecm_pose[:3], dtype=float)

    if hypothesis == "A":
        T_we = pose_to_SE3(ecm_pose)
        p_cam = (np.linalg.inv(T_we) @ np.append(p_psm, 1.0))[:3]
    elif hypothesis == "B":
        p_cam = p_psm.copy()
    elif hypothesis == "C":
        T_we = pose_to_SE3(ecm_pose)
        p = (np.linalg.inv(T_we) @ np.append(p_psm, 1.0))[:3]
        p_cam = np.array([p[0], -p[1], -p[2]])
    elif hypothesis == "D":
        T_we = pose_to_SE3(ecm_pose)
        p = (np.linalg.inv(T_we) @ np.append(p_psm, 1.0))[:3]
        p_cam = np.array([-p[0], -p[1], p[2]])
    elif hypothesis == "E":
        p_cam = np.array([p_psm[0], -p_psm[1], -p_psm[2]])
    elif hypothesis == "F":
        p_cam = np.array([p_psm[0], -p_psm[1], p_psm[2]])  # 180 around x but z stays
        # correct 180 around x: (x, -y, -z) — that's E. use (x, y, -z) instead:
        p_cam = np.array([p_psm[0], p_psm[1], -p_psm[2]])
    elif hypothesis == "G":
        # ROS->OpenCV: (x,y,z)_ros -> (-y, -z, x)_cv (camera optical frame)
        p_cam = np.array([-p_psm[1], -p_psm[2], p_psm[0]])
    elif hypothesis == "H":
        # reverse axis permutation
        p_cam = np.array([p_psm[1], p_psm[2], p_psm[0]])
    else:
        raise ValueError(hypothesis)
    return p_cam


HYPOTHESES = ["A", "B", "C", "D", "E", "F", "G", "H"]


def find_parquet_containing(ep, wanted):
    for pq in sorted((PARQUET_DIR / ep).glob("*.parquet")):
        fns = pd.read_parquet(pq, columns=["frame_n"])["frame_n"].astype(int)
        if set(fns.tolist()) & wanted:
            return pq
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--snippet", required=True)
    ap.add_argument("--frame", type=int, default=None,
                    help="specific frame number; default = middle of snippet")
    args = ap.parse_args()

    with open(CALIB_PATH, "rb") as f:
        calib = pickle.load(f)
    K = calib["ecm_left_rect_K"]

    snip_dir = SEGMENTS_DIR / args.episode / f"snippet_{args.snippet}"
    left_dir = snip_dir / "frames_left"
    fns = sorted(int(p.stem.split("_")[1]) for p in left_dir.glob("frame_*.webp"))
    fn = args.frame if args.frame is not None else fns[len(fns) // 2]
    print(f"using frame {fn}")

    pq = find_parquet_containing(args.episode, {fn})
    if pq is None:
        raise SystemExit("frame not in any parquet")
    df = pd.read_parquet(
        pq,
        columns=["frame_n", "/ECM/custom/setpoint_cp",
                 "/PSM1/custom/setpoint_cp", "/PSM2/custom/setpoint_cp"],
    )
    row = df[df["frame_n"] == fn].iloc[0]
    ecm = np.asarray(row["/ECM/custom/setpoint_cp"], dtype=float)
    psm1 = np.asarray(row["/PSM1/custom/setpoint_cp"], dtype=float)
    psm2 = np.asarray(row["/PSM2/custom/setpoint_cp"], dtype=float)

    print(f"ecm :  pos={ecm[:3]}")
    print(f"PSM1:  pos={psm1[:3]}")
    print(f"PSM2:  pos={psm2[:3]}")

    img_base = cv2.imread(str(left_dir / f"frame_{fn:06d}.webp"))
    if img_base is None:
        raise SystemExit("image not found")

    out_dir = BASE / "outputs" / "psm_probe" / f"{args.episode}_snippet_{args.snippet}_conv"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for h in HYPOTHESES:
        img = img_base.copy()
        p1 = get_p_cam(psm1, ecm, h)
        p2 = get_p_cam(psm2, ecm, h)
        uv1 = project(p1, K)
        uv2 = project(p2, K)

        def draw(uv, color, label, z):
            if uv is None:
                return False
            x, y = int(round(uv[0])), int(round(uv[1]))
            inside = 0 <= x < img.shape[1] and 0 <= y < img.shape[0]
            col = color if inside else (100, 100, 100)
            cv2.circle(img, (x, y), 12, col, 3)
            cv2.circle(img, (x, y), 3, col, -1)
            cv2.putText(img, f"{label} z={z:.2f}", (x + 14, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            return inside

        cv2.putText(img, f"hypothesis {h}  frame {fn}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        in1 = draw(uv1, (0, 255, 0), "PSM1", float(p1[2]))
        in2 = draw(uv2, (0, 128, 255), "PSM2", float(p2[2]))

        out_path = out_dir / f"hyp_{h}_frame_{fn:06d}.jpg"
        cv2.imwrite(str(out_path), img)
        summary.append(
            (h, uv1, p1[2], in1, uv2, p2[2], in2, out_path.name)
        )

    print("\nhyp | PSM1 (u,v,z)                     inside | PSM2 (u,v,z)                     inside | file")
    for (h, uv1, z1, i1, uv2, z2, i2, name) in summary:
        def fmt(uv, z):
            if uv is None:
                return "None                 "
            return f"({uv[0]:6.1f},{uv[1]:6.1f}) z={z:5.3f}"
        print(f"  {h} | {fmt(uv1,z1)}  {str(i1):5} | {fmt(uv2,z2)}  {str(i2):5} | {name}")
    print(f"\nOutputs: {out_dir}")


if __name__ == "__main__":
    main()
