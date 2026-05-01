"""
Convert an existing CRCD snippet at data/Segments/{Episode}/snippet_NNN/ from the legacy
"frames_left/ + frames_right/ + poses.txt + snippet_annotations.json + video_stereo.mp4"
layout to the TUM-RGB-D-with-Replica-style-segmentation layout.

Target layout (per plan in C:/Users/benli/.claude/plans/audit-shows-all-130-shimmering-rain.md):

    {snippet_dir}/
    ├── rgb/frame_NNNNNN.png          PNG, 8-bit, 1280x720 (re-encoded from webp)
    ├── rgbright/frame_NNNNNN.png     same, right camera
    ├── depth/                        empty placeholder (depth model fills later)
    ├── semantic_instance/            16-bit PNG per frame, pixel = category_id+1, 0=bg
    │   └── frame_NNNNNN.png
    ├── rgb.txt / rgbright.txt / depth.txt   TUM index files (timestamp filename)
    ├── associations.txt              TUM RGB-depth pairing (1:1 by frame_n)
    ├── groundtruth.txt               renamed from poses.txt (TUM quaternion, unchanged)
    ├── intrinsics.yaml               camera + depth_scale + dims (intrinsics TBD)
    ├── info_semantic.json            instance ID -> category name mapping
    └── (preserved: scene_motion.json, video_left.mp4, velocity.png, visualization.html,
         overlays/, snippet_NNN_overlay.mp4, snippet_NNN_results.json, snippet_annotations.json)

Operates in place. Old artefacts (frames_left/, frames_right/, video_stereo.mp4, poses.txt,
poses.txt.bak) are kept until --cleanup is passed.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_poses(poses_path):
    """Return list of (timestamp_str, full_line) tuples, one per pose row.
    Strips comment lines but preserves the original line content for re-emit."""
    poses = []
    with open(poses_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            ts = stripped.split()[0]
            poses.append((ts, line.rstrip('\n')))
    return poses


def list_frame_ns(frames_dir, ext='webp'):
    """Return sorted list of frame numbers from frame_NNNNNN.{ext} files."""
    files = sorted(frames_dir.glob(f'frame_*.{ext}'))
    return [int(f.stem.replace('frame_', '')) for f in files]


def reencode_webp_to_png(src_dir, dst_dir):
    """Read all frame_*.webp files in src_dir, save as PNG in dst_dir.
    Returns list of frame numbers written."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob('frame_*.webp'))
    frame_ns = []
    for f in files:
        fr = int(f.stem.replace('frame_', ''))
        img = Image.open(str(f)).convert('RGB')
        img.save(str(dst_dir / f'frame_{fr:06d}.png'), format='PNG', compress_level=3)
        frame_ns.append(fr)
    return frame_ns


def write_index_file(out_path, header_comment, timestamps_by_frame, folder_prefix, frame_ns):
    """Write TUM-style index: '# header' then 'timestamp folder/frame_NNNNNN.png' lines."""
    with open(out_path, 'w') as f:
        f.write(f'# {header_comment}\n')
        f.write('# timestamp filename\n')
        for fr in frame_ns:
            ts = timestamps_by_frame.get(fr)
            if ts is None:
                continue
            f.write(f'{ts} {folder_prefix}/frame_{fr:06d}.png\n')


def write_associations(out_path, timestamps_by_frame, frame_ns):
    """TUM associations.txt: 'ts_rgb rgb/... ts_depth depth/...' — same timestamp & frame_n
    since RGB and depth are 1:1 in our case."""
    with open(out_path, 'w') as f:
        f.write('# RGB-depth associations (1:1 since synthesised)\n')
        f.write('# ts_rgb rgb_path ts_depth depth_path\n')
        for fr in frame_ns:
            ts = timestamps_by_frame.get(fr)
            if ts is None:
                continue
            f.write(f'{ts} rgb/frame_{fr:06d}.png {ts} depth/frame_{fr:06d}.png\n')


def write_groundtruth(poses_lines, gt_path):
    """Re-emit poses.txt content as groundtruth.txt with TUM standard header."""
    with open(gt_path, 'w') as f:
        f.write('# ground truth trajectory\n')
        f.write('# timestamp tx ty tz qx qy qz qw\n')
        for _, line in poses_lines:
            f.write(line + '\n')


def rasterise_segmentations(annotations_json, semantic_dir, frame_ns, height=720, width=1280):
    """For each frame_n, rasterise all polygon annotations into a 16-bit PNG.
    Pixel value = category_id + 1 (so category 0 -> pixel 1, ..., 0 = background).
    If multiple annotations of same category overlap in a frame, last write wins (acceptable
    since CRCD has at most one of each category per frame in practice).

    Returns a dict mapping frame_n -> set of category_ids painted into that frame."""
    semantic_dir.mkdir(parents=True, exist_ok=True)
    by_image = {}
    for ann in annotations_json['annotations']:
        by_image.setdefault(ann['image_id'], []).append(ann)

    painted = {}
    for fr in frame_ns:
        canvas = np.zeros((height, width), dtype=np.uint16)
        cats_in_frame = set()
        for ann in by_image.get(fr, []):
            cat_id = ann['category_id']
            pixel_val = cat_id + 1
            seg = ann.get('segmentation')
            if not seg:
                continue
            for polygon in seg:
                if not polygon:
                    continue
                pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(canvas, [pts], int(pixel_val))
                cats_in_frame.add(cat_id)
        cv2.imwrite(str(semantic_dir / f'frame_{fr:06d}.png'), canvas)
        painted[fr] = cats_in_frame
    return painted


def write_info_semantic(annotations_json, info_path):
    """Replica-style info_semantic.json mapping pixel value -> category name."""
    info = {
        'classes': [
            {'id': c['id'] + 1, 'name': c['name'], 'supercategory': c['supercategory']}
            for c in annotations_json['categories']
        ],
        'background_id': 0,
        'note': 'Pixel value in semantic_instance/*.png = category_id + 1. 0 = background. CRCD has at most one of each category per frame so semantic_instance == semantic_class in practice.',
    }
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)


def write_intrinsics(snippet_dir, frame_ns, segmentation_state='present'):
    """Write intrinsics.yaml. CRCD calibration TBD; record what we know."""
    yaml_path = snippet_dir / 'intrinsics.yaml'
    yaml_str = f"""camera:
  width: 1280
  height: 720
  fps: 60
  fx: TBD             # CRCD da Vinci calibration not yet sourced
  fy: TBD
  cx: TBD
  cy: TBD
  distortion: [TBD, TBD, TBD, TBD, TBD]   # k1 k2 p1 p2 k3
depth:
  png_depth_scale: 10000   # custom: 0.1 mm precision, max 6.55 m
  units: metres
  # NOTE: differs from TUM default (5000) and Replica default (6553.5).
  # Standard TUM/Replica loaders must be configured to read this value, not assume the default.
stereo:
  baseline_m: TBD     # da Vinci stereo baseline TBD
  right_camera_relative_pose: TBD
deviations_from_tum:
  - "Frame names use sequential frame_NNNNNN.png instead of TUM Unix-timestamp filenames."
  - "rgbright/ folder added for stereo right camera (TUM is monocular)."
  - "depth/ scale = 10000 instead of TUM default 5000."
deviations_from_replica:
  - "Pose format is TUM quaternion (groundtruth.txt), not Replica 4x4 matrix (traj.txt)."
  - "Depth scale 10000 instead of Replica's 6553.5."
segmentation:
  state: {segmentation_state}    # present | partial | absent
frames:
  count: {len(frame_ns)}
  first: {frame_ns[0] if frame_ns else 'null'}
  last: {frame_ns[-1] if frame_ns else 'null'}
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_str)


def cleanup_old_artefacts(snippet_dir):
    """Delete legacy artefacts that the new layout supersedes."""
    paths_to_remove = ['frames_left', 'frames_right', 'video_stereo.mp4', 'poses.txt', 'poses.txt.bak']
    removed = []
    for name in paths_to_remove:
        p = snippet_dir / name
        if p.is_dir():
            for f in p.iterdir():
                f.unlink()
            p.rmdir()
            removed.append(name + '/')
        elif p.exists():
            p.unlink()
            removed.append(name)
    return removed


def convert_snippet(snippet_dir, cleanup=False, verbose=True):
    snippet_dir = Path(snippet_dir)
    if not snippet_dir.exists():
        raise FileNotFoundError(snippet_dir)

    poses_path = snippet_dir / 'poses.txt'
    frames_left = snippet_dir / 'frames_left'
    frames_right = snippet_dir / 'frames_right'
    annotations_path = snippet_dir / 'snippet_annotations.json'

    if not poses_path.exists():
        raise FileNotFoundError(f'poses.txt not found in {snippet_dir}')
    if not frames_left.exists():
        raise FileNotFoundError(f'frames_left/ not found in {snippet_dir}')

    poses_lines = load_poses(poses_path)
    if verbose: print(f'  {len(poses_lines)} pose rows')

    frame_ns = list_frame_ns(frames_left, 'webp')
    if verbose: print(f'  {len(frame_ns)} left webp frames ({frame_ns[0]}-{frame_ns[-1]})')

    if len(poses_lines) != len(frame_ns):
        print(f'  WARN: pose count {len(poses_lines)} != frame count {len(frame_ns)} — using min for index files')
    timestamps_by_frame = {fr: poses_lines[i][0] for i, fr in enumerate(frame_ns) if i < len(poses_lines)}

    if verbose: print(f'  re-encoding rgb/ ...')
    rgb_dir = snippet_dir / 'rgb'
    rgb_frame_ns = reencode_webp_to_png(frames_left, rgb_dir)

    if frames_right.exists():
        if verbose: print(f'  re-encoding rgbright/ ...')
        rgbright_dir = snippet_dir / 'rgbright'
        reencode_webp_to_png(frames_right, rgbright_dir)
    else:
        print(f'  WARN: frames_right/ not found — skipping right camera')

    if verbose: print(f'  writing TUM index files ...')
    write_index_file(snippet_dir / 'rgb.txt', f'RGB images for snippet {snippet_dir.name}',
                     timestamps_by_frame, 'rgb', rgb_frame_ns)
    if frames_right.exists():
        write_index_file(snippet_dir / 'rgbright.txt', f'right-camera RGB images for snippet {snippet_dir.name}',
                         timestamps_by_frame, 'rgbright', rgb_frame_ns)
    write_index_file(snippet_dir / 'depth.txt', f'depth images for snippet {snippet_dir.name}',
                     timestamps_by_frame, 'depth', rgb_frame_ns)
    write_associations(snippet_dir / 'associations.txt', timestamps_by_frame, rgb_frame_ns)

    if verbose: print(f'  writing groundtruth.txt ...')
    write_groundtruth(poses_lines, snippet_dir / 'groundtruth.txt')

    (snippet_dir / 'depth').mkdir(exist_ok=True)
    (snippet_dir / 'depth' / '.gitkeep').touch()

    if annotations_path.exists():
        if verbose: print(f'  rasterising segmentations from snippet_annotations.json ...')
        ann = json.load(open(annotations_path))
        painted = rasterise_segmentations(ann, snippet_dir / 'semantic_instance', rgb_frame_ns)
        n_with_anns = sum(1 for fr in rgb_frame_ns if painted.get(fr))
        if verbose: print(f'    {n_with_anns}/{len(rgb_frame_ns)} frames have annotations')
        write_info_semantic(ann, snippet_dir / 'info_semantic.json')
        segmentation_state = 'present' if n_with_anns == len(rgb_frame_ns) else 'partial'
    else:
        if verbose: print(f'  no snippet_annotations.json — writing all-zero placeholder masks for uniform layout')
        sem_dir = snippet_dir / 'semantic_instance'
        sem_dir.mkdir(exist_ok=True)
        zero_mask = np.zeros((720, 1280), dtype=np.uint16)
        for fr in rgb_frame_ns:
            cv2.imwrite(str(sem_dir / f'frame_{fr:06d}.png'), zero_mask)
        with open(snippet_dir / 'info_semantic.json', 'w') as f:
            json.dump({
                'classes': [],
                'background_id': 0,
                'note': 'No CRCD GT annotations available for this snippet (frames outside annotated split range). All semantic_instance/*.png are placeholder all-zero masks.',
            }, f, indent=2)
        segmentation_state = 'absent'


    write_intrinsics(snippet_dir, rgb_frame_ns, segmentation_state=segmentation_state)

    if cleanup:
        removed = cleanup_old_artefacts(snippet_dir)
        if verbose: print(f'  cleanup: removed {removed}')

    return {
        'snippet': snippet_dir.name,
        'n_frames': len(rgb_frame_ns),
        'first_frame': rgb_frame_ns[0] if rgb_frame_ns else None,
        'last_frame': rgb_frame_ns[-1] if rgb_frame_ns else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('snippet_dirs', nargs='+', help='One or more snippet directories to convert')
    p.add_argument('--cleanup', action='store_true', help='Delete old frames_left/, frames_right/, video_stereo.mp4, poses.txt after conversion')
    args = p.parse_args()

    for s in args.snippet_dirs:
        print(f'\n=== {s} ===')
        result = convert_snippet(s, cleanup=args.cleanup)
        print(f'  done: {result}')


if __name__ == '__main__':
    main()
