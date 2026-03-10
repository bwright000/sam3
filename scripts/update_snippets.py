"""
Update snippet boundaries to be GT-bounded and populate with annotations.

For each episode (C_1, E_3, F_3), this script:
  1. Auto-detects split_size from the annotation JSONs
  2. Finds which splits are annotated (have GT at offset 0)
  3. Expands each snippet's start/end to the nearest annotated GT keyframe
     so that the first and last frames are GT frames
  4. Updates the *_snippets.json metadata
  5. Copies the relevant annotation subset (images + annotations) into
     each snippet directory as snippet_annotations.json
  6. Optionally extracts missing frame images from the source video

Directory layout expected:
    data/
      {EP}/{EP}/           <- downloaded annotation package
        split_imgs/        <- frame images per split
        train.json
        test.json
      Segments/{EP}/       <- working snippet directories
        {EP}_snippets.json
        snippet_001/
          frames_left/
        ...

Split sizes (auto-detected):
    C_1: 120 frames/split
    E_3: 120 frames/split
    F_3: 100 frames/split

Usage:
    # Dry run (show proposed changes)
    python scripts/update_snippets.py --episode F_3 --dry-run

    # Apply changes (update JSON + copy annotations)
    python scripts/update_snippets.py --episode F_3

    # All episodes
    python scripts/update_snippets.py --episode C_1 E_3 F_3

    # With frame extraction from video
    python scripts/update_snippets.py --episode F_3 \\
        --video-path "path/to/left.mp4"

    # Validate only (check existing snippets)
    python scripts/update_snippets.py --episode F_3 --validate
"""

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def detect_split_size(ann_data):
    """Auto-detect split size from max filename offset + 1."""
    max_offset = 0
    for img in ann_data["images"]:
        m = re.search(r"/(\d+)\.jpg$", img["file_name"])
        if m:
            max_offset = max(max_offset, int(m.group(1)))
    return max_offset + 1 if max_offset > 0 else 120


def build_annotated_splits(ann_data, split_size):
    """Return set of split numbers that have annotations."""
    splits = set()
    for img in ann_data["images"]:
        m = re.search(r"split_(\d+)/", img["file_name"])
        if m:
            splits.add(int(m.group(1)))
    return splits


def build_frame_index(ann_data, split_size):
    """Build video_frame_number -> list of image_ids mapping.

    Also handles C_1 train.json broken filenames where file_name is
    like './split_14/' (no offset). In that case, recover offset from
    image_id = split_num * split_size + offset.
    """
    frame_to_ids = defaultdict(list)
    for img in ann_data["images"]:
        fname = img["file_name"]
        m = re.search(r"split_(\d+)/(\d+)\.jpg", fname)
        if m:
            split_num = int(m.group(1))
            offset = int(m.group(2))
        else:
            # C_1 broken format: './split_14/'
            m2 = re.search(r"split_(\d+)/?$", fname.rstrip("/"))
            if m2:
                split_num = int(m2.group(1))
                img_id = img["id"]
                if img_id // split_size == split_num:
                    offset = img_id % split_size
                else:
                    continue
            else:
                continue
        video_frame = split_num * split_size + offset
        frame_to_ids[video_frame].append(img["id"])
    return dict(frame_to_ids)


def load_merged_annotations(ann_dir):
    """Load and merge train.json + test.json into a single dataset.

    Returns merged dict with deduplicated categories, merged images,
    and merged annotations.
    """
    merged = {"images": [], "annotations": [], "categories": []}
    seen_img_ids = set()
    seen_ann_ids = set()

    for src in ["train.json", "test.json"]:
        path = ann_dir / src
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            data = json.load(f)

        # Categories (take from first file)
        if not merged["categories"]:
            merged["categories"] = data.get("categories", [])

        for img in data.get("images", []):
            if img["id"] not in seen_img_ids:
                merged["images"].append(img)
                seen_img_ids.add(img["id"])

        for ann in data.get("annotations", []):
            if ann["id"] not in seen_ann_ids:
                merged["annotations"].append(ann)
                seen_ann_ids.add(ann["id"])

    return merged


# ---------------------------------------------------------------------------
# GT-bounded boundary computation
# ---------------------------------------------------------------------------

def compute_gt_bounded_range(motion_start, motion_end, split_size, annotated_splits):
    """Compute snippet boundaries that start and end on annotated GT frames.

    The start boundary is the nearest annotated split AT or BEFORE motion_start.
    The end boundary is the nearest annotated split AT or AFTER motion_end.
    The actual boundary frames are: start = split_num * split_size,
    end = split_num * split_size + (split_size - 1) (full split included).

    Returns (new_start_frame, new_end_frame, start_split, end_split) or
    None if the motion range has no nearby annotated splits (within 5 splits).
    """
    MAX_SEARCH = 5  # max splits to search before giving up

    # Find start split: nearest annotated split whose GT frame <= motion_start
    start_split = motion_start // split_size
    search = 0
    while start_split >= 0 and start_split not in annotated_splits and search < MAX_SEARCH:
        start_split -= 1
        search += 1
    if start_split < 0 or start_split not in annotated_splits:
        # Try forward
        start_split = motion_start // split_size
        search = 0
        while start_split not in annotated_splits and search < MAX_SEARCH:
            start_split += 1
            search += 1
        if start_split not in annotated_splits:
            return None  # no nearby annotated split

    # Find end split: nearest annotated split whose GT frame >= motion_end
    end_split_candidate = motion_end // split_size
    if end_split_candidate * split_size < motion_end:
        end_split_candidate += 1
    end_split = end_split_candidate
    search = 0
    max_split = max(annotated_splits) if annotated_splits else end_split + MAX_SEARCH
    while end_split <= max_split and end_split not in annotated_splits and search < MAX_SEARCH:
        end_split += 1
        search += 1
    if end_split not in annotated_splits:
        return None  # no nearby annotated split

    new_start = start_split * split_size
    new_end = end_split * split_size + (split_size - 1)  # include full end split

    return new_start, new_end, start_split, end_split


# ---------------------------------------------------------------------------
# Annotation subset extraction
# ---------------------------------------------------------------------------

def extract_snippet_annotations(merged_ann, split_size, start_frame, end_frame):
    """Extract the annotation subset covering [start_frame, end_frame].

    Returns a COCO-format dict with only the images and annotations
    within the frame range.
    """
    frame_index = build_frame_index(merged_ann, split_size)

    # Collect image IDs in range
    target_img_ids = set()
    for frame_num in range(start_frame, end_frame + 1):
        for img_id in frame_index.get(frame_num, []):
            target_img_ids.add(img_id)

    # Filter images
    img_id_set = target_img_ids
    filtered_images = [img for img in merged_ann["images"] if img["id"] in img_id_set]
    filtered_anns = [ann for ann in merged_ann["annotations"] if ann["image_id"] in img_id_set]

    return {
        "categories": merged_ann["categories"],
        "images": filtered_images,
        "annotations": filtered_anns,
        "metadata": {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "split_size": split_size,
            "num_images": len(filtered_images),
            "num_annotations": len(filtered_anns),
        },
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_snippet(snippet_meta, snippet_dir, split_size, annotated_splits, merged_ann):
    """Validate a snippet's boundaries, frames, and annotations.

    Returns (passed: bool, issues: list of strings).
    """
    issues = []
    sid = snippet_meta["snippet_id"]
    sf = snippet_meta["start_frame"]
    ef = snippet_meta["end_frame"]
    msf = snippet_meta["motion_start_frame"]
    mef = snippet_meta["motion_end_frame"]

    # Check 1: start_frame is a GT frame
    start_split = sf // split_size
    if sf % split_size != 0:
        issues.append(f"start_frame {sf} is not a GT keyframe (offset {sf % split_size})")
    elif start_split not in annotated_splits:
        issues.append(f"start_frame {sf} is in unannotated split_{start_split}")

    # Check 2: end_frame's split is annotated and end = split*size + (size-1)
    end_split = ef // split_size
    expected_end = end_split * split_size + (split_size - 1)
    if ef != expected_end:
        issues.append(f"end_frame {ef} doesn't include full split (expected {expected_end})")
    if end_split not in annotated_splits:
        issues.append(f"end_frame {ef} is in unannotated split_{end_split}")

    # Check 3: motion range is within snippet bounds
    if msf < sf or mef > ef:
        issues.append(f"motion range {msf}-{mef} extends outside snippet {sf}-{ef}")

    # Check 4: frames_left directory has correct range
    frames_dir = snippet_dir / "frames_left"
    if frames_dir.exists():
        frame_files = sorted(frames_dir.glob("frame_*.webp")) or \
                      sorted(frames_dir.glob("frame_*.png")) or \
                      sorted(frames_dir.glob("frame_*.jpg"))
        if frame_files:
            first_num = int(re.search(r"(\d+)", frame_files[0].stem).group(1))
            last_num = int(re.search(r"(\d+)", frame_files[-1].stem).group(1))
            if first_num != sf:
                issues.append(f"frames start at {first_num}, expected {sf}")
            if last_num != ef:
                issues.append(f"frames end at {last_num}, expected {ef}")
        else:
            issues.append("no frame files found in frames_left/")
    else:
        issues.append("frames_left/ directory missing")

    # Check 5: snippet_annotations.json exists and covers range
    ann_path = snippet_dir / "snippet_annotations.json"
    if ann_path.exists():
        with open(ann_path) as f:
            snip_ann = json.load(f)
        meta = snip_ann.get("metadata", {})
        if meta.get("start_frame") != sf:
            issues.append(f"annotation start_frame mismatch: {meta.get('start_frame')} vs {sf}")
        if meta.get("end_frame") != ef:
            issues.append(f"annotation end_frame mismatch: {meta.get('end_frame')} vs {ef}")
    else:
        issues.append("snippet_annotations.json missing")

    return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def process_episode(episode, data_dir, segments_dir, dry_run=False,
                    validate_only=False, video_path=None):
    """Process a single episode: update snippet boundaries and annotations."""
    print(f"\n{'='*70}")
    print(f"  Episode: {episode}")
    print(f"{'='*70}")

    ann_dir = data_dir / episode / episode
    ep_dir = segments_dir / episode
    snippets_path = ep_dir / f"{episode}_snippets.json"

    # Validate paths
    if not ann_dir.exists():
        print(f"  ERROR: Annotation directory not found: {ann_dir}")
        return False
    if not snippets_path.exists():
        print(f"  ERROR: Snippets file not found: {snippets_path}")
        return False

    # Load annotations (memory-conscious: load once per episode)
    print(f"  Loading annotations from {ann_dir}...")
    merged_ann = load_merged_annotations(ann_dir)
    total_imgs = len(merged_ann["images"])
    total_anns = len(merged_ann["annotations"])
    print(f"  Loaded {total_imgs} images, {total_anns} annotations")

    # Detect split size
    split_size = detect_split_size(merged_ann)
    print(f"  Split size: {split_size}")

    # Build annotated splits set
    annotated_splits = build_annotated_splits(merged_ann, split_size)
    print(f"  Annotated splits: {len(annotated_splits)}")

    # Load snippets
    with open(snippets_path) as f:
        snippets = json.load(f)
    print(f"  Snippets: {len(snippets)}")

    all_passed = True
    updated_snippets = []

    for snippet in snippets:
        sid = snippet["snippet_id"]
        msf = snippet["motion_start_frame"]
        mef = snippet["motion_end_frame"]
        old_sf = snippet["start_frame"]
        old_ef = snippet["end_frame"]
        snippet_dir = ep_dir / f"snippet_{sid}"

        print(f"\n  Snippet {sid}:")
        print(f"    Motion:     {msf} - {mef}")
        print(f"    Old bounds: {old_sf} - {old_ef} ({old_ef - old_sf + 1} frames)")

        # Compute GT-bounded range
        result = compute_gt_bounded_range(
            msf, mef, split_size, annotated_splits
        )
        if result is None:
            print(f"    WARNING: No annotated splits near motion range {msf}-{mef}")
            print(f"    SKIPPING (snippet outside annotation coverage)")
            updated_snippets.append(snippet)
            continue

        new_sf, new_ef, start_split, end_split = result
        new_num = new_ef - new_sf + 1

        changed = new_sf != old_sf or new_ef != old_ef
        marker = " *** CHANGED" if changed else ""
        print(f"    New bounds: {new_sf} - {new_ef} ({new_num} frames){marker}")
        print(f"    GT splits:  {start_split} - {end_split}")

        # Count GT keyframes in new range
        gt_frames = [f for f in range(new_sf, new_ef + 1)
                     if f % split_size == 0 and f // split_size in annotated_splits]
        print(f"    GT frames:  {len(gt_frames)} keyframes")

        if validate_only:
            # Run validation
            passed, issues = validate_snippet(
                {**snippet, "start_frame": new_sf, "end_frame": new_ef},
                snippet_dir, split_size, annotated_splits, merged_ann
            )
            if passed:
                print(f"    VALIDATION: PASSED")
            else:
                print(f"    VALIDATION: FAILED")
                for issue in issues:
                    print(f"      - {issue}")
                all_passed = False
            updated_snippets.append(snippet)
            continue

        # Update snippet metadata
        updated = dict(snippet)
        updated["start_frame"] = new_sf
        updated["end_frame"] = new_ef
        updated["num_frames"] = new_num
        # Recompute padding (frames outside motion range)
        motion_frames = mef - msf + 1
        updated["motion_frames"] = motion_frames
        updated["padding_frames"] = new_num - motion_frames
        updated["duration_s"] = new_num / 60.0  # 60 fps
        updated["split_size"] = split_size
        updated["gt_start_split"] = start_split
        updated["gt_end_split"] = end_split
        updated_snippets.append(updated)

        if dry_run:
            print(f"    [DRY RUN] Would update metadata and extract annotations")
            continue

        # Extract annotation subset for this snippet
        print(f"    Extracting annotation subset...")
        snip_ann = extract_snippet_annotations(merged_ann, split_size, new_sf, new_ef)
        ann_out = snippet_dir / "snippet_annotations.json"
        snippet_dir.mkdir(parents=True, exist_ok=True)
        with open(ann_out, "w") as f:
            json.dump(snip_ann, f)
        print(f"    Saved {snip_ann['metadata']['num_images']} images, "
              f"{snip_ann['metadata']['num_annotations']} annotations "
              f"-> {ann_out.name}")

        # Extract frames from video if provided and needed
        if video_path:
            _extract_missing_frames(snippet_dir, new_sf, new_ef, video_path)

    # Write updated snippets JSON
    if not validate_only and not dry_run:
        # Backup original
        backup_path = snippets_path.with_suffix(".json.bak")
        if not backup_path.exists():
            shutil.copy2(snippets_path, backup_path)
            print(f"\n  Backed up original to {backup_path.name}")

        with open(snippets_path, "w") as f:
            json.dump(updated_snippets, f, indent=2)
        print(f"  Updated {snippets_path.name}")
    elif dry_run:
        print(f"\n  [DRY RUN] Would update {snippets_path.name}")

    # Free memory
    del merged_ann

    if validate_only:
        status = "ALL PASSED" if all_passed else "SOME FAILED"
        print(f"\n  Validation: {status}")

    return all_passed


def _extract_missing_frames(snippet_dir, start_frame, end_frame, video_path):
    """Extract any frames in [start_frame, end_frame] not yet in frames_left/."""
    try:
        import cv2
    except ImportError:
        print("    WARNING: cv2 not available, skipping frame extraction")
        return

    frames_dir = snippet_dir / "frames_left"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Find existing frames
    existing = set()
    for pat in ["frame_*.webp", "frame_*.png", "frame_*.jpg"]:
        for f in frames_dir.glob(pat):
            m = re.search(r"(\d+)", f.stem)
            if m:
                existing.add(int(m.group(1)))

    needed = [fn for fn in range(start_frame, end_frame + 1) if fn not in existing]
    if not needed:
        print(f"    All {end_frame - start_frame + 1} frames present")
        return

    print(f"    Extracting {len(needed)} missing frames from video...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    ERROR: Cannot open video {video_path}")
        return

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted = 0
    for fn in needed:
        if fn < 0 or fn >= total_video_frames:
            print(f"    WARNING: Frame {fn} out of video range (0-{total_video_frames - 1})")
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            continue
        out_path = frames_dir / f"frame_{fn:06d}.webp"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_WEBP_QUALITY, 95])
        extracted += 1

    cap.release()
    print(f"    Extracted {extracted} frames")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Update snippet boundaries to GT-bounded and populate annotations"
    )
    parser.add_argument(
        "--episode", nargs="+", required=True,
        help="Episode(s) to process (e.g., C_1 E_3 F_3)"
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Path to data/ directory (default: data)"
    )
    parser.add_argument(
        "--segments-dir", default="data/Segments",
        help="Path to Segments/ directory (default: data/Segments)"
    )
    parser.add_argument(
        "--video-path", default=None,
        help="Path to full episode video for frame extraction (optional)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show proposed changes without applying"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate existing snippets instead of updating"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    segments_dir = Path(args.segments_dir)

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    all_ok = True
    for episode in args.episode:
        ok = process_episode(
            episode, data_dir, segments_dir,
            dry_run=args.dry_run,
            validate_only=args.validate,
            video_path=Path(args.video_path) if args.video_path else None,
        )
        if not ok:
            all_ok = False

    if args.validate:
        sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
