#!/usr/bin/env python3
"""Build 5 SLAM-criterion backfill snippets at F:/Datasets/CRCD/.

Each snippet was identified by find_slam_candidates.py as meeting the
strict criteria (>=20s span AND >=20% camera motion at the 0.1 mm/s
threshold). Builds them via the existing build_top10_snippets.build_one
function so they match the canonical TUM/Replica layout used by the rest
of the dataset.

Layout per snippet (mirrors top-10 builds):
  F:/Datasets/CRCD/{episode}/snippet_{NNN}/
    rgb/, rgbright/, depth/, semantic_instance/
    rgb.txt, rgbright.txt, depth.txt, associations.txt
    groundtruth.txt, intrinsics.yaml, scene_motion.json
    video_left.mp4

Single-segment style (no cluster_metadata.json) since each is a fixed 20s
window rather than a multi-segment cluster.

After building, updates F:/Datasets/CRCD/{episode}/{episode}_snippets.json
so the annotator's load_snippet picks them up.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.build_top10_snippets import build_one, write_episode_index, OUT_ROOT  # noqa: E402

# Strict-criterion picks from outputs/slam_candidates_strict.json. snippet_id chosen
# to extend each episode's existing index without renumbering existing snippets.
BACKFILL_SPECS = [
    # F_2: NEW episode (no existing snippets) — densest motion among strict picks (35%)
    {'episode': 'F_2', 'snippet_id': '001', 'rank': 11,
     'description': 'strict-criterion backfill: 35% motion, 30% concurrent (NEW episode)',
     'motion_start': 51060, 'motion_end': 52260,
     'sub_segments': None,
     'source_segment_ids': []},

    # G_1: NEW episode (no existing snippets)
    {'episode': 'G_1', 'snippet_id': '001', 'rank': 12,
     'description': 'strict-criterion backfill: 25% motion, 25% concurrent (NEW episode)',
     'motion_start': 28620, 'motion_end': 29820,
     'sub_segments': None,
     'source_segment_ids': []},

    # G_2: extends existing G_2/snippet_001 (which covers segs 13-17 at 39131..42113)
    # snippet_002 = later region (after snippet_001 motion)
    {'episode': 'G_2', 'snippet_id': '002', 'rank': 13,
     'description': 'strict-criterion backfill: 24% motion, 23% concurrent (later region)',
     'motion_start': 43380, 'motion_end': 44580,
     'sub_segments': None,
     'source_segment_ids': []},

    # B_2: NEW episode (no existing snippets)
    {'episode': 'B_2', 'snippet_id': '001', 'rank': 14,
     'description': 'strict-criterion backfill: 23% motion, 23% concurrent (NEW episode)',
     'motion_start': 19260, 'motion_end': 20460,
     'sub_segments': None,
     'source_segment_ids': []},

    # G_2: third snippet, earlier region (BEFORE snippet_001 motion at 39131..42113)
    {'episode': 'G_2', 'snippet_id': '003', 'rank': 15,
     'description': 'strict-criterion backfill: 21% motion, 21% concurrent (earlier region)',
     'motion_start': 29460, 'motion_end': 30660,
     'sub_segments': None,
     'source_segment_ids': []},
]


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for spec in BACKFILL_SPECS:
        snip_dir = OUT_ROOT / spec['episode'] / f"snippet_{spec['snippet_id']}"
        if snip_dir.is_dir() and (snip_dir / 'rgb').is_dir() and any((snip_dir / 'rgb').glob('frame_*.png')):
            print(f"\n=== {spec['episode']}/snippet_{spec['snippet_id']} already built — skipping ===")
            continue
        build_one(spec)

    print('\n=== updating per-episode index files ===')
    for ep in sorted(set(s['episode'] for s in BACKFILL_SPECS)):
        write_episode_index(ep)
        print(f'  {ep}_snippets.json')

    print('\nAll done.')


if __name__ == '__main__':
    main()
