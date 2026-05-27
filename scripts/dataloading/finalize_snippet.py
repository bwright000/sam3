#!/usr/bin/env python3
"""Finalize a snippet to publication-grade form after promotion.

Applies the standard set of post-promotion transformations that the exemplar
(E_3/snippet_002) was hardened with on 2026-05-17:

  1. Polygon geometry repair via `shapely.validation.make_valid`. Handles
     `Polygon`, `MultiPolygon`, AND `GeometryCollection` returns (the latter
     was the silent-data-loss bug we caught visually on the exemplar). After
     repair, recomputes `annotation.area` and `annotation.bbox` from the
     rasterised mask.
  2. Regenerates `semantic_instance/*.png` from the repaired COCO via the
     canonical `promote_replay.paint_from_scratch` rasteriser. This locks in
     raster <-> polygon agreement bit-exactly.
  3. Upgrades `info_semantic.json` to the explicit schema with `coco_id` +
     `pixel_value` fields (legacy basic schema is also accepted as input).
  4. Adds `depth_placeholder: true` sentinel to `intrinsics.yaml` if not
     already present, matching DATASET.md §1's documented depth-absent state.
  5. Renders `snippet_overlay.mp4` from the repaired canonical for visual review.
  6. Scrubs dev artefacts (.bak* files, intermediate JSONs, replay inputs,
     legacy overlay names) so the snippet directory contains ONLY the
     publication-grade file set documented in DATASET.md §4.

After this script, a snippet is ready for `scripts.evaluation.generate_manifest`
and `scripts.evaluation.publication_gate`. The script is idempotent: re-running
it on a snippet that is already finalized is a no-op for everything except the
overlay render (which is regenerated every time so it stays in sync).

Usage:
  python -m scripts.dataloading.finalize_snippet --snippet-dir <path>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from collections import defaultdict, Counter
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import yaml
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.validation import make_valid

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.propagation.promote_replay import (  # noqa: E402
    CANONICAL_CATS, NAME_TO_CANONICAL_ID, PIXEL_FOR_COCO,
    paint_from_scratch,
)
from sam3_annotator.server.rle import (  # noqa: E402
    polygons_to_mask_even_odd, mask_to_polygons_with_holes,
)


# ---------------------------------------------------------------------------
# Polygon repair
# ---------------------------------------------------------------------------

def _ring_from_coords(coords):
    """Flat coord list from a shapely linear-ring coord sequence."""
    pts = list(coords)
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    flat = [c for xy in pts for c in xy]
    return flat if len(flat) >= 6 else None


def _extract_polygon_rings(poly):
    """Extract COCO rings (exterior + interior) from one shapely Polygon."""
    out = []
    if poly.is_empty:
        return out
    flat_ext = _ring_from_coords(poly.exterior.coords)
    if flat_ext:
        out.append(flat_ext)
    for interior in poly.interiors:
        flat_int = _ring_from_coords(interior.coords)
        if flat_int:
            out.append(flat_int)
    return out


def _polygon_to_rings(geom):
    """Recursively walk any shapely geometry tree → list of COCO rings.

    Handles Polygon, MultiPolygon, GeometryCollection. Non-polygon members
    (LineString, Point) from make_valid are skipped — they're zero-area
    artefacts of resolving self-intersections.
    """
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return _extract_polygon_rings(geom)
    if isinstance(geom, MultiPolygon):
        out = []
        for p in geom.geoms:
            out.extend(_extract_polygon_rings(p))
        return out
    if isinstance(geom, GeometryCollection):
        out = []
        for g in geom.geoms:
            out.extend(_polygon_to_rings(g))
        return out
    return []


def repair_polygons(coco: dict, h: int = 720, w: int = 1280) -> dict:
    """Repair self-intersecting / invalid polygons in-place, recompute area + bbox.

    Non-destructive on coverage: rings whose raster area is non-zero are always
    preserved (either as the make_valid output, or — if make_valid produces no
    polygonal geometry — as the original ring, since cv2.fillPoly handles
    self-intersecting rings via its winding rule). Rings whose raster area is
    zero (e.g. 4-point self-intersecting bow-ties that collapse to a
    MultiLineString under make_valid) are dropped — they contribute no
    coverage so dropping them is consistent with the coverage-preservation
    rule, AND removing them lets the snippet pass layer-F polygon validity.

    Returns a stats dict.
    """
    n_repaired = 0
    n_unchanged = 0
    n_kept_invalid = 0  # rings that make_valid couldn't repair but had area, so kept
    n_dropped_zero_area = 0  # rings dropped because raster area == 0

    for ann in coco.get('annotations', []):
        seg = ann.get('segmentation') or []
        if not isinstance(seg, list):
            continue
        new_rings = []
        any_invalid = False
        for ring in seg:
            if not isinstance(ring, list) or len(ring) < 6:
                continue
            pts = [(ring[i], ring[i + 1]) for i in range(0, len(ring), 2)]
            try:
                poly = Polygon(pts)
            except Exception:
                # Couldn't even construct a Polygon — keep the ring; cv2.fillPoly
                # is forgiving of malformed inputs. (This branch is rare.)
                new_rings.append(ring)
                n_kept_invalid += 1
                continue
            if poly.is_valid:
                new_rings.append(ring)
                continue
            any_invalid = True
            try:
                fixed = make_valid(poly)
            except Exception:
                try:
                    fixed = poly.buffer(0)
                except Exception:
                    fixed = None
            repaired = _polygon_to_rings(fixed) if fixed is not None else []
            if not repaired:
                # make_valid produced nothing polygonal. Decide whether to drop
                # the ring or keep it. Drop if it is genuinely 1D / zero-area
                # (a spike or bow-tie that collapses to a line under
                # make_valid) — these contribute no real coverage; any pixels
                # cv2.fillPoly produces from them are line-painting artefacts.
                # Otherwise keep it — non-trivial self-intersecting rings can
                # still be meaningfully rasterised under the winding rule.
                fixed_area = float(fixed.area) if (fixed is not None) else 0.0
                if fixed_area == 0.0:
                    n_dropped_zero_area += 1
                    continue
                new_rings.append(ring)
                n_kept_invalid += 1
                continue
            new_rings.extend(repaired)
        if any_invalid and len(new_rings) > 0:
            # Only mark "repaired" if we actually changed something
            ann['segmentation'] = new_rings
            n_repaired += 1
            mask = np.zeros((h, w), dtype=np.uint8)
            for r in new_rings:
                pts = np.array(r, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], 1)
            ann['area'] = int(mask.sum())
            ys, xs = np.where(mask > 0)
            if len(xs):
                ann['bbox'] = [float(xs.min()), float(ys.min()),
                               float(xs.max() - xs.min()), float(ys.max() - ys.min())]
            else:
                ann['bbox'] = [0.0, 0.0, 0.0, 0.0]
        else:
            n_unchanged += 1

    return {
        'n_repaired': n_repaired,
        'n_unchanged': n_unchanged,
        'n_kept_invalid': n_kept_invalid,
        'n_dropped_zero_area': n_dropped_zero_area,
    }


# ---------------------------------------------------------------------------
# Polygon-level priority resolution
# ---------------------------------------------------------------------------

def resolve_polygon_priority(coco: dict, h: int = 720, w: int = 1280,
                             priority: tuple[str, ...] = ('Tool', 'Gallbladder', 'Liver')
                             ) -> dict:
    """Apply a category priority order at the polygon level (HOLE-AWARE).

    `priority` lists categories highest-first. Each lower-priority cat is
    clipped to exclude pixels claimed by ANY higher-priority cat. The
    default Tool > Gallbladder > Liver encodes the surgical-scene rule: the
    tool is physical foreground, gallbladder sits in front of liver.

    CRITICAL — hole awareness: a tool with an erased interior hole (tissue
    visible through a grasper joint) must NOT clip the tissue that shows
    through that hole. The exclusion mask is therefore rasterised with
    polygons_to_mask_even_odd (parity-XOR), so a higher-priority cat's
    holes are genuinely empty in the exclusion — the lower-priority cat
    survives there. Re-extraction uses mask_to_polygons_with_holes
    (RETR_CCOMP) so any hole created by clipping is preserved too.

    A previous version filled every polygon ring solid (per-ring fillPoly),
    which silently re-filled tool holes and deleted the tissue-through-hole
    regions during the publish. This is the fix for that bug.

    Mutates coco['annotations'] in place. Returns stats dict.
    """
    cats = {c['id']: c['name'] for c in coco['categories']}
    from collections import defaultdict
    by_frame_cat: dict[int, dict[str, dict]] = defaultdict(dict)
    for a in coco.get('annotations', []):
        nm = cats.get(a['category_id'])
        if nm:
            by_frame_cat[a['image_id']][nm] = a

    stats = {
        'frames_with_clipping': 0,
        'liver_anns_clipped': 0, 'liver_clipped_px_total': 0,
        'gallbladder_anns_clipped': 0, 'gallbladder_clipped_px_total': 0,
        'tool_anns_clipped': 0, 'tool_clipped_px_total': 0,
        'anns_emptied': 0,
    }

    def _raster(a) -> np.ndarray:
        """Hole-aware rasterisation of an annotation's polygons."""
        if a is None:
            return np.zeros((h, w), dtype=np.uint8)
        seg = a.get('segmentation') or []
        if not seg:
            return np.zeros((h, w), dtype=np.uint8)
        return polygons_to_mask_even_odd(seg, h, w)

    for fid, frame_anns in by_frame_cat.items():
        masks = {nm: _raster(frame_anns.get(nm)).astype(bool) for nm in priority}

        frame_clipped = False
        claimed = np.zeros((h, w), dtype=bool)
        for nm in priority:                       # high -> low
            a = frame_anns.get(nm)
            orig_bool = masks[nm]
            if a is None or not (a.get('segmentation') or []):
                claimed |= orig_bool
                continue
            clean_bool = orig_bool & ~claimed
            clipped_px = int(orig_bool.sum() - clean_bool.sum())
            # This cat now owns its surviving pixels
            claimed |= clean_bool

            if clipped_px <= 0:
                continue
            frame_clipped = True
            key = nm.lower()
            stats[f'{key}_clipped_px_total'] = stats.get(f'{key}_clipped_px_total', 0) + clipped_px
            stats[f'{key}_anns_clipped'] = stats.get(f'{key}_anns_clipped', 0) + 1

            # Hole-aware re-extraction (RETR_CCOMP) so any interior hole the
            # clip introduced survives in the polygon form.
            new_polys = mask_to_polygons_with_holes(
                clean_bool.astype(np.uint8), min_area=10, epsilon_max_px=0.5)

            if not new_polys:
                a['_drop'] = True
                stats['anns_emptied'] += 1
                continue

            a['segmentation'] = new_polys
            a['area'] = int(clean_bool.sum())
            ys, xs = np.where(clean_bool)
            if len(xs):
                a['bbox'] = [float(xs.min()), float(ys.min()),
                             float(xs.max() - xs.min()), float(ys.max() - ys.min())]
            else:
                a['bbox'] = [0.0, 0.0, 0.0, 0.0]

        if frame_clipped:
            stats['frames_with_clipping'] += 1

    if stats['anns_emptied'] > 0:
        coco['annotations'] = [a for a in coco['annotations'] if not a.pop('_drop', False)]
        for i, a in enumerate(coco['annotations'], 1):
            a['id'] = i

    return stats


# ---------------------------------------------------------------------------
# info_semantic.json upgrade
# ---------------------------------------------------------------------------

def upgrade_info_semantic(info_path: Path) -> dict:
    """Rewrite info_semantic.json in the explicit `coco_id` + `pixel_value` form."""
    classes = []
    for c in CANONICAL_CATS:
        cid = c['id']
        pv = PIXEL_FOR_COCO[cid]
        classes.append({
            'id': pv,
            'coco_id': cid,
            'pixel_value': pv,
            'name': c['name'],
            'supercategory': c['supercategory'],
        })
    new = {
        'classes': classes,
        'background_id': 0,
        'note': (
            'Each class entry carries three IDs: `coco_id` matches '
            'snippet_annotations.json::categories[].id; `pixel_value` is what '
            'appears in semantic_instance/*.png and equals `coco_id + 1`; `id` '
            'is an alias for `pixel_value` retained for Replica-style loader '
            'compatibility. 0 is reserved for background/unlabeled pixels. '
            'This dataset annotates 3 classes (Liver, Gallbladder, Tool); the '
            'source CRCD dataset defines additional Meat / Skin / FBF / PCH '
            'classes that are not annotated in our snippets and are therefore '
            'omitted from this taxonomy.'
        ),
    }
    info_path.write_text(json.dumps(new, indent=2))
    return new


# ---------------------------------------------------------------------------
# intrinsics.yaml depth_placeholder sentinel
# ---------------------------------------------------------------------------

def ensure_depth_placeholder(intr_path: Path) -> bool:
    """Add `depth_placeholder: true` to intrinsics.yaml. Returns True if added."""
    if not intr_path.is_file():
        return False
    intr = yaml.safe_load(intr_path.read_text()) or {}
    if intr.get('depth_placeholder') is True:
        return False
    intr['depth_placeholder'] = True
    intr['depth_placeholder_note'] = (
        'depth/ is intentionally empty (per DATASET.md §1). Depth maps are a '
        'downstream task; SLAM consumers should compute depth from the stereo '
        'pair.'
    )
    intr_path.write_text(yaml.safe_dump(intr, sort_keys=False))
    return True


# ---------------------------------------------------------------------------
# Overlay video render
# ---------------------------------------------------------------------------

_CAT_COLORS_BGR = {
    'Tool': (216, 180, 0), 'tool': (216, 180, 0),
    'Liver': (70, 57, 230), 'liver': (70, 57, 230),
    'Gallbladder': (135, 183, 82), 'gallbladder': (135, 183, 82),
}
_FALLBACK = [(255, 80, 80), (80, 255, 80), (80, 80, 255)]


def _draw_overlay(img, anns, cat_by_id, fill_alpha=0.35):
    out = img.copy()
    for j, a in enumerate(anns):
        cat = cat_by_id.get(a.get('category_id'), '?')
        c = _CAT_COLORS_BGR.get(cat, _FALLBACK[j % len(_FALLBACK)])
        seg = a.get('segmentation') or []
        if not isinstance(seg, list):
            continue
        ov = out.copy()
        drawn = []
        for poly in seg:
            if not isinstance(poly, list) or len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(ov, [pts], c)
            drawn.append(pts)
        out = cv2.addWeighted(ov, fill_alpha, out, 1 - fill_alpha, 0)
        for pts in drawn:
            cv2.polylines(out, [pts], True, c, 2)
        if drawn:
            x, y = int(drawn[0][0][0]), int(drawn[0][0][1])
            cv2.putText(out, cat, (x, max(15, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    return out


def render_overlay(sd: Path, fps: int = 30) -> dict:
    coco = json.loads((sd / 'snippet_annotations.json').read_text())
    cat_by_id = {c['id']: c['name'] for c in coco['categories']}
    by_frame = defaultdict(list)
    for a in coco['annotations']:
        by_frame[int(a['image_id'])].append(a)

    files = sorted((sd / 'rgb').glob('frame_*.png'))
    if not files:
        return {'status': 'no_rgb_frames'}
    first = cv2.imread(str(files[0]))
    h, w = first.shape[:2]
    out_mp4 = sd / 'snippet_overlay.mp4'

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast',
           '-crf', '23', str(out_mp4)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    cov = Counter()
    for f in files:
        num = int(f.stem.split('_')[1])
        img = cv2.imread(str(f))
        if img is None:
            continue
        anns = by_frame.get(num, [])
        for a in anns:
            cov[cat_by_id.get(a['category_id'], '?')] += 1
        out_img = _draw_overlay(img, anns, cat_by_id)
        cv2.putText(out_img, f'#{num}  anns:{len(anns)}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        proc.stdin.write(out_img.tobytes())
    proc.stdin.close()
    rc = proc.wait()
    return {
        'status': 'ok' if rc == 0 else f'ffmpeg_rc={rc}',
        'coverage': dict(cov),
        'mp4_mb': round(out_mp4.stat().st_size / 1e6, 1),
    }


# ---------------------------------------------------------------------------
# Dev-artefact cleanup
# ---------------------------------------------------------------------------

# Publication-grade file set (from DATASET.md §4 and matched by the E_3/snippet_002
# exemplar). Anything not in this allow-list is treated as a dev artefact and
# removed during finalize_snippet.
PUBLISHABLE_FILES = {
    'MANIFEST.json',
    'rgb.txt', 'rgbright.txt', 'depth.txt', 'associations.txt', 'groundtruth.txt',
    'intrinsics.yaml',
    'info_semantic.json', 'snippet_annotations.json',
    'scene_motion.json', 'cluster_metadata.json',
    'snippet_overlay.mp4', 'video_left.mp4',
}
PUBLISHABLE_DIRS = {'rgb', 'rgbright', 'depth', 'semantic_instance', 'overlays'}


def clean_dev_artefacts(sd: Path) -> dict:
    """Remove anything in the snippet dir that isn't part of the publication set.

    Allow-listed: top-level files in PUBLISHABLE_FILES, top-level dirs in
    PUBLISHABLE_DIRS. Everything else at the top level — backups, replay
    inputs, autosaves, intermediate JSONs, legacy overlay filenames — is
    removed.
    """
    removed_files: list[str] = []
    removed_dirs: list[str] = []
    for entry in sd.iterdir():
        name = entry.name
        if entry.is_dir():
            if name in PUBLISHABLE_DIRS:
                continue
            # Remove non-published top-level directories (e.g. combined_masks/,
            # overlays_tool.*/) — but be safe: only the known transient ones.
            if (name.startswith('overlays_tool')
                    or name == 'combined_masks'
                    or name.startswith('__pycache__')):
                import shutil
                shutil.rmtree(entry)
                removed_dirs.append(name)
            # Otherwise leave unknown dirs in place (e.g. _audit/, dataset
            # extras the user has added).
            continue
        if name in PUBLISHABLE_FILES:
            continue
        # Drop anything that matches a known dev-artefact pattern.
        if (
            '.bak' in name
            or name.endswith('.tmp')
            or name == 'session_autosave.json'
            or name.startswith('annotated_masks')
            or name.startswith('tool_detection_stats')
            or name.endswith('_results.json')
            or name == 'combined_annotations.json'
            or name == 'visualization.html'
            or name == 'velocity.png'
            or (name.endswith('.mp4') and name != 'snippet_overlay.mp4' and name != 'video_left.mp4')
        ):
            entry.unlink()
            removed_files.append(name)
    return {
        'removed_files': sorted(removed_files),
        'removed_dirs': sorted(removed_dirs),
        'n_removed': len(removed_files) + len(removed_dirs),
    }


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def finalize(sd: Path) -> dict:
    """Run the full publication-grade finalization on a snippet directory."""
    coco_p = sd / 'snippet_annotations.json'
    info_p = sd / 'info_semantic.json'
    intr_p = sd / 'intrinsics.yaml'

    if not coco_p.is_file():
        return {'status': 'error', 'reason': f'no snippet_annotations.json at {coco_p}'}

    result: dict = {'snippet': f'{sd.parent.name}/{sd.name}', 'steps': []}

    # 1. Polygon repair
    t0 = time.time()
    coco = json.loads(coco_p.read_text())
    repair_stats = repair_polygons(coco)
    result['steps'].append({
        'name': 'polygon_repair', 'elapsed_s': round(time.time() - t0, 1),
        **repair_stats,
    })

    # 2. Polygon-level priority resolution (Gallbladder > Liver > Tool).
    # Clips Tool polygons that extend into tissue regions, and Liver polygons
    # that extend into Gallbladder. Without this, snippet_annotations.json's
    # polygons can disagree with semantic_instance/*.png on shape.
    t0 = time.time()
    priority_stats = resolve_polygon_priority(coco)
    result['steps'].append({
        'name': 'resolve_polygon_priority',
        'elapsed_s': round(time.time() - t0, 1),
        **priority_stats,
    })

    # 2b. Re-run polygon repair to clean up any self-intersections introduced
    # by `cv2.approxPolyDP` during step 2's re-extraction. Without this, the
    # next audit run flags POLY_INVALID on the simplified-but-self-intersecting
    # output.
    t0 = time.time()
    repair2_stats = repair_polygons(coco)
    coco_p.write_text(json.dumps(coco))
    result['steps'].append({
        'name': 'polygon_repair_post_priority',
        'elapsed_s': round(time.time() - t0, 1),
        **repair2_stats,
    })

    # 3. Regenerate semantic_instance/ from priority-resolved COCO
    t0 = time.time()
    sem_dir = sd / 'semantic_instance'
    rgb_dir = sd / 'rgb'
    n_pngs = paint_from_scratch(sem_dir, rgb_dir, coco['annotations'], 720, 1280)
    result['steps'].append({
        'name': 'paint_from_scratch', 'elapsed_s': round(time.time() - t0, 1),
        'n_pngs': n_pngs,
    })

    # 3. Upgrade info_semantic.json to explicit schema
    upgrade_info_semantic(info_p)
    result['steps'].append({'name': 'info_semantic_upgrade'})

    # 4. Add depth_placeholder sentinel
    added = ensure_depth_placeholder(intr_p)
    result['steps'].append({'name': 'depth_placeholder', 'added': added})

    # 5. Render overlay
    t0 = time.time()
    render_result = render_overlay(sd)
    result['steps'].append({
        'name': 'render_overlay', 'elapsed_s': round(time.time() - t0, 1),
        **render_result,
    })

    # 6. Scrub dev artefacts so the snippet directory matches DATASET.md §4
    cleanup_stats = clean_dev_artefacts(sd)
    result['steps'].append({'name': 'clean_dev_artefacts', **cleanup_stats})

    result['status'] = 'ok'
    return result


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--snippet-dir', type=Path, required=True)
    args = ap.parse_args()
    try:
        result = finalize(args.snippet_dir.resolve())
    except Exception as e:
        print(f'ERROR: {type(e).__name__}: {e}', file=sys.stderr)
        traceback.print_exc()
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get('status') == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
