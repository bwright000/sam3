"""Wraps Sam3VideoPredictor tracker for anchor commit and propagation.

One SAM3Service instance per process. A single 'session' is active at a time
(one snippet open). Switching snippets tears down GPU state and re-inits.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

import numpy as np
import torch

from .refinement import mask_to_low_res_logits, preprocess_painted_mask
from .storage import SnippetInfo


class SAM3Service:
    def __init__(self, no_model: bool = False):
        self.no_model = no_model
        self.predictor = None
        self.tracker = None
        self._state = None
        self._images_tensor = None
        self._snippet: Optional[SnippetInfo] = None
        self._lock = threading.Lock()
        # obj_id assignment per category for current snippet
        self._cat_to_objid: dict[str, int] = {}

    # -------- lifecycle --------

    def load_model(self) -> None:
        if self.no_model:
            return
        if self.predictor is not None:
            return
        from sam3.model.sam3_video_predictor import Sam3VideoPredictor
        t0 = time.time()
        self.predictor = Sam3VideoPredictor(apply_temporal_disambiguation=True)
        self.tracker = self.predictor.model.tracker
        self.tracker.backbone = self.predictor.model.detector.backbone
        print(f"[sam3_service] model loaded in {time.time()-t0:.1f}s")

    def reset_tracker_state(self) -> None:
        with self._lock:
            if self._state is not None:
                del self._state
                self._state = None
            if self._images_tensor is not None:
                del self._images_tensor
                self._images_tensor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def open_snippet(self, snippet: SnippetInfo) -> None:
        """Tear down any existing tracker state and prepare for a new snippet.

        Frames are NOT loaded to GPU yet — that happens lazily on first
        commit_anchor / propagate call to save startup time.
        """
        self.reset_tracker_state()
        self._snippet = snippet
        self._cat_to_objid = {}

    def ensure_tracker_loaded(self) -> None:
        """Load the current snippet's frames into tracker GPU state if not done."""
        if self.no_model:
            return
        with self._lock:
            if self._state is not None:
                return
            if self._snippet is None:
                raise RuntimeError("no snippet open")
            from scripts.generate_tool_masks_video import _load_frames_for_tracker
            n = len(self._snippet.frame_files)
            images, vh, vw = _load_frames_for_tracker(
                self._snippet.frame_files, n, self.tracker.image_size
            )
            self._images_tensor = images
            self._state = self.tracker.init_state(
                video_height=vh, video_width=vw, num_frames=n
            )
            self._state["images"] = images

    def cat_to_objid(self, category: str) -> int:
        if category not in self._cat_to_objid:
            self._cat_to_objid[category] = max(self._cat_to_objid.values(), default=0) + 1
        return self._cat_to_objid[category]

    # -------- single-frame preview (SAM3 click/box) --------

    def preview_click(self, frame_idx: int, category: str,
                      points: list[tuple[float, float, int]] | None = None,
                      box: tuple[float, float, float, float] | None = None,
                      ) -> np.ndarray:
        """Run single-frame inference with clicks+box, return binary mask.

        points: list of (x, y, label) in original image coords
        box: (x1, y1, x2, y2) in original image coords
        """
        if self.no_model:
            raise RuntimeError("model disabled (--no-model)")
        self.ensure_tracker_loaded()
        with self._lock:
            obj_id = self.cat_to_objid(category)

            pts_xy = [[p[0], p[1]] for p in (points or [])]
            labels = [p[2] for p in (points or [])]
            kwargs = dict(
                inference_state=self._state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                rel_coordinates=False,
                clear_old_points=True,
            )
            if box is not None:
                kwargs["box"] = [[box[0], box[1], box[2], box[3]]]
            if pts_xy:
                kwargs["points"] = pts_xy
                kwargs["labels"] = labels

            self.tracker.add_new_points_or_box(**kwargs)
            self.tracker.propagate_in_video_preflight(self._state, run_mem_encoder=True)

            for fidx, obj_ids, _, video_res_masks, obj_scores in self.tracker.propagate_in_video(
                self._state, start_frame_idx=frame_idx,
                max_frame_num_to_track=0, reverse=False,
            ):
                for i, oid in enumerate(obj_ids):
                    if int(oid) != obj_id:
                        continue
                    ml = video_res_masks[i]
                    mnp = ml.squeeze(0).cpu().numpy() if isinstance(ml, torch.Tensor) else ml
                    return (mnp > 0.0).astype(np.uint8)
        return np.zeros((self._snippet.height, self._snippet.width), dtype=np.uint8)

    # -------- commit painted mask as anchor --------

    def commit_anchor(self, frame_idx: int, category: str,
                      painted_mask: np.ndarray) -> np.ndarray:
        """Preprocess painted mask and register with the tracker via add_new_mask.

        Returns the refined mask SAM3 actually uses (at video resolution).
        """
        if self.no_model:
            return preprocess_painted_mask(painted_mask)
        self.ensure_tracker_loaded()
        with self._lock:
            obj_id = self.cat_to_objid(category)
            clean = preprocess_painted_mask(painted_mask)
            if clean.sum() == 0:
                return clean
            mask_tensor = torch.from_numpy(clean).float()
            self.tracker.add_new_mask(
                inference_state=self._state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=mask_tensor,
            )
            return clean

    # -------- bidirectional propagation (chunked) --------

    def propagate_bidir(self,
                        anchor_frame_idx: int,
                        max_frames_per_direction: int,
                        progress: Optional[Callable[[dict], None]] = None,
                        ) -> dict[int, dict[int, np.ndarray]]:
        """Propagate from a single anchor frame in both directions.

        Returns: {frame_idx: {obj_id: binary mask}} for all frames touched.
        """
        if self.no_model:
            return {}
        self.ensure_tracker_loaded()
        results: dict[int, dict[int, np.ndarray]] = {}
        with self._lock:
            self.tracker.propagate_in_video_preflight(self._state, run_mem_encoder=True)
            for reverse in (False, True):
                direction = "backward" if reverse else "forward"
                count = 0
                for fidx, obj_ids, _, video_res_masks, obj_scores in self.tracker.propagate_in_video(
                    self._state,
                    start_frame_idx=anchor_frame_idx,
                    max_frame_num_to_track=max_frames_per_direction,
                    reverse=reverse,
                ):
                    if fidx not in results:
                        results[fidx] = {}
                    for i, oid in enumerate(obj_ids):
                        ml = video_res_masks[i]
                        mnp = ml.squeeze(0).cpu().numpy() if isinstance(ml, torch.Tensor) else ml
                        results[fidx][int(oid)] = (mnp > 0.0).astype(np.uint8)
                    count += 1
                    if progress is not None:
                        progress({
                            "direction": direction,
                            "frame_idx": fidx,
                            "count": count,
                            "max": max_frames_per_direction,
                        })
        return results

    @property
    def objid_to_cat(self) -> dict[int, str]:
        return {v: k for k, v in self._cat_to_objid.items()}
