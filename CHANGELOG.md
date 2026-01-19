# SAM3 CPU Compatibility Changelog

This document details all modifications made to Facebook's SAM3 codebase to enable CPU-only inference. These changes allow SAM3 to run on systems without CUDA-capable GPUs.

**Reference:** [PR #173 - Device-agnostic inference](https://github.com/facebookresearch/sam3/pull/173)

---

## Summary

| Category | Files Modified | Changes |
|----------|---------------|---------|
| Conditional Imports | 2 | Made triton/decord imports optional |
| Device Detection | 4 | Auto-detect device from model/input |
| Cache Device Handling | 2 | Ensure cached tensors match input device |
| Autocast Context | 3 | Use nullcontext() fallback for CPU |
| CUDA-specific APIs | 4 | Guard pin_memory() and .cuda() calls |
| Decorator Fixes | 1 | Replace @torch.autocast with device-aware decorator |

---

## Detailed Changes

### 1. `sam3/model/position_encoding.py`

#### Change 1: Device Fallback in Constructor (line 38-40)
**Why:** The position encoder was initialized with `device="cuda"` even when CUDA isn't available, causing initialization failures.

```python
# BEFORE
self.device = device

# AFTER
if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"
self.device = device
```

#### Change 2: Cache Device Mismatch Fix (line 99-102)
**Why:** Cached positional encodings were stored on the initialization device but not moved when retrieved for inputs on a different device.

```python
# BEFORE
if cache_key in self.cache:
    return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)

# AFTER
if cache_key in self.cache:
    # Ensure cached tensor is on the same device as input
    cached = self.cache[cache_key].to(x.device)
    return cached[None].repeat(x.shape[0], 1, 1, 1)
```

---

### 2. `sam3/model/decoder.py`

#### Change 1: Device Initialization Fallback (line 278-281)
**Why:** Hardcoded `device="cuda"` in coordinate cache initialization caused failures on CPU-only systems.

```python
# BEFORE
coords_h, coords_w = self._get_coords(feat_size, feat_size, device="cuda")

# AFTER
init_device = "cuda" if torch.cuda.is_available() else "cpu"
coords_h, coords_w = self._get_coords(feat_size, feat_size, device=init_device)
```

#### Change 2: FFN Autocast Context (line 316-325)
**Why:** `torch.autocast(device_type="cuda")` fails on CPU. Using nullcontext() as fallback.

```python
# BEFORE
def forward_ffn(self, tgt):
    with torch.amp.autocast(device_type="cuda", enabled=False):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
    return tgt + self.dropout4(tgt2)

# AFTER
def forward_ffn(self, tgt):
    from contextlib import nullcontext
    if torch.cuda.is_available():
        ctx = torch.amp.autocast(device_type="cuda", enabled=False)
    else:
        ctx = nullcontext()
    with ctx:
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
    return tgt + self.dropout4(tgt2)
```

#### Change 3: Coord Cache Device Mismatch Fix (line 346-350)
**Why:** Cached coordinates might be on a different device than the input `reference_boxes`.

```python
# BEFORE
coords_h, coords_w = self.compilable_cord_cache

# AFTER
coords_h, coords_w = self.compilable_cord_cache
# Ensure coords are on the same device as reference_boxes (CPU compatibility)
if coords_h.device != reference_boxes.device:
    coords_h = coords_h.to(reference_boxes.device)
    coords_w = coords_w.to(reference_boxes.device)
```

---

### 3. `sam3/model/geometry_encoders.py`

#### Change: Remove pin_memory() Usage (line ~195)
**Why:** `pin_memory()` is a CUDA optimization that fails on CPU with "Cannot access accelerator device when none is available".

```python
# BEFORE
scale = scale.pin_memory().to(device=boxes_xyxy.device, non_blocking=True)

# AFTER
scale = torch.tensor([W, H, W, H], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device)
```

---

### 4. `sam3/model/io_utils.py`

#### Change: Add Device-Aware Helper Function (line 29-33)
**Why:** Multiple `.cuda()` calls throughout the file needed a centralized helper that respects CPU mode.

```python
# ADDED
def _to_device(tensor, offload_to_cpu=False):
    """Move tensor to appropriate device (CUDA if available and not offloading, else CPU)."""
    if offload_to_cpu or not torch.cuda.is_available():
        return tensor
    return tensor.cuda()
```

All subsequent `.cuda()` calls in this file were replaced with `_to_device(tensor)`.

---

### 5. `sam3/model/sam3_tracker_base.py`

#### Change 1: Remove pin_memory() Call (line ~195)
**Why:** `pin_memory()` fails on CPU-only systems.

```python
# BEFORE
pos_enc = torch.tensor(rel_pos_list).pin_memory().to(device=device, non_blocking=True)

# AFTER
pos_enc = torch.tensor(rel_pos_list, device=device) / t_diff_max
```

#### Change 2: Replace .cuda() Calls (multiple locations)
**Why:** Direct `.cuda()` calls fail when CUDA is unavailable.

```python
# BEFORE
feats = prev["maskmem_features"].cuda(non_blocking=True)
maskmem_enc = prev["maskmem_pos_enc"][-1].cuda()

# AFTER
feats = prev["maskmem_features"].to(device=device, non_blocking=True)
maskmem_enc = prev["maskmem_pos_enc"][-1].to(device=device)
```

---

### 6. `sam3/model/sam3_tracking_predictor.py`

#### Change: Device-Aware Autocast Context (line ~85)
**Why:** Hardcoded CUDA autocast fails on CPU.

```python
# BEFORE
self.bf16_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

# AFTER
from contextlib import nullcontext
if torch.cuda.is_available():
    self.bf16_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
else:
    self.bf16_context = nullcontext()
```

---

### 7. `sam3/model/sam3_video_predictor.py`

#### Change 1: CPU Fallback for Model Initialization (line ~52-54)
**Why:** Model was always moved to CUDA regardless of availability.

```python
# BEFORE
self.model = self.model.cuda()

# AFTER
if torch.cuda.is_available():
    self.model = self.model.cuda()
```

#### Change 2: Guard Multi-GPU Logging (line ~450)
**Why:** NCCL warm-up with `.cuda()` fails on CPU-only systems.

```python
# This code path only executes in multi-GPU scenarios which require CUDA
# No changes needed as it's naturally guarded by the multi-GPU context
```

---

### 8. `sam3/model/sam3_video_inference.py`

#### Change 1: Add Device-Aware Autocast Decorator (line 32-43)
**Why:** `@torch.autocast(device_type="cuda")` decorator fails on CPU.

```python
# ADDED
def cuda_autocast_if_available(dtype=torch.bfloat16):
    """Decorator that applies torch.autocast only when CUDA is available."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=dtype):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator
```

#### Change 2: Replace Autocast Decorators (lines 815, 923)
**Why:** Original decorators assumed CUDA availability.

```python
# BEFORE
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def warm_up_compilation(self):

# AFTER
@cuda_autocast_if_available(dtype=torch.bfloat16)
def warm_up_compilation(self):
```

---

### 9. `sam3/model/vl_combiner.py`

#### Change: Auto-Detect Device from Model Parameters
**Why:** Hardcoded device defaults caused issues when model was on CPU.

```python
# BEFORE
device = "cuda"

# AFTER
device = next(self.parameters()).device
```

---

### 10. `sam3/sam/transformer.py`

#### Change: Device Fallback in RoPEAttention (line 286)
**Why:** Hardcoded CUDA device for frequency computation.

```python
# BEFORE
device = torch.device("cuda")

# AFTER
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
```

---

### 11. `sam3/model/sam3_tracker_utils.py`

#### Change: Conditional Triton Import (line 10-18)
**Why:** Triton is CUDA-only and fails to import on CPU systems.

```python
# BEFORE
from sam3.model.edt import edt_triton

# AFTER
_HAS_TRITON = False
edt_triton = None
if torch.cuda.is_available():
    try:
        from sam3.model.edt import edt_triton
        _HAS_TRITON = True
    except ImportError:
        pass
```

#### Change: Fallback to Slow Implementation (line 163-165)
**Why:** When Triton is unavailable, use OpenCV-based CPU implementation.

```python
# ADDED at start of sample_one_point_from_error_center()
if not _HAS_TRITON:
    return sample_one_point_from_error_center_slow(gt_masks, pred_masks, padding)
```

---

### 12. `sam3/train/data/sam3_image_dataset.py`

#### Change: Conditional Decord Import (line 20-26)
**Why:** Decord may not be installed on all systems and isn't needed for image inference.

```python
# BEFORE
from decord import cpu, VideoReader

# AFTER
try:
    from decord import cpu, VideoReader
    _HAS_DECORD = True
except ImportError:
    _HAS_DECORD = False
    cpu = None
    VideoReader = None
```

---

### 13. `sam3/model_builder.py`

#### Change: Explicit CPU Device Handling (line 552-563)
**Why:** Original code only handled CUDA, not explicit CPU requests.

```python
# BEFORE
def _setup_device_and_mode(model, device, eval_mode):
    model = model.cuda()
    ...

# AFTER
def _setup_device_and_mode(model, device, eval_mode):
    """Setup model device and evaluation mode."""
    if device == "cuda":
        model = model.cuda()
    elif device == "cpu":
        model = model.cpu()
    else:
        model = model.to(device)
    if eval_mode:
        model.eval()
    return model
```

---

### 14. `sam3/model/sam3_image_processor.py`

#### Change: Auto-Detect Device from Model (line 17-21)
**Why:** Processor should inherit device from the model rather than assuming CUDA.

```python
# BEFORE
def __init__(self, model, resolution=1008, device="cuda", ...):

# AFTER
def __init__(self, model, resolution=1008, device=None, ...):
    if device is None:
        device = next(model.parameters()).device
        device = str(device) if hasattr(device, '__str__') else "cpu"
```

---

## Testing

To verify CPU compatibility:

```bash
# Force CPU mode
set CUDA_VISIBLE_DEVICES=
python scripts/test_crcd_prompts.py --device cpu --image "data/split_imgs/split_0/00000.jpg" --prompts "liver,gallbladder,tool"
```

## Performance Notes

| Metric | CPU Mode | GPU Mode (Reference) |
|--------|----------|---------------------|
| Model Load | 30-60s | 10-20s |
| Inference/Image | 30-120s | 1-5s |
| RAM Usage | 8-16 GB | 4-8 GB VRAM |

CPU inference is significantly slower but functional for batch processing or systems without compatible GPUs.
