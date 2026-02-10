"""
Centralized AMP (Automatic Mixed Precision) dtype selection.

Ampere+ GPUs (A100, A10, etc.) support native bfloat16 and reliable autocast.
Older GPUs (Tesla T4, V100) have unreliable autocast behavior across SAM3's
multi-component architecture (detector → tracker → memory encoder), causing
scattered dtype mismatches at conv layer boundaries.

Strategy:
  - Ampere+ (compute cap >= 8.0): bfloat16 autocast as Meta designed
  - Older GPUs (T4, V100): disable autocast, run in float32
"""

import torch


def _detect_gpu_capability():
    """Detect GPU compute capability and set AMP strategy."""
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            if props.major >= 8:  # Ampere+ (A100, A10, H100, etc.)
                return torch.bfloat16, True
            else:  # Turing (T4), Volta (V100), etc.
                return torch.float32, False
        except Exception:
            pass
    return torch.bfloat16, True  # default for non-CUDA


AMP_DTYPE, AUTOCAST_ENABLED = _detect_gpu_capability()
