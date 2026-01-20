"""
Device configuration module for SAM3 on limited hardware.

Supports:
- CPU mode (primary for limited hardware)
- CUDA mode with memory optimizations
- MPS mode (Apple Silicon)
- Automatic hardware detection and configuration

Device Priority: CUDA > MPS > CPU
"""

import sys
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DeviceConfig:
    """Configuration for device and memory settings."""
    device: str
    dtype: torch.dtype
    use_compile: bool
    use_half_precision: bool
    vram_gb: Optional[float]
    ram_gb: Optional[float]


def is_mps_available() -> bool:
    """Check if MPS (Apple Metal Performance Shaders) is available."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_default_device() -> str:
    """Get the best available device with priority: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    elif is_mps_available():
        return "mps"
    else:
        return "cpu"


def get_system_info() -> dict:
    """Get system hardware information."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": None,
        "cuda_vram_gb": None,
        "cuda_compute_capability": None,
        "mps_available": is_mps_available(),
    }

    if info["cuda_available"] and info["cuda_device_count"] > 0:
        props = torch.cuda.get_device_properties(0)
        info["cuda_device_name"] = props.name
        info["cuda_vram_gb"] = props.total_memory / (1024 ** 3)
        info["cuda_compute_capability"] = f"{props.major}.{props.minor}"

    return info


def print_system_info():
    """Print system hardware information."""
    info = get_system_info()

    print("=" * 50)
    print("System Hardware Information")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {info['cuda_available']}")
    print(f"MPS available: {info['mps_available']}")

    if info["cuda_available"]:
        print(f"CUDA device: {info['cuda_device_name']}")
        print(f"CUDA VRAM: {info['cuda_vram_gb']:.2f} GB")
        print(f"Compute capability: {info['cuda_compute_capability']}")
    elif info["mps_available"]:
        print("Apple Metal Performance Shaders (MPS) detected")
    else:
        print("Running in CPU-only mode")

    print(f"Default device: {get_default_device()}")
    print("=" * 50)


def configure_device(
    device: Optional[str] = None,
    force_cpu: bool = False,
    verbose: bool = True
) -> DeviceConfig:
    """
    Configure device settings based on hardware capabilities.

    Args:
        device: Explicit device selection ('cpu', 'cuda', or 'mps')
        force_cpu: Force CPU mode regardless of other device availability
        verbose: Print configuration details

    Returns:
        DeviceConfig with optimal settings for the hardware

    Device Priority: CUDA > MPS > CPU
    """
    info = get_system_info()

    # Determine device with priority: CUDA > MPS > CPU
    if force_cpu or device == "cpu":
        selected_device = "cpu"
    elif device == "cuda" and info["cuda_available"]:
        selected_device = "cuda"
    elif device == "mps" and info["mps_available"]:
        selected_device = "mps"
    elif device is None:
        # Auto-detect best available device
        selected_device = get_default_device()
    else:
        # Fallback to CPU if requested device not available
        selected_device = "cpu"

    # Configure dtype and optimizations based on device
    vram = None

    if selected_device == "cpu":
        # CPU mode: use float32, no compile (not beneficial on CPU)
        dtype = torch.float32
        use_compile = False
        use_half = False

    elif selected_device == "mps":
        # MPS mode (Apple Silicon): use float16 (bfloat16 not supported)
        dtype = torch.float16
        use_compile = False  # torch.compile has limited MPS support
        use_half = True

    else:
        # CUDA mode: configure based on VRAM
        vram = info["cuda_vram_gb"]

        # Determine if we can use half precision
        # GTX 970 (compute 5.2) has limited fp16 support
        compute_major = int(info["cuda_compute_capability"].split(".")[0])

        if compute_major >= 7:  # Volta and newer (RTX, etc.)
            dtype = torch.float16
            use_half = True
            use_compile = True
        elif compute_major >= 6:  # Pascal (GTX 10xx)
            dtype = torch.float16
            use_half = True
            use_compile = False  # Compile may not help on older GPUs
        else:  # Maxwell and older (GTX 9xx)
            dtype = torch.float32
            use_half = False
            use_compile = False

    config = DeviceConfig(
        device=selected_device,
        dtype=dtype,
        use_compile=use_compile,
        use_half_precision=use_half,
        vram_gb=vram if selected_device == "cuda" else None,
        ram_gb=None,  # Could add psutil for RAM detection
    )

    if verbose:
        print_device_config(config)

    return config


def print_device_config(config: DeviceConfig):
    """Print the device configuration."""
    print("=" * 50)
    print("SAM3 Device Configuration")
    print("=" * 50)
    print(f"Device: {config.device}")
    print(f"Data type: {config.dtype}")
    print(f"torch.compile: {'enabled' if config.use_compile else 'disabled'}")
    print(f"Half precision: {'enabled' if config.use_half_precision else 'disabled'}")

    if config.device == "cuda" and config.vram_gb:
        print(f"VRAM available: {config.vram_gb:.2f} GB")

        # Memory warning for SAM3
        if config.vram_gb < 8:
            print("\n*** WARNING: SAM3 (848M params) typically requires 8+ GB VRAM ***")
            print("*** Consider using CPU mode for better stability ***")
        elif config.vram_gb < 16:
            print("\nNote: Limited VRAM - may experience OOM on large images")

    elif config.device == "mps":
        print("\nRunning in MPS mode (Apple Silicon)")
        print("Note: Some operations may fall back to CPU")
        print("Note: Using float16 (bfloat16 not supported on MPS)")

    else:
        print("\nRunning in CPU mode")
        print("Note: Inference will be slow (~30-120 seconds per image)")

    print("=" * 50)


def get_torch_device(config: DeviceConfig) -> torch.device:
    """Get torch.device from config."""
    return torch.device(config.device)


def estimate_memory_requirements():
    """Print estimated memory requirements for SAM3."""
    print("=" * 50)
    print("SAM3 Memory Requirements (Estimated)")
    print("=" * 50)
    print("Model weights: ~3.5 GB")
    print("Inference (per image):")
    print("  - GPU (fp16): ~8-12 GB VRAM")
    print("  - GPU (fp32): ~16-24 GB VRAM")
    print("  - CPU: ~12-16 GB RAM")
    print("=" * 50)


if __name__ == "__main__":
    # Run diagnostics
    print_system_info()
    estimate_memory_requirements()

    # Test configuration
    print("\nTesting CPU configuration:")
    config_cpu = configure_device(device="cpu")

    if torch.cuda.is_available():
        print("\nTesting CUDA configuration:")
        config_cuda = configure_device(device="cuda")

    if is_mps_available():
        print("\nTesting MPS configuration:")
        config_mps = configure_device(device="mps")

    print("\nAuto-detected best device:")
    config_auto = configure_device(device=None)
