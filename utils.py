"""
Utility functions for device detection and configuration.
"""

import torch
from typing import Tuple


def get_optimal_device() -> Tuple[str, str]:
    """
    Automatically detect the best available device.

    Returns:
        Tuple of (device_name, device_type)
        - device_name: 'cuda', 'mps', or 'cpu'
        - device_type: Human-readable description
    """
    if torch.cuda.is_available():
        device = 'cuda'
        device_type = f'CUDA ({torch.cuda.get_device_name(0)})'
    elif torch.backends.mps.is_available():
        device = 'mps'
        device_type = 'MPS (Apple Silicon)'
    else:
        device = 'cpu'
        device_type = 'CPU'

    return device, device_type


def get_optimal_dtype(device: str) -> str:
    """
    Get optimal dtype for the given device.

    Args:
        device: Device name ('cuda', 'mps', or 'cpu')

    Returns:
        Optimal dtype as string
    """
    if device == 'cuda':
        # CUDA supports bfloat16 and float16
        return 'bfloat16'
    elif device == 'mps':
        # MPS has better support for float16 than bfloat16
        return 'float16'
    else:
        # CPU: use float32 for best compatibility
        return 'float32'


def configure_device_and_dtype(device: str = None, dtype: str = None) -> Tuple[str, str]:
    """
    Configure device and dtype with automatic detection.

    Args:
        device: Optional device override ('cuda', 'mps', 'cpu', or 'auto')
        dtype: Optional dtype override ('float32', 'float16', 'bfloat16', or 'auto')

    Returns:
        Tuple of (device, dtype)
    """
    # Auto-detect device if not specified or set to 'auto'
    if device is None or device == 'auto':
        device, device_type = get_optimal_device()
        print(f"Auto-detected device: {device_type}")
    else:
        # Validate specified device
        if device not in ['cuda', 'mps', 'cpu']:
            raise ValueError(f"Invalid device: {device}. Must be 'cuda', 'mps', 'cpu', or 'auto'")

        # Check if specified device is available
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            device = 'cpu'
        elif device == 'mps' and not torch.backends.mps.is_available():
            print("Warning: MPS not available, falling back to CPU")
            device = 'cpu'

    # Auto-detect dtype if not specified or set to 'auto'
    if dtype is None or dtype == 'auto':
        dtype = get_optimal_dtype(device)
        print(f"Auto-selected dtype: {dtype}")
    else:
        # Validate dtype
        if dtype not in ['float32', 'float16', 'bfloat16']:
            raise ValueError(f"Invalid dtype: {dtype}")

        # Warn about suboptimal choices
        if device == 'mps' and dtype == 'bfloat16':
            print("Warning: MPS has limited bfloat16 support, consider using float16")
        if device == 'cpu' and dtype != 'float32':
            print("Warning: CPU works best with float32")

    return device, dtype


def get_gpu_memory_info():
    """
    Get detailed GPU memory information.

    Returns:
        List of dicts with GPU memory info
    """
    if not torch.cuda.is_available():
        return []

    gpu_info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = props.total_memory / 1024**3

        gpu_info.append({
            'id': i,
            'name': torch.cuda.get_device_name(i),
            'total_memory': total,
            'allocated': allocated,
            'reserved': reserved,
            'free': total - reserved
        })

    return gpu_info


def print_device_info():
    """Print detailed device information."""
    print("\n" + "="*60)
    print("Device Information")
    print("="*60)

    # CUDA info
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        gpu_count = torch.cuda.device_count()
        print(f"  - Device count: {gpu_count}")

        for i in range(gpu_count):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / 1024**3
            print(f"    - Memory: {total_mem:.1f} GB")
            print(f"    - Compute Capability: {props.major}.{props.minor}")

        # Multi-GPU recommendation
        if gpu_count > 1:
            total_memory = sum(props.total_memory for props in
                             [torch.cuda.get_device_properties(i) for i in range(gpu_count)])
            print(f"\n  💡 Multi-GPU Setup Detected:")
            print(f"    - Total VRAM: {total_memory / 1024**3:.1f} GB")
            print(f"    - Recommended for models: up to {int(total_memory / 1024**3 / 2)}B parameters")

    else:
        print(f"✗ CUDA not available")

    # MPS info
    if torch.backends.mps.is_available():
        print(f"✓ MPS available (Apple Silicon)")
        print(f"  - Built with MPS: {torch.backends.mps.is_built()}")
    else:
        print(f"✗ MPS not available")

    # CPU info
    print(f"✓ CPU always available")
    print(f"  - PyTorch version: {torch.__version__}")

    # Optimal device
    device, device_type = get_optimal_device()
    print(f"\n⭐ Optimal device: {device_type}")
    print("="*60 + "\n")


def move_to_device(tensor: torch.Tensor, device: str) -> torch.Tensor:
    """
    Move tensor to device with proper error handling.

    Args:
        tensor: Input tensor
        device: Target device

    Returns:
        Tensor on target device
    """
    try:
        return tensor.to(device)
    except Exception as e:
        print(f"Warning: Failed to move tensor to {device}: {e}")
        print(f"Falling back to CPU")
        return tensor.to('cpu')


def setup_multi_gpu(num_gpus: int = None, max_memory_per_gpu: str = None) -> dict:
    """
    Setup multi-GPU configuration for large models.

    Args:
        num_gpus: Number of GPUs to use (None = all available)
        max_memory_per_gpu: Max memory per GPU (e.g., "22GB")

    Returns:
        Device map configuration dict
    """
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        return {"": "cpu"}

    available_gpus = torch.cuda.device_count()

    if num_gpus is None:
        num_gpus = available_gpus
    else:
        num_gpus = min(num_gpus, available_gpus)

    print(f"Setting up multi-GPU: using {num_gpus}/{available_gpus} GPUs")

    # Build max_memory dict
    max_memory = {}
    for i in range(num_gpus):
        if max_memory_per_gpu:
            max_memory[i] = max_memory_per_gpu
        else:
            # Leave 2GB buffer per GPU
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory / 1024**3
            max_memory[i] = f"{int(total - 2)}GB"

    # Add CPU fallback
    max_memory["cpu"] = "64GB"

    print(f"Max memory configuration: {max_memory}")

    return max_memory


def get_model_size_gb(model_name: str) -> float:
    """
    Estimate model size in GB based on name.

    Args:
        model_name: HuggingFace model name

    Returns:
        Estimated size in GB
    """
    # Extract size from model name
    import re

    # Common patterns: "7B", "13B", "32B", "70B"
    match = re.search(r'(\d+)B', model_name, re.IGNORECASE)

    if match:
        size_b = int(match.group(1))
        # Assume bfloat16: 2 bytes per parameter
        return size_b * 2
    else:
        # Common model sizes
        if '2b' in model_name.lower() or 'gemma-2' in model_name.lower():
            return 4
        elif '3b' in model_name.lower():
            return 6
        elif '7b' in model_name.lower():
            return 14
        else:
            return 10  # Default estimate
