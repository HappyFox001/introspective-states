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


def print_device_info():
    """Print detailed device information."""
    print("\n" + "="*60)
    print("Device Information")
    print("="*60)

    # CUDA info
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  - Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    - Memory: {props.total_memory / 1024**3:.1f} GB")
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
