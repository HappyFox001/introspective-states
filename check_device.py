#!/usr/bin/env python
"""
Device detection and information tool.
Run this to check what acceleration is available on your system.
"""

import torch
from utils import print_device_info, get_optimal_device, configure_device_and_dtype


def main():
    """Main function to display device information."""
    print_device_info()

    # Show recommendations
    print("\n" + "="*60)
    print("Recommendations")
    print("="*60)

    device, device_type = get_optimal_device()

    if device == 'cuda':
        print("✓ CUDA detected - Optimal setup:")
        print("  device: 'cuda'")
        print("  dtype: 'bfloat16'  (recommended)")
        print("\nExpected performance:")
        print("  - Vector building: ~5-10 min for 32 samples")
        print("  - Experiment (100 trials): ~1-2 hours")

    elif device == 'mps':
        print("✓ MPS detected (Apple Silicon) - Optimal setup:")
        print("  device: 'mps'")
        print("  dtype: 'float16'  (recommended)")
        print("\nExpected performance:")
        print("  - Vector building: ~10-20 min for 32 samples")
        print("  - Experiment (100 trials): ~2-4 hours")
        print("\nNote: MPS is slower than CUDA but much faster than CPU")

    else:
        print("⚠ No GPU acceleration detected - CPU only:")
        print("  device: 'cpu'")
        print("  dtype: 'float32'  (required)")
        print("\nExpected performance:")
        print("  - Vector building: ~30-60 min for 32 samples")
        print("  - Experiment (100 trials): ~10-20 hours")
        print("\n⚠ CPU mode is very slow. Consider using a GPU if possible.")

    # Test configuration
    print("\n" + "="*60)
    print("Testing Configuration")
    print("="*60)

    try:
        test_device, test_dtype = configure_device_and_dtype('auto', 'auto')
        print(f"✓ Auto-configuration successful")
        print(f"  Selected device: {test_device}")
        print(f"  Selected dtype: {test_dtype}")

        # Test tensor operations
        print("\nTesting tensor operations...")
        test_tensor = torch.randn(100, 100)
        test_tensor = test_tensor.to(test_device)
        result = torch.matmul(test_tensor, test_tensor.T)
        print(f"✓ Matrix multiplication works on {test_device}")

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")

    print("\n" + "="*60)
    print("Usage in config file:")
    print("="*60)
    print("""
# config/experiment_config.yaml
model:
  device: "auto"  # Automatically detects best device
  dtype: "auto"   # Automatically selects optimal dtype

# Or manually specify:
model:
  device: "cuda"   # or "mps" or "cpu"
  dtype: "float16" # or "bfloat16" or "float32"
""")


if __name__ == '__main__':
    main()
