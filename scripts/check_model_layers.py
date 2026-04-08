"""
Check model layer structure and print layer information.
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoConfig

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import configure_device_and_dtype, setup_multi_gpu, get_model_size_gb


def check_model_layers(model_name: str, multi_gpu: bool = False, num_gpus: int = None):
    """
    Load model and print layer information.

    Args:
        model_name: HuggingFace model name
        multi_gpu: Whether to use multi-GPU
        num_gpus: Number of GPUs to use
    """
    print(f"Checking model: {model_name}")
    print("=" * 70)

    # Estimate model size
    model_size = get_model_size_gb(model_name)
    print(f"\nEstimated model size: {model_size:.1f} GB")

    # Load config first (lightweight)
    print("\nLoading model config...")
    config = AutoConfig.from_pretrained(model_name)

    print(f"\nModel Configuration:")
    print(f"  Model Type: {config.model_type}")
    print(f"  Hidden Size: {config.hidden_size}")
    print(f"  Number of Attention Heads: {config.num_attention_heads}")
    print(f"  Number of Hidden Layers: {config.num_hidden_layers}")
    print(f"  Vocab Size: {config.vocab_size}")

    if hasattr(config, 'num_key_value_heads'):
        print(f"  Number of KV Heads (GQA): {config.num_key_value_heads}")

    # Calculate total parameters (rough estimate)
    total_params = (
        config.vocab_size * config.hidden_size +  # Embedding
        config.num_hidden_layers * (
            4 * config.hidden_size * config.hidden_size +  # MLP
            4 * config.hidden_size * config.hidden_size    # Attention
        )
    )
    print(f"  Estimated Parameters: {total_params / 1e9:.2f}B")

    # Suggested injection layers
    num_layers = config.num_hidden_layers
    suggested_layers = [
        0,  # First layer
        num_layers // 4,  # Early (25%)
        num_layers // 2,  # Middle (50%)
        3 * num_layers // 4,  # Late (75%)
        num_layers - 1  # Last layer
    ]

    print(f"\nSuggested Injection Layers (0-{num_layers-1}):")
    print(f"  Early layers (0-{num_layers//3}): Lexical/syntactic features")
    print(f"  Middle layers ({num_layers//3}-{2*num_layers//3}): Semantic/conceptual features")
    print(f"  Late layers ({2*num_layers//3}-{num_layers-1}): Task-specific/output planning")
    print(f"\n  Recommended test layers: {suggested_layers}")

    # Load full model if requested
    print("\n" + "=" * 70)
    load_full = input("Load full model to inspect layer names? (y/n): ").strip().lower()

    if load_full == 'y':
        print("\nLoading full model...")

        device, dtype = configure_device_and_dtype('auto', 'auto')

        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16
        }

        if multi_gpu and device == 'cuda':
            print("Using multi-GPU setup...")
            max_memory = setup_multi_gpu(num_gpus, None)

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype_map[dtype],
                device_map="auto",
                max_memory=max_memory
            )
            print(f"\n✓ Model distributed across GPUs")

            if hasattr(model, 'hf_device_map'):
                print(f"\nDevice Map:")
                for name, device in model.hf_device_map.items():
                    print(f"  {name}: {device}")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype_map[dtype],
                device_map=device
            )
            print(f"\n✓ Model loaded on {device}")

        # Find layer structure
        print("\n" + "=" * 70)
        print("Layer Structure:")
        print("=" * 70)

        layers = None
        layer_attr_name = None

        # Try different model architectures
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
            layer_attr_name = 'model.model.layers'
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
            layer_attr_name = 'model.transformer.h'
        elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
            layers = model.gpt_neox.layers
            layer_attr_name = 'model.gpt_neox.layers'

        if layers is not None:
            print(f"\nFound {len(layers)} layers at: {layer_attr_name}")
            print(f"\nLayer indices: 0 to {len(layers) - 1}")

            # Print first layer structure
            print(f"\nFirst Layer Structure (Layer 0):")
            print(f"  Type: {type(layers[0]).__name__}")
            print(f"  Submodules:")
            for name, module in layers[0].named_children():
                print(f"    - {name}: {type(module).__name__}")

            # Print last layer structure
            print(f"\nLast Layer Structure (Layer {len(layers) - 1}):")
            print(f"  Type: {type(layers[-1]).__name__}")
            print(f"  Submodules:")
            for name, module in layers[-1].named_children():
                print(f"    - {name}: {type(module).__name__}")

            # Memory usage per layer (if on GPU)
            if device == 'cuda' or (multi_gpu and torch.cuda.is_available()):
                print(f"\nGPU Memory Usage:")
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1e9
                    reserved = torch.cuda.memory_reserved(i) / 1e9
                    print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        else:
            print("\nCould not automatically detect layer structure.")
            print("Model architecture:")
            print(model)

    print("\n" + "=" * 70)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Check model layer structure")
    parser.add_argument('--model', type=str,
                       default='Qwen/Qwen2.5-32B-Instruct',
                       help='Model name or path')
    parser.add_argument('--multi-gpu', action='store_true',
                       help='Use multi-GPU setup')
    parser.add_argument('--num-gpus', type=int, default=None,
                       help='Number of GPUs to use (default: all)')

    args = parser.parse_args()

    check_model_layers(args.model, args.multi_gpu, args.num_gpus)


if __name__ == '__main__':
    main()
