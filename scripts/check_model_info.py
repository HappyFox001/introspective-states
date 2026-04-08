"""
Quick check of model layer information (config only, no model loading).
Useful for verifying layer settings before running experiments.
"""

import argparse
from transformers import AutoConfig


def check_model_info(model_name: str):
    """
    Check model configuration and suggest injection layers.

    Args:
        model_name: HuggingFace model name
    """
    print(f"Checking model: {model_name}")
    print("=" * 70)

    # Load config (lightweight, no model weights)
    print("\nLoading model config...")
    config = AutoConfig.from_pretrained(model_name)

    print(f"\n{'Model Configuration':^70}")
    print("=" * 70)
    print(f"  Model Type: {config.model_type}")
    print(f"  Hidden Size: {config.hidden_size}")
    print(f"  Number of Attention Heads: {config.num_attention_heads}")
    print(f"  Number of Hidden Layers: {config.num_hidden_layers}")
    print(f"  Vocab Size: {config.vocab_size}")

    if hasattr(config, 'num_key_value_heads'):
        print(f"  Number of KV Heads (GQA): {config.num_key_value_heads}")

    if hasattr(config, 'intermediate_size'):
        print(f"  MLP Intermediate Size: {config.intermediate_size}")

    # Estimate parameters
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size

    # Rough estimate
    embedding_params = vocab_size * hidden_size
    attention_params = num_layers * 4 * hidden_size * hidden_size
    mlp_params = num_layers * 8 * hidden_size * hidden_size  # Assuming 4x expansion

    total_params = embedding_params + attention_params + mlp_params
    print(f"  Estimated Parameters: {total_params / 1e9:.2f}B")
    print(f"  Estimated Size (FP16): {total_params * 2 / 1e9:.2f} GB")
    print(f"  Estimated Size (BF16): {total_params * 2 / 1e9:.2f} GB")

    # Suggest injection layers
    print(f"\n{'Suggested Injection Layers':^70}")
    print("=" * 70)

    early_layer = num_layers // 4
    middle_layer = num_layers // 2
    late_layer = 3 * num_layers // 4
    last_layer = num_layers - 1

    print(f"\nLayer ranges (total {num_layers} layers, indexed 0-{last_layer}):")
    print(f"  Early layers   (0-{num_layers//3:2d}): Lexical/syntactic processing")
    print(f"  Middle layers ({num_layers//3:2d}-{2*num_layers//3:2d}): Semantic/conceptual representations")
    print(f"  Late layers   ({2*num_layers//3:2d}-{last_layer:2d}): Task-specific/output planning")

    suggested_layers = [0, early_layer, middle_layer, late_layer, last_layer]
    print(f"\nRecommended test layers: {suggested_layers}")
    print(f"  Layer  0: First layer (baseline)")
    print(f"  Layer {early_layer:2d}: Early processing (~25%)")
    print(f"  Layer {middle_layer:2d}: Middle layer (~50%) - often best for concepts")
    print(f"  Layer {late_layer:2d}: Late processing (~75%)")
    print(f"  Layer {last_layer:2d}: Last layer (output)")

    # Generate config snippet
    print(f"\n{'Config Snippet for experiment_config.yaml':^70}")
    print("=" * 70)
    print("injection:")
    print(f"  layers: {suggested_layers}")
    print(f"  alphas: [0.0, 0.5, 1.0, 2.0, 4.0]")
    print("")
    print("vector_extraction:")
    print(f"  layers: {suggested_layers}")

    print("\n" + "=" * 70)
    print("Done! Copy the config snippet above to your config file.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Quick check of model layer information (config only)"
    )
    parser.add_argument(
        '--model', type=str,
        default='Qwen/Qwen2.5-32B-Instruct',
        help='Model name or path'
    )

    args = parser.parse_args()

    check_model_info(args.model)


if __name__ == '__main__':
    main()
