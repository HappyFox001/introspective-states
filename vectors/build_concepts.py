"""
Build concept vectors using contrastive pairs.
Extracts directional representations for style/persona/stance.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import yaml
import argparse


class ConceptVectorBuilder:
    """Build concept vectors through contrastive activation differences."""

    def __init__(self, model_name: str, device: str = 'cuda', dtype: str = 'bfloat16'):
        """
        Initialize builder with model.

        Args:
            model_name: HuggingFace model name
            device: Device to run on
            dtype: Data type for model
        """
        print(f"Loading model: {model_name}")

        self.model_name = model_name
        self.device = device

        # Load model
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype_map[dtype],
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Ensure padding token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

        print(f"Model loaded on {device} with dtype {dtype}")
        print(f"Number of layers: {self.model.config.num_hidden_layers}")

    def get_activation(self, prompt: str, layer_idx: int, token_position: str = 'last') -> torch.Tensor:
        """
        Get residual stream activation for a prompt.

        Args:
            prompt: Input prompt
            layer_idx: Layer index to extract from
            token_position: 'last' or 'first' token position

        Returns:
            Activation tensor of shape (hidden_dim,)
        """
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        # Forward pass with output_hidden_states
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Get hidden states for target layer
        hidden_states = outputs.hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)

        # Select token position
        if token_position == 'last':
            activation = hidden_states[0, -1, :]  # Last token
        elif token_position == 'first':
            activation = hidden_states[0, 0, :]  # First token
        else:
            raise ValueError(f"Unknown token_position: {token_position}")

        return activation.cpu()

    def build_contrastive_vector(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        layer_idx: int,
        normalization: str = 'zscore'
    ) -> Tuple[np.ndarray, Dict]:
        """
        Build concept vector through contrastive pairs.

        Args:
            positive_prompts: Prompts for positive concept (e.g., "formal")
            negative_prompts: Prompts for negative concept (e.g., "neutral")
            layer_idx: Layer to extract activations from
            normalization: 'zscore', 'unit', or 'none'

        Returns:
            Concept vector (numpy array) and metadata dict
        """
        assert len(positive_prompts) == len(negative_prompts), \
            "Must have equal number of positive and negative prompts"

        n_pairs = len(positive_prompts)
        print(f"  Building vector from {n_pairs} contrastive pairs at layer {layer_idx}...")

        deltas = []

        for pos_prompt, neg_prompt in tqdm(zip(positive_prompts, negative_prompts),
                                           total=n_pairs,
                                           desc=f"Layer {layer_idx}",
                                           leave=False):
            # Get activations
            pos_act = self.get_activation(pos_prompt, layer_idx, token_position='last')
            neg_act = self.get_activation(neg_prompt, layer_idx, token_position='last')

            # Compute difference
            delta = (pos_act - neg_act).numpy()
            deltas.append(delta)

        # Stack and aggregate
        deltas = np.stack(deltas, axis=0)  # (n_pairs, hidden_dim)
        mean_delta = np.mean(deltas, axis=0)  # (hidden_dim,)

        # Normalize
        if normalization == 'zscore':
            # Z-score normalization
            mean_delta = (mean_delta - np.mean(mean_delta)) / (np.std(mean_delta) + 1e-8)
        elif normalization == 'unit':
            # Unit normalization
            mean_delta = mean_delta / (np.linalg.norm(mean_delta) + 1e-8)
        elif normalization == 'none':
            pass
        else:
            raise ValueError(f"Unknown normalization: {normalization}")

        # Metadata
        metadata = {
            'n_pairs': n_pairs,
            'layer': layer_idx,
            'normalization': normalization,
            'mean_norm': float(np.linalg.norm(mean_delta)),
            'std_across_pairs': float(np.std(np.linalg.norm(deltas, axis=1)))
        }

        return mean_delta, metadata

    def generate_contrastive_prompts(
        self,
        base_texts: List[str],
        positive_style: str,
        negative_style: str,
        system_prompts: Dict[str, str]
    ) -> Tuple[List[str], List[str]]:
        """
        Generate contrastive prompt pairs from base texts.

        Args:
            base_texts: Neutral base texts
            positive_style: Style name for positive prompts (e.g., "formal")
            negative_style: Style name for negative prompts (e.g., "neutral")
            system_prompts: Dict mapping style names to system prompts

        Returns:
            Tuple of (positive_prompts, negative_prompts)
        """
        positive_prompts = []
        negative_prompts = []

        pos_system = system_prompts.get(positive_style, "")
        neg_system = system_prompts.get(negative_style, "")

        for text in base_texts:
            # Format prompts with system message and task
            pos_prompt = self._format_prompt(pos_system, text)
            neg_prompt = self._format_prompt(neg_system, text)

            positive_prompts.append(pos_prompt)
            negative_prompts.append(neg_prompt)

        return positive_prompts, negative_prompts

    def _format_prompt(self, system_msg: str, text: str) -> str:
        """Format prompt with system message."""
        # Use chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Please summarize: {text}"}
            ]
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return prompt
            except:
                pass

        # Fallback: simple concatenation
        return f"{system_msg}\n\nUser: Please summarize: {text}\n\nAssistant:"


def load_base_texts(data_path: Path, n_samples: int) -> List[str]:
    """Load base texts from neutral corpus."""
    texts = []

    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            texts.append(item['text'])

            if len(texts) >= n_samples:
                break

    return texts


def main():
    parser = argparse.ArgumentParser(description="Build concept vectors")
    parser.add_argument('--config', type=str, default='config/experiment_config.yaml')
    parser.add_argument('--prompts-config', type=str, default='config/prompts.yaml')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--output-dir', type=str, default='vectors')
    parser.add_argument('--model', type=str, default=None,
                        help='Override model from config')
    parser.add_argument('--concepts', type=str, nargs='+', default=None,
                        help='Specific concepts to build (default: all)')

    args = parser.parse_args()

    # Load configs
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    with open(args.prompts_config, 'r') as f:
        prompts_config = yaml.safe_load(f)

    # Override model if specified
    model_name = args.model or config['model']['name']
    device = config['model']['device']
    dtype = config['model']['dtype']

    # Initialize builder
    builder = ConceptVectorBuilder(model_name, device, dtype)

    # Load base texts
    data_path = Path(args.data_dir) / 'neutral_corpus.jsonl'
    n_samples = config['vector_extraction']['n_samples']

    print(f"\nLoading {n_samples} base texts from {data_path}...")
    base_texts = load_base_texts(data_path, n_samples)
    print(f"Loaded {len(base_texts)} texts")

    # Get system prompts
    system_prompts = prompts_config['system_prompts']

    # Determine which concepts to build
    concepts_to_build = args.concepts or list(config['concepts'].keys())

    # Build vectors for each concept
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = config['vector_extraction']['layers']
    normalization = config['vector_extraction']['normalization']

    for concept_name in concepts_to_build:
        if concept_name not in config['concepts']:
            print(f"Warning: Concept '{concept_name}' not found in config, skipping")
            continue

        concept_config = config['concepts'][concept_name]
        positive_style = concept_config['positive']
        negative_style = concept_config['negative']

        print(f"\n{'='*60}")
        print(f"Building concept: {concept_name}")
        print(f"  Positive: {positive_style}")
        print(f"  Negative: {negative_style}")
        print(f"{'='*60}")

        # Generate contrastive prompts
        pos_prompts, neg_prompts = builder.generate_contrastive_prompts(
            base_texts,
            positive_style,
            negative_style,
            system_prompts
        )

        # Build vectors for each layer
        concept_dir = output_dir / concept_name
        concept_dir.mkdir(parents=True, exist_ok=True)

        all_metadata = {
            'concept': concept_name,
            'positive': positive_style,
            'negative': negative_style,
            'model': model_name,
            'n_samples': len(base_texts),
            'layers': {}
        }

        for layer_idx in layers:
            vector, metadata = builder.build_contrastive_vector(
                pos_prompts,
                neg_prompts,
                layer_idx,
                normalization
            )

            # Save vector
            output_path = concept_dir / f'layer_{layer_idx}.npz'
            np.savez(
                output_path,
                vector=vector,
                metadata=metadata
            )

            all_metadata['layers'][layer_idx] = metadata
            print(f"  Saved layer {layer_idx} to {output_path}")
            print(f"    Norm: {metadata['mean_norm']:.4f}")

        # Save overall metadata
        metadata_path = concept_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)

        print(f"  Saved metadata to {metadata_path}")

    print("\n" + "="*60)
    print("Concept vector building complete!")
    print(f"Vectors saved in: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
