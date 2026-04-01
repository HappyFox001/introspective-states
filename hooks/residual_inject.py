"""
Residual stream injection mechanism.
Implements hooks for injecting concept vectors during model forward pass.
"""

import torch
import numpy as np
from typing import Optional, List, Tuple, Callable


class ResidualInjector:
    """Inject concept vectors into residual stream during forward pass."""

    def __init__(
        self,
        model: torch.nn.Module,
        vector: np.ndarray,
        layer_idx: int,
        alpha: float = 1.0,
        token_range: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize injector.

        Args:
            model: The language model
            vector: Concept vector to inject (numpy array)
            layer_idx: Layer index to inject at
            alpha: Injection strength multiplier
            token_range: Optional (start, end) token indices to inject over.
                        If None, injects over all tokens.
        """
        self.model = model
        self.vector = torch.tensor(vector, dtype=torch.float32)
        self.layer_idx = layer_idx
        self.alpha = alpha
        self.token_range = token_range

        self.hook_handle = None
        self._injection_count = 0

    def _injection_hook(self, module, input, output):
        """
        Hook function that injects vector into residual stream.

        Args:
            module: The layer module
            input: Input tuple to the module
            output: Output tuple from the module

        Returns:
            Modified output with injection
        """
        # Output is typically a tuple (hidden_states, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Move vector to same device and dtype as hidden states
        vector = self.vector.to(hidden_states.device).to(hidden_states.dtype)

        # Inject into specified token range
        batch_size, seq_len, hidden_dim = hidden_states.shape

        if self.token_range is not None:
            start, end = self.token_range
            # Clamp to sequence length
            start = max(0, min(start, seq_len))
            end = max(0, min(end, seq_len))

            # Inject
            hidden_states[:, start:end, :] += self.alpha * vector
        else:
            # Inject into all tokens
            hidden_states += self.alpha * vector

        self._injection_count += 1

        # Return modified output
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        else:
            return hidden_states

    def enable(self):
        """Enable injection by registering hook."""
        if self.hook_handle is not None:
            print("Warning: Hook already enabled")
            return

        # Get target layer
        # For most HF models, layers are in model.model.layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # For GPT-style models
            layers = self.model.transformer.h
        else:
            raise ValueError("Could not find layers in model")

        target_layer = layers[self.layer_idx]

        # Register forward hook
        self.hook_handle = target_layer.register_forward_hook(self._injection_hook)

        self._injection_count = 0

    def disable(self):
        """Disable injection by removing hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def __enter__(self):
        """Context manager entry."""
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disable()

    def get_injection_count(self) -> int:
        """Get number of times injection was applied."""
        return self._injection_count


class MultiLayerInjector:
    """Inject vectors across multiple layers simultaneously."""

    def __init__(
        self,
        model: torch.nn.Module,
        vectors: List[np.ndarray],
        layer_indices: List[int],
        alphas: List[float],
        token_range: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize multi-layer injector.

        Args:
            model: The language model
            vectors: List of concept vectors (one per layer)
            layer_indices: List of layer indices
            alphas: List of injection strengths (one per layer)
            token_range: Optional token range for injection
        """
        assert len(vectors) == len(layer_indices) == len(alphas), \
            "Must have equal length vectors, layers, and alphas"

        self.injectors = [
            ResidualInjector(model, vec, layer, alpha, token_range)
            for vec, layer, alpha in zip(vectors, layer_indices, alphas)
        ]

    def enable(self):
        """Enable all injectors."""
        for injector in self.injectors:
            injector.enable()

    def disable(self):
        """Disable all injectors."""
        for injector in self.injectors:
            injector.disable()

    def __enter__(self):
        """Context manager entry."""
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disable()


def load_concept_vector(concept_dir: str, layer_idx: int) -> np.ndarray:
    """
    Load a concept vector from disk.

    Args:
        concept_dir: Directory containing concept vectors
        layer_idx: Layer index to load

    Returns:
        Concept vector as numpy array
    """
    from pathlib import Path

    vector_path = Path(concept_dir) / f'layer_{layer_idx}.npz'

    if not vector_path.exists():
        raise FileNotFoundError(f"Vector file not found: {vector_path}")

    data = np.load(vector_path)
    return data['vector']


def compute_token_range_from_prompt(
    tokenizer,
    full_prompt: str,
    prefix_to_skip: str = "",
    suffix_to_skip: str = ""
) -> Tuple[int, int]:
    """
    Compute token range for injection by skipping prefix/suffix.

    Args:
        tokenizer: The tokenizer
        full_prompt: Full prompt string
        prefix_to_skip: Prefix string to skip (e.g., system prompt)
        suffix_to_skip: Suffix string to skip

    Returns:
        (start_idx, end_idx) token range
    """
    # Tokenize full prompt
    full_tokens = tokenizer.encode(full_prompt)

    start_idx = 0
    end_idx = len(full_tokens)

    # Compute prefix length
    if prefix_to_skip:
        prefix_tokens = tokenizer.encode(prefix_to_skip)
        start_idx = len(prefix_tokens)

    # Compute suffix length
    if suffix_to_skip:
        suffix_tokens = tokenizer.encode(suffix_to_skip)
        end_idx = len(full_tokens) - len(suffix_tokens)

    return start_idx, end_idx
