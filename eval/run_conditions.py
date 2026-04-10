"""
Run experimental conditions (C0-C4) for introspection evaluation.
"""

import json
import torch
import yaml
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from hooks.residual_inject import ResidualInjector, load_concept_vector
from utils import configure_device_and_dtype, print_device_info, setup_multi_gpu, get_model_size_gb


class IntrospectionExperiment:
    """Run introspection experiments with different conditions."""

    def __init__(
        self,
        model_name: str,
        config: Dict,
        prompts_config: Dict,
        device: str = 'auto',
        multi_gpu: bool = False,
        num_gpus: int = None,
        max_memory_per_gpu: str = None
    ):
        """
        Initialize experiment runner.

        Args:
            model_name: HuggingFace model name
            config: Experiment configuration dict
            prompts_config: Prompts configuration dict
            device: Device to run on ('auto', 'cuda', 'mps', or 'cpu')
            multi_gpu: Enable multi-GPU model parallelism
            num_gpus: Number of GPUs to use (None = all)
            max_memory_per_gpu: Max memory per GPU (e.g., "22GB")
        """
        print(f"Initializing experiment with model: {model_name}")

        # Estimate model size
        model_size = get_model_size_gb(model_name)
        print(f"Estimated model size: {model_size:.1f} GB")

        # Configure device and dtype
        self.device, dtype = configure_device_and_dtype(
            device,
            config['model'].get('dtype', 'auto')
        )

        self.multi_gpu = multi_gpu

        self.model_name = model_name
        self.config = config
        self.prompts_config = prompts_config

        # Load model and tokenizer
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16
        }

        # Multi-GPU setup for large models
        if multi_gpu and self.device == 'cuda':
            print("Enabling multi-GPU model parallelism...")
            max_memory = setup_multi_gpu(num_gpus, max_memory_per_gpu)

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype_map[dtype],
                device_map="auto",  # Automatic distribution
                max_memory=max_memory
            )
            print(f"✓ Model distributed across {num_gpus or torch.cuda.device_count()} GPUs")

        # Single GPU or MPS
        elif self.device == 'mps':
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype_map[dtype]
            )
            self.model = self.model.to('mps')

        # Single GPU or CPU
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype_map[dtype],
                device_map=self.device
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

        print(f"✓ Model loaded with dtype {dtype}")
        print(f"✓ Number of layers: {self.model.config.num_hidden_layers}")

        # Print device map for multi-GPU
        if multi_gpu and hasattr(self.model, 'hf_device_map'):
            print(f"✓ Device map: {self.model.hf_device_map}")

    def format_prompt(
        self,
        task_content: str,
        system_style: str,
        include_introspection: bool = True,
        concept_name: str = None
    ) -> str:
        """
        Format prompt with system message and optional introspection probe.

        Args:
            task_content: Main task content
            system_style: Style for system prompt ('formal', 'neutral', etc.)
            include_introspection: Whether to include introspection prompt
            concept_name: Concept name for binary introspection (e.g., 'formal_neutral')

        Returns:
            Formatted prompt string
        """
        # Get system prompt
        system_prompt = self.prompts_config['system_prompts'].get(
            system_style,
            self.prompts_config['system_prompts']['neutral']
        )

        # Get introspection prompt
        introspection_prompt = ""
        if include_introspection:
            # Use concept-specific binary introspection if available
            if concept_name and 'binary_introspection' in self.prompts_config['introspection_prompts']:
                binary_prompts = self.prompts_config['introspection_prompts']['binary_introspection']
                if concept_name in binary_prompts:
                    introspection_prompt = binary_prompts[concept_name]
                else:
                    # Fallback to general introspection
                    introspection_prompt = self.prompts_config['introspection_prompts']['full_introspection']
            else:
                introspection_prompt = self.prompts_config['introspection_prompts']['full_introspection']

        # Combine
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{introspection_prompt}\n\n{task_content}"}
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

        # Fallback
        full_content = f"{introspection_prompt}\n\n{task_content}" if include_introspection else task_content
        return f"{system_prompt}\n\nUser: {full_content}\n\nAssistant:"

    def generate_with_injection(
        self,
        prompt: str,
        concept_vector: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
        alpha: float = 1.0,
        temperature: float = 1.0,
        max_new_tokens: int = 512
    ) -> str:
        """
        Generate text with optional concept injection.

        Args:
            prompt: Input prompt
            concept_vector: Optional concept vector to inject
            layer_idx: Layer to inject at
            alpha: Injection strength
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        # Setup injection if provided
        injector = None
        if concept_vector is not None and layer_idx is not None:
            injector = ResidualInjector(
                self.model,
                concept_vector.cpu().numpy(),
                layer_idx,
                alpha=alpha
            )
            injector.enable()

        try:
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            return generated_text

        finally:
            # Clean up injection
            if injector is not None:
                injector.disable()

    def generate_batch_with_injection(
        self,
        prompts: List[str],
        concept_vector: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
        alpha: float = 1.0,
        temperature: float = 1.0,
        max_new_tokens: int = 512
    ) -> List[str]:
        """
        Generate text for a batch of prompts with the same injection parameters.

        NOTE: All prompts in the batch share the same injection (same vector, layer, alpha).
        For different injection parameters, call this function separately.

        Args:
            prompts: List of input prompts
            concept_vector: Optional concept vector to inject (same for all)
            layer_idx: Layer to inject at (same for all)
            alpha: Injection strength (same for all)
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate

        Returns:
            List of generated texts
        """
        # Tokenize batch with padding
        inputs = self.tokenizer(
            prompts,
            padding=True,
            return_tensors='pt',
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Setup injection if provided
        injector = None
        if concept_vector is not None and layer_idx is not None:
            injector = ResidualInjector(
                self.model,
                concept_vector.cpu().numpy(),
                layer_idx,
                alpha=alpha
            )
            injector.enable()

        try:
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Decode all outputs
            generated_texts = []
            for i in range(len(prompts)):
                # Find where the input ends for this sample
                # (need to handle variable length inputs due to padding)
                input_ids = inputs['input_ids'][i]
                # Find first pad token or use full length
                pad_mask = input_ids == self.tokenizer.pad_token_id
                if pad_mask.any():
                    input_length = pad_mask.nonzero()[0].item()
                else:
                    input_length = len(input_ids)

                # Decode only the generated part
                generated_tokens = outputs[i][input_length:]
                generated_text = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True
                )
                generated_texts.append(generated_text)

            return generated_texts

        finally:
            # Clean up injection
            if injector is not None:
                injector.disable()

    def run_single_trial(
        self,
        task_data: Dict,
        condition: str,
        concept_name: str,
        layer_idx: int,
        alpha: float,
        vector_dir: Path
    ) -> Dict:
        """
        Run a single trial under specified condition.

        Args:
            task_data: Task data dict (with 'text' or 'problem' field)
            condition: Condition name ('C0', 'C1', etc.)
            concept_name: Concept to test (e.g., 'formal_neutral')
            layer_idx: Layer to inject at
            alpha: Injection strength
            vector_dir: Directory containing concept vectors

        Returns:
            Result dict with output and metadata
        """
        condition_config = self.config['conditions'][condition]

        # Get task content
        if 'text' in task_data:
            task_content = self.prompts_config['neutral_corpus_task'].format(
                text=task_data['text']
            )
        elif 'problem' in task_data:
            task_content = self.prompts_config['step_reasoning_task'].format(
                problem=task_data['problem']
            )
        else:
            raise ValueError("Task data must have 'text' or 'problem' field")

        # Determine external style
        concept_config = self.config['concepts'][concept_name]
        positive_style = concept_config['positive']
        negative_style = concept_config['negative']

        external_style_map = {
            'neutral': 'neutral',
            'target': positive_style,
            'opposite': negative_style
        }

        external_style = external_style_map[condition_config['external_style']]

        # Format prompt
        prompt = self.format_prompt(
            task_content,
            external_style,
            include_introspection=True,
            concept_name=concept_name
        )

        # Load concept vector if needed
        concept_vector = None
        if condition_config['inject']:
            inject_concept = condition_config['inject_concept']

            if inject_concept == 'target':
                # Load positive concept vector
                vector_path = vector_dir / concept_name / f'layer_{layer_idx}.npz'
                data = torch.load(vector_path) if vector_path.suffix == '.pt' else None

                if data is None:
                    import numpy as np
                    data = np.load(vector_path)
                    concept_vector = torch.tensor(data['vector'])
                else:
                    concept_vector = data
            else:
                raise ValueError(f"Unknown inject_concept: {inject_concept}")

        # Generate
        output = self.generate_with_injection(
            prompt,
            concept_vector,
            layer_idx if condition_config['inject'] else None,
            alpha,
            temperature=self.config['evaluation']['temperature'],
            max_new_tokens=self.config['evaluation']['max_new_tokens']
        )

        # Return result
        result = {
            'task_id': task_data['id'],
            'condition': condition,
            'concept': concept_name,
            'layer': layer_idx,
            'alpha': alpha,
            'external_style': external_style,
            'injected': condition_config['inject'],
            'prompt': prompt,
            'output': output,
            'task_data': task_data
        }

        return result

    def run_experiment(
        self,
        task_dataset: List[Dict],
        conditions: List[str],
        concepts: List[str],
        layers: List[int],
        alphas: List[float],
        vector_dir: Path,
        output_path: Path,
        n_trials_per_condition: Optional[int] = None,
        batch_size: int = None
    ):
        """
        Run full experiment across conditions with batched generation.

        Args:
            task_dataset: List of task data dicts
            conditions: List of condition names to run
            concepts: List of concept names to test
            layers: List of layer indices
            alphas: List of alpha values
            vector_dir: Directory containing concept vectors
            output_path: Path to save results
            n_trials_per_condition: Max trials per condition (None = all)
            batch_size: Batch size for generation (None = use config or 1)
        """
        # Limit trials if specified
        if n_trials_per_condition is not None:
            task_dataset = task_dataset[:n_trials_per_condition]

        # Get batch size from config if not specified
        if batch_size is None:
            batch_size = self.config.get('evaluation', {}).get('parallel', {}).get('batch_size', 1)

        # Count total trials (accounting for skipped non-injection conditions)
        total_trials = 0
        for _ in task_dataset:
            for condition in conditions:
                for _ in concepts:
                    for layer in layers:
                        for alpha in alphas:
                            if not self.config['conditions'][condition]['inject']:
                                if alpha != 0.0 or layer != layers[0]:
                                    continue
                            total_trials += 1

        print(f"\nRunning {total_trials} trials with batch_size={batch_size}:")
        print(f"  Tasks: {len(task_dataset)}")
        print(f"  Conditions: {conditions}")
        print(f"  Concepts: {concepts}")
        print(f"  Layers: {layers}")
        print(f"  Alphas: {alphas}")
        if batch_size > 1:
            print(f"  Estimated speedup: {batch_size}x")
            print(f"  Estimated time: {total_trials * 7 / batch_size / 60:.1f} minutes (vs {total_trials * 7 / 60:.1f} minutes)")

        # Prepare output file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Reorganize trials by (condition, concept, layer, alpha) for batching
        # This groups trials with the same injection parameters together
        trial_groups = {}
        for condition in conditions:
            for concept in concepts:
                for layer in layers:
                    for alpha in alphas:
                        # Skip injection params for non-injection conditions
                        if not self.config['conditions'][condition]['inject']:
                            if alpha != 0.0 or layer != layers[0]:
                                continue

                        key = (condition, concept, layer, alpha)
                        trial_groups[key] = []

                        for task_data in task_dataset:
                            trial_groups[key].append(task_data)

        # Run trials in batches
        with open(output_path, 'w') as f:
            pbar = tqdm(total=total_trials, desc="Running trials")

            for (condition, concept, layer, alpha), task_list in trial_groups.items():
                # Load concept vector once per group
                condition_config = self.config['conditions'][condition]
                concept_vector = None

                if condition_config['inject']:
                    vector_path = vector_dir / concept / f'layer_{layer}.npz'
                    try:
                        import numpy as np
                        data = np.load(vector_path)
                        concept_vector = torch.tensor(data['vector'])
                    except Exception as e:
                        print(f"\nWarning: Could not load vector for {concept} layer {layer}: {e}")

                # Process tasks in batches
                for batch_start in range(0, len(task_list), batch_size):
                    batch_end = min(batch_start + batch_size, len(task_list))
                    batch_tasks = task_list[batch_start:batch_end]

                    # Prepare prompts for this batch
                    batch_prompts = []
                    batch_metadata = []

                    for task_data in batch_tasks:
                        # Get task content
                        if 'text' in task_data:
                            task_content = self.prompts_config['neutral_corpus_task'].format(
                                text=task_data['text']
                            )
                        elif 'problem' in task_data:
                            task_content = self.prompts_config['step_reasoning_task'].format(
                                problem=task_data['problem']
                            )
                        else:
                            continue

                        # Determine external style
                        concept_config = self.config['concepts'][concept]
                        positive_style = concept_config['positive']
                        negative_style = concept_config['negative']

                        external_style_map = {
                            'neutral': 'neutral',
                            'target': positive_style,
                            'opposite': negative_style
                        }
                        external_style = external_style_map[condition_config['external_style']]

                        # Format prompt
                        prompt = self.format_prompt(
                            task_content,
                            external_style,
                            include_introspection=True,
                            concept_name=concept
                        )

                        batch_prompts.append(prompt)
                        batch_metadata.append({
                            'task_data': task_data,
                            'external_style': external_style
                        })

                    if not batch_prompts:
                        continue

                    try:
                        # Generate batch
                        if len(batch_prompts) == 1 or batch_size == 1:
                            # Single trial - use original method
                            outputs = [self.generate_with_injection(
                                batch_prompts[0],
                                concept_vector,
                                layer if condition_config['inject'] else None,
                                alpha,
                                temperature=self.config['evaluation']['temperature'],
                                max_new_tokens=self.config['evaluation']['max_new_tokens']
                            )]
                        else:
                            # Batch generation
                            outputs = self.generate_batch_with_injection(
                                batch_prompts,
                                concept_vector,
                                layer if condition_config['inject'] else None,
                                alpha,
                                temperature=self.config['evaluation']['temperature'],
                                max_new_tokens=self.config['evaluation']['max_new_tokens']
                            )

                        # Write results
                        for i, (output, metadata) in enumerate(zip(outputs, batch_metadata)):
                            result = {
                                'task_id': metadata['task_data']['id'],
                                'condition': condition,
                                'concept': concept,
                                'layer': layer,
                                'alpha': alpha,
                                'external_style': metadata['external_style'],
                                'injected': condition_config['inject'],
                                'prompt': batch_prompts[i],
                                'output': output,
                                'task_data': metadata['task_data']
                            }
                            f.write(json.dumps(result) + '\n')

                        f.flush()

                    except Exception as e:
                        print(f"\nError in batch ({condition}, {concept}, layer {layer}, alpha {alpha}): {e}")
                        import traceback
                        traceback.print_exc()

                    finally:
                        pbar.update(len(batch_prompts))

            pbar.close()

        print(f"\nResults saved to: {output_path}")


def load_task_dataset(data_path: Path, task_type: str) -> List[Dict]:
    """Load task dataset from JSONL file."""
    dataset = []

    with open(data_path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))

    print(f"Loaded {len(dataset)} samples from {data_path}")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Run introspection experimental conditions")
    parser.add_argument('--config', type=str, default='config/experiment_config.yaml')
    parser.add_argument('--prompts-config', type=str, default='config/prompts.yaml')
    parser.add_argument('--task', type=str, choices=['neutral_corpus', 'step_reasoning'],
                        required=True, help='Task type')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Override data path')
    parser.add_argument('--vector-dir', type=str, default='vectors',
                        help='Directory containing concept vectors')
    parser.add_argument('--output-dir', type=str, default='output/json',
                        help='Output directory')
    parser.add_argument('--conditions', type=str, nargs='+',
                        default=['C0', 'C1', 'C2', 'C3', 'C4'],
                        help='Conditions to run')
    parser.add_argument('--concepts', type=str, nargs='+', default=None,
                        help='Concepts to test (default: all)')
    parser.add_argument('--n-trials', type=int, default=None,
                        help='Number of trials per condition (default: all)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for generation (default: from config, typically 4-8)')

    args = parser.parse_args()

    # Load configs
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    with open(args.prompts_config, 'r') as f:
        prompts_config = yaml.safe_load(f)

    # Load task dataset
    if args.data_path is None:
        data_filename = 'neutral_corpus.jsonl' if args.task == 'neutral_corpus' else 'step_reasoning.jsonl'
        data_path = Path('data') / data_filename
    else:
        data_path = Path(args.data_path)

    task_dataset = load_task_dataset(data_path, args.task)

    # Setup experiment
    model_name = config['model']['name']
    device = config['model']['device']

    # Multi-GPU settings
    multi_gpu_config = config['model'].get('multi_gpu', {})
    multi_gpu = multi_gpu_config.get('enabled', False)
    num_gpus = multi_gpu_config.get('num_gpus', None)
    max_memory = multi_gpu_config.get('max_memory_per_gpu', None)

    experiment = IntrospectionExperiment(
        model_name,
        config,
        prompts_config,
        device,
        multi_gpu=multi_gpu,
        num_gpus=num_gpus,
        max_memory_per_gpu=max_memory
    )

    # Determine concepts to test
    concepts = args.concepts or list(config['concepts'].keys())

    # Determine layers and alphas
    layers = config['injection']['layers']
    alphas = config['injection']['alphas']

    # Override n_trials if specified
    n_trials = args.n_trials or config['evaluation']['n_trials_per_condition']

    # Output path
    output_path = Path(args.output_dir) / f"{args.task}_results.jsonl"

    # Run experiment
    experiment.run_experiment(
        task_dataset,
        args.conditions,
        concepts,
        layers,
        alphas,
        Path(args.vector_dir),
        output_path,
        n_trials,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
