"""
Run prefill intentionality attribution experiment.
Tests whether models can distinguish intentional vs. prefilled outputs.
"""

import json
import torch
import yaml
import argparse
import sys
import random
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent))
from hooks.residual_inject import ResidualInjector
from utils import configure_device_and_dtype, print_device_info


class PrefillExperiment:
    """Run prefill intentionality attribution experiments."""

    def __init__(self, model_name: str, config: Dict, prompts_config: Dict, device: str = 'auto'):
        """Initialize experiment."""
        print(f"Loading model: {model_name}")

        # Configure device and dtype
        self.device, dtype = configure_device_and_dtype(
            device,
            config['model'].get('dtype', 'auto')
        )

        self.model_name = model_name
        self.config = config
        self.prompts_config = prompts_config

        # Load model
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16
        }

        # For MPS, use device_map='auto' instead of 'mps'
        if self.device == 'mps':
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype_map[dtype]
            )
            self.model = self.model.to('mps')
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
        print(f"✓ Model loaded on {self.device}")

    def run_prefill_trial(
        self,
        topic: str,
        prefill_text: str,
        inject_vector: torch.Tensor = None,
        layer_idx: int = None,
        alpha: float = 1.0
    ) -> Dict:
        """
        Run a single prefill trial.

        Args:
            topic: Topic to write about
            prefill_text: Text to prefill as "model's output"
            inject_vector: Optional vector to inject (to make prefill seem intentional)
            layer_idx: Layer to inject at
            alpha: Injection strength

        Returns:
            Result dict
        """
        # Step 1: Initial request
        initial_prompt = self.prompts_config['prefill_prompts']['initial_request'].format(
            topic=topic
        )

        # Format with chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": initial_prompt}]
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                prompt = f"User: {initial_prompt}\n\nAssistant:"
        else:
            prompt = f"User: {initial_prompt}\n\nAssistant:"

        # Add prefilled text
        full_context = prompt + " " + prefill_text

        # Step 2: Intentionality check prompt
        check_prompt = self.prompts_config['prefill_prompts']['intentionality_check'].format(
            previous_output=prefill_text
        )

        # Format check prompt
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "user", "content": initial_prompt},
                {"role": "assistant", "content": prefill_text},
                {"role": "user", "content": check_prompt}
            ]
            try:
                check_full_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                check_full_prompt = f"{full_context}\n\nUser: {check_prompt}\n\nAssistant:"
        else:
            check_full_prompt = f"{full_context}\n\nUser: {check_prompt}\n\nAssistant:"

        # Setup injection if provided
        injector = None
        if inject_vector is not None and layer_idx is not None:
            injector = ResidualInjector(
                self.model,
                inject_vector.cpu().numpy(),
                layer_idx,
                alpha=alpha
            )
            injector.enable()

        try:
            # Generate response to intentionality check
            inputs = self.tokenizer(check_full_prompt, return_tensors='pt').to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=1.0,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            return {
                'topic': topic,
                'prefill_text': prefill_text,
                'injected': inject_vector is not None,
                'layer': layer_idx if inject_vector is not None else None,
                'alpha': alpha if inject_vector is not None else None,
                'response': response
            }

        finally:
            if injector is not None:
                injector.disable()

    def generate_prefill_pairs(self, topics: List[Dict], n_pairs: int) -> List[Dict]:
        """
        Generate topic-prefill pairs for experiment.

        Args:
            topics: List of topic dicts
            n_pairs: Number of pairs to generate

        Returns:
            List of (topic, prefill_text) pairs
        """
        # Prefill templates for different styles
        prefill_templates = {
            'formal': "The {topic} represents a significant aspect of contemporary society.",
            'casual': "So, {topic} is pretty interesting when you think about it.",
            'technical': "From a technical perspective, {topic} involves multiple interrelated factors."
        }

        pairs = []

        for i in range(n_pairs):
            topic_data = topics[i % len(topics)]
            topic = topic_data['topic']

            # Randomly select a prefill style
            style = random.choice(list(prefill_templates.keys()))
            prefill_text = prefill_templates[style].format(topic=topic)

            pairs.append({
                'topic': topic,
                'prefill_text': prefill_text,
                'prefill_style': style
            })

        return pairs


def load_topics(data_path: Path) -> List[Dict]:
    """Load topics from JSONL file."""
    topics = []

    with open(data_path, 'r') as f:
        for line in f:
            topics.append(json.loads(line))

    return topics


def main():
    parser = argparse.ArgumentParser(description="Run prefill intentionality experiment")
    parser.add_argument('--config', type=str, default='config/experiment_config.yaml')
    parser.add_argument('--prompts-config', type=str, default='config/prompts.yaml')
    parser.add_argument('--topics', type=str, default='data/topics.jsonl')
    parser.add_argument('--vector-dir', type=str, default='vectors')
    parser.add_argument('--output-dir', type=str, default='output/json')
    parser.add_argument('--concept', type=str, default='formal_neutral',
                        help='Concept to use for injection')
    parser.add_argument('--layer', type=int, default=8,
                        help='Layer to inject at')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Injection strength')
    parser.add_argument('--n-pairs', type=int, default=50,
                        help='Number of topic-prefill pairs to test')

    args = parser.parse_args()

    # Load configs
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    with open(args.prompts_config, 'r') as f:
        prompts_config = yaml.safe_load(f)

    # Load topics
    topics = load_topics(Path(args.topics))
    print(f"Loaded {len(topics)} topics")

    # Initialize experiment
    model_name = config['model']['name']
    device = config['model']['device']

    experiment = PrefillExperiment(model_name, config, prompts_config, device)

    # Generate prefill pairs
    print(f"Generating {args.n_pairs} topic-prefill pairs...")
    pairs = experiment.generate_prefill_pairs(topics, args.n_pairs)

    # Load concept vector
    import numpy as np
    vector_path = Path(args.vector_dir) / args.concept / f'layer_{args.layer}.npz'
    vector_data = np.load(vector_path)
    concept_vector = torch.tensor(vector_data['vector'])

    # Run experiment
    output_path = Path(args.output_dir) / f'prefill_{args.concept}_results.jsonl'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning prefill experiment...")
    print(f"  Concept: {args.concept}")
    print(f"  Layer: {args.layer}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Pairs: {args.n_pairs}")

    with open(output_path, 'w') as f:
        for pair in tqdm(pairs, desc="Running trials"):
            # Condition 1: No injection (baseline - should apologize/disavow)
            result_no_inject = experiment.run_prefill_trial(
                pair['topic'],
                pair['prefill_text'],
                inject_vector=None
            )
            result_no_inject['condition'] = 'no_inject'
            result_no_inject['prefill_style'] = pair['prefill_style']
            f.write(json.dumps(result_no_inject) + '\n')

            # Condition 2: With injection (should accept as intentional)
            result_inject = experiment.run_prefill_trial(
                pair['topic'],
                pair['prefill_text'],
                inject_vector=concept_vector,
                layer_idx=args.layer,
                alpha=args.alpha
            )
            result_inject['condition'] = 'inject'
            result_inject['prefill_style'] = pair['prefill_style']
            f.write(json.dumps(result_inject) + '\n')

            f.flush()

    print(f"\nResults saved to: {output_path}")
    print("\nNext step: Grade results to compute apology rate")


if __name__ == '__main__':
    main()
