"""
Fetch and prepare datasets for introspection experiments.
Uses HuggingFace datasets for neutral corpus and reasoning tasks.
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset
from tqdm import tqdm
import yaml


def fetch_wikipedia_neutral_corpus(n_samples: int = 200, min_length: int = 100, max_length: int = 500, seed: int = 42) -> List[Dict]:
    """
    Fetch neutral text from Wikipedia.

    Args:
        n_samples: Number of samples to fetch
        min_length: Minimum text length in characters
        max_length: Maximum text length in characters
        seed: Random seed for reproducibility

    Returns:
        List of dicts with 'text' and 'id' fields
    """
    print(f"Fetching Wikipedia neutral corpus ({n_samples} samples)...")

    # Load Wikipedia dataset (20220301.en subset is smaller and faster)
    dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)

    random.seed(seed)
    samples = []

    for idx, item in enumerate(tqdm(dataset, desc="Scanning Wikipedia")):
        text = item['text'].strip()

        # Filter by length
        if min_length <= len(text) <= max_length:
            # Clean up the text
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            clean_text = ' '.join(lines)

            if len(clean_text) >= min_length:
                samples.append({
                    'id': f'wiki_{idx}',
                    'text': clean_text[:max_length],
                    'source': 'wikipedia'
                })

        if len(samples) >= n_samples:
            break

    print(f"Collected {len(samples)} Wikipedia samples")
    return samples


def fetch_gsm8k_reasoning(n_samples: int = 200, difficulty: str = 'easy', seed: int = 42) -> List[Dict]:
    """
    Fetch math reasoning problems from GSM8K.

    Args:
        n_samples: Number of samples to fetch
        difficulty: Filter difficulty (not directly supported, we'll filter by length)
        seed: Random seed

    Returns:
        List of dicts with 'problem', 'answer', and 'id' fields
    """
    print(f"Fetching GSM8K reasoning problems ({n_samples} samples)...")

    # Load GSM8K dataset
    dataset = load_dataset("gsm8k", "main", split="train")

    random.seed(seed)
    samples = []

    # Filter by problem complexity (use length as proxy for difficulty)
    max_problem_length = 200 if difficulty == 'easy' else 400

    for idx, item in enumerate(tqdm(dataset, desc="Processing GSM8K")):
        problem = item['question'].strip()
        answer = item['answer'].strip()

        # Filter by length
        if len(problem) <= max_problem_length:
            samples.append({
                'id': f'gsm8k_{idx}',
                'problem': problem,
                'answer': answer,
                'source': 'gsm8k'
            })

        if len(samples) >= n_samples:
            break

    # Shuffle for variety
    random.shuffle(samples)

    print(f"Collected {len(samples)} GSM8K samples")
    return samples


def fetch_common_topics(n_samples: int = 100, seed: int = 42) -> List[Dict]:
    """
    Generate common topics for prefill and intentional control experiments.

    Args:
        n_samples: Number of topics to generate
        seed: Random seed

    Returns:
        List of dicts with 'topic' field
    """
    topics = [
        "the weather today",
        "artificial intelligence",
        "climate change",
        "healthy eating",
        "space exploration",
        "renewable energy",
        "education systems",
        "mental health",
        "urban planning",
        "ocean conservation",
        "scientific research",
        "digital privacy",
        "sustainable agriculture",
        "medical innovation",
        "transportation systems",
        "cultural diversity",
        "economic development",
        "environmental protection",
        "technological progress",
        "social equality",
    ]

    random.seed(seed)

    # Extend with variations if needed
    samples = []
    for i in range(n_samples):
        topic = topics[i % len(topics)]
        samples.append({
            'id': f'topic_{i}',
            'topic': topic
        })

    print(f"Generated {len(samples)} topic samples")
    return samples


def save_dataset(data: List[Dict], output_path: Path):
    """Save dataset to JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Saved {len(data)} items to {output_path}")


def main():
    """Main function to fetch all datasets."""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch datasets for introspection experiments")
    parser.add_argument('--config', type=str, default='config/experiment_config.yaml',
                        help='Path to experiment config')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory for datasets')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)

    # Fetch neutral corpus
    if 'neutral_corpus' in config['data']:
        corpus_config = config['data']['neutral_corpus']
        corpus_data = fetch_wikipedia_neutral_corpus(
            n_samples=corpus_config.get('n_samples', 200),
            min_length=corpus_config.get('min_length', 100),
            max_length=corpus_config.get('max_length', 500),
            seed=args.seed
        )
        save_dataset(corpus_data, output_dir / 'neutral_corpus.jsonl')

    # Fetch reasoning problems
    if 'step_reasoning' in config['data']:
        reasoning_config = config['data']['step_reasoning']
        reasoning_data = fetch_gsm8k_reasoning(
            n_samples=reasoning_config.get('n_samples', 200),
            difficulty=reasoning_config.get('difficulty', 'easy'),
            seed=args.seed
        )
        save_dataset(reasoning_data, output_dir / 'step_reasoning.jsonl')

    # Generate topics for prefill/control experiments
    topics_data = fetch_common_topics(n_samples=100, seed=args.seed)
    save_dataset(topics_data, output_dir / 'topics.jsonl')

    print("\nDataset preparation complete!")
    print(f"Files saved in: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
