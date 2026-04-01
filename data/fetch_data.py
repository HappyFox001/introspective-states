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

    random.seed(seed)
    samples = []

    # Try multiple data sources in order of preference
    data_sources = [
        # Option 1: Use wikimedia/wikipedia (new format)
        ("wikimedia/wikipedia", "20231101.en", "text"),
        # Option 2: Use c4 (cleaned Common Crawl)
        ("allenai/c4", "en", "text"),
        # Option 3: Use simple wikipedia
        ("wikimedia/wikipedia", "20231101.simple", "text"),
    ]

    dataset = None
    for dataset_name, config_name, text_field in data_sources:
        try:
            print(f"  Trying {dataset_name} ({config_name})...")
            dataset = load_dataset(
                dataset_name,
                config_name,
                split="train",
                streaming=True,
                trust_remote_code=False
            )
            print(f"  ✓ Successfully loaded {dataset_name}")
            break
        except Exception as e:
            print(f"  ✗ Failed to load {dataset_name}: {e}")
            continue

    # Fallback: Use built-in sample texts if all datasets fail
    if dataset is None:
        print("  All remote datasets failed, using built-in sample texts...")
        return generate_builtin_neutral_texts(n_samples, min_length, max_length, seed)

    # Collect samples from dataset
    try:
        for idx, item in enumerate(tqdm(dataset, desc="Scanning dataset", total=n_samples*5)):
            # Handle different field names
            text = item.get('text') or item.get('title', '') + ' ' + item.get('text', '')
            text = text.strip()

            # Filter by length
            if min_length <= len(text) <= max_length * 2:
                # Clean up the text
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                clean_text = ' '.join(lines)

                if len(clean_text) >= min_length:
                    samples.append({
                        'id': f'corpus_{idx}',
                        'text': clean_text[:max_length],
                        'source': dataset_name if dataset else 'builtin'
                    })

            if len(samples) >= n_samples:
                break

            # Safety limit
            if idx > n_samples * 50:
                print(f"  Warning: Scanned {idx} items but only found {len(samples)} valid samples")
                break

    except Exception as e:
        print(f"  Error while collecting samples: {e}")
        if len(samples) < n_samples:
            print("  Falling back to built-in texts...")
            return generate_builtin_neutral_texts(n_samples, min_length, max_length, seed)

    print(f"Collected {len(samples)} neutral corpus samples")
    return samples


def generate_builtin_neutral_texts(n_samples: int = 200, min_length: int = 100, max_length: int = 500, seed: int = 42) -> List[Dict]:
    """
    Generate neutral text samples from built-in templates.
    Fallback when online datasets are not accessible.

    Args:
        n_samples: Number of samples to generate
        min_length: Minimum text length
        max_length: Maximum text length
        seed: Random seed

    Returns:
        List of text samples
    """
    random.seed(seed)

    # Sample neutral topics and templates
    topics = [
        "climate change", "renewable energy", "space exploration", "artificial intelligence",
        "biodiversity", "urban planning", "public health", "education systems",
        "scientific research", "technological innovation", "economic development",
        "cultural heritage", "environmental conservation", "medical science",
        "transportation systems", "agricultural practices", "ocean ecosystems",
        "geological processes", "atmospheric phenomena", "historical events"
    ]

    templates = [
        "The study of {topic} has evolved significantly over recent decades. Researchers have identified multiple factors that influence outcomes in this field. Contemporary approaches emphasize evidence-based methods and systematic analysis. Various institutions worldwide contribute to advancing knowledge in this domain.",

        "{topic} represents an important area of inquiry in modern science. Multiple disciplines contribute insights through empirical observation and theoretical modeling. Current understanding reflects decades of accumulated research across different contexts. Ongoing investigations continue to refine existing frameworks.",

        "Research on {topic} encompasses diverse methodological approaches. Scientists employ both qualitative and quantitative techniques to examine relevant phenomena. Data collection occurs across multiple settings and time scales. Findings inform policy decisions and practical applications in various sectors.",

        "The field of {topic} involves systematic investigation of complex systems. Scholars analyze patterns, mechanisms, and outcomes using rigorous protocols. Evidence accumulates through peer-reviewed studies and collaborative research initiatives. Results are disseminated through academic publications and conferences.",

        "{topic} has garnered attention from researchers worldwide. Investigations span multiple levels of analysis and temporal scales. Methodologies include observational studies, controlled experiments, and computational modeling. Conclusions are subject to ongoing validation and refinement."
    ]

    samples = []
    for i in range(n_samples):
        topic = random.choice(topics)
        template = random.choice(templates)
        text = template.format(topic=topic.title())

        # Add some variation
        sentences = text.split('. ')
        num_sentences = random.randint(2, 4)
        selected_sentences = sentences[:num_sentences]
        final_text = '. '.join(selected_sentences)
        if not final_text.endswith('.'):
            final_text += '.'

        samples.append({
            'id': f'builtin_{i}',
            'text': final_text,
            'source': 'builtin'
        })

    print(f"Generated {len(samples)} built-in neutral text samples")
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

    random.seed(seed)
    samples = []

    try:
        # Try to load GSM8K dataset
        print("  Loading GSM8K dataset from HuggingFace...")
        dataset = load_dataset("openai/gsm8k", "main", split="train", trust_remote_code=False)
        print("  ✓ Successfully loaded GSM8K")

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

    except Exception as e:
        print(f"  ✗ Failed to load GSM8K: {e}")
        print("  Falling back to built-in reasoning problems...")
        samples = generate_builtin_reasoning_problems(n_samples, seed)

    print(f"Collected {len(samples)} reasoning problem samples")
    return samples


def generate_builtin_reasoning_problems(n_samples: int = 200, seed: int = 42) -> List[Dict]:
    """
    Generate simple reasoning problems as fallback.

    Args:
        n_samples: Number of problems to generate
        seed: Random seed

    Returns:
        List of problem dicts
    """
    random.seed(seed)

    problem_templates = [
        {
            "template": "A store has {a} apples. They sell {b} apples in the morning and {c} apples in the afternoon. How many apples are left?",
            "answer": lambda a, b, c: a - b - c
        },
        {
            "template": "John walks {a} kilometers per day. How many kilometers does he walk in {b} days?",
            "answer": lambda a, b: a * b
        },
        {
            "template": "A box contains {a} red balls and {b} blue balls. What is the total number of balls?",
            "answer": lambda a, b: a + b
        },
        {
            "template": "Sarah has {a} dollars. She earns {b} dollars more. How much does she have now?",
            "answer": lambda a, b: a + b
        },
        {
            "template": "A class has {a} students. They are divided into groups of {b} students each. How many groups are there?",
            "answer": lambda a, b: a // b
        }
    ]

    samples = []
    for i in range(n_samples):
        template_data = random.choice(problem_templates)
        template = template_data["template"]
        answer_func = template_data["answer"]

        # Generate random numbers
        if template.count('{') == 2:
            a = random.randint(10, 50)
            b = random.randint(1, 20)
            problem = template.format(a=a, b=b)
            answer = answer_func(a, b)
        elif template.count('{') == 3:
            a = random.randint(20, 100)
            b = random.randint(1, 20)
            c = random.randint(1, 20)
            problem = template.format(a=a, b=b, c=c)
            answer = answer_func(a, b, c)
        else:
            continue

        samples.append({
            'id': f'builtin_reasoning_{i}',
            'problem': problem,
            'answer': f"#### {answer}",
            'source': 'builtin'
        })

    print(f"Generated {len(samples)} built-in reasoning problems")
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
