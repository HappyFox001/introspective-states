"""
Grade introspection outputs and compute metrics.
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np
from tqdm import tqdm


class IntrospectionGrader:
    """Grade introspection JSON outputs and compute metrics."""

    def __init__(self, config: Dict):
        """
        Initialize grader.

        Args:
            config: Experiment configuration dict
        """
        self.config = config
        self.schema = config['evaluation']['introspection_schema']

    def extract_json_from_output(self, output: str) -> Optional[Dict]:
        """
        Extract JSON object from model output.

        Args:
            output: Model output string

        Returns:
            Parsed JSON dict or None if extraction fails
        """
        # Try to find JSON block
        json_pattern = r'\{[^{}]*\}'

        matches = re.findall(json_pattern, output, re.DOTALL)

        for match in matches:
            try:
                parsed = json.loads(match)

                # Validate required fields
                if 'detection' in parsed or 'state_identification' in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue

        return None

    def grade_detection(self, result: Dict, ground_truth_injected: bool) -> Dict:
        """
        Grade detection accuracy.

        Args:
            result: Result dict with output
            ground_truth_injected: Whether injection actually occurred

        Returns:
            Grading dict with scores
        """
        parsed_json = self.extract_json_from_output(result['output'])

        if parsed_json is None:
            return {
                'valid_json': False,
                'detection_response': None,
                'detection_correct': False
            }

        detection = parsed_json.get('detection', 'uncertain')

        # Determine correctness
        if ground_truth_injected:
            # Should detect "yes"
            correct = detection == 'yes'
        else:
            # Should detect "no"
            correct = detection == 'no'

        return {
            'valid_json': True,
            'detection_response': detection,
            'detection_correct': correct
        }

    def grade_identification(self, result: Dict, target_concept: str) -> Dict:
        """
        Grade state identification accuracy.

        Args:
            result: Result dict with output
            target_concept: Target concept name (e.g., 'formal')

        Returns:
            Grading dict with scores
        """
        parsed_json = self.extract_json_from_output(result['output'])

        if parsed_json is None:
            return {
                'valid_json': False,
                'identification_response': None,
                'identification_correct': False
            }

        identification = parsed_json.get('state_identification', 'uncertain')

        # Check if matches target concept
        # Handle compound concept names like 'formal_neutral'
        target_parts = target_concept.split('_')
        target_positive = target_parts[0]

        correct = identification == target_positive

        return {
            'valid_json': True,
            'identification_response': identification,
            'identification_correct': correct,
            'target_concept': target_positive
        }

    def grade_source_attribution(self, result: Dict, condition: str) -> Dict:
        """
        Grade source attribution accuracy.

        Args:
            result: Result dict with output
            condition: Condition name ('C0', 'C1', etc.)

        Returns:
            Grading dict with scores
        """
        parsed_json = self.extract_json_from_output(result['output'])

        if parsed_json is None:
            return {
                'valid_json': False,
                'source_response': None,
                'source_correct': False
            }

        source = parsed_json.get('source_attribution', 'uncertain')

        # Determine expected source based on condition
        condition_config = self.config['conditions'][condition]

        if condition == 'C0':
            # No injection, neutral external -> intrinsic/uncertain
            expected = ['intrinsic', 'uncertain']
        elif condition == 'C1':
            # External only
            expected = ['external', 'both']
        elif condition == 'C2':
            # Internal only
            expected = ['internal', 'both']
        elif condition == 'C3':
            # Conflict - complex, accept internal or both
            expected = ['internal', 'both', 'uncertain']
        elif condition == 'C4':
            # Consistent - both external and internal
            expected = ['both', 'external', 'internal']
        else:
            expected = ['uncertain']

        correct = source in expected

        return {
            'valid_json': True,
            'source_response': source,
            'source_correct': correct,
            'expected_sources': expected
        }

    def grade_result(self, result: Dict) -> Dict:
        """
        Grade a single result across all metrics.

        Args:
            result: Result dict from experiment

        Returns:
            Complete grading dict
        """
        condition = result['condition']
        concept = result['concept']
        injected = result['injected']

        grading = {
            'task_id': result['task_id'],
            'condition': condition,
            'concept': concept,
            'layer': result['layer'],
            'alpha': result['alpha'],
            'injected': injected
        }

        # Grade detection
        detection_grade = self.grade_detection(result, injected)
        grading.update(detection_grade)

        # Grade identification (only if injected)
        if injected:
            identification_grade = self.grade_identification(result, concept)
            grading.update(identification_grade)

        # Grade source attribution
        source_grade = self.grade_source_attribution(result, condition)
        grading.update(source_grade)

        return grading

    def compute_aggregate_metrics(self, gradings: List[Dict]) -> Dict:
        """
        Compute aggregate metrics across gradings.

        Args:
            gradings: List of grading dicts

        Returns:
            Aggregate metrics dict
        """
        # Group by condition, concept, layer, alpha
        groups = defaultdict(list)

        for grading in gradings:
            key = (
                grading['condition'],
                grading['concept'],
                grading['layer'],
                grading['alpha']
            )
            groups[key].append(grading)

        # Compute metrics for each group
        metrics = []

        for key, group_gradings in groups.items():
            condition, concept, layer, alpha = key

            valid_json_count = sum(1 for g in group_gradings if g.get('valid_json', False))
            total_count = len(group_gradings)

            # Detection metrics
            detection_correct = [g for g in group_gradings if g.get('detection_correct', False)]
            detection_accuracy = len(detection_correct) / total_count if total_count > 0 else 0.0

            # Identification metrics (only for injected conditions)
            injected_gradings = [g for g in group_gradings if g.get('injected', False)]
            if injected_gradings:
                identification_correct = [g for g in injected_gradings if g.get('identification_correct', False)]
                identification_accuracy = len(identification_correct) / len(injected_gradings)
            else:
                identification_accuracy = None

            # Source attribution metrics
            source_correct = [g for g in group_gradings if g.get('source_correct', False)]
            source_accuracy = len(source_correct) / total_count if total_count > 0 else 0.0

            metrics.append({
                'condition': condition,
                'concept': concept,
                'layer': layer,
                'alpha': alpha,
                'n_trials': total_count,
                'valid_json_rate': valid_json_count / total_count if total_count > 0 else 0.0,
                'detection_accuracy': detection_accuracy,
                'identification_accuracy': identification_accuracy,
                'source_accuracy': source_accuracy
            })

        return metrics


def load_results(results_path: Path) -> List[Dict]:
    """Load experimental results from JSONL file."""
    results = []

    with open(results_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))

    return results


def main():
    parser = argparse.ArgumentParser(description="Grade introspection outputs")
    parser.add_argument('--config', type=str, default='config/experiment_config.yaml')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to results JSONL file')
    parser.add_argument('--output-dir', type=str, default='output/json',
                        help='Output directory for grades and metrics')

    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load results
    results_path = Path(args.results)
    print(f"Loading results from {results_path}...")
    results = load_results(results_path)
    print(f"Loaded {len(results)} results")

    # Initialize grader
    grader = IntrospectionGrader(config)

    # Grade all results
    print("Grading results...")
    gradings = []

    for result in tqdm(results, desc="Grading"):
        grading = grader.grade_result(result)
        gradings.append(grading)

    # Save individual gradings
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gradings_path = output_dir / f"{results_path.stem}_gradings.jsonl"
    with open(gradings_path, 'w') as f:
        for grading in gradings:
            f.write(json.dumps(grading) + '\n')

    print(f"Saved gradings to {gradings_path}")

    # Compute aggregate metrics
    print("Computing aggregate metrics...")
    metrics = grader.compute_aggregate_metrics(gradings)

    # Save metrics
    metrics_path = output_dir / f"{results_path.stem}_metrics.jsonl"
    with open(metrics_path, 'w') as f:
        for metric in metrics:
            f.write(json.dumps(metric) + '\n')

    print(f"Saved metrics to {metrics_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY METRICS")
    print("="*60)

    for metric in metrics:
        if metric['alpha'] == 1.0:  # Show only alpha=1.0 for brevity
            print(f"\n{metric['condition']} | {metric['concept']} | Layer {metric['layer']}")
            print(f"  Valid JSON: {metric['valid_json_rate']:.2%}")
            print(f"  Detection Accuracy: {metric['detection_accuracy']:.2%}")
            if metric['identification_accuracy'] is not None:
                print(f"  Identification Accuracy: {metric['identification_accuracy']:.2%}")
            print(f"  Source Attribution Accuracy: {metric['source_accuracy']:.2%}")


if __name__ == '__main__':
    main()
