"""
Visualize introspection experiment results.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def load_metrics(metrics_path: Path) -> pd.DataFrame:
    """Load metrics from JSONL file into DataFrame."""
    metrics = []

    with open(metrics_path, 'r') as f:
        for line in f:
            metrics.append(json.loads(line))

    return pd.DataFrame(metrics)


def plot_alpha_sweep(df: pd.DataFrame, output_dir: Path):
    """
    Plot accuracy vs injection strength (alpha) curves.

    Args:
        df: Metrics DataFrame
        output_dir: Output directory for figures
    """
    # Filter to single layer for clarity
    layer = df['layer'].mode()[0]  # Most common layer
    df_layer = df[df['layer'] == layer]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metrics_to_plot = [
        ('detection_accuracy', 'Detection Accuracy'),
        ('identification_accuracy', 'Identification Accuracy'),
        ('source_accuracy', 'Source Attribution Accuracy')
    ]

    for idx, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[idx]

        # Plot each condition
        for condition in sorted(df_layer['condition'].unique()):
            df_cond = df_layer[df_layer['condition'] == condition]

            # Group by alpha and average across concepts
            grouped = df_cond.groupby('alpha')[metric].mean()

            ax.plot(grouped.index, grouped.values, marker='o', label=condition, linewidth=2)

        ax.set_xlabel('Injection Strength (α)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1])

    plt.tight_layout()
    output_path = output_dir / 'alpha_sweep.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved alpha sweep plot to {output_path}")
    plt.close()


def plot_layer_sensitivity(df: pd.DataFrame, output_dir: Path):
    """
    Plot accuracy vs layer depth.

    Args:
        df: Metrics DataFrame
        output_dir: Output directory for figures
    """
    # Filter to alpha=1.0 for clarity
    df_alpha = df[df['alpha'] == 1.0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metrics_to_plot = [
        ('detection_accuracy', 'Detection Accuracy'),
        ('identification_accuracy', 'Identification Accuracy'),
        ('source_accuracy', 'Source Attribution Accuracy')
    ]

    for idx, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[idx]

        # Plot each condition
        for condition in sorted(df_alpha['condition'].unique()):
            df_cond = df_alpha[df_alpha['condition'] == condition]

            # Group by layer and average across concepts
            grouped = df_cond.groupby('layer')[metric].mean()

            ax.plot(grouped.index, grouped.values, marker='o', label=condition, linewidth=2)

        ax.set_xlabel('Layer Index', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1])

    plt.tight_layout()
    output_path = output_dir / 'layer_sensitivity.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved layer sensitivity plot to {output_path}")
    plt.close()


def plot_condition_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Plot condition comparison heatmap.

    Args:
        df: Metrics DataFrame
        output_dir: Output directory for figures
    """
    # Filter to alpha=1.0 and middle layer
    middle_layer = sorted(df['layer'].unique())[len(df['layer'].unique()) // 2]
    df_filtered = df[(df['alpha'] == 1.0) & (df['layer'] == middle_layer)]

    # Pivot for heatmap
    metrics = ['detection_accuracy', 'identification_accuracy', 'source_accuracy']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Create pivot table
        pivot = df_filtered.pivot_table(
            values=metric,
            index='concept',
            columns='condition',
            aggfunc='mean'
        )

        # Plot heatmap
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={'label': 'Accuracy'}
        )

        ax.set_title(metric.replace('_', ' ').title(), fontsize=13, fontweight='bold')
        ax.set_xlabel('Condition', fontsize=12)
        ax.set_ylabel('Concept', fontsize=12)

    plt.tight_layout()
    output_path = output_dir / 'condition_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved condition comparison plot to {output_path}")
    plt.close()


def plot_concept_breakdown(df: pd.DataFrame, output_dir: Path):
    """
    Plot per-concept performance breakdown.

    Args:
        df: Metrics DataFrame
        output_dir: Output directory for figures
    """
    # Filter to C2 (Internal-Only) condition, alpha=1.0
    df_filtered = df[(df['condition'] == 'C2') & (df['alpha'] == 1.0)]

    # Group by concept and layer
    grouped = df_filtered.groupby(['concept', 'layer']).agg({
        'detection_accuracy': 'mean',
        'identification_accuracy': 'mean'
    }).reset_index()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, metric in enumerate(['detection_accuracy', 'identification_accuracy']):
        ax = axes[idx]

        for concept in sorted(grouped['concept'].unique()):
            df_concept = grouped[grouped['concept'] == concept]

            ax.plot(
                df_concept['layer'],
                df_concept[metric],
                marker='o',
                label=concept,
                linewidth=2
            )

        ax.set_xlabel('Layer Index', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(
            metric.replace('_', ' ').title() + ' (C2, α=1.0)',
            fontsize=13,
            fontweight='bold'
        )
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1])

    plt.tight_layout()
    output_path = output_dir / 'concept_breakdown.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved concept breakdown plot to {output_path}")
    plt.close()


def generate_summary_table(df: pd.DataFrame, output_dir: Path):
    """
    Generate markdown summary table.

    Args:
        df: Metrics DataFrame
        output_dir: Output directory
    """
    # Filter to alpha=1.0 and middle layer
    middle_layer = sorted(df['layer'].unique())[len(df['layer'].unique()) // 2]
    df_filtered = df[(df['alpha'] == 1.0) & (df['layer'] == middle_layer)]

    # Group by condition and compute means
    summary = df_filtered.groupby('condition').agg({
        'detection_accuracy': ['mean', 'std'],
        'identification_accuracy': ['mean', 'std'],
        'source_accuracy': ['mean', 'std'],
        'valid_json_rate': 'mean'
    }).round(3)

    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

    # Save as markdown
    output_path = output_dir / 'summary_table.md'
    with open(output_path, 'w') as f:
        f.write(f"# Summary Metrics (Layer {middle_layer}, α=1.0)\n\n")
        f.write(summary.to_markdown())

    print(f"Saved summary table to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize introspection results")
    parser.add_argument('--metrics', type=str, required=True,
                        help='Path to metrics JSONL file')
    parser.add_argument('--output-dir', type=str, default='output/figures',
                        help='Output directory for plots')

    args = parser.parse_args()

    # Load metrics
    metrics_path = Path(args.metrics)
    print(f"Loading metrics from {metrics_path}...")
    df = load_metrics(metrics_path)
    print(f"Loaded {len(df)} metric entries")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set plotting style
    sns.set_style('whitegrid')
    plt.rcParams['font.size'] = 10

    # Generate plots
    print("\nGenerating plots...")

    plot_alpha_sweep(df, output_dir)
    plot_layer_sensitivity(df, output_dir)
    plot_condition_comparison(df, output_dir)
    plot_concept_breakdown(df, output_dir)

    # Generate summary table
    generate_summary_table(df, output_dir)

    print("\nVisualization complete!")
    print(f"All outputs saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
