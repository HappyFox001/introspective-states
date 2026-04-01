#!/bin/bash

# Full pipeline script for introspection experiments
# Usage: bash run_full_pipeline.sh [--model MODEL_NAME] [--concepts CONCEPT1 CONCEPT2 ...] [--n-trials N]

set -e  # Exit on error

# Default parameters
MODEL="google/gemma-2b-it"
CONCEPTS=("formal_neutral" "cautious_assertive" "empathetic_neutral")
N_TRIALS=100
TASKS=("neutral_corpus" "step_reasoning")

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --concepts)
      shift
      CONCEPTS=()
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        CONCEPTS+=("$1")
        shift
      done
      ;;
    --n-trials)
      N_TRIALS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "=================================="
echo "Introspection Experiment Pipeline"
echo "=================================="
echo "Model: $MODEL"
echo "Concepts: ${CONCEPTS[@]}"
echo "Trials per condition: $N_TRIALS"
echo "Tasks: ${TASKS[@]}"
echo "=================================="
echo

# Step 1: Fetch data (if not already present)
echo "[Step 1/5] Fetching datasets..."
if [ ! -f "data/neutral_corpus.jsonl" ] || [ ! -f "data/step_reasoning.jsonl" ]; then
  python data/fetch_data.py --config config/experiment_config.yaml
else
  echo "Datasets already exist, skipping fetch."
fi
echo "✓ Data ready"
echo

# Step 2: Build concept vectors
echo "[Step 2/5] Building concept vectors..."
for concept in "${CONCEPTS[@]}"; do
  if [ ! -d "vectors/$concept" ]; then
    echo "  Building $concept..."
    python vectors/build_concepts.py \
      --config config/experiment_config.yaml \
      --prompts-config config/prompts.yaml \
      --model "$MODEL" \
      --concepts "$concept"
  else
    echo "  $concept vectors already exist, skipping."
  fi
done
echo "✓ Concept vectors ready"
echo

# Step 3: Run experiments
echo "[Step 3/5] Running experimental conditions..."
for task in "${TASKS[@]}"; do
  echo "  Task: $task"

  for concept in "${CONCEPTS[@]}"; do
    echo "    Concept: $concept"

    OUTPUT_FILE="output/json/${task}_${concept}_results.jsonl"

    if [ ! -f "$OUTPUT_FILE" ]; then
      python eval/run_conditions.py \
        --task "$task" \
        --conditions C0 C1 C2 C3 C4 \
        --concepts "$concept" \
        --n-trials "$N_TRIALS" \
        --output-dir output/json
    else
      echo "      Results already exist, skipping."
    fi
  done
done
echo "✓ Experiments complete"
echo

# Step 4: Grade results
echo "[Step 4/5] Grading introspection outputs..."
for task in "${TASKS[@]}"; do
  for concept in "${CONCEPTS[@]}"; do
    RESULTS_FILE="output/json/${task}_${concept}_results.jsonl"

    if [ -f "$RESULTS_FILE" ]; then
      echo "  Grading $task - $concept..."
      python scoring/grade_introspection.py \
        --results "$RESULTS_FILE" \
        --output-dir output/json
    fi
  done
done
echo "✓ Grading complete"
echo

# Step 5: Generate visualizations
echo "[Step 5/5] Generating visualizations..."
for task in "${TASKS[@]}"; do
  for concept in "${CONCEPTS[@]}"; do
    METRICS_FILE="output/json/${task}_${concept}_results_metrics.jsonl"

    if [ -f "$METRICS_FILE" ]; then
      echo "  Plotting $task - $concept..."
      python analysis/plot_results.py \
        --metrics "$METRICS_FILE" \
        --output-dir "output/figures/${task}_${concept}"
    fi
  done
done
echo "✓ Visualizations complete"
echo

# Summary
echo "=================================="
echo "Pipeline Complete!"
echo "=================================="
echo
echo "Results saved in:"
echo "  - Raw results: output/json/"
echo "  - Metrics: output/json/*_metrics.jsonl"
echo "  - Figures: output/figures/"
echo
echo "Next steps:"
echo "  1. Review summary tables in output/figures/*/summary_table.md"
echo "  2. Examine plots in output/figures/*/"
echo "  3. Check detailed results in output/json/"
echo
