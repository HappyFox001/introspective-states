"""
Quick test script to verify setup and basic functionality.
Run this after installation to check if everything works.
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('datasets', 'Datasets'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('yaml', 'PyYAML'),
        ('tqdm', 'tqdm')
    ]

    failed = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT FOUND")
            failed.append(name)

    if failed:
        print(f"\nMissing packages: {', '.join(failed)}")
        print("Please run: pip install -r requirements.txt")
        return False

    print("✓ All imports successful\n")
    return True


def test_configs():
    """Test if configuration files exist and are valid."""
    print("Testing configuration files...")

    configs = [
        'config/experiment_config.yaml',
        'config/prompts.yaml'
    ]

    failed = []
    for config_path in configs:
        path = Path(config_path)
        if not path.exists():
            print(f"  ✗ {config_path} - NOT FOUND")
            failed.append(config_path)
            continue

        try:
            import yaml
            with open(path, 'r') as f:
                yaml.safe_load(f)
            print(f"  ✓ {config_path}")
        except Exception as e:
            print(f"  ✗ {config_path} - INVALID: {e}")
            failed.append(config_path)

    if failed:
        print(f"\nConfiguration issues: {', '.join(failed)}")
        return False

    print("✓ All configs valid\n")
    return True


def test_directories():
    """Test if required directories exist."""
    print("Testing directory structure...")

    required_dirs = [
        'data',
        'vectors',
        'hooks',
        'eval',
        'scoring',
        'analysis',
        'output',
        'output/json',
        'output/figures',
        'output/docs'
    ]

    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            print(f"  Creating {dir_path}...")
            path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"  ✓ {dir_path}")

    print("✓ All directories ready\n")
    return True


def test_model_loading():
    """Test if a small model can be loaded (optional, requires GPU/CPU)."""
    print("Testing model loading (optional)...")
    print("  This may take a few minutes and requires ~2GB memory...")

    try:
        import torch
        from transformers import AutoTokenizer

        # Try loading tokenizer only (lightweight test)
        model_name = "google/gemma-2b-it"
        print(f"  Loading tokenizer for {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"  ✓ Tokenizer loaded")

        # Test basic tokenization
        test_text = "Hello, world!"
        tokens = tokenizer(test_text, return_tensors='pt')
        print(f"  ✓ Tokenization works")

        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  ⚠ CUDA not available (CPU only)")

        print("✓ Model infrastructure works\n")
        return True

    except Exception as e:
        print(f"  ⚠ Model loading failed: {e}")
        print("  This is OK for initial setup, but you'll need model access for experiments.\n")
        return True  # Don't fail the test, just warn


def test_scripts_exist():
    """Test if all main scripts exist."""
    print("Testing script files...")

    scripts = [
        'data/fetch_data.py',
        'vectors/build_concepts.py',
        'hooks/residual_inject.py',
        'eval/run_conditions.py',
        'scoring/grade_introspection.py',
        'analysis/plot_results.py'
    ]

    failed = []
    for script_path in scripts:
        path = Path(script_path)
        if not path.exists():
            print(f"  ✗ {script_path} - NOT FOUND")
            failed.append(script_path)
        else:
            print(f"  ✓ {script_path}")

    if failed:
        print(f"\nMissing scripts: {', '.join(failed)}")
        return False

    print("✓ All scripts present\n")
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("Introspective States - Setup Verification")
    print("="*60)
    print()

    tests = [
        ("Package imports", test_imports),
        ("Configuration files", test_configs),
        ("Directory structure", test_directories),
        ("Script files", test_scripts_exist),
        ("Model infrastructure", test_model_loading)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with error: {e}\n")
            results.append((test_name, False))

    # Summary
    print("="*60)
    print("Test Summary")
    print("="*60)

    all_passed = True
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {test_name}")
        if not success:
            all_passed = False

    print()

    if all_passed:
        print("✓ Setup verification complete!")
        print("\nYou can now run experiments with:")
        print("  bash run_full_pipeline.sh")
        print("\nOr step by step:")
        print("  1. python data/fetch_data.py")
        print("  2. python vectors/build_concepts.py --concepts formal_neutral")
        print("  3. python eval/run_conditions.py --task neutral_corpus --n-trials 10")
        print("  4. python scoring/grade_introspection.py --results output/json/neutral_corpus_results.jsonl")
        print("  5. python analysis/plot_results.py --metrics output/json/neutral_corpus_results_metrics.jsonl")
        return 0
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
