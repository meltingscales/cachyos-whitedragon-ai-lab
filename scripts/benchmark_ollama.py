#!/usr/bin/env python3
"""
Ollama Model Benchmark Script
Tests all available Ollama models with simple queries to identify crashes.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Configuration
LOG_DIR = Path(__file__).parent.parent / "logs"
TEST_PROMPT = "Hello, please respond with 'OK' if you can read this."
TIMEOUT_SECONDS = 240


def setup_logging():
    """Create logs directory and return log file path."""
    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"ollama_benchmark_{timestamp}.log"
    return log_file


def log_message(log_file, message, print_to_console=True):
    """Write message to log file and optionally print to console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"

    if print_to_console:
        print(log_line)

    with open(log_file, "a") as f:
        f.write(log_line + "\n")


def get_installed_models():
    """Get list of all installed Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return []

        # Parse ollama list output
        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:  # Header + at least one model
            return []

        models = []
        for line in lines[1:]:  # Skip header
            parts = line.split()
            if parts:
                models.append(parts[0])  # First column is model name

        return models

    except Exception as e:
        print(f"Error getting model list: {e}")
        return []


def test_model(model_name, log_file, cpu_only=False):
    """Test a single model with a simple query."""
    log_message(log_file, f"Testing model: {model_name}")

    start_time = time.time()

    try:
        # Prepare environment for CPU-only mode if requested
        env = os.environ.copy()
        if cpu_only:
            env["OLLAMA_NUM_GPU"] = "0"

        # Run ollama with the test prompt
        result = subprocess.run(
            ["ollama", "run", model_name, TEST_PROMPT],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env=env
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            log_message(log_file, f"✓ {model_name}: SUCCESS ({elapsed:.2f}s)")
            if result.stdout:
                log_message(log_file, f"  Response preview: {result.stdout[:100]}...", print_to_console=False)
            return {
                "model": model_name,
                "status": "success",
                "elapsed_seconds": round(elapsed, 2),
                "response_length": len(result.stdout)
            }
        else:
            log_message(log_file, f"✗ {model_name}: FAILED (exit code {result.returncode}, {elapsed:.2f}s)")
            if result.stderr:
                log_message(log_file, f"  Error: {result.stderr[:200]}", print_to_console=False)
            return {
                "model": model_name,
                "status": "failed",
                "elapsed_seconds": round(elapsed, 2),
                "error": result.stderr[:500] if result.stderr else "Unknown error"
            }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        log_message(log_file, f"✗ {model_name}: TIMEOUT ({elapsed:.2f}s)")
        return {
            "model": model_name,
            "status": "timeout",
            "elapsed_seconds": round(elapsed, 2)
        }

    except Exception as e:
        elapsed = time.time() - start_time
        log_message(log_file, f"✗ {model_name}: CRASHED ({elapsed:.2f}s) - {str(e)}")
        return {
            "model": model_name,
            "status": "crashed",
            "elapsed_seconds": round(elapsed, 2),
            "error": str(e)
        }


def main():
    """Main benchmark function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Benchmark Ollama models for stability testing"
    )
    parser.add_argument(
        "--mode",
        choices=["gpu", "cpu", "both"],
        default="gpu",
        help="Run mode: gpu (default), cpu (CPU-only), or both (test each model with GPU then CPU)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Ollama Model Benchmark - GPU Stability Test")
    print("=" * 70)
    print()

    log_file = setup_logging()
    log_message(log_file, "Starting Ollama model benchmark")
    log_message(log_file, f"Mode: {args.mode.upper()}")
    log_message(log_file, f"Test prompt: {TEST_PROMPT}")
    log_message(log_file, f"Timeout: {TIMEOUT_SECONDS}s")
    log_message(log_file, "")

    # Get all models
    models = get_installed_models()

    if not models:
        log_message(log_file, "No Ollama models found!")
        log_message(log_file, "Run 'ollama list' to check installed models")
        return 1

    log_message(log_file, f"Found {len(models)} models to test")
    log_message(log_file, "")

    # Test each model
    results = []
    cpu_only = args.mode == "cpu"
    test_both = args.mode == "both"

    for i, model in enumerate(models, 1):
        if test_both:
            # Test with GPU first
            log_message(log_file, f"[{i}/{len(models)} - GPU] " + "-" * 60)
            result_gpu = test_model(model, log_file, cpu_only=False)
            result_gpu["mode"] = "GPU"
            results.append(result_gpu)
            log_message(log_file, "")
            time.sleep(2)

            # Then test with CPU
            log_message(log_file, f"[{i}/{len(models)} - CPU] " + "-" * 60)
            result_cpu = test_model(model, log_file, cpu_only=True)
            result_cpu["mode"] = "CPU"
            results.append(result_cpu)
            log_message(log_file, "")
        else:
            # Test with single mode
            mode_str = "CPU" if cpu_only else "GPU"
            log_message(log_file, f"[{i}/{len(models)} - {mode_str}] " + "-" * 60)
            result = test_model(model, log_file, cpu_only=cpu_only)
            result["mode"] = mode_str
            results.append(result)
            log_message(log_file, "")

        # Small delay between tests
        if i < len(models):
            time.sleep(2)

    # Summary
    log_message(log_file, "=" * 70)
    log_message(log_file, "BENCHMARK SUMMARY")
    log_message(log_file, "=" * 70)

    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    timeout_count = sum(1 for r in results if r["status"] == "timeout")
    crashed_count = sum(1 for r in results if r["status"] == "crashed")

    log_message(log_file, f"Total models tested: {len(results)}")
    log_message(log_file, f"✓ Successful: {success_count}")
    log_message(log_file, f"✗ Failed: {failed_count}")
    log_message(log_file, f"✗ Timeout: {timeout_count}")
    log_message(log_file, f"✗ Crashed: {crashed_count}")
    log_message(log_file, "")

    if crashed_count > 0 or timeout_count > 0:
        log_message(log_file, "Models with issues (should consider removing):")
        for result in results:
            if result["status"] in ["crashed", "timeout", "failed"]:
                log_message(log_file, f"  - {result['model']} ({result['status']})")
        log_message(log_file, "")

    if success_count > 0:
        log_message(log_file, "Working models:")
        for result in results:
            if result["status"] == "success":
                mode_info = f" [{result.get('mode', 'GPU')}]" if test_both or cpu_only else ""
                log_message(log_file, f"  - {result['model']}{mode_info} ({result['elapsed_seconds']}s)")

    # Save detailed JSON results
    json_file = log_file.with_suffix(".json")
    with open(json_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "test_config": {
                "prompt": TEST_PROMPT,
                "timeout_seconds": TIMEOUT_SECONDS
            },
            "summary": {
                "total": len(results),
                "success": success_count,
                "failed": failed_count,
                "timeout": timeout_count,
                "crashed": crashed_count
            },
            "results": results
        }, f, indent=2)

    log_message(log_file, "")
    log_message(log_file, f"Detailed results saved to: {json_file}")
    log_message(log_file, f"Log file: {log_file}")

    return 0 if crashed_count == 0 and timeout_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
