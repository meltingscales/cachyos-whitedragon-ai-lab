#!/usr/bin/env python3
"""
GPU VRAM Stress Test
Tests GPU memory allocation and operations incrementally to find crash threshold.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch not found. Install with: pip install torch")

try:
    import cudf
    import cupy as cp
    HAS_CUDF = True
except ImportError:
    HAS_CUDF = False
    print("⚠️  cuDF/CuPy not found (optional for pandas GPU operations)")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠️  Pandas not found. Install with: pip install pandas")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("⚠️  NumPy not found. Install with: pip install numpy")


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def get_gpu_info():
    """Get GPU information"""
    if not HAS_TORCH:
        return None

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        return {
            "name": gpu_name,
            "total_memory_gb": round(total_memory, 2),
            "device": device
        }
    return None


def clear_gpu_memory():
    """Clear GPU memory cache"""
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(0.5)


def get_current_memory_usage():
    """Get current GPU memory usage"""
    if HAS_TORCH and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return {
            "allocated_gb": round(allocated, 3),
            "reserved_gb": round(reserved, 3)
        }
    return None


def test_pytorch_allocation(size_gb: float, operations: bool = True) -> Dict:
    """
    Test PyTorch GPU memory allocation and operations

    Args:
        size_gb: Amount of memory to allocate in GB
        operations: Whether to perform operations on allocated memory

    Returns:
        Dict with test results
    """
    result = {
        "success": False,
        "size_gb": size_gb,
        "error": None,
        "duration_s": 0,
        "operations_performed": 0
    }

    if not HAS_TORCH:
        result["error"] = "PyTorch not available"
        return result

    if not torch.cuda.is_available():
        result["error"] = "CUDA/ROCm not available"
        return result

    try:
        start_time = time.time()

        # Calculate tensor size for target memory
        # float32 = 4 bytes, so for X GB we need (X * 1024^3) / 4 elements
        elements = int((size_gb * 1024**3) / 4)
        rows = int(elements ** 0.5)
        cols = elements // rows

        print(f"  Allocating {rows:,} x {cols:,} tensor (~{size_gb:.1f} GB)...")

        # Allocate tensor on GPU
        tensor = torch.randn(rows, cols, device='cuda', dtype=torch.float32)
        torch.cuda.synchronize()

        memory_info = get_current_memory_usage()
        print(f"  Allocated: {memory_info['allocated_gb']:.2f} GB, Reserved: {memory_info['reserved_gb']:.2f} GB")

        ops_count = 0
        if operations:
            # Perform various operations
            print("  Performing operations...")

            # Matrix multiplication
            result_tensor = torch.mm(tensor[:1000, :1000], tensor[:1000, :1000].T)
            torch.cuda.synchronize()
            ops_count += 1

            # Element-wise operations
            tensor = tensor * 2.0 + 1.0
            torch.cuda.synchronize()
            ops_count += 1

            # Reduction operations
            mean_val = tensor.mean()
            std_val = tensor.std()
            torch.cuda.synchronize()
            ops_count += 2

            print(f"  Operations completed (mean={mean_val:.4f}, std={std_val:.4f})")

            # Clean up intermediate results
            del result_tensor

        duration = time.time() - start_time

        # Clean up
        del tensor
        clear_gpu_memory()

        result["success"] = True
        result["duration_s"] = round(duration, 3)
        result["operations_performed"] = ops_count

    except RuntimeError as e:
        result["error"] = str(e)
        clear_gpu_memory()
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
        clear_gpu_memory()

    return result


def test_cudf_operations(size_gb: float) -> Dict:
    """
    Test cuDF (GPU-accelerated pandas) operations

    Args:
        size_gb: Amount of memory to allocate in GB

    Returns:
        Dict with test results
    """
    result = {
        "success": False,
        "size_gb": size_gb,
        "error": None,
        "duration_s": 0,
        "operations_performed": 0
    }

    if not HAS_CUDF:
        result["error"] = "cuDF not available"
        return result

    try:
        start_time = time.time()

        # Calculate dataframe size for target memory
        # Estimate ~40 bytes per row with 4 float columns
        rows = int((size_gb * 1024**3) / 40)

        print(f"  Creating cuDF DataFrame with {rows:,} rows (~{size_gb:.1f} GB)...")

        # Create GPU DataFrame
        df = cudf.DataFrame({
            'a': cp.random.randn(rows),
            'b': cp.random.randn(rows),
            'c': cp.random.randn(rows),
            'd': cp.random.randn(rows)
        })

        ops_count = 0
        print("  Performing cuDF operations...")

        # Various pandas-like operations
        df['e'] = df['a'] + df['b']
        ops_count += 1

        df['f'] = df['c'] * df['d']
        ops_count += 1

        mean_vals = df.mean()
        ops_count += 1

        grouped = df.groupby(df['a'] > 0)['b'].mean()
        ops_count += 1

        sorted_df = df.sort_values('a')
        ops_count += 1

        print(f"  cuDF operations completed")

        duration = time.time() - start_time

        # Clean up
        del df, sorted_df, grouped, mean_vals
        cp.get_default_memory_pool().free_all_blocks()

        result["success"] = True
        result["duration_s"] = round(duration, 3)
        result["operations_performed"] = ops_count

    except Exception as e:
        result["error"] = str(e)
        if HAS_CUDF:
            cp.get_default_memory_pool().free_all_blocks()

    return result


def run_stress_test(
    start_gb: int = 1,
    max_gb: Optional[int] = None,
    increment_gb: int = 1,
    test_cudf: bool = False,
    operations: bool = True
):
    """
    Run incremental GPU VRAM stress test

    Args:
        start_gb: Starting allocation size in GB
        max_gb: Maximum allocation size in GB (None for GPU max)
        increment_gb: Increment size in GB
        test_cudf: Whether to test cuDF operations
        operations: Whether to perform operations on allocated memory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"gpu_stress_test_{timestamp}.log"
    json_file = log_dir / f"gpu_stress_test_{timestamp}.json"

    results = {
        "timestamp": timestamp,
        "gpu_info": None,
        "pytorch_tests": [],
        "cudf_tests": [],
        "threshold_gb": None,
        "max_successful_gb": None
    }

    def log(msg: str):
        """Log message to both console and file"""
        print(msg)
        with open(log_file, 'a') as f:
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp_str}] {msg}\n")

    log(f"{Colors.CYAN}{Colors.BOLD}=== GPU VRAM Stress Test ==={Colors.ENDC}")
    log("")

    # Get GPU info
    gpu_info = get_gpu_info()
    if not gpu_info:
        log(f"{Colors.RED}✗ No GPU detected or PyTorch not available{Colors.ENDC}")
        return

    results["gpu_info"] = gpu_info
    log(f"{Colors.BOLD}GPU:{Colors.ENDC} {gpu_info['name']}")
    log(f"{Colors.BOLD}Total VRAM:{Colors.ENDC} {gpu_info['total_memory_gb']:.2f} GB")
    log("")

    # Set max_gb if not specified
    if max_gb is None:
        max_gb = int(gpu_info['total_memory_gb'] * 0.95)  # 95% of total
        log(f"Max test size: {max_gb} GB (95% of total VRAM)")

    log(f"{Colors.BOLD}Test configuration:{Colors.ENDC}")
    log(f"  Start: {start_gb} GB")
    log(f"  Max: {max_gb} GB")
    log(f"  Increment: {increment_gb} GB")
    log(f"  Operations: {operations}")
    log(f"  cuDF test: {test_cudf}")
    log("")

    max_successful_gb = 0
    threshold_gb = None

    # Run PyTorch tests
    log(f"{Colors.BOLD}=== PyTorch Memory Tests ==={Colors.ENDC}")
    current_size = start_gb

    while current_size <= max_gb:
        log("")
        log(f"{Colors.CYAN}Testing {current_size} GB allocation...{Colors.ENDC}")

        result = test_pytorch_allocation(current_size, operations)
        results["pytorch_tests"].append(result)

        if result["success"]:
            log(f"{Colors.GREEN}✓ Success{Colors.ENDC} - Duration: {result['duration_s']:.2f}s, Operations: {result['operations_performed']}")
            max_successful_gb = current_size
        else:
            log(f"{Colors.RED}✗ Failed{Colors.ENDC} - Error: {result['error']}")
            threshold_gb = current_size
            log("")
            log(f"{Colors.YELLOW}Threshold found: Unable to allocate {threshold_gb} GB{Colors.ENDC}")
            log(f"{Colors.GREEN}Maximum successful allocation: {max_successful_gb} GB{Colors.ENDC}")
            break

        current_size += increment_gb
        time.sleep(1)  # Brief pause between tests

    if threshold_gb is None:
        log("")
        log(f"{Colors.GREEN}All tests passed up to {max_gb} GB!{Colors.ENDC}")
        max_successful_gb = max_gb

    results["max_successful_gb"] = max_successful_gb
    results["threshold_gb"] = threshold_gb

    # Run cuDF tests if requested
    if test_cudf and HAS_CUDF:
        log("")
        log(f"{Colors.BOLD}=== cuDF (GPU Pandas) Tests ==={Colors.ENDC}")

        # Test up to max successful size
        current_size = start_gb
        while current_size <= min(max_successful_gb, max_gb):
            log("")
            log(f"{Colors.CYAN}Testing {current_size} GB cuDF DataFrame...{Colors.ENDC}")

            result = test_cudf_operations(current_size)
            results["cudf_tests"].append(result)

            if result["success"]:
                log(f"{Colors.GREEN}✓ Success{Colors.ENDC} - Duration: {result['duration_s']:.2f}s, Operations: {result['operations_performed']}")
            else:
                log(f"{Colors.RED}✗ Failed{Colors.ENDC} - Error: {result['error']}")
                break

            current_size += increment_gb
            time.sleep(1)

    # Summary
    log("")
    log(f"{Colors.BOLD}=== Summary ==={Colors.ENDC}")
    log(f"GPU: {gpu_info['name']} ({gpu_info['total_memory_gb']:.2f} GB)")
    log(f"Maximum successful allocation: {Colors.GREEN}{max_successful_gb} GB{Colors.ENDC}")
    if threshold_gb:
        log(f"Failure threshold: {Colors.RED}{threshold_gb} GB{Colors.ENDC}")
    log(f"Utilization: {(max_successful_gb / gpu_info['total_memory_gb'] * 100):.1f}%")
    log("")
    log(f"Results saved to:")
    log(f"  Log: {log_file}")
    log(f"  JSON: {json_file}")

    # Save JSON results
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="GPU VRAM stress test - incrementally allocate memory to find crash threshold",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--start', '-s',
        type=int,
        default=1,
        help='Starting allocation size in GB (default: 1)'
    )

    parser.add_argument(
        '--max', '-m',
        type=int,
        default=None,
        help='Maximum allocation size in GB (default: 95%% of GPU memory)'
    )

    parser.add_argument(
        '--increment', '-i',
        type=int,
        default=1,
        help='Increment size in GB (default: 1)'
    )

    parser.add_argument(
        '--cudf',
        action='store_true',
        help='Also test cuDF (GPU-accelerated pandas) operations'
    )

    parser.add_argument(
        '--no-operations',
        action='store_true',
        help='Skip operations, only test allocation'
    )

    args = parser.parse_args()

    # Check dependencies
    missing_deps = []
    if not HAS_TORCH:
        missing_deps.append("torch")
    if not HAS_NUMPY:
        missing_deps.append("numpy")
    if not HAS_PANDAS:
        missing_deps.append("pandas")

    if missing_deps:
        print(f"{Colors.RED}Missing required dependencies: {', '.join(missing_deps)}{Colors.ENDC}")
        print(f"\nInstall with: pip install {' '.join(missing_deps)}")
        sys.exit(1)

    if args.cudf and not HAS_CUDF:
        print(f"{Colors.YELLOW}Warning: cuDF requested but not available. Skipping cuDF tests.{Colors.ENDC}")
        print("Install cuDF: https://docs.rapids.ai/install\n")

    run_stress_test(
        start_gb=args.start,
        max_gb=args.max,
        increment_gb=args.increment,
        test_cudf=args.cudf and HAS_CUDF,
        operations=not args.no_operations
    )


if __name__ == "__main__":
    main()
