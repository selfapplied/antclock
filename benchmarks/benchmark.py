#!/usr/bin/env python3
"""CE Benchmark Runner - Simple execution of available benchmarks."""

import platform
import os

# Hardware acceleration hints for Apple Silicon
if platform.system() == 'Darwin':  # macOS
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')

from benchmarks.ce.synthetic import main as run_synthetic
from benchmarks.standard.standard import run_real_benchmarks

if __name__ == "__main__":
    print("🧬 Running CE synthetic benchmarks...")
    try:
        run_synthetic()
        print("✅ Synthetic benchmarks completed")
    except Exception as e:
        print(f"❌ Synthetic benchmarks failed: {e}")

    print("\n🎯 Attempting standard benchmarks...")
    try:
        run_real_benchmarks()
        print("✅ Standard benchmarks completed")
    except Exception as e:
        print(f"⚠️  Standard benchmarks failed (expected in sandbox): {e}")
