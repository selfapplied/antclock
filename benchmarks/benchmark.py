#!/usr/bin/env python3
"""CE Benchmark Runner - Simple execution of available benchmarks."""

import platform
import os


def setup_device():
    """Configure hardware acceleration based on available devices."""
    device_info = {"device": "cpu", "name": "CPU"}
    
    try:
        import torch
        if torch.cuda.is_available():
            # CUDA GPU available
            device_info["device"] = "cuda"
            device_info["name"] = torch.cuda.get_device_name(0)
            print(f"🚀 CUDA GPU detected: {device_info['name']}")
        elif platform.system() == 'Darwin' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Apple Silicon MPS available
            os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
            os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')
            device_info["device"] = "mps"
            device_info["name"] = "Apple Silicon (MPS)"
            print(f"🍎 Apple Silicon MPS detected")
        else:
            print("💻 Using CPU (no GPU acceleration available)")
    except ImportError:
        print("⚠️  PyTorch not available, using CPU")
    
    return device_info


# Configure hardware acceleration
DEVICE_INFO = setup_device()

from benchmarks.ce.synthetic import main as run_synthetic
from benchmarks.standard.standard import run_real_benchmarks

if __name__ == "__main__":
    print(f"🔧 Device: {DEVICE_INFO['name']} ({DEVICE_INFO['device']})")
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
