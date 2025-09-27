#!/usr/bin/env python3
import torch
import subprocess
import os

def check_gpu():
    print("🔍 GPU Information:")
    print("=" * 40)
    
    # Check CUDA availability
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Total Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            
            # Current memory usage
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            free = (props.total_memory - torch.cuda.memory_reserved(i)) / 1024**3
            
            print(f"  Memory Allocated: {allocated:.1f} GB")
            print(f"  Memory Reserved: {reserved:.1f} GB")
            print(f"  Memory Free: {free:.1f} GB")
    
    # Try nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\n🖥️  nvidia-smi output:")
            print(result.stdout)
    except:
        print("\nnvidia-smi not available")

if __name__ == "__main__":
    check_gpu()