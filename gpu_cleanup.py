#!/usr/bin/env python3
"""
GPU Memory Cleanup Utility

This script helps clean up GPU memory and processes that might be blocking
model training due to CUDA out of memory errors.
"""

import subprocess
import sys
import torch
import gc
import os

def clear_gpu_memory():
    """Clear PyTorch GPU memory cache"""
    if torch.cuda.is_available():
        print("🧹 Clearing PyTorch GPU memory cache...")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        
        # Show memory status
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            free = total - reserved
            
            print(f"GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {free:.1f}GB free")
        
        print("✅ GPU memory cleared")
    else:
        print("❌ No CUDA GPUs available")

def show_gpu_processes():
    """Show current GPU processes"""
    try:
        print("🔍 Current GPU processes:")
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            print("PID\tProcess\t\tMemory (MB)")
            print("-" * 40)
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 3:
                    pid, process, memory = parts[0], parts[1], parts[2]
                    print(f"{pid}\t{process[:15]}\t{memory}")
        else:
            print("No GPU processes found")
            
    except FileNotFoundError:
        print("❌ nvidia-smi not found. Please install NVIDIA drivers.")
    except Exception as e:
        print(f"❌ Error checking GPU processes: {e}")

def kill_python_gpu_processes():
    """Kill Python processes using GPU (except current process and parents)"""
    try:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            python_pids = []
            
            # Get current process and parent PIDs to avoid killing them
            current_pid = str(os.getpid())
            parent_pid = str(os.getppid())
            
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 2:
                    pid, process = parts[0], parts[1]
                    if 'python' in process.lower():
                        # Don't kill current process or parent process
                        if pid != current_pid and pid != parent_pid:
                            python_pids.append(pid)
                        else:
                            print(f"⚠️  Skipping current/parent process {pid}")
            
            if python_pids:
                print(f"🔫 Found {len(python_pids)} Python GPU processes to kill")
                for pid in python_pids:
                    try:
                        subprocess.run(["kill", "-9", pid], check=True)
                        print(f"✅ Killed process {pid}")
                    except subprocess.CalledProcessError:
                        print(f"❌ Failed to kill process {pid}")
            else:
                print("No killable Python GPU processes found")
        else:
            print("No GPU processes found")
            
    except Exception as e:
        print(f"❌ Error killing processes: {e}")

def reset_gpu():
    """Reset GPU (requires root/admin)"""
    try:
        print("🔄 Attempting GPU reset...")
        result = subprocess.run(["nvidia-smi", "--gpu-reset"], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ GPU reset successful")
        else:
            print(f"❌ GPU reset failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error resetting GPU: {e}")

def main():
    """Main cleanup interface"""
    print("=" * 50)
    print("🚀 GPU Memory Cleanup Utility")
    print("=" * 50)
    print(f"Current PID: {os.getpid()}, Parent PID: {os.getppid()}")
    
    while True:
        print("\nOptions:")
        print("1. Clear PyTorch GPU memory cache")
        print("2. Show current GPU processes")
        print("3. Kill Python GPU processes")
        print("4. Reset GPU (requires admin)")
        print("5. Full cleanup (1 + 3)")
        print("6. Exit")
        
        try:
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                clear_gpu_memory()
            elif choice == '2':
                show_gpu_processes()
            elif choice == '3':
                kill_python_gpu_processes()
            elif choice == '4':
                reset_gpu()
            elif choice == '5':
                print("🧹 Starting full cleanup (memory + processes)...")
                clear_gpu_memory()
                kill_python_gpu_processes()
                print("✅ Full cleanup completed (current process preserved)")
            elif choice == '6':
                print("👋 Goodbye!")
                break
            else:
                print("Please enter 1-6")
                
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()