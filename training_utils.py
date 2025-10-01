"""
Training utilities to eliminate code duplication in main.py
"""
import os
import torch
import subprocess
from transformers import AutoModelForCausalLM

def recreate_model_with_config(base_model: str, config: dict):
    """Recreate model with specific configuration - eliminates 6+ duplications"""
    # Clear existing model
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # Load with new config
    return AutoModelForCausalLM.from_pretrained(base_model, **config)

def run_testing_script(script_name: str) -> bool:
    """Run testing scripts safely - eliminates 8+ duplications"""
    try:
        subprocess.run(["python", script_name], check=True)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to run {script_name}: {e}")
        return False

def handle_training_error_menu(model, base_model, training_args, trainer_components):
    """Unified error handling menu - eliminates 2 massive duplications"""
    options = {
        '1': ('Smart memory optimization', lambda: _apply_memory_optimization(model, base_model, '4GB', 16)),
        '2': ('Extreme memory efficiency', lambda: _apply_memory_optimization(model, base_model, '2GB', 64)),
        '3': ('Switch to CPU training', lambda: _switch_to_cpu(model, base_model, training_args)),
        '4': ('Run GPU cleanup utility', lambda: run_testing_script('gpu_cleanup.py')),
        '5': ('Change base model', lambda: 'restart'),
        '6': ('Change training configuration', lambda: 'restart'),
        '7': ('Skip training and test existing model', lambda: _test_existing_model()),
        '8': ('Exit', lambda: False)
    }
    
    while True:
        print("\n📋 Training Error Recovery Options:")
        for key, (desc, _) in options.items():
            print(f"  {key}. {desc}")
        
        try:
            choice = input(f"\nSelect option (1-8) [1]: ").strip() or '1'
            if choice in options:
                result = options[choice][1]()
                if result in [True, False, 'restart']:
                    return result
            else:
                print("Please enter 1-8")
        except (EOFError, KeyboardInterrupt):
            return False

def _apply_memory_optimization(model, base_model, memory_limit, batch_steps):
    """Apply memory optimization settings"""
    config = {
        'torch_dtype': torch.float32,
        'device_map': "auto", 
        'low_cpu_mem_usage': True,
        'max_memory': {0: memory_limit}
    }
    try:
        new_model = recreate_model_with_config(base_model, config)
        print(f"[SUCCESS] Applied {memory_limit} memory optimization")
        return new_model
    except Exception as e:
        print(f"[ERROR] Memory optimization failed: {e}")
        return None

def _switch_to_cpu(model, base_model, training_args):
    """Switch model to CPU training"""
    config = {
        'torch_dtype': torch.float32,
        'low_cpu_mem_usage': True,
        'use_cache': False
    }
    try:
        new_model = recreate_model_with_config(base_model, config).cpu()
        training_args.use_cpu = True
        training_args.fp16 = False
        print("[SUCCESS] Switched to CPU training")
        return new_model
    except Exception as e:
        print(f"[ERROR] CPU switch failed: {e}")
        return None

def _test_existing_model():
    """Test existing model if available"""
    model_dir = "./models/jvm_troubleshooting_model"
    if os.path.exists(model_dir):
        print(f"[INFO] Found existing model at {model_dir}")
        print("\n🧪 Testing Options:")
        print("  1. Interactive testing (test_model.py)")
        print("  2. Quick batch testing (quick_test.py)")
        print("  3. Skip testing")
        
        try:
            choice = input("\nChoose testing option (1-3) [1]: ").strip() or '1'
            if choice == '1':
                run_testing_script('test_model.py')
            elif choice == '2':
                run_testing_script('quick_test.py')
        except (EOFError, KeyboardInterrupt):
            pass
    else:
        print("[WARNING] No existing model found")
        print("[INFO] Try: python model_utils.py recover")
    return False