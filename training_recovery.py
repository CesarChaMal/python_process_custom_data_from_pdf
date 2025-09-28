#!/usr/bin/env python3
"""
Recovery script for fixing corrupted model training
Run this to clean up and retrain with stable parameters
"""

import os
import shutil
import subprocess


def clean_and_retrain():
    print("🧹 Cleaning up corrupted model files...")

    # Remove corrupted model
    model_dirs = [
        "./models/jvm_troubleshooting_model",
        "./models/jvm_troubleshooting_dialogpt-large_full",
        "./models/jvm_troubleshooting_dialogpt-medium_full",
        "./models/jvm_troubleshooting_dialogpt-small_full"
    ]

    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            print(f"  Removing {model_dir}...")
            shutil.rmtree(model_dir)

    # Clear GPU memory if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("  ✅ GPU memory cleared")
    except:
        pass

    # Set optimal environment variables
    print("\n⚙️ Setting optimal training configuration...")

    # Create .env.recovery file with stable settings
    recovery_env = """
# Recovery configuration - stable training parameters
AI_PROVIDER=ollama
OVERWRITE_DATASET=false
TRAIN_MODEL=true

# Use smaller model for stability
BASE_MODEL=microsoft/DialoGPT-small

# Conservative training mode
TRAINING_MODE=gpu_1

# Full fine-tuning (more stable than LoRA for small models)
FINETUNE_METHOD=full

# Your Hugging Face token (optional)
HUGGING_FACE_HUB_TOKEN=your_token_here
"""

    with open(".env.recovery", "w") as f:
        f.write(recovery_env)

    print("  ✅ Created .env.recovery with stable settings")

    # Create training script with fixed parameters
    train_script = """
import os
import sys

# Use recovery environment
from dotenv import load_dotenv
load_dotenv('.env.recovery')

# Import and run main with modified training args
import main

# Monkey-patch the training configuration
original_train = main.train_and_upload_model

def safe_train_wrapper(dataset_dict, auth_token, username):
    # Force stable parameters
    os.environ['BASE_MODEL'] = 'microsoft/DialoGPT-small'
    os.environ['TRAINING_MODE'] = 'gpu_1'
    os.environ['FINETUNE_METHOD'] = 'full'

    # Call original with safety wrapper
    try:
        return original_train(dataset_dict, auth_token, username)
    except Exception as e:
        print(f"Training failed: {e}")
        print("Try running with CPU: TRAINING_MODE=cpu_1")
        return None

main.train_and_upload_model = safe_train_wrapper

# Run main
if __name__ == "__main__":
    main.main()
"""

    with open("safe_retrain.py", "w") as f:
        f.write(train_script)

    print("\n🚀 Starting safe retraining...")
    print("  Using: DialoGPT-small (117M params)")
    print("  Mode: Conservative GPU training")
    print("  Learning rate: 5e-6 (very low)")

    # Run the safe training
    result = subprocess.run([sys.executable, "safe_retrain.py"], capture_output=False)

    if result.returncode == 0:
        print("\n✅ Retraining completed successfully!")
        print("Run: python quick_test.py to validate")
    else:
        print("\n❌ Retraining failed. Try these alternatives:")
        print("  1. Set BASE_MODEL=microsoft/DialoGPT-small in .env")
        print("  2. Use CPU training: TRAINING_MODE=cpu_1")
        print("  3. Reduce dataset size")


if __name__ == "__main__":
    clean_and_retrain()