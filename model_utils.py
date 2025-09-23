#!/usr/bin/env python3
"""
Model Utility Functions
Handles model recovery, downloading, and management.
"""

import os
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

def copy_from_checkpoint():
    """Copy model files from the latest checkpoint."""
    model_dir = "./models/jvm_troubleshooting_model"
    
    # Find the latest checkpoint
    checkpoints = []
    if os.path.exists(model_dir):
        for item in os.listdir(model_dir):
            if item.startswith("checkpoint-") and os.path.isdir(os.path.join(model_dir, item)):
                checkpoint_num = int(item.split("-")[1])
                checkpoints.append((checkpoint_num, item))
    
    if not checkpoints:
        print("[ERROR] No checkpoints found to copy from")
        return False
    
    # Get the latest checkpoint
    latest_checkpoint = max(checkpoints)[1]
    checkpoint_path = os.path.join(model_dir, latest_checkpoint)
    
    print(f"[INFO] Copying model from {latest_checkpoint}...")
    
    # Files to copy
    files_to_copy = [
        "model.safetensors",
        "pytorch_model.bin"  # fallback
    ]
    
    copied = False
    for file_name in files_to_copy:
        src_file = os.path.join(checkpoint_path, file_name)
        dst_file = os.path.join(model_dir, file_name)
        
        if os.path.exists(src_file):
            try:
                shutil.copy2(src_file, dst_file)
                print(f"[SUCCESS] Copied {file_name}")
                copied = True
                break
            except Exception as e:
                print(f"[ERROR] Failed to copy {file_name}: {e}")
    
    return copied

def download_from_huggingface(model_id="CesarChaMal/jvm_troubleshooting_model", auth_token=None):
    """Download model from Hugging Face Hub."""
    model_dir = "./models/jvm_troubleshooting_model"
    
    try:
        print(f"[INFO] Downloading model from Hugging Face: {model_id}")
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Download the model
        snapshot_download(
            repo_id=model_id,
            local_dir=model_dir,
            token=auth_token,
            ignore_patterns=["*.git*", "README.md", ".gitattributes"]
        )
        
        print("[SUCCESS] Model downloaded from Hugging Face")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to download model from Hugging Face: {e}")
        return False

def test_model_loading():
    """Test if the model can be loaded successfully."""
    model_path = "./models/jvm_troubleshooting_model"
    
    if not os.path.exists(model_path):
        print("[ERROR] Model directory not found")
        return False
    
    try:
        print("[INFO] Testing model loading...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        print("[SUCCESS] Model loads successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return False

def recover_model(auth_token=None):
    """Try to recover the model using available options."""
    print("[INFO] Attempting to recover model...")
    
    # Option 1: Copy from checkpoint
    print("\n1. Trying to copy from local checkpoint...")
    if copy_from_checkpoint():
        if test_model_loading():
            print("[SUCCESS] Model recovered from checkpoint")
            return True
        else:
            print("[WARNING] Checkpoint copy failed, trying Hugging Face...")
    
    # Option 2: Download from Hugging Face
    print("\n2. Trying to download from Hugging Face...")
    if download_from_huggingface(auth_token=auth_token):
        if test_model_loading():
            print("[SUCCESS] Model recovered from Hugging Face")
            return True
    
    print("[ERROR] Could not recover model using any method")
    return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "copy":
            copy_from_checkpoint()
        elif sys.argv[1] == "download":
            download_from_huggingface()
        elif sys.argv[1] == "test":
            test_model_loading()
        elif sys.argv[1] == "recover":
            recover_model()
    else:
        print("Usage: python model_utils.py [copy|download|test|recover]")