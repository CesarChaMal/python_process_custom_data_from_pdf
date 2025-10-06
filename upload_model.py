#!/usr/bin/env python3
"""
Model Upload Utility for Hugging Face Hub

Standalone script to upload trained models to Hugging Face Hub.
Can be used independently or called from the main workflow.

Usage:
    python upload_model.py                    # Interactive mode
    python upload_model.py --model-dir ./models/my_model --model-id user/model-name
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError
from check_hf import check_hf_connection
from create_model_card import generate_and_upload_model_card

load_dotenv()

def upload_model_to_hf(model_dir, model_id, auth_token, base_model=None, finetune_method=None, train_size=None, test_size=None):
    """
    Upload model to Hugging Face Hub with model card generation.
    
    Args:
        model_dir (str): Local model directory path
        model_id (str): HF model repository ID (username/model-name)
        auth_token (str): HF authentication token
        base_model (str): Base model used for fine-tuning (optional)
        finetune_method (str): Fine-tuning method used (optional)
        train_size (int): Training dataset size (optional)
        test_size (int): Test dataset size (optional)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"[INFO] Uploading model to Hugging Face Hub as {model_id}...")
        print(f"[INFO] Local model directory: {model_dir}")
        
        # Initialize Hugging Face API
        api = HfApi()
        
        # Create repository (ignore if already exists)
        try:
            create_repo(repo_id=model_id, token=auth_token, repo_type="model", exist_ok=True)
        except HfHubHTTPError:
            pass  # Repository already exists
        
        # Upload all model files
        api.upload_folder(
            folder_path=model_dir,
            repo_id=model_id,
            token=auth_token,
            repo_type="model"
        )
        
        print(f"[SUCCESS] Model uploaded to: https://huggingface.co/{model_id}")
        
        # Generate and upload model card if metadata provided
        if base_model:
            print("[INFO] Generating model card...")
            generate_and_upload_model_card(
                model_id=model_id,
                auth_token=auth_token,
                base_model=base_model,
                finetune_method=finetune_method or "full",
                train_size=train_size or 100,
                test_size=test_size or 50
            )
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to upload model: {e}")
        return False

def main():
    # Show usage if no args provided
    if len(sys.argv) == 1:
        print("🚀 Model Upload Utility")
        print("=" * 40)
        print("\n📋 Quick Usage:")
        print("  python upload_model.py                    # Upload default model")
        print("  python upload_model.py --model-id user/model-name")
        print("\n💡 Examples:")
        print("  # Upload with auto-detected settings")
        print("  python upload_model.py")
        print("\n  # Upload specific model")
        print("  python upload_model.py --model-dir ./models/my_model --model-id myuser/my-model")
        print("\n  # Upload with metadata")
        print("  python upload_model.py --base-model microsoft/DialoGPT-medium --finetune-method full")
        print("\n🔧 Setup:")
        print("  1. Set HUGGING_FACE_HUB_TOKEN in .env file")
        print("  2. Ensure model exists locally")
        print("  3. Run upload command")
        print("\n📖 Full help: python upload_model.py --help")
        sys.exit(0)
    
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    parser.add_argument("--model-dir", default="./models/jvm_troubleshooting_model", help="Local model directory")
    parser.add_argument("--model-id", help="HF model ID (username/model-name)")
    parser.add_argument("--base-model", help="Base model used for fine-tuning")
    parser.add_argument("--finetune-method", choices=["full", "lora"], default="full", help="Fine-tuning method")
    parser.add_argument("--train-size", type=int, help="Training dataset size")
    parser.add_argument("--test-size", type=int, help="Test dataset size")
    
    args = parser.parse_args()
    
    print("🚀 Model Upload Utility")
    print("=" * 40)
    
    # Validate model directory exists
    if not os.path.exists(args.model_dir):
        print(f"[ERROR] Model directory not found: {args.model_dir}")
        return 1
    
    # Get HF token
    auth_token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    if not auth_token:
        print("[ERROR] HUGGING_FACE_HUB_TOKEN not found in environment")
        print("💡 Setup:")
        print("  1. Get token from: https://huggingface.co/settings/tokens")
        print("  2. Add to .env: HUGGING_FACE_HUB_TOKEN=hf_your_token_here")
        return 1
    
    # Validate HF connection
    if not check_hf_connection():
        print("[ERROR] Hugging Face connection failed")
        return 1
    
    # Get model ID if not provided
    if not args.model_id:
        try:
            username = HfApi(token=auth_token).whoami()["name"]
            model_name = os.path.basename(args.model_dir.rstrip('/'))
            args.model_id = f"{username}/{model_name}"
            print(f"[INFO] Using model ID: {args.model_id}")
        except Exception as e:
            print(f"[ERROR] Could not determine model ID: {e}")
            return 1
    
    # Upload model
    success = upload_model_to_hf(
        model_dir=args.model_dir,
        model_id=args.model_id,
        auth_token=auth_token,
        base_model=args.base_model,
        finetune_method=args.finetune_method,
        train_size=args.train_size,
        test_size=args.test_size
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())