#!/usr/bin/env python3
"""
Dataset Upload Utility - Standalone Hugging Face Dataset Upload

This script provides standalone functionality to upload datasets to Hugging Face Hub
with validation, error handling, and CLI interface.

Features:
- HF connection validation
- Repository creation/update
- Dataset upload with progress tracking
- CLI interface for standalone use
- Integration with main pipeline

Author: CesarChaMal
License: MIT
"""

import os
import sys
import argparse
from datasets import DatasetDict, load_from_disk
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError
from check_hf import check_hf_connection


def upload_dataset_to_hf(dataset_path: str, dataset_name: str, auth_token: str, username: str = None) -> bool:
    """
    Upload dataset to Hugging Face Hub with validation and error handling.
    
    Args:
        dataset_path (str): Local path to the dataset
        dataset_name (str): Name for the dataset repository
        auth_token (str): Hugging Face authentication token
        username (str): HF username (optional, will be fetched if not provided)
        
    Returns:
        bool: True if upload successful, False otherwise
    """
    
    # Validate dataset exists locally
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found at {dataset_path}")
        return False
    
    # Validate HF connection
    print("[INFO] Validating Hugging Face connection...")
    if not check_hf_connection():
        print("[ERROR] Hugging Face connection failed")
        return False
    
    try:
        # Get username if not provided
        if not username:
            api = HfApi(token=auth_token)
            username = api.whoami()["name"]
            print(f"[INFO] Using HF username: {username}")
        
        # Load dataset
        print(f"[INFO] Loading dataset from {dataset_path}...")
        dataset_dict = load_from_disk(dataset_path)
        
        # Validate dataset structure
        if not isinstance(dataset_dict, DatasetDict):
            print("[ERROR] Invalid dataset format - expected DatasetDict")
            return False
        
        if 'train' not in dataset_dict or 'test' not in dataset_dict:
            print("[ERROR] Dataset missing train/test splits")
            return False
        
        train_size = len(dataset_dict['train'])
        test_size = len(dataset_dict['test'])
        print(f"[INFO] Dataset loaded: {train_size} train + {test_size} test examples")
        
        # Create repository ID
        repo_id = f"{username}/{dataset_name}"
        print(f"[INFO] Target repository: {repo_id}")
        
        # Create or update repository
        print("[INFO] Creating/updating repository...")
        try:
            create_repo(repo_id=repo_id, token=auth_token, repo_type="dataset")
            print(f"[SUCCESS] Repository {repo_id} created!")
        except HfHubHTTPError as e:
            if "already exists" in str(e):
                print(f"[INFO] Repository {repo_id} already exists, updating...")
            else:
                print(f"[ERROR] Failed to create repository: {e}")
                return False
        
        # Upload dataset
        print("[INFO] Uploading dataset to Hugging Face Hub...")
        dataset_dict.push_to_hub(repo_id, token=auth_token)
        
        print(f"[SUCCESS] Dataset uploaded successfully!")
        print(f"[INFO] View at: https://huggingface.co/datasets/{repo_id}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Dataset upload failed: {e}")
        return False


def main():
    """CLI interface for standalone dataset upload."""
    parser = argparse.ArgumentParser(
        description="Upload dataset to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python upload_dataset.py --dataset ./dataset/jvm_troubleshooting_guide --name jvm_troubleshooting_guide
  python upload_dataset.py --dataset ./dataset/my_data --name my_dataset --username myuser
        """
    )
    
    parser.add_argument(
        "--dataset", "-d",
        required=True,
        help="Path to local dataset directory"
    )
    
    parser.add_argument(
        "--name", "-n", 
        required=True,
        help="Dataset name for Hugging Face repository"
    )
    
    parser.add_argument(
        "--token", "-t",
        help="Hugging Face token (or set HUGGING_FACE_HUB_TOKEN env var)"
    )
    
    parser.add_argument(
        "--username", "-u",
        help="Hugging Face username (will be auto-detected if not provided)"
    )
    
    args = parser.parse_args()
    
    # Get token from args or environment
    auth_token = args.token or os.getenv('HUGGING_FACE_HUB_TOKEN')
    if not auth_token:
        print("[ERROR] Hugging Face token required!")
        print("Provide via --token argument or HUGGING_FACE_HUB_TOKEN environment variable")
        sys.exit(1)
    
    # Upload dataset
    success = upload_dataset_to_hf(
        dataset_path=args.dataset,
        dataset_name=args.name,
        auth_token=auth_token,
        username=args.username
    )
    
    if success:
        print("\n✅ Dataset upload completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Dataset upload failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()