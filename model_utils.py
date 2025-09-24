#!/usr/bin/env python3
"""
JVM Model Recovery and Management Utilities

This utility script provides comprehensive model management functionality for
the JVM troubleshooting model. It handles common scenarios where model files
go missing or become corrupted, offering multiple recovery strategies.

Features:
- Model recovery from local checkpoints
- Model download from Hugging Face Hub
- Model integrity testing and validation
- Automated recovery with fallback options
- Command-line interface for manual operations

Usage:
    python model_utils.py recover    # Automatic recovery
    python model_utils.py copy       # Copy from checkpoint
    python model_utils.py download   # Download from HF Hub
    python model_utils.py test       # Test model loading

Common Use Cases:
- Model files missing after Git operations
- Corrupted model files
- Moving between different environments
- CI/CD pipeline model deployment

Author: CesarChaMal
License: MIT
"""

# =============================================================================
# IMPORTS AND DEPENDENCIES
# =============================================================================

import os
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM  # Model loading
from huggingface_hub import snapshot_download  # HF Hub downloads
import torch

# =============================================================================
# LOCAL CHECKPOINT RECOVERY FUNCTIONS
# =============================================================================

def copy_from_checkpoint():
    """
    Recover model files from the most recent training checkpoint.
    
    This function searches for training checkpoints in the model directory
    and copies the model files from the latest checkpoint to the main model
    directory. This is useful when:
    - Main model files are missing but checkpoints exist
    - Training completed but final model wasn't saved properly
    - Model files were accidentally deleted
    
    Returns:
        bool: True if model was successfully copied, False otherwise
        
    Note:
        Checkpoints are created during training and contain complete model state.
        This function prioritizes .safetensors format over .bin for better safety.
    """
    
    model_dir = "./models/jvm_troubleshooting_model"
    
    print("🔍 Searching for training checkpoints...")
    
    # =============================================================================
    # CHECKPOINT DISCOVERY
    # =============================================================================
    
    # Find all available checkpoints
    checkpoints = []
    if os.path.exists(model_dir):
        for item in os.listdir(model_dir):
            if item.startswith("checkpoint-") and os.path.isdir(os.path.join(model_dir, item)):
                try:
                    # Extract checkpoint number for sorting
                    checkpoint_num = int(item.split("-")[1])
                    checkpoints.append((checkpoint_num, item))
                except ValueError:
                    # Skip malformed checkpoint names
                    continue
    
    if not checkpoints:
        print("❌ No training checkpoints found")
        print("💡 Checkpoints are created during model training")
        print("   Try running: python main.py (with TRAIN_MODEL=true)")
        return False
    
    # Sort checkpoints and get the latest one
    checkpoints.sort()
    latest_checkpoint_num, latest_checkpoint = checkpoints[-1]
    checkpoint_path = os.path.join(model_dir, latest_checkpoint)
    
    print(f"✅ Found {len(checkpoints)} checkpoints")
    print(f"💾 Using latest checkpoint: {latest_checkpoint} (step {latest_checkpoint_num})")
    
    # =============================================================================
    # MODEL FILE COPYING
    # =============================================================================
    
    # Priority order for model files (safetensors preferred for safety)
    model_files_to_copy = [
        "model.safetensors",    # Preferred format (safer)
        "pytorch_model.bin",    # Fallback format
        "adapter_model.safetensors",  # LoRA adapter (if exists)
    ]
    
    # Essential files that should also be copied
    essential_files = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json"
    ]
    
    copied_files = []
    
    # Copy model files (at least one must succeed)
    model_copied = False
    for file_name in model_files_to_copy:
        src_file = os.path.join(checkpoint_path, file_name)
        dst_file = os.path.join(model_dir, file_name)
        
        if os.path.exists(src_file):
            try:
                shutil.copy2(src_file, dst_file)
                print(f"✅ Copied model file: {file_name}")
                copied_files.append(file_name)
                model_copied = True
                break  # Only need one model file
            except Exception as e:
                print(f"⚠️  Failed to copy {file_name}: {e}")
    
    if not model_copied:
        print("❌ No model files could be copied from checkpoint")
        return False
    
    # Copy essential configuration files
    for file_name in essential_files:
        src_file = os.path.join(checkpoint_path, file_name)
        dst_file = os.path.join(model_dir, file_name)
        
        if os.path.exists(src_file):
            try:
                shutil.copy2(src_file, dst_file)
                copied_files.append(file_name)
            except Exception as e:
                print(f"⚠️  Failed to copy {file_name}: {e}")
    
    print(f"✨ Successfully copied {len(copied_files)} files from checkpoint")
    return True

# =============================================================================
# HUGGING FACE HUB DOWNLOAD FUNCTIONS
# =============================================================================

def download_from_huggingface(model_id="CesarChaMal/jvm_troubleshooting_model", auth_token=None):
    """
    Download the trained model from Hugging Face Hub.
    
    This function downloads the complete model from the Hugging Face Hub,
    including all necessary files for inference. It's useful when:
    - Setting up the model in a new environment
    - Local model files are corrupted or missing
    - Deploying to production servers
    - Sharing the model across team members
    
    Args:
        model_id (str): Hugging Face model repository ID
        auth_token (str): HF authentication token (optional for public models)
        
    Returns:
        bool: True if download successful, False otherwise
        
    Note:
        This downloads the entire model repository, which may be several hundred MB.
        Ensure you have sufficient disk space and network bandwidth.
    """
    
    model_dir = "./models/jvm_troubleshooting_model"
    
    print(f"🌐 Downloading model from Hugging Face Hub...")
    print(f"🏷️  Repository: {model_id}")
    
    try:
        # =============================================================================
        # DOWNLOAD PREPARATION
        # =============================================================================
        
        # Create local model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Check if directory is empty or has existing files
        existing_files = os.listdir(model_dir) if os.path.exists(model_dir) else []
        if existing_files:
            print(f"⚠️  Found {len(existing_files)} existing files in model directory")
            print("🗑️  These will be overwritten by the download")
        
        # =============================================================================
        # MODEL DOWNLOAD
        # =============================================================================
        
        print("📦 Starting download (this may take several minutes)...")
        
        # Download the complete model repository
        downloaded_path = snapshot_download(
            repo_id=model_id,
            local_dir=model_dir,
            token=auth_token,
            # Ignore unnecessary files to save space and time
            ignore_patterns=[
                "*.git*",           # Git metadata
                "README.md",        # Documentation
                ".gitattributes",   # Git LFS config
                "*.md",             # Other markdown files
                "training_args.json", # Training configuration
                "trainer_state.json", # Training state
                "*.log"             # Log files
            ],
            # Resume partial downloads if interrupted
            resume_download=True
        )
        
        # =============================================================================
        # DOWNLOAD VALIDATION
        # =============================================================================
        
        # Verify essential files were downloaded
        essential_files = [
            "config.json",
            "tokenizer_config.json",
        ]
        
        model_files = [
            "model.safetensors",
            "pytorch_model.bin"
        ]
        
        # Check for essential configuration files
        missing_essential = []
        for file_name in essential_files:
            if not os.path.exists(os.path.join(model_dir, file_name)):
                missing_essential.append(file_name)
        
        # Check for at least one model file
        model_file_found = any(
            os.path.exists(os.path.join(model_dir, f)) for f in model_files
        )
        
        if missing_essential:
            print(f"⚠️  Warning: Missing essential files: {missing_essential}")
        
        if not model_file_found:
            print("❌ No model files found in download")
            return False
        
        # Count downloaded files
        downloaded_files = os.listdir(model_dir)
        total_size = sum(
            os.path.getsize(os.path.join(model_dir, f)) 
            for f in downloaded_files 
            if os.path.isfile(os.path.join(model_dir, f))
        )
        
        print(f"✅ Successfully downloaded {len(downloaded_files)} files")
        print(f"📊 Total size: {total_size / (1024*1024):.1f} MB")
        print(f"💾 Model saved to: {model_dir}")
        
        return True
        
    except Exception as e:
        # Handle various download errors with specific guidance
        error_msg = str(e).lower()
        
        print(f"❌ Failed to download model from Hugging Face")
        print(f"🔍 Error: {e}")
        
        # Provide specific troubleshooting based on error type
        if "404" in error_msg or "not found" in error_msg:
            print("\n💡 Model Not Found - Possible solutions:")
            print(f"   • Verify model ID is correct: {model_id}")
            print("   • Check if model is public or requires authentication")
            print("   • Ensure model exists on Hugging Face Hub")
            
        elif "401" in error_msg or "403" in error_msg:
            print("\n💡 Authentication Error - Possible solutions:")
            print("   • Provide authentication token for private models")
            print("   • Check token permissions")
            print("   • Verify account access to the model")
            
        elif "network" in error_msg or "connection" in error_msg:
            print("\n💡 Network Error - Possible solutions:")
            print("   • Check internet connection")
            print("   • Try again later (server may be busy)")
            print("   • Check firewall/proxy settings")
            
        else:
            print("\n💡 General troubleshooting:")
            print("   • Ensure sufficient disk space")
            print("   • Check write permissions to model directory")
            print("   • Try manual download from Hugging Face website")
        
        return False

# =============================================================================
# MODEL VALIDATION AND TESTING FUNCTIONS
# =============================================================================

def test_model_loading():
    """
    Test if the model can be loaded successfully and is functional.
    
    This function performs comprehensive validation of the model files:
    - Checks if model directory exists
    - Validates essential files are present
    - Tests tokenizer loading
    - Tests model loading
    - Performs basic inference test
    
    Returns:
        bool: True if model loads and works correctly, False otherwise
        
    Note:
        This function helps identify corrupted files, missing dependencies,
        or configuration issues before attempting to use the model.
    """
    
    model_path = "./models/jvm_troubleshooting_model"
    
    print("🧪 Testing model integrity and loading...")
    
    # =============================================================================
    # DIRECTORY AND FILE VALIDATION
    # =============================================================================
    
    if not os.path.exists(model_path):
        print("❌ Model directory not found")
        print(f"💾 Expected location: {model_path}")
        print("💡 Run model recovery: python model_utils.py recover")
        return False
    
    # Check for essential files
    essential_files = {
        "config.json": "Model configuration",
        "tokenizer_config.json": "Tokenizer configuration",
    }
    
    model_files = {
        "model.safetensors": "Model weights (SafeTensors format)",
        "pytorch_model.bin": "Model weights (PyTorch format)"
    }
    
    print("📁 Checking essential files...")
    
    # Validate essential configuration files
    missing_files = []
    for file_name, description in essential_files.items():
        file_path = os.path.join(model_path, file_name)
        if os.path.exists(file_path):
            print(f"  ✅ {file_name} - {description}")
        else:
            print(f"  ❌ {file_name} - {description} (MISSING)")
            missing_files.append(file_name)
    
    # Check for at least one model file
    model_file_found = False
    for file_name, description in model_files.items():
        file_path = os.path.join(model_path, file_name)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"  ✅ {file_name} - {description} ({file_size:.1f} MB)")
            model_file_found = True
            break
    
    if not model_file_found:
        print("  ❌ No model weight files found")
        missing_files.append("model weights")
    
    if missing_files:
        print(f"\n⚠️  Missing critical files: {', '.join(missing_files)}")
        print("💡 Try model recovery to fix missing files")
        return False
    
    # =============================================================================
    # TOKENIZER LOADING TEST
    # =============================================================================
    
    try:
        print("\n🔤 Testing tokenizer loading...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Basic tokenizer validation
        test_text = "What are common JVM memory issues?"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        
        print(f"  ✅ Tokenizer loaded successfully")
        print(f"  ✅ Vocabulary size: {tokenizer.vocab_size:,}")
        print(f"  ✅ Test encoding/decoding: OK")
        
    except Exception as e:
        print(f"  ❌ Tokenizer loading failed: {e}")
        print("💡 Tokenizer files may be corrupted")
        return False
    
    # =============================================================================
    # MODEL LOADING TEST
    # =============================================================================
    
    try:
        print("\n🤖 Testing model loading...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",  # Use appropriate dtype
            low_cpu_mem_usage=True  # Optimize memory usage
        )
        
        # Model validation
        param_count = model.num_parameters()
        model_size_mb = param_count * 4 / (1024*1024)  # Rough estimate
        
        print(f"  ✅ Model loaded successfully")
        print(f"  ✅ Parameters: {param_count:,}")
        print(f"  ✅ Estimated size: {model_size_mb:.1f} MB")
        
        # Set to evaluation mode
        model.eval()
        
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        
        # Provide specific error guidance
        error_msg = str(e).lower()
        if "out of memory" in error_msg:
            print("💡 Insufficient RAM - try on a machine with more memory")
        elif "safetensors" in error_msg:
            print("💡 SafeTensors error - model file may be corrupted")
        elif "config" in error_msg:
            print("💡 Configuration error - config.json may be invalid")
        else:
            print("💡 Model files may be corrupted or incompatible")
        
        return False
    
    # =============================================================================
    # BASIC INFERENCE TEST
    # =============================================================================
    
    try:
        print("\n💬 Testing basic inference...")
        
        # Simple inference test
        test_input = "### Human: What is JVM?\n### Assistant:"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=10,
                do_sample=False,  # Deterministic for testing
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"  ✅ Inference test successful")
        print(f"  ✅ Generated response length: {len(response)} characters")
        
    except Exception as e:
        print(f"  ❌ Inference test failed: {e}")
        print("💡 Model loads but cannot generate responses")
        return False
    
    print("\n✨ All tests passed! Model is ready for use.")
    return True

# =============================================================================
# AUTOMATED MODEL RECOVERY FUNCTIONS
# =============================================================================

def recover_model(auth_token=None):
    """
    Attempt to recover the model using all available methods.
    
    This function tries multiple recovery strategies in order of preference:
    1. Copy from local training checkpoints (fastest)
    2. Download from Hugging Face Hub (requires internet)
    
    The function automatically validates each recovery attempt and only
    considers it successful if the model can be loaded and used.
    
    Args:
        auth_token (str): Hugging Face authentication token (optional)
        
    Returns:
        bool: True if model was successfully recovered, False otherwise
        
    Note:
        This is the recommended function to use when model files are missing
        or corrupted. It handles all recovery scenarios automatically.
    """
    
    print("🔧 JVM Model Recovery Utility")
    print("=" * 40)
    print("🎯 Attempting to recover missing or corrupted model files...")
    
    # =============================================================================
    # RECOVERY STRATEGY 1: LOCAL CHECKPOINT
    # =============================================================================
    
    print("\n💾 Strategy 1: Copy from local training checkpoint")
    print("-" * 50)
    
    if copy_from_checkpoint():
        print("🧪 Validating recovered model...")
        if test_model_loading():
            print("\n✨ SUCCESS: Model recovered from local checkpoint!")
            print("💡 The model is ready for use")
            return True
        else:
            print("⚠️  Checkpoint files copied but model validation failed")
            print("🔄 Trying next recovery method...")
    else:
        print("⚠️  No usable checkpoints found locally")
        print("🔄 Trying next recovery method...")
    
    # =============================================================================
    # RECOVERY STRATEGY 2: HUGGING FACE DOWNLOAD
    # =============================================================================
    
    print("\n🌐 Strategy 2: Download from Hugging Face Hub")
    print("-" * 50)
    
    if download_from_huggingface(auth_token=auth_token):
        print("🧪 Validating downloaded model...")
        if test_model_loading():
            print("\n✨ SUCCESS: Model recovered from Hugging Face Hub!")
            print("💡 The model is ready for use")
            return True
        else:
            print("⚠️  Model downloaded but validation failed")
            print("🔍 This may indicate a corrupted download")
    else:
        print("⚠️  Download from Hugging Face failed")
    
    # =============================================================================
    # RECOVERY FAILED
    # =============================================================================
    
    print("\n❌ MODEL RECOVERY FAILED")
    print("=" * 40)
    print("🚨 Could not recover the model using any available method")
    
    print("\n💡 Possible solutions:")
    print("   1. Train a new model:")
    print("      python main.py (with TRAIN_MODEL=true in .env)")
    print("   2. Check internet connection and try again")
    print("   3. Verify Hugging Face authentication if using private models")
    print("   4. Contact support if the issue persists")
    
    print("\n📊 Recovery attempt summary:")
    print("   • Local checkpoint: Failed or not available")
    print("   • Hugging Face download: Failed or validation error")
    
    return False

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    """
    Command-line interface for model management operations.
    
    Provides easy access to all model recovery and management functions
    through simple command-line arguments.
    """
    
    import sys
    from dotenv import load_dotenv
    
    # Load environment variables for authentication
    load_dotenv()
    auth_token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        print(f"🔧 JVM Model Utilities - {command.upper()} Operation")
        print("=" * 50)
        
        try:
            if command == "copy":
                print("💾 Copying model from local checkpoint...")
                success = copy_from_checkpoint()
                
            elif command == "download":
                print("🌐 Downloading model from Hugging Face Hub...")
                success = download_from_huggingface(auth_token=auth_token)
                
            elif command == "test":
                print("🧪 Testing model loading and functionality...")
                success = test_model_loading()
                
            elif command == "recover":
                print("🔄 Starting automated model recovery...")
                success = recover_model(auth_token=auth_token)
                
            else:
                print(f"❌ Unknown command: {command}")
                success = False
            
            # Display final result
            if success:
                print(f"\n✨ {command.upper()} operation completed successfully!")
            else:
                print(f"\n❌ {command.upper()} operation failed")
                sys.exit(1)
                
        except KeyboardInterrupt:
            print(f"\n\n⏹️  {command.upper()} operation interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n💥 Unexpected error during {command}: {e}")
            sys.exit(1)
    
    else:
        # Display usage information
        print("🔧 JVM Model Management Utilities")
        print("=" * 40)
        print("\n📚 Usage: python model_utils.py <command>")
        print("\n📎 Available Commands:")
        print("   recover  - Automatically recover model (recommended)")
        print("   copy     - Copy model from local checkpoint")
        print("   download - Download model from Hugging Face Hub")
        print("   test     - Test model loading and functionality")
        
        print("\n💡 Examples:")
        print("   python model_utils.py recover    # Auto-recover missing model")
        print("   python model_utils.py test       # Validate existing model")
        print("   python model_utils.py download   # Force download from HF Hub")
        
        print("\n🔍 Common Use Cases:")
        print("   • Model files missing after Git operations")
        print("   • Corrupted model files")
        print("   • Setting up model in new environment")
        print("   • Validating model integrity")