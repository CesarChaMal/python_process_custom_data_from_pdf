#!/bin/bash

# ============================================================================
# PDF to Q&A Dataset Generator - Cross-Platform Launcher Script
# ============================================================================
#
# This script provides a comprehensive setup and execution environment for the
# PDF to Q&A Dataset Generator ML pipeline. It handles environment detection,
# dependency management, interactive configuration, and automated execution.
#
# FEATURES:
# - Cross-platform Python detection (python/python3/py commands)
# - Automatic virtual environment management
# - Interactive AI provider configuration (Ollama/OpenAI)
# - Model training setup with multiple base model options
# - Dependency validation and installation
# - Environment file (.env) management
# - Pre-flight checks for AI services
# - Post-execution model testing options
#
# MACHINE LEARNING PIPELINE:
# 1. PDF Text Extraction - Extract content from PDF documents
# 2. AI-Powered Q&A Generation - Create training datasets using LLMs
# 3. Dataset Management - Structure and validate training data
# 4. Model Fine-tuning - Train custom conversational models
# 5. Model Testing - Interactive validation of trained models
# 6. Model Deployment - Upload to Hugging Face Hub
#
# SUPPORTED ENVIRONMENTS:
# - Linux/Unix systems
# - macOS
# - Windows (Git Bash, WSL, Cygwin)
# - Cross-platform Python detection
#
# USAGE EXAMPLES:
#   ./run.sh                    # Interactive setup with default settings
#   AI_PROVIDER=openai ./run.sh # Override AI provider for this run
#   AI_MODEL=gpt-4 ./run.sh     # Use specific model temporarily
#
# CONFIGURATION OPTIONS:
#   Edit .env file to set persistent configuration:
#   - AI_PROVIDER: ollama (local) or openai (cloud)
#   - AI_MODEL: Specific model name (optional)
#   - OPENAI_API_KEY: Required for OpenAI provider
#   - HUGGING_FACE_HUB_TOKEN: Optional for model/dataset upload
#   - TRAIN_MODEL: Enable/disable model fine-tuning
#   - BASE_MODEL: Pre-trained model for fine-tuning
#   - FINETUNE_METHOD: full or lora (Parameter Efficient Fine-Tuning)
#   - OVERWRITE_DATASET: Force dataset regeneration
#
# TECHNICAL REQUIREMENTS:
# - Python 3.8+ with pip
# - For Ollama: Local Ollama server running on port 11434
# - For OpenAI: Valid API key with sufficient credits
# - For training: GPU recommended but not required
# - For uploads: Hugging Face account and token
#
# Author: Generated from PDF to Q&A Dataset Generator Pipeline
# Purpose: Automated ML pipeline setup and execution
# ============================================================================

# Enable strict error handling
# -e: Exit on any command failure
# This ensures the script stops if any step fails
set -e

echo "============================================================================"
echo "PDF to Q&A Dataset Generator - ML Pipeline Launcher"
echo "============================================================================"
echo "Starting automated setup and execution..."

# ============================================================================
# PYTHON ENVIRONMENT DETECTION
# ============================================================================
# Cross-platform Python detection to handle different installation methods:
# - 'python' - Standard Linux/Mac installation
# - 'python3' - Explicit Python 3 command
# - 'py' - Windows Python Launcher
# This ensures compatibility across different operating systems and Python setups

echo "[1/8] Detecting Python installation..."
PYTHON_CMD=""

# Try 'python' command first (most common)
if command -v python &> /dev/null && python --version &> /dev/null; then
    PYTHON_CMD="python"
    echo "✓ Found Python via 'python' command"
# Try 'python3' command (explicit Python 3)
elif command -v python3 &> /dev/null && python3 --version &> /dev/null; then
    PYTHON_CMD="python3"
    echo "✓ Found Python via 'python3' command"
# Try 'py' command (Windows Python Launcher)
elif command -v py &> /dev/null && py --version &> /dev/null; then
    PYTHON_CMD="py"
    echo "✓ Found Python via 'py' command (Windows)"
else
    echo "❌ [ERROR] Python is not installed or not working."
    echo "Please install Python 3.8+ first from https://python.org"
    echo "Supported commands: python, python3, py"
    exit 1
fi

# Display detected Python version for verification
echo "Using Python command: $PYTHON_CMD"
echo -n "Python version: "
$PYTHON_CMD --version

# ============================================================================
# VIRTUAL ENVIRONMENT MANAGEMENT
# ============================================================================
# Clean virtual environment setup to ensure consistent dependencies
# This prevents conflicts with system-wide Python packages

echo ""
echo "[2/8] Setting up virtual environment..."

# Remove existing virtual environment to ensure clean state
# This prevents issues with corrupted or outdated environments
if [ -d ".venv" ]; then
    echo "🗑️  Removing existing virtual environment for clean setup..."
    rm -rf .venv
fi

# Create new virtual environment
echo "📦 Creating fresh virtual environment..."
$PYTHON_CMD -m venv .venv

# Validate virtual environment creation
if [ ! -d ".venv" ]; then
    echo "❌ [ERROR] Failed to create virtual environment"
    echo "This might be due to:"
    echo "- Insufficient disk space"
    echo "- Permission issues"
    echo "- Python venv module not installed"
    exit 1
fi

echo "✓ Virtual environment created successfully"

# ============================================================================
# VIRTUAL ENVIRONMENT ACTIVATION
# ============================================================================
# Cross-platform virtual environment activation
# Different operating systems use different activation scripts

echo ""
echo "[3/8] Activating virtual environment..."

# Detect operating system and use appropriate activation script
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows environments (Git Bash, MSYS2, etc.)
    echo "🪟 Detected Windows environment"
    source .venv/Scripts/activate
else
    # Unix-like environments (Linux, macOS, WSL)
    echo "🐧 Detected Unix-like environment"
    source .venv/bin/activate
fi

echo "✓ Virtual environment activated"

# ============================================================================
# DEPENDENCY INSTALLATION
# ============================================================================
# Install required Python packages for the ML pipeline

echo ""
echo "[4/8] Installing ML pipeline dependencies..."
echo "📥 Installing packages from requirements.txt..."

# Install dependencies with progress indication
# --upgrade ensures latest compatible versions
# --quiet reduces output verbosity
pip install --upgrade pip  # Ensure latest pip version
pip install -r requirements.txt

echo "✓ Dependencies installed successfully"

# ============================================================================
# ENVIRONMENT CONFIGURATION CHECK
# ============================================================================
# Validate and prepare environment configuration

echo ""
echo "[5/8] Checking environment configuration..."

# Check if .env file exists for persistent configuration
if [ ! -f ".env" ]; then
    echo "⚠️  [WARNING] .env file not found"
    echo "Creating .env file from template..."
    
    # Create basic .env file if .env.example exists
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✓ Created .env from .env.example template"
    else
        # Create minimal .env file
        touch .env
        echo "✓ Created empty .env file"
    fi
    
    echo "You can manually edit .env file later for persistent configuration"
else
    echo "✓ Found existing .env configuration file"
fi

# ============================================================================
# INTERACTIVE CONFIGURATION SETUP
# ============================================================================
# Guide user through ML pipeline configuration options
# This ensures proper setup for different use cases and environments

echo ""
echo "[6/8] Interactive Configuration Setup"
echo "============================================================================"
echo "Configure your ML pipeline settings:"
echo ""
echo "🤖 1. AI Provider Selection:"
echo "   1) Ollama (Local LLM - Free, Private, Requires local setup)"
echo "   2) OpenAI (Cloud API - Paid, High quality, Easy setup)"
echo ""
read -p "Choose AI provider (1-2) [1]: " provider_choice
provider_choice=${provider_choice:-1}  # Default to Ollama

# Configure AI provider based on user selection
if [ "$provider_choice" = "2" ]; then
    AI_PROVIDER="openai"
    echo ""
    echo "📡 OpenAI Configuration:"
    echo "Available models:"
    echo "   - gpt-4o-mini: Fast, cost-effective, good quality"
    echo "   - gpt-3.5-turbo: Balanced speed and quality"
    echo "   - gpt-4: Highest quality, slower, more expensive"
    echo ""
    read -p "Enter OpenAI model [gpt-4o-mini]: " ai_model
    ai_model=${ai_model:-gpt-4o-mini}
else
    AI_PROVIDER="ollama"
    echo ""
    echo "🏠 Ollama Configuration:"
    echo "Using local Ollama server for private, offline processing"
    echo "Recommended model: cesarchamal/qa-expert (optimized for Q&A generation)"
    echo ""
    read -p "Enter Ollama model [cesarchamal/qa-expert]: " ai_model
    ai_model=${ai_model:-cesarchamal/qa-expert}
fi

echo ""
echo "📊 2. Dataset Management:"
echo "   1) Use existing dataset if found (Faster, reuse previous work)"
echo "   2) Always overwrite existing dataset (Fresh generation, slower)"
echo ""
read -p "Choose dataset option (1-2) [1]: " dataset_choice
dataset_choice=${dataset_choice:-1}  # Default to reuse existing

# Set dataset overwrite behavior
if [ "$dataset_choice" = "2" ]; then
    OVERWRITE_DATASET="true"
    echo "✓ Will regenerate dataset from scratch"
else
    OVERWRITE_DATASET="false"
    echo "✓ Will reuse existing dataset if available"
fi

echo ""
echo "🧠 3. Model Training Configuration:"
echo "   1) Skip model training (Dataset generation only)"
echo "   2) Train custom model after dataset creation (Full ML pipeline)"
echo ""
echo "Note: Training requires significant computational resources and time"
read -p "Choose training option (1-2) [1]: " train_choice
train_choice=${train_choice:-1}  # Default to skip training

# Configure model training if selected
if [ "$train_choice" = "2" ]; then
    TRAIN_MODEL="true"
    echo "✓ Model training enabled"
    
    echo ""
    echo "🔧 Fine-tuning Method Selection:"
    echo "   1) Full fine-tuning (Updates all model parameters)"
    echo "      - Best quality results"
    echo "      - Requires more memory and time"
    echo "      - Recommended for production use"
    echo ""
    echo "   2) LoRA fine-tuning (Parameter Efficient Fine-Tuning)"
    echo "      - Faster training with less memory"
    echo "      - Good quality with efficiency"
    echo "      - Recommended for experimentation"
    echo ""
    read -p "Choose fine-tuning method (1-2) [1]: " finetune_method
    finetune_method=${finetune_method:-1}
    
    if [ "$finetune_method" = "2" ]; then
        FINETUNE_METHOD="lora"
        echo "✓ Using LoRA (Parameter Efficient Fine-Tuning)"
    else
        FINETUNE_METHOD="full"
        echo "✓ Using full fine-tuning"
    fi
    
    echo ""
    echo "🏗️  Base Model Selection:"
    echo "Choose the pre-trained model to fine-tune:"
    echo ""
    echo "   1) microsoft/DialoGPT-small (117M params)"
    echo "      - Fast training and inference"
    echo "      - Lower memory requirements"
    echo "      - Good for prototyping"
    echo ""
    echo "   2) microsoft/DialoGPT-medium (345M params) [RECOMMENDED]"
    echo "      - Balanced performance and quality"
    echo "      - Moderate resource requirements"
    echo "      - Best overall choice"
    echo ""
    echo "   3) microsoft/DialoGPT-large (762M params)"
    echo "      - Highest quality responses"
    echo "      - Requires significant resources"
    echo "      - Best for production deployment"
    echo ""
    echo "   4) distilgpt2 (82M params)"
    echo "      - Very fast, minimal resources"
    echo "      - Basic conversational ability"
    echo ""
    echo "   5) gpt2 (124M params)"
    echo "      - Standard GPT-2 model"
    echo "      - General purpose"
    echo ""
    echo "   6) Custom model (Enter your own)"
    echo ""
    read -p "Choose base model (1-6) [2]: " model_choice
    model_choice=${model_choice:-2}  # Default to DialoGPT-medium
    
    # Set base model based on user selection
    case $model_choice in
        1) base_model="microsoft/DialoGPT-small" 
           echo "✓ Selected DialoGPT-small (fast, lightweight)" ;;
        2) base_model="microsoft/DialoGPT-medium" 
           echo "✓ Selected DialoGPT-medium (recommended balance)" ;;
        3) base_model="microsoft/DialoGPT-large" 
           echo "✓ Selected DialoGPT-large (highest quality)" ;;
        4) base_model="distilgpt2" 
           echo "✓ Selected DistilGPT-2 (very fast)" ;;
        5) base_model="gpt2" 
           echo "✓ Selected GPT-2 (standard)" ;;
        6) 
            echo ""
            echo "Enter custom model name (e.g., 'microsoft/DialoGPT-medium'):"
            read -p "Custom model: " base_model
            echo "✓ Selected custom model: $base_model"
            ;;
        *) base_model="microsoft/DialoGPT-medium" 
           echo "✓ Using default: DialoGPT-medium" ;;
    esac
else
    TRAIN_MODEL="false"
    FINETUNE_METHOD="full"
    base_model="microsoft/DialoGPT-medium"  # Default to medium for better quality
    echo "✓ Model training disabled - dataset generation only"
fi

# ============================================================================
# ENVIRONMENT FILE CONFIGURATION
# ============================================================================
# Update .env file with user selections for persistent configuration

echo ""
echo "[7/8] Updating configuration file..."
echo "💾 Saving settings to .env file for future use..."

# Create or update .env file with proper escaping
if [ ! -f ".env" ]; then
    touch .env
fi

# Remove existing entries and add new ones
grep -v "^AI_PROVIDER=" .env > .env.tmp 2>/dev/null || touch .env.tmp
echo "AI_PROVIDER=$AI_PROVIDER" >> .env.tmp

grep -v "^AI_MODEL=" .env.tmp > .env.tmp2 2>/dev/null || touch .env.tmp2
echo "AI_MODEL=$ai_model" >> .env.tmp2

grep -v "^OVERWRITE_DATASET=" .env.tmp2 > .env.tmp3 2>/dev/null || touch .env.tmp3
echo "OVERWRITE_DATASET=$OVERWRITE_DATASET" >> .env.tmp3

grep -v "^TRAIN_MODEL=" .env.tmp3 > .env.tmp4 2>/dev/null || touch .env.tmp4
echo "TRAIN_MODEL=$TRAIN_MODEL" >> .env.tmp4

grep -v "^FINETUNE_METHOD=" .env.tmp4 > .env.tmp5 2>/dev/null || touch .env.tmp5
echo "FINETUNE_METHOD=$FINETUNE_METHOD" >> .env.tmp5

grep -v "^BASE_MODEL=" .env.tmp5 > .env.new 2>/dev/null || touch .env.new
echo "BASE_MODEL=$base_model" >> .env.new

# Replace original .env file
mv .env.new .env
rm -f .env.tmp .env.tmp2 .env.tmp3 .env.tmp4 .env.tmp5 2>/dev/null || true

# Display final configuration summary
echo ""
echo "============================================================================"
echo "📋 CONFIGURATION SUMMARY"
echo "============================================================================"
echo "🤖 AI Provider: $AI_PROVIDER"
echo "🧠 AI Model: $ai_model"
echo "📊 Overwrite Dataset: $OVERWRITE_DATASET"
echo "🏋️  Train Model: $TRAIN_MODEL"
if [ "$TRAIN_MODEL" = "true" ]; then
    echo "🏗️  Base Model: $base_model"
    echo "🔧 Fine-tuning Method: $FINETUNE_METHOD"
fi
echo "============================================================================"

# ============================================================================
# PRE-FLIGHT VALIDATION
# ============================================================================
# Validate required files and services before starting the ML pipeline

echo ""
echo "[8/8] Performing pre-flight checks..."

# Check if source PDF file exists
echo "📄 Checking for PDF input file..."
if [ ! -f "jvm_troubleshooting_guide.pdf" ]; then
    echo "❌ [ERROR] PDF file 'jvm_troubleshooting_guide.pdf' not found"
    echo ""
    echo "Required: Place your PDF file in the current directory with the name:"
    echo "         'jvm_troubleshooting_guide.pdf'"
    echo ""
    echo "The PDF should contain the content you want to convert into Q&A format."
    exit 1
fi
echo "✓ Found PDF input file"

# Validate AI provider setup and connectivity
if [ "$AI_PROVIDER" = "ollama" ]; then
    echo "🏠 Validating Ollama setup..."
    
    # Check if Ollama server is running
    if ! curl -s http://localhost:11434/api/tags > /dev/null; then
        echo "❌ [ERROR] Ollama server is not running on localhost:11434"
        echo ""
        echo "To fix this:"
        echo "1. Install Ollama from https://ollama.ai"
        echo "2. Start the server: ollama serve"
        echo "3. Run this script again"
        echo ""
        echo "Ollama provides local, private AI processing without API costs."
        exit 1
    fi
    echo "✓ Ollama server is running"
    
    # Check if required model is available
    echo "🔍 Checking for required model..."
    if ! curl -s http://localhost:11434/api/tags | grep -q "$ai_model"; then
        echo "⬇️  Model '$ai_model' not found locally. Downloading..."
        echo "This may take several minutes depending on model size."
        ollama pull "$ai_model"
        echo "✓ Model downloaded successfully"
    else
        echo "✓ Model '$ai_model' is available"
    fi
elif [ "$AI_PROVIDER" = "openai" ]; then
    echo "📡 Validating OpenAI setup..."
    
    # Check if API key is configured
    OPENAI_KEY=$(grep "^OPENAI_API_KEY=" .env 2>/dev/null | cut -d'=' -f2 | tr -d '"')
    if [ -z "$OPENAI_KEY" ] || [ "$OPENAI_KEY" = "your_openai_key_here" ]; then
        echo "❌ [ERROR] OpenAI API key not configured"
        echo ""
        echo "To fix this:"
        echo "1. Get an API key from https://platform.openai.com/api-keys"
        echo "2. Add it to your .env file: OPENAI_API_KEY=sk-your-key-here"
        echo "3. Ensure you have sufficient credits in your OpenAI account"
        echo ""
        echo "Note: OpenAI charges per API call. Check pricing at https://openai.com/pricing"
        exit 1
    fi
    echo "✓ OpenAI API key configured"
    echo "✓ Using model: $ai_model"
else
    echo "❌ [ERROR] Unsupported AI provider: $AI_PROVIDER"
    echo "Supported providers: 'ollama' (local) or 'openai' (cloud)"
    echo "Please check your .env configuration."
    exit 1
fi

# Pre-download base model for training if enabled
if [ "$TRAIN_MODEL" = "true" ]; then
    echo ""
    echo "🏗️  Preparing base model for fine-tuning..."
    echo "📥 Pre-downloading: $base_model"
    echo "This ensures the model is available before training starts."
    
    # Download tokenizer and model to local cache
    $PYTHON_CMD -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('$base_model')
print('Downloading model...')
AutoModelForCausalLM.from_pretrained('$base_model')
print('✓ Base model ready for fine-tuning')
"
fi

# ============================================================================
# ML PIPELINE EXECUTION
# ============================================================================
# Execute the main ML pipeline with validated configuration

echo ""
echo "============================================================================"
echo "🚀 STARTING ML PIPELINE EXECUTION"
echo "============================================================================"
echo "✅ All pre-flight checks passed"
echo "🔄 Executing PDF to Q&A Dataset Generator pipeline..."
echo ""

# Execute the main Python script with all configuration ready
$PYTHON_CMD main.py

echo ""
echo "============================================================================"
echo "🎉 ML PIPELINE COMPLETED SUCCESSFULLY!"
echo "============================================================================"

# ============================================================================
# POST-EXECUTION MODEL TESTING
# ============================================================================
# Offer interactive model testing if a trained model is available

echo ""
echo "🧪 Post-Execution Options:"

# Check if trained model exists and offer testing
if [ -d "./models/jvm_troubleshooting_model" ]; then
    echo "✅ Trained model found at: ./models/jvm_troubleshooting_model"
    echo ""
    echo "Available testing options:"
    echo "1. Interactive testing with conversation memory (test_model.py)"
    echo "2. Quick batch testing (quick_test.py)"
    echo "3. Skip testing"
    echo ""
    read -p "Choose testing option (1-3) [3]: " test_choice
    test_choice=${test_choice:-3}
    
    case $test_choice in
        1)
            echo "🚀 Starting interactive model testing..."
            $PYTHON_CMD test_model.py
            ;;
        2)
            echo "🚀 Running quick batch validation..."
            $PYTHON_CMD quick_test.py
            ;;
        *)
            echo "✓ Skipping model testing"
            echo "You can test the model later using:"
            echo "  python test_model.py              # Interactive testing with memory"
            echo "  python quick_test.py              # Quick validation"
            ;;
    esac
elif [ "$TRAIN_MODEL" = "true" ]; then
    echo "⚠️  [WARNING] Model training was enabled but no model found"
    echo "Check the training logs above for any errors."
    echo "You can try running 'python model_utils.py recover' to restore the model."
else
    echo "ℹ️  No model training was performed (dataset generation only)"
    echo "To train a model, run this script again and select training option."
fi

echo ""
echo "============================================================================"
echo "📚 NEXT STEPS:"
echo "============================================================================"
echo "• Dataset: Check ./dataset/jvm_troubleshooting_guide/ for generated Q&A pairs"
if [ "$TRAIN_MODEL" = "true" ]; then
    echo "• Model: Check ./models/jvm_troubleshooting_model/ for trained model files"
fi
echo "• Testing: Use test_model.py for interactive testing with conversation memory"
echo "• Upload: Models and datasets can be uploaded to Hugging Face Hub"
echo "• Documentation: See README.md for detailed usage instructions"
echo "============================================================================"
echo "Thank you for using the PDF to Q&A Dataset Generator! 🎯"