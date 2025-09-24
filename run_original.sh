#!/bin/bash

# PDF to Q&A Dataset Generator Launcher
# 
# Usage Examples:
#   ./run.sh                    # Use default settings (Ollama)
#   AI_PROVIDER=openai ./run.sh # Use OpenAI for this run
#   AI_MODEL=gpt-4 ./run.sh     # Use specific model
#
# Configuration:
#   Edit .env file to set:
#   - AI_PROVIDER (ollama/openai)
#   - AI_MODEL (optional)
#   - OPENAI_API_KEY (for OpenAI)
#   - HUGGING_FACE_HUB_TOKEN (optional)
#   - TRAIN_MODEL (true/false)
#   - BASE_MODEL (for fine-tuning)

set -e

echo "Starting PDF to Q&A Dataset Generator..."

# Check if Python is installed (try multiple variants)
PYTHON_CMD=""
if command -v python &> /dev/null && python --version &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null && python3 --version &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v py &> /dev/null && py --version &> /dev/null; then
    PYTHON_CMD="py"
else
    echo "[ERROR] Python is not installed or not working. Please install Python 3.8+ first."
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version

# Remove broken venv if it exists and recreate
if [ -d ".venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf .venv
fi

echo "Creating virtual environment..."
$PYTHON_CMD -m venv .venv

if [ ! -d ".venv" ]; then
    echo "[ERROR] Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "[WARNING] .env file not found. Please create one with your configuration"
    echo "You can copy from .env.example and update the values."
fi

# Interactive configuration
echo "Configuration Setup"
echo "1. Select AI Provider:"
echo "   1) Ollama (default)"
echo "   2) OpenAI"
read -p "Choose provider (1-2) [1]: " provider_choice
provider_choice=${provider_choice:-1}

if [ "$provider_choice" = "2" ]; then
    AI_PROVIDER="openai"
    echo "Enter OpenAI model (gpt-4o-mini, gpt-3.5-turbo, gpt-4) [gpt-4o-mini]:"
    read -p "Model: " ai_model
    ai_model=${ai_model:-gpt-4o-mini}
else
    AI_PROVIDER="ollama"
    echo "Enter Ollama model [cesarchamal/qa-expert]:"
    read -p "Model: " ai_model
    ai_model=${ai_model:-cesarchamal/qa-expert}
fi

echo "2. Dataset handling:"
echo "   1) Use existing dataset if found (default)"
echo "   2) Always overwrite existing dataset"
read -p "Choose option (1-2) [1]: " dataset_choice
dataset_choice=${dataset_choice:-1}

if [ "$dataset_choice" = "2" ]; then
    OVERWRITE_DATASET="true"
else
    OVERWRITE_DATASET="false"
fi

echo "3. Model training:"
echo "   1) Skip model training (default)"
echo "   2) Train model after dataset creation"
read -p "Choose option (1-2) [1]: " train_choice
train_choice=${train_choice:-1}

if [ "$train_choice" = "2" ]; then
    TRAIN_MODEL="true"
    
    echo "Select fine-tuning method:"
    echo "   1) Full fine-tuning (all parameters, best quality)"
    echo "   2) LoRA fine-tuning (efficient, faster)"
    read -p "Choose method (1-2) [1]: " finetune_method
    finetune_method=${finetune_method:-1}
    
    if [ "$finetune_method" = "2" ]; then
        FINETUNE_METHOD="lora"
    else
        FINETUNE_METHOD="full"
    fi
    
    echo "Select base model:"
    echo "   1) microsoft/DialoGPT-small (fast, lightweight)"
    echo "   2) microsoft/DialoGPT-medium (balanced)"
    echo "   3) microsoft/DialoGPT-large (best quality, slow)"
    echo "   4) distilgpt2 (very fast, basic)"
    echo "   5) gpt2 (standard)"
    echo "   6) Custom model"
    read -p "Choose model (1-6) [1]: " model_choice
    model_choice=${model_choice:-1}
    
    case $model_choice in
        1) base_model="microsoft/DialoGPT-small" ;;
        2) base_model="microsoft/DialoGPT-medium" ;;
        3) base_model="microsoft/DialoGPT-large" ;;
        4) base_model="distilgpt2" ;;
        5) base_model="gpt2" ;;
        6) 
            echo "Enter custom model name:"
            read -p "Model: " base_model
            ;;
        *) base_model="microsoft/DialoGPT-small" ;;
    esac
else
    TRAIN_MODEL="false"
    FINETUNE_METHOD="full"
    base_model="microsoft/DialoGPT-small"
fi

# Update .env file
echo "Updating .env configuration..."

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

echo "Using AI provider: $AI_PROVIDER"
echo "Model: $ai_model"
echo "Overwrite dataset: $OVERWRITE_DATASET"
echo "Train model: $TRAIN_MODEL"
if [ "$TRAIN_MODEL" = "true" ]; then
    echo "Base model: $base_model"
    echo "Fine-tuning method: $FINETUNE_METHOD"
fi

# Check if PDF exists
if [ ! -f "jvm_troubleshooting_guide.pdf" ]; then
    echo "[WARNING] PDF file 'jvm_troubleshooting_guide.pdf' not found in current directory"
    echo "Please place your PDF file with this name to continue."
    exit 1
fi

# Check AI provider requirements
if [ "$AI_PROVIDER" = "ollama" ]; then
    echo "Checking Ollama connection..."
    if ! curl -s http://localhost:11434/api/tags > /dev/null; then
        echo "[ERROR] Ollama is not running on localhost:11434"
        echo "Please start Ollama first: ollama serve"
        exit 1
    fi
    
    echo "Checking available models..."
    if ! curl -s http://localhost:11434/api/tags | grep -q "cesarchamal/qa-expert"; then
        echo "[WARNING] Model 'cesarchamal/qa-expert' not found. Pulling model..."
        ollama pull cesarchamal/qa-expert
    fi
elif [ "$AI_PROVIDER" = "openai" ]; then
    echo "Checking OpenAI configuration..."
    OPENAI_KEY=$(grep "^OPENAI_API_KEY=" .env 2>/dev/null | cut -d'=' -f2 | tr -d '"')
    if [ -z "$OPENAI_KEY" ] || [ "$OPENAI_KEY" = "your_openai_key_here" ]; then
        echo "[ERROR] OpenAI API key not configured in .env file"
        echo "Please set OPENAI_API_KEY in your .env file"
        exit 1
    fi
    echo "[SUCCESS] OpenAI configuration found"
else
    echo "[ERROR] Unsupported AI provider: $AI_PROVIDER"
    echo "Please set AI_PROVIDER to 'ollama' or 'openai' in .env file"
    exit 1
fi

# Pre-download base model if training is enabled
if [ "$TRAIN_MODEL" = "true" ]; then
    echo "Pre-downloading base model: $base_model"
    $PYTHON_CMD -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('$base_model'); AutoModelForCausalLM.from_pretrained('$base_model'); print('[SUCCESS] Base model downloaded')"
fi

echo "All checks passed. Starting PDF processing..."
$PYTHON_CMD main.py

echo "Process completed successfully!"

# Check if model exists and offer testing
if [ -d "./models/jvm_troubleshooting_model" ]; then
    echo ""
    read -p "Do you want to test the trained model? (y/N): " test_model
    if [ "${test_model,,}" = "y" ]; then
        echo "Starting model testing..."
        $PYTHON_CMD test_model.py
    fi
elif [ "$TRAIN_MODEL" = "true" ]; then
    echo "[WARNING] Model training was enabled but no model found at ./models/jvm_troubleshooting_model"
fi