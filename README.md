# PDF to Q&A Dataset Generator

## Overview

A comprehensive machine learning pipeline that transforms PDF documents into intelligent question-answer datasets and trains custom conversational AI models. This project demonstrates end-to-end ML workflow from data extraction to model deployment, featuring automated dataset generation, fine-tuning capabilities, and interactive model testing.

## What This Project Does

This tool automates the entire process of creating domain-specific AI assistants from PDF documentation:

1. **Document Processing**: Extracts and preprocesses text content from PDF files using PyMuPDF
2. **AI-Powered Dataset Generation**: Uses large language models (Ollama or OpenAI) to generate contextual question-answer pairs
3. **Dataset Management**: Creates structured datasets with train/test splits and uploads to Hugging Face Hub
4. **Model Fine-tuning**: Trains custom conversational models using state-of-the-art techniques (full fine-tuning or LoRA)
5. **Interactive Testing**: Provides a chat interface to test and validate the trained models
6. **Model Deployment**: Uploads trained models to Hugging Face Hub for sharing and deployment

## Key Machine Learning Concepts Demonstrated

- **Transfer Learning**: Fine-tuning pre-trained language models for domain-specific tasks
- **Parameter Efficient Fine-tuning (PEFT)**: Using LoRA for efficient model adaptation
- **Dataset Engineering**: Automated generation and preprocessing of training data
- **Conversational AI**: Building question-answering systems with proper prompt formatting
- **Model Evaluation**: Interactive testing and validation of trained models
- **MLOps**: Automated pipeline from data to deployment with version control

## Technical Architecture

```
PDF Document → Text Extraction → AI-Generated Q&A → Dataset Creation → Model Training → Deployment
     ↓              ↓                 ↓               ↓              ↓            ↓
  PyMuPDF      Preprocessing    Ollama/OpenAI    HuggingFace    Transformers   HF Hub
```

## Features

### Core Functionality
- **PDF Text Extraction**: Advanced text processing with PyMuPDF for clean content extraction
- **AI-Powered Q&A Generation**: Leverages Ollama (local) or OpenAI (cloud) for intelligent question-answer pair creation
- **Multi-Model Support**: Compatible with various language models (GPT-4, Llama, DialoGPT, etc.)
- **Automated Dataset Creation**: Generates structured train/test splits with proper formatting
- **Hugging Face Integration**: Seamless upload and management of datasets and models

### Advanced ML Features
- **Custom Model Training**: Fine-tune pre-trained models on domain-specific data
- **LoRA Support**: Parameter-efficient fine-tuning for faster training and lower memory usage
- **Interactive Model Testing**: Real-time chat interface for model validation
- **Model Deployment**: Automated upload to Hugging Face Hub for sharing and production use
- **Local Caching**: Efficient storage and reuse of datasets and models

### Developer Experience
- **Cross-Platform Support**: Works on Windows, Linux, and macOS
- **Interactive Setup**: Guided configuration for different AI providers and training options
- **Comprehensive Logging**: Detailed progress tracking and debugging information
- **Error Handling**: Robust error management with helpful troubleshooting guides

## Prerequisites

### System Requirements
- **Operating Systems**: Pop!_OS, Ubuntu, Windows (Git Bash), WSL Ubuntu, macOS, Linux
- **Python 3.8+** with pip
- **AI Provider**:
  - **Ollama**: Running locally on port 11434
  - **OpenAI**: Valid API key
- **Hugging Face account and token** (optional for model/dataset upload)

### Platform-Specific Notes
- **Pop!_OS/Ubuntu**: Native support, all features available
- **Windows Git Bash**: Full compatibility, uses Windows Python paths
- **WSL Ubuntu**: Native Linux environment, GPU passthrough supported
- **macOS/Linux**: Standard Unix environment support

## Quick Start

### Pop!_OS / Ubuntu / WSL Ubuntu
```bash
# Clone and setup
git clone <repository-url>
cd python_process_custom_data_from_pdf

# Run setup script
./run.sh
```

### Windows Git Bash
```bash
# Clone and setup
git clone <repository-url>
cd python_process_custom_data_from_pdf

# Run setup script (Git Bash automatically detects Windows)
./run.sh
```

### Alternative Windows (Command Prompt)
```cmd
# Clone and setup
git clone <repository-url>
cd python_process_custom_data_from_pdf

# Run setup script
run.bat
```

## Manual Setup

### Pop!_OS / Ubuntu / WSL Ubuntu / macOS / Linux
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your configuration

# Run the application
python main.py
```

### Windows (Git Bash / Command Prompt)
```bash
# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Git Bash
# .venv\Scripts\activate.bat    # Command Prompt

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env with your configuration

# Run the application
python main.py
```

## Configuration

Configure your AI provider in `.env`:

### Using Ollama (Default)
```bash
AI_PROVIDER=ollama
AI_MODEL=cesarchamal/qa-expert  # optional
OVERWRITE_DATASET=false  # true to overwrite without prompt
HUGGING_FACE_HUB_TOKEN=your_hf_token_here
```

### Using OpenAI
```bash
AI_PROVIDER=openai
AI_MODEL=gpt-4o-mini  # optional: gpt-4o-mini, gpt-3.5-turbo, gpt-4
OVERWRITE_DATASET=false  # true to overwrite without prompt
OPENAI_API_KEY=your_openai_key_here
HUGGING_FACE_HUB_TOKEN=your_hf_token_here
```

## Usage Examples

### Example 1: Using Ollama (Default) - Pop!_OS/Ubuntu/WSL
```bash
# 1. Start Ollama
ollama serve

# 2. Configure .env
echo "AI_PROVIDER=ollama" > .env
echo "HUGGING_FACE_HUB_TOKEN=hf_xxx" >> .env

# 3. Run
./run.sh
```

### Example 1b: Using Ollama - Windows Git Bash
```bash
# 1. Start Ollama (in separate terminal)
ollama serve

# 2. Configure .env (Git Bash)
echo "AI_PROVIDER=ollama" > .env
echo "HUGGING_FACE_HUB_TOKEN=hf_xxx" >> .env

# 3. Run
./run.sh
```

### Example 2: Using OpenAI GPT-4o-mini
```bash
# 1. Configure .env
echo "AI_PROVIDER=openai" > .env
echo "AI_MODEL=gpt-4o-mini" >> .env
echo "OPENAI_API_KEY=sk-xxx" >> .env
echo "HUGGING_FACE_HUB_TOKEN=hf_xxx" >> .env

# 2. Run
./run.sh
```

### Example 3: Using OpenAI GPT-4
```bash
# 1. Configure .env
echo "AI_PROVIDER=openai" > .env
echo "AI_MODEL=gpt-4" >> .env
echo "OPENAI_API_KEY=sk-xxx" >> .env

# 2. Run (local only, no HF upload)
python main.py
```

### Example 4: Custom Ollama Model
```bash
# 1. Pull custom model
ollama pull llama3.2

# 2. Configure .env
echo "AI_PROVIDER=ollama" > .env
echo "AI_MODEL=llama3.2" >> .env

# 3. Run
./run.sh
```

### Example 5: Force Overwrite Existing Dataset
```bash
# 1. Configure .env to overwrite without prompt
echo "OVERWRITE_DATASET=true" >> .env

# 2. Run (will overwrite existing dataset)
python main.py
```

### Example 6: Use Existing Dataset
```bash
# 1. Configure .env to keep existing
echo "OVERWRITE_DATASET=false" >> .env

# 2. Run (will use existing dataset if found)
python main.py
```

### Example 7: Train Model with Better Quality
```bash
# 1. Use interactive setup for training
./run.sh
# Choose: Train model after dataset creation
# Select: DialoGPT-medium for better quality
```

### Example 8: Test Trained Model
```bash
# Interactive testing with conversation memory
python test_model.py

# Quick batch validation
python quick_test.py

# Recover missing model files
python model_utils.py recover
```

## Available Base Models

| Model | Size | Speed | Quality | Use Case | Default |
|-------|------|-------|---------|----------|----------|
| `microsoft/DialoGPT-small` | 117M | Fast | Good | Quick prototyping | |
| `microsoft/DialoGPT-medium` | 345M | Medium | Better | Balanced performance | ✅ |
| `microsoft/DialoGPT-large` | 762M | Slow | Best | Production quality | |
| `distilgpt2` | 82M | Very Fast | Basic | Lightweight deployment | |
| `gpt2` | 124M | Fast | Standard | General purpose | |

**Note**: DialoGPT-medium is now the default for better response quality.

## Usage Steps

1. **Place PDF**: Add your PDF file as `jvm_troubleshooting_guide.pdf`
2. **Configure**: Set up `.env` with your AI provider and tokens
3. **Run**: Execute `./run.sh` (Linux/Mac) or `run.bat` (Windows)
4. **Output**: Dataset generated locally and optionally uploaded to Hugging Face

## Output

- **Local dataset**: `./dataset/jvm_troubleshooting_guide/`
- **Local model**: `./models/jvm_troubleshooting_model/`
- **Hugging Face Dataset**: `https://huggingface.co/datasets/CesarChaMal/jvm_troubleshooting_guide`
- **Hugging Face Model**: `https://huggingface.co/CesarChaMal/jvm_troubleshooting_model`

## Model Testing

### Interactive Testing with Memory
```bash
# Test trained model with conversation memory
python test_model.py
```

**Available Commands:**
- `quit`, `exit`, `q` - Exit the assistant
- `help` - Show help message
- `examples` - Show example questions
- `defaults` - Test with default questions
- `history` - Show conversation history
- `clear` - Clear conversation history
- `context` - Show current context sent to model

**New Feature**: The model now remembers your conversation context for better follow-up responses!

### Quick Batch Testing
```bash
# Quick validation with 11 predefined questions
python quick_test.py

# If model is missing, recover it first
python model_utils.py recover
```

### Default Test Questions
The testing scripts include these JVM troubleshooting questions:
1. What are common JVM memory issues?
2. How do I troubleshoot OutOfMemoryError?
3. What JVM parameters should I tune for performance?
4. How do I analyze garbage collection logs?
5. What causes high CPU usage in JVM applications?
6. How do I debug memory leaks in Java applications?
7. What are the best practices for JVM monitoring?
8. How do I optimize JVM startup time?
9. What tools can I use for JVM profiling?
10. How do I handle StackOverflowError?
11. What are the differences between heap and non-heap memory?

## Troubleshooting

### Test Connections
```bash
# Test Hugging Face connection
python check_hf.py

# Test Ollama connection
curl http://localhost:11434/api/tags

# Test trained model
python test_model.py
```

### GPU Memory Management

The system now includes **intelligent GPU memory analysis** and **automatic error recovery**:

#### 🤖 AI-Powered Error Resolution
When training fails due to GPU memory issues, the system:
- **Analyzes your hardware** (GPU model, memory usage, available VRAM)
- **Tracks failure patterns** (attempt count, error types, model compatibility)
- **Provides smart recommendations** based on your specific situation
- **Offers unlimited retry attempts** with different strategies

#### 🔧 GPU Memory Recovery Options
1. **Smart memory optimization** - For high-VRAM GPUs (8GB+)
2. **Extreme memory efficiency** - Ultra-conservative settings
3. **CPU training fallback** - Automatic CPU switching
4. **GPU cleanup utility** - `python gpu_cleanup.py`
5. **Model/config changes** - Return to selection menus
6. **Skip to testing** - Test existing models without retraining

#### 📊 System Analysis Features
```
🔍 System Analysis:
  • GPU: NVIDIA GeForce RTX 4090
  • Memory Usage: 49.4% of total capacity
  • Available Memory: 7.9GB (50.6%)
  • Failure Attempt: #1
  • Model Size: microsoft/DialoGPT-large (~774M params)

🤖 AI Recommendations:
  1. Option 1: Smart memory optimization - You have sufficient VRAM
  2. Option 4: GPU cleanup utility - Clear background processes first
  3. Option 2: Extreme memory efficiency - Conservative approach

💡 Recommended: Option 1
```

### Common Issues
- **Ollama not running**: Start with `ollama serve`
- **Model not found**: Pull with `ollama pull model-name`
- **OpenAI errors**: Check API key and billing
- **HF upload fails**: Verify token permissions
- **CUDA out of memory**: Use the intelligent error recovery system
- **Training hangs**: The system now provides non-interactive fallbacks
- **GPU processes blocking**: Use `python gpu_cleanup.py` for automated cleanup
- **Model too large for GPU**: System recommends smaller models automatically
- **Repeated training failures**: AI suggests CPU training or model changes
- **Virtual environment corruption**: Robust cleanup handles stubborn files

### Model Recovery

If your trained model files are missing (common after Git operations), use the model recovery utility:

```bash
# Automatic recovery (tries checkpoint first, then Hugging Face)
python model_utils.py recover

# Manual options
python model_utils.py copy      # Copy from local checkpoint
python model_utils.py download  # Download from Hugging Face
python model_utils.py test      # Test if model loads correctly
```

### GPU Cleanup Utility

For advanced GPU memory management:

```bash
# Interactive GPU cleanup tool
python gpu_cleanup.py
```

**Features:**
- Clear PyTorch GPU memory cache
- Show current GPU processes
- Kill Python GPU processes
- Reset GPU (requires admin)
- Full cleanup automation

**Why models go missing**: Large model files are excluded from Git to prevent repository size issues. The trained models are automatically uploaded to Hugging Face Hub and can be recovered from there.

## Advanced Features

### Intelligent Training Management
- **AI-Powered Error Analysis**: Automatic hardware analysis and failure pattern recognition
- **Smart Recommendations**: Context-aware suggestions based on your specific setup
- **Unlimited Retry System**: Keep trying different strategies until success
- **Dynamic Memory Management**: Automatic model size and batch size optimization
- **Fallback Strategies**: CPU training, model changes, and testing options

### Model Quality Improvements
- **Better Default Model**: DialoGPT-medium (345M parameters) instead of small (117M)
- **Improved Training**: Lower learning rate, gradient clipping, more epochs
- **Longer Context**: 768 tokens vs 512 for better understanding
- **Conversation Memory**: Interactive testing remembers conversation history
- **Better Data Processing**: Enhanced filtering and preprocessing

### Robust Environment Management
- **Advanced Virtual Environment Cleanup**: Handles stubborn files and process locks
- **Cross-Platform Compatibility**: Works on Windows, Linux, and macOS
- **Process Management**: Automatic cleanup of conflicting Python processes
- **Memory Optimization**: GPU memory fragmentation detection and resolution

## References

### Video Tutorial Series

This project was inspired by the "Beginner's Guide to DS, ML, and AI" video series. Watch in order:

**Part 1**: **[Beginner's Guide to DS, ML, and AI - [1] Process Your Own PDF Doc into LLM Finetune-Ready Format](https://www.youtube.com/watch?v=hr2kSC1evQM)** - Learn how to extract and process PDF documents into training-ready datasets

**Part 2**: **[Beginner's Guide to DS, ML, and AI - [2] Fine-tune Llama2-7b LLM Using Custom Data](https://www.youtube.com/watch?v=tDkY2gpvylE)** - Complete walkthrough of fine-tuning large language models with your custom dataset

**Part 3**: **[Beginner's Guide to DS, ML, and AI - [3] Deploy Inference Endpoint on HuggingFace](https://www.youtube.com/watch?v=382yy-mCeCA)** - Deploy your trained model to production using Hugging Face inference endpoints

**Source Repository**: [WYNAssociates GitHub](https://github.com/CesarChaMal/WYNAssociates/tree/main) - Contains the complete code examples and implementations demonstrated throughout the video series

### Additional Resources

- **[ML_CONCEPTS.md](ML_CONCEPTS.md)**: Comprehensive guide to machine learning concepts for beginners
- **Hugging Face Transformers**: Official documentation and tutorials
- **PyMuPDF Documentation**: PDF processing and text extraction
- **Ollama Documentation**: Local LLM deployment and usagee Hub for model storage (local models excluded from Git)

