# PDF to Q&A Dataset Generator

Automatically converts PDF documents into question-answer datasets using AI (Ollama or OpenAI) and uploads them to Hugging Face Hub.

## Features

- Extract text content from PDF files
- Generate Q&A pairs using Ollama or OpenAI
- Support for multiple AI models
- Create train/test dataset splits
- Upload datasets to Hugging Face Hub
- **Train custom models** from generated datasets
- **Interactive model testing** with Q&A interface
- Upload trained models to Hugging Face Hub
- Local dataset and model caching

## Prerequisites

- Python 3.8+
- AI Provider:
  - **Ollama**: Running locally on port 11434
  - **OpenAI**: Valid API key
- Hugging Face account and token (optional)

## Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd python_process_custom_data_from_pdf

# Run setup script
./run.sh        # Linux/Mac
# or
run.bat         # Windows
```

## Manual Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
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

### Example 1: Using Ollama (Default)
```bash
# 1. Start Ollama
ollama serve

# 2. Configure .env
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

### Example 7: Train Model from Dataset
```bash
# 1. Configure .env for automatic model training
echo "TRAIN_MODEL=true" >> .env

# 2. Run (will train model after dataset creation)
python main.py
```

### Example 8: Custom Base Model
```bash
# 1. Configure custom base model
echo "BASE_MODEL=microsoft/DialoGPT-medium" >> .env

# 2. Run with model training
python main.py
```

### Example 9: Test Existing Model
```bash
# Interactive testing
python test_model.py

# Quick batch test
python quick_test.py
```

## Available Base Models

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `microsoft/DialoGPT-small` | 117M | Fast | Good | Quick prototyping |
| `microsoft/DialoGPT-medium` | 345M | Medium | Better | Balanced performance |
| `microsoft/DialoGPT-large` | 762M | Slow | Best | Production quality |
| `distilgpt2` | 82M | Very Fast | Basic | Lightweight deployment |
| `gpt2` | 124M | Fast | Standard | General purpose |

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

### Interactive Testing
```bash
# Test trained model interactively
python test_model.py
```

**Available Commands:**
- `quit`, `exit`, `q` - Exit the assistant
- `help` - Show help message
- `examples` - Show example questions
- `defaults` - Test with default questions

### Quick Batch Testing
```bash
# Quick test with predefined questions
python quick_test.py
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

### Common Issues
- **Ollama not running**: Start with `ollama serve`
- **Model not found**: Pull with `ollama pull model-name`
- **OpenAI errors**: Check API key and billing
- **HF upload fails**: Verify token permissions
- **Unicode encoding errors**: Fixed in latest version (removed all emoji characters)
- **Large model files**: Use Hugging Face Hub for model storage (local models excluded from Git)
- **Model generates poor responses**: Try retraining with more data or different parameters
- **Model testing fails**: Ensure model exists in `./models/jvm_troubleshooting_model/`
- **Git push failures**: Large model files are now excluded via .gitignore
