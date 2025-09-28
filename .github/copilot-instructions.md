# Copilot Instructions for `python_process_custom_data_from_pdf`

## Overview
This project automates the creation of domain-specific AI assistants from PDF documents. It includes components for text extraction, dataset generation, model fine-tuning, and deployment. The architecture is designed to streamline the machine learning pipeline from data to deployment.

## Key Components

### 1. **Document Processing**
- Extracts and preprocesses text content from PDF files using PyMuPDF.
- Relevant file: `main.py` (entry point for processing).

### 2. **Dataset Generation**
- Uses large language models (Ollama or OpenAI) to generate question-answer pairs.
- Relevant files: `main.py`, `model_utils.py`.

### 3. **Model Training**
- Fine-tunes conversational models using Hugging Face Transformers.
- Supports full fine-tuning and LoRA (parameter-efficient fine-tuning).
- Relevant files: `main.py`, `test_model.py`.

### 4. **Deployment**
- Uploads datasets and models to Hugging Face Hub.
- Relevant files: `main.py`, `create_model_card.py`.

## Developer Workflows

### Setup
- Clone the repository and set up the environment:
  ```bash
  git clone <repository-url>
  cd python_process_custom_data_from_pdf
  ./run.sh  # For Linux/macOS
  run.bat   # For Windows
  ```

### Manual Setup
- Create a virtual environment and install dependencies:
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # Linux/macOS
  .venv\Scripts\activate   # Windows
  pip install -r requirements.txt
  ```

### Running the Application
- Configure `.env` with your AI provider (Ollama or OpenAI).
- Run the main script:
  ```bash
  python main.py
  ```

### Testing
- Use `test_model.py` for model validation.
- Run `quick_test.py` for quick functionality checks.

## Project-Specific Conventions

### Dataset Management
- Datasets are stored in `dataset/` with train/test splits.
- Use `OVERWRITE_DATASET` in `.env` to control overwriting behavior.

### Model Artifacts
- Models are saved in `models/` with versioned checkpoints.
- Use `create_model_card.py` to generate model documentation.

### Logging
- Logs are stored in `training_log.json` for debugging and progress tracking.

## External Dependencies
- **PyMuPDF**: For PDF text extraction.
- **Hugging Face Transformers**: For model fine-tuning and deployment.
- **Ollama/OpenAI**: For AI-powered dataset generation.

## Integration Points
- Hugging Face Hub: Upload datasets and models.
- AI Providers: Configure in `.env` (`AI_PROVIDER`, `AI_MODEL`).

## Examples

### Example `.env` Configuration
```env
AI_PROVIDER=ollama
AI_MODEL=cesarchamal/qa-expert
OVERWRITE_DATASET=false
HUGGING_FACE_HUB_TOKEN=your_hf_token_here
```

### Example Command
- Run the pipeline end-to-end:
  ```bash
  python main.py
  ```

## Notes
- Refer to `README.md` for detailed usage instructions.
- Ensure Python 3.8+ is installed.